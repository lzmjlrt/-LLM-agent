import logging
import re
from typing import Dict, Iterable, List, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


TOOL_USAGE_HINT_KEYWORDS = (
    "不会用",
    "怎么用",
    "如何用",
    "怎么安装",
    "如何安装",
    "怎么操作",
    "如何操作",
    "说明书",
    "教程",
    "步骤",
)


class QueryPlanningResult(BaseModel):
    rewritten_query: str = Field(description="面向检索的主查询")
    multi_queries: List[str] = Field(default_factory=list, description="多查询改写结果")
    decomposition_queries: List[str] = Field(default_factory=list, description="复杂问题拆解后的子查询")
    needs_decomposition: bool = Field(default=False, description="是否需要做查询分解")


class RetrievalAssessmentResult(BaseModel):
    decision: str = Field(description="检索判定结果：sufficient / insufficient / out_of_scope")
    reason: str = Field(description="判定原因")


def rewrite_query(user_review: str) -> str:
    """将用户原评论重写为更适合检索说明书的查询。"""
    raw_query = (user_review or "").strip()
    if not raw_query:
        return ""

    if any(keyword in raw_query for keyword in TOOL_USAGE_HINT_KEYWORDS):
        return f"{raw_query} 产品使用方法 操作步骤 常见问题 故障排查"
    return raw_query


def heuristic_decompose_query(user_review: str, max_parts: int = 3) -> List[str]:
    raw_query = (user_review or "").strip()
    if not raw_query:
        return []
    fragments = re.split(r"[；;。！？\?]|(?:并且|以及|同时|还有|另外)", raw_query)
    normalized = []
    for fragment in fragments:
        cleaned = re.sub(r"\s+", " ", fragment).strip(" ，,。.？?！!；;")
        if len(cleaned) >= 4 and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized[:max_parts]


def _normalize_queries(queries: Iterable[str], max_items: int) -> List[str]:
    normalized: List[str] = []
    for query in queries:
        cleaned = re.sub(r"\s+", " ", str(query or "")).strip()
        if len(cleaned) < 2:
            continue
        if cleaned in normalized:
            continue
        normalized.append(cleaned)
        if len(normalized) >= max_items:
            break
    return normalized


def _build_first_pass_queries(user_review: str, deterministic_rewrite: str) -> List[str]:
    return _normalize_queries([user_review, deterministic_rewrite], max_items=2)


def plan_retrieval_queries(
    user_review: str,
    query_planner_llm=None,
    max_multi_queries: int = 4,
    max_decomposition_queries: int = 3,
) -> Dict[str, object]:
    raw_query = (user_review or "").strip()
    deterministic_rewrite = rewrite_query(raw_query)

    if query_planner_llm is None:
        heuristic_sub_queries = heuristic_decompose_query(raw_query, max_parts=max_decomposition_queries)
        fallback_multi = _normalize_queries(
            [
                deterministic_rewrite,
                f"{deterministic_rewrite} 关键步骤",
                f"{deterministic_rewrite} 常见故障排查",
            ],
            max_items=max_multi_queries,
        )
        return {
            "rewritten_query": deterministic_rewrite,
            "multi_queries": [query for query in fallback_multi if query != deterministic_rewrite],
            "decomposition_queries": heuristic_sub_queries,
            "needs_decomposition": len(heuristic_sub_queries) >= 2,
        }

    planner_prompt = (
        "你是一个检索规划助手。请针对用户问题生成检索计划，目标是提升说明书检索召回率。\n"
        "要求：\n"
        "1) rewritten_query: 给出一个最适合检索的主查询；\n"
        "2) multi_queries: 给出最多4个不同视角改写查询（短句、去重）；\n"
        "3) decomposition_queries: 如果问题包含多个子问题，拆成最多3个子查询；\n"
        "4) needs_decomposition: 仅当问题复杂且确实需要拆分时置为 true。\n"
        f"用户问题：{raw_query}"
    )
    planner_chain = query_planner_llm.with_structured_output(QueryPlanningResult)
    planning_result = planner_chain.invoke(planner_prompt)

    rewritten_query = re.sub(r"\s+", " ", (planning_result.rewritten_query or "")).strip() or deterministic_rewrite
    multi_queries = _normalize_queries(planning_result.multi_queries, max_items=max_multi_queries)
    decomposition_queries = _normalize_queries(
        planning_result.decomposition_queries,
        max_items=max_decomposition_queries,
    )
    needs_decomposition = bool(planning_result.needs_decomposition)
    if not decomposition_queries and needs_decomposition:
        decomposition_queries = heuristic_decompose_query(raw_query, max_parts=max_decomposition_queries)
    if len(decomposition_queries) < 2:
        needs_decomposition = False

    return {
        "rewritten_query": rewritten_query,
        "multi_queries": [query for query in multi_queries if query != rewritten_query],
        "decomposition_queries": decomposition_queries,
        "needs_decomposition": needs_decomposition,
    }


def _doc_key(doc: Document) -> Tuple[str, str, str]:
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    return doc.page_content, source, page


def _extract_query_terms(query: str) -> List[str]:
    return re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]+", query.lower())


def rrf_fuse_and_rerank(
    query: str,
    dense_docs: Iterable[Document],
    bm25_docs: Iterable[Document],
    top_k: int = 3,
) -> List[Document]:
    ranked_with_scores = rrf_fuse_and_rerank_with_scores(query, dense_docs, bm25_docs, top_k=top_k)
    return [doc for doc, _score in ranked_with_scores]


def rrf_fuse_and_rerank_with_scores(
    query: str,
    dense_docs: Iterable[Document],
    bm25_docs: Iterable[Document],
    top_k: int = 3,
) -> List[Tuple[Document, float]]:
    """融合向量检索和 BM25 检索结果，并做轻量 rerank。"""
    dense_list = list(dense_docs)
    bm25_list = list(bm25_docs)
    query_terms = _extract_query_terms(query)

    scored_docs: Dict[Tuple[str, str, str], Tuple[Document, float]] = {}
    dense_rank = {_doc_key(doc): idx + 1 for idx, doc in enumerate(dense_list)}
    bm25_rank = {_doc_key(doc): idx + 1 for idx, doc in enumerate(bm25_list)}

    for doc in dense_list + bm25_list:
        key = _doc_key(doc)
        rrf_score = 0.0
        if key in dense_rank:
            rrf_score += 1.0 / (60 + dense_rank[key])
        if key in bm25_rank:
            rrf_score += 1.0 / (60 + bm25_rank[key])

        content_lower = doc.page_content.lower()
        overlap_count = sum(1 for term in query_terms if term in content_lower)
        final_score = rrf_score + (0.02 * overlap_count)
        scored_docs[key] = (doc, final_score)

    ranked = sorted(scored_docs.values(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def _extract_vector_documents(vector_store) -> List[Document]:
    docstore = getattr(vector_store, "docstore", None)
    doc_dict = getattr(docstore, "_dict", None)
    if isinstance(doc_dict, dict):
        return [doc for doc in doc_dict.values() if isinstance(doc, Document)]
    return []


def _aggregate_query_results(
    queries: List[str],
    vector_store,
    bm25_retriever,
    per_query_top_k: int = 4,
    dense_k: int = 6,
) -> List[Tuple[Document, float, List[str]]]:
    merged_results: Dict[Tuple[str, str, str], Tuple[Document, float, set[str]]] = {}
    for query in queries:
        dense_docs = vector_store.similarity_search(query, k=dense_k)
        bm25_docs = bm25_retriever.invoke(query) if bm25_retriever else []
        ranked_docs = rrf_fuse_and_rerank_with_scores(
            query,
            dense_docs,
            bm25_docs,
            top_k=per_query_top_k,
        )
        for rank, (doc, score) in enumerate(ranked_docs):
            key = _doc_key(doc)
            boost_score = score + (0.01 / (rank + 1))
            if key in merged_results:
                stored_doc, old_score, query_set = merged_results[key]
                query_set.add(query)
                merged_results[key] = (stored_doc, old_score + boost_score, query_set)
            else:
                merged_results[key] = (doc, boost_score, {query})
    ranked_merged = sorted(merged_results.values(), key=lambda item: item[1], reverse=True)
    return [(doc, score, sorted(list(query_set))) for doc, score, query_set in ranked_merged]


def _merge_ranked_results(
    primary: List[Tuple[Document, float, List[str]]],
    extra: List[Tuple[Document, float, List[str]]],
) -> List[Tuple[Document, float, List[str]]]:
    merged = primary + extra
    deduped: Dict[Tuple[str, str, str], Tuple[Document, float, List[str]]] = {}
    for doc, score, hit_queries in merged:
        key = _doc_key(doc)
        if key in deduped:
            old_doc, old_score, old_queries = deduped[key]
            combined_queries = sorted(set(old_queries + hit_queries))
            deduped[key] = (old_doc, max(old_score, score), combined_queries)
        else:
            deduped[key] = (doc, score, hit_queries)
    return sorted(deduped.values(), key=lambda item: item[1], reverse=True)


def _build_judge_context(
    query: str,
    rewritten_query: str,
    ranked_results: List[Tuple[Document, float, List[str]]],
    max_docs: int = 3,
) -> str:
    doc_lines = []
    for idx, (doc, score, hit_queries) in enumerate(ranked_results[:max_docs], start=1):
        clipped = re.sub(r"\s+", " ", doc.page_content).strip()
        clipped = clipped[:260]
        doc_lines.append(
            f"[候选{idx}] score={score:.4f} 命中查询={','.join(hit_queries[:2])} 内容={clipped}"
        )
    docs_block = "\n".join(doc_lines) if doc_lines else "[无候选文档]"
    return (
        "你是客服检索质量裁判。请判断当前检索候选是否足够回答用户问题。\n"
        "decision 只能是: sufficient / insufficient / out_of_scope。\n"
        "- sufficient: 文档与问题直接相关，足以回答核心问题。\n"
        "- insufficient: 仍在产品范围内，但文档不够相关或信息不足。\n"
        "- out_of_scope: 问题明显超出本产品说明书范围。\n"
        f"用户问题: {query}\n"
        f"当前主查询: {rewritten_query}\n"
        f"候选文档:\n{docs_block}"
    )


def assess_retrieval_quality(
    user_query: str,
    rewritten_query: str,
    ranked_results: List[Tuple[Document, float, List[str]]],
    quality_judge_llm=None,
) -> Dict[str, str]:
    if not ranked_results:
        return {"decision": "insufficient", "reason": "没有召回到文档"}

    top_score = ranked_results[0][1]
    # 高置信命中直接通过，避免每次都依赖 LLM 判定导致过度“检索不足”。
    if len(ranked_results) >= 2 and top_score >= 0.08:
        return {"decision": "sufficient", "reason": "首轮召回命中较好"}
    if quality_judge_llm is None:
        if top_score < 0.05 and len(ranked_results) < 2:
            return {"decision": "insufficient", "reason": "召回分数偏低且候选过少"}
        return {"decision": "sufficient", "reason": "启发式判定候选可用"}

    judge_prompt = _build_judge_context(user_query, rewritten_query, ranked_results)
    judge_chain = quality_judge_llm.with_structured_output(RetrievalAssessmentResult)
    judge_result = judge_chain.invoke(judge_prompt)
    decision = re.sub(r"\s+", "", (judge_result.decision or "").strip().lower())
    if decision not in ("sufficient", "insufficient", "out_of_scope"):
        decision = "insufficient"
    reason = re.sub(r"\s+", " ", (judge_result.reason or "")).strip() or "未提供原因"
    return {"decision": decision, "reason": reason}


def _format_retrieval_output(
    query_strategy: str,
    stage_label: str,
    judge_reason: str,
    ranked_results: List[Tuple[Document, float, List[str]]],
) -> str:
    top_results = ranked_results[:3]
    doc_blocks = []
    for idx, (doc, _score, hit_queries) in enumerate(top_results, start=1):
        source = str(doc.metadata.get("source", "unknown"))
        page = str(doc.metadata.get("page", ""))
        query_hint = " | ".join(hit_queries[:2])
        doc_blocks.append(
            f"文档{idx}（source={source}, page={page}, 命中查询={query_hint}）：{doc.page_content}"
        )
    header = (
        "检索策略："
        f"\n- 查询输入：{query_strategy}"
        f"\n- 当前阶段：{stage_label}"
        f"\n- 质量判定：{judge_reason}"
        "\n- 结果如下：\n"
    )
    return header + "\n\n".join(doc_blocks)


def create_rag_tool(vector_store, query_planner_llm=None):
    """创建一个使用特定向量数据库的 RAG 工具。"""
    vector_docs = _extract_vector_documents(vector_store)
    bm25_retriever = None
    if vector_docs:
        try:
            bm25_retriever = BM25Retriever.from_documents(vector_docs)
            bm25_retriever.k = 6
            logger.info("BM25 检索器初始化成功，文档数=%s", len(vector_docs))
        except Exception as err:
            logger.warning("BM25 检索器初始化失败，将仅使用向量检索: %s", err)

    @tool
    def read_instructions(user_review: str) -> str:
        """
        当用户评论中提到具体产品不会使用的问题时，调用此工具从说明书PDF中检索信息并返回相关内容。
        """
        raw_query = (user_review or "").strip()
        deterministic_rewrite = rewrite_query(raw_query)
        stage1_queries = _build_first_pass_queries(raw_query, deterministic_rewrite)

        logger.info(
            "--- [工具执行] 原查询='%s' | 首轮查询=%s ---",
            raw_query,
            stage1_queries,
        )

        stage1_results = _aggregate_query_results(stage1_queries, vector_store, bm25_retriever)
        stage1_judge = assess_retrieval_quality(
            raw_query,
            deterministic_rewrite,
            stage1_results,
            quality_judge_llm=query_planner_llm,
        )
        if stage1_judge["decision"] == "sufficient":
            return _format_retrieval_output(
                query_strategy=" + ".join(stage1_queries),
                stage_label="original+rewrite",
                judge_reason=stage1_judge["reason"],
                ranked_results=stage1_results,
            )
        if stage1_judge["decision"] == "out_of_scope":
            return "当前问题可能超出本产品说明书范围。请咨询产品使用、维护或售后相关问题。"

        query_plan = plan_retrieval_queries(raw_query, query_planner_llm=query_planner_llm)
        rewritten_query = str(query_plan["rewritten_query"])
        multi_queries = list(query_plan["multi_queries"])
        decomposition_queries = list(query_plan["decomposition_queries"])
        needs_decomposition = bool(query_plan["needs_decomposition"])
        expanded_candidates = [rewritten_query] + multi_queries
        expanded_queries = [q for q in _normalize_queries(expanded_candidates, max_items=5) if q not in stage1_queries]

        logger.info(
            "--- [工具升级] 改写='%s' | multi=%s | decomposition=%s ---",
            rewritten_query,
            multi_queries,
            decomposition_queries,
        )

        if expanded_queries:
            stage2_results = _aggregate_query_results(expanded_queries, vector_store, bm25_retriever)
            stage2_merged = _merge_ranked_results(stage1_results, stage2_results)
            stage2_judge = assess_retrieval_quality(
                raw_query,
                rewritten_query,
                stage2_merged,
                quality_judge_llm=query_planner_llm,
            )
            if stage2_judge["decision"] == "sufficient":
                return _format_retrieval_output(
                    query_strategy=" + ".join(stage1_queries + expanded_queries),
                    stage_label="multi-query",
                    judge_reason=stage2_judge["reason"],
                    ranked_results=stage2_merged,
                )
            if stage2_judge["decision"] == "out_of_scope":
                return "当前问题可能超出本产品说明书范围。请咨询产品使用、维护或售后相关问题。"
        else:
            stage2_merged = stage1_results

        if needs_decomposition and decomposition_queries:
            stage3_results = _aggregate_query_results(decomposition_queries, vector_store, bm25_retriever)
            stage3_merged = _merge_ranked_results(stage2_merged, stage3_results)
            stage3_judge = assess_retrieval_quality(
                raw_query,
                rewritten_query,
                stage3_merged,
                quality_judge_llm=query_planner_llm,
            )
            if stage3_judge["decision"] == "sufficient":
                return _format_retrieval_output(
                    query_strategy=" + ".join(stage1_queries + expanded_queries + decomposition_queries),
                    stage_label="decomposition",
                    judge_reason=stage3_judge["reason"],
                    ranked_results=stage3_merged,
                )
            if stage3_judge["decision"] == "out_of_scope":
                return "当前问题可能超出本产品说明书范围。请咨询产品使用、维护或售后相关问题。"

        return (
            "我已尝试主查询、Multi Query 和查询分解，但仍未检索到足够相关的说明书内容。"
            "请补充具体型号、报错现象或操作步骤后再试。"
        )

    return [read_instructions]
