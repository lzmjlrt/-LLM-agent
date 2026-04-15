import logging
import re
from typing import Dict, Iterable, List, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.tools import tool

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


def rewrite_query(user_review: str) -> str:
    """将用户原评论重写为更适合检索说明书的查询。"""
    raw_query = (user_review or "").strip()
    if not raw_query:
        return ""

    if any(keyword in raw_query for keyword in TOOL_USAGE_HINT_KEYWORDS):
        return f"{raw_query} 产品使用方法 操作步骤 常见问题 故障排查"
    return raw_query


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
    return [doc for doc, _score in ranked[:top_k]]


def _extract_vector_documents(vector_store) -> List[Document]:
    docstore = getattr(vector_store, "docstore", None)
    doc_dict = getattr(docstore, "_dict", None)
    if isinstance(doc_dict, dict):
        return [doc for doc in doc_dict.values() if isinstance(doc, Document)]
    return []


def create_rag_tool(vector_store):
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
        rewritten_query = rewrite_query(user_review)
        logger.info("--- [工具执行] 原查询='%s' | 重写查询='%s' ---", user_review, rewritten_query)

        dense_docs = vector_store.similarity_search(rewritten_query, k=6)
        bm25_docs = bm25_retriever.invoke(rewritten_query) if bm25_retriever else []
        similar_docs = rrf_fuse_and_rerank(rewritten_query, dense_docs, bm25_docs, top_k=3)

        if not similar_docs:
            similar_docs = dense_docs[:2]

        doc_input = ""
        for i, doc in enumerate(similar_docs):
            doc_input += f"文档{i + 1}：{doc.page_content}\n\n"
        return doc_input if doc_input else "在说明书中未找到相关内容。"

    return [read_instructions]
