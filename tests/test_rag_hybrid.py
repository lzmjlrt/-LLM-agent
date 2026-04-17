import unittest

from langchain_core.documents import Document

from agent.rag.tools import (
    _build_first_pass_queries,
    QueryPlanningResult,
    RetrievalAssessmentResult,
    assess_retrieval_quality,
    heuristic_decompose_query,
    plan_retrieval_queries,
    rewrite_query,
    rrf_fuse_and_rerank,
)


class FakeStructuredPlanner:
    def __init__(self, result):
        self._result = result

    def invoke(self, _):
        return self._result


class FakePlannerLLM:
    def __init__(self, result):
        self._result = result

    def with_structured_output(self, _):
        return FakeStructuredPlanner(self._result)


class FakeJudgeLLM:
    def __init__(self, result):
        self._result = result

    def with_structured_output(self, _):
        return FakeStructuredPlanner(self._result)


class FailIfInvokedStructuredJudge:
    def invoke(self, _):
        raise AssertionError("LLM judge should not be invoked for high-confidence retrieval")


class FailIfInvokedJudgeLLM:
    def with_structured_output(self, _):
        return FailIfInvokedStructuredJudge()


class TestRagHybridRetrieval(unittest.TestCase):
    def test_rewrite_query_for_usage_issue(self):
        rewritten = rewrite_query("这个产品不会用，怎么安装？")
        self.assertIn("产品使用方法", rewritten)
        self.assertIn("操作步骤", rewritten)

    def test_rrf_fuse_and_rerank_merges_dense_and_bm25(self):
        dense_docs = [
            Document(page_content="更换刀头步骤说明", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="清洗方法与保养", metadata={"source": "a.pdf", "page": 2}),
        ]
        bm25_docs = [
            Document(page_content="安装教程与常见问题", metadata={"source": "b.pdf", "page": 1}),
            Document(page_content="更换刀头步骤说明", metadata={"source": "a.pdf", "page": 1}),
        ]

        ranked = rrf_fuse_and_rerank(
            query="如何安装和更换刀头",
            dense_docs=dense_docs,
            bm25_docs=bm25_docs,
            top_k=3,
        )
        self.assertTrue(len(ranked) >= 2)
        merged_contents = [doc.page_content for doc in ranked]
        self.assertIn("更换刀头步骤说明", merged_contents)
        self.assertIn("安装教程与常见问题", merged_contents)

    def test_heuristic_decompose_query(self):
        sub_queries = heuristic_decompose_query("怎么充电，同时怎么更换刀头？另外清洗步骤是什么？")
        self.assertTrue(len(sub_queries) >= 2)

    def test_first_pass_queries_contains_original_and_rewrite(self):
        queries = _build_first_pass_queries(
            "剃须刀怎么充电",
            "剃须刀怎么充电 产品使用方法 操作步骤 常见问题 故障排查",
        )
        self.assertEqual(len(queries), 2)
        self.assertEqual(queries[0], "剃须刀怎么充电")

    def test_plan_retrieval_queries_with_llm_multi_and_decomposition(self):
        fake_plan = QueryPlanningResult(
            rewritten_query="剃须刀充电与刀头更换步骤",
            multi_queries=[
                "剃须刀怎么充电",
                "剃须刀刀头如何更换",
                "剃须刀充电注意事项",
            ],
            decomposition_queries=[
                "剃须刀充电步骤",
                "剃须刀更换刀头步骤",
            ],
            needs_decomposition=True,
        )
        planner_llm = FakePlannerLLM(fake_plan)
        plan = plan_retrieval_queries(
            "这款剃须刀怎么充电和更换刀头？",
            query_planner_llm=planner_llm,
        )
        self.assertEqual(plan["rewritten_query"], "剃须刀充电与刀头更换步骤")
        self.assertTrue(len(plan["multi_queries"]) >= 2)
        self.assertTrue(plan["needs_decomposition"])
        self.assertTrue(len(plan["decomposition_queries"]) >= 2)

    def test_assess_retrieval_quality_heuristic_for_empty_results(self):
        decision = assess_retrieval_quality(
            user_query="剃须刀怎么充电",
            rewritten_query="剃须刀充电步骤",
            ranked_results=[],
            quality_judge_llm=None,
        )
        self.assertEqual(decision["decision"], "insufficient")

    def test_assess_retrieval_quality_with_llm_out_of_scope(self):
        judge_llm = FakeJudgeLLM(
            RetrievalAssessmentResult(
                decision="out_of_scope",
                reason="问题与产品说明书无关",
            )
        )
        ranked_results = [
            (Document(page_content="刀头更换", metadata={"source": "a.pdf", "page": 1}), 0.12, ["刀头更换"])
        ]
        decision = assess_retrieval_quality(
            user_query="今天上海天气如何",
            rewritten_query="上海天气",
            ranked_results=ranked_results,
            quality_judge_llm=judge_llm,
        )
        self.assertEqual(decision["decision"], "out_of_scope")
        self.assertIn("无关", decision["reason"])

    def test_assess_retrieval_quality_skips_llm_for_high_confidence(self):
        ranked_results = [
            (Document(page_content="充电步骤详解", metadata={"source": "a.pdf", "page": 1}), 0.21, ["剃须刀怎么充电"]),
            (Document(page_content="充电指示灯说明", metadata={"source": "a.pdf", "page": 2}), 0.13, ["剃须刀怎么充电"]),
        ]
        decision = assess_retrieval_quality(
            user_query="剃须刀怎么充电",
            rewritten_query="剃须刀充电步骤",
            ranked_results=ranked_results,
            quality_judge_llm=FailIfInvokedJudgeLLM(),
        )
        self.assertEqual(decision["decision"], "sufficient")


if __name__ == "__main__":
    unittest.main()
