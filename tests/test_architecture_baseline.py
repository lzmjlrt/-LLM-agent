import unittest

from langchain_core.documents import Document

import agent.graph_workflow as graph_workflow
import agent.rag_setup as rag_setup
from agent.rag.tools import create_rag_tool
from agent.workflow.router import route_after_analysis
from agent.workflow.schema import ReviewQuality


class _FakeDocStore:
    def __init__(self, docs):
        self._dict = {f"doc-{idx}": doc for idx, doc in enumerate(docs)}


class _FakeVectorStore:
    def __init__(self, docs):
        self.docstore = _FakeDocStore(docs)
        self._docs = docs

    def similarity_search(self, _query, k=6):
        return self._docs[:k]


class TestArchitectureBaseline(unittest.TestCase):
    def _state(self, quality: str, emotion: str, require_tool_use: bool):
        return {
            "review_quality": ReviewQuality(
                quality=quality,
                emotion=emotion,
                key_information=[],
                require_tool_use=require_tool_use,
            )
        }

    def test_route_priority_contract(self):
        # 路由顺序契约：tool_use > default > negative > positive/neutral
        self.assertEqual(
            route_after_analysis(self._state("default", "负面", True)),
            "generate_negative_reply",
        )
        self.assertEqual(
            route_after_analysis(self._state("default", "负面", False)),
            "generate_default_reply",
        )
        self.assertEqual(
            route_after_analysis(self._state("normal", "负面", False)),
            "generate_negative_reply",
        )
        self.assertEqual(
            route_after_analysis(self._state("normal", "中性", False)),
            "generate_positive_reply",
        )

    def test_review_quality_fields_contract(self):
        self.assertEqual(
            set(ReviewQuality.model_fields.keys()),
            {"quality", "emotion", "key_information", "require_tool_use"},
        )

    def test_compatibility_exports_contract(self):
        self.assertEqual(
            set(graph_workflow.__all__),
            {"create_graph", "route_after_analysis", "AgentState", "ReviewQuality"},
        )
        self.assertEqual(
            set(rag_setup.__all__),
            {"get_vector_store", "create_rag_tool"},
        )

    def test_rag_tool_name_contract(self):
        vector_store = _FakeVectorStore(
            [
                Document(page_content="剃须刀充电步骤", metadata={"source": "es.pdf", "page": 1}),
                Document(page_content="刀头更换说明", metadata={"source": "es.pdf", "page": 2}),
            ]
        )
        tools = create_rag_tool(vector_store)
        self.assertEqual(len(tools), 1)
        self.assertEqual(getattr(tools[0], "name", ""), "read_instructions")


if __name__ == "__main__":
    unittest.main()
