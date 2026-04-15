import unittest

from langchain_core.documents import Document

from agent.rag.tools import rewrite_query, rrf_fuse_and_rerank


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


if __name__ == "__main__":
    unittest.main()
