from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)


def create_rag_tool(vector_store):
    """创建一个使用特定向量数据库的 RAG 工具。"""

    @tool
    def read_instructions(user_review: str) -> str:
        """
        当用户评论中提到具体产品不会使用的问题时，调用此工具从说明书PDF中检索信息并返回相关内容。
        """
        logger.info("--- [工具执行] 正在为问题 '%s' 检索说明书 ---", user_review)
        similar_docs = vector_store.similarity_search(user_review, k=2)
        doc_input = ""
        for i, doc in enumerate(similar_docs):
            doc_input += f"文档{i + 1}：{doc.page_content}\n\n"
        return doc_input if doc_input else "在说明书中未找到相关内容。"

    return [read_instructions]
