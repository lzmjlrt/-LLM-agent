from langchain_community.embeddings import DashScopeEmbeddings, OpenAIEmbeddings

from agent import config


def create_embeddings(embedding_provider: str, embedding_api_key: str):
    """根据提供商创建嵌入模型。"""
    if embedding_provider == "DashScope (Alibaba)":
        return DashScopeEmbeddings(
            model=config.DASHSCOPE_EMBEDDING_MODEL_NAME,
            dashscope_api_key=embedding_api_key,
        )

    if embedding_provider == "OpenAI":
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL_NAME,
            api_key=embedding_api_key,
        )

    raise ValueError(f"不支持的嵌入模型提供商: {embedding_provider}")
