from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatOpenAI

from agent import config


def create_llm(llm_provider: str, llm_api_key: str):
    """根据提供商配置初始化聊天模型。"""
    if llm_provider == "DeepSeek":
        return init_chat_model(
            config.DEEPSEEK_MODEL_NAME,
            model_provider="deepseek",
            base_url=config.DEEPSEEK_BASE_URL,
            api_key=llm_api_key,
        )

    if llm_provider == "OpenAI":
        return ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL_NAME,
            api_key=llm_api_key,
            temperature=0,
        )

    raise ValueError(f"不支持的模型提供商: {llm_provider}")
