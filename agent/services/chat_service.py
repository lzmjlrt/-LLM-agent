import logging
import os
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from openai import APIError, APITimeoutError, AuthenticationError, BadRequestError, RateLimitError

from agent import config
from agent.errors import AgentInitializationError, AgentRuntimeError
from agent.factories.model_factory import create_llm
from agent.graph_workflow import create_graph
from agent.rag.cache import build_faiss_cache_path
from agent.rag_setup import create_rag_tool, get_vector_store

logger = logging.getLogger(__name__)


def initialize_agent_runtime(
    llm_provider: str,
    llm_api_key: str,
    uploaded_file,
    embedding_provider: str,
    embedding_api_key: str,
):
    """初始化并返回可调用的图应用。"""
    try:
        os.makedirs(config.TEMP_DIR_PATH, exist_ok=True)
    except OSError as err:
        raise AgentInitializationError("初始化失败：无法创建临时目录。", str(err)) from err

    pdf_path = os.path.join(config.TEMP_DIR_PATH, uploaded_file.name)
    try:
        with open(pdf_path, "wb") as file_obj:
            file_obj.write(uploaded_file.getbuffer())
    except OSError as err:
        raise AgentInitializationError("初始化失败：无法保存上传的 PDF。", str(err)) from err

    try:
        llm = create_llm(llm_provider, llm_api_key)
    except ValueError as err:
        raise AgentInitializationError("初始化失败：模型配置无效。", str(err)) from err

    embedding_model_name = (
        config.DASHSCOPE_EMBEDDING_MODEL_NAME
        if embedding_provider == "DashScope (Alibaba)"
        else config.OPENAI_EMBEDDING_MODEL_NAME
    )
    faiss_path = build_faiss_cache_path(pdf_path, embedding_provider, embedding_model_name)
    logger.info("RAG 缓存路径: %s", faiss_path)

    try:
        vector_store = get_vector_store(pdf_path, faiss_path, embedding_provider, embedding_api_key)
    except ValueError as err:
        raise AgentInitializationError("初始化失败：嵌入模型配置无效。", str(err)) from err
    except (OSError, RuntimeError) as err:
        raise AgentInitializationError("初始化失败：知识库构建或加载失败。", str(err)) from err

    tools = create_rag_tool(vector_store)
    checkpointer = InMemorySaver()
    return create_graph(llm, tools, checkpointer=checkpointer)


def invoke_agent(app, user_prompt: str, thread_id: str):
    request_id = str(uuid.uuid4())
    logger.info("开始处理请求 request_id=%s thread_id=%s", request_id, thread_id)
    try:
        result = app.invoke(
            {"original_review": user_prompt, "request_id": request_id},
            config={"configurable": {"thread_id": thread_id}},
        )
        reply = result.get("finally_reply", "抱歉，我暂时无法生成回复，请稍后重试。")
        logger.info("请求完成 request_id=%s thread_id=%s", request_id, thread_id)
        return reply, request_id
    except (BadRequestError, AuthenticationError, RateLimitError, APITimeoutError, APIError) as err:
        logger.exception("模型服务错误 request_id=%s thread_id=%s", request_id, thread_id)
        raise AgentRuntimeError(
            "请求模型服务失败，请检查模型配置或稍后重试。",
            f"request_id={request_id}\nthread_id={thread_id}\n{err}",
        ) from err
    except (ValueError, RuntimeError) as err:
        logger.exception("业务处理错误 request_id=%s thread_id=%s", request_id, thread_id)
        raise AgentRuntimeError(
            "处理请求时发生错误。",
            f"request_id={request_id}\nthread_id={thread_id}\n{err}",
        ) from err
