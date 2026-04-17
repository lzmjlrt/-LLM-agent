import logging
import os
import uuid

from langgraph.checkpoint.memory import InMemorySaver

from agent import config
from agent.event_names import EVENT_RAG_INDEX_LIFECYCLE
from agent.errors import AgentInitializationError
from agent.factories.model_factory import create_llm
from agent.graph_workflow import create_graph
from agent.logging_utils import log_event
from agent.rag.cache import build_faiss_cache_path
from agent.rag_setup import create_rag_tool, get_vector_store
from agent.services.upload_store import persist_uploaded_manual

logger = logging.getLogger(__name__)


def initialize_agent_runtime(
    llm_provider: str,
    llm_api_key: str,
    uploaded_file,
    embedding_provider: str,
    embedding_api_key: str,
    thread_id: str | None = None,
):
    """初始化并返回可调用的图应用。"""
    runtime_thread_id = (thread_id or "").strip() or str(uuid.uuid4())
    try:
        os.makedirs(config.UPLOADS_DIR_PATH, exist_ok=True)
        os.makedirs(config.FAISS_CACHE_DIR_PATH, exist_ok=True)
        os.makedirs(config.CONVERSATION_STORE_DIR_PATH, exist_ok=True)
    except OSError as err:
        raise AgentInitializationError("初始化失败：无法创建缓存目录。", str(err)) from err

    try:
        pdf_path, pdf_hash = persist_uploaded_manual(uploaded_file, runtime_thread_id)
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
    is_incremental = os.path.exists(faiss_path)
    logger.info("RAG 缓存路径: %s", faiss_path)
    log_event(
        EVENT_RAG_INDEX_LIFECYCLE,
        thread_id=runtime_thread_id,
        pdf_hash=pdf_hash[:16],
        embedding_provider=embedding_provider,
        embedding_model=embedding_model_name,
        faiss_path=faiss_path,
        incremental=is_incremental,
    )

    try:
        vector_store = get_vector_store(pdf_path, faiss_path, embedding_provider, embedding_api_key)
    except ValueError as err:
        raise AgentInitializationError("初始化失败：嵌入模型配置无效。", str(err)) from err
    except (OSError, RuntimeError) as err:
        raise AgentInitializationError("初始化失败：知识库构建或加载失败。", str(err)) from err

    tools = create_rag_tool(vector_store)
    checkpointer = InMemorySaver()
    return create_graph(llm, tools, checkpointer=checkpointer)


__all__ = ["initialize_agent_runtime"]
