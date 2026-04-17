import logging
import os
import uuid
import hashlib
import json
import time
from datetime import UTC, datetime

from langgraph.checkpoint.memory import InMemorySaver
from openai import APIError, APITimeoutError, AuthenticationError, BadRequestError, RateLimitError

from agent import config
from agent.errors import AgentInitializationError, AgentRuntimeError
from agent.factories.model_factory import create_llm
from agent.graph_workflow import create_graph
from agent.logging_utils import log_event
from agent.rag.cache import build_faiss_cache_path
from agent.rag_setup import create_rag_tool, get_vector_store

logger = logging.getLogger(__name__)


def _sanitize_filename(file_name: str) -> str:
    raw_name = (file_name or "uploaded_manual.pdf").strip()
    safe_chars = []
    for char in raw_name:
        if char.isalnum() or char in ("-", "_", "."):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars).strip("._")
    return sanitized or "uploaded_manual.pdf"


def _persist_uploaded_manual(uploaded_file, thread_id: str) -> tuple[str, str]:
    file_bytes = bytes(uploaded_file.getbuffer())
    pdf_hash = hashlib.sha256(file_bytes).hexdigest()

    upload_dir = os.path.join(config.UPLOADS_DIR_PATH, thread_id)
    os.makedirs(upload_dir, exist_ok=True)

    safe_name = _sanitize_filename(getattr(uploaded_file, "name", "manual.pdf"))
    pdf_name = f"{pdf_hash[:12]}_{safe_name}"
    pdf_path = os.path.join(upload_dir, pdf_name)
    if not os.path.exists(pdf_path):
        with open(pdf_path, "wb") as file_obj:
            file_obj.write(file_bytes)

    metadata = {
        "thread_id": thread_id,
        "pdf_hash": pdf_hash,
        "filename": safe_name,
        "saved_path": pdf_path,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    metadata_path = os.path.join(upload_dir, "index_meta.json")
    with open(metadata_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    return pdf_path, pdf_hash


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
    except OSError as err:
        raise AgentInitializationError("初始化失败：无法创建缓存目录。", str(err)) from err

    try:
        pdf_path, pdf_hash = _persist_uploaded_manual(uploaded_file, runtime_thread_id)
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
        "rag_index_lifecycle",
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


def invoke_agent(app, user_prompt: str, thread_id: str):
    request_id = str(uuid.uuid4())
    start_ts = time.perf_counter()
    logger.info("开始处理请求 request_id=%s thread_id=%s", request_id, thread_id)
    try:
        result = app.invoke(
            {"original_review": user_prompt, "request_id": request_id},
            config={"configurable": {"thread_id": thread_id}},
        )
        reply = result.get("finally_reply", "抱歉，我暂时无法生成回复，请稍后重试。")
        duration_ms = int((time.perf_counter() - start_ts) * 1000)
        logger.info("请求完成 request_id=%s thread_id=%s", request_id, thread_id)
        log_event(
            "agent_invoke_success",
            request_id=request_id,
            thread_id=thread_id,
            duration_ms=duration_ms,
            query_length=len(user_prompt or ""),
            reply_length=len(reply or ""),
        )
        return reply, request_id
    except (BadRequestError, AuthenticationError, RateLimitError, APITimeoutError, APIError) as err:
        logger.exception("模型服务错误 request_id=%s thread_id=%s", request_id, thread_id)
        duration_ms = int((time.perf_counter() - start_ts) * 1000)
        log_event(
            "agent_invoke_model_error",
            request_id=request_id,
            thread_id=thread_id,
            duration_ms=duration_ms,
            error_type=type(err).__name__,
        )
        raise AgentRuntimeError(
            "请求模型服务失败，请检查模型配置或稍后重试。",
            f"request_id={request_id}\nthread_id={thread_id}\n{err}",
        ) from err
    except (ValueError, RuntimeError) as err:
        logger.exception("业务处理错误 request_id=%s thread_id=%s", request_id, thread_id)
        duration_ms = int((time.perf_counter() - start_ts) * 1000)
        log_event(
            "agent_invoke_business_error",
            request_id=request_id,
            thread_id=thread_id,
            duration_ms=duration_ms,
            error_type=type(err).__name__,
        )
        raise AgentRuntimeError(
            "处理请求时发生错误。",
            f"request_id={request_id}\nthread_id={thread_id}\n{err}",
        ) from err
