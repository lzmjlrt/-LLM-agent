import logging
import time
import uuid

from openai import APIError, APITimeoutError, AuthenticationError, BadRequestError, RateLimitError

from agent import config
from agent.event_names import (
    EVENT_AGENT_INVOKE_BUSINESS_ERROR,
    EVENT_AGENT_INVOKE_MODEL_ERROR,
    EVENT_AGENT_INVOKE_SUCCESS,
    EVENT_CONVERSATION_MEMORY_LOADED,
    EVENT_CONVERSATION_MEMORY_UPDATED,
    EVENT_SEMANTIC_CACHE_HIT,
)
from agent.errors import AgentRuntimeError
from agent.logging_utils import log_event
from agent.services.conversation_memory import (
    append_conversation_turn,
    build_conversation_context,
    find_cached_reply,
    load_thread_memory,
    save_thread_memory,
)

logger = logging.getLogger(__name__)


def invoke_agent(app, user_prompt: str, thread_id: str):
    request_id = str(uuid.uuid4())
    start_ts = time.perf_counter()
    logger.info("开始处理请求 request_id=%s thread_id=%s", request_id, thread_id)
    try:
        memory = load_thread_memory(thread_id)
        log_event(
            EVENT_CONVERSATION_MEMORY_LOADED,
            request_id=request_id,
            thread_id=thread_id,
            turn_count=len(memory.get("turns", [])),
            cache_count=len(memory.get("qa_cache", [])),
        )

        if config.SEMANTIC_CACHE_ENABLED:
            cached_reply, similarity_score = find_cached_reply(
                memory,
                user_prompt,
                config.SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
            )
            if cached_reply:
                memory = append_conversation_turn(
                    memory,
                    user_prompt,
                    cached_reply,
                    config.CONVERSATION_HISTORY_LIMIT,
                )
                save_thread_memory(thread_id, memory)
                duration_ms = int((time.perf_counter() - start_ts) * 1000)
                log_event(
                    EVENT_SEMANTIC_CACHE_HIT,
                    request_id=request_id,
                    thread_id=thread_id,
                    duration_ms=duration_ms,
                    similarity=round(similarity_score, 4),
                )
                return cached_reply, request_id

        conversation_context = build_conversation_context(memory)
        invoke_payload = {
            "original_review": user_prompt,
            "request_id": request_id,
        }
        if conversation_context:
            invoke_payload["conversation_context"] = conversation_context

        result = app.invoke(
            invoke_payload,
            config={"configurable": {"thread_id": thread_id}},
        )
        reply = result.get("finally_reply", "抱歉，我暂时无法生成回复，请稍后重试。")
        memory = append_conversation_turn(
            memory,
            user_prompt,
            reply,
            config.CONVERSATION_HISTORY_LIMIT,
        )
        save_thread_memory(thread_id, memory)
        duration_ms = int((time.perf_counter() - start_ts) * 1000)
        logger.info("请求完成 request_id=%s thread_id=%s", request_id, thread_id)
        log_event(
            EVENT_AGENT_INVOKE_SUCCESS,
            request_id=request_id,
            thread_id=thread_id,
            duration_ms=duration_ms,
            query_length=len(user_prompt or ""),
            reply_length=len(reply or ""),
        )
        log_event(
            EVENT_CONVERSATION_MEMORY_UPDATED,
            request_id=request_id,
            thread_id=thread_id,
            turn_count=len(memory.get("turns", [])),
            cache_count=len(memory.get("qa_cache", [])),
        )
        return reply, request_id
    except (BadRequestError, AuthenticationError, RateLimitError, APITimeoutError, APIError) as err:
        logger.exception("模型服务错误 request_id=%s thread_id=%s", request_id, thread_id)
        duration_ms = int((time.perf_counter() - start_ts) * 1000)
        log_event(
            EVENT_AGENT_INVOKE_MODEL_ERROR,
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
            EVENT_AGENT_INVOKE_BUSINESS_ERROR,
            request_id=request_id,
            thread_id=thread_id,
            duration_ms=duration_ms,
            error_type=type(err).__name__,
        )
        raise AgentRuntimeError(
            "处理请求时发生错误。",
            f"request_id={request_id}\nthread_id={thread_id}\n{err}",
        ) from err


__all__ = ["invoke_agent"]
