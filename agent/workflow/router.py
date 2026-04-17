import logging

from agent.event_names import EVENT_ROUTE_DECISION
from agent.logging_utils import log_event
from agent.workflow.constants import (
    EMOTION_NEGATIVE,
    INTENT_GENERAL_QUERY,
    QUALITY_DEFAULT,
    ROUTE_GENERATE_DEFAULT_REPLY,
    ROUTE_GENERATE_NEGATIVE_REPLY,
    ROUTE_GENERATE_POSITIVE_REPLY,
    ROUTE_REASON_NEGATIVE_EMOTION,
    ROUTE_REASON_POSITIVE_OR_NEUTRAL,
    ROUTE_REASON_QUALITY_DEFAULT,
    ROUTE_REASON_TOOL_USE,
)
from agent.workflow.schema import AgentState

logger = logging.getLogger(__name__)


def route_after_analysis(state: AgentState) -> str:
    """决策路由。"""
    request_id = state.get("request_id", "n/a")
    decision_reason = state.get("tool_decision_reason", "未记录")
    query_intent = state.get("query_intent", INTENT_GENERAL_QUERY)
    intent_reason = state.get("intent_reason", "未记录")
    logger.info("--- [决策] 正在根据分析结果进行路由... --- request_id=%s", request_id)
    analysis_result = state["review_quality"]
    if analysis_result.require_tool_use:
        log_event(
            EVENT_ROUTE_DECISION,
            request_id=request_id,
            route_target=ROUTE_GENERATE_NEGATIVE_REPLY,
            reason=ROUTE_REASON_TOOL_USE,
            intent=query_intent,
        )
        logger.info(
            "--- [决策结果] -> 需要调用工具 (路由至负面评论处理节点), 意图=%s, 意图原因=%s, 原因=%s --- request_id=%s",
            query_intent,
            intent_reason,
            decision_reason,
            request_id,
        )
        return ROUTE_GENERATE_NEGATIVE_REPLY
    if analysis_result.quality == QUALITY_DEFAULT:
        log_event(
            EVENT_ROUTE_DECISION,
            request_id=request_id,
            route_target=ROUTE_GENERATE_DEFAULT_REPLY,
            reason=ROUTE_REASON_QUALITY_DEFAULT,
            intent=query_intent,
        )
        logger.info("--- [决策结果] -> 无效评论 --- request_id=%s", request_id)
        return ROUTE_GENERATE_DEFAULT_REPLY
    if analysis_result.emotion == EMOTION_NEGATIVE:
        log_event(
            EVENT_ROUTE_DECISION,
            request_id=request_id,
            route_target=ROUTE_GENERATE_NEGATIVE_REPLY,
            reason=ROUTE_REASON_NEGATIVE_EMOTION,
            intent=query_intent,
        )
        logger.info("--- [决策结果] -> 负面评论 --- request_id=%s", request_id)
        return ROUTE_GENERATE_NEGATIVE_REPLY

    log_event(
        EVENT_ROUTE_DECISION,
        request_id=request_id,
        route_target=ROUTE_GENERATE_POSITIVE_REPLY,
        reason=ROUTE_REASON_POSITIVE_OR_NEUTRAL,
        intent=query_intent,
    )
    logger.info("--- [决策结果] -> 正面/中性评论 --- request_id=%s", request_id)
    return ROUTE_GENERATE_POSITIVE_REPLY
