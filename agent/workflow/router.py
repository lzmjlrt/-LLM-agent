from agent.workflow.schema import AgentState
import logging

from agent.logging_utils import log_event

logger = logging.getLogger(__name__)


def route_after_analysis(state: AgentState) -> str:
    """决策路由。"""
    request_id = state.get("request_id", "n/a")
    decision_reason = state.get("tool_decision_reason", "未记录")
    query_intent = state.get("query_intent", "unknown")
    intent_reason = state.get("intent_reason", "未记录")
    logger.info("--- [决策] 正在根据分析结果进行路由... --- request_id=%s", request_id)
    analysis_result = state["review_quality"]
    if analysis_result.require_tool_use:
        log_event(
            "route_decision",
            request_id=request_id,
            route_target="generate_negative_reply",
            reason="tool_use",
            intent=query_intent,
        )
        logger.info(
            "--- [决策结果] -> 需要调用工具 (路由至负面评论处理节点), 意图=%s, 意图原因=%s, 原因=%s --- request_id=%s",
            query_intent,
            intent_reason,
            decision_reason,
            request_id,
        )
        return "generate_negative_reply"
    if analysis_result.quality == "default":
        log_event(
            "route_decision",
            request_id=request_id,
            route_target="generate_default_reply",
            reason="quality_default",
            intent=query_intent,
        )
        logger.info("--- [决策结果] -> 无效评论 --- request_id=%s", request_id)
        return "generate_default_reply"
    if analysis_result.emotion == "负面":
        log_event(
            "route_decision",
            request_id=request_id,
            route_target="generate_negative_reply",
            reason="negative_emotion",
            intent=query_intent,
        )
        logger.info("--- [决策结果] -> 负面评论 --- request_id=%s", request_id)
        return "generate_negative_reply"

    log_event(
        "route_decision",
        request_id=request_id,
        route_target="generate_positive_reply",
        reason="positive_or_neutral",
        intent=query_intent,
    )
    logger.info("--- [决策结果] -> 正面/中性评论 --- request_id=%s", request_id)
    return "generate_positive_reply"
