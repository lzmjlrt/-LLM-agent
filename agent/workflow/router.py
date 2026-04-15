from agent.workflow.schema import AgentState
import logging

logger = logging.getLogger(__name__)


def route_after_analysis(state: AgentState) -> str:
    """决策路由。"""
    request_id = state.get("request_id", "n/a")
    decision_reason = state.get("tool_decision_reason", "未记录")
    logger.info("--- [决策] 正在根据分析结果进行路由... --- request_id=%s", request_id)
    analysis_result = state["review_quality"]
    if analysis_result.require_tool_use:
        logger.info(
            "--- [决策结果] -> 需要调用工具 (路由至负面评论处理节点), 原因=%s --- request_id=%s",
            decision_reason,
            request_id,
        )
        return "generate_negative_reply"
    if analysis_result.quality == "default":
        logger.info("--- [决策结果] -> 无效评论 --- request_id=%s", request_id)
        return "generate_default_reply"
    if analysis_result.emotion == "负面":
        logger.info("--- [决策结果] -> 负面评论 --- request_id=%s", request_id)
        return "generate_negative_reply"

    logger.info("--- [决策结果] -> 正面/中性评论 --- request_id=%s", request_id)
    return "generate_positive_reply"
