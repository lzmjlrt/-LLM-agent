import logging

from langchain_core.messages import HumanMessage

from agent.event_names import (
    EVENT_ANALYZE_RESULT,
    EVENT_ANSWER_VALIDATION,
)
from agent.logging_utils import log_event
from agent.workflow.constants import (
    INTENT_ADDITIONAL_QUERY,
    INTENT_GENERAL_QUERY,
    INTENT_GRAPHRAG_QUERY,
    QUALITY_DEFAULT,
    ROUTE_GENERATE_DEFAULT_REPLY,
    ROUTE_GENERATE_NEGATIVE_REPLY,
    ROUTE_GENERATE_POSITIVE_REPLY,
)
from agent.workflow.intent_rules import (
    classify_query_intent,
    decide_tool_usage,
    detect_human_review_need,
    normalize_quality_for_meaningful_query,
)
from agent.workflow.reply_policies import (
    build_additional_info_reply,
    build_human_handoff_reply,
    build_out_of_scope_reply,
    validate_generated_reply,
)
from agent.workflow.schema import AgentState

logger = logging.getLogger(__name__)


def create_nodes(structred_llm, agent_executor):
    """创建工作流节点。"""

    def analyze_review(state: AgentState) -> dict:
        request_id = state.get("request_id", "n/a")
        logger.info("--- [节点执行] 正在分析评论 --- request_id=%s", request_id)
        original_review = state["original_review"]
        review_analysis = structred_llm.invoke(original_review)
        query_intent, intent_reason = classify_query_intent(original_review)

        normalized_quality, quality_reason = normalize_quality_for_meaningful_query(
            original_review,
            review_analysis.quality,
        )
        review_analysis.quality = normalized_quality
        final_tool_use, decision_reason = decide_tool_usage(
            original_review,
            bool(review_analysis.require_tool_use),
        )
        needs_additional_info = query_intent == INTENT_ADDITIONAL_QUERY
        out_of_scope = "超出" in intent_reason

        if query_intent == INTENT_GRAPHRAG_QUERY:
            review_analysis.require_tool_use = True
            decision_reason = f"{decision_reason}; 意图={query_intent}"
        elif query_intent == INTENT_ADDITIONAL_QUERY:
            review_analysis.quality = QUALITY_DEFAULT
            review_analysis.require_tool_use = False
        else:
            review_analysis.require_tool_use = final_tool_use
            if out_of_scope:
                review_analysis.quality = QUALITY_DEFAULT

        needs_human_review, human_reason = detect_human_review_need(original_review)
        logger.info(
            "--- [判定结果] 意图=%s, 意图原因=%s, 工具调用=%s, 工具原因=%s, 质量=%s, 质量原因=%s, 需补充=%s, 超范围=%s, 需人工=%s, 人工原因=%s --- request_id=%s",
            query_intent,
            intent_reason,
            review_analysis.require_tool_use,
            decision_reason,
            review_analysis.quality,
            quality_reason,
            needs_additional_info,
            out_of_scope,
            needs_human_review,
            human_reason,
            request_id,
        )
        log_event(
            EVENT_ANALYZE_RESULT,
            request_id=request_id,
            query_intent=query_intent,
            intent_reason=intent_reason,
            quality=review_analysis.quality,
            emotion=review_analysis.emotion,
            require_tool_use=review_analysis.require_tool_use,
            needs_additional_info=needs_additional_info,
            out_of_scope=out_of_scope,
            needs_human_review=needs_human_review,
        )
        return {
            "review_quality": review_analysis,
            "tool_decision_reason": decision_reason,
            "query_intent": query_intent,
            "intent_reason": intent_reason,
            "needs_additional_info": needs_additional_info,
            "out_of_scope": out_of_scope,
            "needs_human_review": needs_human_review,
            "human_review_reason": human_reason,
        }

    def generate_negative_reply(state: AgentState) -> dict:
        request_id = state.get("request_id", "n/a")
        logger.info("--- [节点执行] 正在处理负面评论 --- request_id=%s", request_id)
        if state.get("needs_human_review"):
            human_reason = state.get("human_review_reason", "需人工复核")
            return {"finally_reply": build_human_handoff_reply(human_reason)}
        conversation_context = (state.get("conversation_context") or "").strip()
        prompt_content = (
            f"你是一个专业客服。用户消息：'{state['original_review']}'。\n"
            "用户可能是负面反馈，也可能是在咨询产品使用问题。\n"
            "你的任务是输出专业、清晰的中文回复：\n"
            "1. 如果用户是使用咨询（如怎么充电/怎么更换/注意事项），你**必须**调用 `read_instructions` 工具并给出步骤化说明。\n"
            "2. 如果用户是负面反馈（如质量、物流、体验差），先表达歉意，再给出解决方案。\n"
            "3. 如果工具返回内容与问题不相关，不要硬编步骤，改为给出稳妥建议并建议联系客服。\n"
        )
        if conversation_context:
            prompt_content += f"\n历史会话参考信息（仅供辅助判断）：\n{conversation_context}\n"
        response = agent_executor.invoke({"messages": [HumanMessage(content=prompt_content)]})
        return {"finally_reply": response["messages"][-1].content}

    def generate_positive_reply(state: AgentState) -> dict:
        request_id = state.get("request_id", "n/a")
        logger.info("--- [节点执行] 正在处理正面/中性评论 --- request_id=%s", request_id)
        if state.get("needs_human_review"):
            human_reason = state.get("human_review_reason", "需人工复核")
            return {"finally_reply": build_human_handoff_reply(human_reason)}
        return {"finally_reply": "亲，非常感谢您的认可与支持！您的满意是我们不断前行的动力，期待您的再次光临！"}

    def generate_default_reply(state: AgentState) -> dict:
        request_id = state.get("request_id", "n/a")
        logger.info("--- [节点执行] 正在处理无效评论 --- request_id=%s", request_id)
        if state.get("needs_human_review"):
            human_reason = state.get("human_review_reason", "需人工复核")
            return {"finally_reply": build_human_handoff_reply(human_reason)}
        if state.get("needs_additional_info"):
            return {"finally_reply": build_additional_info_reply()}
        if state.get("out_of_scope"):
            return {"finally_reply": build_out_of_scope_reply()}
        return {"finally_reply": "感谢您的评价！"}

    def validate_reply(state: AgentState) -> dict:
        request_id = state.get("request_id", "n/a")
        query_intent = state.get("query_intent", INTENT_GENERAL_QUERY)
        valid, reason, final_reply = validate_generated_reply(
            original_review=state.get("original_review", ""),
            generated_reply=state.get("finally_reply", ""),
            query_intent=query_intent,
            needs_additional_info=bool(state.get("needs_additional_info", False)),
            out_of_scope=bool(state.get("out_of_scope", False)),
        )
        if not valid:
            logger.warning("--- [回复校验] 未通过, 原因=%s --- request_id=%s", reason, request_id)
        else:
            logger.info("--- [回复校验] 通过, 原因=%s --- request_id=%s", reason, request_id)

        log_event(
            EVENT_ANSWER_VALIDATION,
            request_id=request_id,
            query_intent=query_intent,
            answer_valid=valid,
            reason=reason,
        )
        return {
            "finally_reply": final_reply,
            "answer_valid": valid,
            "answer_validation_reason": reason,
        }

    return {
        "analyze_review": analyze_review,
        ROUTE_GENERATE_NEGATIVE_REPLY: generate_negative_reply,
        ROUTE_GENERATE_POSITIVE_REPLY: generate_positive_reply,
        ROUTE_GENERATE_DEFAULT_REPLY: generate_default_reply,
        "validate_reply": validate_reply,
    }


__all__ = ["create_nodes"]
