from langchain_core.messages import HumanMessage
import logging
import re

from agent.workflow.schema import AgentState

logger = logging.getLogger(__name__)

TOOL_RULE_KEYWORDS = (
    "不会用",
    "怎么用",
    "如何用",
    "怎么安装",
    "如何安装",
    "怎么操作",
    "如何操作",
    "说明书",
    "教程",
    "步骤",
    "更换",
    "充电",
    "电量",
    "续航",
    "刀头",
    "清洗",
    "保养",
    "维护",
    "注意事项",
)

MEANINGFUL_QUERY_HINTS = (
    "什么",
    "哪些",
    "注意",
    "事项",
    "可以",
    "是否",
    "怎么",
    "如何",
    "吗",
    "?",
    "？",
)

HUMAN_REVIEW_HIGH_RISK_KEYWORDS = (
    "退款",
    "退货",
    "投诉",
    "举报",
    "维权",
    "赔偿",
    "起火",
    "爆炸",
    "漏电",
    "割伤",
    "受伤",
    "过敏",
    "安全隐患",
)

USAGE_ACTION_TERMS = (
    "充电",
    "电量",
    "续航",
    "使用",
    "操作",
    "安装",
    "清洗",
    "保养",
    "维护",
    "更换",
    "刀头",
    "注意",
    "事项",
    "指示灯",
    "开机",
    "关机",
)

QUESTION_TERMS = (
    "怎么",
    "如何",
    "怎样",
    "有什么",
    "有哪些",
    "吗",
    "？",
    "?",
)


def detect_usage_question(review_text: str) -> tuple[bool, str]:
    raw_text = (review_text or "").strip()
    if not raw_text:
        return False, "空输入"
    question_like = any(term in raw_text for term in QUESTION_TERMS)
    usage_like = any(term in raw_text for term in USAGE_ACTION_TERMS)
    pattern_match = re.search(r"(怎么|如何|怎样).{0,8}(充电|使用|操作|安装|清洗|保养|更换|维护)", raw_text)

    if (question_like and usage_like) or pattern_match:
        return True, "规则识别为使用咨询问题"
    return False, "未识别为使用咨询问题"


def decide_tool_usage(review_text: str, llm_requires_tool: bool) -> tuple[bool, str]:
    lowered = review_text.lower()
    matched_keywords = [kw for kw in TOOL_RULE_KEYWORDS if kw in lowered]
    if matched_keywords:
        return True, f"规则命中关键词: {', '.join(matched_keywords[:3])}"
    usage_question, usage_reason = detect_usage_question(review_text)
    if usage_question:
        return True, usage_reason
    if llm_requires_tool:
        return True, "LLM判定需要调用工具"
    return False, "LLM判定无需调用工具"


def normalize_quality_for_meaningful_query(review_text: str, llm_quality: str) -> tuple[str, str]:
    """防止有效问题被误判为 default。"""
    normalized_quality = (llm_quality or "").strip().lower()
    raw_text = (review_text or "").strip()
    if normalized_quality != "default":
        return normalized_quality, "保持LLM质量判定"

    if len(raw_text) >= 6 and any(hint in raw_text for hint in MEANINGFUL_QUERY_HINTS):
        return "normal", "规则覆盖：检测到有效提问语句"
    return "default", "保持LLM default判定"


def detect_human_review_need(review_text: str) -> tuple[bool, str]:
    raw_text = (review_text or "").strip().lower()
    matched = [kw for kw in HUMAN_REVIEW_HIGH_RISK_KEYWORDS if kw in raw_text]
    if matched:
        return True, f"命中高风险关键词: {', '.join(matched[:3])}"
    return False, "未命中高风险关键词"


def build_human_handoff_reply(reason: str) -> str:
    return (
        "非常抱歉给您带来不便。您的问题涉及需要人工重点跟进的事项，"
        "我已为您转交人工客服优先处理，请您稍候，我们会尽快与您联系。"
        f"\n\n（转人工原因：{reason}）"
    )


def create_nodes(structred_llm, agent_executor):
    """创建工作流节点。"""

    def analyze_review(state: AgentState) -> dict:
        request_id = state.get("request_id", "n/a")
        logger.info("--- [节点执行] 正在分析评论 --- request_id=%s", request_id)
        review_analysis = structred_llm.invoke(state["original_review"])
        normalized_quality, quality_reason = normalize_quality_for_meaningful_query(
            state["original_review"],
            review_analysis.quality,
        )
        review_analysis.quality = normalized_quality
        final_tool_use, decision_reason = decide_tool_usage(
            state["original_review"],
            bool(review_analysis.require_tool_use),
        )
        review_analysis.require_tool_use = final_tool_use
        needs_human_review, human_reason = detect_human_review_need(state["original_review"])
        logger.info(
            "--- [判定结果] 工具调用=%s, 工具原因=%s, 质量=%s, 质量原因=%s, 需人工=%s, 人工原因=%s --- request_id=%s",
            final_tool_use,
            decision_reason,
            normalized_quality,
            quality_reason,
            needs_human_review,
            human_reason,
            request_id,
        )
        return {
            "review_quality": review_analysis,
            "tool_decision_reason": decision_reason,
            "needs_human_review": needs_human_review,
            "human_review_reason": human_reason,
        }

    def generate_negative_reply(state: AgentState) -> dict:
        request_id = state.get("request_id", "n/a")
        logger.info("--- [节点执行] 正在处理负面评论 --- request_id=%s", request_id)
        if state.get("needs_human_review"):
            human_reason = state.get("human_review_reason", "需人工复核")
            return {"finally_reply": build_human_handoff_reply(human_reason)}
        prompt_content = (
            f"你是一个专业客服。用户消息：'{state['original_review']}'。\n"
            "用户可能是负面反馈，也可能是在咨询产品使用问题。\n"
            "你的任务是输出专业、清晰的中文回复：\n"
            "1. 如果用户是使用咨询（如怎么充电/怎么更换/注意事项），你**必须**调用 `read_instructions` 工具并给出步骤化说明。\n"
            "2. 如果用户是负面反馈（如质量、物流、体验差），先表达歉意，再给出解决方案。\n"
            "3. 如果工具返回内容与问题不相关，不要硬编步骤，改为给出稳妥建议并建议联系客服。\n"
        )
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
        return {"finally_reply": "感谢您的评价！"}

    return {
        "analyze_review": analyze_review,
        "generate_negative_reply": generate_negative_reply,
        "generate_positive_reply": generate_positive_reply,
        "generate_default_reply": generate_default_reply,
    }
