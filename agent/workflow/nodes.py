from langchain_core.messages import HumanMessage
import logging
import re

from agent.logging_utils import log_event
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

SCOPE_HINT_KEYWORDS = (
    "剃须刀",
    "刀头",
    "充电",
    "电量",
    "续航",
    "说明书",
    "安装",
    "操作",
    "使用",
    "清洗",
    "保养",
    "维护",
    "注意事项",
    "客服",
    "售后",
    "退换",
    "维修",
)

OUT_OF_SCOPE_HINT_KEYWORDS = (
    "天气",
    "股票",
    "电影",
    "明星",
    "体育",
    "足球",
    "篮球",
    "政治",
    "编程",
    "代码",
    "数学",
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


def is_question_like(review_text: str) -> bool:
    raw_text = (review_text or "").strip()
    if not raw_text:
        return False
    return any(term in raw_text for term in QUESTION_TERMS)


def is_in_scope_query(review_text: str) -> bool:
    raw_text = (review_text or "").strip().lower()
    if not raw_text:
        return False
    return any(keyword in raw_text for keyword in SCOPE_HINT_KEYWORDS)


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


def detect_additional_info_need(review_text: str) -> tuple[bool, str]:
    raw_text = (review_text or "").strip()
    if not raw_text:
        return True, "输入为空，需补充问题描述"
    if not is_question_like(raw_text):
        return False, "非问句，无需补充信息"

    vague_terms = ("这个", "那个", "这款", "有问题", "不好用", "怎么办", "帮我看看", "麻烦看看")
    if len(raw_text) <= 12 and any(term in raw_text for term in vague_terms):
        usage_question, _ = detect_usage_question(raw_text)
        if not usage_question:
            return True, "问题过于笼统，需补充产品/场景细节"

    if any(term in raw_text for term in ("这个", "那个", "这款")) and not is_in_scope_query(raw_text):
        return True, "缺少明确产品上下文，需补充信息"

    return False, "信息充分，无需补充"


def classify_query_intent(review_text: str) -> tuple[str, str]:
    """将查询分类为 general-query / additional-query / graphrag-query。"""
    raw_text = (review_text or "").strip()
    if not raw_text:
        return "additional-query", "空输入，需补充问题描述"

    usage_question, usage_reason = detect_usage_question(raw_text)
    if usage_question:
        return "graphrag-query", usage_reason

    matched_tool_keywords = [kw for kw in TOOL_RULE_KEYWORDS if kw in raw_text.lower()]
    if matched_tool_keywords:
        return "graphrag-query", f"规则命中知识检索关键词: {', '.join(matched_tool_keywords[:3])}"

    additional_needed, additional_reason = detect_additional_info_need(raw_text)
    if additional_needed:
        return "additional-query", additional_reason

    out_of_scope_hit = [kw for kw in OUT_OF_SCOPE_HINT_KEYWORDS if kw in raw_text.lower()]
    if out_of_scope_hit:
        return "general-query", f"疑似超出经营范围: {', '.join(out_of_scope_hit[:2])}"

    if is_question_like(raw_text) and not is_in_scope_query(raw_text):
        return "general-query", "问题疑似超出产品/售后范围"

    return "general-query", "默认一般查询"


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


def build_additional_info_reply() -> str:
    return (
        "亲～为了更准确帮您处理，麻烦再补充下具体型号或您遇到的步骤问题，我马上为您继续排查。"
    )


def build_out_of_scope_reply() -> str:
    return (
        "亲～这个问题可能不在当前产品客服处理范围内。若您有剃须刀使用、维护或售后问题，我可以马上帮您。"
    )


def build_human_handoff_reply(reason: str) -> str:
    return (
        "非常抱歉给您带来不便。您的问题涉及需要人工重点跟进的事项，"
        "我已为您转交人工客服优先处理，请您稍候，我们会尽快与您联系。"
        f"\n\n（转人工原因：{reason}）"
    )


def validate_generated_reply(
    original_review: str,
    generated_reply: str,
    query_intent: str,
    needs_additional_info: bool,
    out_of_scope: bool,
) -> tuple[bool, str, str]:
    """对最终回复做轻量校验，必要时给出兜底回复。"""
    reply = (generated_reply or "").strip()
    if not reply:
        fallback = "抱歉，我暂时没有组织好回复，请您再描述一下问题，我马上继续帮您处理。"
        return False, "回复为空", fallback

    if needs_additional_info:
        if "补充" not in reply and "麻烦" not in reply and "具体" not in reply:
            return False, "补充信息场景回复不匹配", build_additional_info_reply()

    if out_of_scope:
        if "范围" not in reply and "产品" not in reply:
            return False, "超范围场景回复不匹配", build_out_of_scope_reply()

    if query_intent == "graphrag-query":
        template_like = (
            "亲，非常感谢您的认可与支持" in reply
            or reply == "感谢您的评价！"
        )
        if template_like:
            fallback = (
                "抱歉，我刚刚没有准确回答到您的使用问题。"
                "您可以再告诉我具体型号或操作环节，我马上为您继续查询说明书。"
            )
            return False, "使用咨询场景回复模板化", fallback
        if len(reply) < 12:
            fallback = (
                "抱歉，我需要更多细节来给您准确步骤。"
                "请告诉我您是在哪一步遇到问题，我马上继续帮您。"
            )
            return False, "使用咨询场景回复过短", fallback

    if query_intent == "additional-query":
        if "麻烦" not in reply and "补充" not in reply:
            return False, "澄清场景未引导补充信息", build_additional_info_reply()

    return True, "回复通过校验", reply


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
        needs_additional_info = query_intent == "additional-query"
        out_of_scope = "超出" in intent_reason

        if query_intent == "graphrag-query":
            review_analysis.require_tool_use = True
            decision_reason = f"{decision_reason}; 意图={query_intent}"
        elif query_intent == "additional-query":
            review_analysis.quality = "default"
            review_analysis.require_tool_use = False
        else:
            review_analysis.require_tool_use = final_tool_use
            if out_of_scope:
                review_analysis.quality = "default"

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
            "analyze_result",
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
        if state.get("needs_additional_info"):
            return {"finally_reply": build_additional_info_reply()}
        if state.get("out_of_scope"):
            return {"finally_reply": build_out_of_scope_reply()}
        return {"finally_reply": "感谢您的评价！"}

    def validate_reply(state: AgentState) -> dict:
        request_id = state.get("request_id", "n/a")
        query_intent = state.get("query_intent", "general-query")
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
            "answer_validation",
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
        "generate_negative_reply": generate_negative_reply,
        "generate_positive_reply": generate_positive_reply,
        "generate_default_reply": generate_default_reply,
        "validate_reply": validate_reply,
    }
