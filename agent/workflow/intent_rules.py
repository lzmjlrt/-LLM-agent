import re

from agent.workflow.constants import (
    INTENT_ADDITIONAL_QUERY,
    INTENT_GENERAL_QUERY,
    INTENT_GRAPHRAG_QUERY,
    QUALITY_DEFAULT,
    QUALITY_NORMAL,
)

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

HUMAN_HANDOFF_REQUEST_PATTERNS = (
    r"(转|要|找|接).{0,4}人工",
    r"人工客服",
    r"人工处理",
    r"客服(介入|处理|跟进)",
    r"联系.{0,2}客服",
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
        return INTENT_ADDITIONAL_QUERY, "空输入，需补充问题描述"

    usage_question, usage_reason = detect_usage_question(raw_text)
    if usage_question:
        return INTENT_GRAPHRAG_QUERY, usage_reason

    matched_tool_keywords = [kw for kw in TOOL_RULE_KEYWORDS if kw in raw_text.lower()]
    if matched_tool_keywords:
        return INTENT_GRAPHRAG_QUERY, f"规则命中知识检索关键词: {', '.join(matched_tool_keywords[:3])}"

    additional_needed, additional_reason = detect_additional_info_need(raw_text)
    if additional_needed:
        return INTENT_ADDITIONAL_QUERY, additional_reason

    out_of_scope_hit = [kw for kw in OUT_OF_SCOPE_HINT_KEYWORDS if kw in raw_text.lower()]
    if out_of_scope_hit:
        return INTENT_GENERAL_QUERY, f"疑似超出经营范围: {', '.join(out_of_scope_hit[:2])}"

    if is_question_like(raw_text) and not is_in_scope_query(raw_text):
        return INTENT_GENERAL_QUERY, "问题疑似超出产品/售后范围"

    return INTENT_GENERAL_QUERY, "默认一般查询"


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
    if normalized_quality != QUALITY_DEFAULT:
        return normalized_quality, "保持LLM质量判定"

    if len(raw_text) >= 6 and any(hint in raw_text for hint in MEANINGFUL_QUERY_HINTS):
        return QUALITY_NORMAL, "规则覆盖：检测到有效提问语句"
    return QUALITY_DEFAULT, "保持LLM default判定"


def detect_human_review_need(review_text: str) -> tuple[bool, str]:
    raw_text = (review_text or "").strip()
    lowered = raw_text.lower()
    matched = [kw for kw in HUMAN_REVIEW_HIGH_RISK_KEYWORDS if kw in lowered]
    if matched:
        return True, f"命中高风险关键词: {', '.join(matched[:3])}"
    handoff_hit = [pattern for pattern in HUMAN_HANDOFF_REQUEST_PATTERNS if re.search(pattern, raw_text)]
    if handoff_hit:
        return True, "命中人工服务诉求"
    return False, "未命中高风险关键词"


__all__ = [
    "classify_query_intent",
    "decide_tool_usage",
    "detect_additional_info_need",
    "detect_human_review_need",
    "detect_usage_question",
    "is_in_scope_query",
    "is_question_like",
    "normalize_quality_for_meaningful_query",
]
