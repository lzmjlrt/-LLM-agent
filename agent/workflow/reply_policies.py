from agent.workflow.constants import (
    INTENT_ADDITIONAL_QUERY,
    INTENT_GRAPHRAG_QUERY,
)


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
    _ = original_review
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

    if query_intent == INTENT_GRAPHRAG_QUERY:
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

    if query_intent == INTENT_ADDITIONAL_QUERY:
        if "麻烦" not in reply and "补充" not in reply:
            return False, "澄清场景未引导补充信息", build_additional_info_reply()

    return True, "回复通过校验", reply


__all__ = [
    "build_additional_info_reply",
    "build_human_handoff_reply",
    "build_out_of_scope_reply",
    "validate_generated_reply",
]
