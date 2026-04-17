from typing import List, NotRequired, TypedDict

from pydantic import BaseModel, Field


class ReviewQuality(BaseModel):
    """评价质量分类器的输出要求。"""

    quality: str = Field(description="是否是正常评论，或者是默认的回复内容，返回值为normal或者default")
    emotion: str = Field(description="情感倾向。只返回正面，负面，中性这三个词")
    key_information: List[str] = Field(
        description="提取评价里用户对产品不会用的关键信息，返回一个列表，列表里是用户对产品的关键信息"
    )
    require_tool_use: bool = Field(description="判读用户的评论是否需要调用工具来获取更多信息，True表示需要，False表示不需要")


class AgentState(TypedDict):
    original_review: str
    review_quality: ReviewQuality
    finally_reply: str
    request_id: NotRequired[str]
    tool_decision_reason: NotRequired[str]
    query_intent: NotRequired[str]
    intent_reason: NotRequired[str]
    needs_additional_info: NotRequired[bool]
    out_of_scope: NotRequired[bool]
    needs_human_review: NotRequired[bool]
    human_review_reason: NotRequired[str]
    answer_valid: NotRequired[bool]
    answer_validation_reason: NotRequired[str]
