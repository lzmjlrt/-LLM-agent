import unittest

from agent.workflow.constants import (
    INTENT_ADDITIONAL_QUERY,
    INTENT_GENERAL_QUERY,
    INTENT_GRAPHRAG_QUERY,
    QUALITY_DEFAULT,
    QUALITY_NORMAL,
)
from agent.workflow.intent_rules import (
    classify_query_intent,
    detect_additional_info_need,
    decide_tool_usage,
    detect_human_review_need,
    normalize_quality_for_meaningful_query,
)


class TestIntentRules(unittest.TestCase):
    def test_intent_detects_graphrag_query(self):
        intent, _reason = classify_query_intent("剃须刀怎么充电？")
        self.assertEqual(intent, INTENT_GRAPHRAG_QUERY)

    def test_intent_detects_additional_query(self):
        intent, _reason = classify_query_intent("这个怎么办？")
        self.assertEqual(intent, INTENT_ADDITIONAL_QUERY)

    def test_intent_detects_general_query(self):
        intent, reason = classify_query_intent("今天北京天气如何？")
        self.assertEqual(intent, INTENT_GENERAL_QUERY)
        self.assertIn("超出", reason)

    def test_normalize_quality_for_question(self):
        normalized, _reason = normalize_quality_for_meaningful_query(
            "这款剃须刀有哪些注意事项？",
            QUALITY_DEFAULT,
        )
        self.assertEqual(normalized, QUALITY_NORMAL)

    def test_decide_tool_usage_with_rule(self):
        need_tool, reason = decide_tool_usage("怎么安装和更换刀头？", False)
        self.assertTrue(need_tool)
        self.assertIn("规则", reason)

    def test_detect_additional_info_need(self):
        need_more, reason = detect_additional_info_need("那个怎么办？")
        self.assertTrue(need_more)
        self.assertIn("补充", reason)

    def test_detect_human_review_need(self):
        need_human, reason = detect_human_review_need("这个产品漏电并且伤人，要求赔偿")
        self.assertTrue(need_human)
        self.assertIn("高风险", reason)

    def test_detect_human_review_need_for_manual_handoff_request(self):
        need_human, reason = detect_human_review_need("我要转人工客服处理")
        self.assertTrue(need_human)
        self.assertIn("人工服务", reason)


if __name__ == "__main__":
    unittest.main()
