import unittest

from agent.workflow.nodes import (
    decide_tool_usage,
    detect_human_review_need,
    detect_usage_question,
    normalize_quality_for_meaningful_query,
)


class TestToolDecision(unittest.TestCase):
    def test_rule_hit_forces_tool_use(self):
        use_tool, reason = decide_tool_usage("这个产品我不会用，怎么安装？", False)
        self.assertTrue(use_tool)
        self.assertIn("规则命中关键词", reason)

    def test_llm_fallback_when_no_rule_hit(self):
        use_tool, reason = decide_tool_usage("物流太慢了", True)
        self.assertTrue(use_tool)
        self.assertEqual(reason, "LLM判定需要调用工具")

    def test_no_rule_no_llm_request(self):
        use_tool, reason = decide_tool_usage("包装有点破损", False)
        self.assertFalse(use_tool)
        self.assertEqual(reason, "LLM判定无需调用工具")

    def test_default_quality_overridden_for_question_like_input(self):
        normalized_quality, reason = normalize_quality_for_meaningful_query(
            "这个剃须刀有什么注意的重要事项吗？",
            "default",
        )
        self.assertEqual(normalized_quality, "normal")
        self.assertIn("规则覆盖", reason)

    def test_human_review_detects_high_risk_keywords(self):
        need_human, reason = detect_human_review_need("这个产品漏电，要求退款")
        self.assertTrue(need_human)
        self.assertIn("命中高风险关键词", reason)

    def test_charge_question_is_detected_as_usage_query(self):
        is_usage, reason = detect_usage_question("你这个剃须刀怎么充电的？")
        self.assertTrue(is_usage)
        self.assertIn("使用咨询", reason)

    def test_charge_question_forces_tool_use(self):
        use_tool, reason = decide_tool_usage("你这个剃须刀怎么充电的？", False)
        self.assertTrue(use_tool)
        self.assertTrue("规则" in reason or "使用咨询" in reason)


if __name__ == "__main__":
    unittest.main()
