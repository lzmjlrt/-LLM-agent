import unittest

from agent.workflow.nodes import (
    classify_query_intent,
    detect_additional_info_need,
    decide_tool_usage,
    detect_human_review_need,
    detect_usage_question,
    normalize_quality_for_meaningful_query,
    validate_generated_reply,
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

    def test_classify_query_intent_for_usage_question(self):
        intent, reason = classify_query_intent("这款剃须刀怎么充电？")
        self.assertEqual(intent, "graphrag-query")
        self.assertTrue("咨询" in reason or "关键词" in reason)

    def test_classify_query_intent_for_vague_question(self):
        intent, _reason = classify_query_intent("这个有问题怎么办？")
        self.assertEqual(intent, "additional-query")

    def test_classify_query_intent_for_out_of_scope_question(self):
        intent, reason = classify_query_intent("今天上海天气怎么样？")
        self.assertEqual(intent, "general-query")
        self.assertIn("超出", reason)

    def test_detect_additional_info_need(self):
        need_more, _reason = detect_additional_info_need("这个怎么办？")
        self.assertTrue(need_more)

    def test_validate_generated_reply_rejects_template_for_rag_query(self):
        valid, reason, fallback = validate_generated_reply(
            original_review="你这个剃须刀怎么充电？",
            generated_reply="亲，非常感谢您的认可与支持！您的满意是我们不断前行的动力，期待您的再次光临！",
            query_intent="graphrag-query",
            needs_additional_info=False,
            out_of_scope=False,
        )
        self.assertFalse(valid)
        self.assertIn("模板化", reason)
        self.assertIn("继续查询", fallback)

    def test_validate_generated_reply_passes_normal_reply(self):
        valid, reason, final_reply = validate_generated_reply(
            original_review="你这个剃须刀怎么充电？",
            generated_reply="您可以先确认电量指示灯状态，再按说明书步骤连接充电线进行充电。",
            query_intent="graphrag-query",
            needs_additional_info=False,
            out_of_scope=False,
        )
        self.assertTrue(valid)
        self.assertIn("通过", reason)
        self.assertIn("充电", final_reply)


if __name__ == "__main__":
    unittest.main()
