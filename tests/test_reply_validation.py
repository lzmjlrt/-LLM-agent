import unittest

from agent.workflow.constants import (
    INTENT_ADDITIONAL_QUERY,
    INTENT_GRAPHRAG_QUERY,
)
from agent.workflow.reply_policies import (
    build_additional_info_reply,
    validate_generated_reply,
)


class TestReplyValidation(unittest.TestCase):
    def test_reject_template_reply_for_graphrag(self):
        valid, reason, fallback = validate_generated_reply(
            original_review="剃须刀怎么充电？",
            generated_reply="亲，非常感谢您的认可与支持！您的满意是我们不断前行的动力，期待您的再次光临！",
            query_intent=INTENT_GRAPHRAG_QUERY,
            needs_additional_info=False,
            out_of_scope=False,
        )
        self.assertFalse(valid)
        self.assertIn("模板化", reason)
        self.assertIn("继续查询", fallback)

    def test_additional_query_requires_clarification(self):
        valid, reason, final_reply = validate_generated_reply(
            original_review="这个怎么办？",
            generated_reply="感谢您的评价！",
            query_intent=INTENT_ADDITIONAL_QUERY,
            needs_additional_info=True,
            out_of_scope=False,
        )
        self.assertFalse(valid)
        self.assertIn("补充信息", reason)
        self.assertEqual(final_reply, build_additional_info_reply())

    def test_pass_normal_graphrag_reply(self):
        valid, reason, final_reply = validate_generated_reply(
            original_review="剃须刀怎么充电？",
            generated_reply="您可以先检查电量指示灯，再按说明书步骤连接充电线进行充电。",
            query_intent=INTENT_GRAPHRAG_QUERY,
            needs_additional_info=False,
            out_of_scope=False,
        )
        self.assertTrue(valid)
        self.assertIn("通过", reason)
        self.assertIn("充电", final_reply)


if __name__ == "__main__":
    unittest.main()
