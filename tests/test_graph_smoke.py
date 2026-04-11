import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.workflow.graph import create_graph
from agent.workflow.schema import ReviewQuality


class FakeStructuredLLM:
    def __init__(self, review_quality):
        self.review_quality = review_quality

    def invoke(self, _):
        return self.review_quality


class FakeLLM:
    def __init__(self, review_quality):
        self.review_quality = review_quality

    def with_structured_output(self, _, **__):
        return FakeStructuredLLM(self.review_quality)


class FakeAgentExecutor:
    def invoke(self, _):
        return {"messages": [SimpleNamespace(content="已收到反馈，我们会尽快处理。")]}


class TestGraphSmoke(unittest.TestCase):
    @patch("agent.workflow.graph.create_react_agent", return_value=FakeAgentExecutor())
    def test_graph_invoke_returns_final_reply(self, _):
        fake_llm = FakeLLM(
            ReviewQuality(
                quality="normal",
                emotion="负面",
                key_information=[],
                require_tool_use=False,
            )
        )
        app = create_graph(fake_llm, tools=[])
        result = app.invoke({"original_review": "不会用这个产品"})
        self.assertIn("finally_reply", result)
        self.assertEqual(result["finally_reply"], "已收到反馈，我们会尽快处理。")


if __name__ == "__main__":
    unittest.main()
