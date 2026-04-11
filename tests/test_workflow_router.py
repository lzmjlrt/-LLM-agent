import unittest

from agent.workflow.router import route_after_analysis
from agent.workflow.schema import ReviewQuality


class TestRouteAfterAnalysis(unittest.TestCase):
    def test_tool_route_has_highest_priority(self):
        state = {
            "review_quality": ReviewQuality(
                quality="default",
                emotion="中性",
                key_information=[],
                require_tool_use=True,
            )
        }
        self.assertEqual(route_after_analysis(state), "generate_negative_reply")

    def test_route_to_default_for_default_quality(self):
        state = {
            "review_quality": ReviewQuality(
                quality="default",
                emotion="负面",
                key_information=[],
                require_tool_use=False,
            )
        }
        self.assertEqual(route_after_analysis(state), "generate_default_reply")

    def test_route_to_negative_for_negative_sentiment(self):
        state = {
            "review_quality": ReviewQuality(
                quality="normal",
                emotion="负面",
                key_information=[],
                require_tool_use=False,
            )
        }
        self.assertEqual(route_after_analysis(state), "generate_negative_reply")

    def test_route_to_positive_for_non_negative(self):
        state = {
            "review_quality": ReviewQuality(
                quality="normal",
                emotion="正面",
                key_information=[],
                require_tool_use=False,
            )
        }
        self.assertEqual(route_after_analysis(state), "generate_positive_reply")


if __name__ == "__main__":
    unittest.main()
