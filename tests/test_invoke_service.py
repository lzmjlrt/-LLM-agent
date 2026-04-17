import unittest
from unittest.mock import patch

from agent.errors import AgentRuntimeError
from agent.services.invoke_service import invoke_agent


class FakeApp:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.invoke_calls = 0
        self.last_payload = None
        self.last_config = None

    def invoke(self, payload, config=None):
        self.invoke_calls += 1
        self.last_payload = payload
        self.last_config = config
        if self._error is not None:
            raise self._error
        return self._response


class TestInvokeService(unittest.TestCase):
    @patch("agent.services.invoke_service.save_thread_memory")
    @patch("agent.services.invoke_service.append_conversation_turn", side_effect=lambda m, *_: m)
    @patch("agent.services.invoke_service.find_cached_reply", return_value=("缓存命中回复", 0.98))
    @patch("agent.services.invoke_service.load_thread_memory", return_value={"summary": "", "turns": [], "qa_cache": []})
    @patch("agent.services.invoke_service.config.SEMANTIC_CACHE_ENABLED", True)
    def test_cache_hit_skips_graph_invoke(self, _load, _find, _append, save_memory):
        app = FakeApp(response={"finally_reply": "不应被调用"})
        reply, request_id = invoke_agent(app, "怎么充电", "thread-test")
        self.assertEqual(reply, "缓存命中回复")
        self.assertTrue(request_id)
        self.assertEqual(app.invoke_calls, 0)
        save_memory.assert_called_once()

    @patch("agent.services.invoke_service.save_thread_memory")
    @patch("agent.services.invoke_service.append_conversation_turn", side_effect=lambda m, *_: m)
    @patch("agent.services.invoke_service.build_conversation_context", return_value="历史摘要：用户曾咨询充电问题")
    @patch("agent.services.invoke_service.find_cached_reply", return_value=(None, 0.42))
    @patch("agent.services.invoke_service.load_thread_memory", return_value={"summary": "", "turns": [], "qa_cache": []})
    @patch("agent.services.invoke_service.config.SEMANTIC_CACHE_ENABLED", True)
    def test_cache_miss_invokes_graph(self, _load, _find, _context, _append, save_memory):
        app = FakeApp(response={"finally_reply": "这是新回答"})
        reply, request_id = invoke_agent(app, "如何更换刀头", "thread-test")
        self.assertEqual(reply, "这是新回答")
        self.assertTrue(request_id)
        self.assertEqual(app.invoke_calls, 1)
        self.assertEqual(app.last_config, {"configurable": {"thread_id": "thread-test"}})
        self.assertIn("conversation_context", app.last_payload)
        save_memory.assert_called_once()

    @patch("agent.services.invoke_service.save_thread_memory")
    @patch("agent.services.invoke_service.append_conversation_turn", side_effect=lambda m, *_: m)
    @patch("agent.services.invoke_service.build_conversation_context", return_value="")
    @patch("agent.services.invoke_service.find_cached_reply", return_value=(None, 0.0))
    @patch("agent.services.invoke_service.load_thread_memory", return_value={"summary": "", "turns": [], "qa_cache": []})
    @patch("agent.services.invoke_service.config.SEMANTIC_CACHE_ENABLED", True)
    def test_invoke_wraps_business_error(self, _load, _find, _context, _append, _save):
        app = FakeApp(error=ValueError("boom"))
        with self.assertRaises(AgentRuntimeError):
            invoke_agent(app, "你好", "thread-test")


if __name__ == "__main__":
    unittest.main()
