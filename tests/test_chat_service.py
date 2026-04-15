import unittest

from agent.errors import AgentRuntimeError
from agent.services.chat_service import invoke_agent


class FakeApp:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.last_config = None

    def invoke(self, _, config=None):
        self.last_config = config
        if self._error is not None:
            raise self._error
        return self._response


class TestChatService(unittest.TestCase):
    def test_invoke_agent_returns_reply_and_request_id(self):
        app = FakeApp(response={"finally_reply": "测试回复"})
        reply, request_id = invoke_agent(app, "你好", "thread-test")
        self.assertEqual(reply, "测试回复")
        self.assertTrue(request_id)
        self.assertEqual(app.last_config, {"configurable": {"thread_id": "thread-test"}})

    def test_invoke_agent_wraps_runtime_error(self):
        app = FakeApp(error=ValueError("boom"))
        with self.assertRaises(AgentRuntimeError):
            invoke_agent(app, "你好", "thread-test")


if __name__ == "__main__":
    unittest.main()
