import unittest

from agent.errors import ConfigValidationError
from agent.factories.config_validation import validate_runtime_config


class FakeUploadFile:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getbuffer(self):
        return self._data


class TestConfigValidation(unittest.TestCase):
    def test_valid_config_passes(self):
        uploaded_file = FakeUploadFile("manual.pdf", b"pdf-bytes")
        validate_runtime_config(
            llm_provider="OpenAI",
            llm_api_key="sk-test",
            uploaded_file=uploaded_file,
            embedding_provider="OpenAI",
            embedding_api_key="sk-embed",
        )

    def test_invalid_llm_provider_raises(self):
        uploaded_file = FakeUploadFile("manual.pdf", b"pdf-bytes")
        with self.assertRaises(ConfigValidationError):
            validate_runtime_config(
                llm_provider="Unknown",
                llm_api_key="sk-test",
                uploaded_file=uploaded_file,
                embedding_provider="OpenAI",
                embedding_api_key="sk-embed",
            )

    def test_missing_pdf_raises(self):
        with self.assertRaises(ConfigValidationError):
            validate_runtime_config(
                llm_provider="DeepSeek",
                llm_api_key="sk-test",
                uploaded_file=None,
                embedding_provider="DashScope (Alibaba)",
                embedding_api_key="sk-embed",
            )

    def test_empty_pdf_raises(self):
        uploaded_file = FakeUploadFile("manual.pdf", b"")
        with self.assertRaises(ConfigValidationError):
            validate_runtime_config(
                llm_provider="DeepSeek",
                llm_api_key="sk-test",
                uploaded_file=uploaded_file,
                embedding_provider="DashScope (Alibaba)",
                embedding_api_key="sk-embed",
            )


if __name__ == "__main__":
    unittest.main()
