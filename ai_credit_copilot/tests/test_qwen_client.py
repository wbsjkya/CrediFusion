import unittest

from ai_credit_copilot.config import QwenSettings
from ai_credit_copilot.qwen_client import QwenClient, QwenConfigurationError


class QwenClientTest(unittest.TestCase):
    def test_endpoint_is_normalised(self):
        settings = QwenSettings("key", "https://example.test/v1/", "qwen3.5", 30, 500)
        self.assertEqual(settings.chat_completions_url, "https://example.test/v1/chat/completions")

    def test_unconfigured_client_never_makes_a_request(self):
        client = QwenClient(QwenSettings("", "https://example.test/v1", "qwen3.5", 30, 500))
        with self.assertRaises(QwenConfigurationError):
            client.complete([{"role": "user", "content": "hello"}])
