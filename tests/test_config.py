import os
import unittest
from unittest.mock import patch

from config import get_api_key_for_provider, get_provider_config


class ConfigTests(unittest.TestCase):
    def test_get_provider_config_falls_back_to_openai(self) -> None:
        config = get_provider_config("unknown-provider")
        self.assertEqual(config["api_base"], "https://api.openai.com/v1")
        self.assertEqual(config["model"], "gpt-4o")

    def test_get_embedding_config_returns_expected_model(self) -> None:
        config = get_provider_config("modelscope", for_embedding=True)
        self.assertEqual(config["model"], "Qwen/Qwen3-Embedding-8B")

    def test_get_embedding_config_supports_none_provider(self) -> None:
        config = get_provider_config("none", for_embedding=True)
        self.assertEqual(config["model"], "")
        self.assertEqual(config["api_key_env"], "")

    def test_get_api_key_for_provider_reads_from_environment(self) -> None:
        with patch.dict(os.environ, {"MODELSCOPE_API_KEY": "test-key"}, clear=False):
            api_key = get_api_key_for_provider("modelscope")
        self.assertEqual(api_key, "test-key")


if __name__ == "__main__":
    unittest.main()
