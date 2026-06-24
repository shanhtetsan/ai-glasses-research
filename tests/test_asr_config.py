import os
import unittest
from unittest.mock import patch

from asr_config import load_whisper_config


class WhisperConfigTests(unittest.TestCase):
    def test_defaults_to_base_english(self):
        with patch.dict(os.environ, {}, clear=True):
            config = load_whisper_config()

        self.assertEqual(config.model, "base")
        self.assertEqual(config.language, "en")

    def test_reads_model_and_language_from_environment(self):
        with patch.dict(os.environ, {"WHISPER_MODEL": "small", "WHISPER_LANG": "zh"}, clear=True):
            config = load_whisper_config()

        self.assertEqual(config.model, "small")
        self.assertEqual(config.language, "zh")

    def test_allows_auto_language_detection(self):
        with patch.dict(os.environ, {"WHISPER_LANG": "auto"}, clear=True):
            config = load_whisper_config()

        self.assertIsNone(config.language)


if __name__ == "__main__":
    unittest.main()
