# asr_config.py
# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class WhisperConfig:
    model: str
    language: Optional[str]


def _clean_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def load_whisper_config() -> WhisperConfig:
    """Read Whisper ASR configuration from the environment."""
    language = _clean_env("WHISPER_LANG", "en")
    if language and language.lower() in {"auto", "detect", "none"}:
        language = None

    return WhisperConfig(
        model=_clean_env("WHISPER_MODEL", "base") or "base",
        language=language,
    )
