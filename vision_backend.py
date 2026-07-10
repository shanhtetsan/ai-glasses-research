# vision_backend.py
# -*- coding: utf-8 -*-
#
# Single switch point for the vision/chat model. Reads MODEL_BACKEND (loaded
# from .env by app_main before this import) and re-exports the chosen backend's
# stream_chat() and OmniStreamPiece. Both backends expose an identical
# interface, so downstream code imports only from here and never changes:
#
#   MODEL_BACKEND=gemini  → gemini_client  (Google Gemini, cloud, text-only)
#   MODEL_BACKEND=qwen    → omni_client    (local Qwen2.5-Omni-3B on MPS, text-only)
#
import os

_BACKEND = os.getenv("MODEL_BACKEND", "gemini").strip().lower()

if _BACKEND == "qwen":
    # Heavy: loads the 3B model onto MPS at import time.
    from omni_client import stream_chat, OmniStreamPiece
elif _BACKEND == "gemini":
    from gemini_client import stream_chat, OmniStreamPiece
else:
    raise ValueError(
        f"Unknown MODEL_BACKEND={_BACKEND!r} — expected 'gemini' or 'qwen'"
    )

print(f"[VISION] Active backend: {_BACKEND!r}")

__all__ = ["stream_chat", "OmniStreamPiece", "_BACKEND"]
