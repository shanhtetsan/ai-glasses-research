---
name: asr-whisper
description: DashScope Paraformer streaming ASR replaced with local Whisper (buffer + transcribe on STOP)
metadata:
  type: project
---

Replaced the DashScope Paraformer real-time streaming ASR with local OpenAI Whisper.

**Why:** Decouples from DashScope cloud API; works fully offline.

**How it works:** `ws_audio` buffers incoming PCM bytes into a `bytearray`. On `STOP`, converts int16 PCM → float32, runs `_whisper_model.transcribe()` in a thread executor (non-blocking), then pipes result to `start_ai_with_text_custom`. Hotword detection still works via `has_hotword()` from `asr_core.py`.

**Config:** `WHISPER_MODEL` env var (default `base`), `WHISPER_LANG` env var (default `zh`).
