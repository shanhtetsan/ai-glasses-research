# Smart Glasses AI Integration — DEVLOG

**Project:** AI-powered smart glasses for BLV (Blind and Low-Vision) navigation  
**PI:** Prof. Hao Tang (BMCC)  
**Role:** AI Integration   
**Reference Repo:** [AI-FanGe/OpenAIglasses_for_Navigation](https://github.com/AI-FanGe/OpenAIglasses_for_Navigation)  
**Hardware:** XIAO ESP32-S3 (camera + mic) → WiFi → Laptop inference server  

---

## Session 001 — 2026-06-03

### What was done
- Cloned reference repo to local machine
- Set up Python virtual environment (`python3 -m venv venv`)
- Attempted `pip install -r requirements.txt` — failed on `numpy-1.24.3` (too old for Python 3.14, tried to build from source)
- Fixed by installing `numpy>=1.26` first (pre-built Apple Silicon wheel), then re-running requirements install — succeeded
- Created `.env` file with API keys
- Obtained Anthropic API key (console.anthropic.com)
- Obtained OpenAI API key (platform.openai.com)
- Opened project in VS Code, confirmed venv interpreter selected

### Environment
- Machine: MacBook Pro M5, 24GB RAM
- OS: macOS
- Python: 3.14
- GPU backend: Apple MPS (no CUDA — repo was written for NVIDIA, will need `device='mps'` swaps)

### Issues encountered
| Issue | Cause | Fix |
|---|---|---|
| `numpy-1.24.3` build failure | Too old for Python 3.14, no pre-built wheel | `pip install "numpy>=1.26"` first |
| CUDA device calls will fail | Repo targets NVIDIA GPU | Swap `device='cuda'` → `device='mps' if torch.backends.mps.is_available() else 'cpu'` wherever encountered |

---

## Tech Decisions Log

### Decision 001 — English Pipeline over DashScope (Alibaba)
**Date:** 2026-06-03  
**Decision:** Replace the original Alibaba DashScope pipeline with an English-language stack  
**Chosen stack:**
- ASR: **Whisper** (OpenAI, runs locally on M5)
- LLM + TTS: **OpenAI** `gpt-4o-audio-preview` (single API call, streaming)

**Why not DashScope:**
- Project is being tested in the US, not China
- Paraformer ASR is tuned for Chinese speech — poor English accuracy
- DashScope is a paid Chinese cloud service, adds external dependency
- Tang confirmed English pipeline from the start

**Why Whisper:**
- Runs fully locally — no API key, no cost, no latency to external server
- Apple M5 handles it well
- Best-in-class English ASR accuracy

**Why OpenAI over Claude for `omni_client.py`:**
- This file does LLM + TTS in a single streaming call (same pattern as Qwen-Omni-Turbo)
- Claude API is text-only — no audio output support
- OpenAI `gpt-4o-audio-preview` supports combined text + audio streaming, matching the existing architecture exactly
- Avoids breaking the interface that `navigation_master.py` and `app_main.py` depend on

**Why Claude API is still in the project:**
- Available for any pure text/reasoning tasks if needed later
- Tang confirmed he will cover API costs

---

## Planned Next Sessions

- [ ] Read and understand `omni_client.py` fully before rewriting
- [ ] Read `asr_core.py` — understand Paraformer interface to plan Whisper swap
- [ ] Rewrite `omni_client.py` → OpenAI `gpt-4o-audio-preview`
- [ ] Rewrite `asr_core.py` → Whisper
- [ ] Fix all `device='cuda'` → MPS/CPU for Apple Silicon
- [ ] Test pipeline end-to-end without ESP32 (webcam/mic as input)
- [ ] Integrate ESP32-S3 once pipeline is confirmed working
- [ ] Document all voice command changes (Chinese → English)


---

## Session 002 — 2026-06-03

### What was done
- Translated all Chinese comments to English across all core files (originals preserved on the line above each translation)
- Translated Chinese README.md to English → saved as `README_EN.md`
- Rewrote `omni_client.py` — replaced Alibaba DashScope with OpenAI `gpt-4o-audio-preview`
- Translated Chinese voice guidance strings in `navigation_master.py` to English
- Replaced all five core files in the project: `app_main.py`, `asr_core.py`, `omni_client.py`, `navigation_master.py`, `audio_player.py`
- Set up VS Code workspace with tab groups and Mac Spaces for project organization
- Installed Claude Code VS Code extension for in-editor AI assistance

### Issues identified (not yet fixed)
| Issue | File | Notes |
|---|---|---|
| `audio_player.py` uses prerecorded Chinese `.wav` files | `audio_player.py` | English guidance strings from `navigation_master.py` will not match Chinese keys in `AUDIO_MAP` — voice output will silently fail |
| `asr_core.py` still wired to DashScope Paraformer | `asr_core.py`, `app_main.py` | ASR not yet swapped to Whisper — this is the next major task |
| `app_main.py` still imports `dashscope` | `app_main.py` | Will break on run until DashScope import is removed and Whisper is wired in |
| CUDA device calls throughout codebase | multiple files | Need to swap `device='cuda'` → `device='mps'` for Apple Silicon |

---

## Tech Decisions Log (continued)

### Decision 002 — OpenAI TTS instead of prerecorded Chinese `.wav` files
**Date:** 2026-06-03  
**Decision:** Replace `audio_player.py`'s prerecorded file lookup system with dynamic OpenAI TTS  
**Chosen approach:** Call OpenAI TTS API (`tts-1` model) at runtime to generate English speech for any guidance text

**Why not keep prerecorded files:**
- All existing `.wav` files are Chinese — English guidance strings will never match
- Re-recording English audio files manually is time-consuming and not scalable
- Dynamic TTS is more flexible — any new guidance text works automatically

**Why OpenAI TTS:**
- Same API key already in use for `omni_client.py`
- `tts-1` is fast and cheap — suitable for real-time navigation guidance
- Supports streaming so first audio chunk arrives quickly
- Voice options (alloy, nova, shimmer etc.) can be tuned for clarity

**Tradeoff:**
- Adds ~300-500ms latency per TTS call vs instant prerecorded playback
- Requires internet connection (same requirement as `omni_client.py`)
- Acceptable for research/demo context

---

## Planned Next Sessions

- [ ] Swap `asr_core.py` → Whisper (replace Paraformer ASR)
- [ ] Remove DashScope import from `app_main.py`
- [ ] Rewrite `audio_player.py` `play_voice_text()` → OpenAI TTS dynamic generation
- [ ] Fix all `device='cuda'` → `device='mps'` for Apple Silicon
- [ ] Test full pipeline end-to-end without ESP32 (laptop webcam + mic)
- [ ] Integrate ESP32-S3 once pipeline confirmed working

### Session 003 — Qwen2.5-Omni-3B Audio Test

**Result:** Text output ✅ Audio output ✅ (saved to test_response.wav)

**Voice quality:** Poor — audio is muffled/unclear on both Chelsie (default) and Ethan voices

**Likely cause:** 3B model is too small for high quality TTS — the talker component is undertrained at this size

**Next test:** Qwen2.5-Omni-7B — same architecture, larger model, expected better voice quality

**Fallback plan:** If 7B audio still poor → use Qwen2.5-Omni-3B/7B for LLM (text) + OpenAI TTS for voice separately