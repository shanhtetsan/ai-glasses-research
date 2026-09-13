# app_main.py
# -*- coding: utf-8 -*-
import os, sys, time, json, asyncio, base64, csv, tempfile, wave, hmac, logging, uuid
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Tuple, List, Callable, Set, Deque
from collections import deque
from dataclasses import dataclass
import re


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


STABILITY_MODE = env_bool("STABILITY_MODE", True)

# Add after other imports:

# extract_english_label (item-search voice command) needs the `openai`
# package for its DashScope-compatible client. Not installed in the
# cloud/gemini_live-only deploy (see requirements-cloud.txt) — degrade to
# item-search-disabled rather than crash the whole server on import (see
# the extract_english_label is None guard at its one call site).
if not STABILITY_MODE:
    try:
        from qwen_extractor import extract_english_label
    except ImportError as e:
        print(f"[ITEM_SEARCH] openai not installed, item-search label extraction disabled: {e}")
        extract_english_label = None
else:
    extract_english_label = None

# Blind-path/cross-street navigation + obstacle detection need torch and
# ultralytics. navigation_master.py, workflow_blindpath.py,
# workflow_crossstreet.py, and obstacle_detector_client.py all import torch
# at their own module level, so importing *any* of them pulls torch in
# transitively even without the `import torch` below — all four have to be
# inside this same guard, not just the last three. Not installed in the
# cloud/gemini_live-only deploy (see requirements-cloud.txt) — degrade to
# navigation-disabled rather than crash the whole server on import, same
# pattern as yolomedia below. load_navigation_models() already tolerates
# these being None.
try:
    if STABILITY_MODE:
        raise ImportError("disabled by STABILITY_MODE")
    from navigation_master import NavigationMaster, OrchestratorResult
    # New: import blind-path navigator
    from workflow_blindpath import BlindPathNavigator
    # New: import cross-street navigator
    from workflow_crossstreet import CrossStreetNavigator
    from obstacle_detector_client import ObstacleDetectorClient
    import torch
    from ultralytics import YOLO
except ImportError as e:
    print(f"[NAVIGATION] torch/ultralytics not installed, navigation disabled: {e}")
    NavigationMaster = None
    OrchestratorResult = None
    BlindPathNavigator = None
    CrossStreetNavigator = None
    ObstacleDetectorClient = None
    torch = None
    YOLO = None

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
import uvicorn
import cv2
import numpy as np
from perception_orientation import (
    RgbCanonicalizerTelemetry,
    canonicalize_thermal_payload,
    log_rgb_canonicalizer_health,
    map_thermal_to_rgb_normalized,
    queue_latest_raw_rgb,
    run_latest_rgb_canonicalizer,
    summarize_thermal_grid,
)
from perception_fusion import (
    HAND_HANDEDNESS_AUTHORITATIVE_THRESHOLD,
    HandTargetGuidanceTracker,
    build_hand_fusion_fact,
    compact_guidance_for_log,
    compact_hand_facts_for_log,
    extract_requested_target,
    guidance_anchor_for_utterance,
    guidance_anchor_reason,
    is_hand_perception_request,
    is_target_directed_request,
    should_wait_for_explicit_target,
)
try:
    import audioop
except ModuleNotFoundError:
    class _AudioopCompat:
        @staticmethod
        def ratecv(fragment, width, channels, inrate, outrate, state):
            if width != 2 or channels != 1 or inrate % outrate:
                raise RuntimeError("audioop fallback supports mono PCM16 integer downsampling only")
            samples = np.frombuffer(fragment, dtype="<i2")
            return samples[::inrate // outrate].astype("<i2", copy=False).tobytes(), None
    audioop = _AudioopCompat()
if not STABILITY_MODE:
    import general_detector  # prompt-free YOLOE general object detection (lazy-loaded)
else:
    general_detector = None
import bridge_io
import threading
# import yolomedia  # must be in the same directory as app_main.py, filename is yolomedia.py

try:
    if STABILITY_MODE:
        raise RuntimeError("disabled by STABILITY_MODE")
    import yolomedia
except Exception:
    yolomedia = None
DEBUG     = False  # set True to enable verbose navigation/recorder/YOLO logs
DEBUG_VAD = False   # set False once VAD_SILENCE_RMS is tuned for your mic

# ---- Windows event loop policy ----
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ---- .env ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import os
_gk = os.getenv("GEMINI_API_KEY") or ""
print("Gemini Key =", "(configured)" if _gk else "(not set)")

# ---- Active AI backend selection (for comparing regular Gemini / Gemini Live / Qwen) ----
# Set via env var, e.g.:  AI_BACKEND=gemini_regular python app_main.py
# "gemini_live"    — Gemini Live handles ASR + conversation + spoken reply end-to-end.
# "gemini_regular" — Gemini Live is used only for real-time ASR (TEXT response
#                     mode, no spoken reply from it); the transcribed text is
#                     sent to gemini_client.stream_chat (gemini-2.5-flash) and
#                     the reply is spoken via local TTS.
# "qwen"           — same as gemini_regular, but the transcribed text goes to
#                     omni_client.stream_chat (local Qwen2.5-Omni-3B) instead.
_VALID_BACKENDS = ("gemini_live", "gemini_regular", "qwen")
AI_BACKEND = os.getenv("AI_BACKEND", "gemini_live").strip().lower()
if AI_BACKEND not in _VALID_BACKENDS:
    print(f"[BACKEND] WARNING: unknown AI_BACKEND={AI_BACKEND!r}, falling back to 'gemini_live'")
    AI_BACKEND = "gemini_live"

print("=" * 60)
print(f"  ACTIVE AI BACKEND: {AI_BACKEND}")
print("=" * 60)



# ---- [DASHSCOPE FALLBACK] Remote ASR — commented out, kept for reference ----
# from dashscope import audio as dash_audio
# API_KEY     = os.getenv("DASHSCOPE_API_KEY", "YOUR_DASHSCOPE_API_KEY")
# if not API_KEY:
#     raise RuntimeError("DASHSCOPE_API_KEY is not set")
# MODEL        = "paraformer-realtime-v2"
# AUDIO_FMT    = "pcm"
# CHUNK_MS     = 20
# BYTES_CHUNK  = SAMPLE_RATE * CHUNK_MS // 1000 * 2
# SILENCE_20MS = bytes(BYTES_CHUNK)
# ---------------------------------------------------------------------------

# ---- Local Whisper ASR (replaces DashScope) ----
# ESP32 sends PCM16 at 16 kHz; Whisper expects float32 at 16 kHz — same rate.
SAMPLE_RATE = 16000
from asr_config import load_whisper_config
_WHISPER_CFG = load_whisper_config()
# openai-whisper imports torch internally, so this needs the same guard as
# navigation above. _run_whisper_and_dispatch (gemini_regular/qwen's local
# ASR path — gemini_live doesn't use this at all) already early-returns on
# `_whisper_model is None`, so no other call site needs touching.
try:
    if STABILITY_MODE and AI_BACKEND == "gemini_live":
        raise ImportError("local Whisper disabled by STABILITY_MODE")
    import whisper as _whisper_lib
    print(f"[...] Loading Whisper model {_WHISPER_CFG.model!r} (lang={_WHISPER_CFG.language or 'auto'})...")
    _whisper_model = _whisper_lib.load_model(_WHISPER_CFG.model)
    print("[OK] Whisper model ready")
except ImportError as e:
    print(f"[WHISPER] torch/whisper not installed, local ASR disabled: {e}")
    _whisper_model = None

# ---- Server-side Voice Activity Detection (VAD) tuning ----
# The ESP32 streams PCM16 continuously; these constants control when the
# server decides the user has finished speaking and fires Whisper.
#
# VAD_SILENCE_RMS   — RMS amplitude (int16 scale, 0–32767) of a 20 ms chunk
#                     that is treated as "silent".
#                     Background/noise floor is typically 50–150.
#                     Normal speech is 300–3 000+.
#                     Raise if ambient noise falsely triggers speech detection;
#                     lower if soft voices are missed.
VAD_SILENCE_RMS    = 300
JPEG_QUALITY       = 80

# All frames downstream of ingest are already in canonical portrait space.
# Kept in the settings response for compatibility; active display/inference
# rotation must remain zero to prevent a second physical rotation.
CAMERA_ROTATION_DEG = 0

# VAD_SILENCE_MS    — milliseconds of continuous silence (after speech has
#                     been detected) that trigger auto-transcription.
#                     700 ms = 35 chunks × 20 ms. Raise for slower speakers;
#                     lower for snappier response.
VAD_SILENCE_MS     = 700

# VAD_MIN_SPEECH_MS — minimum speech duration (ms) before silence can fire
#                     Whisper.  Prevents spurious triggers from a brief click
#                     or microphone pop.  300 ms = 15 chunks × 20 ms.
VAD_MIN_SPEECH_MS  = 300

# Derived chunk counts (ESP32 sends exactly 20 ms chunks at 16 kHz / PCM16).
_VAD_CHUNK_MS         = 20
VAD_SILENCE_CHUNKS    = VAD_SILENCE_MS    // _VAD_CHUNK_MS   # 35
VAD_MIN_SPEECH_CHUNKS = VAD_MIN_SPEECH_MS // _VAD_CHUNK_MS   # 15

# Hard cap on pcm_buffer size. Without this, if vad_speech_detected never
# flips true (spoke too softly, held further from the mic, etc.), the buffer
# just keeps growing unflushed — and whatever you say on your NEXT attempt
# gets appended onto that leftover audio instead of starting fresh, so
# Whisper ends up transcribing both attempts merged into one blob. This caps
# how long that can go on: past this many seconds with no detected speech,
# the buffer is discarded and VAD state resets clean.
VAD_MAX_BUFFER_SECONDS = 12
VAD_MAX_BUFFER_BYTES = SAMPLE_RATE * 2 * VAD_MAX_BUFFER_SECONDS  # 16-bit mono PCM

# ---- Import our modules ----
import audio_stream                # module import (not just names) so we can
                                    # assign audio_stream.current_ai_task and
                                    # have is_playing_now() see the update —
                                    # `from audio_stream import current_ai_task`
                                    # would bind a stale local copy instead.
from audio_stream import (
    register_stream_route,         # mount /stream.wav
    broadcast_pcm16_realtime,      # distribute 16k PCM to all connected clients in real time
    hard_reset_audio,              # master switch for audio + AI playback
    BYTES_PER_20MS_16K,
    is_playing_now,
    cancel_current_ai,
)
from vision_backend import stream_chat, OmniStreamPiece
from asr_core import (
    ASRCallback,
    set_current_recognition,
    stop_current_recognition,
    INTERRUPT_KEYWORDS,
    _normalize_cn,
)
from audio_player import initialize_audio_system, play_voice_text

from gemini_live_client import GeminiLiveClient
gemini_live = GeminiLiveClient()

def _has_hotword(text: str) -> bool:
    """Return True if text contains any interrupt keyword (mirrors ASRCallback logic)."""
    t = _normalize_cn(text)
    if not t:
        return False
    for w in INTERRUPT_KEYWORDS:
        if w and _normalize_cn(w) in t:
            return True
    return False

# ---- Gemini Live callback wiring ----
# Rolling per-turn buffers for the transcripts Gemini streams back to us.
_input_text_buf: List[str] = []   # what the user said this turn (from input_audio_transcription)
_output_text_buf: List[str] = []  # what Gemini said this turn (from output_audio_transcription)
_esp32_tts_started: bool = False  # whether we've sent TTS:START to the ESP32 for the current turn
_TTS_CHUNK = 2040                 # fits TTSChunk.data[2048] on the firmware side


@dataclass(frozen=True)
class _TTSQueueItem:
    turn_id: int
    chunk: Optional[bytes] = None
    terminal_status: Optional[str] = None


_tts_send_queue: asyncio.Queue[Any] = asyncio.Queue()
_tts_sender_task: Optional[asyncio.Task] = None
_vision_submitted_for_turn: bool = False
_thermal_submitted_for_turn: bool = False
_perception_submitted_for_turn: bool = False
_vision_frame_sequence_for_turn: Optional[int] = None
_vision_frame_backend_ns_for_turn: Optional[int] = None  # same frame's capture time, same clock as detectors' backend_received_monotonic_ns
_vision_intent_for_turn: bool = False   # latched True if is_explicit_vision_request() ever fired this turn
_thermal_intent_for_turn: bool = False  # latched True if is_thermal_request() ever fired this turn
_TTS_TARGET_LEAD_SEC = 2.0      # audio allowed to sit buffered on the device

# Persistent audioop.ratecv state, kept across _on_audio calls within a turn
# so consecutive chunks resample smoothly instead of clicking at boundaries.
# Reset whenever a new response turn starts (on turn_complete/interrupted).
_ratecv_state_8k = None


def _log_gemini_timing(event: str, turn_id: Optional[int], **fields) -> None:
    parts = [
        f"event={event}",
        f"mono_ns={time.monotonic_ns()}",
        f"generation={gemini_live.session_generation}",
        f"turn_id={turn_id if turn_id is not None else 0}",
    ]
    parts.extend(f"{key}={value}" for key, value in fields.items())
    print("[GEMINI-TIMING] " + " ".join(parts), flush=True)


def _age_ms_since(started_ns: Optional[int]) -> Optional[float]:
    if started_ns is None:
        return None
    return round(max(0.0, (time.monotonic_ns() - started_ns) / 1_000_000), 3)


def _detector_perception_telemetry(raw: dict) -> dict:
    """Shape one detector's raw latest_* response into the persisted turn record."""
    return {
        "source_frame_id": raw.get("source_frame_id"),
        "backend_frame_received_ns": raw.get("backend_received_monotonic_ns"),
        "inference_completed_ns": raw.get("inference_completed_monotonic_ns"),
        "request_ms": raw.get("request_ms"),
        "inference_ms": raw.get("inference_ms"),
        "completion_age_ms": raw.get("age_ms"),
        "backend_frame_age_ms": _age_ms_since(raw.get("backend_received_monotonic_ns")),
        "service_healthy": raw.get("service_healthy"),
    }


def _frame_alignment(
    gemini_frame_id: Optional[int],
    gemini_backend_ns: Optional[int],
    yolo_raw: dict,
) -> Tuple[Optional[int], Optional[float]]:
    """How far the RGB frame sent to Gemini has drifted from the frame YOLO's
    detections are based on: frame-sequence count and wall-clock time, both
    gemini minus yolo. Null whenever either side's frame id is missing."""
    yolo_frame_id = yolo_raw.get("source_frame_id")
    if gemini_frame_id is None or yolo_frame_id is None:
        return None, None
    frame_id_delta = gemini_frame_id - yolo_frame_id
    yolo_backend_ns = yolo_raw.get("backend_received_monotonic_ns")
    time_delta_ms = None
    if gemini_backend_ns is not None and yolo_backend_ns is not None:
        time_delta_ms = round((gemini_backend_ns - yolo_backend_ns) / 1_000_000.0, 3)
    return frame_id_delta, time_delta_ms


def _clear_gemini_turn_state(reason: str) -> None:
    """Drop only ephemeral per-turn grounding/transcription state."""
    global _vision_submitted_for_turn, _thermal_submitted_for_turn, _perception_submitted_for_turn
    global _vision_frame_sequence_for_turn, _vision_frame_backend_ns_for_turn
    global _vision_intent_for_turn, _thermal_intent_for_turn
    had_state = bool(
        _input_text_buf
        or _output_text_buf
        or _vision_submitted_for_turn
        or _thermal_submitted_for_turn
    )
    _input_text_buf.clear()
    _output_text_buf.clear()
    _vision_submitted_for_turn = False
    _thermal_submitted_for_turn = False
    _perception_submitted_for_turn = False
    _vision_frame_sequence_for_turn = None
    _vision_frame_backend_ns_for_turn = None
    _vision_intent_for_turn = False
    _thermal_intent_for_turn = False
    if had_state:
        print(
            f"[GEMINI-TURN] state_cleared reason={reason} "
            f"generation={gemini_live.session_generation}",
            flush=True,
        )


def _mark_gemini_playing() -> None:
    """Make is_playing_now() return True for the duration of the current
    Gemini Live turn, so ws_audio's mic-mute guard actually engages while
    Gemini is speaking.

    Idempotent — the turn's first _on_audio call creates the marker task;
    later chunks in the same turn are no-ops since it's still running.
    Cleared by cancel_current_ai() in _on_turn_complete (normal end) or by
    hard_reset_audio() -> cancel_current_ai() in _on_interrupted (barge-in).
    The 1h sleep is just a safety net in case a turn ever ends without
    either callback firing — both normal paths clear it well before that.
    """
    if audio_stream.current_ai_task is not None and not audio_stream.current_ai_task.done():
        return

    async def _sentinel():
        await asyncio.sleep(3600)

    task = asyncio.create_task(_sentinel())

    def _clear(t: asyncio.Task) -> None:
        if audio_stream.current_ai_task is t:
            audio_stream.current_ai_task = None

    task.add_done_callback(_clear)
    audio_stream.current_ai_task = task


def _clear_tts_send_queue() -> None:
    while True:
        try:
            _tts_send_queue.get_nowait()
            _tts_send_queue.task_done()
        except asyncio.QueueEmpty:
            return


def _enqueue_tts_chunk(chunk: bytes, turn_id: int) -> None:
    _tts_send_queue.put_nowait(_TTSQueueItem(turn_id=turn_id, chunk=bytes(chunk)))
    latency_tracker.mark_tts_queued(_tts_send_queue.qsize(), turn_id)


def _enqueue_tts_terminal(status: str, turn_id: Optional[int] = None) -> None:
    correlated_turn_id = latency_tracker.active_turn_id if turn_id is None else turn_id
    _tts_send_queue.put_nowait(_TTSQueueItem(
        turn_id=correlated_turn_id or 0,
        terminal_status=status,
    ))


async def _send_esp32_audio_text(ws: WebSocket, message: str) -> None:
    lock = _esp32_audio_send_lock if ws is esp32_audio_ws else None
    if lock is None:
        await ws.send_text(message)
        return
    async with lock:
        await ws.send_text(message)


async def _send_esp32_audio_bytes(ws: WebSocket, chunk: bytes) -> None:
    lock = _esp32_audio_send_lock if ws is esp32_audio_ws else None
    if lock is None:
        await ws.send_bytes(chunk)
        return
    async with lock:
        await ws.send_bytes(chunk)


async def _paced_tts_sender() -> None:
    """Sole owner of paced Gemini Live TTS writes to the ESP32 websocket.

    Paces against a virtual playback clock rather than sleeping one chunk's
    duration per send: the clock accumulates the real duration of everything
    written, so time spent inside send_bytes() cannot make the stream drift
    slower than realtime and starve the firmware's I2S writer. At most
    _TTS_TARGET_LEAD_SEC of audio is ever in flight, keeping qTTS well under
    TTS_QUEUE_DEPTH while still holding a cushion against WiFi jitter.

    Latency turns are finalized by the callbacks that end them, not here —
    draining can lag a turn's end by seconds, and a fast follow-up would
    otherwise mark an already-completed turn as interrupted and evict it
    from the rolling median/p95.
    """
    global _esp32_tts_started
    owner_ws: Optional[WebSocket] = None
    owner_turn_id = 0
    playback_clock = 0.0
    gate_reopen_handle: Optional[asyncio.TimerHandle] = None
    gate_generation = 0

    while True:
        item = await _tts_send_queue.get()
        try:
            if item.terminal_status is not None:
                terminal_turn_id = item.turn_id or owner_turn_id
                terminal_ws = owner_ws
                if (_esp32_tts_started and owner_ws and
                        owner_ws.client_state == WebSocketState.CONNECTED):
                    try:
                        # RESET discards whatever is still queued on the device
                        # (correct for a barge-in); END lets it play out.
                        await _send_esp32_audio_text(
                            owner_ws,
                            f"TTS:RESET:{terminal_turn_id}"
                            if item.terminal_status == "interrupted"
                            else f"TTS:END:{terminal_turn_id}"
                        )
                    except Exception:
                        pass
                _esp32_tts_started = False
                gate_generation += 1
                if gate_reopen_handle is not None:
                    gate_reopen_handle.cancel()
                    gate_reopen_handle = None
                if terminal_ws is not None:
                    # END is ordered after all PCM. Keep freshness suppressed
                    # until the existing virtual playback clock says that PCM
                    # has drained; RESET discards it and can reopen immediately.
                    reopen_delay = (
                        0.0
                        if item.terminal_status == "interrupted"
                        else max(0.0, playback_clock - time.monotonic())
                    )
                    reopen_generation = gate_generation

                    def _reopen_audio_freshness(
                        expected_generation: int = reopen_generation,
                        expected_ws: Optional[WebSocket] = terminal_ws,
                    ) -> None:
                        if (
                            expected_generation == gate_generation
                            and esp32_audio_ws is expected_ws
                            and not _esp32_tts_started
                        ):
                            audio_freshness_tracker.set_gate(
                                expected_streaming=True,
                                reason="streaming",
                            )

                    if reopen_delay == 0.0:
                        _reopen_audio_freshness()
                    else:
                        gate_reopen_handle = asyncio.get_running_loop().call_later(
                            reopen_delay,
                            _reopen_audio_freshness,
                        )
                owner_ws = None
                owner_turn_id = 0
                playback_clock = 0.0
                continue

            chunk = item.chunk
            if chunk is None:
                continue
            ws = esp32_audio_ws
            if not ws or ws.client_state != WebSocketState.CONNECTED:
                _esp32_tts_started = False
                owner_ws = None
                owner_turn_id = 0
                playback_clock = 0.0
                continue

            if (owner_ws is not ws or owner_turn_id != item.turn_id
                    or not _esp32_tts_started):
                gate_generation += 1
                if gate_reopen_handle is not None:
                    gate_reopen_handle.cancel()
                    gate_reopen_handle = None
                await _send_esp32_audio_text(
                    ws, f"TTS:START:{item.turn_id}"
                )
                audio_freshness_tracker.set_gate(
                    expected_streaming=False,
                    reason="tts",
                )
                owner_ws = ws
                owner_turn_id = item.turn_id
                _esp32_tts_started = True
                playback_clock = time.monotonic()
                latency_tracker.mark("tts_start_sent", item.turn_id)

            now = time.monotonic()
            if playback_clock < now:
                playback_clock = now
            lead = playback_clock - now
            if lead > _TTS_TARGET_LEAD_SEC:
                await asyncio.sleep(lead - _TTS_TARGET_LEAD_SEC)
            await _send_esp32_audio_bytes(ws, chunk)
            latency_tracker.mark_tts_sent(len(chunk), item.turn_id)
            playback_clock += len(chunk) / (audio_stream.STREAM_SR * 2)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _esp32_tts_started = False
            owner_ws = None
            owner_turn_id = 0
            playback_clock = 0.0
            print(f"[TTS-WS] paced send failed: {e}", flush=True)
        finally:
            _tts_send_queue.task_done()


async def _on_audio(pcm24k: bytes):
    """Gemini Live streams 24kHz PCM16 audio deltas.

    The ESP32 TTS websocket needs 8kHz PCM: audio_stream.STREAM_SR is 8000,
    so a single 24k->8k resample feeds it directly.
    """
    global _ratecv_state_8k
    turn_id = _ensure_turn()
    first_audio = latency_tracker.mark("first_gemini_audio_received", turn_id)
    if first_audio:
        _log_gemini_timing(
            "on_audio_begin", turn_id, bytes=len(pcm24k)
        )

    _mark_gemini_playing()

    try:
        pcm8k, _ratecv_state_8k = audioop.ratecv(pcm24k, 2, 1, 24000, 8000, _ratecv_state_8k)
    except Exception as e:
        print(f"[Gemini Live] audio resample failed: {e}", flush=True)
        return

    if pcm8k:
        for i in range(0, len(pcm8k), _TTS_CHUNK):
            _enqueue_tts_chunk(pcm8k[i:i + _TTS_CHUNK], turn_id)
            if first_audio and i == 0:
                _log_gemini_timing(
                    "first_tts_chunk_queued",
                    turn_id,
                    bytes=min(_TTS_CHUNK, len(pcm8k)),
                    queue_depth=_tts_send_queue.qsize(),
                )


def build_perception_state(utterance: str = "") -> tuple[Optional[dict], dict]:
    """Build fresh YOLO + hand facts for Gemini in canonical RGB coordinates.

    Returns (perception, raw_metrics) where raw_metrics carries the exact
    yolo/hand_state dicts read here, so telemetry callers observe the same
    cache snapshot used for fusion rather than risking a second, later read
    of a cache that may have advanced (YOLO paces at its own interval).
    """
    requested_target = extract_requested_target(utterance)
    guidance_anchor = guidance_anchor_for_utterance(utterance)
    anchor_reason = guidance_anchor_reason(utterance)
    yolo = yolo_client.latest_perception(
        display_rotation_deg=0,
        requested_target=requested_target,
    )
    hand_state = hand_client.latest_hands()
    raw_metrics = {"yolo": yolo, "hands": hand_state}

    # Gate on backend_frame_age_ms (time since the RGB frame was received),
    # not completion_age_ms (time since inference finished on it) — the
    # latter under-counts staleness by exactly that frame's request_ms, which
    # can matter once a request queues up or a service slows down.
    # completion_age_ms is still recorded via _detector_perception_telemetry.
    yolo_backend_age_ms = _age_ms_since(yolo.get("backend_received_monotonic_ns"))
    yolo_fresh = bool(
        yolo.get("enabled")
        and yolo.get("service_healthy") is not False
        and yolo_backend_age_ms is not None
        and yolo_backend_age_ms <= yolo_client.settings.stale_after_sec * 1000.0
    )

    hands_backend_age_ms = _age_ms_since(hand_state.get("backend_received_monotonic_ns"))
    hands_fresh = bool(
        hand_state.get("enabled")
        and hand_state.get("service_healthy") is not False
        and hands_backend_age_ms is not None
        and hands_backend_age_ms <= hand_client.settings.stale_after_sec * 1000.0
    )

    if not yolo_fresh and not hands_fresh:
        return None, raw_metrics

    objects = []
    if yolo_fresh:
        for obj in (yolo.get("objects") or [])[:8]:
            if isinstance(obj, dict):
                objects.append(dict(obj))

    hands = []
    if hands_fresh:
        for hand in (hand_state.get("hands") or [])[:2]:
            compact = build_hand_fusion_fact(hand)
            if compact is not None:
                hands.append(compact)

    target_selection = (
        dict(yolo.get("guidance_target") or {})
        if yolo_fresh
        else {
            "target_state": "uncertain",
            "stable": False,
            "requested_target": requested_target,
        }
    )
    selected_hand = None
    if hands:
        def _hand_priority(item: dict) -> tuple:
            try:
                handedness_score = float(item.get("handedness_score", 0.0))
            except (TypeError, ValueError):
                handedness_score = 0.0
            return (
                bool(item.get("gesture_authoritative") and item.get("gesture") == "pointing"),
                bool(item.get("gesture_authoritative")),
                bool(item.get("handedness_authoritative")),
                handedness_score,
                -int(item.get("hand_index", 0)),
            )
        selected_hand = max(hands, key=_hand_priority)

    guidance = None
    if selected_hand is not None and yolo_fresh:
        guidance = hand_target_guidance.build(
            target_selection,
            selected_hand,
            guidance_anchor=guidance_anchor,
            anchor_reason=anchor_reason,
            requested_target=requested_target,
            objects_age_ms=yolo.get("age_ms"),
            hands_age_ms=hand_state.get("age_ms"),
            object_frame_id=yolo.get("frame_id"),
            hand_frame_id=hand_state.get("frame_id"),
        )

    return {
        "coordinate_space": "canonical_rgb_normalized_2d",
        "requested_target": requested_target,
        "guidance_anchor": guidance_anchor,
        "guidance_anchor_reason": anchor_reason,
        "objects_fresh": yolo_fresh,
        "objects_age_ms": yolo.get("age_ms") if yolo_fresh else None,
        "object_frame_id": yolo.get("frame_id") if yolo_fresh else None,
        "objects": objects,
        "hands_fresh": hands_fresh,
        "hands_age_ms": hand_state.get("age_ms") if hands_fresh else None,
        "hand_frame_id": hand_state.get("frame_id") if hands_fresh else None,
        "handedness_semantics": {
            "tracker_label": "anatomical_handedness",
            "image_position": "independent canonical RGB left/center/right fact",
            "authoritative_score_threshold": HAND_HANDEDNESS_AUTHORITATIVE_THRESHOLD,
            "gesture_source": "landmark_geometry",
            "gesture_authoritative_only_when_flagged": True,
        },
        "hands": hands,
        "target_selection": target_selection,
        "guidance": guidance,
    }, raw_metrics


async def _on_input_transcription(text: str):
    """User speech transcript, streamed incrementally by Gemini Live."""
    global _vision_submitted_for_turn, _thermal_submitted_for_turn, _perception_submitted_for_turn
    global _vision_frame_sequence_for_turn, _vision_frame_backend_ns_for_turn
    global _vision_intent_for_turn, _thermal_intent_for_turn
    if not text:
        return
    turn_id = _ensure_turn()
    latency_tracker.mark("first_input_transcription", turn_id)
    combined = append_transcription_delta(_input_text_buf, text)
    try:
        # Tagged so the UI can tell this apart from the AI's partial text —
        # see the note in _on_turn_complete about why this also needs a
        # ui_broadcast_final once the turn ends.
        await ui_broadcast_partial("(user) " + combined)
    except Exception:
        pass
    wants_vision = is_explicit_vision_request(combined)
    wants_thermal = is_thermal_request(combined)
    if wants_vision:
        _vision_intent_for_turn = True
    if wants_thermal:
        _thermal_intent_for_turn = True
    if not _vision_submitted_for_turn and (wants_vision or wants_thermal):
        _log_gemini_timing(
            "vision_intent_detected",
            turn_id,
            thermal_intent="yes" if wants_thermal else "no",
        )
        frame = latest_rgb.snapshot()
        if frame.data is not None:
            frame_age_ms = max(
                0.0, (time.monotonic() - frame.timestamp) * 1000
            )
            latency_tracker.mark("vision_frame_selected", turn_id)
            _vision_frame_sequence_for_turn = frame.sequence
            # Same conversion yolo_client.py/hand_client.py use for their
            # backend_received_monotonic_ns, so the two are directly comparable.
            _vision_frame_backend_ns_for_turn = int(frame.timestamp * 1_000_000_000)
            _log_gemini_timing(
                "vision_snapshot",
                turn_id,
                sequence=frame.sequence,
                age_ms=round(frame_age_ms, 3),
                bytes=len(frame.data),
            )
            _vision_submitted_for_turn = await gemini_live.send_image(
                frame.data,
                turn_id=turn_id,
                sequence=frame.sequence,
                source="pre_response",
            )
            print(
                f"[VISION] generation={gemini_live.session_generation} "
                f"turn_id={turn_id} pre-response latest_rgb sequence={frame.sequence} "
                f"submitted={_vision_submitted_for_turn}",
                flush=True,
            )

    # Send fresh specialized perception with the same RGB vision turn.
    if _vision_submitted_for_turn and not _perception_submitted_for_turn:
        perception, perception_raw = build_perception_state(combined)
        latency_tracker.mark("perception_state_built", turn_id)
        perception_bytes = None

        if perception is None:
            print(
                f"[PERCEPTION] turn_id={turn_id} no fresh detector state; not submitted",
                flush=True,
            )
        else:
            perception_payload = (
                "PERCEPTION_STATE "
                + json.dumps(perception, separators=(",", ":"))
                + " Use these fresh detector results as supplemental visual evidence. "
                  "MediaPipe handedness is ANATOMICAL handedness. When "
                  "handedness_authoritative is true, you MUST use that anatomical "
                  "Left/Right label and MUST NOT override it by visually guessing from "
                  "image position. A Right hand can appear at image_left and a Left "
                  "hand can appear at image_right; image position and anatomical "
                  "handedness are different facts. Gesture labels such as fist or open "
                  "palm are backend landmark-geometry facts; when gesture_authoritative "
                  "is true, you MUST use that gesture and MUST NOT override it from the "
                  "image. When target_state is uncertain, do not guess or give directional "
                  "guidance. When target_state is not_found, say the requested target was "
                  "not found and NEVER substitute another detected object. The requested_target "
                  "and guidance_anchor fields are deterministic current-turn backend facts. "
                  "Use the selected anchor and do not substitute a different hand landmark. "
                  "When held_through_miss is true, the bbox is the last validated fresh cached "
                  "position held for one temporary miss, not a new visual observation. "
                  "When target_state is stable, guidance.horizontal_relation and "
                  "guidance.vertical_relation are deterministic backend-computed facts in "
                  "canonical RGB coordinates: do not reverse or reinterpret them from the "
                  "image. Convert stable relations into concise instructions such as "
                  "'Move your right hand left.' Object detections may be incomplete. "
                  "Normalized offsets are 2D image coordinates only; NEVER describe dx_norm "
                  "or dy_norm as meters, inches, physical depth, forward/backward movement, "
                  "or physical distance such as '8 inches forward'. Depth and contact are "
                  "unavailable."
            )

            perception_bytes = len(perception_payload.encode("utf-8"))
            _perception_submitted_for_turn = await gemini_live.send_text(
                perception_payload,
                turn_id=turn_id,
                source="perception_facts",
            )
            if _perception_submitted_for_turn:
                latency_tracker.mark("perception_state_sent", turn_id)

            print(
                f"[PERCEPTION] generation={gemini_live.session_generation} "
                f"turn_id={turn_id} "
                f"objects={len(perception['objects'])} "
                f"hands={len(perception['hands'])} "
                f"hand_facts={compact_hand_facts_for_log(perception['hands'])} "
                f"submitted={_perception_submitted_for_turn}",
                flush=True,
            )
            if _perception_submitted_for_turn and perception.get("guidance") is not None:
                print(
                    "[GUIDANCE] " + compact_guidance_for_log(perception["guidance"]),
                    flush=True,
                )

        frame_id_delta, frame_time_delta_ms = _frame_alignment(
            _vision_frame_sequence_for_turn,
            _vision_frame_backend_ns_for_turn,
            perception_raw["yolo"],
        )
        latency_tracker.update_perception(turn_id, {
            "yolo": _detector_perception_telemetry(perception_raw["yolo"]),
            "hands": _detector_perception_telemetry(perception_raw["hands"]),
            "sent": bool(perception is not None and _perception_submitted_for_turn),
            "bytes": perception_bytes,
            "gemini_rgb_frame_id": _vision_frame_sequence_for_turn,
            "frame_id_delta": frame_id_delta,
            "frame_time_delta_ms": frame_time_delta_ms,
        })

    # Thermal goes as structured text, never as the colorized heatmap.
    if wants_thermal and not _thermal_submitted_for_turn:
        build_started_ns = time.monotonic_ns()
        _log_gemini_timing("thermal_build_begin", turn_id)
        facts = build_thermal_facts()
        _log_gemini_timing(
            "thermal_build_end",
            turn_id,
            build_ms=round(
                (time.monotonic_ns() - build_started_ns) / 1_000_000, 3
            ),
            facts_available="yes" if facts is not None else "no",
        )
        if facts is None:
            print("[THERMAL] no recent thermal frame; facts not submitted", flush=True)
        else:
            _thermal_submitted_for_turn = await gemini_live.send_text(
                "THERMAL_MEASUREMENTS " + json.dumps(facts, separators=(",", ":")),
                turn_id=turn_id,
                source="thermal_facts",
            )
            print(
                f"[THERMAL] generation={gemini_live.session_generation} "
                f"turn_id={turn_id} facts submitted "
                f"ahead={facts['directly_ahead_mean_c']}C "
                f"max={facts['scene_max_c']}C submitted={_thermal_submitted_for_turn}",
                flush=True,
            )
    if _has_hotword(combined):
        async with interrupt_lock:
            print(f"[HOTWORD] '{combined}' -> full reset", flush=True)
            await full_system_reset("Hotword interrupt")

async def _on_output_transcription(text: str):
    """Gemini's spoken-response transcript, streamed incrementally."""
    if not text:
        return
    _output_text_buf.append(text)
    try:
        await ui_broadcast_partial("[AI] " + "".join(_output_text_buf))
    except Exception:
        pass

async def _on_turn_complete():
    """Fires once Gemini has finished a full spoken response.

    Only relevant for AI_BACKEND == "gemini_live" — gemini_regular/qwen
    don't connect to Gemini Live at all (they use local Whisper ASR instead,
    see ws_audio / _run_whisper_and_dispatch), so this callback never fires
    for those backends.

    NOTE: there's no pre-Gemini gate on the live audio stream — Gemini hears
    and may respond to everything the user says, including navigation/command
    phrases like "开始导航". This handler still runs the command dispatcher
    afterwards so the app's own state (navigation mode, item search, etc.)
    stays correct, but the user will already have heard Gemini's spoken
    reply to that utterance by the time it fires. If that's not acceptable,
    consider gating `streaming` in ws_audio so mic audio isn't forwarded to
    Gemini at all while in a restrictive navigation state.
    """
    global omni_conversation_active, omni_previous_nav_state
    global _ratecv_state_8k, _vision_submitted_for_turn, _thermal_submitted_for_turn, _perception_submitted_for_turn
    global _vision_intent_for_turn, _thermal_intent_for_turn

    turn_id = _ensure_turn()
    active_turn = latency_tracker.snapshot().get("current_active_turn") or {}
    integrity_degraded = bool(active_turn.get("audio_integrity_degraded"))
    if integrity_degraded:
        loss = active_turn.get("mic_loss") or {}
        print(
            f"[AUDIO-INTEGRITY] turn_id={turn_id} status=degraded "
            f"epoch={loss.get('epoch', 0)} chunks={loss.get('chunks', 0)} "
            f"duration_ms={loss.get('duration_ms', 0)} "
            f"reason={loss.get('reason', 'unknown')}",
            flush=True,
        )
        try:
            await ui_broadcast_final(
                "[Audio integrity degraded — repeat the question if the response seems wrong.]"
            )
        except Exception:
            pass
    latency_tracker.mark("turn_complete", turn_id)
    _enqueue_tts_terminal("completed", turn_id)
    await _finalize_latency_turn("completed", turn_id)
    # New turn next time — don't carry resample state across turn boundaries
    _ratecv_state_8k = None
    _vision_submitted_for_turn = False
    _thermal_submitted_for_turn = False
    _perception_submitted_for_turn = False
    _vision_intent_for_turn = False
    _thermal_intent_for_turn = False


    # Turn is over — clear the is_playing_now() marker set by _mark_gemini_playing()
    # in _on_audio. (The interrupted/barge-in case is cleared separately, via
    # hard_reset_audio() -> cancel_current_ai() in _on_interrupted.)
    await cancel_current_ai()

    # Broadcast the user's transcript as a FINAL message first, so it's
    # actually visible/persisted in the UI (previously it only ever went out
    # as a partial, which the AI's own partial immediately overwrote — it
    # never showed up because it was never sent as a final).
    user_text = "".join(_input_text_buf).strip()
    _input_text_buf.clear()
    if user_text:
        try:
            await ui_broadcast_final("(user) " + user_text)
        except Exception:
            pass

    # Signal "finished" to any /stream.wav listeners
    await _signal_stream_finished()

    final_ai_text = "".join(_output_text_buf).strip() or "(empty response)"
    print(f"[AI] {final_ai_text}", flush=True)
    try:
        await ui_broadcast_final("[AI] " + final_ai_text)
    except Exception:
        pass
    _output_text_buf.clear()

    if user_text and not STABILITY_MODE:
        async with interrupt_lock:
            # gemini_live already spoke the reply live over audio; this call
            # is only for side effects (nav start/stop, item search, etc.)
            await start_ai_with_text_custom(user_text)

    omni_conversation_active = False
    if orchestrator and omni_previous_nav_state:
        orchestrator.force_state(omni_previous_nav_state)
        if DEBUG: print(f"[OMNI] Dialogue ended, restored to {omni_previous_nav_state} mode")
        omni_previous_nav_state = None

RECONNECT_APOLOGY_TIMEOUT_SEC = 30.0
RECONNECT_APOLOGY_POLL_SEC = 0.5
RECONNECT_APOLOGY_PROMPT = (
    "Say exactly this and nothing else: "
    "\"Sorry, I lost the connection there — could you say that again?\""
)


async def _apologize_for_dropped_turn(reason: str, turn_id: int) -> None:
    """Best-effort notice for a turn abandoned by a session drop, not a user
    barge-in — session resumption restores conversation context, but there's
    no evidence (in this SDK or this app's own history — even the planned
    goaway rotation path abandons active turns the same way) that a specific
    in-flight generation survives a reconnect. Rather than leave the user
    wondering whether they were heard, wait for the session to come back and
    have Gemini say so directly, reusing the existing audio/TTS pipeline.

    Never raises — a missed apology is strictly better than crashing
    anything downstream of it. Skips quietly if the user has already moved
    on to a fresh turn (e.g. they spoke again during the reconnect gap)
    rather than talking over them, and gives up quietly if the session
    doesn't come back within RECONNECT_APOLOGY_TIMEOUT_SEC.
    """
    try:
        waited = 0.0
        while not gemini_live.connected and waited < RECONNECT_APOLOGY_TIMEOUT_SEC:
            await asyncio.sleep(RECONNECT_APOLOGY_POLL_SEC)
            waited += RECONNECT_APOLOGY_POLL_SEC
        if not gemini_live.connected:
            print(
                f"[GEMINI-RECONNECT-APOLOGY] turn_id={turn_id} reason={reason} "
                "gave_up=yes (session did not come back in time)",
                flush=True,
            )
            return
        if latency_tracker.active_turn_id is not None:
            print(
                f"[GEMINI-RECONNECT-APOLOGY] turn_id={turn_id} reason={reason} "
                "skipped=new_turn_already_active",
                flush=True,
            )
            return
        sent = await gemini_live.send_text(
            RECONNECT_APOLOGY_PROMPT, source="reconnect_apology"
        )
        print(
            f"[GEMINI-RECONNECT-APOLOGY] turn_id={turn_id} reason={reason} "
            f"waited_sec={waited:.1f} sent={sent}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[GEMINI-RECONNECT-APOLOGY] turn_id={turn_id} reason={reason} "
            f"failed error={type(exc).__name__}",
            flush=True,
        )


async def _on_interrupted(reason: str = "user_barge_in"):
    """An active turn was cut off — either the user barged in
    (reason="user_barge_in") or the session itself was dropped/rotated out
    from under it (any other reason: "receive_error", "normal_receive_end",
    "goaway", etc.). Either way the ESP32 needs the same TTS-stop cleanup so
    tts_playing clears and the mic un-mutes; only a connection-caused
    interruption that actually had a turn in flight also gets a spoken
    apology once the session comes back (see _apologize_for_dropped_turn).
    """
    global _ratecv_state_8k
    print(f"[Gemini Live] Response interrupted reason={reason}", flush=True)
    turn_id = latency_tracker.active_turn_id

    # Same as _on_turn_complete: tell the firmware the TTS stream is over so
    # tts_playing clears and the mic un-mutes. Without this, a barge-in left
    # the ESP32 stuck in TTS mode until the next full turn happened to send
    # its own TTS:START/TTS:END pair.
    _clear_tts_send_queue()
    if turn_id is not None:
        latency_tracker.mark("interrupted", turn_id)
    _enqueue_tts_terminal("interrupted", turn_id)
    if turn_id is not None:
        await _finalize_latency_turn("interrupted", turn_id)
    _ratecv_state_8k = None
    _clear_gemini_turn_state("interrupted")
    await hard_reset_audio("gemini_interrupted")

    if reason != "user_barge_in" and turn_id is not None:
        asyncio.create_task(_apologize_for_dropped_turn(reason, turn_id))


async def _on_gemini_session_transition(reason: str, _generation: int):
    _clear_gemini_turn_state(f"session_{reason}")

gemini_live.on_audio = _on_audio
gemini_live.on_input_transcription = _on_input_transcription
gemini_live.on_output_transcription = _on_output_transcription
gemini_live.on_turn_complete = _on_turn_complete
gemini_live.on_interrupted = _on_interrupted
gemini_live.on_session_transition = _on_gemini_session_transition
gemini_live.turn_id_provider = lambda: latency_tracker.active_turn_id

# ---- Helper for the non-live backends (gemini_regular / qwen) ----
# These backends are text-only for this comparison — no TTS, no audio out.
# Only AI_BACKEND == "gemini_live" touches audio at all; ASR for the other
# two still runs through Gemini Live (in TEXT response mode, see
# startup_gemini) purely so transcription stays consistent across all three
# tests, but their replies are text-in/text-out to the UI only.

async def _signal_stream_finished():
    """Tell any /stream.wav listeners the current (gemini_live) audio
    response is over. Only used by the gemini_live path — the text-only
    backends never open an audio stream in the first place."""
    from audio_stream import stream_clients  # local import to avoid circular dependency
    for sc in list(stream_clients):
        if not sc.abort_event.is_set():
            try: sc.q.put_nowait(b"\x00" * BYTES_PER_20MS_16K)  # one frame of silence
            except Exception: pass
            try: sc.q.put_nowait(None)
            except Exception: pass

async def run_backend_turn(user_text: str):
    """Generate a text reply using the selected non-live backend and
    broadcast it to the UI. No audio synthesis — gemini_regular/qwen are
    being compared as text backends here, with Gemini Live handling the
    full audio pipeline separately when AI_BACKEND == "gemini_live"."""
    global omni_conversation_active, omni_previous_nav_state

    if _backend_stream_chat is None:
        print(f"[BACKEND] No stream_chat available for AI_BACKEND={AI_BACKEND!r}", flush=True)
        return

    content_list = []
    if last_frames and (not STABILITY_MODE or is_explicit_vision_request(user_text)):
        try:
            _, jpeg_bytes = last_frames[-1]
            img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        except Exception:
            pass
    content_list.append({"type": "text", "text": user_text})

    txt_buf: List[str] = []
    try:
        async for piece in _backend_stream_chat(content_list, voice="Cherry", audio_format="wav"):
            if piece.text_delta:
                txt_buf.append(piece.text_delta)
                try:
                    await ui_broadcast_partial("[AI] " + "".join(txt_buf))
                except Exception:
                    pass
    except Exception as e:
        print(f"[BACKEND:{AI_BACKEND}] generation failed: {e}", flush=True)
        try:
            await ui_broadcast_final(f"[AI] Error occurred: {e}")
        except Exception:
            pass
        txt_buf = []

    final_text = "".join(txt_buf).strip()
    if final_text:
        print(f"[AI] {final_text}", flush=True)
        try:
            await ui_broadcast_final("[AI] " + final_text)
        except Exception:
            pass

    omni_conversation_active = False
    if orchestrator and omni_previous_nav_state:
        orchestrator.force_state(omni_previous_nav_state)
        if DEBUG: print(f"[OMNI] Dialogue ended, restored to {omni_previous_nav_state} mode")
        omni_previous_nav_state = None

# ---- Synchronous recorder ----
import sync_recorder
import signal
import atexit
from stability_runtime import (
    MSG_TYPE_CAM, MSG_TYPE_THERMAL, MSG_TYPE_IMU, MSG_TYPE_STATUS,
    AudioFreshnessTracker, LatencyTracker, LatestFrameStore,
    PCM_20MS_BYTES_16K_MONO, RecordingPipeline, VisionController,
    append_transcription_delta, parse_mic_loss_message, parse_sensor_message,
)
from yolo_client import YoloClientSettings, YoloShadowClient
from hand_client import HandClientSettings, HandTrackingClient
from research_exporter import (
    DEFAULT_SESSION_STATE_PATH, DEFAULT_SESSION_TTL_SEC,
    ResearchExporter, ResearchExporterSettings, SessionGate,
)

# ---- IMU UDP ----
UDP_IP   = "0.0.0.0"
UDP_PORT = 12345

app = FastAPI()

# ====== State and containers ======
app.mount("/static", StaticFiles(directory="static"), name="static")

ui_clients: Dict[int, WebSocket] = {}
current_partial: str = ""
recent_finals: List[str] = []
RECENT_MAX = 50
# Legacy chat backends read this cache; strict mode must retain only the
# newest JPEG, matching latest_rgb's single-value contract.
last_frames: Deque[Tuple[float, bytes]] = deque(maxlen=1 if STABILITY_MODE else 10)
latest_rgb = LatestFrameStore()
latest_thermal = LatestFrameStore()
rgb_canonicalizer_telemetry = RgbCanonicalizerTelemetry()

# Research platform integration: bounded, best-effort, off by default. See
# research_exporter.py's module docstring for the full safety contract —
# nothing here can block or slow the camera/audio/Gemini hot paths.
research_session_gate = SessionGate(
    state_path=os.getenv("RESEARCH_SESSION_STATE_PATH", DEFAULT_SESSION_STATE_PATH),
    default_ttl_sec=float(os.getenv("RESEARCH_SESSION_DEFAULT_TTL_SEC", str(DEFAULT_SESSION_TTL_SEC))),
)
research_exporter = ResearchExporter(
    ResearchExporterSettings.from_env(),
    latest_rgb,
    research_session_gate,
)

yolo_settings = YoloClientSettings.from_env()
yolo_client = YoloShadowClient(
    yolo_settings,
    latest_rgb,
    rotation_provider=lambda: 0,
    on_event=research_exporter.publish_event,
)
hand_target_guidance = HandTargetGuidanceTracker(
    deadband=yolo_settings.guidance_deadband,
)
hand_settings = HandClientSettings.from_env()
hand_client = HandTrackingClient(hand_settings, latest_rgb)
latency_tracker = LatencyTracker(history_size=200)
latency_tracker = LatencyTracker(history_size=200, on_event=research_exporter.publish_event)
audio_freshness_tracker = AudioFreshnessTracker(
    stale_after_ms=float(os.getenv("AUDIO_STALE_AFTER_MS", "500")),
    recent_window_sec=2.0,
)
_latency_csv_lock = threading.Lock()

# Free-text label for a block of turns under manual test (e.g. "far-room-walking"),
# set via POST /api/session/start and cleared via POST /api/session/stop. Not
# related to research_session_gate above — that's an external research-platform
# activation gate; this is purely a label carried on this app's own turn telemetry.
_telemetry_session_id: Optional[str] = None
_telemetry_session_label: Optional[str] = None

# Snapshot of (session_id, session_label) taken the moment each turn_id is first
# created, so a turn started mid-session keeps that session's tag even if
# /api/session/stop (or a new /api/session/start) fires before the turn finishes —
# the transition itself is what a labeled test block is trying to capture.
_turn_session_tags: Dict[int, Tuple[Optional[str], Optional[str]]] = {}


def _tag_turn_session(turn_id: int) -> int:
    _turn_session_tags.setdefault(turn_id, (_telemetry_session_id, _telemetry_session_label))
    return turn_id


def _ensure_turn() -> int:
    return _tag_turn_session(latency_tracker.ensure_turn())


def _start_turn(turn_id: Optional[int] = None) -> int:
    return _tag_turn_session(latency_tracker.start_turn(turn_id))

# Persisted per-turn telemetry (latency timestamps, tts stats, and the
# perception record) as JSONL on the Fly volume, alongside backend.log.
# Rotating handler caps disk usage; a missing /data (e.g. local dev outside
# Docker) disables this with one warning instead of failing every turn.
_TURN_TELEMETRY_PATH = os.getenv("TURN_TELEMETRY_PATH", "/data/turn_telemetry.jsonl")
_TURN_TELEMETRY_MAX_BYTES = int(os.getenv("TURN_TELEMETRY_MAX_BYTES", str(20 * 1024 * 1024)))
_TURN_TELEMETRY_BACKUP_COUNT = int(os.getenv("TURN_TELEMETRY_BACKUP_COUNT", "3"))
_turn_telemetry_logger = logging.getLogger("turn_telemetry")
_turn_telemetry_logger.setLevel(logging.INFO)
_turn_telemetry_logger.propagate = False
_turn_telemetry_enabled = False
try:
    _turn_telemetry_handler = RotatingFileHandler(
        _TURN_TELEMETRY_PATH,
        maxBytes=_TURN_TELEMETRY_MAX_BYTES,
        backupCount=_TURN_TELEMETRY_BACKUP_COUNT,
        encoding="utf-8",
    )
    _turn_telemetry_handler.setFormatter(logging.Formatter("%(message)s"))
    _turn_telemetry_logger.addHandler(_turn_telemetry_handler)
    _turn_telemetry_enabled = True
except OSError as exc:
    print(f"[TURN-TELEMETRY] disabled, cannot open {_TURN_TELEMETRY_PATH}: {exc}", flush=True)

# Canonical 32x24 Celsius grid shared by facts, heatmap, and calibration.
latest_thermal_matrix: Optional[np.ndarray] = None
latest_imu: Dict[str, Any] = {"timestamp": 0.0, "sequence": 0, "data": None}
latest_device_status: Dict[str, Any] = {"timestamp": 0.0, "data": None}
thermal_display_config: Dict[str, Any] = {
    "palette": "inferno",
    "auto_range": True,
    "min_c": 15.0,
    "max_c": 40.0,
    "hotspot": True,
    "labels": True,
    "interpolation": "cubic",
    # Neutral coarse 2D alignment. These map canonical thermal normalized
    # coordinates into canonical RGB; they never change temperature data.
    "calibration_offset_x": 0.0,
    "calibration_offset_y": 0.0,
    "calibration_scale_x": 1.0,
    "calibration_scale_y": 1.0,
}
VISION_FRAME_MAX_AGE_SEC = float(os.getenv("VISION_FRAME_MAX_AGE_SEC", "3.0"))
VISION_MIN_INTERVAL_SEC = float(os.getenv("VISION_MIN_INTERVAL_SEC", "2.0"))

camera_viewers: Set[WebSocket] = set()
thermal_viewers: Set[WebSocket] = set()
esp32_camera_ws: Optional[WebSocket] = None
imu_ws_clients: Set[WebSocket] = set()
esp32_audio_ws: Optional[WebSocket] = None
_esp32_audio_send_lock: Optional[asyncio.Lock] = None

# True while the ESP32 has explicitly placed the microphone in START mode.
# Gemini Live receives a low-rate stream of current camera frames during this
# window, allowing natural visual questions without a hard-coded phrase list.
mic_streaming = False
GEMINI_VIDEO_INTERVAL_SEC = max(
    0.25, float(os.getenv("GEMINI_VIDEO_INTERVAL_SEC", "0.75"))
)
_last_gemini_video_submit = 0.0

# The video pump above only runs while mic_streaming is true, so any gap
# between conversations longer than that leaves the Gemini Live session with
# nothing at all being sent. Gemini's server closes such idle sessions with
# WS close code 1008 after ~150s (measured empirically from backend.log —
# not documented). This loop sends an occasional frame during those gaps to
# keep the session alive; see _gemini_keepalive_loop().
GEMINI_KEEPALIVE_INTERVAL_SEC = max(
    30.0, float(os.getenv("GEMINI_KEEPALIVE_INTERVAL_SEC", "60"))
)
_gemini_keepalive_task: Optional[asyncio.Task] = None

# Same drop-to-latest / rate-limited shape as the Gemini video pump above,
# decoupled to its own interval so research sampling never competes with
# Gemini's cadence. research_exporter.publish_event() is itself a no-op
# unless ENABLE_RESEARCH_EXPORT is set AND a research session is active, so
# this call is free (one attribute read) the rest of the time.
RESEARCH_FRAME_SAMPLE_INTERVAL_SEC = max(
    1.0, float(os.getenv("RESEARCH_FRAME_SAMPLE_INTERVAL_SEC", "5.0"))
)
_last_research_frame_sample = 0.0

# Bounded ingest hand-offs keep ESP32 receive loops independent of Gemini,
# OpenCV, and slow browser viewers. Drop-oldest preserves real-time behavior.
ESP_AUDIO_INGEST_QUEUE_MAX = 12
ESP_THERMAL_INGEST_QUEUE_MAX = 1
ESP_AUDIO_HEALTH_INTERVAL_SEC = 10.0
ESP_AUDIO_GAP_BUCKET_LIMITS_MS = (30, 50, 100, 250, 500, 1000)
backend_metrics = {
    "camera_sensor_connects": 0,
    "camera_sensor_disconnects": 0,
    "audio_connects": 0,
    "audio_disconnects": 0,
    "invalid_sensor_packets": 0,
    "last_audio_activity": 0.0,
}


def _append_latency_csv(record: dict) -> None:
    csv_path = os.getenv("LATENCY_LOG_CSV", "").strip()
    if not csv_path:
        return
    fields = (
        "turn_id", "status",
        "speech_end_to_first_gemini_audio_ms",
        "speech_end_to_first_tts_send_ms",
        "first_mic_to_first_gemini_audio_ms",
        "gemini_audio_to_first_tts_send_ms",
        "backend_turn_total_ms",
        "speech_end_to_first_i2s_ms",
        "tts_chunks", "tts_bytes", "max_queue_depth",
    )
    row = {field: record.get(field) for field in fields}
    with _latency_csv_lock:
        needs_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)


async def _append_latency_csv_safely(record: dict) -> None:
    try:
        await asyncio.to_thread(_append_latency_csv, record)
    except Exception as exc:
        print(f"[LATENCY] CSV append failed: {exc}", flush=True)


def _append_turn_telemetry(record: dict) -> None:
    _turn_telemetry_logger.info(json.dumps(record, separators=(",", ":")))


async def _append_turn_telemetry_safely(record: dict) -> None:
    try:
        await asyncio.to_thread(_append_turn_telemetry, record)
    except Exception as exc:
        print(f"[TURN-TELEMETRY] append failed: {exc}", flush=True)


async def _finalize_latency_turn(status: str, turn_id: int) -> None:
    record = latency_tracker.finish(status, turn_id)
    if record is None:
        return
    # Read here rather than at turn-start: these buffers/flags are still the
    # ones for `turn_id` at every call site (all four call this before the
    # buffers/flags get cleared), so this is the full-turn content.
    record["input_transcription"] = "".join(_input_text_buf).strip() or None
    record["response_text"] = "".join(_output_text_buf).strip() or None
    record["vision_intent"] = _vision_intent_for_turn
    record["thermal_intent"] = _thermal_intent_for_turn
    # Unlike the above, session tag IS taken at turn-start (see _tag_turn_session)
    # so a session boundary crossed mid-turn doesn't erase which block it was in.
    session_id, session_label = _turn_session_tags.pop(turn_id, (None, None))
    record["session_id"] = session_id
    record["session_label"] = session_label
    print("[LATENCY] " + json.dumps(record, separators=(",", ":")), flush=True)
    if status == "completed" and os.getenv("LATENCY_LOG_CSV", "").strip():
        asyncio.create_task(_append_latency_csv_safely(record))
    if _turn_telemetry_enabled:
        asyncio.create_task(_append_turn_telemetry_safely(record))

recording_pipeline = RecordingPipeline(
    sync_recorder.record_frame,
    maxsize=2,
    on_failure=sync_recorder.stop_recording,
)
recording_audio_pipeline = RecordingPipeline(
    lambda pcm: sync_recorder.record_audio(pcm, text="[Gemini audio]"),
    maxsize=8,
    on_failure=sync_recorder.stop_recording,
)
vision_controller = VisionController(
    latest_rgb,
    gemini_live.send_image,
    max_age_sec=VISION_FRAME_MAX_AGE_SEC,
    min_interval_sec=VISION_MIN_INTERVAL_SEC,
)


def _device_id(ws: WebSocket) -> str:
    client = getattr(ws, "client", None)
    return f"{client.host}:{client.port}" if client else "unknown"


_VISION_PHRASES = (
    "what is in front", "what's in front", "describe this scene",
    "describe the scene", "read this sign", "what am i holding",
    "is there a chair", "look at this", "what do you see",
)


def is_explicit_vision_request(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if should_wait_for_explicit_target(normalized):
        return False
    return (
        any(phrase in normalized for phrase in _VISION_PHRASES)
        or is_hand_perception_request(normalized)
        or (
            is_target_directed_request(normalized)
            and extract_requested_target(normalized) is not None
        )
    )

# ---- Thermal facts for Gemini ----
# Gemini receives measurements, never the colorized heatmap: palette and
# auto-range change every frame, so color-reading is unrepeatable. These come
# straight off the canonical float32 grid in latest_thermal_matrix.
THERMAL_FACT_MAX_AGE_SEC = float(os.getenv("THERMAL_FACT_MAX_AGE_SEC", "3.0"))
# Fraction of the canonical thermal grid treated as "directly ahead" for raw
# temperature summaries. RGB alignment is separately represented by the coarse
# tunable 2D mapping below; neither path claims depth or removes parallax.
THERMAL_CENTER_FRACTION = 0.4

_THERMAL_PHRASES = (
    "hot", "warm", "cold", "cool", "temperature", "burn", "heat",
    "safe to touch", "safe to hold", "boiling", "is the stove", "is the oven",
)


def is_thermal_request(text: str) -> bool:
    normalized = (text or "").strip().lower()
    return any(phrase in normalized for phrase in _THERMAL_PHRASES)


def build_thermal_facts() -> Optional[dict]:
    """Structured thermal measurements for the current view, or None.

    Reported on a coarse 3x3 grid rather than per object: there is no
    bounding-box source in STABILITY_MODE (YOLO disabled), and 3x3 answers the
    questions that matter to a BLV user without asserting a spatial precision
    this hardware has not been calibrated for.
    """
    matrix = latest_thermal_matrix
    if matrix is None:
        return None
    snapshot = latest_thermal.snapshot()
    if snapshot.data is None:
        return None
    age = time.monotonic() - snapshot.timestamp
    if age > THERMAL_FACT_MAX_AGE_SEC:
        return None
    facts = summarize_thermal_grid(matrix, age, THERMAL_CENTER_FRACTION)
    if facts is None:
        return None
    hotspot_norm = facts.get("hotspot", {}).get("thermal_norm")
    if isinstance(hotspot_norm, list) and len(hotspot_norm) == 2:
        mapped = map_thermal_to_rgb_normalized(
            hotspot_norm[0],
            hotspot_norm[1],
            offset_x=thermal_display_config["calibration_offset_x"],
            offset_y=thermal_display_config["calibration_offset_y"],
            scale_x=thermal_display_config["calibration_scale_x"],
            scale_y=thermal_display_config["calibration_scale_y"],
        )
        facts["hotspot"]["rgb_norm_coarse"] = [round(value, 6) for value in mapped]
    facts["spatial_alignment_note"] = (
        "Canonical thermal coordinates mapped to canonical RGB by a coarse "
        "tunable 2D affine alignment. This is not depth, distance, or 3D "
        "calibration and remains subject to parallax. Temperatures are unwarped."
    )
    return facts


async def request_gemini_vision(reason: str) -> dict:
    result = await vision_controller.request(reason)
    print(
        f"[VISION] trigger={reason!r} ok={result.get('ok')} "
        f"result={result.get('error', 'submitted')}",
        flush=True,
    )
    return result

# Global variables for blind-path navigation
blind_path_navigator = None
navigation_active = False
yolo_seg_model = None
obstacle_detector = None

# Global variables for cross-street navigation
cross_street_navigator = None
cross_street_active = False
orchestrator = None

# Omni conversation state flags
omni_conversation_active = False  # marks whether an omni conversation is in progress
omni_previous_nav_state = None  # saves the navigation state before omni was activated, for restoration

# General object-detection (prompt-free YOLOE) state.
# Boxes + labels are drawn on the video stream only — no audio, no text narration.
general_detect_active = False       # True while "detect objects" mode is running

# Model loading function
def load_navigation_models():
    """Load the models required for blind-path navigation."""
    global yolo_seg_model, obstacle_detector

    if YOLO is None or torch is None:
        print("[NAVIGATION] torch/ultralytics unavailable — navigation disabled")
        return

    try:
        seg_model_path = os.getenv(
            "BLIND_PATH_MODEL",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "yolo-seg.pt"),
        )
        #print(f"[NAVIGATION] Trying to load model: {seg_model_path}")

        if os.path.exists(seg_model_path):
            if DEBUG: print(f"[NAVIGATION] Model file found, starting load...")
            yolo_seg_model = YOLO(seg_model_path)

            # Force the model onto GPU
            if torch.cuda.is_available():
                yolo_seg_model.to("cuda")
                if DEBUG: print(f"[NAVIGATION] Blind-path segmentation model loaded and moved to GPU: {yolo_seg_model.device}")
            else:
                if DEBUG: print("[NAVIGATION] CUDA not available, model remains on CPU")

            # Test whether the model runs correctly
            try:
                test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                results = yolo_seg_model.predict(
                    test_img,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    verbose=False
                )
                if DEBUG: print(f"[NAVIGATION] Model test succeeded, supported class count: {len(yolo_seg_model.names) if hasattr(yolo_seg_model, 'names') else 'unknown'}")
            except Exception as e:
                print(f"[NAVIGATION] Model test failed: {e}")
        else:
            print(f"[NAVIGATION] Error: model file not found: {seg_model_path}")

        # Use ObstacleDetectorClient instead of YOLO directly
        obstacle_model_path = os.getenv(
            "OBSTACLE_MODEL",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "yoloe-11l-seg.pt"),
        )
        if DEBUG: print(f"[NAVIGATION] Attempting to load obstacle detection model: {obstacle_model_path}")

        if os.path.exists(obstacle_model_path):
            if DEBUG: print(f"[NAVIGATION] Obstacle detection model file found, starting load...")
            try:
                # Use YOLO-E wrapped inside ObstacleDetectorClient
                obstacle_detector = ObstacleDetectorClient(model_path=obstacle_model_path)
                if DEBUG: print(f"[NAVIGATION] YOLO-E obstacle detector loaded successfully")

                # Test the obstacle detection functionality
                if DEBUG:
                    try:
                        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                        cv2.rectangle(test_img, (200, 200), (400, 400), (255, 255, 255), -1)
                        test_results = obstacle_detector.detect(test_img)
                        print(f"[NAVIGATION] YOLO-E detection test: {len(test_results)} objects")
                    except Exception as e:
                        print(f"[NAVIGATION] YOLO-E detection test failed: {e}")

            except Exception as e:
                print(f"[NAVIGATION] Obstacle detector load failed: {e}")
                import traceback
                traceback.print_exc()
                obstacle_detector = None
        else:
            if DEBUG: print(f"[NAVIGATION] Warning: obstacle detection model file not found: {obstacle_model_path}")

    except Exception as e:
        print(f"[NAVIGATION] Model load failed: {e}")
        import traceback
        traceback.print_exc()

# Load models at program startup
if not STABILITY_MODE:
    if DEBUG: print("[NAVIGATION] Loading navigation models...")
    load_navigation_models()
    if DEBUG: print(f"[NAVIGATION] Model loading complete - yolo_seg_model: {yolo_seg_model is not None}")
else:
    print("[STABILITY] Navigation, YOLO, item search, and traffic-light processing bypassed")

# Recording starts in the FastAPI startup hook so importing the module for
# tests does not create empty files.

# Register exit handler to ensure recordings are saved on Ctrl+C
def cleanup_on_exit():
    """Clean up resources on program exit."""
    print("\n[SYSTEM] Shutting down recorder...")
    try:
        sync_recorder.stop_recording()
        print("[SYSTEM] Recording files saved")
    except Exception as e:
        print(f"[SYSTEM] Error while shutting down recorder: {e}")

def signal_handler(sig, frame):
    """Handle Ctrl+C / SIGTERM signals."""
    print("\n[SYSTEM] Interrupt signal received, shutting down safely...")
    cleanup_on_exit()
    import sys
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # termination signal
atexit.register(cleanup_on_exit)               # also called on normal exit

if DEBUG: print("[RECORDER] Exit handler registered")



# Pre-load only outside strict stabilization mode.
if not STABILITY_MODE:
    try:
        import trafficlight_detection
        if DEBUG: print("[TRAFFIC_LIGHT] Pre-loading traffic-light detection model...")
        if trafficlight_detection.init_model():
            if DEBUG: print("[TRAFFIC_LIGHT] Traffic-light detection model pre-loaded successfully")
            try:
                test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = trafficlight_detection.process_single_frame(test_img)
                if DEBUG: print("[TRAFFIC_LIGHT] Model warmup complete")
            except Exception as e:
                print(f"[TRAFFIC_LIGHT] Model warmup failed: {e}")
    except Exception as e:
        print(f"[TRAFFIC_LIGHT] Traffic-light model pre-load error: {e}")

# ============== Key: system-level "hard reset" master switch =================
interrupt_lock = asyncio.Lock()

# ============== YOLO media thread management =================
yolomedia_thread: Optional[threading.Thread] = None
yolomedia_stop_event = threading.Event()
yolomedia_running = False
yolomedia_sending_frames = False  # marks whether YOLO has started sending processed frames

# Mapping from item names to YOLO class labels
ITEM_TO_CLASS_MAP = {
    "红牛": "Red_Bull",
    "AD钙奶": "AD_milk",
    "ad钙奶": "AD_milk",
    "钙奶": "AD_milk",
}

async def ui_broadcast_raw(msg: str):
    dead = []
    for k, ws in list(ui_clients.items()):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(k)
    for k in dead:
        ui_clients.pop(k, None)


async def ui_broadcast_partial(text: str):
    global current_partial
    current_partial = text
    await ui_broadcast_raw("PARTIAL:" + text)

async def ui_broadcast_final(text: str):
    global current_partial, recent_finals
    current_partial = ""
    recent_finals.append(text)
    if len(recent_finals) > RECENT_MAX:
        recent_finals = recent_finals[-RECENT_MAX:]
    await ui_broadcast_raw("FINAL:" + text)
    print(f"[ASR/AI FINAL] {text}", flush=True)

async def full_system_reset(reason: str = ""):
    """
    Restore the system to its just-started state:
    1) Stop playback + cancel AI task + disconnect all /stream.wav clients (hard_reset_audio)
    2) Stop the ASR real-time recognition stream (critical)
    3) Clear UI state
    4) Clear recent camera frames (avoid feeding stale frames into the next round)
    5) Notify ESP32: RESET (optional)
    """
    # 1) Audio & AI
    _clear_tts_send_queue()
    _enqueue_tts_terminal("interrupted")
    _clear_gemini_turn_state("full_system_reset")
    await hard_reset_audio(reason or "full_system_reset")

    # 2) ASR
    await stop_current_recognition()

    # 3) UI
    global current_partial, recent_finals
    current_partial = ""
    recent_finals = []

    # 4) Camera frames
    try:
        last_frames.clear()
    except Exception:
        pass

    # 5) Notify ESP32
    try:
        if esp32_audio_ws and (esp32_audio_ws.client_state == WebSocketState.CONNECTED):
            await esp32_audio_ws.send_text("RESET")
    except Exception:
        pass

    if DEBUG: print("[SYSTEM] full reset done.", flush=True)

# ========= Start/Stop YOLO media processing =========
def start_yolomedia_with_target(target_name: str):
    """Start the yolomedia worker thread to search for the specified item."""
    global yolomedia_thread, yolomedia_stop_event, yolomedia_running, yolomedia_sending_frames
    
    # If already running, stop first
    if yolomedia_running:
        stop_yolomedia()
    
    # Look up the corresponding YOLO class label
    yolo_class = ITEM_TO_CLASS_MAP.get(target_name, target_name)
    if DEBUG: print(f"[YOLOMEDIA] Starting with target: {target_name} -> YOLO class: {yolo_class}", flush=True)
    
    yolomedia_stop_event.clear()
    yolomedia_running = True
    yolomedia_sending_frames = False  # reset frame-sending flag
    
    def _run():
        try:
            # Pass the target class name and stop event
            yolomedia.main(headless=True, prompt_name=yolo_class, stop_event=yolomedia_stop_event)
        except Exception as e:
            print(f"[YOLOMEDIA] worker stopped: {e}", flush=True)
        finally:
            global yolomedia_running, yolomedia_sending_frames
            yolomedia_running = False
            yolomedia_sending_frames = False
    
    yolomedia_thread = threading.Thread(target=_run, daemon=True)
    yolomedia_thread.start()
    if DEBUG: print(f"[YOLOMEDIA] background worker started for: {yolo_class}", flush=True)

def stop_yolomedia():
    """Stop the yolomedia worker thread."""
    global yolomedia_thread, yolomedia_stop_event, yolomedia_running, yolomedia_sending_frames
    
    if yolomedia_running:
        if DEBUG: print("[YOLOMEDIA] Stopping worker...", flush=True)
        yolomedia_stop_event.set()

        # Wait for the thread to finish (up to 5 seconds)
        if yolomedia_thread and yolomedia_thread.is_alive():
            yolomedia_thread.join(timeout=5.0)

        yolomedia_running = False
        yolomedia_sending_frames = False
        if DEBUG: print("[YOLOMEDIA] Worker stopped.", flush=True)

# ========= Custom start_ai_with_text, with special command recognition =========
async def start_ai_with_text_custom(user_text: str):
    """Extended AI launch function with special command recognition."""
    global navigation_active, blind_path_navigator, cross_street_active, cross_street_navigator, orchestrator

    # Lower-case once so English phrase matching is case-insensitive. Lower-casing
    # Chinese characters is a no-op so the existing Chinese checks still work.
    user_text = user_text.lower()

    # ---- General object detection (prompt-free YOLOE, continuous stream) ----
    # Draws boxes + labels on the video only — no spoken audio, no text narration.
    # Checked before the orchestrator/navigation guards so it works in any mode.
    global general_detect_active
    if any(k in user_text for k in ["detect objects", "detect object",
                                    "list objects", "what objects",
                                    "检测物体", "识别物体"]):
        if yolomedia_running:
            stop_yolomedia()
        general_detect_active = True
        # Warm up the model in the background so the first frame isn't laggy.
        threading.Thread(target=general_detector.load, daemon=True).start()
        return
    if any(k in user_text for k in ["stop detecting objects", "stop object detection",
                                    "stop detecting", "stop objects",
                                    "停止检测物体", "停止识别物体"]):
        general_detect_active = False
        return

    # In navigation or traffic-light detection mode, only specific words trigger omni dialogue
    if orchestrator:
        current_state = orchestrator.get_state()
        # If in navigation or traffic-light detection mode (not CHAT mode)
        if current_state not in ["CHAT", "IDLE"]:
            # Check whether the utterance is an allowed dialogue trigger keyword
            allowed_keywords = [
                # Chinese
                "帮我看", "帮我看下", "帮我找", "找一下", "看看", "识别一下",
                # English
                "what is this", "what is that", "what's this", "what's that",
                "describe", "look at", "identify", "find",
            ]
            is_allowed_query = any(keyword in user_text for keyword in allowed_keywords)

            # Check whether the utterance is a navigation control command
            nav_control_keywords = [
                # Chinese
                "开始过马路", "过马路结束", "开始导航", "盲道导航", "停止导航", "结束导航",
                "检测红绿灯", "看红绿灯", "停止检测", "停止红绿灯",
                # English
                "start crossing", "stop crossing", "end crossing",
                "start navigation", "stop navigation", "end navigation",
                "detect traffic light", "check traffic light", "stop detection",
            ]
            is_nav_control = any(keyword in user_text for keyword in nav_control_keywords)
            
            # If neither an allowed query nor a navigation control command, discard
            if not is_allowed_query and not is_nav_control:
                if DEBUG:
                    mode_name = "Traffic light detection" if current_state == "TRAFFIC_LIGHT_DETECTION" else "Navigation"
                    print(f"[{mode_name} mode] Discarding non-dialogue audio: {user_text}")
                return  # discard; do not enter omni
    
    # Check for street-crossing commands — use orchestrator to control
    if any(k in user_text for k in ["开始过马路", "帮我过马路",
                                     "start crossing", "help me cross", "cross the street"]):
        # If currently searching for an item, stop first
        if yolomedia_running:
            stop_yolomedia()
            print("[ITEM_SEARCH] Switching from item-search mode to street-crossing")

        if orchestrator:
            orchestrator.start_crossing()
            if DEBUG: print(f"[CROSS_STREET] Street-crossing mode started, state: {orchestrator.get_state()}")
            # Play launch voice prompt and broadcast to UI. Local say-based TTS
            # is only for gemini_regular/qwen — gemini_live speaks via its own
            # _on_audio path, so firing this too would double-speak.
            if AI_BACKEND != "gemini_live":
                play_voice_text("Street crossing mode activated.")
            await ui_broadcast_final("[System] Street-crossing mode started")
        else:
            print("[CROSS_STREET] Warning: navigation master not initialized!")
            if AI_BACKEND != "gemini_live":
                play_voice_text("Failed to start crossing mode, please try again later.")
            await ui_broadcast_final("[System] Navigation system not ready")
        return
    
    if any(k in user_text for k in ["过马路结束", "结束过马路",
                                     "stop crossing", "end crossing", "done crossing"]):
        if orchestrator:
            orchestrator.stop_navigation()
            if DEBUG: print(f"[CROSS_STREET] Navigation stopped, state: {orchestrator.get_state()}")
            # Play stop voice prompt and broadcast to UI
            if AI_BACKEND != "gemini_live":
                play_voice_text("Navigation stopped.")
            await ui_broadcast_final("[System] Street-crossing mode stopped")
        else:
            await ui_broadcast_final("[System] Navigation system not running")
        return
    
    # Check for traffic-light detection command — mutually exclusive with blind-path navigation
    if any(k in user_text for k in ["检测红绿灯", "看红绿灯",
                                     "detect traffic light", "check traffic light",
                                     "what color is the light", "what light"]):
        try:
            import trafficlight_detection
            
            # Switch orchestrator to traffic-light detection mode (pause blind-path navigation)
            if orchestrator:
                orchestrator.start_traffic_light_detection()
                if DEBUG: print(f"[TRAFFIC] Switched to traffic-light detection mode, state: {orchestrator.get_state()}")
            
            # Use main-thread processing instead of a separate thread to avoid dropped frames
            success = trafficlight_detection.init_model()  # initialise model only; do not start a thread
            trafficlight_detection.reset_detection_state()  # reset state

            if success:
                await ui_broadcast_final("[System] Traffic-light detection started")
            else:
                await ui_broadcast_final("[System] Traffic-light model load failed")
        except Exception as e:
            print(f"[TRAFFIC] Failed to start traffic-light detection: {e}")
            await ui_broadcast_final(f"[System] Start failed: {e}")
        return
    
    if any(k in user_text for k in ["停止检测", "停止红绿灯",
                                     "stop detection", "stop traffic light"]):
        try:
            # Restore to dialogue (CHAT) mode
            if orchestrator:
                orchestrator.stop_navigation()  # return to CHAT mode
                if DEBUG: print(f"[TRAFFIC] Traffic-light detection stopped, restored to {orchestrator.get_state()} mode")

            await ui_broadcast_final("[System] Traffic-light detection stopped")
        except Exception as e:
            print(f"[TRAFFIC] Failed to stop traffic-light detection: {e}")
            await ui_broadcast_final(f"[System] Stop failed: {e}")
        return
    
    # Check for navigation commands — use orchestrator to control
    if any(k in user_text for k in ["开始导航", "盲道导航", "帮我导航",
                                     "start navigation", "help me navigate", "navigate me", "blind path"]):
        # If currently searching for an item, stop first
        if yolomedia_running:
            stop_yolomedia()
            print("[ITEM_SEARCH] Switching from item-search mode to blind-path navigation")

        if orchestrator:
            orchestrator.start_blind_path_navigation()
            if DEBUG: print(f"[NAVIGATION] Blind-path navigation started, state: {orchestrator.get_state()}")
            await ui_broadcast_final("[System] Blind-path navigation started")
        else:
            print("[NAVIGATION] Warning: navigation master not initialized!")
            await ui_broadcast_final("[System] Navigation system not ready")
        return
    
    if any(k in user_text for k in ["停止导航", "结束导航",
                                     "stop navigation", "end navigation"]):
        if orchestrator:
            orchestrator.stop_navigation()
            if DEBUG: print(f"[NAVIGATION] Navigation stopped, state: {orchestrator.get_state()}")
            await ui_broadcast_final("[System] Blind-path navigation stopped")
        else:
            await ui_broadcast_final("[System] Navigation system not running")
        return

    nav_cmd_keywords = [
        # Chinese
        "开始过马路", "过马路结束", "开始导航", "盲道导航", "停止导航", "结束导航",
        "立即通过", "现在通过", "继续",
        # English
        "start crossing", "stop crossing", "end crossing",
        "start navigation", "stop navigation", "end navigation",
        "pass now", "go now", "continue",
    ]
    if any(k in user_text for k in nav_cmd_keywords):
        if orchestrator:
            orchestrator.on_voice_command(user_text)
            await ui_broadcast_final("[System] Navigation mode updated")
        else:
            await ui_broadcast_final("[System] Navigation master not initialized")
        return    

    # Check for "帮我找/识别一下xxx" (help me find/identify xxx) command.
    # Try Chinese pattern first, then English ("find <item>" / "look for <item>").
    find_pattern_cn = r"(?:^\s*帮我)?\s*找一下\s*(.+?)(?:。|！|？|$)"
    find_pattern_en = r"\b(?:find|look\s+for)\s+(?:the\s+|a\s+|an\s+|my\s+)?(.+?)[\.\?\!]?\s*$"
    match = re.search(find_pattern_cn, user_text) or re.search(find_pattern_en, user_text)
        
    if match:
        # Extract the Chinese item name
        item_cn = match.group(1).strip()
        if item_cn:
            if extract_english_label is None:
                if DEBUG: print("[ITEM_SEARCH] extract_english_label unavailable (openai not installed), skipping", flush=True)
                await ui_broadcast_final("[System] Item search unavailable")
                return
            # Use local mapping + Qwen to extract the English class label
            label_en, src = extract_english_label(item_cn)
            if DEBUG: print(f"[COMMAND] Finder request: '{item_cn}' -> '{label_en}' (src={src})", flush=True)

            # Switch to item-search mode (pause navigation)
            if orchestrator:
                orchestrator.start_item_search()
                if DEBUG: print(f"[ITEM_SEARCH] Switched to item-search mode, state: {orchestrator.get_state()}")
            
            # Pass the English class label to yolomedia (it will auto-switch to YOLOE when the class is not found)
            start_yolomedia_with_target(label_en)

            # Send a confirmation feedback to the frontend / voice output
            try:
                await ui_broadcast_final(f"[Item Search] Searching for {item_cn}...")
            except Exception:
                pass

            return
    
    # Check for "found it" (找到了) command
    if "找到了" in user_text or "拿到了" in user_text:
        if DEBUG: print("[COMMAND] Found command detected", flush=True)
        # Stop the yolomedia worker
        stop_yolomedia()

        # Stop item-search mode and restore the previous navigation state
        if orchestrator:
            orchestrator.stop_item_search(restore_nav=True)
            current_state = orchestrator.get_state()
            if DEBUG: print(f"[ITEM_SEARCH] Item search ended, current state: {current_state}")
            
            # Give feedback based on the restored state
            if current_state in ["BLINDPATH_NAV", "SEEKING_CROSSWALK", "WAIT_TRAFFIC_LIGHT", "CROSSING", "SEEKING_NEXT_BLINDPATH"]:
                await ui_broadcast_final("[Item Search] Item found, resuming navigation.")
            else:
                await ui_broadcast_final("[Item Search] Item found.")
        else:
            await ui_broadcast_final("[Item Search] Item found.")
        
        return
    
    # When omni dialogue starts, switch to CHAT mode
    global omni_conversation_active, omni_previous_nav_state
    omni_conversation_active = True
    
    # Save the current navigation state and switch to CHAT mode
    if orchestrator:
        current_state = orchestrator.get_state()
        # Only save and switch when already in a navigation mode
        if current_state not in ["CHAT", "IDLE"]:
            omni_previous_nav_state = current_state
            orchestrator.force_state("CHAT")
            if DEBUG: print(f"[OMNI] Dialogue started, switching from {current_state} to CHAT mode")
        else:
            omni_previous_nav_state = None
            if DEBUG: print(f"[OMNI] Dialogue started (already in {current_state} mode)")
    
    # If not a special command, run the original AI dialogue logic
    # But if yolomedia is running, skip normal AI dialogue for now
    if yolomedia_running:
        if DEBUG: print("[AI] YOLO media is running, skipping normal AI response", flush=True)
        return True

    # Plain conversation — not handled here; caller decides what to do with it
    # (the live mic-audio path just uses this for its side effects since
    # Gemini already replied; the typed-PROMPT path forwards it to Gemini).
    return False

# ========= Typed-prompt entry point =========
# gemini_live is the only backend that touches audio at all — it produces
# its own spoken reply via _on_audio. gemini_regular/qwen are text-only for
# this comparison (see run_backend_turn above): no TTS, reply goes to the UI only.
async def start_ai_with_text(user_text: str):
    """Start new AI voice output after a hard reset."""
    async def _runner():
        txt_buf: List[str] = []
        rate_state = None

        # Assemble (image + text) content
        content_list = []
        if last_frames and (not STABILITY_MODE or is_explicit_vision_request(user_text)):
            try:
                _, jpeg_bytes = last_frames[-1]
                img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            except Exception:
                pass
        content_list.append({"type": "text", "text": user_text})

        try:
            async for piece in stream_chat(content_list, voice="Cherry", audio_format="wav"):
                # Text delta — update UI as text arrives
                if piece.text_delta:
                    txt_buf.append(piece.text_delta)
                    try:
                        await ui_broadcast_partial("[AI] " + "".join(txt_buf))
                    except Exception:
                        pass
                # piece.audio_b64 is always None from the local model;
                # audio is produced below via macOS 'say' after the full text is ready.
                #
                # [DASHSCOPE FALLBACK] streaming audio path:
                # if piece.audio_b64:
                #     pcm24 = base64.b64decode(piece.audio_b64)
                #     pcm8k, rate_state = audioop.ratecv(pcm24, 2, 1, 24000, 8000, rate_state)
                #     pcm8k = audioop.mul(pcm8k, 2, 0.60)
                #     if pcm8k:
                #         await broadcast_pcm16_realtime(pcm8k)

            # TTS: feed the complete AI response through the 8 kHz downlink
            full_text = "".join(txt_buf).strip()
            if full_text:
                try:
                    pcm8k = await _say_to_pcm8k(full_text)
                    if pcm8k:
                        # Primary path: enqueue raw mono-16 PCM for the paced
                        # /ws_audio sender. Firmware taskTTSPlay consumes qTTS
                        # and writes to i2sOut.
                        _ws = esp32_audio_ws
                        if _ws and _ws.client_state == WebSocketState.CONNECTED:
                            turn_id = _ensure_turn()
                            _CHUNK = 2040  # fits TTSChunk.data[2048] on the firmware side
                            for _i in range(0, len(pcm8k), _CHUNK):
                                _enqueue_tts_chunk(pcm8k[_i:_i + _CHUNK], turn_id)
                            latency_tracker.mark("turn_complete", turn_id)
                            _enqueue_tts_terminal("completed", turn_id)
                            print(f"[TTS-WS] queued {len(pcm8k)} bytes in {-(-len(pcm8k)//_CHUNK)} chunks", flush=True)
                        else:
                            print("[TTS-WS] esp32_audio_ws not connected — skipping WebSocket send", flush=True)
                        # Also broadcast via /stream.wav so browser clients can hear it
                        await broadcast_pcm16_realtime(pcm8k)
                except Exception as tts_err:
                    print(f"[TTS] say failed: {tts_err}", flush=True)

        except asyncio.CancelledError:
            # Interrupted by a new round
            raise
        except Exception as e:
            try:
                await ui_broadcast_final(f"[AI] Error occurred: {e}")
            except Exception:
                pass
        finally:
            # Mark omni dialogue as ended and restore the previous navigation mode
            global omni_conversation_active, omni_previous_nav_state
            omni_conversation_active = False
            
            # Restore the previous navigation state
            if orchestrator and omni_previous_nav_state:
                orchestrator.force_state(omni_previous_nav_state)
                if DEBUG: print(f"[OMNI] Dialogue ended, restored to {omni_previous_nav_state} mode")
                omni_previous_nav_state = None
            else:
                if DEBUG: print(f"[OMNI] Dialogue ended (no navigation state to restore)")
            
            # On natural completion, send a "finish" signal to the current connection
            from audio_stream import stream_clients  # local import to avoid circular dependency
            for sc in list(stream_clients):
                if not sc.abort_event.is_set():
                    try: sc.q.put_nowait(b"\x00"*BYTES_PER_20MS_16K)  # one frame of silence
                    except Exception: pass
                    try: sc.q.put_nowait(None)
                    except Exception: pass

            final_text = ("".join(txt_buf)).strip() or "(empty response)"
            print(f"[AI] {final_text}", flush=True)
            try:
                await ui_broadcast_final("[AI] " + final_text)
            except Exception:
                pass

    # Hard-reset before actually starting to guarantee absolutely no leftover audio
    await hard_reset_audio("start_ai_with_text")
    loop = asyncio.get_running_loop()
    from audio_stream import current_ai_task as _task_holder  # read/write module-level global
    from audio_stream import __dict__ as _as_dict
    # Set the module-level current_ai_task
    task = loop.create_task(_runner())
    _as_dict["current_ai_task"] = task

    # Clear the handle when the turn ends so a finished task can never be
    # mistaken for an in-progress one. Only clear if it is still *this* task,
    # so a newer turn started in the meantime is left untouched.
    def _clear_task(t: asyncio.Task) -> None:
        if _as_dict.get("current_ai_task") is t:
            _as_dict["current_ai_task"] = None
    task.add_done_callback(_clear_task)

# ---------- Page / Health ----------
@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/api/health")
def health():
    import resource
    now = time.monotonic()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    rgb = latest_rgb.snapshot()
    thermal = latest_thermal.snapshot()
    return JSONResponse({
        "ok": True,
        "stability_mode": STABILITY_MODE,
        "process_rss_bytes": int(rss),
        "sockets": {
            "camera_sensor": int(esp32_camera_ws is not None),
            "audio": int(esp32_audio_ws is not None),
        },
        "viewers": {"rgb": len(camera_viewers), "thermal": len(thermal_viewers), "imu": len(imu_ws_clients)},
        "latest_age_sec": {
            "rgb": None if rgb.data is None else max(0.0, now - rgb.timestamp),
            "thermal": None if thermal.data is None else max(0.0, now - thermal.timestamp),
            "imu": None if not latest_imu["timestamp"] else max(0.0, now - latest_imu["timestamp"]),
        },
        "latest_sequences": {
            "rgb": rgb.sequence,
            "thermal": thermal.sequence,
            "imu": latest_imu["sequence"],
        },
        "rgb_canonicalizer": rgb_canonicalizer_telemetry.health(),
        "device_status": latest_device_status,
        "recording": {
            **recording_pipeline.health(),
            "audio": recording_audio_pipeline.health(),
        },
        "vision": dict(vision_controller.metrics),
        "yolo": yolo_client.health(),
        "hands": hand_client.health(),
        "thermal_calibration": thermal_calibration_diagnostics(),
        "research": research_exporter.health(),
        "audio": {"last_activity": backend_metrics["last_audio_activity"]},
        "audio_freshness": audio_freshness_tracker.snapshot(),
        "connections": dict(backend_metrics),
    })


@app.get("/api/camera-freshness")
def camera_freshness():
    """Compact upstream state for browser-viewer freshness recovery."""
    now = time.monotonic()
    rgb = latest_rgb.snapshot()
    return JSONResponse({
        "camera_socket_connected": esp32_camera_ws is not None,
        "canonical_frame_age_ms": (
            None if rgb.data is None
            else round(max(0.0, now - rgb.timestamp) * 1000.0, 1)
        ),
        "canonical_frame_sequence": rgb.sequence,
    })


@app.get("/api/audio-freshness")
def audio_freshness():
    """Compact read-only socket/gate/PCM freshness state for the browser."""
    return JSONResponse(audio_freshness_tracker.snapshot())


@app.get("/latency/metrics")
def latency_metrics():
    return JSONResponse(latency_tracker.snapshot())


@app.get("/api/yolo/detections")
def yolo_detections():
    """Read-only Phase 2.1 shadow cache; never feeds Gemini or navigation."""
    return JSONResponse({
        "enabled": yolo_settings.enabled,
        **yolo_client.latest_detections(),
    })


@app.get("/api/perception/latest")
def latest_perception():
    """Read-only browser view of the latest YOLO shadow result."""
    return JSONResponse(yolo_client.latest_perception(display_rotation_deg=0))


@app.get("/api/perception/hands/latest")
def latest_hands():
    """Read-only hand cache in the canonical RGB source coordinate space."""
    return JSONResponse(hand_client.latest_hands())


class RecordingCommand(BaseModel):
    active: bool


@app.get("/api/recording")
def recording_status():
    return JSONResponse({
        **recording_pipeline.health(),
        "audio": recording_audio_pipeline.health(),
    })


@app.post("/api/recording")
async def recording_control(command: RecordingCommand):
    if command.active:
        recorder = sync_recorder.get_recorder()
        started = recorder.is_recording or await asyncio.to_thread(sync_recorder.start_recording)
        if not started:
            return JSONResponse({"error": "recorder_start_failed"}, status_code=500)
        recording_pipeline.start()
        recording_audio_pipeline.start()
    else:
        await recording_pipeline.stop()
        await recording_audio_pipeline.stop()
        await asyncio.to_thread(sync_recorder.stop_recording)
    return JSONResponse({
        **recording_pipeline.health(),
        "audio": recording_audio_pipeline.health(),
    })


class VisionRequest(BaseModel):
    reason: str = "authenticated-test"


@app.post("/api/vision")
async def vision_request(request: Request, payload: VisionRequest):
    expected = os.getenv("STABILITY_TEST_TOKEN", "")
    supplied = request.headers.get("x-stability-token", "")
    if not expected or supplied != expected:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    result = await request_gemini_vision(f"api:{payload.reason[:80]}")
    return JSONResponse(result, status_code=200 if result.get("ok") else 409)


class SessionStartRequest(BaseModel):
    label: str


@app.post("/api/session/start")
async def session_start(payload: SessionStartRequest):
    """Tag every turn recorded from now until /api/session/stop with a label
    (e.g. "far-room-walking") for manual test blocks. In-memory only, no auth —
    same trust boundary as the rest of this app's local-network control plane."""
    global _telemetry_session_id, _telemetry_session_label
    _telemetry_session_id = uuid.uuid4().hex
    _telemetry_session_label = payload.label
    return JSONResponse({
        "session_id": _telemetry_session_id,
        "session_label": _telemetry_session_label,
    })


@app.post("/api/session/stop")
async def session_stop():
    global _telemetry_session_id, _telemetry_session_label
    _telemetry_session_id = None
    _telemetry_session_label = None
    return JSONResponse({"stopped": True})


class ResearchSessionCommand(BaseModel):
    active: bool
    session_id: Optional[str] = None
    token: Optional[str] = None
    ttl_sec: Optional[float] = None


@app.post("/internal/research-session")
async def research_session_handshake(request: Request, command: ResearchSessionCommand):
    """Session-state handshake for the standalone research platform.

    The realtime app only ever RECEIVES session state here — it never polls
    the research platform. That direction of dependency is what keeps the
    realtime app's uptime independent of the research platform's uptime
    (§14 of the research-platform architecture plan): if the research
    platform disappears, it simply stops calling this endpoint, the durable
    SessionGate's expires_at lapses on its own (auto-expiry safety net, see
    research_exporter.py), and research_exporter.publish_event() quietly
    goes back to being a no-op. Nothing here is on any camera/audio/Gemini
    hot path.

    Auth is a separate shared secret from the per-session bearer token the
    exporter sends outbound (research_exporter.py's SessionGate.token) —
    this one only ever guards this inbound handshake.
    """
    expected = os.getenv("RESEARCH_SESSION_SHARED_SECRET", "")
    supplied = request.headers.get("x-research-shared-secret", "")
    if not expected or not hmac.compare_digest(supplied, expected):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if command.active:
        if not command.session_id or not command.token:
            return JSONResponse(
                {"error": "session_id and token are required to activate"},
                status_code=400,
            )
        research_session_gate.activate(command.session_id, command.token, ttl_sec=command.ttl_sec)
    else:
        research_session_gate.deactivate()
    return JSONResponse(research_session_gate.snapshot())


class ThermalDisplaySettings(BaseModel):
    palette: Optional[str] = None
    auto_range: Optional[bool] = None
    min_c: Optional[float] = None
    max_c: Optional[float] = None
    hotspot: Optional[bool] = None
    labels: Optional[bool] = None
    interpolation: Optional[str] = None
    calibration_offset_x: Optional[float] = None
    calibration_offset_y: Optional[float] = None
    calibration_scale_x: Optional[float] = None
    calibration_scale_y: Optional[float] = None


def thermal_calibration_diagnostics() -> dict:
    calibration = {
        key: thermal_display_config[key]
        for key in (
            "calibration_offset_x",
            "calibration_offset_y",
            "calibration_scale_x",
            "calibration_scale_y",
        )
    }
    state = {
        "canonical_orientation": (
            "portrait_32x24_rows_by_cols; native_24x32 rotated_90ccw_then_flipped_horizontal"
        ),
        **calibration,
        "hotspot_thermal_norm": None,
        "hotspot_rgb_norm_unclamped": None,
        "hotspot_rgb_norm_display": None,
        "mapping_limitations": "coarse_2d_only; parallax_and_depth_not_calibrated",
    }
    matrix = latest_thermal_matrix
    if matrix is None or matrix.shape != (32, 24) or not np.isfinite(matrix).all():
        return state
    row, col = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    thermal_norm = [(float(col) + 0.5) / matrix.shape[1], (float(row) + 0.5) / matrix.shape[0]]
    mapping_args = {
        "offset_x": calibration["calibration_offset_x"],
        "offset_y": calibration["calibration_offset_y"],
        "scale_x": calibration["calibration_scale_x"],
        "scale_y": calibration["calibration_scale_y"],
    }
    mapped = map_thermal_to_rgb_normalized(*thermal_norm, **mapping_args)
    display = map_thermal_to_rgb_normalized(
        *thermal_norm, **mapping_args, clamp_for_display=True
    )
    state.update({
        "hotspot_thermal_norm": [round(value, 6) for value in thermal_norm],
        "hotspot_rgb_norm_unclamped": [round(value, 6) for value in mapped],
        "hotspot_rgb_norm_display": [round(value, 6) for value in display],
    })
    return state


def thermal_display_state() -> dict:
    return {
        **thermal_display_config,
        "calibration_diagnostics": thermal_calibration_diagnostics(),
    }


@app.get("/api/thermal-display")
def get_thermal_display():
    return JSONResponse(thermal_display_state())


@app.post("/api/thermal-display")
def update_thermal_display(settings: ThermalDisplaySettings):
    values = settings.model_dump(exclude_none=True) if hasattr(settings, "model_dump") else settings.dict(exclude_none=True)
    if "palette" in values and values["palette"] not in {"inferno", "jet", "hot", "magma", "turbo", "bone"}:
        return JSONResponse({"error": "invalid palette"}, status_code=400)
    if "interpolation" in values and values["interpolation"] not in {"nearest", "cubic"}:
        return JSONResponse({"error": "invalid interpolation"}, status_code=400)
    next_min = float(values.get("min_c", thermal_display_config["min_c"]))
    next_max = float(values.get("max_c", thermal_display_config["max_c"]))
    if next_min >= next_max:
        return JSONResponse({"error": "min_c must be less than max_c"}, status_code=400)
    for key in ("calibration_offset_x", "calibration_offset_y"):
        if key in values:
            value = float(values[key])
            if not np.isfinite(value) or not -1.0 <= value <= 1.0:
                return JSONResponse({"error": f"{key} must be finite and between -1 and 1"}, status_code=400)
    for key in ("calibration_scale_x", "calibration_scale_y"):
        if key in values:
            value = float(values[key])
            if not np.isfinite(value) or not 0.1 <= value <= 3.0:
                return JSONResponse({"error": f"{key} must be finite and between 0.1 and 3"}, status_code=400)
    thermal_display_config.update(values)
    if any(key.startswith("calibration_") for key in values):
        calibration = thermal_calibration_diagnostics()
        print(
            "[THERMAL-CAL] "
            f"offset=({calibration['calibration_offset_x']},"
            f"{calibration['calibration_offset_y']}) "
            f"scale=({calibration['calibration_scale_x']},"
            f"{calibration['calibration_scale_y']}) "
            f"hotspot_thermal={calibration['hotspot_thermal_norm']} "
            f"hotspot_rgb={calibration['hotspot_rgb_norm_unclamped']}",
            flush=True,
        )
    return JSONResponse(thermal_display_state())
   


@app.get("/api/imu-validation")
def imu_validation(mode: str = "stationary"):
    allowed = {"stationary", "tilt_forward", "tilt_backward", "tilt_left", "tilt_right", "rotate"}
    if mode not in allowed:
        return JSONResponse({"error": "invalid mode", "allowed": sorted(allowed)}, status_code=400)
    age = None
    if latest_imu["timestamp"]:
        age = time.monotonic() - latest_imu["timestamp"]
    return JSONResponse({
        "mode": mode,
        "stale": age is None or age > 1.5,
        "age_sec": age,
        "sequence": latest_imu["sequence"],
        "sample": latest_imu["data"],
        "expected": {
            "stationary": "acceleration magnitude near 9.81 m/s^2; gyro near 0 deg/s",
            "tilt_forward": "one horizontal acceleration axis changes sign/magnitude consistently",
            "tilt_backward": "same axis as forward changes in the opposite direction",
            "tilt_left": "the other horizontal acceleration axis changes consistently",
            "tilt_right": "same axis as left changes in the opposite direction",
            "rotate": "at least one gyro axis shows a clear non-zero deg/s response",
        }[mode],
    })


@app.post("/api/restart")
async def restart_server():
    """Restart the Python server by re-executing the process (reloads all models).

    Responds first, then re-execs after a short delay so the HTTP reply flushes.
    """
    async def _do_restart():
        await asyncio.sleep(0.5)
        print("[SYSTEM] Restart requested via web UI — re-executing…", flush=True)
        try:
            cleanup_on_exit()  # save recordings; atexit won't fire across execv
        except Exception:
            pass
        os.execv(sys.executable, [sys.executable] + sys.argv)

    asyncio.create_task(_do_restart())
    return JSONResponse({"status": "restarting"})


class CommandPayload(BaseModel):
    text: str


@app.post("/api/command")
async def run_command(payload: CommandPayload):
    """Inject a command exactly as if it had been spoken (dev buttons on the page).

    Runs the same dispatcher as the voice/PROMPT path so every existing command
    phrase (detect objects, start navigation, start crossing, …) works via HTTP.
    """
    text = (payload.text or "").strip()
    if not text:
        return JSONResponse({"error": "empty command"}, status_code=400)
    if STABILITY_MODE:
        if is_explicit_vision_request(text):
            return JSONResponse(await request_gemini_vision(f"test-command:{text[:80]}"))
        return JSONResponse(
            {"error": "experimental commands disabled in STABILITY_MODE"},
            status_code=403,
        )
    async with interrupt_lock:
        await start_ai_with_text_custom(text)
    return JSONResponse({"ran": text})

class SettingsPayload(BaseModel):
    jpeg_quality: Optional[int] = None
    vad_silence_rms: Optional[int] = None
    vad_silence_ms: Optional[int] = None
    vad_min_speech_ms: Optional[int] = None
    camera_rotation_deg: Optional[int] = None

@app.get("/api/settings")
def get_settings():
    return JSONResponse({
        "jpeg_quality": JPEG_QUALITY,
        "vad_silence_rms": VAD_SILENCE_RMS,
        "vad_silence_ms": VAD_SILENCE_MS,
        "vad_min_speech_ms": VAD_MIN_SPEECH_MS,
        "camera_rotation_deg": CAMERA_ROTATION_DEG,
    })

@app.post("/api/settings")
def update_settings(payload: SettingsPayload):
    global JPEG_QUALITY, VAD_SILENCE_RMS, VAD_SILENCE_MS, VAD_MIN_SPEECH_MS
    global VAD_SILENCE_CHUNKS, VAD_MIN_SPEECH_CHUNKS
    if payload.jpeg_quality is not None:
        JPEG_QUALITY = max(1, min(100, payload.jpeg_quality))
    if payload.camera_rotation_deg not in (None, 0):
        return JSONResponse(
            {"error": "camera rotation is fixed at canonical 0 degrees"},
            status_code=400,
        )
    if payload.vad_silence_rms is not None:
        VAD_SILENCE_RMS = max(50, min(5000, payload.vad_silence_rms))
    if payload.vad_silence_ms is not None:
        VAD_SILENCE_MS = max(200, min(3000, payload.vad_silence_ms))
        VAD_SILENCE_CHUNKS = VAD_SILENCE_MS // _VAD_CHUNK_MS
    if payload.vad_min_speech_ms is not None:
        VAD_MIN_SPEECH_MS = max(100, min(2000, payload.vad_min_speech_ms))
        VAD_MIN_SPEECH_CHUNKS = VAD_MIN_SPEECH_MS // _VAD_CHUNK_MS
    return JSONResponse({
        "jpeg_quality": JPEG_QUALITY,
        "vad_silence_rms": VAD_SILENCE_RMS,
        "vad_silence_ms": VAD_SILENCE_MS,
        "vad_min_speech_ms": VAD_MIN_SPEECH_MS,
        "camera_rotation_deg": CAMERA_ROTATION_DEG,
    })

class CameraCommand(BaseModel):
    framesize: Optional[str] = None
    quality: Optional[int] = None
    fps: Optional[int] = None
    exposure_auto: Optional[bool] = None
    exposure_value: Optional[int] = None
    gain_ceiling: Optional[int] = None   # 0=2X .. 6=128X
    aec2: Optional[bool] = None          # extended AEC (night mode)
    ae_level: Optional[int] = None       # AE target bias, -2..+2

@app.post("/api/camera")
async def camera_command(cmd: CameraCommand):
    if STABILITY_MODE:
        return JSONResponse(
            {
                "error": "camera hardware settings are fixed in STABILITY_MODE",
                "active": {"framesize": "QVGA", "quality": 16, "fps": 4, "fb_count": 2},
            },
            status_code=409,
        )
    if esp32_camera_ws is None:
        return JSONResponse({"error": "ESP32 camera not connected"}, status_code=503)
    sent = []
    try:
        if cmd.framesize:
            v = cmd.framesize.upper()
            if v in ("VGA", "SVGA", "XGA"):
                await esp32_camera_ws.send_text(f"SET:FRAMESIZE={v}")
                sent.append(f"FRAMESIZE={v}")
        if cmd.quality is not None:
            q = max(5, min(40, cmd.quality))
            await esp32_camera_ws.send_text(f"SET:QUALITY={q}")
            sent.append(f"QUALITY={q}")
        if cmd.fps is not None:
            f = max(0, min(60, cmd.fps))
            await esp32_camera_ws.send_text(f"SET:FPS={f}")
            sent.append(f"FPS={f}")
        if cmd.exposure_auto is not None:
            await esp32_camera_ws.send_text(f"SET:AE_AUTO={1 if cmd.exposure_auto else 0}")
            sent.append(f"AE_AUTO={1 if cmd.exposure_auto else 0}")
        if cmd.exposure_value is not None:
            v = max(0, min(1200, cmd.exposure_value))
            await esp32_camera_ws.send_text(f"SET:AEC={v}")
            sent.append(f"AEC={v}")
        if cmd.gain_ceiling is not None:
            v = max(0, min(6, cmd.gain_ceiling))
            await esp32_camera_ws.send_text(f"SET:GAINCEIL={v}")
            sent.append(f"GAINCEIL={v}")
        if cmd.aec2 is not None:
            await esp32_camera_ws.send_text(f"SET:AEC2={1 if cmd.aec2 else 0}")
            sent.append(f"AEC2={1 if cmd.aec2 else 0}")
        if cmd.ae_level is not None:
            v = max(-2, min(2, cmd.ae_level))
            await esp32_camera_ws.send_text(f"SET:AE_LEVEL={v}")
            sent.append(f"AE_LEVEL={v}")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"sent": sent})

# Register /stream.wav route
register_stream_route(app)

# ---------- WebSocket: WebUI text (ASR/AI status push) ----------
@app.websocket("/ws_ui")
async def ws_ui(ws: WebSocket):
    await ws.accept()
    ui_clients[id(ws)] = ws
    try:
        init = {"partial": current_partial, "finals": recent_finals[-10:]}
        await ws.send_text("INIT:" + json.dumps(init, ensure_ascii=False))
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        ui_clients.pop(id(ws), None)

# ---------- Whisper + RMS-VAD dispatch (gemini_regular / qwen only) ----------
# This is the same local-ASR flow the app used before the Live migration —
# buffer PCM16, transcribe on detected silence, dispatch the text. Only used
# when AI_BACKEND != "gemini_live" (see the ASR section near the top of this
# file for why: gemini-3.1-flash-live-preview can't be reused as a TEXT-only
# ASR engine, so these two backends need their own ASR again).
async def _run_whisper_and_dispatch(buf: bytes) -> None:
    if not buf or _whisper_model is None:
        return
    if _turn_busy:
        print("[WHISPER] Skipped — previous turn still in progress", flush=True)
        return
    _turn_busy = True
    speech_end_ts = time.time()  # ~when VAD detected end of speech and called this
    try:
        # PCM16 int16 → float32 normalised to [-1, 1] at 16 kHz
        samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
        loop    = asyncio.get_running_loop()
        result  = await loop.run_in_executor(
            None,
            lambda: _whisper_model.transcribe(samples, language=_WHISPER_CFG.language, fp16=False)
        )
        # ASR time is folded into the single total_latency measurement
        # logged later in run_backend_turn(), not reported separately.
        text = (result.get("text") or "").strip()
        print(f"[WHISPER] {text}", flush=True)

        if text:
            await ui_broadcast_final(text)

            if _has_hotword(text):
                async with interrupt_lock:
                    print(f"[HOTWORD] '{text}' → full reset", flush=True)
                    await full_system_reset("Hotword interrupt")
            elif not is_playing_now():
                async with interrupt_lock:
                    handled = await start_ai_with_text_custom(text)
                    if not handled:
                        await run_backend_turn(text)
    except Exception as e:
        print(f"[WHISPER] transcribe error: {e}", flush=True)
    finally:
        # This was missing — without it, _turn_busy stayed True forever
        # after the very first turn, permanently locking out every
        # subsequent attempt ("Skipped — previous turn still in progress").
        _turn_busy = False


# ---------- WebSocket: ESP32 audio entry (ASR uplink) ----------
#
# Changes vs original:
#   - DashScope dash_audio.asr.Recognition, keepalive_loop, ASRCallback all removed.
#   - New flow: START → buffer PCM16, audio frames → extend buffer,
#     STOP → transcribe with Whisper, hotword check, then call start_ai_with_text_custom.
#
# [DASHSCOPE FALLBACK] The original recognition-based flow was:
#   cb = ASRCallback(on_sdk_error=…, post=…, ui_broadcast_partial=…, …)
#   recognition = dash_audio.asr.Recognition(api_key=API_KEY, model=MODEL,
#       format=AUDIO_FMT, sample_rate=SAMPLE_RATE, callback=cb)
#   recognition.start(); await set_current_recognition(recognition)
#   # On each audio frame: recognition.send_audio_frame(msg["bytes"])
#   # keepalive_loop fed silence when idle > 350 ms
#   # On STOP: recognition.send_audio_frame(SILENCE_20MS) x15, then recognition.stop()
#
@app.websocket("/ws_audio")
async def ws_audio(ws: WebSocket):
    global esp32_audio_ws, _esp32_audio_send_lock, mic_streaming
    # Evict-and-replace rather than reject: unlike ws_camera_esp, a stale
    # mic connection blocks the VAD/dispatch pipeline, so a reconnect needs
    # to fail over immediately instead of waiting on ping-timeout detection
    # to notice the old one is dead.
    old_ws = esp32_audio_ws
    if old_ws is not None:
        print("[MIC] New connection superseding previous one")
        try:
            await old_ws.close(code=1001)
        except Exception:
            pass
    esp32_audio_ws = ws
    await ws.accept()
    _esp32_audio_send_lock = asyncio.Lock()
    connection_generation = audio_freshness_tracker.connect()
    device_id = _device_id(ws)
    connected_at = time.monotonic()
    backend_metrics["audio_connects"] += 1
    print(
        f"[WS-INGEST] device={device_id} socket=audio event=connect "
        f"generation={connection_generation}",
        flush=True,
    )

    audio_ingest_q: asyncio.Queue[bytes] = asyncio.Queue(
        maxsize=ESP_AUDIO_INGEST_QUEUE_MAX
    )
    audio_dropped = 0
    audio_dropped_interval = 0
    audio_queue_high_water = 0
    audio_last_receive_at: Optional[float] = None
    audio_receive_count_interval = 0
    audio_pcm_chunk_count_interval = 0
    audio_receive_gap_max_ms = 0.0
    audio_receive_gap_buckets = [
        0 for _ in range(len(ESP_AUDIO_GAP_BUCKET_LIMITS_MS) + 1)
    ]
    audio_health_started_at = time.monotonic()

    def _record_audio_receive(now: float, byte_count: int) -> None:
        nonlocal audio_last_receive_at
        nonlocal audio_receive_count_interval, audio_pcm_chunk_count_interval
        nonlocal audio_receive_gap_max_ms
        if audio_last_receive_at is not None:
            gap_ms = (now - audio_last_receive_at) * 1000
            audio_receive_gap_max_ms = max(audio_receive_gap_max_ms, gap_ms)
            bucket = len(ESP_AUDIO_GAP_BUCKET_LIMITS_MS)
            for index, limit_ms in enumerate(ESP_AUDIO_GAP_BUCKET_LIMITS_MS):
                if gap_ms < limit_ms:
                    bucket = index
                    break
            audio_receive_gap_buckets[bucket] += 1
        audio_last_receive_at = now
        audio_receive_count_interval += 1
        audio_pcm_chunk_count_interval += max(
            1,
            (byte_count + PCM_20MS_BYTES_16K_MONO - 1)
            // PCM_20MS_BYTES_16K_MONO,
        )

    def _emit_audio_health(*, force: bool = False) -> None:
        nonlocal audio_dropped_interval, audio_queue_high_water
        nonlocal audio_receive_count_interval, audio_pcm_chunk_count_interval
        nonlocal audio_receive_gap_max_ms
        nonlocal audio_receive_gap_buckets, audio_health_started_at
        now = time.monotonic()
        interval_sec = now - audio_health_started_at
        if not force and interval_sec < ESP_AUDIO_HEALTH_INTERVAL_SEC:
            return
        print(
            f"[AUDIO-HEALTH] device={device_id} interval_s={interval_sec:.1f} "
            f"received_messages={audio_receive_count_interval} "
            f"received_pcm_chunks={audio_pcm_chunk_count_interval} "
            f"receive_gap_max_ms={audio_receive_gap_max_ms:.1f} "
            "receive_gap_limits_ms=30/50/100/250/500/1000 "
            f"receive_gap_buckets={'/'.join(map(str, audio_receive_gap_buckets))} "
            f"queue_high_water={audio_queue_high_water} "
            f"dropped_delta={audio_dropped_interval} dropped_total={audio_dropped}",
            flush=True,
        )
        audio_dropped_interval = 0
        audio_queue_high_water = audio_ingest_q.qsize()
        audio_receive_count_interval = 0
        audio_pcm_chunk_count_interval = 0
        audio_receive_gap_max_ms = 0.0
        audio_receive_gap_buckets = [
            0 for _ in range(len(ESP_AUDIO_GAP_BUCKET_LIMITS_MS) + 1)
        ]
        audio_health_started_at = now

    async def _gemini_audio_worker():
        while True:
            chunk = await audio_ingest_q.get()
            started = time.monotonic()
            try:
                await gemini_live.send_audio(chunk)
            except Exception as exc:
                print(
                    f"[WS-INGEST] device={device_id} socket=audio "
                    f"event=process_error error={type(exc).__name__}",
                    flush=True,
                )
            finally:
                latency_ms = (time.monotonic() - started) * 1000
                if latency_ms >= 250:
                    print(
                        f"[WS-INGEST] device={device_id} socket=audio "
                        f"event=processed bytes={len(chunk)} "
                        f"queue={audio_ingest_q.qsize()} latency_ms={latency_ms:.1f}",
                        flush=True,
                    )

    audio_worker_task = (
        asyncio.create_task(_gemini_audio_worker())
        if AI_BACKEND == "gemini_live"
        else None
    )

    streaming: bool = False
    pcm_buffer: Optional[bytearray] = None
    # VAD state (reset on every START / STOP / auto-trigger)
    vad_silent_chunks: int   = 0      # consecutive silent 20ms chunks this utterance
    vad_speech_chunks: int   = 0      # speech chunks accumulated this utterance
    vad_speech_detected: bool = False  # True once VAD_MIN_SPEECH_CHUNKS of speech seen

    try:
        while True:
            if WebSocketState and ws.client_state != WebSocketState.CONNECTED:
                break
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                if "Cannot call \"receive\"" in str(e):
                    break
                raise

            if "text" in msg and msg["text"] is not None:
                raw = (msg["text"] or "").strip()
                cmd = raw.upper()

                mic_loss = parse_mic_loss_message(raw)
                if mic_loss is not None:
                    audio_freshness_tracker.record_loss(mic_loss)
                    latency_tracker.mark_audio_integrity_degraded(mic_loss)
                    print(
                        f"[MIC-LOSS] device={device_id} "
                        f"generation={connection_generation} "
                        f"epoch={mic_loss['epoch']} chunks={mic_loss['chunks']} "
                        f"duration_ms={mic_loss['duration_ms']} "
                        f"reason={mic_loss['reason']} "
                        f"turn_id={latency_tracker.active_turn_id or 0}",
                        flush=True,
                    )
                    continue

                if cmd == "MIC_GATE:TTS":
                    audio_freshness_tracker.set_gate(
                        expected_streaming=False, reason="tts"
                    )
                    continue
                if cmd == "MIC_GATE:OPEN":
                    audio_freshness_tracker.set_gate(
                        expected_streaming=True, reason="streaming"
                    )
                    continue
                if cmd == "BARGE_IN":
                    print(
                        f"[BARGE-IN] device={device_id} event=received "
                        f"generation={connection_generation}",
                        flush=True,
                    )
                    audio_freshness_tracker.set_gate(
                        expected_streaming=True, reason="barge_in"
                    )
                    await _on_interrupted()
                    continue

                if raw.startswith("LATENCY:PING:"):
                    parts = raw.split(":")
                    if len(parts) == 4 and parts[2].isdigit() and parts[3].isdigit():
                        # Echo the device timestamp unchanged. The ESP32
                        # computes RTT only with esp_timer_get_time().
                        await _send_esp32_audio_text(
                            ws,
                            f"LATENCY:PONG:{parts[2]}:{parts[3]}"
                        )
                    continue

                if raw.startswith("LATENCY:RTT:"):
                    parts = raw.split(":")
                    if len(parts) == 7:
                        try:
                            values_ms = [int(value) / 1000.0 for value in parts[2:]]
                            latency_tracker.update_rtt(*values_ms)
                        except ValueError:
                            pass
                    continue

                if raw.startswith("LATENCY:DEVICE:"):
                    parts = raw.split(":")
                    if len(parts) == 4:
                        try:
                            latency_tracker.update_device_latency(
                                int(parts[2]), float(parts[3])
                            )
                        except ValueError:
                            pass
                    continue

                if raw.startswith("SPEECH_START:"):
                    try:
                        turn_id = int(raw.split(":", 1)[1])
                    except ValueError:
                        continue
                    active_turn_id = latency_tracker.active_turn_id
                    if active_turn_id is not None and active_turn_id != turn_id:
                        latency_tracker.mark("interrupted", active_turn_id)
                        await _finalize_latency_turn("interrupted", active_turn_id)
                    if active_turn_id != turn_id:
                        _clear_gemini_turn_state("speech_start")
                    _start_turn(turn_id)
                    continue

                if raw.startswith("SPEECH_END:"):
                    try:
                        turn_id = int(raw.split(":", 1)[1])
                    except ValueError:
                        continue
                    if latency_tracker.active_turn_id is None:
                        _start_turn(turn_id)
                    latency_tracker.mark("speech_end_detected", turn_id)
                    continue

                if cmd == "START":
                    print("[MIC] Listening — waiting for speech...")
                    streaming            = True
                    mic_streaming         = AI_BACKEND == "gemini_live"
                    pcm_buffer           = bytearray()
                    vad_silent_chunks    = 0
                    vad_speech_chunks    = 0
                    vad_speech_detected  = False
                    audio_freshness_tracker.set_gate(
                        expected_streaming=True, reason="streaming"
                    )
                    await ui_broadcast_partial("（Recording…）")
                    await _send_esp32_audio_text(ws, "OK:STARTED")

                elif cmd == "STOP":
                    streaming = False
                    if AI_BACKEND == "gemini_live":
                        print("[MIC] Stopped streaming")
                        mic_streaming = False
                        await _send_esp32_audio_text(ws, "OK:STOPPED")
                    else:
                        print("[MIC] Transcribing...")
                        buf = bytes(pcm_buffer) if pcm_buffer else b""
                        pcm_buffer          = None
                        vad_silent_chunks   = 0
                        vad_speech_chunks   = 0
                        vad_speech_detected = False
                        await _send_esp32_audio_text(ws, "OK:STOPPED")
                        await _run_whisper_and_dispatch(buf)

                elif raw.startswith("PROMPT:"):
                    # Device-initiated prompt (bypasses ASR entirely)
                    text = raw[len("PROMPT:"):].strip()
                    if text:
                        async with interrupt_lock:
                            await start_ai_with_text_custom(text)
                        await _send_esp32_audio_text(ws, "OK:PROMPT_ACCEPTED")
                    else:
                        await _send_esp32_audio_text(ws, "ERR:EMPTY_PROMPT")

            elif "bytes" in msg and msg["bytes"] is not None:
                chunk = msg["bytes"]
                if not chunk or len(chunk) % 2:
                    print(
                        f"[WS-INGEST] device={device_id} socket=audio "
                        f"event=invalid_pcm bytes={len(chunk)}",
                        flush=True,
                    )
                    continue
                audio_receive_now = time.monotonic()
                _record_audio_receive(audio_receive_now, len(chunk))
                audio_freshness_tracker.record_pcm(len(chunk))
                latency_tracker.mark_microphone_chunk(
                    now_ns=time.monotonic_ns()
                )
                backend_metrics["last_audio_activity"] = audio_receive_now

                if AI_BACKEND == "gemini_live":
                    # Gemini Live owns the mic entirely in this mode — it runs
                    # its own server-side VAD/turn-detection on the raw stream,
                    # so the local RMS-VAD/Whisper pipeline below must stay out
                    # of the way (it would otherwise fire its own transcription
                    # off the same audio and double-dispatch commands).
                    if streaming and not is_playing_now():
                        # Firmware may aggregate two 20 ms chunks into one
                        # WebSocket frame. Normalize at ingress so queue age,
                        # Gemini pacing, and drop counters retain 20 ms units.
                        for offset in range(0, len(chunk), PCM_20MS_BYTES_16K_MONO):
                            pcm_chunk = chunk[offset:offset + PCM_20MS_BYTES_16K_MONO]
                            if audio_ingest_q.full():
                                try:
                                    audio_ingest_q.get_nowait()
                                    audio_dropped += 1
                                    audio_dropped_interval += 1
                                except asyncio.QueueEmpty:
                                    pass
                            try:
                                audio_ingest_q.put_nowait(pcm_chunk)
                            except asyncio.QueueFull:
                                audio_dropped += 1
                                audio_dropped_interval += 1
                            audio_queue_high_water = max(
                                audio_queue_high_water, audio_ingest_q.qsize()
                            )
                        if audio_dropped and audio_dropped % 50 == 1:
                            print(
                                f"[WS-INGEST] device={device_id} socket=audio "
                                f"event=drop_oldest bytes={len(chunk)} "
                                f"queue={audio_ingest_q.qsize()} dropped={audio_dropped}",
                                flush=True,
                            )
                    _emit_audio_health()
                    continue

                if streaming and pcm_buffer is not None:
                    # Mute the mic while the AI is speaking. Without this the
                    # glasses' own TTS echoes back into the mic, gets VAD-segmented
                    # and can launch a bogus turn — which then blocks the user's
                    # next real question (the "ask twice" symptom). Drop the frame
                    # and reset VAD so nothing accumulates during playback.
                    if is_playing_now():
                        pcm_buffer          = bytearray()
                        vad_silent_chunks   = 0
                        vad_speech_chunks   = 0
                        vad_speech_detected = False
                        continue
                    pcm_buffer.extend(chunk)

                n = len(chunk)
                if n >= 2:
                    s = np.frombuffer(chunk[: n & ~1], dtype=np.int16).astype(np.float32)
                    s -= s.mean()  # strip DC offset from PDM mic before measuring energy
                    rms = float(np.sqrt(np.mean(s ** 2)))
                else:
                    rms = 0.0

                    # [VAD DEBUG] Log every chunk so we can read the real noise floor.
                    # Set DEBUG_VAD = False once VAD_SILENCE_RMS is tuned.
                    if DEBUG_VAD:
                        label = "SPEECH" if rms >= VAD_SILENCE_RMS else "silent"
                        print(
                            f"[VAD DEBUG] rms={rms:6.0f}  thresh={VAD_SILENCE_RMS}"
                            f"  → {label}"
                            f"  speech_chunks={vad_speech_chunks}"
                            f"  silent_chunks={vad_silent_chunks}"
                            f"  detected={vad_speech_detected}",
                            flush=True,
                        )

                if rms >= VAD_SILENCE_RMS:
                    vad_silent_chunks  = 0
                    vad_speech_chunks += max(
                        1, len(chunk) // PCM_20MS_BYTES_16K_MONO
                    )
                    if not vad_speech_detected and vad_speech_chunks >= VAD_MIN_SPEECH_CHUNKS:
                        vad_speech_detected = True
                        print("[MIC] Speech detected", flush=True)
                else:
                    if vad_speech_detected:
                        vad_silent_chunks += max(
                            1, len(chunk) // PCM_20MS_BYTES_16K_MONO
                        )
                        if vad_silent_chunks >= VAD_SILENCE_CHUNKS:
                            print("[MIC] Transcribing...", flush=True)
                            buf = bytes(pcm_buffer)
                            pcm_buffer          = bytearray()
                            vad_silent_chunks   = 0
                            vad_speech_chunks   = 0
                            vad_speech_detected = False
                            await ui_broadcast_partial("（Processing…）")
                            await _run_whisper_and_dispatch(buf)
                    else:
                        vad_speech_chunks = 0

    except Exception as e:
        print(f"\n[WS ERROR] {e}")
    finally:
        streaming  = False
        mic_streaming = False
        pcm_buffer = None
        _emit_audio_health(force=True)
        if audio_worker_task is not None:
            audio_worker_task.cancel()
            try:
                await audio_worker_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        if esp32_audio_ws is ws:
            _clear_tts_send_queue()
            _enqueue_tts_terminal("interrupted")
            esp32_audio_ws = None
            _esp32_audio_send_lock = None
            audio_freshness_tracker.disconnect()
        backend_metrics["audio_disconnects"] += 1
        print(
            f"[WS-INGEST] device={device_id} socket=audio event=disconnect "
            f"reason=client_or_receive_end connected_s={time.monotonic() - connected_at:.1f} "
            f"dropped={audio_dropped}",
            flush=True,
        )

def _process_camera_frame_blocking(data: bytes):
    """Decode one JPEG frame and run detection/navigation on it.

    Pure synchronous CPU work — safe to run in a thread executor. Does NO
    websocket or async I/O. Returns (out_jpeg_bytes | None, guidance_text | None);
    the caller broadcasts the JPEG and speaks the guidance from the event loop.
    """
    if STABILITY_MODE:
        return data, None
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            return (None, None)
    except Exception:
        return (None, None)

    # General object detection takes over the frame while active. Checked before
    # the orchestrator so it works even when nav models aren't loaded (laptop).
    if general_detect_active and not yolomedia_running:
        try:
            annotated, _counts = general_detector.detect(
                bgr, conf=float(os.getenv("GENERAL_DET_CONF", "0.25"))
            )
            out_img = annotated if annotated is not None else bgr
        except Exception as e:
            if DEBUG:
                print(f"[GENERAL_DET] error: {e}")
            out_img = bgr
        # No spoken/text guidance — boxes + labels are drawn on the frame itself.
        ok, enc = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        return (enc.tobytes() if ok else None, None)

    # Orchestrator active and item-search not occupying the frame
    if orchestrator and not yolomedia_running:
        current_state = orchestrator.get_state()

        # Item-search: yolomedia owns the stream; show raw until it starts sending
        if current_state == "ITEM_SEARCH":
            if not yolomedia_sending_frames:
                ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                return (enc.tobytes() if ok else None, None)
            return (None, None)

        out_img = bgr
        guidance = None
        try:
            if current_state == "TRAFFIC_LIGHT_DETECTION":
                import trafficlight_detection
                result = trafficlight_detection.process_single_frame(bgr)
                out_img = result['vis_image'] if result['vis_image'] is not None else bgr
            else:
                res = orchestrator.process_frame(bgr)
                guidance = res.guidance_text
                out_img = res.annotated_image if res.annotated_image is not None else bgr
        except Exception:
            pass
        ok, enc = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        return (enc.tobytes() if ok else None, guidance)

    # Fallback: no orchestrator, or yolomedia running. Passthrough raw unless
    # yolomedia is already sending its own annotated frames.
    if not yolomedia_sending_frames:
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        return (enc.tobytes() if ok else None, None)
    return (None, None)


# 1-byte type prefix used on the merged /ws/camera_thermal socket (see
# ws_camera_thermal_esp below) — must match MSG_TYPE_CAM/MSG_TYPE_THERMAL in
# compile.ino exactly.
async def _handle_camera_frame(data: bytes, frame_counter: int, holder: dict, frame_event: asyncio.Event,
                                gemini_frame_holder: dict, gemini_frame_event: asyncio.Event,
                                gemini_pump_task, received_at: Optional[float] = None) -> int:
    """Fan out one already-canonical RGB frame to every main consumer.

    Shared verbatim by /ws/camera
    (ws_camera_esp) and the merged /ws/camera_thermal (ws_camera_thermal_esp)
    so both stay on identical processing logic."""
    frame_counter += 1
    latest_rgb.update(data, received_at)
    recording_pipeline.enqueue_latest(data)

    # Cheap per-frame bookkeeping stays here; recording runs in the worker.
    try:
        last_frames.append((time.time(), data))
    except Exception:
        pass
    if not STABILITY_MODE:
        bridge_io.push_raw_jpeg(data)

    # Hand the newest frame to the processor. If an older unprocessed
    # frame is still sitting here, it's overwritten (dropped).
    holder["data"] = data
    frame_event.set()

    global _last_gemini_video_submit
    if gemini_pump_task is not None and mic_streaming:
        now = time.monotonic()
        if now - _last_gemini_video_submit >= GEMINI_VIDEO_INTERVAL_SEC:
            _last_gemini_video_submit = now
            gemini_frame_holder["data"] = data
            gemini_frame_event.set()

    global _last_research_frame_sample
    now_mono = time.monotonic()
    if now_mono - _last_research_frame_sample >= RESEARCH_FRAME_SAMPLE_INTERVAL_SEC:
        _last_research_frame_sample = now_mono
        research_exporter.publish_event(
            "CAMERA_FRAME_SAMPLE", {"frame_sequence": frame_counter}
        )

    return frame_counter


def _retain_latest_thermal(data: bytes, received_at: Optional[float] = None) -> np.ndarray:
    """Canonicalize once, then retain the shared 32x24 thermal matrix."""
    global latest_thermal_matrix
    canonical = canonicalize_thermal_payload(data)
    if canonical is None:
        raise ValueError("invalid thermal payload")
    latest_thermal_matrix = canonical
    latest_thermal.update(canonical.astype("<f4", copy=False).tobytes(), received_at)
    return latest_thermal_matrix


async def _handle_thermal_frame(data: bytes, thermal_frame_count: int) -> int:
    """Per-frame thermal ingest: parse the 24x32 float32 grid, colorize,
    broadcast to thermal_viewers. Shared verbatim by /ws/thermal
    (ws_thermal_esp) and the merged /ws/camera_thermal
    (ws_camera_thermal_esp) so both stay on identical processing logic.
    `data` must already have any transport-specific prefix stripped — exactly
    3072 bytes (24*32 float32)."""
    if len(data) != 3072:
        return thermal_frame_count
    frame = _retain_latest_thermal(data)
    thermal_frame_count += 1
    if thermal_frame_count == 1 or thermal_frame_count % 20 == 0:
        print(f"[THERMAL] received {thermal_frame_count} frames, "
              f"forwarding to {len(thermal_viewers)} viewer(s)", flush=True)

    colorized = _colorize_thermal(frame)
    ok, enc = cv2.imencode(".jpg", colorized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if ok and thermal_viewers:
        jpeg_bytes = enc.tobytes()
        stats = json.dumps({"max": float(frame.max()), "min": float(frame.min())})
        dead = []
        for viewer_ws in list(thermal_viewers):
            try:
                await viewer_ws.send_bytes(jpeg_bytes)
                await viewer_ws.send_text(stats)
            except Exception:
                dead.append(viewer_ws)
        for d in dead:
            thermal_viewers.discard(d)
    return thermal_frame_count


def _prepare_thermal_frame_blocking(frame: np.ndarray):
    """CPU-only colorize/JPEG encode for a canonical thermal matrix."""
    if frame.shape != (32, 24) or not np.isfinite(frame).all():
        return None, None
    colorized = _colorize_thermal(frame)
    ok, enc = cv2.imencode(".jpg", colorized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None, None
    stats = json.dumps({"max": float(frame.max()), "min": float(frame.min())})
    return enc.tobytes(), stats


# ---------- WebSocket: ESP32 camera entry (JPEG binary) ----------
@app.websocket("/ws/camera")
async def ws_camera_esp(ws: WebSocket):
    if STABILITY_MODE:
        await ws.close(code=1008, reason="legacy camera endpoint disabled in STABILITY_MODE")
        return
    global esp32_camera_ws, blind_path_navigator, cross_street_navigator, cross_street_active, navigation_active, orchestrator
    if esp32_camera_ws is not None:
        await ws.close(code=1013)
        return
    esp32_camera_ws = ws
    await ws.accept()
    print("[CONNECTED] Camera (ESP32)")

    # Initialize the blind-path navigator
    if blind_path_navigator is None and yolo_seg_model is not None:
        blind_path_navigator = BlindPathNavigator(yolo_seg_model, obstacle_detector)
        if DEBUG: print("[NAVIGATION] Blind-path navigator initialized")
    else:
        if blind_path_navigator is None and yolo_seg_model is None:
            print("[NAVIGATION] Warning: YOLO model not loaded, cannot initialize navigator")

    # Initialize the street-crossing navigator
    if cross_street_navigator is None:
        if yolo_seg_model:
            cross_street_navigator = CrossStreetNavigator(
                seg_model=yolo_seg_model,
                coco_model=None,  # traffic-light detection disabled
                obs_model=None    # obstacle detection also disabled for now (faster)
            )
            if DEBUG: print("[CROSS_STREET] Street-crossing navigator initialized")
        else:
            print("[CROSS_STREET] Error: segmentation model missing, cannot initialize street-crossing navigator")

    if orchestrator is None and blind_path_navigator is not None and cross_street_navigator is not None:
        orchestrator = NavigationMaster(blind_path_navigator, cross_street_navigator)
        if DEBUG: print("[NAV MASTER] Master state machine initialized")
    loop = asyncio.get_running_loop()
    frame_counter = 0

    # ---- Drop-to-latest pipeline ----------------------------------------
    # The receive loop below drains the socket and keeps only the NEWEST frame.
    # A separate processor task runs YOLO/nav on that newest frame in a thread
    # (so it never blocks the receive loop) and broadcasts the annotated result.
    # Frames that arrive while the processor is busy are overwritten (dropped),
    # so detection always runs on the freshest frame instead of a growing
    # backlog — this is what kills the video/detection lag.
    holder = {"data": None}
    frame_event = asyncio.Event()

    async def _processor():
        while True:
            await frame_event.wait()
            frame_event.clear()
            data = holder["data"]
            if data is None:
                continue
            try:
                out_jpeg, guidance = await loop.run_in_executor(
                    None, _process_camera_frame_blocking, data
                )
            except Exception as e:
                out_jpeg, guidance = None, None
                if DEBUG:
                    print(f"[NAV MASTER] processor error: {e}")

            # Speak/broadcast navigation guidance from the event loop
            if guidance:
                try:
                    if AI_BACKEND != "gemini_live":
                        play_voice_text(guidance)
                    await ui_broadcast_final(f"[NAV] {guidance}")
                except Exception:
                    pass

            # Broadcast the annotated frame to browser viewers
            if out_jpeg and camera_viewers:
                dead = []
                for viewer_ws in list(camera_viewers):
                    try:
                        await asyncio.wait_for(viewer_ws.send_bytes(out_jpeg), timeout=0.25)
                    except Exception:
                        dead.append(viewer_ws)
                for d in dead:
                    camera_viewers.discard(d)

    processor_task = asyncio.create_task(_processor())

    # ---- Gemini Live video feed (separate single-slot pipeline) ---------
    # Decoupled from the nav _processor above so a slow/degraded Gemini
    # connection can never add latency to navigation frame processing.
    # send_image() already has its own 3s timeout (see gemini_live_client.py);
    # this drop-to-latest holder/event is the "single in-flight slot" that
    # timeout assumes the caller provides.
    gemini_frame_holder = {"data": None}
    gemini_frame_event = asyncio.Event()

    async def _gemini_image_pump():
        while True:
            await gemini_frame_event.wait()
            gemini_frame_event.clear()
            data = gemini_frame_holder["data"]
            if data is None:
                continue
            try:
                await gemini_live.send_image(data)
            except Exception as e:
                if DEBUG:
                    print(f"[Gemini Live] send_image failed: {e}")

    gemini_pump_task = (
        asyncio.create_task(_gemini_image_pump())
        if AI_BACKEND == "gemini_live"
        else None
    )

    raw_holder = {"data": None}
    raw_event = asyncio.Event()

    async def _publish_canonical(data: bytes, received_at: float):
        nonlocal frame_counter
        frame_counter = await _handle_camera_frame(
            data, frame_counter, holder, frame_event,
            gemini_frame_holder, gemini_frame_event, gemini_pump_task,
            received_at,
        )

    canonicalizer_task = asyncio.create_task(
        run_latest_rgb_canonicalizer(
            raw_holder, raw_event, _publish_canonical,
            rgb_canonicalizer_telemetry,
        )
    )
    canonicalizer_health_task = asyncio.create_task(
        log_rgb_canonicalizer_health(rgb_canonicalizer_telemetry)
    )

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                queue_latest_raw_rgb(
                    msg["bytes"], time.monotonic(), raw_holder, raw_event,
                    rgb_canonicalizer_telemetry,
                )
            elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[CAMERA ERROR] {e}")
    finally:
        canonicalizer_health_task.cancel()
        try:
            await canonicalizer_health_task
        except (asyncio.CancelledError, Exception):
            pass
        canonicalizer_task.cancel()
        try:
            await canonicalizer_task
        except (asyncio.CancelledError, Exception):
            pass
        processor_task.cancel()
        try:
            await processor_task
        except (asyncio.CancelledError, Exception):
            pass
        if gemini_pump_task is not None:
            gemini_pump_task.cancel()
            try:
                await gemini_pump_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        esp32_camera_ws = None
        print("[DISCONNECTED] Camera (ESP32)")

        # Clean up navigation state
        if blind_path_navigator:
            blind_path_navigator.reset()
        if cross_street_navigator:
            cross_street_navigator.reset()
        if orchestrator:
            orchestrator.reset()
            if DEBUG: print("[NAV MASTER] Master reset")

# ---------- WebSocket: ESP32 camera+thermal entry (merged connection) ----------
# One physical connection carrying both camera JPEG and thermal frames,
# multiplexed with a 1-byte MSG_TYPE_CAM/MSG_TYPE_THERMAL prefix (see
# compile.ino's wsCamThermal) instead of the separate /ws/camera + /ws/thermal
# sockets above — merged to avoid the DMA/heap contention crashes seen
# running two TLS sockets' worth of camera+thermal traffic concurrently.
# /ws/camera and /ws/thermal are left in place (unused by current firmware)
# for rollback rather than deleted.
#
# Setup/teardown here is identical to ws_camera_esp (same navigator init,
# same processor_task/gemini_pump_task pipeline, same esp32_camera_ws
# mutual-exclusion global) plus a thermal_frame_count counter borrowed from
# ws_thermal_esp — only the receive loop differs, dispatching by MSG_TYPE
# instead of assuming every binary message is a camera frame.
@app.websocket("/ws/camera_thermal")
async def ws_camera_thermal_esp(ws: WebSocket):
    global esp32_camera_ws, blind_path_navigator, cross_street_navigator, cross_street_active, navigation_active, orchestrator
    if esp32_camera_ws is not None:
        await ws.close(code=1013)
        return
    esp32_camera_ws = ws
    await ws.accept()
    device_id = _device_id(ws)
    connected_at = time.monotonic()
    backend_metrics["camera_sensor_connects"] += 1
    print(f"[WS-INGEST] device={device_id} socket=camera_thermal event=connect", flush=True)

    # Initialize the blind-path navigator
    if blind_path_navigator is None and yolo_seg_model is not None:
        blind_path_navigator = BlindPathNavigator(yolo_seg_model, obstacle_detector)
        if DEBUG: print("[NAVIGATION] Blind-path navigator initialized")
    else:
        if blind_path_navigator is None and yolo_seg_model is None:
            print("[NAVIGATION] Warning: YOLO model not loaded, cannot initialize navigator")

    # Initialize the street-crossing navigator
    if cross_street_navigator is None:
        if yolo_seg_model:
            cross_street_navigator = CrossStreetNavigator(
                seg_model=yolo_seg_model,
                coco_model=None,  # traffic-light detection disabled
                obs_model=None    # obstacle detection also disabled for now (faster)
            )
            if DEBUG: print("[CROSS_STREET] Street-crossing navigator initialized")
        else:
            print("[CROSS_STREET] Error: segmentation model missing, cannot initialize street-crossing navigator")

    if orchestrator is None and blind_path_navigator is not None and cross_street_navigator is not None:
        orchestrator = NavigationMaster(blind_path_navigator, cross_street_navigator)
        if DEBUG: print("[NAV MASTER] Master state machine initialized")
    loop = asyncio.get_running_loop()
    frame_counter = 0
    camera_received_count = 0
    thermal_frame_count = 0
    thermal_dropped = 0
    thermal_ingest_q: asyncio.Queue[tuple[np.ndarray, float]] = asyncio.Queue(
        maxsize=ESP_THERMAL_INGEST_QUEUE_MAX
    )

    async def _thermal_processor():
        nonlocal thermal_frame_count
        while True:
            frame, received_at = await thermal_ingest_q.get()
            jpeg_bytes, stats = await asyncio.get_running_loop().run_in_executor(
                None, _prepare_thermal_frame_blocking, frame
            )
            if jpeg_bytes is None:
                continue
            thermal_frame_count += 1
            dead = []
            for viewer_ws in list(thermal_viewers):
                try:
                    await asyncio.wait_for(viewer_ws.send_bytes(jpeg_bytes), timeout=0.25)
                    await asyncio.wait_for(viewer_ws.send_text(stats), timeout=0.25)
                except Exception:
                    dead.append(viewer_ws)
            for viewer_ws in dead:
                thermal_viewers.discard(viewer_ws)
            latency_ms = (time.monotonic() - received_at) * 1000
            if thermal_frame_count == 1 or thermal_frame_count % 20 == 0:
                print(
                    f"[WS-INGEST] device={device_id} socket=camera_thermal "
                    f"type=thermal bytes={frame.nbytes} queue={thermal_ingest_q.qsize()} "
                    f"latency_ms={latency_ms:.1f}",
                    flush=True,
                )

    thermal_processor_task = asyncio.create_task(_thermal_processor())
    imu_event = asyncio.Event()

    async def _imu_pump():
        while True:
            await imu_event.wait()
            imu_event.clear()
            sample = latest_imu["data"]
            if sample is not None:
                await imu_broadcast(json.dumps(sample))

    imu_pump_task = asyncio.create_task(_imu_pump())

    # ---- Drop-to-latest pipeline (identical to ws_camera_esp) ------------
    holder = {"data": None}
    frame_event = asyncio.Event()

    async def _processor():
        while True:
            await frame_event.wait()
            frame_event.clear()
            data = holder["data"]
            if data is None:
                continue
            try:
                out_jpeg, guidance = await loop.run_in_executor(
                    None, _process_camera_frame_blocking, data
                )
            except Exception as e:
                out_jpeg, guidance = None, None
                if DEBUG:
                    print(f"[NAV MASTER] processor error: {e}")

            if guidance:
                try:
                    if AI_BACKEND != "gemini_live":
                        play_voice_text(guidance)
                    await ui_broadcast_final(f"[NAV] {guidance}")
                except Exception:
                    pass

            if out_jpeg and camera_viewers:
                dead = []
                for viewer_ws in list(camera_viewers):
                    try:
                        await asyncio.wait_for(viewer_ws.send_bytes(out_jpeg), timeout=0.25)
                    except Exception:
                        dead.append(viewer_ws)
                for d in dead:
                    camera_viewers.discard(d)

    processor_task = asyncio.create_task(_processor())

    # ---- Gemini Live video feed (identical to ws_camera_esp) -------------
    gemini_frame_holder = {"data": None}
    gemini_frame_event = asyncio.Event()

    async def _gemini_image_pump():
        while True:
            await gemini_frame_event.wait()
            gemini_frame_event.clear()
            data = gemini_frame_holder["data"]
            if data is None:
                continue
            try:
                await gemini_live.send_image(data)
            except Exception as e:
                if DEBUG:
                    print(f"[Gemini Live] send_image failed: {e}")

    gemini_pump_task = (
        asyncio.create_task(_gemini_image_pump())
        if AI_BACKEND == "gemini_live"
        else None
    )

    raw_holder = {"data": None}
    raw_event = asyncio.Event()

    async def _publish_canonical(data: bytes, received_at: float):
        nonlocal frame_counter
        frame_counter = await _handle_camera_frame(
            data, frame_counter, holder, frame_event,
            gemini_frame_holder, gemini_frame_event, gemini_pump_task,
            received_at,
        )

    canonicalizer_task = asyncio.create_task(
        run_latest_rgb_canonicalizer(
            raw_holder, raw_event, _publish_canonical,
            rgb_canonicalizer_telemetry,
        )
    )
    canonicalizer_health_task = asyncio.create_task(
        log_rgb_canonicalizer_health(rgb_canonicalizer_telemetry)
    )

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                try:
                    msg_type, payload = parse_sensor_message(data)
                except ValueError as exc:
                    backend_metrics["invalid_sensor_packets"] += 1
                    if backend_metrics["invalid_sensor_packets"] % 25 == 1:
                        print(f"[WS-INGEST] invalid sensor packet reason={exc}", flush=True)
                    continue
                if msg_type == MSG_TYPE_CAM:
                    received_at = time.monotonic()
                    camera_received_count += 1
                    queue_latest_raw_rgb(
                        payload, received_at, raw_holder, raw_event,
                        rgb_canonicalizer_telemetry,
                    )
                    if camera_received_count == 1 or camera_received_count % 100 == 0:
                        print(
                            f"[WS-INGEST] device={device_id} socket=camera_thermal "
                            f"type=camera bytes={len(payload)} queue=latest count={camera_received_count}",
                            flush=True,
                        )
                elif msg_type == MSG_TYPE_THERMAL:
                    received_at = time.monotonic()
                    try:
                        canonical_thermal = _retain_latest_thermal(payload, received_at)
                    except ValueError:
                        backend_metrics["invalid_sensor_packets"] += 1
                        continue
                    if thermal_ingest_q.full():
                        try:
                            thermal_ingest_q.get_nowait()
                            thermal_dropped += 1
                        except asyncio.QueueEmpty:
                            pass
                    try:
                        thermal_ingest_q.put_nowait((canonical_thermal, received_at))
                    except asyncio.QueueFull:
                        thermal_dropped += 1
                elif msg_type == MSG_TYPE_IMU:
                    imu_data = payload
                    imu_data["ts"] = imu_data["uptime_ms"]
                    latest_imu.update({
                        "timestamp": time.monotonic(),
                        "sequence": imu_data["sequence"],
                        "data": imu_data,
                    })
                    process_imu_and_maybe_store(imu_data)
                    imu_event.set()
                elif msg_type == MSG_TYPE_STATUS:
                    latest_device_status.update({"timestamp": time.monotonic(), "data": payload})
            elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[CAMERA-THERMAL ERROR] {e}")
    finally:
        canonicalizer_health_task.cancel()
        try:
            await canonicalizer_health_task
        except (asyncio.CancelledError, Exception):
            pass
        canonicalizer_task.cancel()
        try:
            await canonicalizer_task
        except (asyncio.CancelledError, Exception):
            pass
        imu_pump_task.cancel()
        try:
            await imu_pump_task
        except (asyncio.CancelledError, Exception):
            pass
        thermal_processor_task.cancel()
        try:
            await thermal_processor_task
        except (asyncio.CancelledError, Exception):
            pass
        processor_task.cancel()
        try:
            await processor_task
        except (asyncio.CancelledError, Exception):
            pass
        if gemini_pump_task is not None:
            gemini_pump_task.cancel()
            try:
                await gemini_pump_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        esp32_camera_ws = None
        backend_metrics["camera_sensor_disconnects"] += 1
        print(
            f"[WS-INGEST] device={device_id} socket=camera_thermal event=disconnect "
            f"reason=client_or_receive_end connected_s={time.monotonic() - connected_at:.1f} "
            f"camera_frames={frame_counter} thermal_frames={thermal_frame_count} "
            f"thermal_dropped={thermal_dropped}",
            flush=True,
        )

        # Clean up navigation state
        if blind_path_navigator:
            blind_path_navigator.reset()
        if cross_street_navigator:
            cross_street_navigator.reset()
        if orchestrator:
            orchestrator.reset()
            if DEBUG: print("[NAV MASTER] Master reset")

# ---------- WebSocket: browser subscribes to camera frames ----------
@app.websocket("/ws/viewer")
async def ws_viewer(ws: WebSocket):
    await ws.accept()
    camera_viewers.add(ws)
    print(f"[VIEWER] Browser connected. Total viewers: {len(camera_viewers)}", flush=True)
    try:
        while True:
            # Keep the connection alive
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        print("[VIEWER] Browser disconnected", flush=True)
    finally:
        try:
            camera_viewers.remove(ws)
        except Exception:
            pass
        print(f"[VIEWER] Removed. Total viewers: {len(camera_viewers)}", flush=True)

def _colorize_thermal(frame: np.ndarray) -> np.ndarray:
    """Render the canonical thermal grid without another orientation change."""
    cfg = dict(thermal_display_config)
    if cfg["auto_range"]:
        lo, hi = np.percentile(frame, [5, 95])
    else:
        lo, hi = float(cfg["min_c"]), float(cfg["max_c"])
    if hi <= lo:
      hi = lo + 1e-6
    normed = np.clip((frame - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    palette = {
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "magma": cv2.COLORMAP_MAGMA,
        "turbo": cv2.COLORMAP_TURBO,
        "bone": cv2.COLORMAP_BONE,
    }.get(cfg["palette"], cv2.COLORMAP_INFERNO)
    colored = cv2.applyColorMap(normed, palette)
    interpolation = cv2.INTER_NEAREST if cfg["interpolation"] == "nearest" else cv2.INTER_CUBIC
    # Scale from the canonical 32x24 portrait grid with square thermal pixels.
    rows, cols = frame.shape
    scale = 10
    out_w, out_h = cols * scale, rows * scale
    rendered = cv2.resize(colored, (out_w, out_h), interpolation=interpolation)
    if cfg["hotspot"]:
        row, col = np.unravel_index(int(np.argmax(frame)), frame.shape)
        cv2.circle(rendered, (int((col + 0.5) * scale), int((row + 0.5) * scale)), 7, (255, 255, 255), 2)
    if cfg["labels"]:
        cv2.putText(rendered, f"{float(frame.min()):.1f}C - {float(frame.max()):.1f}C",
                    (6, out_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return rendered

# ---------- WebSocket: ESP32 thermal entry ("THRM" + 24x32 float32 binary) ----------
@app.websocket("/ws/thermal")
async def ws_thermal_esp(ws: WebSocket):
    """Dedicated socket for thermal sensor frames — kept separate from the
    camera socket so the two streams never interfere. Colorizes each frame
    and broadcasts the JPEG + max/min temps to thermal_viewers ONLY — NOT
    camera_viewers, which is the RGB camera_esp/viewer pair's frame set."""
    if STABILITY_MODE:
        await ws.close(code=1008, reason="legacy thermal endpoint disabled in STABILITY_MODE")
        return
    await ws.accept()
    print("[CONNECTED] Thermal (ESP32)", flush=True)
    thermal_frame_count = 0
    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                if len(data) >= 4 and data[:4] == b"THRM" and len(data) - 4 == 3072:
                    thermal_frame_count = await _handle_thermal_frame(data[4:], thermal_frame_count)
            elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[THERMAL ERROR] {e}", flush=True)
    finally:
        print("[DISCONNECTED] Thermal (ESP32)", flush=True)

# ---------- WebSocket: browser subscribes to thermal frames ----------
@app.websocket("/ws/thermal_viewer")
async def ws_thermal_viewer(ws: WebSocket):
    await ws.accept()
    thermal_viewers.add(ws)
    print(f"[THERMAL-VIEWER] Browser connected. Total viewers: {len(thermal_viewers)}", flush=True)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        print("[THERMAL-VIEWER] Browser disconnected", flush=True)
    finally:
        thermal_viewers.discard(ws)
        print(f"[THERMAL-VIEWER] Removed. Total viewers: {len(thermal_viewers)}", flush=True)

# ---------- WebSocket: browser subscribes to IMU data ----------
@app.websocket("/ws")
async def ws_imu(ws: WebSocket):
    await ws.accept()
    imu_ws_clients.add(ws)
    try:
        while True:
            msg = await ws.receive()
            if "text" in msg and msg["text"] is not None:
                # Same parsing/processing as the old UDPProto.datagram_received
                # path (see below) — this socket now doubles as the ESP32's
                # ingestion channel, not just the browser-viewer broadcast-out.
                try:
                    d = json.loads(msg["text"])
                    if 'ts' not in d and 'timestamp_ms' in d:
                        d['ts'] = d.pop('timestamp_ms')
                    process_imu_and_maybe_store(d)
                    asyncio.create_task(imu_broadcast(json.dumps(d)))
                except Exception:
                    pass
            elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        imu_ws_clients.discard(ws)

async def imu_broadcast(msg: str):
    if not imu_ws_clients: return
    dead = []
    for ws in list(imu_ws_clients):
        try:
            await asyncio.wait_for(ws.send_text(msg), timeout=0.25)
        except Exception:
            dead.append(ws)
    for ws in dead:
        imu_ws_clients.discard(ws)

# ---------- Server-side IMU estimation (kept as-is) ----------
from math import atan2, hypot, pi
GRAV_BETA   = 0.98
STILL_W     = 0.4
YAW_DB      = 0.08
YAW_LEAK    = 0.2
ANG_EMA     = 0.15
AUTO_REZERO = True
USE_PROJ    = True
FREEZE_STILL= True
G     = 9.807
A_TOL = 0.08 * G
# How many seconds of still IMU data to collect for startup gyro-bias calibration.
# At ~50 Hz this is ~100 samples; if the device is moved the window resets.
GYRO_CAL_SECONDS = 2.0
gLP = {"x":0.0, "y":0.0, "z":0.0}
gOff= {"x":0.0, "y":0.0, "z":0.0}
BIAS_ALPHA = 0.002
yaw  = 0.0
Rf = Pf = Yf = 0.0
ref = {"roll":0.0, "pitch":0.0, "yaw":0.0}
holdStart = 0.0
isStill   = False
last_ts_imu = 0.0
last_wall = 0.0
imu_store: List[Dict[str, Any]] = []

# Startup gyro-bias calibration state
_cal_done     = False  # True once the fast startup calibration has completed
_cal_samples: List[Tuple[float, float, float]] = []  # (wx, wy, wz) still samples
_cal_start_ms = 0.0   # t_ms when the current still window started

def _wrap180(a: float) -> float:
    a = a % 360.0
    if a >= 180.0: a -= 360.0
    if a < -180.0: a += 360.0
    return a

def process_imu_and_maybe_store(d: Dict[str, Any]):
    global gLP, gOff, yaw, Rf, Pf, Yf, ref, holdStart, isStill, last_ts_imu, last_wall
    global _cal_done, _cal_samples, _cal_start_ms

    t_ms = float(d.get("ts", 0.0))
    now_wall = time.monotonic()
    if t_ms <= 0.0:
        t_ms = (now_wall * 1000.0)
    if last_ts_imu <= 0.0 or t_ms <= last_ts_imu or (t_ms - last_ts_imu) > 3000.0:
        dt = 0.02
    else:
        dt = (t_ms - last_ts_imu) / 1000.0
    last_ts_imu = t_ms

    ax = float(((d.get("accel") or {}).get("x", 0.0)))
    ay = float(((d.get("accel") or {}).get("y", 0.0)))
    az = float(((d.get("accel") or {}).get("z", 0.0)))
    wx = float(((d.get("gyro")  or {}).get("x", 0.0)))
    wy = float(((d.get("gyro")  or {}).get("y", 0.0)))
    wz = float(((d.get("gyro")  or {}).get("z", 0.0)))

    gLP["x"] = GRAV_BETA * gLP["x"] + (1.0 - GRAV_BETA) * ax
    gLP["y"] = GRAV_BETA * gLP["y"] + (1.0 - GRAV_BETA) * ay
    gLP["z"] = GRAV_BETA * gLP["z"] + (1.0 - GRAV_BETA) * az
    gmag = hypot(gLP["x"], gLP["y"], gLP["z"]) or 1.0
    gHat = {"x": gLP["x"]/gmag, "y": gLP["y"]/gmag, "z": gLP["z"]/gmag}

    roll  = (atan2(az, ay)   * 180.0 / pi)
    pitch = (atan2(-ax, ay)  * 180.0 / pi)

    aNorm = hypot(ax, ay, az); wNorm = hypot(wx, wy, wz)
    nearFlat = (abs(roll) < 2.0 and abs(pitch) < 2.0)
    stillCond = (abs(aNorm - G) < A_TOL) and (wNorm < STILL_W)

    if stillCond:
        if holdStart <= 0.0: holdStart = t_ms
        if not isStill and (t_ms - holdStart) > 350.0: isStill = True
        gOff["x"] = (1.0 - BIAS_ALPHA)*gOff["x"] + BIAS_ALPHA*wx
        gOff["y"] = (1.0 - BIAS_ALPHA)*gOff["y"] + BIAS_ALPHA*wy
        gOff["z"] = (1.0 - BIAS_ALPHA)*gOff["z"] + BIAS_ALPHA*wz
    else:
        holdStart = 0.0; isStill = False

    # ---- Startup gyro-bias calibration ----
    # Collect still samples until GYRO_CAL_SECONDS of continuous stillness has
    # been observed, then replace gOff with the direct average.  This gives an
    # immediate accurate baseline instead of waiting ~10 s for the slow EMA to
    # converge from zero.  The existing EMA above continues running afterwards
    # to track slow thermal drift.
    if not _cal_done:
        if stillCond:
            if _cal_start_ms <= 0.0:
                _cal_start_ms = t_ms
            _cal_samples.append((wx, wy, wz))
            if (t_ms - _cal_start_ms) >= GYRO_CAL_SECONDS * 1000.0 and len(_cal_samples) >= 10:
                n = len(_cal_samples)
                gOff["x"] = sum(s[0] for s in _cal_samples) / n
                gOff["y"] = sum(s[1] for s in _cal_samples) / n
                gOff["z"] = sum(s[2] for s in _cal_samples) / n
                _cal_done = True
                _cal_samples.clear()
                print(
                    f"[IMU] Gyro bias calibrated: gOff = "
                    f"({gOff['x']:.4f}, {gOff['y']:.4f}, {gOff['z']:.4f})",
                    flush=True,
                )
        else:
            # Device moved — reset the collection window and start fresh
            _cal_samples.clear()
            _cal_start_ms = 0.0

    if USE_PROJ:
        yawdot = ((wx - gOff["x"])*gHat["x"] + (wy - gOff["y"])*gHat["y"] + (wz - gOff["z"])*gHat["z"])
    else:
        yawdot = (wy - gOff["y"])

    if abs(yawdot) < YAW_DB: yawdot = 0.0
    if FREEZE_STILL and stillCond: yawdot = 0.0

    yaw = _wrap180(yaw + yawdot * dt)

    if (YAW_LEAK > 0.0) and nearFlat and stillCond and abs(yaw) > 0.0:
        step = YAW_LEAK * dt * (-1.0 if yaw > 0 else (1.0 if yaw < 0 else 0.0))
        if abs(yaw) <= abs(step): yaw = 0.0
        else: yaw += step

    global Rf, Pf, Yf, ref, last_wall
    Rf = ANG_EMA * roll  + (1.0 - ANG_EMA) * Rf
    Pf = ANG_EMA * pitch + (1.0 - ANG_EMA) * Pf
    Yf = ANG_EMA * yaw   + (1.0 - ANG_EMA) * Yf

    if AUTO_REZERO and nearFlat and (wNorm < STILL_W):
        if holdStart <= 0.0: holdStart = t_ms
        if not isStill and (t_ms - holdStart) > 350.0:
            ref.update({"roll": Rf, "pitch": Pf, "yaw": Yf})
            isStill = True

    R = _wrap180(Rf - ref["roll"])
    P = _wrap180(Pf - ref["pitch"])
    Y = _wrap180(Yf - ref["yaw"])

    now_wall = time.monotonic()
    if last_wall <= 0.0 or (now_wall - last_wall) >= 0.100:
        last_wall = now_wall
        item = {
            "ts": t_ms/1000.0,
            "angles": {"roll": R, "pitch": P, "yaw": Y},
            "accel":  {"x": ax, "y": ay, "z": az},
            "gyro":   {"x": wx, "y": wy, "z": wz},
        }
        imu_store.append(item)

# ---------- UDP: receive IMU data and forward ----------
class UDPProto(asyncio.DatagramProtocol):
    def connection_made(self, transport):
        print(f"[UDP] listening on {UDP_IP}:{UDP_PORT}")
    def datagram_received(self, data, addr):
        try:
            s = data.decode('utf-8', errors='ignore').strip()
            d = json.loads(s)
            if 'ts' not in d and 'timestamp_ms' in d:
                d['ts'] = d.pop('timestamp_ms')
            process_imu_and_maybe_store(d)
            asyncio.create_task(imu_broadcast(json.dumps(d)))
        except Exception:
            pass



# === New: register a send callback for bridge_io (broadcast JPEG to /ws/viewer) ===
@app.on_event("startup")
async def startup_yolo_shadow():
    await yolo_client.start()


@app.on_event("startup")
async def startup_hand_tracking():
    await hand_client.start()


@app.on_event("startup")
async def startup_research_exporter():
    await research_exporter.start()


@app.on_event("startup")
async def on_startup_register_bridge_sender():
    # Save the main thread's event loop
    main_loop = asyncio.get_event_loop()
    
    def _sender(jpeg_bytes: bytes):
        # Note: this function may be called from a non-coroutine thread and must return to the main event loop
        try:
            # Check event loop state to avoid sending after shutdown
            if main_loop.is_closed():
                return
            
            # Mark that YOLO has started sending processed frames
            global yolomedia_sending_frames
            if not yolomedia_sending_frames:
                yolomedia_sending_frames = True
                if DEBUG: print("[YOLOMEDIA] Starting to send processed frames", flush=True)
            
            async def _broadcast():
                if not camera_viewers:
                    return
                dead = []
                for ws in list(camera_viewers):
                    try:
                        await asyncio.wait_for(ws.send_bytes(jpeg_bytes), timeout=0.25)
                    except Exception as e:
                        dead.append(ws)
                for ws in dead:
                    try:
                        camera_viewers.remove(ws)
                    except Exception:
                        pass
            
            # Use the saved main-thread event loop
            future = asyncio.run_coroutine_threadsafe(_broadcast(), main_loop)
            # Do not await the result to avoid blocking the producer thread
        except Exception as e:
            # Only log on unexpected errors
            if "Event loop is closed" not in str(e):
                print(f"[DEBUG] _sender error: {e}", flush=True)

    bridge_io.set_sender(_sender)


@app.on_event("startup")
async def startup_stability_workers():
    if await asyncio.to_thread(sync_recorder.start_recording):
        recording_pipeline.start()
        recording_audio_pipeline.start()
        audio_stream.recording_enqueue_callback = recording_audio_pipeline.enqueue_latest
    else:
        print("[RECORDER] Startup recording failed; ingestion remains active")


@app.on_event("startup")
async def startup_tts_sender():
    global _tts_sender_task
    if _tts_sender_task is None or _tts_sender_task.done():
        _tts_sender_task = asyncio.create_task(_paced_tts_sender())


@app.on_event("startup")
async def on_startup_init_audio():
    """Initialize the audio system at startup."""
    if STABILITY_MODE:
        print("[STABILITY] Local host audio output disabled; ESP32 speaker remains active")
        return
    # Initialize in a background thread to avoid blocking startup
    def _init():
        try:
            initialize_audio_system()
        except Exception as e:
            print(f"[AUDIO] Initialization failed: {e}")
    
    threading.Thread(target=_init, daemon=True).start()

def _log_gemini_connect_failure(task: "asyncio.Task") -> None:
    """Done-callback for the background gemini_live.connect() task below.

    Nothing awaits that task, so an exception in it would otherwise only
    ever surface as asyncio's generic "Task exception was never retrieved"
    warning — easy to miss, and exactly the kind of silent failure that
    made a missing GEMINI_API_KEY hard to diagnose before. This makes it
    loud instead: the server keeps running (non-blocking startup is the
    point), but every send_audio/send_image/send_text call will silently
    no-op until this is fixed and the server restarted, so this needs to
    be impossible to miss in the logs.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        print("=" * 60)
        print("  [Gemini Live] STARTUP CONNECT FAILED")
        print(f"  {type(exc).__name__}: {exc}")
        print("  Server is running, but Gemini Live is NOT connected —")
        print("  every send_audio/send_image/send_text call will silently")
        print("  no-op until this is fixed and the server is restarted.")
        print("=" * 60)

@app.on_event("startup")
async def startup_gemini():
    if AI_BACKEND == "gemini_live":
        task = asyncio.create_task(gemini_live.connect(response_modality="AUDIO"))
        task.add_done_callback(_log_gemini_connect_failure)
    # gemini_regular/qwen don't touch Gemini Live at all — they use local
    # Whisper ASR instead (see the AI_BACKEND != "gemini_live" branch above).


async def _gemini_keepalive_loop() -> None:
    """Sends an occasional idle-time video frame so the Gemini Live session
    never sits long enough to hit its ~150s no-activity close (code 1008).

    Only fires when there's genuinely nothing else going on: no active turn,
    and no real send (audio/image/text) in the last GEMINI_KEEPALIVE_INTERVAL_SEC.
    Tagged source="keepalive" end to end (GEMINI-TIMING lines, this print) so
    it's never mistaken for a real vision send. It also never touches
    latency_tracker or _vision_frame_sequence_for_turn — those are only set
    from _on_input_transcription's own pre_response vision send — so a
    keepalive frame structurally cannot show up as a turn's gemini_rgb_frame_id
    or any other per-turn telemetry field.
    """
    while True:
        await asyncio.sleep(5.0)
        if AI_BACKEND != "gemini_live":
            continue
        if not gemini_live.connected:
            continue
        if gemini_live.has_active_turn():
            continue
        if gemini_live.seconds_since_last_activity() < GEMINI_KEEPALIVE_INTERVAL_SEC:
            continue
        frame = latest_rgb.snapshot()
        if frame.data is None:
            continue
        sent = await gemini_live.send_image(
            frame.data, sequence=frame.sequence, source="keepalive"
        )
        print(
            f"[GEMINI-KEEPALIVE] sent={sent} bytes={len(frame.data)} "
            f"sequence={frame.sequence}",
            flush=True,
        )


@app.on_event("startup")
async def startup_gemini_keepalive():
    global _gemini_keepalive_task
    if AI_BACKEND == "gemini_live" and (
        _gemini_keepalive_task is None or _gemini_keepalive_task.done()
    ):
        _gemini_keepalive_task = asyncio.create_task(_gemini_keepalive_loop())


@app.on_event("startup")
async def on_startup():
    if STABILITY_MODE:
        print("[STABILITY] UDP IMU ingest disabled; using multiplexed sensor WebSocket")
        return
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(lambda: UDPProto(), local_addr=(UDP_IP, UDP_PORT))
    print("[OK] Server running on port 8081")

@app.on_event("shutdown")
async def on_shutdown():
    """Clean up resources when the application shuts down."""
    global _tts_sender_task, _gemini_keepalive_task
    print("[SHUTDOWN] Starting resource cleanup...")
    await hand_client.stop()
    await yolo_client.stop()
    await research_exporter.stop()
    await recording_pipeline.stop()
    await recording_audio_pipeline.stop()
    
    # Stop YOLO media processing
    stop_yolomedia()

    # Stop the Gemini Live session first, so its receive_task can't fire any
    # more _on_audio/_on_turn_complete callbacks while we tear down audio
    # state below.
    if AI_BACKEND == "gemini_live":
        await gemini_live.disconnect()

    active_turn_id = latency_tracker.active_turn_id
    if active_turn_id is not None:
        latency_tracker.mark("interrupted", active_turn_id)
        await _finalize_latency_turn("interrupted", active_turn_id)

    # Stop audio and AI tasks
    await hard_reset_audio("shutdown")
    if _tts_sender_task is not None:
        _tts_sender_task.cancel()
        try:
            await _tts_sender_task
        except asyncio.CancelledError:
            pass
        _tts_sender_task = None

    if _gemini_keepalive_task is not None:
        _gemini_keepalive_task.cancel()
        try:
            await _gemini_keepalive_task
        except asyncio.CancelledError:
            pass
        _gemini_keepalive_task = None

    print("[SHUTDOWN] Resource cleanup complete")

# app_main.py —— after the existing @app.on_event("startup") in the file, add another startup hook


# --- Export interface (optional) ---
def get_last_frames():
    return last_frames

def get_camera_ws():
    return esp32_camera_ws

if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=8081,
        log_level="warning", access_log=False,
        loop="asyncio", workers=1, reload=False,
        # Stock 20s/20s defaults were too aggressive for the ESP32's WiFi —
        # confirmed root cause of "1011 keepalive ping timeout" disconnects.
        # This is a single Config-level setting shared by every websocket
        # route in this process (mic, camera, thermal, viewer, ui, imu) —
        # uvicorn has no per-route override. That's fine here: mic and
        # camera share one physical device, one WiFi radio, and one Arduino
        # loop() scheduling both sockets, so they have the same jitter
        # profile anyway.
        ws_ping_interval=30.0,
        ws_ping_timeout=60.0,
    )
