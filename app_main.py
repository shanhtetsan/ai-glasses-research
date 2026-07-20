# app_main.py
# -*- coding: utf-8 -*-
import os, sys, time, json, asyncio, base64, audioop, tempfile, wave
from typing import Any, Dict, Optional, Tuple, List, Callable, Set, Deque
from collections import deque
from dataclasses import dataclass
import re
# Add after other imports:
from qwen_extractor import extract_english_label
from navigation_master import NavigationMaster, OrchestratorResult 
# New: import blind-path navigator
from workflow_blindpath import BlindPathNavigator
# New: import cross-street navigator
from workflow_crossstreet import CrossStreetNavigator
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from obstacle_detector_client import ObstacleDetectorClient
import general_detector  # prompt-free YOLOE general object detection (lazy-loaded)

import torch


import mediapipe as mp
import bridge_io
import threading
# import yolomedia  # must be in the same directory as app_main.py, filename is yolomedia.py

try:
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
print("Gemini Key =", (_gk[:6] + "..." + _gk[-4:]) if len(_gk) > 12 else "(not set)")

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
import whisper as _whisper_lib
from asr_config import load_whisper_config
_WHISPER_CFG = load_whisper_config()
print(f"[...] Loading Whisper model {_WHISPER_CFG.model!r} (lang={_WHISPER_CFG.language or 'auto'})...")
_whisper_model = _whisper_lib.load_model(_WHISPER_CFG.model)
print("[OK] Whisper model ready")

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
from audio_stream import (
    register_stream_route,         # mount /stream.wav
    broadcast_pcm16_realtime,      # distribute 16k PCM to all connected clients in real time
    hard_reset_audio,              # master switch for audio + AI playback
    BYTES_PER_20MS_16K,
    is_playing_now,
    current_ai_task,
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

# Persistent audioop.ratecv state, kept across _on_audio calls within a turn
# so consecutive chunks resample smoothly instead of clicking at boundaries.
# Reset whenever a new response turn starts (on turn_complete/interrupted).
_ratecv_state_8k = None
_ratecv_state_16k = None

async def _on_audio(pcm24k: bytes):
    """Gemini Live streams 24kHz PCM16 audio deltas.

    Two different downstream consumers need two different sample rates:
      - the ESP32 TTS websocket expects 8kHz (matches its i2s DAC / the old
        macOS-TTS output rate)
      - broadcast_pcm16_realtime / the browser's /stream.wav expects 16kHz
        (see its import comment and BYTES_PER_20MS_16K)
    Resampling once to 8kHz and reusing that buffer for both was the bug
    that made browser audio inaudible/garbled — each consumer now gets its
    own correctly-rated stream.
    """
    global _esp32_tts_started, _ratecv_state_8k, _ratecv_state_16k

    try:
        pcm8k, _ratecv_state_8k = audioop.ratecv(pcm24k, 2, 1, 24000, 8000, _ratecv_state_8k)
        pcm16k, _ratecv_state_16k = audioop.ratecv(pcm24k, 2, 1, 24000, 16000, _ratecv_state_16k)
    except Exception as e:
        print(f"[Gemini Live] audio resample failed: {e}", flush=True)
        return

    if pcm8k:
        _ws = esp32_audio_ws
        if _ws and _ws.client_state == WebSocketState.CONNECTED:
            try:
                if not _esp32_tts_started:
                    await _ws.send_text("TTS:START")
                    _esp32_tts_started = True
                for i in range(0, len(pcm8k), _TTS_CHUNK):
                    await _ws.send_bytes(pcm8k[i:i + _TTS_CHUNK])
            except Exception as e:
                print(f"[TTS-WS] send failed: {e}", flush=True)

    if pcm16k:
        # Browser /stream.wav — this is the one that was getting the wrong
        # sample rate before.
        await broadcast_pcm16_realtime(pcm16k)

async def _on_input_transcription(text: str):
    """User speech transcript, streamed incrementally by Gemini Live."""
    if not text:
        return
    _input_text_buf.append(text)
    try:
        # Tagged so the UI can tell this apart from the AI's partial text —
        # see the note in _on_turn_complete about why this also needs a
        # ui_broadcast_final once the turn ends.
        await ui_broadcast_partial("(user) " + "".join(_input_text_buf))
    except Exception:
        pass
    combined = "".join(_input_text_buf)
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
    global _esp32_tts_started, omni_conversation_active, omni_previous_nav_state
    global _ratecv_state_8k, _ratecv_state_16k

    _ws = esp32_audio_ws
    if _esp32_tts_started and _ws and _ws.client_state == WebSocketState.CONNECTED:
        try:
            await _ws.send_text("TTS:END")
        except Exception:
            pass
    _esp32_tts_started = False
    # New turn next time — don't carry resample state across turn boundaries
    _ratecv_state_8k = None
    _ratecv_state_16k = None

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

    if user_text:
        async with interrupt_lock:
            # gemini_live already spoke the reply live over audio; this call
            # is only for side effects (nav start/stop, item search, etc.)
            await start_ai_with_text_custom(user_text)

    omni_conversation_active = False
    if orchestrator and omni_previous_nav_state:
        orchestrator.force_state(omni_previous_nav_state)
        if DEBUG: print(f"[OMNI] Dialogue ended, restored to {omni_previous_nav_state} mode")
        omni_previous_nav_state = None

async def _on_interrupted():
    """User barged in and cut off Gemini's current response."""
    global _esp32_tts_started, _ratecv_state_8k, _ratecv_state_16k
    print("[Gemini Live] Response interrupted by user", flush=True)
    _esp32_tts_started = False
    _ratecv_state_8k = None
    _ratecv_state_16k = None
    await hard_reset_audio("gemini_interrupted")

gemini_live.on_audio = _on_audio
gemini_live.on_input_transcription = _on_input_transcription
gemini_live.on_output_transcription = _on_output_transcription
gemini_live.on_turn_complete = _on_turn_complete
gemini_live.on_interrupted = _on_interrupted

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
    if last_frames:
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
last_frames: Deque[Tuple[float, bytes]] = deque(maxlen=10)

camera_viewers: Set[WebSocket] = set()
esp32_camera_ws: Optional[WebSocket] = None
imu_ws_clients: Set[WebSocket] = set()
esp32_audio_ws: Optional[WebSocket] = None

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
if DEBUG: print("[NAVIGATION] Loading navigation models...")
load_navigation_models()
if DEBUG: print(f"[NAVIGATION] Model loading complete - yolo_seg_model: {yolo_seg_model is not None}")

# Start synchronous recording
sync_recorder.start_recording()

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



# Pre-load the traffic-light detection model (prevents stutter when entering WAIT_TRAFFIC_LIGHT state)
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
    else:
        if DEBUG: print("[TRAFFIC_LIGHT] Traffic-light detection model pre-load failed")
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
            # Play launch voice prompt and broadcast to UI
            play_voice_text("Street crossing mode activated.")
            await ui_broadcast_final("[System] Street-crossing mode started")
        else:
            print("[CROSS_STREET] Warning: navigation master not initialized!")
            play_voice_text("Failed to start crossing mode, please try again later.")
            await ui_broadcast_final("[System] Navigation system not ready")
        return
    
    if any(k in user_text for k in ["过马路结束", "结束过马路",
                                     "stop crossing", "end crossing", "done crossing"]):
        if orchestrator:
            orchestrator.stop_navigation()
            if DEBUG: print(f"[CROSS_STREET] Navigation stopped, state: {orchestrator.get_state()}")
            # Play stop voice prompt and broadcast to UI
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
        if last_frames:
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
                        # Primary path: send raw mono-16 PCM to ESP32 over /ws_audio WebSocket.
                        # Firmware taskTTSPlay consumes qTTS and writes to i2sOut.
                        _ws = esp32_audio_ws
                        if _ws and _ws.client_state == WebSocketState.CONNECTED:
                            try:
                                await _ws.send_text("TTS:START")
                                _CHUNK = 2040  # fits TTSChunk.data[2048] on the firmware side
                                for _i in range(0, len(pcm8k), _CHUNK):
                                    await _ws.send_bytes(pcm8k[_i:_i + _CHUNK])
                                await _ws.send_text("TTS:END")
                                print(f"[TTS-WS] sent {len(pcm8k)} bytes in {-(-len(pcm8k)//_CHUNK)} chunks", flush=True)
                            except Exception as _ws_err:
                                print(f"[TTS-WS] send failed: {_ws_err}", flush=True)
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

@app.get("/api/health", response_class=PlainTextResponse)
def health():
    return "OK"


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
    async with interrupt_lock:
        await start_ai_with_text_custom(text)
    return JSONResponse({"ran": text})

class SettingsPayload(BaseModel):
    jpeg_quality: Optional[int] = None
    vad_silence_rms: Optional[int] = None
    vad_silence_ms: Optional[int] = None
    vad_min_speech_ms: Optional[int] = None

@app.get("/api/settings")
def get_settings():
    return JSONResponse({
        "jpeg_quality": JPEG_QUALITY,
        "vad_silence_rms": VAD_SILENCE_RMS,
        "vad_silence_ms": VAD_SILENCE_MS,
        "vad_min_speech_ms": VAD_MIN_SPEECH_MS,
    })

@app.post("/api/settings")
def update_settings(payload: SettingsPayload):
    global JPEG_QUALITY, VAD_SILENCE_RMS, VAD_SILENCE_MS, VAD_MIN_SPEECH_MS
    global VAD_SILENCE_CHUNKS, VAD_MIN_SPEECH_CHUNKS
    if payload.jpeg_quality is not None:
        JPEG_QUALITY = max(1, min(100, payload.jpeg_quality))
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
    global esp32_audio_ws
    esp32_audio_ws = ws
    await ws.accept()
    print("[CONNECTED] Mic (ESP32 audio)")

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

                if cmd == "START":
                    print("[MIC] Listening — waiting for speech...")
                    streaming            = True
                    pcm_buffer           = bytearray()
                    vad_silent_chunks    = 0
                    vad_speech_chunks    = 0
                    vad_speech_detected  = False
                    await ui_broadcast_partial("（Recording…）")
                    await ws.send_text("OK:STARTED")

                elif cmd == "STOP":
                    streaming = False
                    if AI_BACKEND == "gemini_live":
                        print("[MIC] Stopped streaming")
                        mic_streaming = False
                        await ws.send_text("OK:STOPPED")
                    else:
                        print("[MIC] Transcribing...")
                        buf = bytes(pcm_buffer) if pcm_buffer else b""
                        pcm_buffer          = None
                        vad_silent_chunks   = 0
                        vad_speech_chunks   = 0
                        vad_speech_detected = False
                        await ws.send_text("OK:STOPPED")
                        await _run_whisper_and_dispatch(buf)

                elif raw.startswith("PROMPT:"):
                    # Device-initiated prompt (bypasses ASR entirely)
                    text = raw[len("PROMPT:"):].strip()
                    if text:
                        async with interrupt_lock:
                            await start_ai_with_text_custom(text)
                        await ws.send_text("OK:PROMPT_ACCEPTED")
                    else:
                        await ws.send_text("ERR:EMPTY_PROMPT")

            elif "bytes" in msg and msg["bytes"] is not None:
                chunk = msg["bytes"]

                if AI_BACKEND == "gemini_live":
                    # Gemini Live owns the mic entirely in this mode — it runs
                    # its own server-side VAD/turn-detection on the raw stream,
                    # so the local RMS-VAD/Whisper pipeline below must stay out
                    # of the way (it would otherwise fire its own transcription
                    # off the same audio and double-dispatch commands).
                    if streaming and not is_playing_now():
                        await gemini_live.send_audio(chunk)
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
                    vad_speech_chunks += 1
                    if not vad_speech_detected and vad_speech_chunks >= VAD_MIN_SPEECH_CHUNKS:
                        vad_speech_detected = True
                        print("[MIC] Speech detected", flush=True)
                else:
                    if vad_speech_detected:
                        vad_silent_chunks += 1
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
        pcm_buffer = None
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        if esp32_audio_ws is ws:
            esp32_audio_ws = None
        print("[DISCONNECTED] Mic (ESP32 audio)")

def _process_camera_frame_blocking(data: bytes):
    """Decode one JPEG frame and run detection/navigation on it.

    Pure synchronous CPU work — safe to run in a thread executor. Does NO
    websocket or async I/O. Returns (out_jpeg_bytes | None, guidance_text | None);
    the caller broadcasts the JPEG and speaks the guidance from the event loop.
    """
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


# ---------- WebSocket: ESP32 camera entry (JPEG binary) ----------
@app.websocket("/ws/camera")
async def ws_camera_esp(ws: WebSocket):
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
                    play_voice_text(guidance)
                    await ui_broadcast_final(f"[NAV] {guidance}")
                except Exception:
                    pass

            # Broadcast the annotated frame to browser viewers
            if out_jpeg and camera_viewers:
                dead = []
                for viewer_ws in list(camera_viewers):
                    try:
                        await viewer_ws.send_bytes(out_jpeg)
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

    gemini_pump_task = asyncio.create_task(_gemini_image_pump()) if AI_BACKEND == "gemini_live" else None

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                frame_counter += 1

                # Cheap per-frame bookkeeping stays in the receive loop so it
                # sees every frame (recording, latest-frame cache, yolomedia feed).
                try:
                    sync_recorder.record_frame(data)
                except Exception as e:
                    if frame_counter % 100 == 0:  # avoid log spam
                        print(f"[RECORDER] Failed to record frame: {e}")
                try:
                    last_frames.append((time.time(), data))
                except Exception:
                    pass
                bridge_io.push_raw_jpeg(data)

                # Hand the newest frame to the processor. If an older unprocessed
                # frame is still sitting here, it's overwritten (dropped).
                holder["data"] = data
                frame_event.set()

                if gemini_pump_task is not None:
                    gemini_frame_holder["data"] = data
                    gemini_frame_event.set()

            elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[CAMERA ERROR] {e}")
    finally:
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

# ---------- WebSocket: ESP32 thermal entry (MLX90640, "THRM" binary) ----------
@app.websocket("/ws/thermal")
async def ws_thermal_esp(ws: WebSocket):
    """Dedicated socket for MLX90640 thermal frames — kept separate from the
    camera socket so the two streams never interfere. Forwards each frame
    straight to the browser viewers."""
    await ws.accept()
    print("[CONNECTED] Thermal (ESP32)", flush=True)
    thermal_frame_count = 0
    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                if len(data) >= 4 and data[:4] == b"THRM":
                    thermal_frame_count += 1
                    if thermal_frame_count == 1 or thermal_frame_count % 20 == 0:
                        print(f"[THERMAL] received {thermal_frame_count} frames "
                              f"({len(data)} bytes), forwarding to {len(camera_viewers)} viewer(s)",
                              flush=True)
                    dead = []
                    for viewer_ws in list(camera_viewers):
                        try:
                            await viewer_ws.send_bytes(data)
                        except Exception:
                            dead.append(viewer_ws)
                    for d in dead:
                        camera_viewers.discard(d)
            elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[THERMAL ERROR] {e}", flush=True)
    finally:
        print("[DISCONNECTED] Thermal (ESP32)", flush=True)

# ---------- WebSocket: browser subscribes to IMU data ----------
@app.websocket("/ws")
async def ws_imu(ws: WebSocket):
    await ws.accept()
    imu_ws_clients.add(ws)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        imu_ws_clients.discard(ws)

async def imu_broadcast(msg: str):
    if not imu_ws_clients: return
    dead = []
    for ws in list(imu_ws_clients):
        try:
            await ws.send_text(msg)
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
                        await ws.send_bytes(jpeg_bytes)
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
async def on_startup_init_audio():
    """Initialize the audio system at startup."""
    # Initialize in a background thread to avoid blocking startup
    def _init():
        try:
            initialize_audio_system()
        except Exception as e:
            print(f"[AUDIO] Initialization failed: {e}")
    
    threading.Thread(target=_init, daemon=True).start()

@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(lambda: UDPProto(), local_addr=(UDP_IP, UDP_PORT))
    print("[OK] Server running on port 8081")

@app.on_event("shutdown")
async def on_shutdown():
    """Clean up resources when the application shuts down."""
    print("[SHUTDOWN] Starting resource cleanup...")
    
    # Stop YOLO media processing
    stop_yolomedia()

    # Stop audio and AI tasks
    await hard_reset_audio("shutdown")
    
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
        loop="asyncio", workers=1, reload=False
    )
