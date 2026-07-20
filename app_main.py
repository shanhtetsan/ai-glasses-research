# app_main.py
# -*- coding: utf-8 -*-
import os, sys, time, json, asyncio, base64, audioop

# ---- Timestamp every print() call, everywhere in this file ----
# Overriding the builtin here means every existing print("[MIC] ...") /
# print("[LATENCY] ...") / etc. call throughout the whole file automatically
# gets a "[HH:MM:SS.mmm]" prefix — no need to touch each individual call
# site. Must happen before any other print() runs, so it's placed as early
# as possible, right after the first imports.
import builtins
from datetime import datetime
_original_print = builtins.print
def _timestamped_print(*args, **kwargs):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
    _original_print(f"[{ts}]", *args, **kwargs)
builtins.print = _timestamped_print

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
DEBUG_VAD = False   # set True temporarily if you need to see per-chunk RMS values again

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

# ---- Latency instrumentation ----
# Prints one line per trial: total_latency — seconds from when you stop
# speaking to when the full response is completely done (Gemini Live
# finishes speaking the reply, or gemini_regular/qwen finish generating
# the text reply). Same definition across all three backends.
def _log_latency(event: str, **fields):
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[LATENCY] backend={AI_BACKEND} event={event} {parts}", flush=True)



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

# ---- ASR ----
# AI_BACKEND == "gemini_live": ASR is handled server-side by Gemini Live
# (input_audio_transcription) — raw mic PCM16 is forwarded straight to
# GeminiLiveClient.send_audio(), no local model needed.
# AI_BACKEND == "gemini_regular" / "qwen": these are text-only backends with
# no ASR of their own (gemini-3.1-flash-live-preview only supports AUDIO
# response modality, so it can't be reused as a TEXT-only ASR engine either —
# confirmed by testing, not just docs). So for these two, local Whisper is
# back exactly as it worked before the Live migration: buffer PCM16 with a
# simple RMS VAD, transcribe on silence, dispatch the text.
SAMPLE_RATE = 16000
_whisper_model = None
if AI_BACKEND != "gemini_live":
    import whisper as _whisper_lib
    print("[...] Loading Whisper model...")
    _whisper_model = _whisper_lib.load_model("base")
    print("[OK] Whisper model ready")

# ---- Voice Activity Detection (VAD) tuning ----
# Used by ws_audio's RMS-based VAD loop for the gemini_regular/qwen backends
# (see above). Has no effect on AI_BACKEND == "gemini_live", which streams
# raw audio continuously and lets Gemini Live's own server-side VAD decide
# turn boundaries.
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
)
from gemini_live_client import GeminiLiveClient
gemini_live = GeminiLiveClient()

# Only import whichever text-generation backend was actually selected —
# importing omni_client.py loads the full Qwen2.5-Omni-3B model into memory,
# so we don't want that happening unless AI_BACKEND=qwen was explicitly asked for.
_backend_stream_chat = None
if AI_BACKEND == "gemini_regular":
    from gemini_client import stream_chat as _backend_stream_chat
elif AI_BACKEND == "qwen":
    from omni_client import stream_chat as _backend_stream_chat

from asr_core import (
    ASRCallback,
    set_current_recognition,
    stop_current_recognition,
    INTERRUPT_KEYWORDS,
    _normalize_cn,
)
from audio_player import initialize_audio_system, play_voice_text

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

# ---- Latency tracking (gemini_live path) ----
# Proxy for "when did the user stop talking": the last time Gemini actually
# transcribed something (input_audio_transcription), NOT the last raw mic
# chunk sent — the ESP32 streams audio continuously the whole time it's
# connected, not just while you're talking, so "last chunk sent" is always
# only ~20ms old regardless of when you actually stopped speaking. That was
# a bug in the original version of this metric; using the last transcription
# update instead is a real anchor to end-of-speech.
_last_transcription_ts = [0.0]
_audio_turn_started = False   # True once we've seen at least one audio byte of the current response

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
    global _audio_turn_started

    _audio_turn_started = True

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
                async def _send_to_esp32():
                    global _esp32_tts_started
                    if not _esp32_tts_started:
                        await _ws.send_text("TTS:START")
                        _esp32_tts_started = True
                    for i in range(0, len(pcm8k), _TTS_CHUNK):
                        await _ws.send_bytes(pcm8k[i:i + _TTS_CHUNK])
                # Same protection as _speak_and_broadcast: don't let a
                # non-draining/broken ESP32 connection block this callback
                # (and therefore Gemini Live's whole receive loop) forever.
                await asyncio.wait_for(_send_to_esp32(), timeout=3.0)
            except asyncio.TimeoutError:
                print("[TTS-WS] send timed out after 3s (ESP32 not draining?) — skipping chunk", flush=True)
            except Exception as e:
                print(f"[TTS-WS] send failed: {e}", flush=True)

    if pcm16k:
        # Browser /stream.wav — this is the one that was getting the wrong
        # sample rate before. Previously wrapped in asyncio.wait_for(), but
        # that forcibly cancels broadcast_pcm16_realtime() mid-execution on
        # timeout — since that function lives in audio_stream.py and we
        # don't control its internals, cancelling it partway through (mid
        # lock, mid queue-write, etc.) is a likely cause of the persistent
        # mic-reconnect storm seen right after that timeout fired. Spawned
        # as an independent background task instead: never blocks this
        # callback, and is allowed to run to completion or fail on its own
        # rather than being torn down mid-flight.
        _spawn_background_task(broadcast_pcm16_realtime(pcm16k), name="broadcast_pcm16_realtime")

async def _on_input_transcription(text: str):
    """User speech transcript, streamed incrementally by Gemini Live."""
    if not text:
        return
    _last_transcription_ts[0] = time.time()
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
    global _ratecv_state_8k, _ratecv_state_16k, _audio_turn_started

    if _audio_turn_started:
        # Unlike the Whisper path, we can't print a start marker in real
        # time here — Gemini Live's own server-side VAD decides when your
        # speech ended, and we only find out which transcription update was
        # the *last* one retroactively, once no more arrive. So both markers
        # print together, right now, with the reconstructed start time.
        print(f"[LATENCY] backend={AI_BACKEND} measurement STARTED (retroactively, at last transcription)", flush=True)
        print(f"[LATENCY] backend={AI_BACKEND} measurement ENDED (audio delivered)", flush=True)
        _log_latency("total_latency", seconds=round(time.time() - _last_transcription_ts[0], 3))
    _audio_turn_started = False

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
            await try_dispatch_command(user_text)

    omni_conversation_active = False
    if orchestrator and omni_previous_nav_state:
        orchestrator.force_state(omni_previous_nav_state)
        if DEBUG: print(f"[OMNI] Dialogue ended, restored to {omni_previous_nav_state} mode")
        omni_previous_nav_state = None

async def _on_interrupted():
    """User barged in and cut off Gemini's current response."""
    global _esp32_tts_started, _ratecv_state_8k, _ratecv_state_16k, _audio_turn_started
    print("[Gemini Live] Response interrupted by user", flush=True)
    _esp32_tts_started = False
    _ratecv_state_8k = None
    _ratecv_state_16k = None
    _audio_turn_started = False
    await hard_reset_audio("gemini_interrupted")

gemini_live.on_audio = _on_audio
gemini_live.on_input_transcription = _on_input_transcription
gemini_live.on_output_transcription = _on_output_transcription
gemini_live.on_turn_complete = _on_turn_complete
gemini_live.on_interrupted = _on_interrupted

# ---- Helpers for the non-live backends (gemini_regular / qwen) ----
# ASR is local Whisper (see ws_audio / _run_whisper_and_dispatch) — Gemini
# Live isn't connected at all for these two backends. Replies are spoken via
# local TTS (_speak_and_broadcast below), matching gemini_live's behavior of
# producing audio output, so total_latency is comparable audio-to-audio
# across all three backends.

async def _signal_stream_finished():
    """Tell any /stream.wav listeners the current audio response is over.
    Used after gemini_live's own audio finishes (_on_turn_complete) and
    after local TTS finishes for gemini_regular/qwen (run_backend_turn)."""
    from audio_stream import stream_clients  # local import to avoid circular dependency
    for sc in list(stream_clients):
        if not sc.abort_event.is_set():
            try: sc.q.put_nowait(b"\x00" * BYTES_PER_20MS_16K)  # one frame of silence
            except Exception: pass
            try: sc.q.put_nowait(None)
            except Exception: pass

async def _local_say_tts(text: str):
    """macOS-only TTS: 'say' -> AIFF -> afconvert -> 16-bit PCM WAV at its
    native sample rate. Used only for the gemini_regular/qwen backends,
    since those return text but no audio of their own (Gemini Live produces
    its own audio and doesn't need this). If you're not on macOS, swap this
    out for your platform's TTS — everything downstream just expects
    (pcm_bytes, sample_rate).
    """
    import tempfile, wave

    word_count = len(text.split())
    char_count = len(text)
    print(f"[TTS TIMING] text length: {word_count} words, {char_count} chars", flush=True)

    async def _wait_with_timeout(proc, timeout, label):
        t_spawn_to_wait_start = time.time()
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
            elapsed = time.time() - t_spawn_to_wait_start
            print(f"[TTS TIMING] {label} finished in {elapsed:.2f}s", flush=True)
        except asyncio.TimeoutError:
            elapsed = time.time() - t_spawn_to_wait_start
            # Seen in practice: 'say'/'afconvert' can take 80+ seconds with
            # zero indication why on a CPU-starved machine (e.g. Whisper
            # transcription + camera processing competing for the same
            # cores), silently ballooning total_latency with no error
            # printed anywhere. Cap it so that can't happen invisibly again.
            print(f"[TTS] {label} timed out after {timeout}s (was still running at {elapsed:.2f}s) — killing and aborting synthesis", flush=True)
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            raise

    with tempfile.TemporaryDirectory() as tmpdir:
        aiff_path = os.path.join(tmpdir, "out.aiff")
        wav_path  = os.path.join(tmpdir, "out.wav")

        t_before_spawn = time.time()
        p1 = await asyncio.create_subprocess_exec(
            "say", "-o", aiff_path, "--", text,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        spawn_elapsed = time.time() - t_before_spawn
        if spawn_elapsed > 0.5:
            # If spawning itself is slow, that points to OS-level process
            # creation being starved (e.g. CPU/scheduler contention) rather
            # than 'say' itself being slow to synthesize.
            print(f"[TTS TIMING] 'say' subprocess spawn took {spawn_elapsed:.2f}s (unusually slow)", flush=True)
        await _wait_with_timeout(p1, 10.0, "'say'")

        t_before_spawn2 = time.time()
        p2 = await asyncio.create_subprocess_exec(
            "afconvert", aiff_path, wav_path, "-f", "WAVE", "-d", "LEI16",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        spawn_elapsed2 = time.time() - t_before_spawn2
        if spawn_elapsed2 > 0.5:
            print(f"[TTS TIMING] 'afconvert' subprocess spawn took {spawn_elapsed2:.2f}s (unusually slow)", flush=True)
        await _wait_with_timeout(p2, 10.0, "'afconvert'")

        with wave.open(wav_path, "rb") as w:
            ch = w.getnchannels()
            sw = w.getsampwidth()
            fr = w.getframerate()
            pcm = w.readframes(w.getnframes())

        if ch == 2:
            pcm = audioop.tomono(pcm, sw, 1, 0)
        return pcm, fr

async def _speak_and_broadcast(text: str):
    """Synthesize text locally and push it out both the ESP32 TTS websocket
    and the browser /stream.wav — same two destinations _on_audio feeds for
    gemini_live, just sourced from local TTS instead of Gemini's own audio."""
    pcm, native_rate = await _local_say_tts(text)
    if not pcm:
        return

    pcm8k, _ = audioop.ratecv(pcm, 2, 1, native_rate, 8000, None)
    pcm16k, _ = audioop.ratecv(pcm, 2, 1, native_rate, 16000, None)

    if pcm8k:
        _ws = esp32_audio_ws
        if _ws and _ws.client_state == WebSocketState.CONNECTED:
            try:
                async def _send_to_esp32():
                    await _ws.send_text("TTS:START")
                    for i in range(0, len(pcm8k), _TTS_CHUNK):
                        await _ws.send_bytes(pcm8k[i:i + _TTS_CHUNK])
                    await _ws.send_text("TTS:END")
                # If the ESP32 isn't draining its socket (e.g. speaker/playback
                # code stuck or broken), send_bytes() can block indefinitely
                # waiting for TCP buffer space — that stalled the whole turn,
                # including the total_latency measurement, for 40+ seconds.
                # Cap it so a dead speaker can't hang the pipeline.
                await asyncio.wait_for(_send_to_esp32(), timeout=8.0)
            except asyncio.TimeoutError:
                print("[TTS-WS] send timed out after 8s (ESP32 not draining?) — skipping", flush=True)
            except Exception as e:
                print(f"[TTS-WS] send failed: {e}", flush=True)

    if pcm16k:
        # Fire-and-forget, not forcibly cancelled — see the matching note in
        # _on_audio. The 65s total_latency outlier confirmed this call can
        # genuinely hang when nothing drains /stream.wav, but forcibly
        # cancelling it mid-execution via asyncio.wait_for() likely corrupted
        # shared state in audio_stream.py and caused the persistent mic
        # reconnect storm seen right after. This still keeps total_latency
        # from including this call's time, without tearing it down mid-flight.
        _spawn_background_task(broadcast_pcm16_realtime(pcm16k), name="broadcast_pcm16_realtime")

async def run_backend_turn(user_text: str, speech_end_ts: Optional[float] = None):
    """Generate a text reply using the selected non-live backend, speak it
    via local TTS, and broadcast both to the UI/ESP32/browser. Mirrors
    gemini_live's audio delivery path so all three backends are directly
    comparable on total_latency (end-of-speech -> audio fully delivered).

    speech_end_ts: time.time() of when the user's speech ended (i.e. right
    before Whisper started transcribing) — total_latency is measured from
    here when available, so it's directly comparable to gemini_live's
    total_latency (also measured from end-of-speech). None for the typed-
    PROMPT path, where there's no preceding ASR step; total_latency falls
    back to measuring from this function's own start in that case.
    """
    global omni_conversation_active, omni_previous_nav_state

    if _backend_stream_chat is None:
        print(f"[BACKEND] No stream_chat available for AI_BACKEND={AI_BACKEND!r}", flush=True)
        return

    t0 = time.time()
    if speech_end_ts is None:
        # Typed-PROMPT path — no preceding Whisper detection already
        # announced the start, so mark it here instead.
        print(f"[LATENCY] backend={AI_BACKEND} measurement STARTED (typed prompt)", flush=True)
    await hard_reset_audio("run_backend_turn")

    content_list = []
    if last_frames:
        try:
            _, jpeg_bytes = last_frames[-1]
            if ENABLE_LOWLIGHT_ENHANCE:
                jpeg_bytes = await _enhance_lowlight_jpeg_async(jpeg_bytes)
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
        try:
            await _speak_and_broadcast(final_text)
        except Exception as e:
            print(f"[TTS] failed: {e}", flush=True)

    await _signal_stream_finished()

    # Measured here, after audio has actually been synthesized and pushed
    # out — not right after text generation — so this is audio-to-audio,
    # the same definition as gemini_live's total_latency.
    print(f"[LATENCY] backend={AI_BACKEND} measurement ENDED (audio delivered)", flush=True)
    _log_latency("total_latency", seconds=round(time.time() - (speech_end_ts or t0), 3))

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

# True exactly while the ESP32 mic is streaming audio to Gemini Live (set in
# ws_audio on START/STOP). Camera frames are only forwarded to Gemini while
# this is True — mirrors Google's own guidance to send video only during
# audio activity. Deliberately NOT gated on omni_conversation_active, since
# that flag is only updated post-hoc (after Gemini has already responded)
# and would always be stale by the time a frame needs to go out.
mic_streaming = False

# Model loading function
def load_navigation_models():
    """Load the models required for blind-path navigation."""
    global yolo_seg_model, obstacle_detector

    try:
        seg_model_path = os.getenv("BLIND_PATH_MODEL", r"C:\Users\Administrator\Desktop\rebuild1002\model\yolo-seg.pt")
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
        obstacle_model_path = os.getenv("OBSTACLE_MODEL", r"C:\Users\Administrator\Desktop\rebuild1002\model\yoloe-11l-seg.pt")
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
async def try_dispatch_command(user_text: str) -> bool:
    """Check user_text for special nav/system commands and run their side effects.

    Returns True if the text was handled as a command (or explicitly
    discarded) and should NOT also be treated as ordinary conversation;
    False if it's plain conversational text with no special meaning.

    (Renamed from start_ai_with_text_custom — it no longer decides whether
    to call the AI itself, since Gemini Live already streams a response to
    everything heard on the mic. Callers use the return value to decide
    whether to additionally forward text to Gemini, e.g. for typed prompts.)
    """
    global navigation_active, blind_path_navigator, cross_street_active, cross_street_navigator, orchestrator
    
    # In navigation or traffic-light detection mode, only specific words trigger omni dialogue
    if orchestrator:
        current_state = orchestrator.get_state()
        # If in navigation or traffic-light detection mode (not CHAT mode)
        if current_state not in ["CHAT", "IDLE"]:
            # Check whether the utterance is an allowed dialogue trigger keyword
            allowed_keywords = ["帮我看", "帮我看下", "帮我找", "找一下", "看看", "识别一下"]
            is_allowed_query = any(keyword in user_text for keyword in allowed_keywords)
            
            # Check whether the utterance is a navigation control command
            nav_control_keywords = ["开始过马路", "过马路结束", "开始导航", "盲道导航", "停止导航", "结束导航", 
                                   "检测红绿灯", "看红绿灯", "停止检测", "停止红绿灯"]
            is_nav_control = any(keyword in user_text for keyword in nav_control_keywords)
            
            # If neither an allowed query nor a navigation control command, discard
            if not is_allowed_query and not is_nav_control:
                if DEBUG:
                    mode_name = "Traffic light detection" if current_state == "TRAFFIC_LIGHT_DETECTION" else "Navigation"
                    print(f"[{mode_name} mode] Discarding non-dialogue audio: {user_text}")
                return True  # discard; do not enter omni
    
    # Check for street-crossing commands — use orchestrator to control
    if "开始过马路" in user_text or "帮我过马路" in user_text:
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
        return True
    
    if "过马路结束" in user_text or "结束过马路" in user_text:
        if orchestrator:
            orchestrator.stop_navigation()
            if DEBUG: print(f"[CROSS_STREET] Navigation stopped, state: {orchestrator.get_state()}")
            # Play stop voice prompt and broadcast to UI
            play_voice_text("Navigation stopped.")
            await ui_broadcast_final("[System] Street-crossing mode stopped")
        else:
            await ui_broadcast_final("[System] Navigation system not running")
        return True
    
    # Check for traffic-light detection command — mutually exclusive with blind-path navigation
    if "检测红绿灯" in user_text or "看红绿灯" in user_text:
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
        return True
    
    if "停止检测" in user_text or "停止红绿灯" in user_text:
        try:
            # Restore to dialogue (CHAT) mode
            if orchestrator:
                orchestrator.stop_navigation()  # return to CHAT mode
                if DEBUG: print(f"[TRAFFIC] Traffic-light detection stopped, restored to {orchestrator.get_state()} mode")

            await ui_broadcast_final("[System] Traffic-light detection stopped")
        except Exception as e:
            print(f"[TRAFFIC] Failed to stop traffic-light detection: {e}")
            await ui_broadcast_final(f"[System] Stop failed: {e}")
        return True
    
    # Check for navigation commands — use orchestrator to control
    if "开始导航" in user_text or "盲道导航" in user_text or "帮我导航" in user_text:
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
        return True
    
    if "停止导航" in user_text or "结束导航" in user_text:
        if orchestrator:
            orchestrator.stop_navigation()
            if DEBUG: print(f"[NAVIGATION] Navigation stopped, state: {orchestrator.get_state()}")
            await ui_broadcast_final("[System] Blind-path navigation stopped")
        else:
            await ui_broadcast_final("[System] Navigation system not running")
        return True

    nav_cmd_keywords = ["开始过马路", "过马路结束", "开始导航", "盲道导航", "停止导航", "结束导航", "立即通过", "现在通过", "继续"]
    if any(k in user_text for k in nav_cmd_keywords):
        if orchestrator:
            orchestrator.on_voice_command(user_text)
            await ui_broadcast_final("[System] Navigation mode updated")
        else:
            await ui_broadcast_final("[System] Navigation master not initialized")
        return True

    # Check for "帮我找/识别一下xxx" (help me find/identify xxx) command
    # Extended regex to support more keywords
    find_pattern = r"(?:^\s*帮我)?\s*找一下\s*(.+?)(?:。|！|？|$)"
    match = re.search(find_pattern, user_text)
        
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

            return True
    
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
        
        return True
    
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
    
    # Not a special command. If yolomedia is running, skip normal AI dialogue for now.
    if yolomedia_running:
        if DEBUG: print("[AI] YOLO media is running, skipping normal AI response", flush=True)
        return True

    # Plain conversation — not handled here; caller decides what to do with it
    # (the live mic-audio path just uses this for its side effects since
    # Gemini already replied; the typed-PROMPT path forwards it to Gemini).
    return False

# ========= Typed-prompt entry point =========
# gemini_live produces its own spoken reply via _on_audio; gemini_regular/qwen
# generate text then speak it via local TTS in run_backend_turn — all three
# ultimately produce audio output.
async def start_ai_with_text(user_text: str):
    """Route typed text to whichever backend is currently selected.

    Used for the /ws_audio 'PROMPT:' text path (device-initiated prompts
    that bypass audio ASR entirely). Ordinary mic audio doesn't go through
    this function — for AI_BACKEND=="gemini_live" it's streamed straight to
    gemini_live.send_audio() from ws_audio and the reply arrives via the
    Live callbacks; for the other backends, ws_audio streams mic audio into
    a TEXT-mode Live session used purely for ASR, and _on_turn_complete
    hands the transcribed text to run_backend_turn() the same way this
    function does for typed prompts.
    """
    await hard_reset_audio("start_ai_with_text")
    if AI_BACKEND == "gemini_live":
        _output_text_buf.clear()
        try:
            await gemini_live.send_text(user_text)
        except Exception as e:
            print(f"[Gemini Live] send_text failed: {e}", flush=True)
            try:
                await ui_broadcast_final(f"[AI] Error occurred: {e}")
            except Exception:
                pass
    else:
        await run_backend_turn(user_text)

# ---------- Page / Health ----------
@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/api/health", response_class=PlainTextResponse)
def health():
    return "OK"

@app.get("/api/backend")
def get_backend():
    return JSONResponse({"backend": AI_BACKEND})

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
        init = {"partial": current_partial, "finals": recent_finals[-10:], "backend": AI_BACKEND}
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
#
# Called via asyncio.create_task() from ws_audio rather than awaited inline,
# so the receive loop stays free to keep reading mic bytes while a turn is
# processed (previously awaiting this here stalled ws.receive() for the
# whole transcribe->generate->speak chain, which could take several
# seconds — long enough for the ESP32's send buffer to back up and the
# firmware to treat the connection as dead and reconnect).
#
# _turn_busy guards against two of these overlapping if you start talking
# again before the previous turn finishes — without it, two concurrent
# calls could both try to write TTS:START/chunks/TTS:END to the same ESP32
# socket at once and interleave their audio.
_turn_busy = False

def _spawn_background_task(coro, name: str = "task"):
    """asyncio.create_task(), but failures actually get printed.

    A fire-and-forget task whose coroutine raises an exception normally
    fails completely silently — no traceback, no error, nothing — unless
    something awaits it or checks task.exception(). Since
    _run_whisper_and_dispatch runs as a background task (so it doesn't
    block ws_audio's receive loop), any bug anywhere in that whole
    transcribe -> generate -> speak chain would otherwise just vanish,
    which is exactly what happened when total_latency stopped showing up
    with no error message. This wrapper makes sure that can't happen again.
    """
    task = asyncio.create_task(coro)
    def _on_done(t: asyncio.Task):
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            import traceback
            print(f"[BACKGROUND TASK ERROR] {name} failed: {exc}", flush=True)
            traceback.print_exception(type(exc), exc, exc.__traceback__)
    task.add_done_callback(_on_done)
    return task

async def _run_whisper_and_dispatch(buf: bytes) -> None:
    global _turn_busy
    if not buf or _whisper_model is None:
        return
    if _turn_busy:
        print("[WHISPER] Skipped — previous turn still in progress", flush=True)
        return
    _turn_busy = True
    speech_end_ts = time.time()  # ~when VAD detected end of speech and called this
    print(f"[LATENCY] backend={AI_BACKEND} measurement STARTED (speech end detected)", flush=True)
    try:
        samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
        loop    = asyncio.get_running_loop()
        result  = await loop.run_in_executor(
            None,
            lambda: _whisper_model.transcribe(samples, language="en", fp16=False)
        )
        # ASR time is folded into the single total_latency measurement
        # logged later in run_backend_turn(), not reported separately.
        text = (result.get("text") or "").strip()
        print(f"[WHISPER] {text}", flush=True)

        if text:
            await ui_broadcast_final("(user) " + text)

            if _has_hotword(text):
                async with interrupt_lock:
                    print(f"[HOTWORD] '{text}' -> full reset", flush=True)
                    await full_system_reset("Hotword interrupt")
            elif not is_playing_now():
                async with interrupt_lock:
                    handled = await try_dispatch_command(text)
                    if not handled:
                        await run_backend_turn(text, speech_end_ts=speech_end_ts)
    except Exception as e:
        print(f"[WHISPER] transcribe error: {e}", flush=True)
    finally:
        # This was missing — without it, _turn_busy stayed True forever
        # after the very first turn, permanently locking out every
        # subsequent attempt ("Skipped — previous turn still in progress").
        _turn_busy = False

# ---------- WebSocket: ESP32 audio entry (ASR uplink) ----------
#
# Branches by AI_BACKEND:
#   gemini_live               -> audio frames forwarded straight to
#                                 gemini_live.send_audio() in real time;
#                                 Gemini's own server-side VAD/turn-detection
#                                 and the _on_input_transcription / _on_audio /
#                                 _on_turn_complete callbacks handle the rest.
#   gemini_regular / qwen     -> local Whisper + RMS VAD (same as pre-Live),
#                                 buffering PCM16 and transcribing on silence.
#
@app.websocket("/ws_audio")
async def ws_audio(ws: WebSocket):
    global esp32_audio_ws, mic_streaming
    esp32_audio_ws = ws
    await ws.accept()
    print("[CONNECTED] Mic (ESP32 audio)")

    streaming: bool = False
    # VAD state — only used when AI_BACKEND != "gemini_live"
    pcm_buffer: Optional[bytearray] = None
    vad_silent_chunks: int = 0
    vad_speech_chunks: int = 0
    vad_speech_detected: bool = False

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
                    streaming = True
                    if AI_BACKEND == "gemini_live":
                        print("[MIC] Streaming to Gemini Live...")
                        mic_streaming = True
                    else:
                        print("[MIC] Listening — waiting for speech...")
                        pcm_buffer          = bytearray()
                        vad_silent_chunks   = 0
                        vad_speech_chunks   = 0
                        vad_speech_detected = False
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
                        # Backgrounded: transcribe -> generate -> speak can take
                        # multiple seconds, and awaiting it here would stop this
                        # loop from reading incoming mic bytes for that whole
                        # window — causing the ESP32's send buffer to back up,
                        # sendBinary() to fail, and the firmware to reconnect.
                        _spawn_background_task(_run_whisper_and_dispatch(buf), name="whisper_dispatch")

                elif raw.startswith("PROMPT:"):
                    # Device-initiated prompt (bypasses ASR entirely)
                    text = raw[len("PROMPT:"):].strip()
                    if text:
                        async with interrupt_lock:
                            handled = await try_dispatch_command(text)
                            if not handled:
                                await start_ai_with_text(text)
                        await ws.send_text("OK:PROMPT_ACCEPTED")
                    else:
                        await ws.send_text("ERR:EMPTY_PROMPT")

            elif "bytes" in msg and msg["bytes"] is not None:
                chunk = msg["bytes"]
                if not streaming:
                    continue

                if AI_BACKEND == "gemini_live":
                    try:
                        await gemini_live.send_audio(chunk)
                    except Exception as e:
                        print(f"[Gemini Live] send_audio failed: {e}", flush=True)
                    continue

                # ---- gemini_regular / qwen: RMS VAD over the PCM buffer ----
                if pcm_buffer is None:
                    continue
                pcm_buffer.extend(chunk)

                if len(pcm_buffer) > VAD_MAX_BUFFER_BYTES:
                    if not vad_speech_detected:
                        # Never triggered — this is stale noise/silence that
                        # was about to leak into whatever you say next. Drop it.
                        print("[MIC] Buffer cap hit with no speech detected — discarding stale audio", flush=True)
                        pcm_buffer          = bytearray()
                        vad_silent_chunks   = 0
                        vad_speech_chunks   = 0
                        vad_speech_detected = False
                        continue
                    else:
                        # Speech was detected but silence never followed
                        # (long continuous speech, or noisy trailing audio
                        # keeps resetting vad_silent_chunks) — force-flush
                        # what we have rather than let it grow indefinitely.
                        print("[MIC] Buffer cap hit mid-speech — forcing transcription", flush=True)
                        buf = bytes(pcm_buffer)
                        pcm_buffer          = bytearray()
                        vad_silent_chunks   = 0
                        vad_speech_chunks   = 0
                        vad_speech_detected = False
                        await ui_broadcast_partial("（Processing…）")
                        _spawn_background_task(_run_whisper_and_dispatch(buf), name="whisper_dispatch")
                        continue

                n = len(chunk)
                if n >= 2:
                    s = np.frombuffer(chunk[: n & ~1], dtype=np.int16).astype(np.float32)
                    s -= s.mean()  # strip DC offset from PDM mic before measuring energy
                    rms = float(np.sqrt(np.mean(s ** 2)))
                else:
                    rms = 0.0

                if DEBUG_VAD:
                    label = "SPEECH" if rms >= VAD_SILENCE_RMS else "silent"
                    print(
                        f"[VAD DEBUG] rms={rms:6.0f}  thresh={VAD_SILENCE_RMS}"
                        f"  -> {label}"
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
                            _spawn_background_task(_run_whisper_and_dispatch(buf), name="whisper_dispatch")
                    else:
                        # Leaky decay instead of a hard reset — a single quiet
                        # chunk (natural micro-pause, softer syllable) was
                        # previously enough to wipe out all progress toward
                        # confirming speech, making detection fragile whenever
                        # the threshold sits close to actual speaking volume.
                        vad_speech_chunks = max(0, vad_speech_chunks - 1)
                        vad_silent_chunks += 1
                        if vad_silent_chunks >= VAD_SILENCE_CHUNKS and len(pcm_buffer) > 0:
                            # Gone quiet again without ever confirming speech —
                            # this attempt failed (too quiet/too short/etc).
                            # Flush now instead of letting it sit in the buffer
                            # for up to VAD_MAX_BUFFER_SECONDS, where it would
                            # glue onto whatever you say on your next attempt.
                            print("[MIC] No speech confirmed, discarding buffered audio", flush=True)
                            pcm_buffer        = bytearray()
                            vad_silent_chunks = 0
                            vad_speech_chunks = 0

    except Exception as e:
        print(f"\n[WS ERROR] {e}")
    finally:
        streaming = False
        pcm_buffer = None
        mic_streaming = False
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        if esp32_audio_ws is ws:
            esp32_audio_ws = None
        print("[DISCONNECTED] Mic (ESP32 audio)")

# ---------- Non-blocking cv2 helpers ----------
# cv2.imdecode/imencode are synchronous, CPU-bound calls. Running them
# directly inside an async def (as the camera handler did before) blocks
# the whole event loop for their duration — including the coroutines that
# keep the Gemini Live websocket's keepalive pings answered and audio
# chunks flowing. Route them through a thread pool instead so a slow frame
# doesn't stall unrelated async work.
async def _cv2_imdecode_async(data: bytes) -> Optional["np.ndarray"]:
    loop = asyncio.get_running_loop()
    def _decode():
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None or bgr.size == 0:
                return None
            return bgr
        except Exception:
            return None
    return await loop.run_in_executor(None, _decode)

async def _cv2_imencode_async(img, quality: int):
    """Returns (ok, jpeg_bytes_or_None)."""
    loop = asyncio.get_running_loop()
    def _encode():
        try:
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            return (ok, enc.tobytes() if ok else None)
        except Exception:
            return (False, None)
    return await loop.run_in_executor(None, _encode)

# ---- Low-light enhancement for the Gemini-bound frame only ----
# Off by default risk: set False at any time to fall back to the exact
# behavior you have today (raw ESP32 JPEG bytes sent straight to Gemini).
# Only ever applied to the throttled ~2fps frame already destined for
# gemini_live.send_image() — never touches the navigation/YOLO path or the
# browser viewer stream, so it cannot affect anything else in the app.
ENABLE_LOWLIGHT_ENHANCE = False

# CLAHE object is expensive-ish to construct; reuse one instance.
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

def _enhance_lowlight_bgr(bgr) -> "np.ndarray":
    """CLAHE (adaptive local contrast) on the luminance channel + a gentle
    gamma lift. Cheap (a few ms on a typical frame size) and safe to run
    synchronously inside the executor thread alongside decode/encode."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    lab = cv2.merge((l, a, b))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Gentle gamma lift (<1.0 brightens) — skip if the frame is already bright
    # enough that lifting it would just wash out highlights.
    mean_l = float(np.mean(l))
    if mean_l < 110:
        gamma = 0.75
        inv_gamma = 1.0 / gamma
        table = (np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)])
                 .astype("uint8"))
        out = cv2.LUT(out, table)

    return out

async def _enhance_lowlight_jpeg_async(data: bytes, quality: int = 85) -> bytes:
    """Decode -> enhance -> re-encode a JPEG, off the event loop.
    Returns the original bytes unchanged on any failure (never raises)."""
    loop = asyncio.get_running_loop()
    def _run():
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return data
            enhanced = _enhance_lowlight_bgr(bgr)
            ok, enc = cv2.imencode(".jpg", enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            return enc.tobytes() if ok else data
        except Exception as e:
            print(f"[LOWLIGHT] enhance failed, using raw frame: {e}", flush=True)
            return data
    return await loop.run_in_executor(None, _run)

async def _encode_for_viewer_async(bgr, quality: int):
    """Same job as _cv2_imencode_async, but applies the low-light enhancement
    first when ENABLE_LOWLIGHT_ENHANCE is on. Used for every frame sent to
    /ws/viewer (the browser UI) so you can visually A/B it, same flag that
    gates the Gemini-bound frame. Falls back to encoding the unmodified
    frame if enhancement raises for any reason."""
    loop = asyncio.get_running_loop()
    def _run():
        img = bgr
        if ENABLE_LOWLIGHT_ENHANCE:
            try:
                img = _enhance_lowlight_bgr(bgr)
            except Exception as e:
                print(f"[LOWLIGHT] viewer enhance failed, using raw frame: {e}", flush=True)
                img = bgr
        try:
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            return (ok, enc.tobytes() if ok else None)
        except Exception:
            return (False, None)
    return await loop.run_in_executor(None, _run)

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
    frame_counter = 0

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                frame_counter += 1

                # Record the raw frame
                try:
                    sync_recorder.record_frame(data)
                except Exception as e:
                    if frame_counter % 100 == 0:  # avoid log spam
                        print(f"[RECORDER] Failed to record frame: {e}")
                
                try:
                    last_frames.append((time.time(), data))
                except Exception:
                    pass

                # Give Gemini Live visual context while the mic is actively
                # streaming (Google's own guidance: send video frames during
                # audio activity). NOTE: this used to gate on
                # omni_conversation_active, but that flag is only ever set
                # True post-hoc inside _on_turn_complete/try_dispatch_command
                # — by the time it flips True, the turn it was meant for is
                # already over, so no frame ever actually went out. Gating on
                # mic_streaming (set live in ws_audio on START/STOP) fixes that.
                # Throttled to ~2 fps since Live API input doesn't need full
                # camera framerate and this avoids saturating the session.
                if mic_streaming and frame_counter % 15 == 0:
                    try:
                        send_data = data
                        if ENABLE_LOWLIGHT_ENHANCE:
                            send_data = await _enhance_lowlight_jpeg_async(data)
                        await gemini_live.send_image(send_data)
                    except Exception as e:
                        print(f"[Gemini Live] send_image failed: {e}", flush=True)

                # Push to bridge_io (for use by yolomedia)
                bridge_io.push_raw_jpeg(data)
                
                # Unified decoding (off the event loop — see _cv2_imdecode_async)
                bgr = await _cv2_imdecode_async(data)
                if bgr is None and frame_counter % 30 == 0:
                    print(f"[JPEG] Decode failed: data length={len(data)}")

                # Hand off to the master state machine first (when item-search is not occupying the frame)
                # In item-search mode, skip navigation processing and let yolomedia take over the frame
                if orchestrator and not yolomedia_running and bgr is not None:
                    current_state = orchestrator.get_state()
                    
                    # Item-search mode: skip frame processing and wait for yolomedia to send processed frames
                    if current_state == "ITEM_SEARCH":
                        # In item-search mode, if yolomedia has not yet started sending frames, show the raw frame
                        if not yolomedia_sending_frames and camera_viewers:
                            ok, jpeg_data = await _encode_for_viewer_async(bgr, JPEG_QUALITY)
                            if ok:
                                dead = []
                                for viewer_ws in list(camera_viewers):
                                    try:
                                        await viewer_ws.send_bytes(jpeg_data)
                                    except Exception:
                                        dead.append(viewer_ws)
                                for d in dead:
                                    camera_viewers.discard(d)
                        continue  # skip subsequent navigation processing
                    
                    out_img = bgr
                    try:
                        # Check whether we are in traffic-light detection mode
                        if current_state == "TRAFFIC_LIGHT_DETECTION":
                            # Traffic-light detection mode: process directly in the main thread to avoid dropped frames
                            import trafficlight_detection
                            result = trafficlight_detection.process_single_frame(bgr, ui_broadcast_callback=ui_broadcast_final)
                            out_img = result['vis_image'] if result['vis_image'] is not None else bgr
                        else:
                            # Other modes: normal navigation processing
                            res = orchestrator.process_frame(bgr)

                            # Voice guidance (throttled internally)
                            # Note: during omni dialogue the mode is CHAT, so no navigation voice is generated
                            if res.guidance_text:
                                try:
                                    # Play voice first, then broadcast to UI
                                    play_voice_text(res.guidance_text)
                                    await ui_broadcast_final(f"[NAV] {res.guidance_text}")
                                except Exception:
                                    pass

                            # Output image
                            out_img = res.annotated_image if res.annotated_image is not None else bgr
                    except Exception as e:
                        if frame_counter % 100 == 0:
                            print(f"[NAV MASTER] Error processing frame: {e}")

                    # Broadcast the image
                    if camera_viewers and out_img is not None:
                        ok, jpeg_data = await _encode_for_viewer_async(out_img, JPEG_QUALITY)
                        if ok:
                            dead = []
                            for viewer_ws in list(camera_viewers):
                                try:
                                    await viewer_ws.send_bytes(jpeg_data)
                                except Exception:
                                    dead.append(viewer_ws)
                            for d in dead:
                                camera_viewers.discard(d)
                    # Handed off to state machine; proceed to next frame
                    continue

                # [Fallback] Item-search is occupying the frame or decoding failed; fall back to the raw frame
                if not yolomedia_sending_frames and camera_viewers:
                    try:
                        if bgr is None:
                            bgr = await _cv2_imdecode_async(data)
                        if bgr is not None:
                            ok, jpeg_data = await _encode_for_viewer_async(bgr, JPEG_QUALITY)
                            if ok:
                                dead = []
                                for viewer_ws in list(camera_viewers):
                                    try:
                                        await viewer_ws.send_bytes(jpeg_data)
                                    except Exception:
                                        dead.append(viewer_ws)
                                for ws in dead:
                                    camera_viewers.discard(ws)
                    except Exception as e:
                        print(f"[CAMERA] Broadcast error: {e}")

            elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[CAMERA ERROR] {e}")
    finally:
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

@app.on_event("startup")
async def startup_gemini():
    if AI_BACKEND == "gemini_live":
        await gemini_live.connect(response_modality="AUDIO")
    # gemini_regular/qwen don't touch Gemini Live at all — they use local
    # Whisper ASR instead (see the AI_BACKEND != "gemini_live" branch above).

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
    await gemini_live.disconnect()
    
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