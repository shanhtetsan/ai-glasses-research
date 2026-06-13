# audio_player.py
# Live English TTS via macOS 'say' → PCM8k → broadcast_pcm16_realtime.
# Replaces the pre-recorded Chinese .wav lookup that was here before.

import os
import asyncio
import threading
import queue
import time
import tempfile
import wave
import audioop

from audio_stream import broadcast_pcm16_realtime

# ---- Lazy recorder import (avoid circular import) ----
_recorder_imported = False
_sync_recorder = None

def _get_recorder():
    global _recorder_imported, _sync_recorder
    if not _recorder_imported:
        try:
            import sync_recorder as sr
            _sync_recorder = sr
            _recorder_imported = True
        except Exception as e:
            print(f"[AUDIO] Could not import recorder: {e}")
            _recorder_imported = True
    return _sync_recorder

# ---- Voice throttle ----
_last_voice_time = 0.0
_last_voice_text = ""
_voice_cooldown = 1.0  # minimum seconds between identical phrases

# ---- Playback queue and worker ----
_audio_queue = queue.PriorityQueue(maxsize=10)
_audio_priority = 0
_worker_thread = None
_worker_loop = None
_is_playing = False
_playing_lock = threading.Lock()
_initialized = False
_last_play_ts = 0.0

# ---- Voice priority definitions (kept for external code that may reference them) ----
VOICE_PRIORITY = {
    'obstacle': 100,
    'direction': 50,
    'straight': 10,
    'other': 30,
}


# ---- TTS: macOS 'say' → 8 kHz PCM16 ----
async def _say_to_pcm8k(text: str) -> bytes:
    """
    Use macOS built-in TTS ('say') to produce 8 kHz PCM16 suitable for
    broadcast_pcm16_realtime / the ESP32 8 kHz downlink.

    Steps:
      1. say   → AIFF  (native macOS TTS)
      2. afconvert → 16-bit PCM WAV
      3. audioop.ratecv → downsample to 8 kHz in Python
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        aiff_path = os.path.join(tmpdir, "out.aiff")
        wav_path  = os.path.join(tmpdir, "out.wav")

        p1 = await asyncio.create_subprocess_exec(
            "say", "-o", aiff_path, "--", text,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await p1.wait()

        p2 = await asyncio.create_subprocess_exec(
            "afconvert", aiff_path, wav_path, "-f", "WAVE", "-d", "LEI16",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await p2.wait()

        with wave.open(wav_path, "rb") as w:
            ch  = w.getnchannels()
            sw  = w.getsampwidth()
            fr  = w.getframerate()
            pcm = w.readframes(w.getnframes())

        if ch == 2:
            pcm = audioop.tomono(pcm, sw, 1, 0)
        if fr != 8000:
            pcm, _ = audioop.ratecv(pcm, sw, 1, fr, 8000, None)
        return pcm


# ---- Audio broadcast with lead/tail silence ----
async def _broadcast_audio_optimized(pcm_data: bytes):
    global _last_play_ts, _is_playing
    try:
        with _playing_lock:
            _is_playing = True

        now = time.monotonic()
        idle_sec = now - (_last_play_ts or now)
        lead_ms = 160 if idle_sec > 3.0 else 60
        tail_ms = 40

        lead_silence = b'\x00' * (lead_ms * 8000 * 2 // 1000)
        tail_silence = b'\x00' * (tail_ms * 8000 * 2 // 1000)
        full_audio = lead_silence + pcm_data + tail_silence

        await broadcast_pcm16_realtime(full_audio)
        _last_play_ts = time.monotonic()
    except Exception as e:
        print(f"[AUDIO] Failed to broadcast audio: {e}")
    finally:
        with _playing_lock:
            _is_playing = False


# ---- Worker thread ----
def _audio_worker():
    global _worker_loop

    _worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_worker_loop)

    async def process_queue():
        while True:
            try:
                priority_data = await asyncio.get_event_loop().run_in_executor(
                    None, _audio_queue.get, True
                )
                if priority_data is None:
                    break
                if isinstance(priority_data, tuple) and len(priority_data) == 2:
                    _, payload = priority_data
                else:
                    payload = priority_data

                if isinstance(payload, str):
                    # Live TTS path
                    try:
                        pcm_data = await _say_to_pcm8k(payload)
                        if pcm_data:
                            await _broadcast_audio_optimized(pcm_data)
                    except Exception as e:
                        print(f"[AUDIO] TTS failed for '{payload}': {e}")
                elif isinstance(payload, (bytes, bytearray)):
                    # Legacy raw-PCM path (kept for backward compat)
                    await _broadcast_audio_optimized(bytes(payload))
            except Exception as e:
                print(f"[AUDIO] Worker thread error: {e}")

    _worker_loop.run_until_complete(process_queue())


# ---- Public API ----
def initialize_audio_system():
    global _initialized, _worker_thread, _last_play_ts
    if _initialized:
        return
    _worker_thread = threading.Thread(target=_audio_worker, daemon=True)
    _worker_thread.start()
    _initialized = True
    _last_play_ts = 0.0
    print("[AUDIO] Audio system initialized (live TTS mode)")


def play_voice_text(text: str):
    """
    Queue an English phrase for live TTS playback.
    Identical throttle, priority, and queue-management logic as before —
    only the voice production mechanism has changed (say → PCM, no WAV files).
    """
    global _last_voice_time, _last_voice_text, _audio_queue, _audio_priority

    if not text:
        return
    if not _initialized:
        initialize_audio_system()

    current_time = time.time()
    if text == _last_voice_text and current_time - _last_voice_time < _voice_cooldown:
        return  # throttle: same phrase too soon

    _last_voice_text = text
    _last_voice_time = current_time

    # Real-time queue management: keep backlog minimal
    queue_size = _audio_queue.qsize()
    with _playing_lock:
        currently_playing = _is_playing

    if queue_size > 0 and not currently_playing:
        _audio_queue = queue.PriorityQueue(maxsize=10)
    elif queue_size > 1 and currently_playing:
        _audio_queue = queue.PriorityQueue(maxsize=10)

    try:
        _audio_priority += 1
        _audio_queue.put_nowait((_audio_priority, text))
    except queue.Full:
        print(f"[AUDIO] Queue full, discarding: {text}")


# Compatibility alias
play_audio_on_esp32 = play_voice_text
