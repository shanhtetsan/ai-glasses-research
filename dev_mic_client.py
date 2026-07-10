#!/usr/bin/env python3
"""Development ESP32 microphone simulator.

Records from the laptop microphone and sends audio to /ws_audio using the same
START -> PCM16 frames -> STOP protocol as the ESP32 firmware.
"""

import argparse
import asyncio
import audioop
import queue
import sys
import wave
from pathlib import Path


SAMPLE_RATE = 16000
CHANNELS = 1


def load_websockets():
    try:
        import websockets
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: websockets. Install with `pip install -r requirements.txt`."
        ) from exc
    return websockets


def list_devices() -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sounddevice. Install with `pip install -r requirements.txt`."
        ) from exc
    print(sd.query_devices())


def wav_to_pcm16_16k(path: Path) -> bytes:
    """Load a PCM WAV file and convert it to mono 16 kHz 16-bit PCM."""
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        frames = audioop.lin2lin(frames, sample_width, 2)
        sample_width = 2

    if channels != 1:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)

    if frame_rate != SAMPLE_RATE:
        frames, _ = audioop.ratecv(frames, sample_width, 1, frame_rate, SAMPLE_RATE, None)

    return frames


def pcm_stats(pcm: bytes) -> tuple[float, int]:
    if not pcm:
        return 0.0, 0
    return audioop.rms(pcm, 2), audioop.max(pcm, 2)


def apply_gain(pcm: bytes, gain: float) -> bytes:
    if gain == 1.0:
        return pcm
    return audioop.mul(pcm, 2, gain)


async def send_pcm_chunks(ws, pcm: bytes, chunk_ms: int, realtime: bool = True) -> None:
    bytes_per_chunk = SAMPLE_RATE * chunk_ms // 1000 * 2
    for offset in range(0, len(pcm), bytes_per_chunk):
        await ws.send(pcm[offset: offset + bytes_per_chunk])
        if realtime:
            await asyncio.sleep(chunk_ms / 1000)


async def record_and_send(args: argparse.Namespace) -> None:
    websockets = load_websockets()
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            "Missing microphone dependency. Install with `pip install -r requirements.txt`, "
            "or test a WAV file with `python dev_mic_client.py --wav test_7b.wav`."
        ) from exc

    audio_queue: queue.Queue[bytes | None] = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[dev-mic] {status}", file=sys.stderr)
        pcm16 = np.clip(indata[:, 0], -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        audio_queue.put(pcm16.tobytes())

    print(f"[dev-mic] connecting to {args.url}")
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send("START")
        try:
            reply = await asyncio.wait_for(ws.recv(), timeout=2.0)
            print(f"[dev-mic] {reply}")
        except asyncio.TimeoutError:
            pass

        print(f"[dev-mic] recording {args.seconds:.1f}s; speak now")
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=args.device,
            blocksize=int(SAMPLE_RATE * args.chunk_ms / 1000),
            callback=callback,
        ):
            end_at = asyncio.get_running_loop().time() + args.seconds
            while asyncio.get_running_loop().time() < end_at:
                try:
                    chunk = audio_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.005)
                    continue
                if chunk:
                    await ws.send(chunk)

        await ws.send("STOP")
        try:
            reply = await asyncio.wait_for(ws.recv(), timeout=2.0)
            print(f"[dev-mic] {reply}")
        except asyncio.TimeoutError:
            pass

    print("[dev-mic] done; check the web UI or server log for Whisper text")


async def send_wav_file(args: argparse.Namespace) -> None:
    websockets = load_websockets()
    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise SystemExit(f"WAV file not found: {wav_path}")

    pcm = wav_to_pcm16_16k(wav_path)
    pcm = apply_gain(pcm, args.gain)
    duration = len(pcm) / (SAMPLE_RATE * 2)
    rms, peak = pcm_stats(pcm)
    print(f"[dev-mic] audio {duration:.2f}s rms={rms:.0f} peak={peak}")
    print(f"[dev-mic] connecting to {args.url}")
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send("START")
        try:
            reply = await asyncio.wait_for(ws.recv(), timeout=2.0)
            print(f"[dev-mic] {reply}")
        except asyncio.TimeoutError:
            pass

        print(f"[dev-mic] sending {wav_path} as 16 kHz PCM16")
        await send_pcm_chunks(ws, pcm, args.chunk_ms, realtime=not args.no_realtime)

        await ws.send("STOP")
        try:
            reply = await asyncio.wait_for(ws.recv(), timeout=2.0)
            print(f"[dev-mic] {reply}")
        except asyncio.TimeoutError:
            pass

    print("[dev-mic] done; check the web UI or server log for Whisper text")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Send laptop mic audio to the AI glasses server.")
    parser.add_argument("--url", default="ws://127.0.0.1:8081/ws_audio")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--chunk-ms", type=int, default=20)
    parser.add_argument("--device", type=int, default=None, help="Input device index.")
    parser.add_argument("--wav", help="Send a local WAV file instead of recording from the microphone.")
    parser.add_argument("--gain", type=float, default=1.0, help="Gain multiplier for --wav audio.")
    parser.add_argument("--no-realtime", action="store_true", help="Send WAV chunks without real-time pacing.")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.wav:
        await send_wav_file(args)
    else:
        await record_and_send(args)


if __name__ == "__main__":
    asyncio.run(main())
