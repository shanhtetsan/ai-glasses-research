"""Small, dependency-free stability primitives shared by the backend and tests."""
from __future__ import annotations

import asyncio
import copy
import inspect
import math
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

MSG_TYPE_CAM = 0x01
MSG_TYPE_THERMAL = 0x02
MSG_TYPE_IMU = 0x03
MSG_TYPE_STATUS = 0x04
THERMAL_PAYLOAD_BYTES = 24 * 32 * 4
MAX_JPEG_PAYLOAD_BYTES = 300 * 1024
IMU_STRUCT = struct.Struct("<IIffffff")
STATUS_STRUCT = struct.Struct("<IIII")


@dataclass(frozen=True)
class LatestFrame:
    data: bytes | None = None
    timestamp: float = 0.0
    sequence: int = 0


class LatestFrameStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = LatestFrame()

    def update(self, data: bytes, timestamp: Optional[float] = None) -> LatestFrame:
        now = time.monotonic() if timestamp is None else timestamp
        with self._lock:
            self._value = LatestFrame(bytes(data), now, self._value.sequence + 1)
            return self._value

    def snapshot(self) -> LatestFrame:
        with self._lock:
            return self._value


class LatencyTracker:
    """Bounded, monotonic-clock-only conversation latency tracking."""

    TIMESTAMP_FIELDS = (
        "turn_created",
        "first_microphone_chunk_received",
        "last_microphone_chunk_received",
        "speech_end_detected",
        "first_input_transcription",
        "first_gemini_audio_received",
        "first_tts_chunk_queued",
        "tts_start_sent",
        "first_binary_tts_chunk_sent",
        "final_binary_tts_chunk_sent",
        "turn_complete",
        "interrupted",
    )

    def __init__(
        self,
        history_size: int = 200,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        self.history_size = history_size
        self.clock = clock
        self._lock = threading.Lock()
        self._history: deque[dict] = deque(maxlen=history_size)
        self._active: dict | None = None
        self._next_turn_id = 1
        self._completed_count = 0
        self._interrupted_count = 0
        self._latest_rtt = {
            "latest_ms": None,
            "average_ms": None,
            "minimum_ms": None,
            "maximum_ms": None,
            "p95_ms": None,
        }
        self._latest_device_metrics: dict | None = None

    @property
    def active_turn_id(self) -> int | None:
        with self._lock:
            return None if self._active is None else self._active["turn_id"]

    def _new_turn(self, turn_id: int, now_ns: int) -> dict:
        timestamps = {name: None for name in self.TIMESTAMP_FIELDS}
        timestamps["turn_created"] = now_ns
        return {
            "turn_id": turn_id,
            "status": "active",
            "timestamps_ns": timestamps,
            "tts_chunks": 0,
            "tts_bytes": 0,
            "max_queue_depth": 0,
            "device_metrics": None,
        }

    def start_turn(
        self,
        turn_id: int | None = None,
        now_ns: int | None = None,
    ) -> int:
        now = self.clock() if now_ns is None else now_ns
        with self._lock:
            if self._active is not None:
                return self._active["turn_id"]
            if turn_id is None or turn_id <= 0:
                turn_id = self._next_turn_id
            self._next_turn_id = max(self._next_turn_id, turn_id + 1)
            self._active = self._new_turn(turn_id, now)
            return turn_id

    def ensure_turn(self, now_ns: int | None = None) -> int:
        active = self.active_turn_id
        return active if active is not None else self.start_turn(now_ns=now_ns)

    def mark(
        self,
        event: str,
        turn_id: int | None = None,
        now_ns: int | None = None,
        first_only: bool = True,
    ) -> bool:
        if event not in self.TIMESTAMP_FIELDS:
            raise ValueError(f"unknown latency event: {event}")
        now = self.clock() if now_ns is None else now_ns
        with self._lock:
            if self._active is None:
                return False
            if turn_id is not None and self._active["turn_id"] != turn_id:
                return False
            timestamps = self._active["timestamps_ns"]
            if first_only and timestamps[event] is not None:
                return False
            timestamps[event] = now
            return True

    def mark_microphone_chunk(
        self,
        turn_id: int | None = None,
        now_ns: int | None = None,
    ) -> bool:
        now = self.clock() if now_ns is None else now_ns
        with self._lock:
            if self._active is None:
                return False
            if turn_id is not None and self._active["turn_id"] != turn_id:
                return False
            timestamps = self._active["timestamps_ns"]
            if timestamps["speech_end_detected"] is not None:
                return False
            if timestamps["first_microphone_chunk_received"] is None:
                timestamps["first_microphone_chunk_received"] = now
            timestamps["last_microphone_chunk_received"] = now
            return True

    def mark_tts_queued(self, queue_depth: int, turn_id: int | None = None) -> bool:
        now = self.clock()
        with self._lock:
            if self._active is None:
                return False
            if turn_id is not None and self._active["turn_id"] != turn_id:
                return False
            timestamps = self._active["timestamps_ns"]
            if timestamps["first_tts_chunk_queued"] is None:
                timestamps["first_tts_chunk_queued"] = now
            self._active["max_queue_depth"] = max(
                self._active["max_queue_depth"], queue_depth
            )
            return True

    def mark_tts_sent(self, byte_count: int, turn_id: int | None = None) -> bool:
        now = self.clock()
        with self._lock:
            if self._active is None:
                return False
            if turn_id is not None and self._active["turn_id"] != turn_id:
                return False
            timestamps = self._active["timestamps_ns"]
            if timestamps["first_binary_tts_chunk_sent"] is None:
                timestamps["first_binary_tts_chunk_sent"] = now
            timestamps["final_binary_tts_chunk_sent"] = now
            self._active["tts_chunks"] += 1
            self._active["tts_bytes"] += max(0, byte_count)
            return True

    @staticmethod
    def _delta_ms(timestamps: dict, start: str, end: str) -> float | None:
        start_ns = timestamps.get(start)
        end_ns = timestamps.get(end)
        if start_ns is None or end_ns is None or end_ns < start_ns:
            return None
        return round((end_ns - start_ns) / 1_000_000.0, 3)

    def _with_derived(self, turn: dict | None) -> dict | None:
        if turn is None:
            return None
        result = copy.deepcopy(turn)
        timestamps = result["timestamps_ns"]
        terminal = "interrupted" if result["status"] == "interrupted" else "turn_complete"
        result.update({
            "speech_end_to_first_gemini_audio_ms": self._delta_ms(
                timestamps, "speech_end_detected", "first_gemini_audio_received"
            ),
            "speech_end_to_first_tts_send_ms": self._delta_ms(
                timestamps, "speech_end_detected", "first_binary_tts_chunk_sent"
            ),
            "first_mic_to_first_gemini_audio_ms": self._delta_ms(
                timestamps, "first_microphone_chunk_received", "first_gemini_audio_received"
            ),
            "gemini_audio_to_first_tts_send_ms": self._delta_ms(
                timestamps, "first_gemini_audio_received", "first_binary_tts_chunk_sent"
            ),
            "backend_turn_total_ms": self._delta_ms(
                timestamps, "turn_created", terminal
            ),
            "speech_end_to_first_i2s_ms": (
                None if result["device_metrics"] is None
                else result["device_metrics"].get("speech_end_to_first_i2s_ms")
            ),
        })
        return result

    def finish(self, status: str, turn_id: int | None = None) -> dict | None:
        if status not in ("completed", "interrupted"):
            raise ValueError("status must be completed or interrupted")
        now = self.clock()
        with self._lock:
            if self._active is None:
                return None
            if turn_id is not None and self._active["turn_id"] != turn_id:
                return None
            timestamps = self._active["timestamps_ns"]
            terminal = "turn_complete" if status == "completed" else "interrupted"
            if timestamps[terminal] is None:
                timestamps[terminal] = now
            self._active["status"] = status
            finished = self._active
            self._history.append(finished)
            self._active = None
            if status == "completed":
                self._completed_count += 1
            else:
                self._interrupted_count += 1
            return self._with_derived(finished)

    def update_device_latency(self, turn_id: int, latency_ms: float) -> bool:
        metrics = {
            "turn_id": turn_id,
            "speech_end_to_first_i2s_ms": round(max(0.0, latency_ms), 3),
        }
        with self._lock:
            target = None
            if self._active is not None and self._active["turn_id"] == turn_id:
                target = self._active
            else:
                for turn in reversed(self._history):
                    if turn["turn_id"] == turn_id:
                        target = turn
                        break
            if target is None:
                return False
            target["device_metrics"] = metrics
            self._latest_device_metrics = dict(metrics)
            return True

    def update_rtt(
        self,
        latest_ms: float,
        average_ms: float,
        minimum_ms: float,
        maximum_ms: float,
        p95_ms: float,
    ) -> None:
        with self._lock:
            self._latest_rtt = {
                "latest_ms": round(max(0.0, latest_ms), 3),
                "average_ms": round(max(0.0, average_ms), 3),
                "minimum_ms": round(max(0.0, minimum_ms), 3),
                "maximum_ms": round(max(0.0, maximum_ms), 3),
                "p95_ms": round(max(0.0, p95_ms), 3),
            }

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return round(ordered[0], 3)
        position = (len(ordered) - 1) * percentile
        lower = math.floor(position)
        upper = math.ceil(position)
        value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
        return round(value, 3)

    @staticmethod
    def _display_metric(value_ms: float | None) -> dict:
        if value_ms is None:
            return {"value_ms": None, "text": "--", "color": "neutral"}
        color = "green" if value_ms < 2500 else ("yellow" if value_ms <= 4000 else "red")
        return {
            "value_ms": value_ms,
            "text": f"{value_ms:.0f} ms",
            "color": color,
        }

    def snapshot(self) -> dict:
        with self._lock:
            history = list(self._history)
            active = self._with_derived(self._active)
            latest_completed = next(
                (self._with_derived(turn) for turn in reversed(history)
                 if turn["status"] == "completed"),
                None,
            )
            latest_backend = (
                self._with_derived(history[-1]) if history else active
            )
            perceived = [
                float(turn["device_metrics"]["speech_end_to_first_i2s_ms"])
                for turn in history
                if turn["status"] == "completed" and turn["device_metrics"] is not None
            ]
            median_ms = self._percentile(perceived, 0.50)
            p90_ms = self._percentile(perceived, 0.90)
            p95_ms = self._percentile(perceived, 0.95)
            display_turn = active or latest_backend
            display = {
                "current_turn": None if active is None else active["turn_id"],
                "status": "idle" if display_turn is None else display_turn["status"],
                "network_rtt": self._display_metric(self._latest_rtt["latest_ms"]),
                "speech_end_to_gemini": self._display_metric(
                    None if display_turn is None
                    else display_turn["speech_end_to_first_gemini_audio_ms"]
                ),
                "speech_end_to_first_tts": self._display_metric(
                    None if display_turn is None
                    else display_turn["speech_end_to_first_tts_send_ms"]
                ),
                "backend_total": self._display_metric(
                    None if display_turn is None
                    else display_turn["backend_turn_total_ms"]
                ),
                "device_end_to_end": self._display_metric(
                    None if display_turn is None
                    else display_turn["speech_end_to_first_i2s_ms"]
                ),
                "median": self._display_metric(median_ms),
                "p95": self._display_metric(p95_ms),
            }
            return {
                "latest_completed_turn": latest_completed,
                "current_active_turn": active,
                "latest_rtt": dict(self._latest_rtt),
                "rolling_median_ms": median_ms,
                "p90_ms": p90_ms,
                "p95_ms": p95_ms,
                "interrupted_count": self._interrupted_count,
                "completed_count": self._completed_count,
                "history_size": len(history),
                "history_limit": self.history_size,
                "latest_backend_metrics": latest_backend,
                "latest_device_metrics": (
                    None if self._latest_device_metrics is None
                    else dict(self._latest_device_metrics)
                ),
                "display": display,
            }


@dataclass(frozen=True)
class SynchronizedFramePair:
    rgb: LatestFrame
    thermal: LatestFrame
    delta_sec: float


class RgbThermalSynchronizer:
    """Keep short RGB/thermal histories and expose the closest valid pair.

    Timestamps are backend receipt times. This is intentionally bounded and
    dependency-free so it remains safe in stability mode. Firmware capture
    timestamps can replace receipt timestamps later without changing callers.
    """

    def __init__(self, history_size: int = 4, max_delta_sec: float = 0.25) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        if max_delta_sec <= 0:
            raise ValueError("max_delta_sec must be positive")
        self.history_size = history_size
        self.max_delta_sec = max_delta_sec
        self._lock = threading.Lock()
        self._rgb: deque[LatestFrame] = deque(maxlen=history_size)
        self._thermal: deque[LatestFrame] = deque(maxlen=history_size)
        self._rgb_sequence = 0
        self._thermal_sequence = 0

    def add_rgb(self, data: bytes, timestamp: Optional[float] = None) -> LatestFrame:
        now = time.monotonic() if timestamp is None else timestamp
        with self._lock:
            self._rgb_sequence += 1
            frame = LatestFrame(bytes(data), now, self._rgb_sequence)
            self._rgb.append(frame)
            return frame

    def add_thermal(self, data: bytes, timestamp: Optional[float] = None) -> LatestFrame:
        now = time.monotonic() if timestamp is None else timestamp
        with self._lock:
            self._thermal_sequence += 1
            frame = LatestFrame(bytes(data), now, self._thermal_sequence)
            self._thermal.append(frame)
            return frame

    def snapshot(self, max_delta_sec: Optional[float] = None) -> SynchronizedFramePair | None:
        threshold = self.max_delta_sec if max_delta_sec is None else max_delta_sec
        with self._lock:
            if not self._rgb or not self._thermal:
                return None
            best_rgb: LatestFrame | None = None
            best_thermal: LatestFrame | None = None
            best_delta = float("inf")
            for rgb in self._rgb:
                for thermal in self._thermal:
                    delta = abs(rgb.timestamp - thermal.timestamp)
                    if delta < best_delta:
                        best_rgb, best_thermal, best_delta = rgb, thermal, delta
            if best_rgb is None or best_thermal is None or best_delta > threshold:
                return None
            return SynchronizedFramePair(best_rgb, best_thermal, best_delta)

    def health(self) -> dict:
        with self._lock:
            return {
                "rgb_history_depth": len(self._rgb),
                "thermal_history_depth": len(self._thermal),
                "max_pair_delta_ms": self.max_delta_sec * 1000.0,
            }


def parse_sensor_message(data: bytes):
    if not data:
        raise ValueError("missing message prefix")
    msg_type, payload = data[0], data[1:]
    if msg_type == MSG_TYPE_CAM:
        if (len(payload) < 4 or len(payload) > MAX_JPEG_PAYLOAD_BYTES
                or not payload.startswith(b"\xff\xd8")):
            raise ValueError("invalid JPEG payload")
        return msg_type, payload
    if msg_type == MSG_TYPE_THERMAL:
        if len(payload) != THERMAL_PAYLOAD_BYTES:
            raise ValueError("invalid thermal payload length")
        return msg_type, payload
    if msg_type == MSG_TYPE_IMU:
        if len(payload) != IMU_STRUCT.size:
            raise ValueError("invalid IMU payload length")
        seq, uptime_ms, ax, ay, az, gx, gy, gz = IMU_STRUCT.unpack(payload)
        if not all(math.isfinite(value) for value in (ax, ay, az, gx, gy, gz)):
            raise ValueError("non-finite IMU value")
        return msg_type, {
            "sequence": seq,
            "uptime_ms": uptime_ms,
            "accel": {"x": ax, "y": ay, "z": az, "units": "m/s^2"},
            "gyro": {"x": gx, "y": gy, "z": gz, "units": "deg/s"},
        }
    if msg_type == MSG_TYPE_STATUS:
        if len(payload) != STATUS_STRUCT.size:
            raise ValueError("invalid status payload length")
        uptime_ms, free_heap, largest_internal, free_psram = STATUS_STRUCT.unpack(payload)
        return msg_type, {
            "uptime_ms": uptime_ms,
            "free_heap": free_heap,
            "largest_internal": largest_internal,
            "free_psram": free_psram,
        }
    raise ValueError("unknown message prefix")


class VisionController:
    def __init__(
        self,
        frames: LatestFrameStore,
        sender: Callable[[bytes], Awaitable[object]],
        max_age_sec: float = 3.0,
        min_interval_sec: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.frames = frames
        self.sender = sender
        self.max_age_sec = max_age_sec
        self.min_interval_sec = min_interval_sec
        self.clock = clock
        self.metrics = {
            "vision_requests": 0,
            "vision_successes": 0,
            "vision_failures": 0,
            "vision_rate_limited": 0,
            "vision_no_recent_frame": 0,
            "last_vision_request_time": 0.0,
        }
        self._lock = asyncio.Lock()

    async def request(self, reason: str) -> dict:
        async with self._lock:
            now = self.clock()
            self.metrics["vision_requests"] += 1
            last = self.metrics["last_vision_request_time"]
            if last and now - last < self.min_interval_sec:
                self.metrics["vision_rate_limited"] += 1
                return {"ok": False, "error": "rate_limited", "reason": reason}
            self.metrics["last_vision_request_time"] = now
            frame = self.frames.snapshot()
            if frame.data is None or now - frame.timestamp > self.max_age_sec:
                self.metrics["vision_no_recent_frame"] += 1
                return {"ok": False, "error": "no_recent_frame", "reason": reason}
            try:
                await self.sender(frame.data)
            except Exception as exc:
                self.metrics["vision_failures"] += 1
                return {"ok": False, "error": type(exc).__name__, "reason": reason}
            self.metrics["vision_successes"] += 1
            return {"ok": True, "sequence": frame.sequence, "reason": reason}


class RecordingPipeline:
    def __init__(
        self,
        writer: Callable[[bytes], object],
        maxsize: int = 2,
        on_failure: Callable[[], object] | None = None,
    ) -> None:
        self.writer = writer
        self.on_failure = on_failure
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=maxsize)
        self.metrics = {
            "recording_active": False,
            "recording_frames_received": 0,
            "recording_frames_written": 0,
            "recording_frames_dropped": 0,
            "recording_errors": 0,
        }
        self._task: asyncio.Task | None = None

    def enqueue_latest(self, frame: bytes) -> None:
        if not self.metrics["recording_active"]:
            return
        self.metrics["recording_frames_received"] += 1
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.metrics["recording_frames_dropped"] += 1
            except asyncio.QueueEmpty:
                pass
        try:
            self.queue.put_nowait(bytes(frame))
        except asyncio.QueueFull:
            self.metrics["recording_frames_dropped"] += 1

    async def _run(self) -> None:
        while True:
            frame = await self.queue.get()
            try:
                if inspect.iscoroutinefunction(self.writer):
                    result = await self.writer(frame)
                else:
                    result = await asyncio.to_thread(self.writer, frame)
                if result is False:
                    raise RuntimeError("recording writer rejected payload")
                self.metrics["recording_frames_written"] += 1
            except Exception:
                self.metrics["recording_errors"] += 1
                self.metrics["recording_active"] = False
                if self.on_failure is not None:
                    try:
                        await asyncio.to_thread(self.on_failure)
                    except Exception:
                        pass
                return

    def start(self) -> None:
        self.metrics["recording_active"] = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self.metrics["recording_active"] = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def health(self) -> dict:
        return {**self.metrics, "recording_queue_depth": self.queue.qsize()}
