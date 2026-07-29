"""Small, dependency-free stability primitives shared by the backend and tests."""
from __future__ import annotations

import asyncio
import inspect
import math
import struct
import threading
import time
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
