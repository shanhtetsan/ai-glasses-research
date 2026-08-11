"""Canonical RGB and thermal orientation helpers.

These transforms define the coordinate system consumed by the backend.  They
are deliberately independent of browser/display settings so semantic facts,
recordings, Gemini, and YOLO all observe the same pixels.
"""
from __future__ import annotations

import asyncio
import json
import math
import statistics
import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np


RGB_REENCODE_QUALITY = 90
THERMAL_RAW_SHAPE = (24, 32)
THERMAL_CANONICAL_SHAPE = (32, 24)
THERMAL_PAYLOAD_BYTES = 24 * 32 * 4
RGB_CANONICAL_HEALTH_INTERVAL_SEC = 30.0


class RgbCanonicalizerTelemetry:
    """Bounded, thread-safe lifecycle and throughput metrics for one worker.

    ``raw_frames_received`` counts submissions, ``canonical_frames_completed``
    counts canonical JPEGs that successfully reached fan-out,
    ``canonical_pending_replaced`` counts pending raw frames discarded by a
    newer submission, and ``canonical_failures`` counts failed transform or
    publish attempts. Timing covers JPEG decode/rotate/encode, not fan-out.
    """

    def __init__(self, history_size: int = 200) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        self._lock = threading.Lock()
        self._timings_ms: deque[float] = deque(maxlen=history_size)
        self._raw_frames_received = 0
        self._canonical_frames_completed = 0
        self._canonical_pending_replaced = 0
        self._canonical_failures = 0
        self._worker_state = "stopped"
        self._last_failure_stage: str | None = None
        self._last_failure_reason: str | None = None

    def record_submission(self, replaced_pending: bool) -> None:
        with self._lock:
            self._raw_frames_received += 1
            if replaced_pending:
                self._canonical_pending_replaced += 1

    def record_completed(self, transform_ms: float) -> None:
        with self._lock:
            self._canonical_frames_completed += 1
            self._timings_ms.append(max(0.0, float(transform_ms)))

    def record_failure(self, stage: str, reason: str) -> int:
        with self._lock:
            self._canonical_failures += 1
            self._last_failure_stage = stage
            self._last_failure_reason = reason
            return self._canonical_failures

    def mark_running(self) -> None:
        with self._lock:
            self._worker_state = "running"

    def mark_stopped(self) -> None:
        with self._lock:
            self._worker_state = "stopped"

    def mark_unexpected_exit(self, reason: str) -> None:
        with self._lock:
            self._worker_state = "unexpected_exit"
            self._last_failure_stage = "worker"
            self._last_failure_reason = reason

    @staticmethod
    def _p95(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]

    def health(self) -> dict[str, Any]:
        with self._lock:
            timings = list(self._timings_ms)
            state = self._worker_state
            return {
                "worker_running": state == "running",
                "worker_alive": state == "running",
                "worker_state": state,
                "raw_frames_received": self._raw_frames_received,
                "canonical_frames_completed": self._canonical_frames_completed,
                "canonical_pending_replaced": self._canonical_pending_replaced,
                "canonical_failures": self._canonical_failures,
                "canonicalization_ms": {
                    "latest": None if not timings else round(timings[-1], 3),
                    "average": None if not timings else round(statistics.fmean(timings), 3),
                    "p95": None if not timings else round(self._p95(timings), 3),
                },
                "timing_history_size": len(timings),
                "timing_history_limit": self._timings_ms.maxlen,
                "last_failure_stage": self._last_failure_stage,
                "last_failure_reason": self._last_failure_reason,
            }


def canonicalize_rgb_jpeg(data: bytes) -> bytes | None:
    """Decode an ESP32 JPEG, rotate 90 degrees CCW, and encode exactly once."""
    try:
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            return None
        canonical = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ok, encoded = cv2.imencode(
            ".jpg",
            canonical,
            [int(cv2.IMWRITE_JPEG_QUALITY), RGB_REENCODE_QUALITY],
        )
        return encoded.tobytes() if ok else None
    except Exception:
        return None


def queue_latest_raw_rgb(
    data: bytes,
    received_at: float,
    raw_holder: dict,
    raw_event: asyncio.Event,
    telemetry: RgbCanonicalizerTelemetry,
) -> None:
    """Non-blocking, single-slot hand-off from socket ingest to canonicalizer."""
    telemetry.record_submission(replaced_pending=raw_holder.get("data") is not None)
    raw_holder["data"] = (data, received_at)
    raw_event.set()


async def run_latest_rgb_canonicalizer(
    raw_holder: dict,
    raw_event: asyncio.Event,
    publish,
    telemetry: RgbCanonicalizerTelemetry,
    transform=canonicalize_rgb_jpeg,
) -> None:
    """Canonicalize at most one in-flight plus one newest pending RGB frame."""
    loop = asyncio.get_running_loop()
    telemetry.mark_running()
    try:
        while True:
            await raw_event.wait()
            raw_event.clear()
            item = raw_holder.get("data")
            raw_holder["data"] = None
            if item is None:
                continue
            data, received_at = item
            stage = "transform"
            started = time.monotonic()
            try:
                canonical = await loop.run_in_executor(None, transform, data)
                transform_ms = (time.monotonic() - started) * 1000.0
                if canonical is None:
                    raise ValueError("transform_rejected")
                stage = "publish"
                await publish(canonical, received_at)
                telemetry.record_completed(transform_ms)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reason = type(exc).__name__
                failure_count = telemetry.record_failure(stage, reason)
                if failure_count == 1 or failure_count % 10 == 0:
                    print(
                        f"[RGB-CANONICAL] frame_failure count={failure_count} "
                        f"stage={stage} reason={reason}",
                        flush=True,
                    )
    except asyncio.CancelledError:
        telemetry.mark_stopped()
        raise
    except Exception as exc:
        telemetry.mark_unexpected_exit(type(exc).__name__)
        print(
            f"[RGB-CANONICAL] worker_exit reason={type(exc).__name__}",
            flush=True,
        )
        raise


async def log_rgb_canonicalizer_health(
    telemetry: RgbCanonicalizerTelemetry,
    interval_sec: float = RGB_CANONICAL_HEALTH_INTERVAL_SEC,
) -> None:
    """Periodically emit compact metrics without retaining log history."""
    while True:
        await asyncio.sleep(interval_sec)
        print(
            "[RGB-CANONICAL-HEALTH] "
            + json.dumps(telemetry.health(), separators=(",", ":")),
            flush=True,
        )


def canonicalize_thermal_payload(data: bytes) -> np.ndarray | None:
    """Convert native 24x32 data to wearer-oriented canonical 32x24.

    The 90-degree counter-clockwise rotation fixes the sensor mounting axis.
    Physical corner testing then showed that matrix's left/right axis was
    reversed relative to canonical RGB, so the horizontal flip belongs here,
    before every downstream thermal consumer.
    """
    if len(data) != THERMAL_PAYLOAD_BYTES:
        return None
    raw = np.frombuffer(data, dtype="<f4").reshape(THERMAL_RAW_SHAPE)
    if not np.isfinite(raw).all():
        return None
    rotated = np.rot90(raw, k=1)
    return np.ascontiguousarray(np.fliplr(rotated), dtype=np.float32)


def describe_grid_position(row: int, col: int, rows: int, cols: int) -> str:
    vertical = ("upper", "middle", "lower")[min(2, int(row * 3 / rows))]
    horizontal = ("left", "centre", "right")[min(2, int(col * 3 / cols))]
    if vertical == "middle" and horizontal == "centre":
        return "directly ahead"
    return f"{vertical} {horizontal}"


def summarize_thermal_grid(
    matrix: np.ndarray,
    age_sec: float,
    center_fraction: float = 0.4,
) -> dict[str, Any] | None:
    """Build spatial facts from an already-canonical thermal matrix."""
    grid = np.asarray(matrix, dtype=np.float32)
    if grid.shape != THERMAL_CANONICAL_SHAPE or not np.isfinite(grid).all():
        return None
    rows, cols = grid.shape
    ambient = float(np.percentile(grid, 20))
    scene_max = float(grid.max())
    hot_row, hot_col = np.unravel_index(int(np.argmax(grid)), grid.shape)
    half = center_fraction / 2
    r0, r1 = int(rows * (0.5 - half)), int(rows * (0.5 + half))
    c0, c1 = int(cols * (0.5 - half)), int(cols * (0.5 + half))
    centre = grid[r0:r1, c0:c1]
    regions = {}
    for ri, rname in enumerate(("upper", "middle", "lower")):
        for ci, cname in enumerate(("left", "centre", "right")):
            block = grid[
                rows * ri // 3:rows * (ri + 1) // 3,
                cols * ci // 3:cols * (ci + 1) // 3,
            ]
            regions[f"{rname}_{cname}"] = round(float(block.mean()), 1)
    return {
        "sensor": "MLX90640 32x24 thermopile array, roughly co-aligned with the camera",
        "measurement_age_sec": round(age_sec, 2),
        "ambient_c": round(ambient, 1),
        "scene_max_c": round(scene_max, 1),
        "scene_min_c": round(float(grid.min()), 1),
        "directly_ahead_mean_c": round(float(centre.mean()), 1),
        "directly_ahead_max_c": round(float(centre.max()), 1),
        "hotspot": {
            "temperature_c": round(scene_max, 1),
            "position": describe_grid_position(int(hot_row), int(hot_col), rows, cols),
            "above_ambient_c": round(scene_max - ambient, 1),
        },
        "region_mean_c": regions,
        "accuracy_note": (
            "Surface temperature estimates, +/-2C typical. Emissivity assumed 0.95; "
            "shiny or metallic surfaces read substantially cooler than they actually "
            "are. Not reliable for burn-safety decisions."
        ),
    }
