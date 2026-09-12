"""Latest-only MediaPipe hand client for the separate perception service.

The client samples the already-canonical ``latest_rgb`` store. It never sits
on camera ingest, Gemini, audio, thermal, or YOLO execution paths.
"""
from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import urlparse


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(
    name: str,
    default: float,
    minimum: float,
    maximum: Optional[float] = None,
) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    value = max(minimum, value)
    return value if maximum is None else min(maximum, value)


@dataclass(frozen=True)
class HandClientSettings:
    enabled: bool = False
    service_url: str = ""
    service_token: str = ""
    min_interval_sec: float = 0.33
    request_timeout_sec: float = 2.0
    stale_after_sec: float = 1.25
    health_interval_sec: float = 30.0

    @classmethod
    def from_env(cls) -> "HandClientSettings":
        enabled = _env_bool("ENABLE_HAND_TRACKING", False)
        if not enabled:
            return cls(enabled=False)
        return cls(
            enabled=True,
            service_url=(
                os.getenv("HAND_SERVICE_URL", "").strip()
                or os.getenv("YOLO_SERVICE_URL", "").strip()
            ).rstrip("/"),
            service_token=(
                os.getenv("HAND_SERVICE_TOKEN", "").strip()
                or os.getenv("YOLO_SERVICE_TOKEN", "").strip()
            ),
            min_interval_sec=_env_float("HAND_MIN_INTERVAL_SEC", 0.33, 0.1),
            request_timeout_sec=_env_float("HAND_REQUEST_TIMEOUT_SEC", 2.0, 0.1),
            stale_after_sec=_env_float("HAND_CACHE_STALE_SEC", 1.25, 0.1),
            health_interval_sec=_env_float("HAND_HEALTH_INTERVAL_SEC", 30.0, 5.0),
        )


def _clamp01(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("coordinate must be finite")
    return max(0.0, min(1.0, number))


def _point2(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"invalid {name}")
    return [_clamp01(value[0]), _clamp01(value[1])]


def validate_hand_result(payload: Any, expected_frame_id: int) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("hand response must be an object")
    frame_id = int(payload["frame_id"])
    if frame_id != expected_frame_id:
        raise ValueError("hand response frame_id mismatch")
    inference_ms = float(payload["inference_ms"])
    width = int(payload["image_width"])
    height = int(payload["image_height"])
    raw_hands = payload["hands"]
    if not math.isfinite(inference_ms) or inference_ms < 0:
        raise ValueError("invalid inference_ms")
    if width <= 0 or height <= 0 or not isinstance(raw_hands, list):
        raise ValueError("invalid image metadata")
    if len(raw_hands) > 2:
        raise ValueError("too many hands")

    hands = []
    for expected_index, raw_hand in enumerate(raw_hands):
        if not isinstance(raw_hand, dict):
            raise ValueError("invalid hand")
        raw_landmarks = raw_hand["landmarks"]
        if not isinstance(raw_landmarks, list) or len(raw_landmarks) != 21:
            raise ValueError("hand must contain 21 landmarks")
        landmarks = []
        for landmark_id, raw_landmark in enumerate(raw_landmarks):
            if not isinstance(raw_landmark, dict) or int(raw_landmark["id"]) != landmark_id:
                raise ValueError("invalid landmark id")
            z = float(raw_landmark["z"])
            if not math.isfinite(z):
                raise ValueError("invalid landmark z")
            landmarks.append({
                "id": landmark_id,
                "x": _clamp01(raw_landmark["x"]),
                "y": _clamp01(raw_landmark["y"]),
                "z": z,
            })
        bbox = raw_hand["bbox_norm"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("invalid bbox_norm")
        bbox_norm = [_clamp01(item) for item in bbox]
        if bbox_norm[0] > bbox_norm[2] or bbox_norm[1] > bbox_norm[3]:
            raise ValueError("invalid bbox ordering")
        hands.append({
            "hand_index": expected_index,
            "handedness": str(raw_hand["handedness"]),
            "handedness_score": _clamp01(raw_hand["handedness_score"]),
            "landmarks": landmarks,
            "index_tip_norm": _point2(raw_hand["index_tip_norm"], "index_tip_norm"),
            "wrist_norm": _point2(raw_hand["wrist_norm"], "wrist_norm"),
            "hand_center_norm": _point2(raw_hand["hand_center_norm"], "hand_center_norm"),
            "bbox_norm": bbox_norm,
        })
    return {
        "frame_id": frame_id,
        "image_width": width,
        "image_height": height,
        "inference_ms": inference_ms,
        "hands": hands,
    }


class HandResultCache:
    def __init__(self, clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._latest: Optional[dict] = None

    def replace(self, result: dict) -> None:
        with self._lock:
            self._latest = copy.deepcopy(result)

    def read(self, stale_after_sec: float) -> dict:
        with self._lock:
            latest = copy.deepcopy(self._latest)
        if latest is None:
            return {"available": False, "stale": False, "age_ms": None, "result": None}
        age_ms = max(
            0.0,
            (self._clock_ns() - latest["inference_completed_monotonic_ns"]) / 1_000_000,
        )
        stale = age_ms > stale_after_sec * 1000.0
        return {
            "available": not stale,
            "stale": stale,
            "age_ms": round(age_ms, 3),
            "result": None if stale else latest,
        }


class HandTrackingClient:
    def __init__(self, settings: HandClientSettings, frames: Any) -> None:
        self.settings = settings
        self.frames = frames
        self.cache = HandResultCache()
        self._task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._http_client: Any = None
        self._httpx: Any = None
        self._running = False
        self._runtime_disabled = False
        self._service_healthy: Optional[bool] = None
        self._unavailable_reason: Optional[str] = None
        self._last_frame_id = 0
        self._last_attempt_at = 0.0
        self._consecutive_failures = 0
        self._lock = threading.Lock()
        self._request_latencies: deque[float] = deque(maxlen=200)
        self._inference_latencies: deque[float] = deque(maxlen=200)
        self._metrics = {
            "requests_started": 0,
            "requests_completed": 0,
            "failures": 0,
            "timeouts": 0,
            "busy_responses": 0,
            "frames_skipped": 0,
            "latest_http_ms": None,
            "latest_inference_ms": None,
            "current_hand_count": 0,
        }

    def _configuration_error(self) -> Optional[str]:
        if not self.settings.service_url:
            return "service_url_missing"
        parsed = urlparse(self.settings.service_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            return "service_url_must_be_https"
        if not self.settings.service_token:
            return "service_token_missing"
        return None

    async def start(self) -> bool:
        if not self.settings.enabled:
            return False
        if self._running:
            return True
        error = self._configuration_error()
        if error:
            self._runtime_disabled = True
            self._unavailable_reason = error
            print(f"[HAND-CLIENT] disabled reason={error}", flush=True)
            return False
        try:
            import httpx
            self._httpx = httpx
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.request_timeout_sec),
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
                follow_redirects=False,
            )
        except Exception as exc:
            self._runtime_disabled = True
            self._unavailable_reason = f"http_client_{type(exc).__name__}"
            return False
        self._running = True
        self._task = asyncio.create_task(self._run(), name="hand-http-latest")
        self._health_task = asyncio.create_task(self._health_logger(), name="hand-http-health")
        return True

    async def _run(self) -> None:
        while self._running:
            wait_for = self.settings.min_interval_sec - (
                time.monotonic() - self._last_attempt_at
            )
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            frame = self.frames.snapshot()
            if frame.data is None or frame.sequence == self._last_frame_id:
                await asyncio.sleep(0.05)
                continue
            if self._last_frame_id and frame.sequence > self._last_frame_id + 1:
                with self._lock:
                    self._metrics["frames_skipped"] += frame.sequence - self._last_frame_id - 1
            self._last_frame_id = frame.sequence
            self._last_attempt_at = time.monotonic()
            await self._request_hands(frame)
            if self._consecutive_failures:
                await asyncio.sleep(min(2 ** (self._consecutive_failures - 1), 30))

    async def _request_hands(self, frame: Any) -> None:
        client = self._http_client
        if client is None:
            return
        started_ns = time.monotonic_ns()
        with self._lock:
            self._metrics["requests_started"] += 1
        try:
            response = await client.post(
                self.settings.service_url + "/v1/hands",
                content=frame.data,
                headers={
                    "Authorization": "Bearer " + self.settings.service_token,
                    "Content-Type": "image/jpeg",
                    "X-Frame-ID": str(frame.sequence),
                    "X-Frame-Received-Monotonic-Ns": str(
                        int(frame.timestamp * 1_000_000_000)
                    ),
                },
            )
            if response.status_code == 429:
                with self._lock:
                    self._metrics["busy_responses"] += 1
                return
            response.raise_for_status()
            validated = validate_hand_result(response.json(), frame.sequence)
            finished_ns = time.monotonic_ns()
            request_ms = max(0.0, (finished_ns - started_ns) / 1_000_000)
            received_at = time.time() - max(0.0, time.monotonic() - frame.timestamp)
            cached = {
                "source_frame_id": frame.sequence,
                "source_received_at": received_at,
                "backend_received_monotonic_ns": int(frame.timestamp * 1_000_000_000),
                "inference_completed_monotonic_ns": finished_ns,
                "request_ms": request_ms,
                **validated,
            }
            self.cache.replace(cached)
            with self._lock:
                self._metrics["requests_completed"] += 1
                self._metrics["latest_http_ms"] = request_ms
                self._metrics["latest_inference_ms"] = validated["inference_ms"]
                self._metrics["current_hand_count"] = len(validated["hands"])
                self._request_latencies.append(request_ms)
                self._inference_latencies.append(validated["inference_ms"])
            self._consecutive_failures = 0
            self._service_healthy = True
            self._unavailable_reason = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            timeout_type = self._httpx.TimeoutException if self._httpx is not None else ()
            timed_out = isinstance(exc, timeout_type)
            with self._lock:
                self._metrics["failures"] += 1
                if timed_out:
                    self._metrics["timeouts"] += 1
                failures = self._metrics["failures"]
            self._consecutive_failures += 1
            self._service_healthy = False
            self._unavailable_reason = "timeout" if timed_out else type(exc).__name__
            if failures == 1 or failures % 10 == 0:
                print(
                    f"[HAND-CLIENT] request failure count={failures} "
                    f"reason={self._unavailable_reason}",
                    flush=True,
                )

    async def _health_logger(self) -> None:
        while self._running:
            await asyncio.sleep(self.settings.health_interval_sec)
            if self._running:
                print(
                    "[HAND-CLIENT-HEALTH] "
                    + json.dumps(self.health(), separators=(",", ":")),
                    flush=True,
                )

    async def stop(self) -> None:
        self._running = False
        for task in (self._task, self._health_task):
            if task is not None:
                task.cancel()
        for task in (self._task, self._health_task):
            if task is not None:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._task = None
        self._health_task = None
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None

    @staticmethod
    def _p95(values: list[float]) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]

    def health(self) -> dict:
        with self._lock:
            metrics = dict(self._metrics)
            request_latencies = list(self._request_latencies)
            inference_latencies = list(self._inference_latencies)
        cache = self.cache.read(self.settings.stale_after_sec)
        return {
            "enabled": self.settings.enabled,
            "worker_running": bool(self._task is not None and not self._task.done()),
            "runtime_disabled": self._runtime_disabled,
            "service_healthy": self._service_healthy,
            "unavailable_reason": self._unavailable_reason,
            **metrics,
            "average_http_ms": (
                None if not request_latencies else round(statistics.fmean(request_latencies), 3)
            ),
            "p95_http_ms": self._p95(request_latencies),
            "average_inference_ms": (
                None if not inference_latencies else round(statistics.fmean(inference_latencies), 3)
            ),
            "p95_inference_ms": self._p95(inference_latencies),
            "latest_result_age_ms": cache["age_ms"],
            "cache_stale": cache["stale"],
        }

    def latest_hands(self) -> dict:
        cache = self.cache.read(self.settings.stale_after_sec)
        with self._lock:
            service_healthy = self._service_healthy
        result = cache["result"]
        available = bool(cache["available"] and service_healthy is not False)
        response = {
            "enabled": self.settings.enabled,
            "available": available,
            "stale": cache["stale"],
            "service_healthy": service_healthy,
            "unavailable_reason": self._unavailable_reason,
            "frame_id": None,
            "frame_sequence": None,
            "captured_at": None,
            "received_at": None,
            "backend_received_monotonic_ns": None,
            "inference_completed_monotonic_ns": None,
            "request_ms": None,
            "age_ms": cache["age_ms"],
            "inference_ms": None,
            "image_width": None,
            "image_height": None,
            "mirrored": False,
            "hands": [],
        }
        if available and result is not None:
            response.update({
                "frame_id": result["frame_id"],
                "frame_sequence": result["source_frame_id"],
                "received_at": result["source_received_at"],
                "backend_received_monotonic_ns": result["backend_received_monotonic_ns"],
                "inference_completed_monotonic_ns": result["inference_completed_monotonic_ns"],
                "request_ms": result["request_ms"],
                "inference_ms": result["inference_ms"],
                "image_width": result["image_width"],
                "image_height": result["image_height"],
                "hands": copy.deepcopy(result["hands"]),
            })
        return response
