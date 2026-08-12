"""Optional latest-frame HTTP client for the separate YOLO shadow service.

No ML dependency is imported here. When disabled, the client creates no HTTP
client and starts no task, preserving the validated backend runtime path.
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
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import urlparse


_DETECTION_LOG_INTERVAL_SEC = 5.0


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
class YoloClientSettings:
    enabled: bool = False
    service_url: str = ""
    service_token: str = ""
    min_interval_sec: float = 1.0
    request_timeout_sec: float = 2.0
    confidence: float = 0.25
    stale_after_sec: float = 3.0
    health_interval_sec: float = 30.0

    @classmethod
    def from_env(cls) -> "YoloClientSettings":
        enabled = _env_bool("ENABLE_YOLO", False)
        if not enabled:
            return cls(enabled=False)
        return cls(
            enabled=True,
            service_url=os.getenv("YOLO_SERVICE_URL", "").strip().rstrip("/"),
            service_token=os.getenv("YOLO_SERVICE_TOKEN", "").strip(),
            min_interval_sec=_env_float("YOLO_MIN_INTERVAL_SEC", 1.0, 0.1),
            request_timeout_sec=_env_float("YOLO_REQUEST_TIMEOUT_SEC", 2.0, 0.1),
            confidence=_env_float("YOLO_CONFIDENCE", 0.25, 0.0, 1.0),
            stale_after_sec=_env_float("YOLO_CACHE_STALE_SEC", 3.0, 0.1),
            health_interval_sec=_env_float("YOLO_HEALTH_INTERVAL_SEC", 30.0, 5.0),
        )


def _clamp01(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("value must be finite")
    return max(0.0, min(1.0, number))


def validate_service_result(payload: Any, expected_frame_id: int) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("YOLO response must be an object")
    frame_id = int(payload["frame_id"])
    if frame_id != expected_frame_id:
        raise ValueError("YOLO response frame_id mismatch")
    inference_ms = float(payload["inference_ms"])
    width = int(payload["image_width"])
    height = int(payload["image_height"])
    objects = payload["objects"]
    if not math.isfinite(inference_ms) or inference_ms < 0:
        raise ValueError("invalid inference_ms")
    if width <= 0 or height <= 0 or not isinstance(objects, list):
        raise ValueError("invalid image metadata")
    validated_objects = []
    for item in objects:
        if not isinstance(item, dict):
            raise ValueError("invalid detection")
        bbox = item["bbox_norm"]
        center = item["center_norm"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("invalid bbox_norm")
        if not isinstance(center, list) or len(center) != 2:
            raise ValueError("invalid center_norm")
        validated_objects.append({
            "class_id": int(item["class_id"]),
            "label": str(item["label"]),
            "confidence": _clamp01(item["confidence"]),
            "bbox_norm": [_clamp01(value) for value in bbox],
            "center_norm": [_clamp01(value) for value in center],
        })
    return {
        "frame_id": frame_id,
        "inference_ms": inference_ms,
        "image_width": width,
        "image_height": height,
        "objects": validated_objects,
    }


class DetectionCache:
    def __init__(self, clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[dict] = None
        self._latest_metadata: dict = {}
        self._clock_ns = clock_ns

    def replace(self, result: dict, metadata: Optional[dict] = None) -> None:
        with self._lock:
            self._latest = copy.deepcopy(result)
            self._latest_metadata = copy.deepcopy(metadata or {})

    def read(self, stale_after_sec: float) -> dict:
        state, _metadata = self.read_with_metadata(stale_after_sec)
        return state

    def read_with_metadata(self, stale_after_sec: float) -> tuple[dict, dict]:
        with self._lock:
            latest = copy.deepcopy(self._latest)
            metadata = copy.deepcopy(self._latest_metadata)
        if latest is None:
            return (
                {"available": False, "stale": False, "age_ms": None, "result": None},
                metadata,
            )
        age_ms = max(
            0.0,
            (self._clock_ns() - latest["inference_completed_monotonic_ns"]) / 1_000_000,
        )
        stale = age_ms > stale_after_sec * 1000
        return (
            {
                "available": not stale,
                "stale": stale,
                "age_ms": round(age_ms, 3),
                "result": None if stale else latest,
            },
            metadata,
        )


class YoloShadowClient:
    def __init__(
        self,
        settings: YoloClientSettings,
        frames: Any,
        rotation_provider: Callable[[], int] = lambda: 0,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.settings = settings
        self.frames = frames
        self.rotation_provider = rotation_provider
        # Optional fire-and-forget hook for external event export (e.g. the
        # research platform's research_exporter.publish_event). Injected
        # rather than imported to keep this module's only external
        # dependency the perception service itself. Never awaited, always
        # guarded so a broken hook can never affect detection polling.
        self._on_event = on_event
        self.cache = DetectionCache()
        self._task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._http_client: Any = None
        self._httpx: Any = None
        self._running = False
        self._runtime_disabled = False
        self._unavailable_reason: Optional[str] = None
        self._service_healthy: Optional[bool] = None
        self._last_frame_id = 0
        self._last_attempt_at = 0.0
        self._last_detection_log_at: Optional[float] = None
        self._last_detection_signature: Optional[tuple[tuple[int, str, int], ...]] = None
        self._consecutive_failures = 0
        self._lock = threading.Lock()
        self._request_latencies: deque[float] = deque(maxlen=200)
        self._inference_latencies: deque[float] = deque(maxlen=200)
        self._metrics = {
            "requests_attempted": 0,
            "requests_completed": 0,
            "request_failures": 0,
            "request_timeouts": 0,
            "frames_skipped": 0,
            "frames_replaced": 0,
            "latest_http_ms": None,
            "latest_network_overhead_ms": None,
            "latest_inference_ms": None,
            "detection_count": 0,
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
        config_error = self._configuration_error()
        if config_error:
            self._runtime_disabled = True
            self._unavailable_reason = config_error
            print(f"[YOLO-CLIENT] disabled reason={config_error}", flush=True)
            return False
        try:
            import httpx  # existing cloud dependency via google-genai
            self._httpx = httpx
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.request_timeout_sec),
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
                follow_redirects=False,
            )
        except Exception as exc:
            self._runtime_disabled = True
            self._unavailable_reason = f"http_client_{type(exc).__name__}"
            print(f"[YOLO-CLIENT] disabled reason={self._unavailable_reason}", flush=True)
            return False
        self._running = True
        self._task = asyncio.create_task(self._run(), name="yolo-http-shadow")
        self._health_task = asyncio.create_task(self._health_logger(), name="yolo-http-health")
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
            self._record_frame_selection(frame.sequence)
            self._last_attempt_at = time.monotonic()
            await self._request_detection(frame)
            if self._consecutive_failures:
                await asyncio.sleep(min(2 ** (self._consecutive_failures - 1), 30))

    def _record_frame_selection(self, frame_id: int) -> None:
        if self._last_frame_id and frame_id > self._last_frame_id + 1:
            skipped = frame_id - self._last_frame_id - 1
            with self._lock:
                self._metrics["frames_skipped"] += skipped
                self._metrics["frames_replaced"] += skipped
        self._last_frame_id = frame_id

    async def _request_detection(self, frame: Any) -> None:
        client = self._http_client
        if client is None:
            return
        try:
            rotation = int(self.rotation_provider())
        except Exception:
            rotation = 0
        if rotation not in (0, 90, 180, 270):
            rotation = 0
        started_ns = time.monotonic_ns()
        with self._lock:
            self._metrics["requests_attempted"] += 1
        try:
            response = await client.post(
                self.settings.service_url + "/v1/detect",
                content=frame.data,
                params={"confidence": self.settings.confidence},
                headers={
                    "Authorization": "Bearer " + self.settings.service_token,
                    "Content-Type": "image/jpeg",
                    "X-Frame-ID": str(frame.sequence),
                    "X-Frame-Received-Monotonic-Ns": str(int(frame.timestamp * 1_000_000_000)),
                    "X-Camera-Rotation-Deg": str(rotation),
                },
            )
            response.raise_for_status()
            validated = validate_service_result(response.json(), frame.sequence)
            finished_ns = time.monotonic_ns()
            request_ms = max(0.0, (finished_ns - started_ns) / 1_000_000)
            inference_ms = validated["inference_ms"]
            cached = {
                "source_frame_id": frame.sequence,
                "backend_received_monotonic_ns": int(frame.timestamp * 1_000_000_000),
                "inference_completed_monotonic_ns": finished_ns,
                "request_ms": request_ms,
                "network_overhead_ms": max(0.0, request_ms - inference_ms),
                **validated,
            }
            self.cache.replace(cached, {"inference_rotation_deg": rotation})
            self._maybe_log_detection_state(validated)
            with self._lock:
                self._metrics["requests_completed"] += 1
                self._metrics["latest_http_ms"] = request_ms
                self._metrics["latest_network_overhead_ms"] = cached["network_overhead_ms"]
                self._metrics["latest_inference_ms"] = inference_ms
                self._metrics["detection_count"] = len(validated["objects"])
                self._request_latencies.append(request_ms)
                self._inference_latencies.append(inference_ms)
            self._consecutive_failures = 0
            self._service_healthy = True
            self._unavailable_reason = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            timeout_type = self._httpx.TimeoutException if self._httpx is not None else ()
            timed_out = isinstance(exc, timeout_type)
            with self._lock:
                self._metrics["request_failures"] += 1
                if timed_out:
                    self._metrics["request_timeouts"] += 1
                failure_count = self._metrics["request_failures"]
            self._consecutive_failures += 1
            self._service_healthy = False
            self._unavailable_reason = "timeout" if timed_out else type(exc).__name__
            if failure_count == 1 or failure_count % 10 == 0:
                print(
                    f"[YOLO-CLIENT] request failure count={failure_count} "
                    f"reason={self._unavailable_reason} backoff_sec="
                    f"{min(2 ** (self._consecutive_failures - 1), 30)}",
                    flush=True,
                )

    def _maybe_log_detection_state(
        self,
        result: dict,
        now: Optional[float] = None,
    ) -> None:
        objects = result["objects"]
        counts = Counter((item["class_id"], item["label"]) for item in objects)
        signature = tuple(sorted(
            (class_id, label, count)
            for (class_id, label), count in counts.items()
        ))
        logged_at = time.monotonic() if now is None else now
        changed = signature != self._last_detection_signature
        interval_elapsed = (
            self._last_detection_log_at is None
            or logged_at - self._last_detection_log_at >= _DETECTION_LOG_INTERVAL_SEC
        )
        if changed and self._on_event is not None:
            try:
                self._on_event("YOLO_CHANGE", {
                    "frame_id": result["frame_id"],
                    "object_count": len(objects),
                    "objects": objects,
                    "inference_ms": result["inference_ms"],
                })
            except Exception:
                pass
        if not changed and not interval_elapsed:
            return

        ordered = sorted(
            objects,
            key=lambda item: (
                item["label"].casefold(),
                item["label"],
                item["class_id"],
                -item["confidence"],
            ),
        )
        labels = ",".join(
            f'{item["label"]}:{item["confidence"]:.2f}' for item in ordered
        ) or "none"
        print(
            f'[YOLO-DETECTION] frame_id={result["frame_id"]} '
            f"objects={len(objects)} labels={labels} "
            f'inference_ms={result["inference_ms"]:.1f}',
            flush=True,
        )
        self._last_detection_signature = signature
        self._last_detection_log_at = logged_at

    async def _health_logger(self) -> None:
        while self._running:
            await asyncio.sleep(self.settings.health_interval_sec)
            if self._running:
                print("[YOLO-CLIENT-HEALTH] " + json.dumps(self.health(), separators=(",", ":")), flush=True)

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
            except Exception as exc:
                print(f"[YOLO-CLIENT] HTTP client close failed reason={type(exc).__name__}", flush=True)
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
            "average_http_ms": None if not request_latencies else round(statistics.fmean(request_latencies), 3),
            "p95_http_ms": self._p95(request_latencies),
            "average_inference_ms": None if not inference_latencies else round(statistics.fmean(inference_latencies), 3),
            "p95_inference_ms": self._p95(inference_latencies),
            "cache_available": cache["available"],
            "cache_stale": cache["stale"],
            "cache_age_ms": cache["age_ms"],
        }

    def latest_detections(self) -> dict:
        return self.cache.read(self.settings.stale_after_sec)

    def latest_perception(self, display_rotation_deg: int = 0) -> dict:
        cache, metadata = self.cache.read_with_metadata(self.settings.stale_after_sec)
        with self._lock:
            service_healthy = self._service_healthy
        response = {
            "enabled": self.settings.enabled,
            "available": cache["available"],
            "stale": cache["stale"],
            "age_ms": cache["age_ms"],
            "service_healthy": service_healthy,
            "frame_id": None,
            "inference_ms": None,
            "image_width": None,
            "image_height": None,
            "inference_rotation_deg": metadata.get("inference_rotation_deg"),
            "display_rotation_deg": display_rotation_deg,
            "mirrored": False,
            "objects": [],
        }
        result = cache["result"]
        if result is not None:
            response.update({
                "frame_id": result["frame_id"],
                "inference_ms": result["inference_ms"],
                "image_width": result["image_width"],
                "image_height": result["image_height"],
                "objects": copy.deepcopy(result["objects"]),
            })
        return response
