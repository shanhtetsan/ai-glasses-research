"""Best-effort event export to the standalone research platform.

Modeled directly on YoloShadowClient in yolo_client.py: bounded queue,
drop-oldest, HTTPS-only service URL, bearer-token auth, short timeout,
exponential backoff, health() snapshot. The two clients talk to different
external Fly/Vercel services, but the "never let an external call touch the
realtime hot path" shape is identical on purpose.

Two independent safety gates keep this module inert by default:

  1. ENABLE_RESEARCH_EXPORT (env, default off) — if unset, publish_event()
     is a zero-cost no-op and no background task or HTTP client is ever
     created. This is a deploy-time switch.
  2. SessionGate.active — even when the exporter is enabled, publish_event()
     is a no-op unless a research platform has explicitly activated a
     session via the /internal/research-session handshake. This is a
     runtime switch, durable across a process restart (see SessionGate).

Every event published here also carries a copy of the current canonical RGB
frame's sequence number and bytes, taken from the same LatestFrameStore
YoloShadowClient already reads (see app_main.py's `latest_rgb`) — no new
frame-capture path, no extra JPEG encode.

Naming: everything here is research_* / ResearchExporter / SessionGate.
Nothing is named recording_* — that name is already owned by
RecordingPipeline and /api/recording in stability_runtime.py / app_main.py,
and reusing it here would collide with an existing, currently-active
feature (raw audio+video disk recording).
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Session gate: durable, disk-backed "is a research session active?" flag.
# ---------------------------------------------------------------------------

DEFAULT_SESSION_STATE_PATH = "research_session_state.json"
DEFAULT_SESSION_TTL_SEC = 6 * 3600.0  # 6 hours — auto-expiry safety net


class SessionGate:
    """Tracks the currently-active research session, if any.

    Backed by a small JSON file (atomic write, 0600 permissions) instead of
    a plain in-memory dict, so an in-place process restart — the
    /api/restart os.execv path, or a supervisor restarting a crashed
    process on the same Fly machine — does not silently drop research
    capture mid-session. This repo has no persistent Fly volume (see
    fly.toml), so this deliberately does NOT survive a fresh deploy or
    machine replacement — only a same-container process restart. That
    matches what "durable across restart" means for this app today; a
    dead research session after a real redeploy is expected and safe
    (research telemetry is disposable by design, see module docstring).

    Every read expires the session on the caller's behalf if `expires_at`
    has passed, so a forgotten/never-ended session cannot stay active
    forever even if no one ever calls deactivate().
    """

    def __init__(
        self,
        state_path: str = DEFAULT_SESSION_STATE_PATH,
        default_ttl_sec: float = DEFAULT_SESSION_TTL_SEC,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = state_path
        self._default_ttl_sec = max(1.0, float(default_ttl_sec))
        self._clock = clock
        self._lock = threading.Lock()
        self._session_id: Optional[str] = None
        self._token: Optional[str] = None
        self._expires_at: Optional[float] = None
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            session_id = data.get("session_id")
            token = data.get("token")
            expires_at = data.get("expires_at")
            if not session_id or not token or expires_at is None:
                return
            expires_at = float(expires_at)
        except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError, OSError):
            return
        with self._lock:
            self._session_id, self._token, self._expires_at = session_id, token, expires_at
            self._expire_if_needed_locked()
        if self._session_id:
            remaining = max(0.0, expires_at - self._clock())
            print(
                f"[RESEARCH-GATE] restored_from_disk session_id={session_id} "
                f"expires_in_sec={remaining:.0f}",
                flush=True,
            )

    def _save_locked(self) -> None:
        """Caller must hold self._lock."""
        payload = {
            "session_id": self._session_id,
            "token": self._token,
            "expires_at": self._expires_at,
        }
        tmp_path = f"{self._path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, self._path)
        except OSError as exc:
            print(f"[RESEARCH-GATE] persist_failed reason={exc}", flush=True)

    def _clear_file_locked(self) -> None:
        """Caller must hold self._lock."""
        try:
            os.remove(self._path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"[RESEARCH-GATE] clear_failed reason={exc}", flush=True)

    def _expire_if_needed_locked(self) -> None:
        """Caller must hold self._lock."""
        if self._expires_at is not None and self._clock() >= self._expires_at:
            self._session_id = None
            self._token = None
            self._expires_at = None
            self._clear_file_locked()

    # -- mutation ------------------------------------------------------------

    def activate(self, session_id: str, token: str, ttl_sec: Optional[float] = None) -> None:
        """Activate (or heartbeat-refresh) a session. Durable immediately."""
        ttl = self._default_ttl_sec if ttl_sec is None else max(1.0, float(ttl_sec))
        ttl = min(ttl, DEFAULT_SESSION_TTL_SEC * 4)  # sanity cap, not caller-controlled forever
        with self._lock:
            self._session_id = str(session_id)
            self._token = str(token)
            self._expires_at = self._clock() + ttl
            self._save_locked()

    def deactivate(self) -> None:
        with self._lock:
            self._session_id = None
            self._token = None
            self._expires_at = None
            self._clear_file_locked()

    # -- reads -----------------------------------------------------------

    @property
    def active(self) -> bool:
        with self._lock:
            self._expire_if_needed_locked()
            return self._session_id is not None

    @property
    def session_id(self) -> Optional[str]:
        with self._lock:
            self._expire_if_needed_locked()
            return self._session_id

    @property
    def token(self) -> Optional[str]:
        with self._lock:
            self._expire_if_needed_locked()
            return self._token

    def snapshot(self) -> dict:
        """Status for /api/health. Deliberately omits the bearer token."""
        with self._lock:
            self._expire_if_needed_locked()
            now = self._clock()
            return {
                "active": self._session_id is not None,
                "session_id": self._session_id,
                "expires_at": self._expires_at,
                "expires_in_sec": (
                    None if self._expires_at is None else round(max(0.0, self._expires_at - now), 1)
                ),
                "state_path": self._path,
            }


# ---------------------------------------------------------------------------
# Exporter: bounded queue -> batched HTTPS POST to the research platform.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResearchExporterSettings:
    enabled: bool = False
    service_url: str = ""
    ingest_path: str = "/api/ingest"
    request_timeout_sec: float = 2.0
    queue_maxsize: int = 64
    batch_max_events: int = 20
    batch_max_wait_sec: float = 1.0
    health_interval_sec: float = 30.0

    @classmethod
    def from_env(cls) -> "ResearchExporterSettings":
        enabled = _env_bool("ENABLE_RESEARCH_EXPORT", False)
        if not enabled:
            return cls(enabled=False)
        return cls(
            enabled=True,
            service_url=os.getenv("RESEARCH_PLATFORM_URL", "").strip().rstrip("/"),
            ingest_path=os.getenv("RESEARCH_INGEST_PATH", "/api/ingest").strip() or "/api/ingest",
            request_timeout_sec=_env_float("RESEARCH_REQUEST_TIMEOUT_SEC", 2.0, 0.1),
            queue_maxsize=int(_env_float("RESEARCH_QUEUE_MAXSIZE", 64, 1, 1000)),
            batch_max_events=int(_env_float("RESEARCH_BATCH_MAX_EVENTS", 20, 1, 200)),
            batch_max_wait_sec=_env_float("RESEARCH_BATCH_MAX_WAIT_SEC", 1.0, 0.05, 30),
            health_interval_sec=_env_float("RESEARCH_HEALTH_INTERVAL_SEC", 30.0, 5.0),
        )


@dataclass
class ResearchEvent:
    event_type: str
    occurred_at: float
    session_id: str
    fields: dict = field(default_factory=dict)
    keyframe_jpeg: Optional[bytes] = None
    keyframe_sequence: Optional[int] = None


class ResearchExporter:
    def __init__(
        self,
        settings: ResearchExporterSettings,
        frames: Any,
        session_gate: SessionGate,
    ) -> None:
        self.settings = settings
        self.frames = frames  # a LatestFrameStore, e.g. latest_rgb — same object YOLO reads
        self.session_gate = session_gate
        self._queue: "asyncio.Queue[ResearchEvent]" = asyncio.Queue(
            maxsize=max(1, settings.queue_maxsize)
        )
        self._task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._http_client: Any = None
        self._httpx: Any = None
        self._running = False
        self._runtime_disabled = False
        self._unavailable_reason: Optional[str] = None
        self._service_healthy: Optional[bool] = None
        self._consecutive_failures = 0
        self._lock = threading.Lock()
        self._metrics = {
            "events_published": 0,
            "events_dropped_disabled": 0,
            "events_dropped_no_session": 0,
            "events_dropped_queue_full": 0,
            "keyframes_attached": 0,
            "batches_attempted": 0,
            "batches_sent": 0,
            "batches_failed": 0,
            "batches_failed_no_token": 0,
            "request_timeouts": 0,
            "latest_http_ms": None,
        }

    # -- configuration -----------------------------------------------------

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
        return None

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> bool:
        if not self.settings.enabled:
            return False
        if self._running:
            return True
        config_error = self._configuration_error()
        if config_error:
            self._runtime_disabled = True
            self._unavailable_reason = config_error
            print(f"[RESEARCH-EXPORTER] disabled reason={config_error}", flush=True)
            return False
        try:
            import httpx  # existing cloud dependency via google-genai (see yolo_client.py)
            self._httpx = httpx
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.request_timeout_sec),
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
                follow_redirects=False,
            )
        except Exception as exc:
            self._runtime_disabled = True
            self._unavailable_reason = f"http_client_{type(exc).__name__}"
            print(f"[RESEARCH-EXPORTER] disabled reason={self._unavailable_reason}", flush=True)
            return False
        self._running = True
        self._task = asyncio.create_task(self._run(), name="research-exporter-sender")
        self._health_task = asyncio.create_task(self._health_logger(), name="research-exporter-health")
        return True

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
                print(f"[RESEARCH-EXPORTER] HTTP client close failed reason={type(exc).__name__}", flush=True)
            self._http_client = None

    # -- producer side: publish_event() -------------------------------------
    #
    # Synchronous, never awaited, never raises. Safe to call from any hot
    # path (camera frame handling, audio integrity marking, turn lifecycle,
    # YOLO detection-change) without any risk of blocking that caller.

    def publish_event(self, event_type: str, fields: Optional[dict] = None) -> bool:
        try:
            return self._publish_event(event_type, fields or {})
        except Exception:
            # Absolute guarantee: a bug in this module can never propagate
            # into camera/audio/Gemini/turn code that calls publish_event.
            return False

    def _publish_event(self, event_type: str, fields: dict) -> bool:
        if not self.settings.enabled or self._runtime_disabled:
            with self._lock:
                self._metrics["events_dropped_disabled"] += 1
            return False
        session_id = self.session_gate.session_id
        if session_id is None:
            with self._lock:
                self._metrics["events_dropped_no_session"] += 1
            return False
        frame = self.frames.snapshot() if self.frames is not None else None
        event = ResearchEvent(
            event_type=event_type,
            occurred_at=time.time(),
            session_id=session_id,
            fields=fields,
            keyframe_jpeg=None if frame is None or frame.data is None else frame.data,
            keyframe_sequence=None if frame is None else frame.sequence,
        )
        return self._enqueue(event)

    def _enqueue(self, event: ResearchEvent) -> bool:
        """Bounded queue, drop-oldest — same pattern as RecordingPipeline
        .enqueue_latest (stability_runtime.py), applied to typed events
        instead of raw video frames."""
        dropped_existing = False
        if self._queue.full():
            try:
                self._queue.get_nowait()
                dropped_existing = True
            except asyncio.QueueEmpty:
                pass
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            with self._lock:
                self._metrics["events_dropped_queue_full"] += 1
            return False
        with self._lock:
            self._metrics["events_published"] += 1
            if dropped_existing:
                self._metrics["events_dropped_queue_full"] += 1
            if event.keyframe_jpeg is not None:
                self._metrics["keyframes_attached"] += 1
        return True

    # -- consumer side: background batching sender --------------------------

    async def _run(self) -> None:
        try:
            while self._running:
                first = await self._queue.get()
                batch = [first]
                deadline = time.monotonic() + self.settings.batch_max_wait_sec
                while len(batch) < self.settings.batch_max_events:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                    batch.append(item)
                await self._send_batch(batch)
                if self._consecutive_failures:
                    await asyncio.sleep(min(2 ** (self._consecutive_failures - 1), 30))
        except asyncio.CancelledError:
            raise

    async def _send_batch(self, batch: list[ResearchEvent]) -> None:
        client = self._http_client
        if client is None or not batch:
            return
        token = self.session_gate.token
        if not token:
            # Session ended mid-flight (expired or deactivated) — nothing
            # valid to authenticate with. Drop rather than send unauthenticated.
            with self._lock:
                self._metrics["batches_failed_no_token"] += 1
            return
        payload = {
            "session_id": self.session_gate.session_id or batch[0].session_id,
            "events": [
                {
                    "type": item.event_type,
                    "occurred_at": item.occurred_at,
                    "fields": item.fields,
                    "keyframe_sequence": item.keyframe_sequence,
                    "keyframe_jpeg_base64": (
                        base64.b64encode(item.keyframe_jpeg).decode("ascii")
                        if item.keyframe_jpeg is not None else None
                    ),
                }
                for item in batch
            ],
        }
        started = time.monotonic()
        with self._lock:
            self._metrics["batches_attempted"] += 1
        try:
            response = await client.post(
                self.settings.service_url + self.settings.ingest_path,
                json=payload,
                headers={"Authorization": "Bearer " + token},
            )
            response.raise_for_status()
            request_ms = (time.monotonic() - started) * 1000.0
            with self._lock:
                self._metrics["batches_sent"] += 1
                self._metrics["latest_http_ms"] = request_ms
            self._consecutive_failures = 0
            self._service_healthy = True
            self._unavailable_reason = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            timeout_type = self._httpx.TimeoutException if self._httpx is not None else ()
            timed_out = isinstance(exc, timeout_type)
            with self._lock:
                self._metrics["batches_failed"] += 1
                if timed_out:
                    self._metrics["request_timeouts"] += 1
                failure_count = self._metrics["batches_failed"]
            self._consecutive_failures += 1
            self._service_healthy = False
            self._unavailable_reason = "timeout" if timed_out else type(exc).__name__
            if failure_count == 1 or failure_count % 10 == 0:
                print(
                    f"[RESEARCH-EXPORTER] batch failure count={failure_count} "
                    f"reason={self._unavailable_reason} backoff_sec="
                    f"{min(2 ** (self._consecutive_failures - 1), 30)}",
                    flush=True,
                )

    async def _health_logger(self) -> None:
        while self._running:
            await asyncio.sleep(self.settings.health_interval_sec)
            if self._running:
                print("[RESEARCH-EXPORTER-HEALTH] " + json.dumps(self.health(), separators=(",", ":")), flush=True)

    # -- status --------------------------------------------------------------

    def health(self) -> dict:
        with self._lock:
            metrics = dict(self._metrics)
        return {
            "enabled": self.settings.enabled,
            "worker_running": bool(self._task is not None and not self._task.done()),
            "runtime_disabled": self._runtime_disabled,
            "service_healthy": self._service_healthy,
            "unavailable_reason": self._unavailable_reason,
            "queue_depth": self._queue.qsize(),
            "queue_maxsize": self.settings.queue_maxsize,
            **metrics,
            "session_gate": self.session_gate.snapshot(),
        }
