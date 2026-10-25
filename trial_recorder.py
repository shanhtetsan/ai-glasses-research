"""Per-trial session recorder: append-only JSONL plus the frames sent to Gemini.

Off unless TRIAL_RECORDER=1. Every public method is synchronous, never awaits,
and never raises: it builds a small dict and does put_nowait() on one bounded
queue. A background task drains the queue and does all file I/O in a worker
thread, so no socket handler or Gemini send ever waits on the disk.

Layout under TRIAL_ROOT (default /data/trials):

    <trial>_<pid>/            or untitled_<ISO8601>/   (see UNTITLED_TRIGGERS:
                              the untitled fallback is opened lazily, at the
                              first event that is evidence, never on a START)
        meta.json             written once, never rewritten
        ground_truth.json     written once, if supplied
        session.jsonl         one event per line, append-only, flushed per line
        frames/<seq>.jpg      frames actually sent to Gemini
    current.json              which session is open, so a restart resumes it

Each line has exactly t, ts, ns, src, turn plus flat event fields.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from trial_config import NullSessionConfig

EVENT_SCHEMA_VERSION = 1
QUEUE_MAXSIZE = 256
UNTITLED_IDLE_CLOSE_SEC = 30 * 60
IMU_STREAM_INTERVAL_SEC = 0.1          # 10 Hz while a turn is active
THERMAL_MAX_AGE_SEC = 3.0
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,31}$")
_TRIAL_TEXT_FIELDS = ("site", "cond", "task", "success_criterion", "prompt_version")

# The only events that open the untitled fallback session. Both are evidence a
# run really happened: a question was asked, or the model produced a turn with
# text in it. Everything else the recorder sees — the mic starting, pump or
# keepalive frames, IMU, session rotations, errors — also happens on an idle
# box after a deploy, and must not create a folder.
UNTITLED_TRIGGERS = frozenset({"speech_start", "turn_end"})


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _flat(prefix: str, value: Any, out: dict) -> None:
    """Flatten one level of nesting into prefix_key entries."""
    if isinstance(value, dict):
        for key, inner in value.items():
            out[f"{prefix}_{key}"] = inner if not isinstance(inner, dict) else json.dumps(inner, separators=(",", ":"))
    else:
        out[prefix] = value


class TrialRecorder:
    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        root: Optional[str] = None,
        queue_maxsize: int = QUEUE_MAXSIZE,
        thermal_summarizer: Optional[Callable[[Any, float, dict], Optional[dict]]] = None,
        static_fields: Optional[Callable[[], dict]] = None,
        on_session_ended: Optional[Callable[[str, str], None]] = None,
        session_config: Optional[Any] = None,
    ) -> None:
        self.enabled = (os.getenv("TRIAL_RECORDER", "") == "1") if enabled is None else bool(enabled)
        self.root = root or os.getenv("TRIAL_ROOT", "/data/trials")
        self._queue_maxsize = max(2, int(queue_maxsize))
        self._queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None
        self._thermal_summarizer = thermal_summarizer
        self._static_fields = static_fields or (lambda: {})
        self._on_session_ended = on_session_ended
        # Per-trial experiment knobs. The override's lifetime is this recorder's
        # session, so the recorder owns installing and dropping it; what the
        # knobs mean is none of its business (see trial_config.py).
        self._config = session_config if session_config is not None else NullSessionConfig()

        # Session state (event-loop thread only).
        self._session: Optional[dict] = None
        self._turn_uids: dict[int, str] = {}       # server turn id -> uid
        self._turn_agg: dict[int, dict] = {}       # per-turn prompt_sent aggregates
        self._current_turn: Optional[int] = None   # last speech_start, until turn_end
        self._n_to_server: dict[int, int] = {}      # SPEECH_START n -> server turn id
        self._last_imu_stream = 0.0

        # Writer-thread state.
        self._fh = None
        self._fh_dir: Optional[str] = None
        self._frames_this_boot: dict[tuple, str] = {}

        self._pending_drops = 0
        self.metrics = {
            "events_enqueued": 0,
            "events_written": 0,
            "events_dropped": 0,
            "events_no_session": 0,
            "frames_written": 0,
            "frames_not_sent": 0,
            "write_errors": 0,
            "internal_errors": 0,
            "last_error": None,
        }

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        if not self.enabled:
            return
        try:
            self._queue = asyncio.Queue(maxsize=self._queue_maxsize)
            await asyncio.to_thread(os.makedirs, self.root, exist_ok=True)
            await self._restore_current()
            self._task = asyncio.create_task(self._writer(), name="trial-recorder-writer")
            print(f"[TRIAL] recorder enabled root={self.root}", flush=True)
        except Exception as exc:
            self._note_internal_error("start", exc)
            self.enabled = False

    async def stop(self) -> None:
        """Flush what is queued. Does NOT end the session: a restart resumes it."""
        if not self.enabled or self._queue is None:
            return
        try:
            await asyncio.wait_for(self._queue.join(), timeout=5.0)
        except Exception:
            pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        await asyncio.to_thread(self._close_fh)

    async def flush(self, timeout: float = 10.0) -> None:
        """Wait until already-enqueued recording writes reach disk."""
        if not self.enabled or self._queue is None:
            return
        await asyncio.wait_for(self._queue.join(), timeout=timeout)

    # ------------------------------------------------------------------ sessions

    async def set_trial(self, meta: dict, ground_truth: Optional[dict] = None, reason: str = "set_trial",
                        config: Any = None) -> dict:
        """Open (or resume) a titled session. Raises ValueError on bad input.

        `config` is the optional per-trial knob block. It is validated before
        anything is opened or closed, installed between closing the previous
        session and opening this one — so the values stamped into meta.json are
        the ones this session will actually run on — and dropped again by
        end_session().
        """
        if not self.enabled:
            raise RuntimeError("trial recorder disabled (set TRIAL_RECORDER=1)")
        trial = str(meta.get("trial", "")).strip()
        pid = str(meta.get("pid", "")).strip()
        if not _SAFE_ID.match(trial) or not _SAFE_ID.match(pid):
            raise ValueError("trial and pid are required: letters, digits, '-' (max 32)")
        parsed_config = self._config.parse(config)
        session_id = f"{trial}_{pid}"
        current = self._session
        if current is not None and current["id"] == session_id:
            # A resume, not a new run: meta.json is already written, so the arm
            # cannot be swapped under it. Agreeing with what is running is fine.
            conflict = self._config.conflict(parsed_config)
            if conflict:
                raise ValueError(conflict)
            self._emit("session_resume", None, reason=reason)
            return self.status()
        if current is not None:
            self.end_session(reason="new_trial")
        self._config.apply(parsed_config)
        clean = {"trial": trial, "pid": pid}
        for key in _TRIAL_TEXT_FIELDS:
            value = meta.get(key)
            clean[key] = None if value is None else str(value)
        await self._open_session(session_id, clean, ground_truth, reason)
        return self.status()

    def ensure_session(self, reason: str) -> None:
        """Called on every START. Resumes an open session; never opens one.

        A START only means the mic socket is streaming. That happens on every
        reconnect after a deploy with nobody in front of the glasses, so
        opening the untitled fallback here wrote a 0-turn folder per deploy.
        The fallback is opened lazily instead, by _open_untitled_for_event,
        at the first event that is actually evidence (see UNTITLED_TRIGGERS).
        A session opened explicitly through set_trial still opens immediately.
        """
        if not self.enabled:
            return
        try:
            self._expire_idle_untitled()
            if self._session is not None:
                self._emit("session_resume", None, reason=reason)
        except Exception as exc:
            self._note_internal_error("ensure_session", exc)

    def _expire_idle_untitled(self) -> None:
        """Close an untitled session that has been sitting idle too long."""
        session = self._session
        if session is not None and session["untitled"]:
            if time.monotonic() - session["last_event_mono"] > UNTITLED_IDLE_CLOSE_SEC:
                self.end_session(reason="idle")

    def _open_untitled_for_event(self, trigger: str) -> bool:
        """Open the untitled fallback for the first real event. True if open after."""
        self._expire_idle_untitled()
        if self._session is not None:
            return True
        if trigger not in UNTITLED_TRIGGERS:
            return False
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        session_id = f"untitled_{stamp}"
        meta = {"trial": None, "pid": None}
        meta.update({key: None for key in _TRIAL_TEXT_FIELDS})
        self._begin_session(session_id, meta, None, f"first_event:{trigger}",
                            used_uids=set(), untitled=True, resumed=False)
        return self._session is not None

    def end_session(self, reason: str) -> Optional[str]:
        if not self.enabled or self._session is None:
            return None
        try:
            session = self._session
            # Every close funnels through here — /internal/trial/end, a new trial
            # replacing this one, and the idle expiry of an untitled session — so
            # this is the one place the per-trial override has to be dropped.
            self._config.clear()
            self._emit("session_end", None, reason=reason, dropped_total=self.metrics["events_dropped"],
                       frames_not_sent=self.metrics["frames_not_sent"])
            self._put({"op": "close", "dir": session["dir"]}, control=True)
            if self._on_session_ended is not None:
                try:
                    self._on_session_ended(session["id"], session["dir"])
                except Exception as exc:
                    self._note_internal_error("on_session_ended", exc)
            self._session = None
            self._turn_uids.clear()
            self._turn_agg.clear()
            self._n_to_server.clear()
            self._current_turn = None
            return session["id"]
        except Exception as exc:
            self._note_internal_error("end_session", exc)
            return None

    async def _open_session(self, session_id, meta, ground_truth, reason) -> None:
        directory = os.path.join(self.root, session_id)
        used_uids, existed, turns = await asyncio.to_thread(self._scan_existing, directory)
        self._begin_session(session_id, meta, ground_truth, reason, used_uids=used_uids,
                            untitled=False, resumed=existed, turns=turns)

    def _begin_session(self, session_id, meta, ground_truth, reason, *, used_uids, untitled, resumed,
                       turns: int = 0) -> None:
        directory = os.path.join(self.root, session_id)
        static = {}
        try:
            static = dict(self._static_fields() or {})
        except Exception as exc:
            self._note_internal_error("static_fields", exc)
        if not meta.get("prompt_version"):
            meta["prompt_version"] = static.get("prompt_version_default")
        static.pop("prompt_version_default", None)
        start_fields = {**meta, **static}
        self._session = {
            "id": session_id,
            "dir": directory,
            "untitled": untitled,
            "trial": meta.get("trial") or session_id,
            "pid": meta.get("pid") or "P0",
            "used_uids": set(used_uids),
            "server_turn_seq": 0,
            "last_event_mono": time.monotonic(),
            "meta": start_fields,
            # Questions asked so far: one per speech_start, seeded from the log
            # on resume so a restart does not reset the count the UI shows.
            "turns": int(turns),
        }
        self._turn_uids.clear()
        self._turn_agg.clear()
        self._n_to_server.clear()
        self._current_turn = None
        self._put({
            "op": "open", "dir": directory, "id": session_id,
            "meta": {**start_fields, "session_id": session_id, "created_ts": _iso_now(),
                     "schema_version": EVENT_SCHEMA_VERSION},
            "ground_truth": ground_truth,
        }, control=True)
        if resumed:
            # Carries the same observed fields as session_start. meta.json is
            # written once, at the first open, so after a restart this line is
            # the only record of what the resumed process is really running —
            # notably that a per-trial override did not survive the restart.
            self._emit("session_resume", None, reason=reason, **start_fields)
        else:
            self._emit("session_start", None, **start_fields)
        print(f"[TRIAL] session {'resumed' if resumed else 'opened'} id={session_id} reason={reason}", flush=True)

    # ------------------------------------------------------------------ turn ids

    def turn_uid(self, server_turn_id: Optional[int]) -> Optional[str]:
        if server_turn_id is None or self._session is None:
            return None
        uid = self._turn_uids.get(server_turn_id)
        if uid is None:
            # A turn the server opened itself (no SPEECH_START:<n>).
            self._session["server_turn_seq"] += 1
            uid = self._unique_uid(f"s{self._session['server_turn_seq']}")
            self._turn_uids[server_turn_id] = uid
        return uid

    def _unique_uid(self, n_part: str) -> str:
        session = self._session
        base = f"{session['trial']}.{session['pid']}.{n_part}"
        uid, k = base, 1
        while uid in session["used_uids"]:
            k += 1
            uid = f"{base}.r{k}"
        session["used_uids"].add(uid)
        return uid

    # ------------------------------------------------------------------ events

    def speech_start(self, n: int, server_turn_id: int, thermal: Optional[tuple], imu: Optional[dict]) -> None:
        if not self.enabled:
            return
        try:
            if not self._open_untitled_for_event("speech_start"):
                return
            uid = self._unique_uid(str(int(n)))
            self._turn_uids[server_turn_id] = uid
            self._turn_agg[server_turn_id] = {"frames": 0, "seq_first": None, "seq_last": None,
                                              "thermal_text": None, "audio_bytes": 0, "prompt_sent": False,
                                              "speech_ended": False}
            self._current_turn = server_turn_id
            self._n_to_server[int(n)] = server_turn_id
            self._session["turns"] += 1
            self._emit("speech_start", uid, n=int(n))
            matrix, captured_at, calibration = thermal if thermal else (None, None, None)
            age_sec = None if captured_at is None else max(0.0, time.monotonic() - captured_at)
            self._emit("thermal", uid, _thermal=(matrix, age_sec, calibration))
            self._emit("imu", uid, **self._imu_fields(imu, "speech_start"))
        except Exception as exc:
            self._note_internal_error("speech_start", exc)

    def speech_end(self, n: int, server_turn_id: Optional[int]) -> None:
        if not self.enabled or self._session is None:
            return
        # Tag by the n the app sent, not whatever turn the server has active:
        # a late reply to an earlier question can be holding the server turn.
        owner = self._n_to_server.get(int(n), server_turn_id)
        agg = self._turn_agg.get(owner) if owner is not None else None
        if agg is not None:
            agg["speech_ended"] = True
        self._emit("speech_end", self.turn_uid(owner), n=int(n))

    def imu_sample(self, imu: Optional[dict]) -> None:
        """Called per IMU packet; records at 10 Hz only while a turn is active."""
        if not self.enabled or self._session is None or self._current_turn is None:
            return
        now = time.monotonic()
        if now - self._last_imu_stream < IMU_STREAM_INTERVAL_SEC:
            return
        self._last_imu_stream = now
        self._emit("imu", self.turn_uid(self._current_turn), **self._imu_fields(imu, "stream"))

    @staticmethod
    def _imu_fields(imu: Optional[dict], at: str) -> dict:
        data = (imu or {}).get("data") if imu else None
        if not data:
            return {"at": at, "seq": None, "accel": None, "gyro": None, "uptime_ms": None, "age_ms": None}
        accel = data.get("accel") or {}
        gyro = data.get("gyro") or {}
        stamp = imu.get("timestamp") or 0.0
        return {
            "at": at,
            "seq": data.get("sequence"),
            "accel": [accel.get("x"), accel.get("y"), accel.get("z")],
            "gyro": [gyro.get("x"), gyro.get("y"), gyro.get("z")],
            "uptime_ms": data.get("uptime_ms"),
            "age_ms": round((time.monotonic() - stamp) * 1000, 1) if stamp else None,
        }

    def frame_sent(self, server_turn_id: Optional[int], seq: Optional[int], jpeg: bytes,
                   captured_at: Optional[float], source: str) -> None:
        if not self.enabled or self._session is None:
            return
        try:
            uid = self.turn_uid(server_turn_id)
            age_ms = None if captured_at is None else round(max(0.0, time.monotonic() - captured_at) * 1000, 1)
            agg = self._turn_agg.get(server_turn_id) if server_turn_id is not None else None
            if agg is not None and seq is not None:
                agg["frames"] += 1
                if agg["seq_first"] is None:
                    agg["seq_first"] = seq
                agg["seq_last"] = seq
            self._emit("frame_sent", uid, seq=seq, bytes=len(jpeg), age_ms=age_ms, source=source,
                       _jpeg=jpeg)
        except Exception as exc:
            self._note_internal_error("frame_sent", exc)

    def frame_not_sent(self) -> None:
        if self.enabled:
            self.metrics["frames_not_sent"] += 1

    def note_thermal_text(self, server_turn_id: Optional[int], text: str) -> None:
        agg = self._turn_agg.get(server_turn_id) if self.enabled and server_turn_id is not None else None
        if agg is not None:
            agg["thermal_text"] = text

    def note_audio(self, byte_count: int) -> None:
        """Mic PCM forwarded to Gemini (16 kHz mono s16 -> 32 bytes per ms)."""
        if not self.enabled or self._current_turn is None:
            return
        agg = self._turn_agg.get(self._current_turn)
        if agg is not None:
            agg["audio_bytes"] += byte_count

    def prompt_sent(self, server_turn_id: Optional[int], final: bool = False) -> None:
        """What the model had when it started answering this question.

        Live has no discrete prompt: emitted at the first model audio after
        this turn's speech_end (earlier audio is the tail of a previous reply),
        or at turn_end if that never happened. Once per turn.
        """
        if not self.enabled or self._session is None or server_turn_id is None:
            return
        agg = self._turn_agg.get(server_turn_id)
        if agg is None or agg["prompt_sent"]:
            return
        if not final and not agg["speech_ended"]:
            return
        agg["prompt_sent"] = True
        self._emit("prompt_sent", self.turn_uid(server_turn_id), frames=agg["frames"],
                   seq_first=agg["seq_first"], seq_last=agg["seq_last"],
                   thermal_text=agg["thermal_text"], audio_ms=round(agg["audio_bytes"] / 32.0, 1))

    def turn_end(self, server_turn_id: int, record: dict, generation: Optional[int]) -> None:
        if not self.enabled:
            return
        try:
            if self._session is None:
                # A turn the server opened on its own still counts as a run, but
                # only once it carries words. An empty or cancelled turn on an
                # idle box is not evidence and must not create a folder.
                has_text = bool(str(record.get("input_transcription") or "").strip()
                                or str(record.get("response_text") or "").strip())
                if not has_text or not self._open_untitled_for_event("turn_end"):
                    return
            uid = self.turn_uid(server_turn_id)
            self.prompt_sent(server_turn_id, final=True)
            question = record.get("input_transcription")
            reply = record.get("response_text")
            if question:
                self._emit("response_text", uid, text=question, kind="input_transcription")
            if reply:
                self._emit("response_text", uid, text=reply, kind="output_transcription")
            fields: dict = {"status": record.get("status"), "server_turn_id": server_turn_id,
                            "generation": generation, "question": question, "reply": reply}
            timestamps = record.get("timestamps_ns") or {}
            created = timestamps.get("turn_created")
            for name, value in timestamps.items():
                fields[f"{name}_ms"] = (None if value is None or created is None
                                        else round((value - created) / 1_000_000, 3))
            skip = {"turn_id", "status", "timestamps_ns", "input_transcription", "response_text"}
            for key, value in record.items():
                if key in skip:
                    continue
                if key == "session_id":
                    key = "telemetry_session_id"
                elif key == "session_label":
                    key = "telemetry_session_label"
                _flat(key, value, fields)
            self._emit("turn_end", uid, **fields)
        except Exception as exc:
            self._note_internal_error("turn_end", exc)
        finally:
            self._turn_agg.pop(server_turn_id, None)
            if self._current_turn == server_turn_id:
                self._current_turn = None

    def session_rotate(self, reason: str, generation: int, age_sec: Optional[float], resumed: bool) -> None:
        if not self.enabled or self._session is None:
            return
        self._emit("session_rotate", None, reason=reason, generation=generation,
                   age_sec=None if age_sec is None else round(age_sec, 3), resumed=bool(resumed))

    def mic_streaming(self, value: bool, reason: str, conn: Optional[int]) -> None:
        if not self.enabled or self._session is None:
            return
        self._emit("mic_streaming", None, value=bool(value), reason=reason, conn=conn)

    def error(self, where: str, exc: Any, turn: Optional[int] = None) -> None:
        if not self.enabled or self._session is None:
            return
        kind = type(exc).__name__ if isinstance(exc, BaseException) else "error"
        self._emit("error", self.turn_uid(turn), where=where, kind=kind, detail=str(exc)[:300])

    # ------------------------------------------------------------------ queue

    def _emit(self, event_type: str, turn: Optional[str], **fields) -> None:
        try:
            if self._session is None:
                self.metrics["events_no_session"] += 1
                return
            self._session["last_event_mono"] = time.monotonic()
            item = {"t": event_type, "ts": _iso_now(), "ns": time.monotonic_ns(),
                    "src": "backend", "turn": turn}
            item.update(fields)
            self._put({"op": "event", "dir": self._session["dir"], "event": item})
        except Exception as exc:
            self._note_internal_error("emit", exc)

    def _put(self, item: dict, control: bool = False) -> None:
        queue = self._queue
        if queue is None:
            return
        # Session open/close must never be lost: make room by dropping an event.
        if control and queue.full():
            try:
                queue.get_nowait()
                queue.task_done()
                self._pending_drops += 1
                self.metrics["events_dropped"] += 1
            except asyncio.QueueEmpty:
                pass
        if not control:
            if queue.full() or (self._pending_drops and queue.qsize() >= self._queue_maxsize - 1):
                self._pending_drops += 1
                self.metrics["events_dropped"] += 1
                return
            if self._pending_drops:
                drop = {"t": "drop", "ts": _iso_now(), "ns": time.monotonic_ns(), "src": "backend",
                        "turn": None, "dropped_count": self._pending_drops}
                queue.put_nowait({"op": "event", "dir": item["dir"], "event": drop})
                self._pending_drops = 0
        queue.put_nowait(item)
        self.metrics["events_enqueued"] += 1

    # ------------------------------------------------------------------ writer

    async def _writer(self) -> None:
        queue = self._queue
        while True:
            batch = [await queue.get()]
            while len(batch) < 64:
                try:
                    batch.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            try:
                await asyncio.to_thread(self._write_batch, batch)
            except Exception as exc:
                self._note_internal_error("writer", exc)
            finally:
                for _ in batch:
                    queue.task_done()

    def _write_batch(self, batch: list) -> None:
        for item in batch:
            try:
                op = item["op"]
                if op == "event":
                    self._write_event(item["dir"], item["event"])
                elif op == "open":
                    self._write_open(item)
                elif op == "close":
                    self._close_fh()
                    self._write_current(None)
            except Exception as exc:
                self.metrics["write_errors"] += 1
                self.metrics["last_error"] = f"{type(exc).__name__}: {exc}"[:300]

    def _ensure_fh(self, directory: str):
        if self._fh is None or self._fh_dir != directory:
            self._close_fh()
            os.makedirs(os.path.join(directory, "frames"), exist_ok=True)
            self._fh = open(os.path.join(directory, "session.jsonl"), "a", encoding="utf-8")
            self._fh_dir = directory
        return self._fh

    def _close_fh(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
        self._fh = None
        self._fh_dir = None

    def _write_open(self, item: dict) -> None:
        directory = item["dir"]
        os.makedirs(os.path.join(directory, "frames"), exist_ok=True)
        for name, payload in (("meta.json", item["meta"]), ("ground_truth.json", item.get("ground_truth"))):
            if payload is None:
                continue
            try:
                with open(os.path.join(directory, name), "x", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
                    fh.write("\n")
            except FileExistsError:
                pass  # written once; a resume never rewrites it
        self._write_current(item["id"])
        self._ensure_fh(directory)

    def _write_current(self, session_id: Optional[str]) -> None:
        path = os.path.join(self.root, "current.json")
        if session_id is None:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            return
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"session_id": session_id}, fh)
        os.replace(tmp, path)

    def _write_event(self, directory: str, event: dict) -> None:
        jpeg = event.pop("_jpeg", None)
        thermal = event.pop("_thermal", None)
        if jpeg is not None:
            event["file"] = self._write_frame(directory, event.get("seq"), jpeg)
        if thermal is not None:
            event.update(self._thermal_fields(*thermal))
        fh = self._ensure_fh(directory)
        fh.write(json.dumps(event, separators=(",", ":"), default=str) + "\n")
        fh.flush()
        self.metrics["events_written"] += 1

    def _write_frame(self, directory: str, seq: Optional[int], jpeg: bytes) -> str:
        key = (directory, seq)
        if seq is not None and key in self._frames_this_boot:
            return self._frames_this_boot[key]   # same frame sent twice (pump + pre_response)
        stem = str(seq) if seq is not None else "noseq"
        k = 0
        while True:
            name = f"{stem}.jpg" if k == 0 else f"{stem}_{k}.jpg"
            try:
                with open(os.path.join(directory, "frames", name), "xb") as fh:
                    fh.write(jpeg)
                break
            except FileExistsError:
                k += 1   # seq restarted after a process restart: never overwrite
        rel = f"frames/{name}"
        if seq is not None:
            self._frames_this_boot[key] = rel
        self.metrics["frames_written"] += 1
        return rel

    def _thermal_fields(self, matrix, age_sec, calibration) -> dict:
        empty = {"ahead_c": None, "ambient_c": None, "hotspot_c": None, "hotspot_xy": None,
                 "hotspot_rgb_xy": None, "age_ms": None if age_sec is None else round(age_sec * 1000, 1)}
        if matrix is None or age_sec is None or age_sec > THERMAL_MAX_AGE_SEC or self._thermal_summarizer is None:
            return empty
        facts = self._thermal_summarizer(matrix, age_sec, calibration or {})
        if not facts:
            return empty
        hotspot = facts.get("hotspot") or {}
        return {
            "ahead_c": facts.get("directly_ahead_mean_c"),
            "ambient_c": facts.get("ambient_c"),
            "hotspot_c": hotspot.get("temperature_c"),
            "hotspot_xy": hotspot.get("thermal_norm"),
            "hotspot_rgb_xy": hotspot.get("rgb_norm_coarse"),
            "age_ms": round(age_sec * 1000, 1),
        }

    # ------------------------------------------------------------------ restore

    @staticmethod
    def _scan_existing(directory: str) -> tuple[set, bool, int]:
        """Returns (turn uids already used, the folder existed, speech_start count)."""
        used: set = set()
        turns = 0
        path = os.path.join(directory, "session.jsonl")
        if not os.path.exists(path):
            return used, os.path.isdir(directory), turns
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("turn"):
                    used.add(event["turn"])
                if event.get("t") == "speech_start":
                    turns += 1
        return used, True, turns

    async def _restore_current(self) -> None:
        def _read():
            try:
                with open(os.path.join(self.root, "current.json"), "r", encoding="utf-8") as fh:
                    session_id = json.load(fh).get("session_id")
                directory = os.path.join(self.root, str(session_id))
                with open(os.path.join(directory, "meta.json"), "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                return session_id, meta
            except (OSError, ValueError):
                return None, None
        session_id, meta = await asyncio.to_thread(_read)
        if not session_id or not meta:
            return
        directory = os.path.join(self.root, session_id)
        used, _, turns = await asyncio.to_thread(self._scan_existing, directory)
        untitled = str(session_id).startswith("untitled_")
        trial_meta = {"trial": meta.get("trial"), "pid": meta.get("pid")}
        trial_meta.update({key: meta.get(key) for key in _TRIAL_TEXT_FIELDS})
        self._begin_session(session_id, trial_meta, None, "process_restart",
                            used_uids=used, untitled=untitled, resumed=True, turns=turns)

    # ------------------------------------------------------------------ status

    def _note_internal_error(self, where: str, exc: BaseException) -> None:
        self.metrics["internal_errors"] += 1
        self.metrics["last_error"] = f"{where}: {type(exc).__name__}: {exc}"[:300]

    def status(self) -> dict:
        session = self._session
        return {
            "enabled": self.enabled,
            "root": self.root,
            "session_id": None if session is None else session["id"],
            "untitled": None if session is None else session["untitled"],
            # Questions asked in the open session, for the live count on /trials.
            "turn_count": None if session is None else session["turns"],
            "meta": None if session is None else dict(session["meta"]),
            # What the backend is running right now, and which of it was set by
            # this trial rather than by the environment.
            "config": self._config.observed(),
            "config_overrides": self._config.overrides(),
            "queue_depth": None if self._queue is None else self._queue.qsize(),
            "queue_maxsize": self._queue_maxsize,
            "pending_drops": self._pending_drops,
            **self.metrics,
        }
