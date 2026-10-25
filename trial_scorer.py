"""Offline scoring for append-only trial recordings.

Question windows are defined by event order: each window begins at a
``speech_start`` and ends immediately before the next ``speech_start`` (or at
the end of the session).  Turn UIDs are evidence, not window boundaries; this
is what preserves replies which Gemini Live files under a server-opened turn.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol


RUBRIC_VERSION = "blind-visibility-v2"  # v2: judge sees thermal evidence
LABELS = ("right", "bad_guess", "dead_refusal", "useful_refusal")
SPATIAL_FRAMES = ("body-relative", "object-relative", "room-relative", "mixed", "none")
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"

# Free-tier gemini-2.5-flash is 5 requests/minute. One judge call per question
# window with no pacing exhausts that partway through a 9-window trial, so the
# scorer paces itself to this ceiling and honours the server's own retryDelay.
DEFAULT_JUDGE_RPM = 5.0
DEFAULT_JUDGE_MAX_ATTEMPTS = 5
MAX_RETRY_SLEEP_SEC = 120.0
_RETRY_STATUSES = ("RESOURCE_EXHAUSTED", "UNAVAILABLE")
_RETRY_CODES = (429, 503)
_RETRY_DELAY_RE = re.compile(r"retry[-_]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s?", re.I)

JUDGE_INSTRUCTION = """You are scoring one spoken smart-glasses question independently.
The user is blind or has low vision, so an unsupported confident claim is more harmful than
an admission of uncertainty. Use only the submitted camera frames, the thermal evidence, and
the room ground truth. The ground truth is an answer key for this offline evaluation; it was
not available to the live assistant.

The glasses also carry a thermal camera. THERMAL SENT TO ASSISTANT is the exact thermal text
the live assistant received for this question; temperatures, hotspots, and hotspot positions
that match it are supported by the sensor, not hallucinations, even when nothing hot is visible
in the RGB frames. THERMAL SNAPSHOT AT SPEECH START is the recorder's reading of the same sensor
when the question began. Temperature claims that match neither are unsupported.

Labels:
- right: the target is visible in this window's frames and the answer describes it correctly.
- bad_guess: the answer names or locates something not present, asserts depth/distance, or
  asserts clearance/safety not established by the frames.
- dead_refusal: it says it cannot see the target and gives nothing else actionable.
- useful_refusal: it says it cannot see the target and also gives actionable visible context,
  such as visible objects, spatial structure, or the direction in which the room opens.

Spatial frame must be exactly one of: body-relative, object-relative, room-relative, mixed,
none. Body-relative uses the user's body/viewpoint (left, right, ahead). Object-relative
locates something relative to another object. Room-relative uses stable room structure.
Mixed means two or more of those frames are used. Keep why to one short line. Return only the
requested structured object."""
PROMPT_SHA256 = hashlib.sha256(JUDGE_INSTRUCTION.encode("utf-8")).hexdigest()

_NUMBER = (
    r"(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|a|an|"
    r"few|couple(?:\s+of)?)"
)
_UNIT = r"(?:met(?:er|re)s?|feet|foot|inches?|centimet(?:er|re)s?|cm|yards?|steps?|paces?)"
_DISTANCE_PATTERNS = (
    # An approximation word alone is not enough: real answers also say
    # "roughly 22 degrees" and "about 91 degrees" for thermal readings.
    re.compile(rf"\b(?:about|roughly)\s+{_NUMBER}(?:[ -]+{_UNIT}(?:\s+away)?|\s+away)\b", re.I),
    re.compile(rf"\b{_NUMBER}[ -]+{_UNIT}(?:\s+away)?\b", re.I),
    re.compile(rf"\b{_UNIT}\b", re.I),
)
_WORDS_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


class TrialScoringError(RuntimeError):
    """The recording is incomplete or cannot be scored safely."""


class JudgeRetryableError(TrialScoringError):
    """A judge call failed in a way that is worth retrying (429 / 503).

    ``retry_after`` is the server's own requested delay in seconds when it
    supplied one, otherwise None and the caller backs off exponentially.
    """

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _error_code(exc: BaseException) -> Optional[int]:
    for attribute in ("code", "status_code", "http_status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _error_status(exc: BaseException) -> str:
    value = getattr(exc, "status", None)
    return value.upper() if isinstance(value, str) else ""


def is_retryable_judge_error(exc: BaseException) -> bool:
    """429 RESOURCE_EXHAUSTED and 503 UNAVAILABLE are transient; retry those."""
    if isinstance(exc, JudgeRetryableError):
        return True
    if _error_code(exc) in _RETRY_CODES or _error_status(exc) in _RETRY_STATUSES:
        return True
    text = str(exc).upper()
    if any(status in text for status in _RETRY_STATUSES):
        return True
    # Bare "429"/"503" only when it reads as a status, not as part of a number.
    return bool(re.search(r"(?<!\d)(?:429|503)(?!\d)", text))


def retry_delay_seconds(exc: BaseException) -> Optional[float]:
    """The server's requested retryDelay, from a structured field or the text."""
    if isinstance(exc, JudgeRetryableError) and exc.retry_after is not None:
        return exc.retry_after

    def _walk(value: Any, depth: int = 0) -> Optional[float]:
        if depth > 6:
            return None
        if isinstance(value, dict):
            for key, inner in value.items():
                if isinstance(key, str) and key.lower().replace("_", "") == "retrydelay":
                    parsed = _parse_delay(inner)
                    if parsed is not None:
                        return parsed
                found = _walk(inner, depth + 1)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for inner in value:
                found = _walk(inner, depth + 1)
                if found is not None:
                    return found
        return None

    for attribute in ("details", "response_json", "error", "body"):
        found = _walk(getattr(exc, attribute, None))
        if found is not None:
            return found
    match = _RETRY_DELAY_RE.search(str(exc))
    return _parse_delay(match.group(1)) if match else None


def _parse_delay(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, float(value))
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*s?\s*", value)
        if match:
            return max(0.0, float(match.group(1)))
    return None


def _judge_rpm() -> float:
    raw = os.getenv("TRIAL_JUDGE_RPM", "").strip()
    if not raw:
        return DEFAULT_JUDGE_RPM
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_JUDGE_RPM


def _judge_max_attempts() -> int:
    raw = os.getenv("TRIAL_JUDGE_MAX_ATTEMPTS", "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_JUDGE_MAX_ATTEMPTS
    except ValueError:
        return DEFAULT_JUDGE_MAX_ATTEMPTS


class RateLimiter:
    """Spaces calls to at most ``rpm`` per minute. rpm <= 0 disables pacing."""

    def __init__(self, rpm: Optional[float] = None, sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.rpm = _judge_rpm() if rpm is None else float(rpm)
        self._interval = 60.0 / self.rpm if self.rpm > 0 else 0.0
        self._sleep = sleep
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> float:
        if self._interval <= 0:
            return 0.0
        with self._lock:
            now = self._monotonic()
            wait = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self._interval
        if wait > 0:
            self._sleep(wait)
        return wait


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with full jitter, in seconds."""
    ceiling = min(MAX_RETRY_SLEEP_SEC, 2.0 ** attempt)
    return random.uniform(0.0, ceiling)


class WindowJudge(Protocol):
    model: str

    def judge(self, question: str, answer: str, frame_paths: list[Path], ground_truth: dict,
              thermal: dict) -> dict:
        ...


def thermal_evidence(window: "QuestionWindow") -> dict:
    """The thermal facts behind a window: what Gemini was sent, and the speech_start snapshot."""
    sent = (window.prompt or {}).get("thermal_text")
    return {
        "sent_to_assistant": sent if isinstance(sent, str) and sent.strip() else None,
        "snapshot_at_speech_start": window.thermal or None,
    }


def judge_evidence_text(question: str, answer: str, ground_truth: dict, thermal: dict) -> str:
    sent = thermal.get("sent_to_assistant")
    snapshot = thermal.get("snapshot_at_speech_start")
    return (
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\n"
        "THERMAL SENT TO ASSISTANT:\n" + (sent or "none — no thermal text was sent for this question") + "\n\n"
        "THERMAL SNAPSHOT AT SPEECH START:\n"
        + (json.dumps(snapshot, ensure_ascii=False, sort_keys=True) if snapshot else "none recorded") + "\n\n"
        "GROUND TRUTH:\n" + json.dumps(ground_truth, ensure_ascii=False, sort_keys=True)
    )


@dataclass(frozen=True)
class QuestionWindow:
    index: int
    turn_uid: Optional[str]
    question: str
    answer: str
    answer_arrived_under: Optional[str]
    frames: list[dict]
    thermal: Optional[dict]
    imu_start: Optional[dict]
    imu_samples: list[dict]
    latency: Optional[dict]
    prompt: Optional[dict]


def detect_distance_units(answer: str) -> Optional[str]:
    """Return the earliest deterministic distance match, preserving its text."""
    matches = [match for pattern in _DISTANCE_PATTERNS if (match := pattern.search(answer or ""))]
    if not matches:
        return None
    return min(matches, key=lambda item: (item.start(), -len(item.group(0)))).group(0)


def count_words(answer: str) -> int:
    return len(_WORDS_RE.findall(answer or ""))


def deterministic_spatial_frame(answer: str) -> str:
    """Conservative fallback used only when distance forces a label without a judge."""
    text = (answer or "").lower()
    body = bool(re.search(
        r"\b(?:your|to your|ahead|straight ahead|behind you|in front of you|away|"
        r"on your left|on your right|to the left|to the right)\b", text
    ))
    room = bool(re.search(
        r"\b(?:room|wall|corner|doorway|entrance|exit|ceiling|floor|far end|near end)\b", text
    ))
    obj = bool(re.search(
        r"\b(?:on|under|above|below|beside|next to|between|behind|in front of)\s+"
        r"(?:the|a|an)\s+(?!room\b|wall\b|doorway\b|entrance\b|floor\b|ceiling\b)", text
    ))
    found = [name for name, present in (
        ("body-relative", body), ("object-relative", obj), ("room-relative", room)
    ) if present]
    return "mixed" if len(found) > 1 else (found[0] if found else "none")


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise TrialScoringError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrialScoringError(f"{path.name} must contain a JSON object")
    return value


def read_session_events(trial_dir: Path) -> list[dict]:
    path = trial_dir / "session.jsonl"
    events: list[dict] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except ValueError as exc:
                    raise TrialScoringError(
                        f"session.jsonl line {line_number} is invalid JSON: {exc}"
                    ) from exc
                if not isinstance(event, dict):
                    raise TrialScoringError(f"session.jsonl line {line_number} is not an object")
                events.append(event)
    except OSError as exc:
        raise TrialScoringError(f"cannot read session.jsonl: {exc}") from exc
    return events


def _without_base(event: Optional[dict]) -> Optional[dict]:
    if event is None:
        return None
    return {key: value for key, value in event.items() if key not in {"t", "ts", "ns", "src", "turn"}}


def build_question_windows(events: list[dict]) -> list[QuestionWindow]:
    starts = [index for index, event in enumerate(events) if event.get("t") == "speech_start"]
    windows: list[QuestionWindow] = []
    for position, start_index in enumerate(starts):
        stop_index = starts[position + 1] if position + 1 < len(starts) else len(events)
        chunk = events[start_index:stop_index]
        turn_uid = events[start_index].get("turn")

        input_parts = [
            str(event.get("text", "")).strip()
            for event in chunk
            if event.get("t") == "response_text"
            and event.get("kind") == "input_transcription"
            and event.get("turn") == turn_uid
            and str(event.get("text", "")).strip()
        ]
        output_events = [
            event for event in chunk
            if event.get("t") == "response_text"
            and event.get("kind") == "output_transcription"
            and str(event.get("text", "")).strip()
        ]

        # Compatibility with early Part 1 recordings which may only carry
        # question/reply on turn_end rather than duplicated response_text rows.
        if not input_parts:
            input_parts = [
                str(event.get("question", "")).strip()
                for event in chunk
                if event.get("t") == "turn_end" and event.get("turn") == turn_uid
                and str(event.get("question", "")).strip()
            ]
        if not output_events:
            output_events = [
                {**event, "text": event.get("reply")}
                for event in chunk
                if event.get("t") == "turn_end" and str(event.get("reply", "")).strip()
            ]

        other_uids: list[str] = []
        for event in output_events:
            owner = event.get("turn")
            if owner and owner != turn_uid and owner not in other_uids:
                other_uids.append(owner)

        frames = [
            {
                "seq": event.get("seq"), "file": event.get("file"), "turn": event.get("turn"),
                "ts": event.get("ts"), "source": event.get("source"), "age_ms": event.get("age_ms"),
            }
            for event in chunk if event.get("t") == "frame_sent" and event.get("file")
        ]
        thermal_event = next((
            event for event in chunk
            if event.get("t") == "thermal" and event.get("turn") == turn_uid
        ), None)
        imu_start_event = next((
            event for event in chunk
            if event.get("t") == "imu" and event.get("turn") == turn_uid
            and event.get("at") == "speech_start"
        ), None)
        imu_samples = [
            _without_base(event) for event in chunk
            if event.get("t") == "imu" and event.get("turn") == turn_uid
        ]
        latency_event = next((
            event for event in chunk
            if event.get("t") == "turn_end" and event.get("turn") == turn_uid
        ), None)
        prompt_event = next((
            event for event in chunk
            if event.get("t") == "prompt_sent" and event.get("turn") == turn_uid
        ), None)

        windows.append(QuestionWindow(
            index=position + 1,
            turn_uid=turn_uid,
            question=" ".join(input_parts).strip(),
            answer=" ".join(str(event.get("text", "")).strip() for event in output_events).strip(),
            answer_arrived_under=",".join(other_uids) or None,
            frames=frames,
            thermal=_without_base(thermal_event),
            imu_start=_without_base(imu_start_event),
            imu_samples=[sample for sample in imu_samples if sample is not None],
            latency=_without_base(latency_event),
            prompt=_without_base(prompt_event),
        ))
    return windows


class GeminiWindowJudge:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.model = (
            model
            or os.getenv("TRIAL_JUDGE_MODEL")
            or os.getenv("GEMINI_MODEL_ID")
            or DEFAULT_JUDGE_MODEL
        )
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise TrialScoringError("google-genai is required for trial scoring") from exc
            api_key = self._api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise TrialScoringError("GEMINI_API_KEY is required for trial scoring")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def judge(self, question: str, answer: str, frame_paths: list[Path], ground_truth: dict,
              thermal: dict) -> dict:
        try:
            from google.genai import types
        except ImportError as exc:
            raise TrialScoringError("google-genai is required for trial scoring") from exc

        evidence = judge_evidence_text(question, answer, ground_truth, thermal)
        contents: list[Any] = [JUDGE_INSTRUCTION, evidence]
        for path in frame_paths:
            try:
                contents.append(types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg"))
            except OSError as exc:
                raise TrialScoringError(f"cannot read frame {path.name}: {exc}") from exc

        schema = {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": list(LABELS)},
                "spatial_frame": {"type": "string", "enum": list(SPATIAL_FRAMES)},
                "why": {"type": "string"},
            },
            "required": ["label", "spatial_frame", "why"],
            "additionalProperties": False,
        }
        try:
            response = self._get_client().models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_json_schema=schema,
                ),
            )
            result = json.loads(response.text)
        except TrialScoringError:
            raise
        except Exception as exc:
            # Preserve retryability: wrapping a 429 in a plain TrialScoringError
            # would hide the quota signal (and the server's retryDelay) from the
            # retry loop in judge_window.
            if is_retryable_judge_error(exc):
                raise JudgeRetryableError(
                    f"Gemini judge failed: {type(exc).__name__}: {exc}",
                    retry_delay_seconds(exc),
                ) from exc
            raise TrialScoringError(f"Gemini judge failed: {type(exc).__name__}: {exc}") from exc
        return _validate_judgment(result)


def judge_window(
    judge: WindowJudge,
    question: str,
    answer: str,
    frame_paths: list[Path],
    ground_truth: dict,
    thermal: dict,
    *,
    limiter: Optional[RateLimiter] = None,
    max_attempts: Optional[int] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict, int]:
    """Call the judge with pacing and bounded retries. Returns (judgment, attempts)."""
    attempts_allowed = _judge_max_attempts() if max_attempts is None else max(1, int(max_attempts))
    last: BaseException = TrialScoringError("judge was never called")
    attempt = 0
    for attempt in range(1, attempts_allowed + 1):
        if limiter is not None:
            limiter.acquire()
        try:
            return _validate_judgment(judge.judge(question, answer, frame_paths, ground_truth, thermal)), attempt
        except Exception as exc:
            last = exc
            if attempt >= attempts_allowed or not is_retryable_judge_error(exc):
                break
            requested = retry_delay_seconds(exc)
            delay = _backoff_delay(attempt) if requested is None else min(requested, MAX_RETRY_SLEEP_SEC)
            print(
                f"[TRIAL-SCORE] judge retry {attempt}/{attempts_allowed} in {delay:.1f}s: "
                f"{type(exc).__name__}: {str(exc)[:160]}",
                flush=True,
            )
            sleep(delay)
    raise TrialScoringError(
        f"judge failed after {attempt} attempt(s): {type(last).__name__}: {last}"
    ) from last


def _validate_judgment(result: Any) -> dict:
    if not isinstance(result, dict):
        raise TrialScoringError("judge returned a non-object")
    label = result.get("label")
    spatial_frame = result.get("spatial_frame")
    why = " ".join(str(result.get("why", "")).split())
    if label not in LABELS:
        raise TrialScoringError(f"judge returned invalid label: {label!r}")
    if spatial_frame not in SPATIAL_FRAMES:
        raise TrialScoringError(f"judge returned invalid spatial_frame: {spatial_frame!r}")
    if not why:
        raise TrialScoringError("judge returned an empty reason")
    return {"label": label, "spatial_frame": spatial_frame, "why": why[:500]}


def score_window(
    window: QuestionWindow,
    trial_dir: Path,
    ground_truth: dict,
    judge: WindowJudge,
    *,
    limiter: Optional[RateLimiter] = None,
    max_attempts: Optional[int] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    distance = detect_distance_units(window.answer)
    consulted = distance is None
    if distance is not None:
        judgment = {
            "label": "bad_guess",
            "spatial_frame": deterministic_spatial_frame(window.answer),
            "why": f"Deterministic distance detector matched {distance!r}; distance claims are forced to bad_guess.",
        }
    else:
        frame_paths = [trial_dir / str(frame["file"]) for frame in window.frames]
        judgment, _ = judge_window(
            judge, window.question, window.answer, frame_paths, ground_truth,
            thermal_evidence(window), limiter=limiter, max_attempts=max_attempts, sleep=sleep,
        )

    return {
        "window": window.index,
        "turn_uid": window.turn_uid,
        "question": window.question,
        "answer": window.answer,
        "answer_arrived_under": window.answer_arrived_under,
        "frame_count": len(window.frames),
        "distance_units": distance,
        "words": count_words(window.answer),
        "spatial_frame": judgment["spatial_frame"],
        "label": judgment["label"],
        "why": judgment["why"],
        "judge_model": judge.model,
        "judge_consulted": consulted,
        "rubric_version": RUBRIC_VERSION,
        "prompt_sha256": PROMPT_SHA256,
    }


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _csv_text(scores: list[dict]) -> str:
    if not scores:
        return ""
    import io
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(scores[0]))
    writer.writeheader()
    writer.writerows(scores)
    return output.getvalue()


def _summary(scores: list[dict], events: list[dict], unscored: Optional[list[dict]] = None) -> dict:
    counts = Counter(score["label"] for score in scores)
    total = len(scores)
    summary = {
        "turn_count": total,
        "counts": {label: counts[label] for label in LABELS},
        "bad_guess_rate": counts["bad_guess"] / total if total else 0.0,
        "useful_refusal_rate": counts["useful_refusal"] / total if total else 0.0,
        "distance_flags": sum(score["distance_units"] is not None for score in scores),
        "misattributed_turns": sum(score["answer_arrived_under"] is not None for score in scores),
        "dropped_events": sum(
            int(event.get("dropped_count") or 0) for event in events if event.get("t") == "drop"
        ),
    }
    # Added only when something actually failed, so a clean trial's summary.json
    # keeps exactly the keys (and the rates' denominators) it has always had.
    if unscored:
        summary["unscored"] = unscored
        summary["unscored_count"] = len(unscored)
        summary["window_count"] = total + len(unscored)
    return summary


def _window_fingerprint(window: QuestionWindow) -> tuple:
    """What a stored score must still match for it to be reusable."""
    return (window.turn_uid, window.question, window.answer, len(window.frames))


def _score_fingerprint(score: dict) -> tuple:
    return (
        score.get("turn_uid"),
        score.get("question") or "",
        score.get("answer") or "",
        int(score.get("frame_count") or 0),
    )


def load_existing_scores(trial_dir: Path) -> dict[int, dict]:
    """Previously written per-window scores, keyed by window index.

    Only records produced by this rubric and judge prompt are returned, so a
    rubric change still re-judges everything.
    """
    path = trial_dir / "scores.jsonl"
    existing: dict[int, dict] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return existing
    except OSError:
        return existing
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("label") not in LABELS:
            continue
        if record.get("rubric_version") != RUBRIC_VERSION or record.get("prompt_sha256") != PROMPT_SHA256:
            continue
        try:
            existing[int(record["window"])] = record
        except (KeyError, TypeError, ValueError):
            continue
    return existing


def _prune_unscored_frames(trial_dir: Path, windows: list[QuestionWindow]) -> list[str]:
    keep = {
        str(frame["file"]) for window in windows for frame in window.frames if frame.get("file")
    }
    frames_dir = trial_dir / "frames"
    removed: list[str] = []
    if not frames_dir.is_dir():
        return removed
    for path in frames_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        relative = path.relative_to(trial_dir).as_posix()
        if relative not in keep:
            path.unlink()
            removed.append(relative)
    return sorted(removed)


def score_trial_folder(
    trial_dir: str | os.PathLike[str],
    *,
    judge: Optional[WindowJudge] = None,
    prune_frames: bool = True,
    resume: bool = True,
    limiter: Optional[RateLimiter] = None,
    max_attempts: Optional[int] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Score one closed trial and atomically replace its derived score files.

    Judge calls are paced to TRIAL_JUDGE_RPM and retried on 429/503. A window
    that still cannot be judged is recorded in ``summary["unscored"]`` with the
    reason and scoring continues, so a trial is never left half-scored with no
    record of why. With ``resume`` (the default) a window whose stored score
    still matches the log is reused instead of re-judged, which makes a re-run
    after a quota failure cost only the windows that are actually missing.
    """
    directory = Path(trial_dir)
    if not directory.is_dir():
        raise TrialScoringError(f"trial not found: {directory.name}")
    ground_truth = _read_json(directory / "ground_truth.json")
    events = read_session_events(directory)
    if not any(event.get("t") == "session_end" for event in events):
        raise TrialScoringError("trial is still open (session_end is missing)")
    windows = build_question_windows(events)
    if not windows:
        raise TrialScoringError("session contains no question windows")

    active_judge = judge or GeminiWindowJudge()
    active_limiter = limiter if limiter is not None else RateLimiter(sleep=sleep)
    previous = load_existing_scores(directory) if resume else {}

    scores: list[dict] = []
    unscored: list[dict] = []
    reused = 0
    for window in windows:
        carried = previous.get(window.index)
        if carried is not None and _score_fingerprint(carried) == _window_fingerprint(window):
            scores.append(carried)
            reused += 1
            continue
        try:
            scores.append(score_window(
                window, directory, ground_truth, active_judge,
                limiter=active_limiter, max_attempts=max_attempts, sleep=sleep,
            ))
        except Exception as exc:
            # Permanent for this window only. Record why and keep going: losing
            # the remaining windows to one bad call is what this guards against.
            reason = f"{type(exc).__name__}: {exc}"
            print(f"[TRIAL-SCORE] window {window.index} unscored — {reason[:200]}", flush=True)
            unscored.append({
                "window": window.index,
                "turn_uid": window.turn_uid,
                "question": window.question,
                "reason": reason[:500],
            })

    summary = _summary(scores, events, unscored)

    jsonl = "".join(json.dumps(score, ensure_ascii=False, separators=(",", ":")) + "\n" for score in scores)
    _atomic_write_text(directory / "scores.jsonl", jsonl)
    _atomic_write_text(directory / "scores.csv", _csv_text(scores))
    _atomic_write_text(
        directory / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    removed = _prune_unscored_frames(directory, windows) if prune_frames else []
    return {
        "trial_id": directory.name,
        "summary": summary,
        "pruned_frames": removed,
        "scored_windows": len(scores),
        "reused_windows": reused,
        "unscored_windows": unscored,
    }

