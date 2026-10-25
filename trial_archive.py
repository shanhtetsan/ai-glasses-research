"""Read trial archives from R2 first, with a local cache fallback."""
from __future__ import annotations

import io
import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Optional

from trial_scorer import build_question_windows, read_session_events
from trial_storage import TrialObjectNotFound, TrialObjectStorage, TrialStorageError


_SAFE_TRIAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,64}$")


class TrialArchiveError(RuntimeError):
    pass


class TrialNotFound(TrialArchiveError):
    pass


def _json_bytes(body: bytes, name: str):
    try:
        return json.loads(body)
    except (UnicodeDecodeError, ValueError) as exc:
        raise TrialArchiveError(f"invalid archived {name}: {exc}") from exc


def _local_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as exc:
        raise TrialArchiveError(f"cannot read {path.name}: {exc}") from exc


def integrate_heading(imu_samples: list[dict]) -> list[dict]:
    """Integrate angular velocity projected onto gravity into relative heading."""
    heading = 0.0
    previous_time: Optional[float] = None
    output: list[dict] = []
    for sample in imu_samples:
        gyro = sample.get("gyro")
        accel = sample.get("accel")
        if not isinstance(gyro, list) or len(gyro) != 3:
            continue
        try:
            vector = [float(value) for value in gyro]
            if isinstance(accel, list) and len(accel) == 3:
                gravity = [float(value) for value in accel]
                magnitude = math.sqrt(sum(value * value for value in gravity))
                yaw_rate = sum(vector[i] * gravity[i] / magnitude for i in range(3)) if magnitude else vector[2]
            else:
                yaw_rate = vector[2]
            timestamp = float(sample.get("uptime_ms")) / 1000.0
        except (TypeError, ValueError):
            continue
        if previous_time is not None:
            delta = timestamp - previous_time
            if 0.0 < delta <= 1.0:
                heading = ((heading + yaw_rate * delta + 180.0) % 360.0) - 180.0
        previous_time = timestamp
        output.append({
            "uptime_ms": sample.get("uptime_ms"),
            "heading_deg": round(heading, 3),
        })
    return output


class TrialArchive:
    def __init__(self, local_root: str | os.PathLike[str], storage: Optional[TrialObjectStorage]) -> None:
        self.local_root = Path(local_root)
        self.storage = storage

    @staticmethod
    def _validate_id(trial_id: str) -> str:
        if not _SAFE_TRIAL_ID.fullmatch(str(trial_id)):
            raise TrialNotFound("trial not found")
        return str(trial_id)

    def _local_dir(self, trial_id: str) -> Path:
        safe = self._validate_id(trial_id)
        path = (self.local_root / safe).resolve()
        root = self.local_root.resolve()
        if root not in path.parents:
            raise TrialNotFound("trial not found")
        return path

    def _remote_key(self, trial_id: str, relative: str) -> str:
        safe = self._validate_id(trial_id)
        clean = Path(relative)
        if clean.is_absolute() or ".." in clean.parts:
            raise TrialNotFound("trial object not found")
        return f"trials/{safe}/{clean.as_posix()}"

    def get_bytes(self, trial_id: str, relative: str) -> tuple[bytes, str, str]:
        key = self._remote_key(trial_id, relative)
        if self.storage is not None:
            try:
                item = self.storage.get_object(key)
                return item.body, item.content_type, "r2"
            except TrialObjectNotFound:
                pass
        path = self._local_dir(trial_id) / relative
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            return path.read_bytes(), "application/octet-stream", "local"
        except FileNotFoundError as exc:
            raise TrialNotFound(f"trial object not found: {relative}") from exc
        except OSError as exc:
            raise TrialArchiveError(f"cannot read trial object: {exc}") from exc

    def get_json(self, trial_id: str, relative: str, default=None):
        try:
            body, _, _ = self.get_bytes(trial_id, relative)
        except TrialNotFound:
            return default
        return _json_bytes(body, relative)

    def _remote_trial_ids(self) -> list[str]:
        if self.storage is None:
            return []
        return [str(item["trial_id"]) for item in self.storage.list_trials()]

    def _local_trial_ids(self) -> list[str]:
        try:
            return [
                path.name for path in self.local_root.iterdir()
                if path.is_dir() and _SAFE_TRIAL_ID.fullmatch(path.name)
                and (path / "meta.json").is_file()
            ]
        except FileNotFoundError:
            return []

    def list_trials(self) -> list[dict]:
        remote_ids = self._remote_trial_ids()
        ids = remote_ids + sorted(set(self._local_trial_ids()) - set(remote_ids))
        rows: list[dict] = []
        for trial_id in ids:
            meta = self.get_json(trial_id, "meta.json", {}) or {}
            summary = self.get_json(trial_id, "summary.json", {}) or {}
            scored = bool(summary)
            if scored:
                turn_count = summary.get("turn_count", 0)
            else:
                # Unscored trials only exist locally (upload requires scoring).
                # Count windows so the list never shows an unscored trial as 0 turns.
                try:
                    turn_count = len(build_question_windows(self._events(trial_id)))
                except (TrialArchiveError, TrialNotFound, OSError, ValueError):
                    turn_count = None
            rows.append({
                "id": trial_id,
                "trial": meta.get("trial"),
                "participant": meta.get("pid"),
                "site": meta.get("site"),
                "condition": meta.get("cond"),
                "prompt_version": meta.get("prompt_version"),
                "scored": scored,
                "turn_count": turn_count,
                # None, not 0.0: an unscored trial has no rate, not a perfect one.
                "bad_guess_rate": summary.get("bad_guess_rate", 0.0) if scored else None,
                "date": meta.get("created_ts"),
                "archived": trial_id in remote_ids,
            })
        return sorted(rows, key=lambda row: str(row.get("date") or ""), reverse=True)

    def _events(self, trial_id: str) -> list[dict]:
        body, _, source = self.get_bytes(trial_id, "session.jsonl")
        if source == "local":
            return read_session_events(self._local_dir(trial_id))
        events = []
        for line_number, line in enumerate(body.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise TrialArchiveError(f"invalid session.jsonl line {line_number}: {exc}") from exc
            events.append(value)
        return events

    def get_trial(self, trial_id: str) -> dict:
        meta = self.get_json(trial_id, "meta.json")
        if meta is None:
            raise TrialNotFound("trial not found")
        ground_truth = self.get_json(trial_id, "ground_truth.json", {}) or {}
        summary = self.get_json(trial_id, "summary.json", {}) or {}
        scores = {}
        try:
            score_body, _, _ = self.get_bytes(trial_id, "scores.jsonl")
            for line in score_body.decode("utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    scores[int(item["window"])] = item
        except TrialNotFound:
            pass

        windows = []
        for window in build_question_windows(self._events(trial_id)):
            frames = []
            for frame in window.frames:
                frames.append({
                    **frame,
                    "url": f"/trials/{trial_id}/{frame['file']}",
                })
            score = scores.get(window.index)
            windows.append({
                "number": window.index,
                "turn_uid": window.turn_uid,
                "question": window.question,
                "answer": window.answer,
                "answer_arrived_under": window.answer_arrived_under,
                "frames": frames,
                "thermal": window.thermal,
                "imu_start": window.imu_start,
                "imu_samples": window.imu_samples,
                "heading": integrate_heading(window.imu_samples),
                "latency": window.latency,
                "session_generation": (window.latency or {}).get("generation"),
                "prompt": {
                    "version": meta.get("prompt_version"),
                    "sha256": meta.get("prompt_sha256"),
                    "sent": window.prompt,
                },
                "target": meta.get("task"),
                "score": score,
            })
        return {
            "id": trial_id,
            "meta": meta,
            "ground_truth": ground_truth,
            "summary": summary,
            "scored": bool(summary),
            "windows": windows,
        }

    def zip_bytes(self, trial_id: str) -> bytes:
        safe = self._validate_id(trial_id)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            if self.storage is not None:
                try:
                    manifest_object = self.storage.get_object(f"trials/{safe}/_complete.json")
                    manifest = _json_bytes(manifest_object.body, "_complete.json")
                    for item in manifest.get("objects", []):
                        relative = str(item.get("key", ""))
                        if relative:
                            body, _, _ = self.get_bytes(safe, relative)
                            archive.writestr(f"{safe}/{relative}", body)
                    return buffer.getvalue()
                except TrialObjectNotFound:
                    pass
            directory = self._local_dir(safe)
            if not directory.is_dir():
                raise TrialNotFound("trial not found")
            for path in sorted(item for item in directory.rglob("*") if item.is_file()):
                archive.write(path, f"{safe}/{path.relative_to(directory).as_posix()}")
        return buffer.getvalue()

