"""The phone's own copy of a trial: ZIP validation, install, and frame linking.

The phone records every frame it captures and uploads them after the trial,
so the archive can show which frames never reached the server. Everything
here is blocking file work; callers run it off the event loop.

A phone frame carries its own sequence number in a JPEG COM segment
("seq=<n>"). The server reads it from the raw bytes on arrival, before the
re-encode drops it, and logs it on frame_recv; the phone's frames/<n>.jpg use
the same n, which is what links the two copies.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

PHONE_DIR = "phone"
MAX_BYTES = 300 * 1024 * 1024
FREE_SPACE_MARGIN = 16 * 1024 * 1024
PUSH_STATE = ".phone_push.json"
MANIFEST_KEYS = ("session_id", "app_version", "device", "frames", "imu_samples", "started_ts", "ended_ts")
REQUIRED_FILES = ("manifest.json", "imu.jsonl", "phone.jsonl")
# Kept locally once R2 has confirmed the phone copy; the rest is R2-only.
KEEP_AFTER_PUSH = ("manifest.json", "phone.jsonl")
MISSING_RANGES_SHOWN = 50

_FRAME_NAME = re.compile(r"^frames/(\d{1,9})\.jpg$")
_SEQ_COMMENT = re.compile(rb"^seq=(\d{1,9})$")
_STAGING_PREFIXES = (".phone-staging-", ".phone-old-")


class PhoneCaptureInvalid(ValueError):
    """The upload is not an acceptable phone copy (422)."""


class PhoneCaptureNoSpace(OSError):
    """Not enough free disk to take the upload (507)."""


# ---------------------------------------------------------------- frame linking

def jpeg_comment_seq(data: bytes) -> Optional[int]:
    """The phone's frame number from a "seq=<n>" COM segment, or None.

    Walks marker segments from SOI and stops at SOS, so it reads the header
    only, never the image data.
    """
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    i = 2
    end = len(data)
    while i + 4 <= end:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:          # fill byte
            i += 1
            continue
        if marker in (0xDA, 0xD9):  # start of scan / end of image
            return None
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:
            i += 2
            continue
        length = (data[i + 2] << 8) | data[i + 3]
        if length < 2 or i + 2 + length > end:
            return None
        if marker == 0xFE:
            match = _SEQ_COMMENT.match(bytes(data[i + 4:i + 2 + length]).strip(b"\x00 \r\n"))
            if match:
                return int(match.group(1))
        i += 2 + length
    return None


def missing_ranges(captured: Iterable[int], received: Iterable[int]) -> list[list[int]]:
    """Phone frame numbers captured but never received, as [first, last] runs."""
    missing = sorted(set(captured) - set(received))
    ranges: list[list[int]] = []
    for seq in missing:
        if ranges and seq == ranges[-1][1] + 1:
            ranges[-1][1] = seq
        else:
            ranges.append([seq, seq])
    return ranges


def frame_seq(name: str) -> Optional[int]:
    """frames/<n>.jpg -> n, relative to the phone folder."""
    match = _FRAME_NAME.match(name)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------- ZIP install

def _bad_name(name: str) -> bool:
    if not name or name.startswith("/") or "\\" in name or ":" in name:
        return True
    return any(part in ("", ".", "..") for part in name.rstrip("/").split("/"))


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return (info.external_attr >> 16) & 0o170000 == 0o120000


def _read_manifest(archive: zipfile.ZipFile, trial_id: str) -> dict:
    try:
        manifest = json.loads(archive.read("manifest.json"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise PhoneCaptureInvalid(f"manifest.json is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PhoneCaptureInvalid("manifest.json must be a JSON object")
    missing = [key for key in MANIFEST_KEYS if key not in manifest]
    if missing:
        raise PhoneCaptureInvalid("manifest.json is missing " + ", ".join(missing))
    if manifest["session_id"] != trial_id:
        raise PhoneCaptureInvalid(
            f"manifest session_id {manifest['session_id']!r} does not match trial {trial_id!r}"
        )
    for key in ("frames", "imu_samples"):
        value = manifest[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PhoneCaptureInvalid(f"manifest {key} must be a non-negative integer")
    return manifest


def _count_jsonl(archive: zipfile.ZipFile, name: str) -> int:
    count = 0
    with archive.open(name) as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                raise PhoneCaptureInvalid(f"{name} line {number} is not valid JSON") from exc
            count += 1
    return count


def validate(archive: zipfile.ZipFile, trial_id: str, max_bytes: int = MAX_BYTES) -> tuple[dict, list]:
    """Check names, sizes and counts. Returns (manifest, entries to extract)."""
    entries: list[zipfile.ZipInfo] = []
    seen_names: set[str] = set()
    seen_seqs: set[int] = set()
    total = 0
    for info in archive.infolist():
        name = info.filename
        if _bad_name(name):
            raise PhoneCaptureInvalid(f"unsafe path in ZIP: {name!r}")
        if _is_symlink(info):
            raise PhoneCaptureInvalid(f"symlink in ZIP: {name!r}")
        if info.flag_bits & 0x1:
            raise PhoneCaptureInvalid(f"encrypted entry in ZIP: {name!r}")
        if name in seen_names:
            raise PhoneCaptureInvalid(f"duplicate entry in ZIP: {name!r}")
        seen_names.add(name)
        if info.is_dir():
            if name != "frames/":
                raise PhoneCaptureInvalid(f"unexpected folder in ZIP: {name!r}")
            continue
        seq = frame_seq(name)
        if seq is not None:
            if seq in seen_seqs:
                raise PhoneCaptureInvalid(f"frame {seq} appears twice")
            seen_seqs.add(seq)
        elif name not in REQUIRED_FILES:
            raise PhoneCaptureInvalid(f"unexpected file in ZIP: {name!r}")
        total += info.file_size
        if total > max_bytes:
            raise PhoneCaptureInvalid(f"ZIP unpacks to more than {max_bytes} bytes")
        entries.append(info)
    absent = [name for name in REQUIRED_FILES if name not in seen_names]
    if absent:
        raise PhoneCaptureInvalid("ZIP is missing " + ", ".join(absent))
    manifest = _read_manifest(archive, trial_id)
    if len(seen_seqs) != manifest["frames"]:
        raise PhoneCaptureInvalid(
            f"manifest says {manifest['frames']} frames, ZIP has {len(seen_seqs)}"
        )
    imu_count = _count_jsonl(archive, "imu.jsonl")
    if imu_count != manifest["imu_samples"]:
        raise PhoneCaptureInvalid(
            f"manifest says {manifest['imu_samples']} IMU samples, imu.jsonl has {imu_count}"
        )
    return manifest, entries


def _extract(archive: zipfile.ZipFile, entries: list, staging: Path) -> None:
    (staging / "frames").mkdir(parents=True)
    for info in entries:
        target = staging / info.filename
        with archive.open(info) as source, open(target, "wb") as sink:
            head = source.read(2)
            if frame_seq(info.filename) is not None and head != b"\xff\xd8":
                raise PhoneCaptureInvalid(f"{info.filename} is not a JPEG")
            sink.write(head)
            shutil.copyfileobj(source, sink, 1024 * 1024)


def _clear_leftovers(trial_dir: Path) -> None:
    for child in trial_dir.iterdir():
        if child.is_dir() and child.name.startswith(_STAGING_PREFIXES):
            shutil.rmtree(child, ignore_errors=True)


def install(zip_path: str, trial_dir: str, trial_id: str, max_bytes: int = MAX_BYTES) -> dict:
    """Validate the ZIP, extract it beside the trial, and swap it in as phone/.

    Nothing is written to phone/ unless every check passes: the ZIP unpacks
    into a dot-named staging folder (which R2 uploads and the archive skip),
    then two renames replace phone/. The caller holds the trial's upload lock.
    """
    directory = Path(trial_dir)
    _clear_leftovers(directory)
    token = secrets.token_hex(4)
    staging = directory / f".phone-staging-{token}"
    try:
        try:
            archive = zipfile.ZipFile(zip_path)
        except zipfile.BadZipFile as exc:
            raise PhoneCaptureInvalid(f"not a ZIP file: {exc}") from exc
        with archive:
            manifest, entries = validate(archive, trial_id, max_bytes)
            needed = sum(info.file_size for info in entries) + FREE_SPACE_MARGIN
            if shutil.disk_usage(directory).free < needed:
                raise PhoneCaptureNoSpace(f"not enough free space on the trial volume for {needed} bytes")
            try:
                _extract(archive, entries, staging)
            except zipfile.BadZipFile as exc:
                raise PhoneCaptureInvalid(f"corrupt ZIP entry: {exc}") from exc
        final = directory / PHONE_DIR
        old = directory / f".phone-old-{token}"
        if final.exists():
            os.rename(final, old)
        os.rename(staging, final)
        shutil.rmtree(old, ignore_errors=True)
        write_push_state(trial_dir, {"status": "pending", "frames": manifest["frames"]})
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------- R2 push state

def write_push_state(trial_dir: str, state: dict) -> None:
    state = {**state, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = Path(trial_dir) / PUSH_STATE
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_push_state(trial_dir: str) -> Optional[dict]:
    try:
        return json.loads((Path(trial_dir) / PUSH_STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def push_pending(trial_dir: str) -> bool:
    """A phone copy is on disk and R2 has not confirmed it."""
    if not (Path(trial_dir) / PHONE_DIR).is_dir():
        return False
    state = read_push_state(trial_dir) or {}
    return state.get("status") != "done"


def prune_after_push(trial_dir: str) -> None:
    """R2 has the phone copy: keep only the small files on the 1 GB volume."""
    phone = Path(trial_dir) / PHONE_DIR
    for child in phone.iterdir():
        if child.name in KEEP_AFTER_PUSH:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
