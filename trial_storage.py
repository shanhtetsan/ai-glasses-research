"""Swappable object storage for scored trial folders.

Cloudflare R2 is exposed through its S3-compatible endpoint.  The application
depends only on the three methods on ``TrialObjectStorage`` so another S3
endpoint (or an adapter for GCS) can replace it without touching routes.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol


TRIAL_PREFIX = "trials"
UPLOAD_RECEIPT = ".upload.json"
COMPLETE_OBJECT = "_complete.json"
_SAFE_TRIAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,64}$")


class TrialStorageError(RuntimeError):
    pass


class TrialStorageUnavailable(TrialStorageError):
    pass


class TrialObjectNotFound(TrialStorageError):
    pass


@dataclass(frozen=True)
class StoredObject:
    key: str
    body: bytes
    content_type: str
    etag: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)


class TrialObjectStorage(Protocol):
    def put_folder(self, folder: str | os.PathLike[str]) -> dict:
        ...

    def get_object(self, key: str) -> StoredObject:
        ...

    def list_trials(self) -> list[dict]:
        ...


@dataclass(frozen=True)
class S3StorageConfig:
    endpoint_url: str
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    bucket: str
    region: str = "auto"

    @classmethod
    def from_r2_env(cls) -> Optional["S3StorageConfig"]:
        names = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
        values = {name: os.getenv(name, "").strip() for name in names}
        if not any(values.values()):
            return None
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise TrialStorageUnavailable("incomplete R2 configuration; missing " + ", ".join(missing))
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
        if not endpoint:
            endpoint = f"https://{values['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
        return cls(
            endpoint_url=endpoint,
            access_key_id=values["R2_ACCESS_KEY_ID"],
            secret_access_key=values["R2_SECRET_ACCESS_KEY"],
            bucket=values["R2_BUCKET"],
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _content_type(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        return "application/x-ndjson"
    if path.suffix.lower() == ".csv":
        return "text/csv; charset=utf-8"
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


class S3TrialStorage:
    def __init__(
        self,
        config: S3StorageConfig,
        *,
        client: Any = None,
        attempts: int = 4,
        base_backoff_sec: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._client = client
        self.attempts = max(1, int(attempts))
        self.base_backoff_sec = max(0.0, float(base_backoff_sec))
        self._sleep = sleep

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise TrialStorageUnavailable("boto3 is required for R2 trial storage") from exc
            self._client = boto3.client(
                service_name="s3",
                endpoint_url=self.config.endpoint_url,
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                region_name=self.config.region,
            )
        return self._client

    def _retry(self, operation: str, callback: Callable[[], Any]) -> Any:
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.attempts + 1):
            try:
                return callback()
            except Exception as exc:
                response = getattr(exc, "response", None)
                error = response.get("Error", {}) if isinstance(response, dict) else {}
                if str(error.get("Code", "")) in {"404", "NoSuchKey", "NotFound"}:
                    raise TrialObjectNotFound(operation) from exc
                last_error = exc
                if attempt < self.attempts:
                    self._sleep(min(self.base_backoff_sec * (2 ** (attempt - 1)), 8.0))
        raise TrialStorageError(
            f"{operation} failed after {self.attempts} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def _put_bytes(self, key: str, body: bytes, content_type: str, sha256: str) -> None:
        self._retry(
            f"put {key}",
            lambda: self._get_client().put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
                Metadata={"sha256": sha256},
            ),
        )

    def _verify(self, key: str, size: int, sha256: str) -> None:
        head = self._retry(
            f"verify {key}",
            lambda: self._get_client().head_object(Bucket=self.config.bucket, Key=key),
        )
        remote_size = int(head.get("ContentLength", -1))
        remote_sha = str((head.get("Metadata") or {}).get("sha256", ""))
        if remote_size != size or remote_sha != sha256:
            raise TrialStorageError(
                f"verification failed for {key}: local size/sha256={size}/{sha256}, "
                f"remote={remote_size}/{remote_sha or 'missing'}"
            )

    def put_folder(self, folder: str | os.PathLike[str]) -> dict:
        directory = Path(folder)
        trial_id = directory.name
        if not directory.is_dir():
            raise TrialStorageError(f"trial folder not found: {directory}")
        if not _SAFE_TRIAL_ID.fullmatch(trial_id):
            raise TrialStorageError(f"unsafe trial id: {trial_id!r}")

        files = sorted(
            path for path in directory.rglob("*")
            if path.is_file() and path.name != UPLOAD_RECEIPT and not path.name.endswith(".tmp")
        )
        if not files:
            raise TrialStorageError(f"trial folder is empty: {trial_id}")

        prefix = f"{TRIAL_PREFIX}/{trial_id}/"
        objects: list[dict] = []
        for path in files:
            relative = path.relative_to(directory).as_posix()
            key = prefix + relative
            body = path.read_bytes()
            digest = hashlib.sha256(body).hexdigest()
            self._put_bytes(key, body, _content_type(path), digest)
            objects.append({"key": relative, "size": len(body), "sha256": digest})

        # Verification happens before the completion marker is published, so
        # list_trials never advertises a partially uploaded first attempt.
        for item in objects:
            self._verify(prefix + item["key"], item["size"], item["sha256"])

        manifest = {
            "trial_id": trial_id,
            "prefix": prefix,
            "uploaded_at": _utc_now(),
            "object_count": len(objects),
            "total_bytes": sum(item["size"] for item in objects),
            "objects": objects,
        }
        manifest_body = (json.dumps(manifest, separators=(",", ":")) + "\n").encode("utf-8")
        manifest_sha = hashlib.sha256(manifest_body).hexdigest()
        manifest_key = prefix + COMPLETE_OBJECT
        self._put_bytes(manifest_key, manifest_body, "application/json", manifest_sha)
        self._verify(manifest_key, len(manifest_body), manifest_sha)
        return manifest

    def get_object(self, key: str) -> StoredObject:
        clean = str(key).strip().lstrip("/")
        if not clean.startswith(TRIAL_PREFIX + "/") or ".." in Path(clean).parts:
            raise TrialStorageError("invalid trial object key")
        response = self._retry(
            f"get {clean}",
            lambda: self._get_client().get_object(Bucket=self.config.bucket, Key=clean),
        )
        body = response["Body"].read()
        return StoredObject(
            key=clean,
            body=body,
            content_type=response.get("ContentType") or "application/octet-stream",
            etag=response.get("ETag"),
            metadata=dict(response.get("Metadata") or {}),
        )

    def list_trials(self) -> list[dict]:
        paginator = self._get_client().get_paginator("list_objects_v2")
        manifests: list[dict] = []
        try:
            pages = paginator.paginate(Bucket=self.config.bucket, Prefix=TRIAL_PREFIX + "/")
            keys = [
                item.get("Key") for page in pages for item in page.get("Contents", [])
                if str(item.get("Key", "")).endswith("/" + COMPLETE_OBJECT)
            ]
            for key in keys:
                try:
                    value = json.loads(self.get_object(str(key)).body)
                except (TrialStorageError, ValueError, TypeError):
                    continue
                if isinstance(value, dict) and _SAFE_TRIAL_ID.fullmatch(str(value.get("trial_id", ""))):
                    manifests.append(value)
        except Exception as exc:
            if isinstance(exc, TrialStorageError):
                raise
            raise TrialStorageError(f"list trials failed: {type(exc).__name__}: {exc}") from exc
        return sorted(manifests, key=lambda item: str(item.get("uploaded_at", "")), reverse=True)


def build_r2_storage() -> Optional[S3TrialStorage]:
    config = S3StorageConfig.from_r2_env()
    return None if config is None else S3TrialStorage(config)


def write_upload_receipt(trial_dir: str | os.PathLike[str], manifest: dict) -> None:
    path = Path(trial_dir) / UPLOAD_RECEIPT
    temporary = path.with_name(path.name + ".tmp")
    payload = {
        "trial_id": manifest.get("trial_id"),
        "uploaded_at": manifest.get("uploaded_at"),
        "object_count": manifest.get("object_count"),
        "total_bytes": manifest.get("total_bytes"),
    }
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)

