"""POST /internal/trial/{id}/phone_capture, R2 push, ZIP, and the N / M / K numbers."""
import io
import json
import os
import shutil
import tempfile
import threading
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from starlette.testclient import TestClient

import app_main
import phone_capture
from phone_capture import jpeg_comment_seq, missing_ranges
from stability_runtime import MSG_TYPE_CAM
from trial_archive import TrialArchive
from trial_storage import S3StorageConfig, S3TrialStorage

TRIAL = "T20_PSIM"


# ---------------------------------------------------------------- fixtures

class _NotFound(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    """In-memory S3 with real 404s and delete, enough for put_subfolder."""

    def __init__(self):
        self.objects = {}
        self.fail = False

    def put_object(self, **kwargs):
        if self.fail:
            raise OSError("R2 outage")
        self.objects[kwargs["Key"]] = {
            "body": bytes(kwargs["Body"]),
            "content_type": kwargs["ContentType"],
            "metadata": dict(kwargs.get("Metadata") or {}),
        }

    def _get(self, key):
        if key not in self.objects:
            raise _NotFound(key)
        return self.objects[key]

    def head_object(self, **kwargs):
        value = self._get(kwargs["Key"])
        return {"ContentLength": len(value["body"]), "Metadata": value["metadata"]}

    def get_object(self, **kwargs):
        value = self._get(kwargs["Key"])
        return {"Body": io.BytesIO(value["body"]), "ContentType": value["content_type"],
                "Metadata": value["metadata"], "ETag": "e"}

    def delete_object(self, **kwargs):
        self.objects.pop(kwargs["Key"], None)

    def get_paginator(self, name):
        client = self

        class Paginator:
            def paginate(self, **kwargs):
                return [{"Contents": [{"Key": key} for key in sorted(client.objects)
                                      if key.startswith(kwargs["Prefix"])]}]
        return Paginator()


def _storage(fake):
    config = S3StorageConfig(endpoint_url="http://r2", access_key_id="a", secret_access_key="b", bucket="t")
    return S3TrialStorage(config, client=fake, attempts=1, sleep=lambda _s: None)


def _jpeg(seq=None) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((8, 8, 3), 128, dtype=np.uint8))
    assert ok
    data = encoded.tobytes()
    if seq is None:
        return data
    text = f"seq={seq}".encode()
    segment = b"\xff\xfe" + (len(text) + 2).to_bytes(2, "big") + text
    return data[:2] + segment + data[2:]


def _capture_zip(trial_id=TRIAL, seqs=range(1, 11), imu=5, manifest=None, extra=None) -> bytes:
    seqs = list(seqs)
    body = {"session_id": trial_id, "app_version": "1.0", "device": "iPhone",
            "frames": len(seqs), "imu_samples": imu,
            "started_ts": "2026-10-26T10:00:00Z", "ended_ts": "2026-10-26T10:20:00Z"}
    body.update(manifest or {})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.json", json.dumps(body))
        archive.writestr("imu.jsonl", "".join(json.dumps({"i": i}) + "\n" for i in range(imu)))
        archive.writestr("phone.jsonl", json.dumps({"t": "start"}) + "\n")
        for seq in seqs:
            archive.writestr(f"frames/{seq}.jpg", _jpeg(seq))
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _make_trial(root, trial_id=TRIAL, events=(), ready=False):
    directory = Path(root) / trial_id
    directory.mkdir(parents=True)
    (directory / "meta.json").write_text(json.dumps({"trial": "T20", "pid": "PSIM"}))
    lines = [{"t": "session_start", "ts": "x"}, *events, {"t": "session_end", "ts": "y"}]
    (directory / "session.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
    if ready:   # everything the main /upload needs
        for name in ("ground_truth.json", "summary.json"):
            (directory / name).write_text("{}")
        (directory / "scores.jsonl").write_text("")
        (directory / "scores.csv").write_text("")
    return directory


# ---------------------------------------------------------------- route tests

class PhoneCaptureRouteTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.open_id = None
        self.fake = FakeS3()
        self.recorder = types.SimpleNamespace(
            enabled=True, root=self.root, status=lambda: {"session_id": self.open_id})
        self.patches = [
            mock.patch.object(app_main, "trial_recorder", self.recorder),
            mock.patch.object(app_main, "trial_object_storage", _storage(self.fake)),
            mock.patch.object(app_main, "PHONE_CAPTURE_TMP", self.tmp),
            mock.patch.object(app_main, "_trial_upload_locks", {}),
            mock.patch.dict(os.environ, {"TRIAL_TOKEN": ""}),
        ]
        for patch in self.patches:
            patch.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])
        self.client = TestClient(app_main.app)

    def post(self, body, trial_id=TRIAL, **kwargs):
        return self.client.post(f"/internal/trial/{trial_id}/phone_capture", content=body,
                                headers={"content-type": "application/zip"}, **kwargs)

    def assertNothingWritten(self, directory):
        self.assertFalse((directory / "phone").exists())
        self.assertEqual([p for p in directory.iterdir() if p.name.startswith(".phone")], [])
        self.assertEqual(os.listdir(self.tmp), [])
        self.assertEqual(self.fake.objects, {})

    def test_valid_upload_lands_archives_and_prunes(self):
        directory = _make_trial(self.root)
        response = self.post(_capture_zip())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["r2"]["status"], "uploaded")
        # R2 has every file plus the phone's own marker; the trial marker is untouched.
        keys = set(self.fake.objects)
        self.assertIn(f"trials/{TRIAL}/phone/_complete.json", keys)
        self.assertIn(f"trials/{TRIAL}/phone/frames/7.jpg", keys)
        self.assertNotIn(f"trials/{TRIAL}/_complete.json", keys)
        # Confirmed, so the next /upload will not push it again.
        self.assertEqual(phone_capture.read_push_state(str(directory))["status"], "done")
        self.assertFalse(phone_capture.push_pending(str(directory)))
        # Local copy pruned to the small files once R2 confirmed.
        self.assertEqual(sorted(os.listdir(directory / "phone")), ["manifest.json", "phone.jsonl"])
        self.assertEqual(os.listdir(self.tmp), [])
        # The trial ZIP carries the phone frames, read back from R2.
        archive = TrialArchive(self.root, _storage(self.fake))
        with zipfile.ZipFile(io.BytesIO(archive.zip_bytes(TRIAL))) as zf:
            names = set(zf.namelist())
        self.assertIn(f"{TRIAL}/phone/frames/7.jpg", names)
        self.assertIn(f"{TRIAL}/meta.json", names)
        self.assertFalse(any("/." in name for name in names))
        # R2's trial listing does not mistake the phone marker for a trial.
        self.assertEqual(_storage(self.fake).list_trials(), [])
        rows = archive.list_trials()
        self.assertTrue(rows[0]["phone_copy"])

    def test_without_r2_everything_stays_local(self):
        directory = _make_trial(self.root)
        with mock.patch.object(app_main, "trial_object_storage", None):
            response = self.post(_capture_zip())
        self.assertEqual(response.json()["r2"], {"status": "not_configured"})
        self.assertEqual(len(os.listdir(directory / "phone" / "frames")), 10)
        self.assertTrue((directory / "phone" / "imu.jsonl").is_file())

    def test_open_trial_is_409(self):
        directory = _make_trial(self.root)
        self.open_id = TRIAL
        response = self.post(_capture_zip())
        self.assertEqual(response.status_code, 409)
        self.assertNothingWritten(directory)

    def test_path_traversal_is_422_and_writes_nothing(self):
        directory = _make_trial(self.root)
        for name in ("../escape.jpg", "/abs.jpg", "frames/../../x.jpg", "frames\\1.jpg", "notes.txt"):
            with self.subTest(name=name):
                response = self.post(_capture_zip(extra={name: b"x"}))
                self.assertEqual(response.status_code, 422, response.text)
                self.assertNothingWritten(directory)
        self.assertFalse((Path(self.root).parent / "escape.jpg").exists())

    def test_manifest_mismatches_are_422(self):
        directory = _make_trial(self.root)
        cases = {
            "frames": _capture_zip(manifest={"frames": 11}),
            "imu": _capture_zip(manifest={"imu_samples": 4}),
            "session": _capture_zip(manifest={"session_id": "T99_PSIM"}),
            "not a zip": b"PK not really",
        }
        for label, body in cases.items():
            with self.subTest(label):
                response = self.post(body)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertNothingWritten(directory)

    def test_reupload_replaces_cleanly_locally_and_in_r2(self):
        directory = _make_trial(self.root)
        with mock.patch.object(app_main, "trial_object_storage", None):
            self.assertEqual(self.post(_capture_zip(seqs=range(1, 11))).status_code, 200)
            self.assertEqual(self.post(_capture_zip(seqs=range(1, 4))).status_code, 200)
        self.assertEqual(sorted(os.listdir(directory / "phone" / "frames")), ["1.jpg", "2.jpg", "3.jpg"])
        self.assertEqual([p for p in directory.iterdir() if p.name.startswith(".phone-")], [])
        # Same through R2: frames the new copy no longer has are deleted there too.
        self.assertEqual(self.post(_capture_zip(seqs=range(1, 11))).status_code, 200)
        response = self.post(_capture_zip(seqs=range(1, 4)))
        self.assertEqual(response.json()["r2"]["stale_deleted"], 7)
        frames = sorted(k for k in self.fake.objects if "/phone/frames/" in k)
        self.assertEqual(len(frames), 3)

    def test_too_large_is_413_with_and_without_length(self):
        directory = _make_trial(self.root)
        body = _capture_zip()
        with mock.patch.object(phone_capture, "MAX_BYTES", len(body) - 1):
            declared = self.post(body)
            chunked = self.post(iter([body[:500], body[500:]]))
        self.assertEqual(declared.status_code, 413)
        self.assertEqual(chunked.status_code, 413)
        self.assertNothingWritten(directory)

    def test_short_disk_is_507(self):
        directory = _make_trial(self.root)
        with mock.patch.object(app_main, "_free_bytes", lambda path: 1024):
            response = self.post(_capture_zip())
        self.assertEqual(response.status_code, 507)
        self.assertNothingWritten(directory)

    def test_failed_r2_push_is_retried_by_upload(self):
        directory = _make_trial(self.root, ready=True)
        self.fake.fail = True
        response = self.post(_capture_zip())
        self.assertEqual(response.json()["r2"]["status"], "failed")
        self.assertEqual(phone_capture.read_push_state(str(directory))["status"], "failed")
        self.assertEqual(len(os.listdir(directory / "phone" / "frames")), 10)   # kept

        self.fake.fail = False
        upload = self.client.post(f"/internal/trial/{TRIAL}/upload")
        self.assertEqual(upload.status_code, 200, upload.text)
        self.assertEqual(upload.json()["phone"]["status"], "uploaded")
        self.assertIn(f"trials/{TRIAL}/phone/_complete.json", self.fake.objects)
        self.assertEqual(phone_capture.read_push_state(str(directory))["status"], "done")
        # The trial's own upload carries no phone/ objects of its own.
        main = json.loads(self.fake.objects[f"trials/{TRIAL}/_complete.json"]["body"])
        self.assertFalse(any(item["key"].startswith("phone/") for item in main["objects"]))
        self.assertFalse(any(item["key"].startswith(".") for item in main["objects"]))


# ---------------------------------------------------------------- linking

class LinkingTests(unittest.TestCase):
    def test_reads_seq_from_com_segment(self):
        self.assertEqual(jpeg_comment_seq(_jpeg(412)), 412)
        self.assertIsNone(jpeg_comment_seq(_jpeg()))
        self.assertIsNone(jpeg_comment_seq(b"\xff\xd8\xff\xfe\x00"))   # truncated
        self.assertIsNone(jpeg_comment_seq(b"not a jpeg"))

    def test_seq_survives_only_until_reencode(self):
        # Why the socket reads it first: the phone re-encode drops COM.
        from perception_orientation import canonicalize_phone_rgb_jpeg
        self.assertIsNone(jpeg_comment_seq(canonicalize_phone_rgb_jpeg(_jpeg(5))))

    def test_missing_ranges(self):
        self.assertEqual(missing_ranges(range(1, 11), [1, 2, 3, 7, 10]), [[4, 6], [8, 9]])
        self.assertEqual(missing_ranges(range(1, 4), [1, 2, 3]), [])


class ViewerNumbersTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def _install(self, events, seqs=range(1, 11)):
        directory = _make_trial(self.root, events=events)
        zip_path = os.path.join(self.root, "c.zip")
        Path(zip_path).write_bytes(_capture_zip(seqs=seqs))
        phone_capture.install(zip_path, str(directory), TRIAL)
        return TrialArchive(self.root, None).get_trial(TRIAL)["phone_copy"]

    def test_gap_and_missing_ranges(self):
        received = [1, 2, 3, *range(7, 11)]
        events = [{"t": "frame_recv", "source": "phone", "phone_seq": s, "n": i}
                  for i, s in enumerate(received, 1)]
        events.append({"t": "frame_recv", "source": "glasses", "phone_seq": None, "n": 99})
        phone = self._install(events)
        self.assertEqual((phone["frames"], phone["reached"], phone["gap"]), (10, 7, 3))
        self.assertEqual(phone["missing_ranges"], [[4, 6]])
        self.assertEqual(phone["missing_total"], 3)

    def test_unlinked_frames_do_not_list_everything_missing(self):
        events = [{"t": "frame_recv", "source": "phone", "phone_seq": None, "n": i} for i in range(4)]
        phone = self._install(events)
        self.assertEqual((phone["reached"], phone["gap"]), (4, 6))
        self.assertFalse(phone["linked"])
        self.assertIsNone(phone["missing_ranges"])

    def test_no_phone_copy_is_none(self):
        _make_trial(self.root)
        self.assertIsNone(TrialArchive(self.root, None).get_trial(TRIAL)["phone_copy"])


class SocketFrameRecvTests(unittest.TestCase):
    def test_phone_seq_is_logged_before_reencode_and_null_for_glasses(self):
        recorder = mock.Mock()
        got = threading.Event()

        async def capture(data, frame_counter, *args, **kwargs):
            got.set()
            return frame_counter + 1

        frame = _jpeg(412)
        with mock.patch.object(app_main, "_handle_camera_frame", capture), \
                mock.patch.object(app_main, "trial_recorder", recorder), \
                mock.patch.object(app_main, "esp32_camera_ws", None), \
                mock.patch.object(app_main, "camera_source", None):
            client = TestClient(app_main.app)
            for path in ("/ws/camera_thermal?source=phone", "/ws/camera_thermal"):
                got.clear()
                with client.websocket_connect(path) as ws:
                    ws.send_bytes(bytes([MSG_TYPE_CAM]) + frame)
                    self.assertTrue(got.wait(5))
        calls = [c.args for c in recorder.frame_recv.call_args_list]
        self.assertEqual(calls, [("phone", 412, 1, len(frame)), ("glasses", None, 1, len(frame))])


if __name__ == "__main__":
    unittest.main()
