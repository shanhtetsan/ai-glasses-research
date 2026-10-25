import io
import json
import tempfile
import unittest
from pathlib import Path

from trial_storage import S3StorageConfig, S3TrialStorage, TrialStorageError


class FakePaginator:
    def __init__(self, client):
        self.client = client

    def paginate(self, **kwargs):
        prefix = kwargs["Prefix"]
        return [{"Contents": [{"Key": key} for key in sorted(self.client.objects) if key.startswith(prefix)]}]


class FakeS3:
    def __init__(self, fail_puts=0):
        self.objects = {}
        self.fail_puts = fail_puts
        self.put_calls = 0

    def put_object(self, **kwargs):
        self.put_calls += 1
        if self.fail_puts:
            self.fail_puts -= 1
            raise OSError("temporary outage")
        body = kwargs["Body"]
        if hasattr(body, "read"):
            body = body.read()
        self.objects[kwargs["Key"]] = {
            "body": bytes(body),
            "content_type": kwargs["ContentType"],
            "metadata": dict(kwargs.get("Metadata") or {}),
        }
        return {"ETag": "etag"}

    def head_object(self, **kwargs):
        value = self.objects[kwargs["Key"]]
        return {"ContentLength": len(value["body"]), "Metadata": value["metadata"]}

    def get_object(self, **kwargs):
        value = self.objects[kwargs["Key"]]
        return {
            "Body": io.BytesIO(value["body"]),
            "ContentType": value["content_type"],
            "Metadata": value["metadata"],
            "ETag": "etag",
        }

    def get_paginator(self, name):
        if name != "list_objects_v2":
            raise AssertionError(name)
        return FakePaginator(self)


class TrialStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name) / "T02_PSIM"
        (self.folder / "frames").mkdir(parents=True)
        (self.folder / "meta.json").write_text('{"trial":"T02"}\n', encoding="utf-8")
        (self.folder / "frames" / "7.jpg").write_bytes(b"jpeg")
        self.client = FakeS3()
        self.storage = S3TrialStorage(
            S3StorageConfig("https://example.invalid", "key", "secret", "bucket"),
            client=self.client,
            base_backoff_sec=0,
            sleep=lambda _: None,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_put_folder_verifies_and_publishes_completion_marker_last(self):
        manifest = self.storage.put_folder(self.folder)
        self.assertEqual(manifest["trial_id"], "T02_PSIM")
        self.assertEqual(manifest["object_count"], 2)
        self.assertIn("trials/T02_PSIM/meta.json", self.client.objects)
        self.assertIn("trials/T02_PSIM/frames/7.jpg", self.client.objects)
        complete = json.loads(self.client.objects["trials/T02_PSIM/_complete.json"]["body"])
        self.assertEqual(complete["objects"], manifest["objects"])

    def test_put_retries_with_backoff(self):
        client = FakeS3(fail_puts=2)
        sleeps = []
        storage = S3TrialStorage(
            self.storage.config, client=client, attempts=3, base_backoff_sec=0.25, sleep=sleeps.append
        )
        storage.put_folder(self.folder)
        self.assertEqual(sleeps, [0.25, 0.5])

    def test_get_object_and_list_trials(self):
        self.storage.put_folder(self.folder)
        item = self.storage.get_object("trials/T02_PSIM/meta.json")
        self.assertEqual(item.body, b'{"trial":"T02"}\n')
        self.assertEqual([trial["trial_id"] for trial in self.storage.list_trials()], ["T02_PSIM"])

    def test_get_rejects_keys_outside_trial_prefix(self):
        with self.assertRaises(TrialStorageError):
            self.storage.get_object("../secret")

    def test_verification_mismatch_fails_without_completion_marker(self):
        original = self.client.head_object

        def wrong_head(**kwargs):
            value = original(**kwargs)
            value["ContentLength"] += 1
            return value

        self.client.head_object = wrong_head
        with self.assertRaises(TrialStorageError):
            self.storage.put_folder(self.folder)
        self.assertNotIn("trials/T02_PSIM/_complete.json", self.client.objects)


if __name__ == "__main__":
    unittest.main()
