import asyncio
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from hand_client import (
    HandClientSettings,
    HandResultCache,
    HandTrackingClient,
    validate_hand_result,
)
from stability_runtime import LatestFrameStore
from yolo_client import YoloClientSettings, YoloShadowClient


def _hand(frame_id=1, handedness="Left"):
    landmarks = [
        {"id": index, "x": index / 20, "y": 1 - index / 20, "z": -index / 100}
        for index in range(21)
    ]
    return {
        "hand_index": 0,
        "handedness": handedness,
        "handedness_score": 0.98,
        "landmarks": landmarks,
        "index_tip_norm": [0.4, 0.6],
        "wrist_norm": [0.0, 1.0],
        "hand_center_norm": [0.5, 0.5],
        "bbox_norm": [0.0, 0.0, 1.0, 1.0],
    }


def _result(frame_id=1, hands=None, inference_ms=40.0):
    return {
        "frame_id": frame_id,
        "image_width": 640,
        "image_height": 480,
        "inference_ms": inference_ms,
        "hands": [_hand(frame_id)] if hands is None else hands,
    }


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http_{self.status_code}")

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class _Client:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    async def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    async def aclose(self):
        return None


class _SlowClient:
    def __init__(self):
        self.calls = []
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.second_started = asyncio.Event()

    async def post(self, *args, **kwargs):
        frame_id = int(kwargs["headers"]["X-Frame-ID"])
        self.calls.append((frame_id, kwargs["content"]))
        if len(self.calls) == 1:
            self.first_started.set()
            await self.release_first.wait()
        elif len(self.calls) == 2:
            self.second_started.set()
        return _Response(_result(frame_id))

    async def aclose(self):
        return None


class HandClientContractTests(unittest.TestCase):
    def test_schema_has_exactly_21_normalized_landmarks_and_max_two_hands(self):
        result = validate_hand_result(_result(), 1)
        self.assertEqual(len(result["hands"][0]["landmarks"]), 21)
        self.assertEqual(result["hands"][0]["index_tip_norm"], [0.4, 0.6])
        with self.assertRaises(ValueError):
            validate_hand_result(_result(hands=[_hand(), _hand(), _hand()]), 1)
        malformed = _result()
        malformed["hands"][0]["landmarks"].pop()
        with self.assertRaises(ValueError):
            validate_hand_result(malformed, 1)

    def test_cache_replaces_latest_and_hides_stale_landmarks(self):
        now = [1_000_000_000]
        cache = HandResultCache(clock_ns=lambda: now[0])
        cache.replace({"source_frame_id": 1, "inference_completed_monotonic_ns": now[0]})
        cache.replace({"source_frame_id": 3, "inference_completed_monotonic_ns": now[0]})
        self.assertEqual(cache.read(1.25)["result"]["source_frame_id"], 3)
        now[0] += 1_250_000_001
        stale = cache.read(1.25)
        self.assertTrue(stale["stale"])
        self.assertFalse(stale["available"])
        self.assertIsNone(stale["result"])

    def test_api_states_distinguish_disabled_unavailable_zero_and_stale(self):
        disabled = HandTrackingClient(HandClientSettings(enabled=False), LatestFrameStore())
        self.assertFalse(disabled.latest_hands()["enabled"])

        now = [2_000_000_000]
        client = HandTrackingClient(HandClientSettings(enabled=True), LatestFrameStore())
        client.cache = HandResultCache(clock_ns=lambda: now[0])
        self.assertFalse(client.latest_hands()["available"])
        client.cache.replace({
            "source_frame_id": 7,
            "source_received_at": 100.0,
            "inference_completed_monotonic_ns": now[0],
            **_result(7, hands=[]),
        })
        zero = client.latest_hands()
        self.assertTrue(zero["available"])
        self.assertEqual(zero["hands"], [])
        now[0] += 1_250_000_001
        stale = client.latest_hands()
        self.assertTrue(stale["stale"])
        self.assertEqual(stale["hands"], [])

    def test_yolo_cache_is_independent(self):
        frames = LatestFrameStore()
        hand = HandTrackingClient(HandClientSettings(enabled=True), frames)
        yolo = YoloShadowClient(YoloClientSettings(enabled=True), frames)
        hand.cache.replace({
            "source_frame_id": 1,
            "source_received_at": 1.0,
            "inference_completed_monotonic_ns": time.monotonic_ns(),
            **_result(1),
        })
        self.assertTrue(hand.latest_hands()["available"])
        self.assertFalse(yolo.latest_detections()["available"])

    def test_app_wiring_is_read_only_and_firmware_socket_count_is_unchanged(self):
        app_source = Path("app_main.py").read_text(encoding="utf-8")
        endpoint = app_source.split('@app.get("/api/perception/hands/latest")', 1)[1]
        endpoint = endpoint.split("class RecordingCommand", 1)[0]
        self.assertIn("hand_client.latest_hands()", endpoint)
        self.assertNotIn("_request_hands", endpoint)
        camera_handler = app_source.split("async def _handle_camera_frame", 1)[1]
        camera_handler = camera_handler.split("def _retain_latest_thermal", 1)[0]
        self.assertNotIn("hand_client", camera_handler)
        gemini_source = Path("gemini_live_client.py").read_text(encoding="utf-8")
        self.assertNotIn("hand_client", gemini_source)
        self.assertNotIn("/v1/hands", gemini_source)
        firmware = Path("compile/compile.ino").read_text(encoding="utf-8")
        self.assertEqual(firmware.count("WebsocketsClient "), 2)


class HandClientFailureTests(unittest.IsolatedAsyncioTestCase):
    def _client_and_frame(self):
        frames = LatestFrameStore()
        frame = frames.update(b"jpeg", time.monotonic())
        client = HandTrackingClient(
            HandClientSettings(
                enabled=True,
                service_url="https://hands.example.test",
                service_token="secret",
            ),
            frames,
        )
        client._httpx = SimpleNamespace(TimeoutException=TimeoutError)
        return client, frame

    async def test_success_uses_canonical_jpeg_without_rotation_header(self):
        client, frame = self._client_and_frame()
        fake = _Client(_Response(_result(frame.sequence)))
        client._http_client = fake
        await client._request_hands(frame)
        latest = client.latest_hands()
        self.assertTrue(latest["available"])
        self.assertEqual(latest["frame_sequence"], frame.sequence)
        self.assertIsNone(latest["captured_at"])
        self.assertIsNotNone(latest["received_at"])
        _args, kwargs = fake.calls[0]
        self.assertNotIn("X-Camera-Rotation-Deg", kwargs["headers"])
        self.assertNotIn("secret", str(latest))

    async def test_timeout_429_and_malformed_json_are_nonfatal(self):
        client, frame = self._client_and_frame()
        client._http_client = _Client(TimeoutError())
        await client._request_hands(frame)
        self.assertEqual(client.health()["timeouts"], 1)
        self.assertFalse(asyncio.current_task().cancelled())

        client._http_client = _Client(_Response({}, status_code=429))
        await client._request_hands(frame)
        self.assertEqual(client.health()["busy_responses"], 1)

        client._http_client = _Client(_Response(ValueError("bad json")))
        await client._request_hands(frame)
        self.assertEqual(client.health()["failures"], 2)
        self.assertFalse(client.latest_hands()["available"])

    async def test_slow_request_dispatches_only_newest_frame_next(self):
        frames = LatestFrameStore()
        frames.update(b"frame-1", time.monotonic())
        client = HandTrackingClient(
            HandClientSettings(
                enabled=True,
                service_url="https://hands.example.test",
                service_token="secret",
                min_interval_sec=0.01,
            ),
            frames,
        )
        slow = _SlowClient()
        client._http_client = slow
        client._httpx = SimpleNamespace(TimeoutException=TimeoutError)
        client._running = True
        client._task = asyncio.create_task(client._run())
        await asyncio.wait_for(slow.first_started.wait(), timeout=1)
        frames.update(b"frame-2", time.monotonic())
        frames.update(b"frame-3", time.monotonic())
        frames.update(b"frame-4", time.monotonic())
        slow.release_first.set()
        await asyncio.wait_for(slow.second_started.wait(), timeout=1)
        await client.stop()
        self.assertEqual(slow.calls[:2], [(1, b"frame-1"), (4, b"frame-4")])
        self.assertEqual(client.health()["frames_skipped"], 2)


if __name__ == "__main__":
    unittest.main()
