import asyncio
import io
import sys
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import httpx
from fastapi import FastAPI, Header, Request

from stability_runtime import LatestFrameStore
from yolo_client import (
    DetectionCache,
    YoloClientSettings,
    YoloShadowClient,
    validate_service_result,
)
from yolo_service.core import normalize_detection


def _service_result(frame_id=1, inference_ms=42.5):
    return {
        "frame_id": frame_id,
        "inference_ms": inference_ms,
        "image_width": 320,
        "image_height": 240,
        "objects": [{
            "class_id": 0,
            "label": "person",
            "confidence": 0.93,
            "bbox_norm": [0.1, 0.05, 0.7, 0.95],
            "center_norm": [0.4, 0.5],
        }],
    }


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http_{self.status_code}")

    def json(self):
        return self.payload


class _FakeClient:
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


class _SlowLatestOnlyClient:
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
        return _FakeResponse(_service_result(frame_id))

    async def aclose(self):
        return None


class YoloClientContractTests(unittest.TestCase):
    @staticmethod
    def _guidance_object(label, confidence, bbox):
        return {
            "label": label,
            "confidence": confidence,
            "bbox_norm": bbox,
        }

    def test_service_response_is_clamped_and_validated(self):
        payload = _service_result()
        payload["objects"][0]["bbox_norm"] = [-1, 0.2, 2, 0.9]
        payload["objects"][0]["confidence"] = 1.2
        result = validate_service_result(payload, expected_frame_id=1)
        self.assertEqual(result["objects"][0]["bbox_norm"], [0.0, 0.2, 1.0, 0.9])
        self.assertEqual(result["objects"][0]["confidence"], 1.0)

    def test_malformed_or_mismatched_response_is_rejected(self):
        for payload in ({}, [], _service_result(frame_id=2)):
            with self.assertRaises((KeyError, TypeError, ValueError)):
                validate_service_result(payload, expected_frame_id=1)

    def test_service_normalizes_pixel_boxes(self):
        detection = normalize_detection({
            "class_id": 2,
            "label": "car",
            "confidence": 0.8,
            "bbox_xyxy": [-10, 24, 400, 260],
        }, 320, 240)
        self.assertEqual(detection["bbox_norm"], [0.0, 0.1, 1.0, 1.0])
        self.assertEqual(detection["center_norm"], [0.5, 0.55])

    def test_cache_replacement_and_staleness(self):
        now = [2_000_000_000]
        cache = DetectionCache(clock_ns=lambda: now[0])
        cache.replace({"source_frame_id": 1, "inference_completed_monotonic_ns": now[0]})
        cache.replace({"source_frame_id": 2, "inference_completed_monotonic_ns": now[0]})
        self.assertEqual(cache.read(1)["result"]["source_frame_id"], 2)
        now[0] += 1_000_000_001
        stale = cache.read(1)
        self.assertTrue(stale["stale"])
        self.assertFalse(stale["available"])
        self.assertIsNone(stale["result"])

    def test_latest_perception_returns_safe_empty_and_stale_states(self):
        now = [2_000_000_000]
        client = YoloShadowClient(YoloClientSettings(enabled=True), LatestFrameStore())
        client.cache = DetectionCache(clock_ns=lambda: now[0])

        empty = client.latest_perception(display_rotation_deg=0)
        self.assertFalse(empty["available"])
        self.assertFalse(empty["stale"])
        self.assertIsNone(empty["frame_id"])
        self.assertEqual(empty["objects"], [])

        cached = {
            "source_frame_id": 7,
            "backend_received_monotonic_ns": now[0],
            "inference_completed_monotonic_ns": now[0],
            "request_ms": 42.0,
            "frame_id": 7,
            "inference_ms": 12.5,
            "image_width": 240,
            "image_height": 320,
            "objects": _service_result()["objects"],
        }
        client.cache.replace(cached, {"inference_rotation_deg": 90})
        fresh = client.latest_perception(display_rotation_deg=0)
        self.assertTrue(fresh["available"])
        self.assertEqual(fresh["frame_id"], 7)
        self.assertEqual(fresh["inference_rotation_deg"], 90)
        self.assertEqual(fresh["display_rotation_deg"], 0)

        now[0] += 3_000_000_001
        stale = client.latest_perception(display_rotation_deg=0)
        self.assertFalse(stale["available"])
        self.assertTrue(stale["stale"])
        self.assertIsNone(stale["frame_id"])
        self.assertEqual(stale["objects"], [])

    def test_latest_selection_counts_replaced_frames(self):
        client = YoloShadowClient(YoloClientSettings(), LatestFrameStore())
        client._record_frame_selection(10)
        client._record_frame_selection(14)
        health = client.health()
        self.assertEqual(health["frames_skipped"], 3)
        self.assertEqual(health["frames_replaced"], 3)

    def test_requested_target_beats_higher_confidence_unrelated_object(self):
        client = YoloShadowClient(
            YoloClientSettings(
                guidance_persistence_frames=2,
                guidance_max_missed_frames=1,
            ),
            LatestFrameStore(),
        )
        objects = [
            self._guidance_object("keyboard", 0.82, [0.05, 0.1, 0.4, 0.4]),
            self._guidance_object("cup", 0.71, [0.55, 0.35, 0.75, 0.65]),
            self._guidance_object("book", 0.65, [0.1, 0.6, 0.5, 0.9]),
        ]
        client._update_target_stability(objects, 1)
        client._update_target_stability(objects, 2)
        requested = client._guidance_target("cup")
        generic = client._guidance_target(None)
        self.assertEqual(requested["target_state"], "stable")
        self.assertEqual(requested["target"], "cup")
        self.assertEqual(requested["requested_target"], "cup")
        self.assertEqual(generic["target"], "keyboard")

    def test_requested_target_absent_or_below_threshold_never_substitutes(self):
        client = YoloShadowClient(
            YoloClientSettings(guidance_persistence_frames=2),
            LatestFrameStore(),
        )
        keyboard = self._guidance_object(
            "keyboard", 0.90, [0.05, 0.1, 0.4, 0.4]
        )
        client._update_target_stability([keyboard], 1)
        client._update_target_stability([keyboard], 2)
        absent = client._guidance_target("bottle")
        self.assertEqual(absent["target_state"], "not_found")
        self.assertEqual(absent["target"], "bottle")

        low_cup = self._guidance_object("cup", 0.49, [0.5, 0.3, 0.7, 0.7])
        client._update_target_stability([keyboard, low_cup], 3)
        client._update_target_stability([keyboard, low_cup], 4)
        below = client._guidance_target("cup")
        self.assertEqual(below["target_state"], "uncertain")
        self.assertEqual(below["target"], "cup")
        self.assertNotEqual(below.get("target"), "keyboard")

    def test_requested_target_is_per_call_and_resets_for_unrelated_turn(self):
        client = YoloShadowClient(
            YoloClientSettings(guidance_persistence_frames=1),
            LatestFrameStore(),
        )
        objects = [
            self._guidance_object("keyboard", 0.82, [0.05, 0.1, 0.4, 0.4]),
            self._guidance_object("cup", 0.71, [0.55, 0.35, 0.75, 0.65]),
        ]
        client._update_target_stability(objects, 1)
        requested_turn = client._guidance_target("cup")
        unrelated_turn = client._guidance_target(None)
        self.assertEqual(requested_turn["requested_target"], "cup")
        self.assertIsNone(unrelated_turn["requested_target"])
        self.assertEqual(unrelated_turn["target_binding"], "generic")
        self.assertEqual(unrelated_turn["target"], "keyboard")

    def test_requested_target_miss_grace_ignores_competing_object(self):
        client = YoloShadowClient(
            YoloClientSettings(
                guidance_persistence_frames=2,
                guidance_max_missed_frames=1,
            ),
            LatestFrameStore(),
        )
        cup = self._guidance_object("cup", 0.80, [0.5, 0.3, 0.7, 0.7])
        keyboard = self._guidance_object(
            "keyboard", 0.99, [0.05, 0.1, 0.4, 0.4]
        )
        client._update_target_stability([cup], 1)
        client._update_target_stability([cup], 2)
        client._update_target_stability([keyboard], 3)
        held = client._guidance_target("cup")
        client._update_target_stability([keyboard], 4)
        expired = client._guidance_target("cup")
        self.assertEqual(held["target_state"], "stable")
        self.assertTrue(held["held_through_miss"])
        self.assertEqual(held["target_bbox"], [0.5, 0.3, 0.7, 0.7])
        self.assertEqual(expired["target_state"], "uncertain")
        self.assertEqual(expired["target"], "cup")

    def test_detection_log_is_change_triggered_and_rate_limited(self):
        client = YoloShadowClient(YoloClientSettings(), LatestFrameStore())
        result = _service_result(frame_id=10, inference_ms=132.44)
        output = io.StringIO()
        with redirect_stdout(output):
            client._maybe_log_detection_state(result, now=100.0)
            result["frame_id"] = 11
            result["objects"][0]["confidence"] = 0.90
            client._maybe_log_detection_state(result, now=104.9)
            result["frame_id"] = 12
            client._maybe_log_detection_state(result, now=105.0)
            result["frame_id"] = 13
            result["objects"].append({**result["objects"][0], "confidence": 0.80})
            client._maybe_log_detection_state(result, now=105.1)

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(
            lines[0],
            "[YOLO-DETECTION] frame_id=10 objects=1 labels=person:0.93 inference_ms=132.4",
        )
        self.assertIn("frame_id=12 objects=1 labels=person:0.90", lines[1])
        self.assertIn(
            "frame_id=13 objects=2 labels=person:0.90,person:0.80",
            lines[2],
        )

    def test_detection_log_sorts_objects_and_formats_zero_detections(self):
        client = YoloShadowClient(YoloClientSettings(), LatestFrameStore())
        result = _service_result(frame_id=20, inference_ms=128.12)
        result["objects"] = [
            {**result["objects"][0], "class_id": 2, "label": "laptop", "confidence": 0.74},
            {**result["objects"][0], "class_id": 0, "label": "person", "confidence": 0.81},
            {**result["objects"][0], "class_id": 0, "label": "person", "confidence": 0.91},
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            client._maybe_log_detection_state(result, now=200.0)
            result["frame_id"] = 21
            result["objects"] = []
            client._maybe_log_detection_state(result, now=200.1)

        self.assertEqual(output.getvalue().splitlines(), [
            "[YOLO-DETECTION] frame_id=20 objects=3 "
            "labels=laptop:0.74,person:0.91,person:0.81 inference_ms=128.1",
            "[YOLO-DETECTION] frame_id=21 objects=0 labels=none inference_ms=128.1",
        ])

    def test_frozen_camera_and_gemini_paths_do_not_reference_yolo(self):
        source = Path("app_main.py").read_text(encoding="utf-8")
        camera_handler = source.split("async def _handle_camera_frame", 1)[1].split(
            "def _retain_latest_thermal", 1
        )[0]
        gemini_client = Path("gemini_live_client.py").read_text(encoding="utf-8")
        self.assertNotIn("yolo_client", camera_handler)
        self.assertNotIn("yolo", gemini_client.lower())
        service_dockerfile = Path("yolo_service/Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY yolov8n.pt", service_dockerfile)
        self.assertNotIn("yoloe-11l-seg-pf.pt", service_dockerfile)

    def test_perception_endpoint_only_reads_latest_shadow_state(self):
        source = Path("app_main.py").read_text(encoding="utf-8")
        endpoint = source.split('@app.get("/api/perception/latest")', 1)[1].split(
            "class RecordingCommand", 1
        )[0]
        self.assertIn("yolo_client.latest_perception", endpoint)
        self.assertNotIn("_request_detection", endpoint)
        self.assertNotIn("service_token", endpoint.lower())


class YoloClientFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_starts_no_network_worker_or_http_import(self):
        with mock.patch.dict("os.environ", {
            "ENABLE_YOLO": "false",
            "YOLO_REQUEST_TIMEOUT_SEC": "invalid",
        }):
            settings = YoloClientSettings.from_env()
        client = YoloShadowClient(settings, LatestFrameStore())
        prior_httpx = sys.modules.get("httpx")
        self.assertFalse(await client.start())
        self.assertIsNone(client._task)
        self.assertIsNone(client._http_client)
        self.assertIs(sys.modules.get("httpx"), prior_httpx)

    async def test_enabled_invalid_numeric_configuration_uses_safe_defaults(self):
        with mock.patch.dict("os.environ", {
            "ENABLE_YOLO": "true",
            "YOLO_SERVICE_URL": "https://example.test",
            "YOLO_SERVICE_TOKEN": "secret",
            "YOLO_MIN_INTERVAL_SEC": "invalid",
            "YOLO_REQUEST_TIMEOUT_SEC": "nan",
            "YOLO_CONFIDENCE": "infinity",
        }, clear=False):
            settings = YoloClientSettings.from_env()
        self.assertEqual(settings.min_interval_sec, 1.0)
        self.assertEqual(settings.request_timeout_sec, 2.0)
        self.assertEqual(settings.confidence, 0.25)

    async def test_missing_url_or_token_degrades_without_task(self):
        for settings, reason in (
            (YoloClientSettings(enabled=True, service_token="token"), "service_url_missing"),
            (YoloClientSettings(enabled=True, service_url="https://example.test"), "service_token_missing"),
            (YoloClientSettings(enabled=True, service_url="http://example.test", service_token="token"), "service_url_must_be_https"),
        ):
            client = YoloShadowClient(settings, LatestFrameStore())
            self.assertFalse(await client.start())
            self.assertEqual(client.health()["unavailable_reason"], reason)
            self.assertIsNone(client._task)

    async def test_success_updates_cache_and_sends_secret_only_in_header(self):
        frames = LatestFrameStore()
        frame = frames.update(b"jpeg", time.monotonic())
        settings = YoloClientSettings(
            enabled=True,
            service_url="https://yolo.example.test",
            service_token="top-secret",
        )
        client = YoloShadowClient(settings, frames, rotation_provider=lambda: 90)
        fake = _FakeClient(_FakeResponse(_service_result(frame.sequence)))
        client._http_client = fake
        client._httpx = mock.Mock(TimeoutException=TimeoutError)
        output = io.StringIO()
        with redirect_stdout(output):
            await client._request_detection(frame)
        self.assertTrue(client.latest_detections()["available"])
        self.assertEqual(client.health()["requests_completed"], 1)
        args, kwargs = fake.calls[0]
        self.assertNotIn("top-secret", args[0])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer top-secret")
        self.assertEqual(kwargs["headers"]["X-Camera-Rotation-Deg"], "90")
        self.assertNotIn("top-secret", output.getvalue())
        perception = client.latest_perception(display_rotation_deg=0)
        self.assertEqual(perception["inference_rotation_deg"], 90)
        self.assertNotIn("top-secret", str(perception))

    async def test_slow_request_dispatches_newest_frame_next_without_backlog(self):
        frames = LatestFrameStore()
        frames.update(b"frame-1", time.monotonic())
        client = YoloShadowClient(
            YoloClientSettings(
                enabled=True,
                service_url="https://yolo.example.test",
                service_token="secret",
                min_interval_sec=0.1,
            ),
            frames,
        )
        slow = _SlowLatestOnlyClient()
        client._http_client = slow
        client._httpx = mock.Mock(TimeoutException=TimeoutError)
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
        self.assertEqual(client.health()["frames_replaced"], 2)

    async def test_fake_yolo_service_integration(self):
        fake_service = FastAPI()

        @fake_service.post("/v1/detect")
        async def detect(
            request: Request,
            authorization: str = Header(...),
            x_frame_id: int = Header(...),
        ):
            self.assertEqual(authorization, "Bearer integration-secret")
            self.assertEqual(await request.body(), b"integration-jpeg")
            return _service_result(x_frame_id, inference_ms=7.5)

        frames = LatestFrameStore()
        frame = frames.update(b"integration-jpeg", time.monotonic())
        client = YoloShadowClient(
            YoloClientSettings(
                enabled=True,
                service_url="https://fake-yolo.test",
                service_token="integration-secret",
            ),
            frames,
        )
        transport = httpx.ASGITransport(app=fake_service)
        client._http_client = httpx.AsyncClient(transport=transport)
        client._httpx = httpx
        try:
            await client._request_detection(frame)
        finally:
            await client._http_client.aclose()
        cached = client.latest_detections()
        self.assertTrue(cached["available"])
        self.assertEqual(cached["result"]["frame_id"], frame.sequence)
        self.assertEqual(cached["result"]["inference_ms"], 7.5)

    async def test_network_and_malformed_response_failures_are_contained(self):
        frames = LatestFrameStore()
        frame = frames.update(b"jpeg", time.monotonic())
        client = YoloShadowClient(
            YoloClientSettings(enabled=True, service_url="https://x", service_token="secret"),
            frames,
        )
        client._httpx = mock.Mock(TimeoutException=TimeoutError)
        for outcome in (OSError("dns"), _FakeResponse({"bad": True})):
            client._http_client = _FakeClient(outcome)
            output = io.StringIO()
            with redirect_stdout(output):
                await client._request_detection(frame)
        self.assertEqual(client.health()["request_failures"], 2)
        self.assertFalse(client.latest_detections()["available"])
        self.assertFalse(asyncio.current_task().cancelled())

    async def test_timeout_is_counted_and_does_not_escape(self):
        frames = LatestFrameStore()
        frame = frames.update(b"jpeg", time.monotonic())
        client = YoloShadowClient(
            YoloClientSettings(enabled=True, service_url="https://x", service_token="secret"),
            frames,
        )
        client._httpx = mock.Mock(TimeoutException=TimeoutError)
        client._http_client = _FakeClient(TimeoutError())
        await client._request_detection(frame)
        self.assertEqual(client.health()["request_timeouts"], 1)
        self.assertFalse(asyncio.current_task().cancelled())


if __name__ == "__main__":
    unittest.main()
