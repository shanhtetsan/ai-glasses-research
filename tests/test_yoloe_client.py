import asyncio
import io
import sys
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

import httpx
from fastapi import FastAPI, Header, Request

from stability_runtime import LatestFrameStore
from yolo_client import DetectionCache
from yoloe_client import (
    YoloeClientSettings,
    YoloeShadowClient,
    validate_obstacle_result,
)


def _obstacle_result(frame_id=1, inference_ms=300.0):
    return {
        "frame_id": frame_id,
        "inference_ms": inference_ms,
        "image_width": 320,
        "image_height": 240,
        "objects": [{
            "class_id": 4,
            "label": "bollard",
            "confidence": 0.61,
            "bbox_norm": [0.2, 0.2, 0.4, 0.9],
            "center_norm": [0.3, 0.55],
            "mask_coverage_norm": 0.06,
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


class YoloeClientContractTests(unittest.TestCase):
    def test_service_response_is_clamped_and_validated(self):
        payload = _obstacle_result()
        payload["objects"][0]["bbox_norm"] = [-1, 0.2, 2, 0.9]
        payload["objects"][0]["confidence"] = 1.2
        payload["objects"][0]["mask_coverage_norm"] = -0.5
        result = validate_obstacle_result(payload, expected_frame_id=1)
        self.assertEqual(result["objects"][0]["bbox_norm"], [0.0, 0.2, 1.0, 0.9])
        self.assertEqual(result["objects"][0]["confidence"], 1.0)
        self.assertEqual(result["objects"][0]["mask_coverage_norm"], 0.0)

    def test_malformed_or_mismatched_response_is_rejected(self):
        for payload in ({}, [], _obstacle_result(frame_id=2)):
            with self.assertRaises((KeyError, TypeError, ValueError)):
                validate_obstacle_result(payload, expected_frame_id=1)

    def test_latest_perception_returns_safe_empty_and_stale_states(self):
        now = [2_000_000_000]
        client = YoloeShadowClient(YoloeClientSettings(enabled=True), LatestFrameStore())
        client.cache = DetectionCache(clock_ns=lambda: now[0])

        empty = client.latest_perception(display_rotation_deg=0)
        self.assertFalse(empty["available"])
        self.assertIsNone(empty["frame_id"])
        self.assertEqual(empty["objects"], [])

        cached = {
            "source_frame_id": 3,
            "backend_received_monotonic_ns": now[0],
            "inference_completed_monotonic_ns": now[0],
            "request_ms": 42.0,
            "frame_id": 3,
            "inference_ms": 300.0,
            "image_width": 240,
            "image_height": 320,
            "objects": _obstacle_result()["objects"],
        }
        client.cache.replace(cached, {"inference_rotation_deg": 0})
        fresh = client.latest_perception(display_rotation_deg=0)
        self.assertTrue(fresh["available"])
        self.assertEqual(fresh["frame_id"], 3)
        self.assertEqual(fresh["source_frame_id"], 3)

        now[0] += 4_000_000_001  # exceeds the 4.0s default stale_after_sec
        stale = client.latest_perception(display_rotation_deg=0)
        self.assertFalse(stale["available"])
        self.assertTrue(stale["stale"])
        self.assertEqual(stale["objects"], [])

    def test_latest_selection_counts_replaced_frames(self):
        client = YoloeShadowClient(YoloeClientSettings(), LatestFrameStore())
        client._record_frame_selection(10)
        client._record_frame_selection(14)
        health = client.health()
        self.assertEqual(health["frames_skipped"], 3)
        self.assertEqual(health["frames_replaced"], 3)

    def test_detection_log_is_change_triggered_and_rate_limited(self):
        client = YoloeShadowClient(YoloeClientSettings(), LatestFrameStore())
        result = _obstacle_result(frame_id=10, inference_ms=300.0)
        output = io.StringIO()
        with redirect_stdout(output):
            client._maybe_log_detection_state(result, now=100.0)
            result["frame_id"] = 11
            result["objects"][0]["confidence"] = 0.55
            client._maybe_log_detection_state(result, now=104.9)
            result["frame_id"] = 12
            client._maybe_log_detection_state(result, now=105.0)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("[YOLOE-DETECTION] frame_id=10"))


class YoloeClientFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_starts_no_network_worker_or_http_import(self):
        with mock.patch.dict("os.environ", {"ENABLE_YOLOE_OBSTACLES": "false"}):
            settings = YoloeClientSettings.from_env()
        client = YoloeShadowClient(settings, LatestFrameStore())
        prior_httpx = sys.modules.get("httpx")
        self.assertFalse(await client.start())
        self.assertIsNone(client._task)
        self.assertIsNone(client._http_client)
        self.assertIs(sys.modules.get("httpx"), prior_httpx)

    async def test_settings_fall_back_to_shared_yolo_service_credentials(self):
        with mock.patch.dict("os.environ", {
            "ENABLE_YOLOE_OBSTACLES": "true",
            "YOLO_SERVICE_URL": "https://shared.example.test",
            "YOLO_SERVICE_TOKEN": "shared-secret",
        }, clear=False):
            settings = YoloeClientSettings.from_env()
        self.assertEqual(settings.service_url, "https://shared.example.test")
        self.assertEqual(settings.service_token, "shared-secret")

    async def test_missing_url_or_token_degrades_without_task(self):
        for settings, reason in (
            (YoloeClientSettings(enabled=True, service_token="token"), "service_url_missing"),
            (YoloeClientSettings(enabled=True, service_url="https://example.test"), "service_token_missing"),
            (YoloeClientSettings(enabled=True, service_url="http://example.test", service_token="token"), "service_url_must_be_https"),
        ):
            client = YoloeShadowClient(settings, LatestFrameStore())
            self.assertFalse(await client.start())
            self.assertEqual(client.health()["unavailable_reason"], reason)
            self.assertIsNone(client._task)

    async def test_success_updates_cache_and_sends_secret_only_in_header(self):
        frames = LatestFrameStore()
        frame = frames.update(b"jpeg", time.monotonic())
        settings = YoloeClientSettings(
            enabled=True,
            service_url="https://obstacles.example.test",
            service_token="top-secret",
        )
        client = YoloeShadowClient(settings, frames, rotation_provider=lambda: 180)
        fake = _FakeClient(_FakeResponse(_obstacle_result(frame.sequence)))
        client._http_client = fake
        client._httpx = mock.Mock(TimeoutException=TimeoutError)
        output = io.StringIO()
        with redirect_stdout(output):
            await client._request_detection(frame)
        self.assertTrue(client.latest_perception()["available"])
        self.assertEqual(client.health()["requests_completed"], 1)
        args, kwargs = fake.calls[0]
        self.assertIn("/v1/obstacles", args[0])
        self.assertNotIn("top-secret", args[0])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer top-secret")
        self.assertEqual(kwargs["headers"]["X-Camera-Rotation-Deg"], "180")
        self.assertNotIn("top-secret", output.getvalue())

    async def test_fake_obstacle_service_integration(self):
        fake_service = FastAPI()

        @fake_service.post("/v1/obstacles")
        async def obstacles(
            request: Request,
            authorization: str = Header(...),
            x_frame_id: int = Header(...),
        ):
            self.assertEqual(authorization, "Bearer integration-secret")
            self.assertEqual(await request.body(), b"integration-jpeg")
            return _obstacle_result(x_frame_id, inference_ms=295.0)

        frames = LatestFrameStore()
        frame = frames.update(b"integration-jpeg", time.monotonic())
        client = YoloeShadowClient(
            YoloeClientSettings(
                enabled=True,
                service_url="https://fake-obstacles.test",
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
        perception = client.latest_perception()
        self.assertTrue(perception["available"])
        self.assertEqual(perception["frame_id"], frame.sequence)
        self.assertEqual(perception["objects"][0]["label"], "bollard")

    async def test_network_and_malformed_response_failures_are_contained(self):
        frames = LatestFrameStore()
        frame = frames.update(b"jpeg", time.monotonic())
        client = YoloeShadowClient(
            YoloeClientSettings(enabled=True, service_url="https://x", service_token="secret"),
            frames,
        )
        client._httpx = mock.Mock(TimeoutException=TimeoutError)
        for outcome in (OSError("dns"), _FakeResponse({"bad": True})):
            client._http_client = _FakeClient(outcome)
            output = io.StringIO()
            with redirect_stdout(output):
                await client._request_detection(frame)
        self.assertEqual(client.health()["request_failures"], 2)
        self.assertFalse(client.latest_perception()["available"])
        self.assertFalse(asyncio.current_task().cancelled())

    async def test_timeout_is_counted_and_does_not_escape(self):
        frames = LatestFrameStore()
        frame = frames.update(b"jpeg", time.monotonic())
        client = YoloeShadowClient(
            YoloeClientSettings(enabled=True, service_url="https://x", service_token="secret"),
            frames,
        )
        client._httpx = mock.Mock(TimeoutException=TimeoutError)
        client._http_client = _FakeClient(TimeoutError())
        await client._request_detection(frame)
        self.assertEqual(client.health()["request_timeouts"], 1)
        self.assertFalse(asyncio.current_task().cancelled())


if __name__ == "__main__":
    unittest.main()
