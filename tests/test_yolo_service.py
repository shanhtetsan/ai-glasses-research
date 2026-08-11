import asyncio
import importlib
import json
import sys
import types
import unittest
from unittest import mock

import httpx


def _import_service_with_ml_stubs():
    cv2 = types.ModuleType("cv2")
    cv2.ROTATE_90_CLOCKWISE = 0
    cv2.ROTATE_180 = 1
    cv2.ROTATE_90_COUNTERCLOCKWISE = 2
    cv2.IMREAD_COLOR = 1
    cv2.rotate = lambda image, _rotation: image
    cv2.imdecode = lambda _encoded, _mode: None

    torch = types.ModuleType("torch")
    torch.set_num_threads = lambda _count: None
    torch.set_num_interop_threads = lambda _count: None
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = mock.Mock

    with mock.patch.dict(sys.modules, {
        "cv2": cv2,
        "torch": torch,
        "ultralytics": ultralytics,
    }):
        sys.modules.pop("yolo_service.app", None)
        return importlib.import_module("yolo_service.app")


service = _import_service_with_ml_stubs()


class YoloServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        service.SERVICE_TOKEN = "service-secret"
        service._model_loaded = True
        service._model_error = None
        service._model = object()
        with service._metrics_lock:
            service._metrics.update({
                "requests": 0,
                "successful_inference": 0,
                "failures": 0,
                "overload_rejections": 0,
                "latest_inference_ms": None,
                "detection_count": 0,
            })
            service._latencies.clear()

    async def _request(self, method, path, **kwargs):
        transport = httpx.ASGITransport(app=service.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://yolo-service.test",
        ) as client:
            return await client.request(method, path, **kwargs)

    async def test_healthz_reports_model_and_aggregate_telemetry(self):
        response = await self._request("GET", "/healthz")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["model_loaded"])
        self.assertEqual(body["device"], service.DEVICE)
        self.assertIn("process_rss_bytes", body)
        self.assertIn("average_inference_ms", body)
        self.assertIn("p95_inference_ms", body)

        service._model_loaded = False
        response = await self._request("GET", "/healthz")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["ok"])

    async def test_detect_authenticates_and_uses_mocked_inference(self):
        expected = {
            "frame_id": 1234,
            "inference_ms": 12.5,
            "image_width": 320,
            "image_height": 240,
            "objects": [{
                "class_id": 0,
                "label": "person",
                "confidence": 0.9,
                "bbox_norm": [0.1, 0.2, 0.8, 0.9],
                "center_norm": [0.45, 0.55],
            }],
        }
        infer = mock.Mock(return_value=expected)
        with mock.patch.object(service, "_infer", infer):
            unauthorized = await self._request(
                "POST",
                "/v1/detect",
                content=b"jpeg",
                headers={"X-Frame-ID": "1234"},
            )
            self.assertEqual(unauthorized.status_code, 401)
            response = await self._request(
                "POST",
                "/v1/detect?confidence=0.4",
                content=b"jpeg",
                headers={
                    "Authorization": "Bearer service-secret",
                    "Content-Type": "image/jpeg",
                    "X-Frame-ID": "1234",
                    "X-Camera-Rotation-Deg": "90",
                    "X-Frame-Received-Monotonic-Ns": "999",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        infer.assert_called_once_with(b"jpeg", 1234, 90, 0.4)
        health = (await self._request("GET", "/healthz")).json()
        self.assertEqual(health["successful_inference"], 1)
        self.assertEqual(health["detection_count"], 1)

    async def test_detect_rejects_busy_request_without_queueing(self):
        self.assertTrue(service._admission.acquire(blocking=False))
        try:
            response = await self._request(
                "POST",
                "/v1/detect",
                content=b"jpeg",
                headers={
                    "Authorization": "Bearer service-secret",
                    "X-Frame-ID": "1",
                },
            )
        finally:
            service._admission.release()
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["detail"], "inference busy")
        self.assertEqual(service._metrics["overload_rejections"], 1)
        self.assertEqual(service._metrics["requests"], 0)


if __name__ == "__main__":
    unittest.main()
