import asyncio
import importlib
import json
import sys
import types
import unittest
from pathlib import Path
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
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = mock.Mock
    ultralytics.YOLOE = mock.Mock

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
        service._hand_model_loaded = True
        service._hand_model_error = None
        service._hand_model = object()
        service._seg_model_loaded = True
        service._seg_model_error = None
        service._seg_model = object()
        service._obstacle_model_loaded = True
        service._obstacle_model_error = None
        service._obstacle_model = object()
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
            service._hand_metrics.update({
                "hand_inference_count": 0,
                "hand_inference_failures": 0,
                "hand_busy_rejections": 0,
                "latest_hand_inference_ms": None,
                "latest_hand_count": 0,
            })
            service._hand_latencies.clear()
            service._seg_metrics.update({
                "seg_requests": 0,
                "seg_successful_inference": 0,
                "seg_failures": 0,
                "seg_overload_rejections": 0,
                "latest_seg_inference_ms": None,
                "seg_detection_count": 0,
            })
            service._seg_latencies.clear()
            service._obstacle_metrics.update({
                "obstacle_requests": 0,
                "obstacle_successful_inference": 0,
                "obstacle_failures": 0,
                "obstacle_overload_rejections": 0,
                "latest_obstacle_inference_ms": None,
                "obstacle_detection_count": 0,
            })
            service._obstacle_latencies.clear()

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
        self.assertTrue(body["hand_model_loaded"])
        self.assertIn("hand_inference_count", body)
        self.assertTrue(body["seg_model_loaded"])
        self.assertIn("seg_requests", body)
        self.assertIn("average_seg_inference_ms", body)
        self.assertTrue(body["obstacle_model_loaded"])
        self.assertIn("obstacle_requests", body)
        self.assertIn("average_obstacle_inference_ms", body)

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

    async def test_segment_authenticates_and_uses_mocked_inference(self):
        expected = {
            "frame_id": 55,
            "inference_ms": 200.0,
            "image_width": 320,
            "image_height": 240,
            "objects": [{
                "class_id": 1,
                "label": "blind_path",
                "confidence": 0.8,
                "bbox_norm": [0.0, 0.4, 1.0, 1.0],
                "center_norm": [0.5, 0.7],
                "mask_coverage_norm": 0.35,
            }],
        }
        infer = mock.Mock(return_value=expected)
        with mock.patch.object(service, "_infer_segment", infer):
            unauthorized = await self._request(
                "POST",
                "/v1/segment",
                content=b"jpeg",
                headers={"X-Frame-ID": "55"},
            )
            self.assertEqual(unauthorized.status_code, 401)
            response = await self._request(
                "POST",
                "/v1/segment?confidence=0.4",
                content=b"jpeg",
                headers={
                    "Authorization": "Bearer service-secret",
                    "Content-Type": "image/jpeg",
                    "X-Frame-ID": "55",
                    "X-Camera-Rotation-Deg": "0",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        infer.assert_called_once_with(b"jpeg", 55, 0, 0.4)
        health = (await self._request("GET", "/healthz")).json()
        self.assertEqual(health["seg_successful_inference"], 1)
        self.assertEqual(health["seg_detection_count"], 1)

    async def test_segment_rejects_busy_request_without_queueing(self):
        self.assertTrue(service._seg_admission.acquire(blocking=False))
        try:
            response = await self._request(
                "POST",
                "/v1/segment",
                content=b"jpeg",
                headers={"Authorization": "Bearer service-secret", "X-Frame-ID": "1"},
            )
        finally:
            service._seg_admission.release()
        self.assertEqual(response.status_code, 429)
        self.assertEqual(service._seg_metrics["seg_overload_rejections"], 1)
        self.assertEqual(service._seg_metrics["seg_requests"], 0)

    async def test_segment_unavailable_returns_503(self):
        service._seg_model_loaded = False
        response = await self._request(
            "POST",
            "/v1/segment",
            content=b"jpeg",
            headers={"Authorization": "Bearer service-secret", "X-Frame-ID": "1"},
        )
        self.assertEqual(response.status_code, 503)

    async def test_obstacles_authenticates_and_uses_mocked_inference(self):
        expected = {
            "frame_id": 77,
            "inference_ms": 300.0,
            "image_width": 320,
            "image_height": 240,
            "objects": [{
                "class_id": 4,
                "label": "bollard",
                "confidence": 0.6,
                "bbox_norm": [0.2, 0.2, 0.4, 0.9],
                "center_norm": [0.3, 0.55],
                "mask_coverage_norm": 0.05,
            }],
        }
        infer = mock.Mock(return_value=expected)
        with mock.patch.object(service, "_infer_obstacles", infer):
            unauthorized = await self._request(
                "POST",
                "/v1/obstacles",
                content=b"jpeg",
                headers={"X-Frame-ID": "77"},
            )
            self.assertEqual(unauthorized.status_code, 401)
            response = await self._request(
                "POST",
                "/v1/obstacles",
                content=b"jpeg",
                headers={
                    "Authorization": "Bearer service-secret",
                    "Content-Type": "image/jpeg",
                    "X-Frame-ID": "77",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        infer.assert_called_once_with(b"jpeg", 77, 0, service.OBSTACLE_DEFAULT_CONFIDENCE)
        health = (await self._request("GET", "/healthz")).json()
        self.assertEqual(health["obstacle_successful_inference"], 1)
        self.assertEqual(health["obstacle_detection_count"], 1)

    async def test_obstacles_rejects_busy_request_without_queueing(self):
        self.assertTrue(service._obstacle_admission.acquire(blocking=False))
        try:
            response = await self._request(
                "POST",
                "/v1/obstacles",
                content=b"jpeg",
                headers={"Authorization": "Bearer service-secret", "X-Frame-ID": "1"},
            )
        finally:
            service._obstacle_admission.release()
        self.assertEqual(response.status_code, 429)
        self.assertEqual(service._obstacle_metrics["obstacle_overload_rejections"], 1)

    def test_seg_model_loader_rejects_mismatched_class_names(self):
        fake_model = types.SimpleNamespace(
            to=lambda _device: None,
            names={0: "something_else", 1: "blind_path"},
        )
        with mock.patch.object(service, "SEG_MODEL_PATH", __file__), \
                mock.patch.object(service, "YOLO", mock.Mock(return_value=fake_model)):
            service._load_seg_model()
        self.assertFalse(service._seg_model_loaded)
        self.assertEqual(service._seg_model_error, "RuntimeError")

    def test_obstacle_model_loader_loads_precomputed_embeddings_once(self):
        fake_model = types.SimpleNamespace(
            to=lambda _device: None,
            set_classes=mock.Mock(),
        )
        fake_load = mock.Mock(return_value="embeddings")
        with mock.patch.object(service, "OBSTACLE_MODEL_PATH", __file__), \
                mock.patch.object(service, "OBSTACLE_EMBEDDINGS_PATH", __file__), \
                mock.patch.object(service, "YOLOE", mock.Mock(return_value=fake_model)), \
                mock.patch.object(service.torch, "load", fake_load, create=True):
            service._load_obstacle_model()
        self.assertTrue(service._obstacle_model_loaded)
        fake_load.assert_called_once()
        fake_model.set_classes.assert_called_once_with(service.OBSTACLE_WHITELIST, "embeddings")
        self.assertFalse(hasattr(fake_model, "get_text_pe"))

    def test_obstacle_model_loader_fails_when_embeddings_missing(self):
        fake_model = types.SimpleNamespace(to=lambda _device: None)
        with mock.patch.object(service, "OBSTACLE_MODEL_PATH", __file__), \
                mock.patch.object(service, "OBSTACLE_EMBEDDINGS_PATH", "/no/such/embeddings.pt"), \
                mock.patch.object(service, "YOLOE", mock.Mock(return_value=fake_model)):
            service._load_obstacle_model()
        self.assertFalse(service._obstacle_model_loaded)
        self.assertEqual(service._obstacle_model_error, "FileNotFoundError")

    def test_mask_coverage_ratio_matches_thresholded_pixel_fraction(self):
        class _FakeTensor:
            def __init__(self, array):
                self._array = array
                self.dtype = "float32"

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._array

        mask = service.np.zeros((4, 4), dtype=service.np.float32)
        mask[0:2, 0:2] = 1.0  # 4 of 16 pixels covered
        prediction = types.SimpleNamespace(
            masks=types.SimpleNamespace(data=[_FakeTensor(mask)]),
        )
        ratios = service._mask_coverage_ratios(prediction)
        self.assertEqual(ratios, [0.25])

    def test_normalize_segmentation_adds_clamped_mask_coverage(self):
        detection = service.normalize_segmentation({
            "class_id": 1,
            "label": "blind_path",
            "confidence": 0.8,
            "bbox_xyxy": [0, 0, 320, 240],
            "mask_coverage_norm": 1.4,
        }, 320, 240)
        self.assertEqual(detection["bbox_norm"], [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(detection["mask_coverage_norm"], 1.0)

    async def test_hands_requires_auth_and_accepts_valid_zero_hand_result(self):
        expected = {
            "frame_id": 23,
            "image_width": 640,
            "image_height": 480,
            "inference_ms": 40.0,
            "hands": [],
        }
        infer = mock.Mock(return_value=expected)
        with mock.patch.object(service, "_infer_hands", infer):
            unauthorized = await self._request(
                "POST", "/v1/hands", content=b"jpeg", headers={"X-Frame-ID": "23"}
            )
            response = await self._request(
                "POST",
                "/v1/hands",
                content=b"jpeg",
                headers={
                    "Authorization": "Bearer service-secret",
                    "Content-Type": "image/jpeg",
                    "X-Frame-ID": "23",
                    "X-Frame-Received-Monotonic-Ns": "999",
                },
            )
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        infer.assert_called_once_with(b"jpeg", 23)
        self.assertEqual(service._hand_metrics["hand_inference_count"], 1)
        self.assertEqual(service._hand_metrics["latest_hand_count"], 0)

    async def test_hands_invalid_jpeg_is_rejected_safely(self):
        service._hand_model = object()
        response = await self._request(
            "POST",
            "/v1/hands",
            content=b"not-a-jpeg",
            headers={
                "Authorization": "Bearer service-secret",
                "X-Frame-ID": "1",
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(service._hand_metrics["hand_inference_failures"], 1)

    async def test_hands_busy_rejects_without_queueing(self):
        self.assertTrue(service._hand_admission.acquire(blocking=False))
        try:
            response = await self._request(
                "POST",
                "/v1/hands",
                content=b"jpeg",
                headers={
                    "Authorization": "Bearer service-secret",
                    "X-Frame-ID": "1",
                },
            )
        finally:
            service._hand_admission.release()
        self.assertEqual(response.status_code, 429)
        self.assertEqual(service._hand_metrics["hand_busy_rejections"], 1)
        self.assertEqual(service._hand_metrics["hand_inference_count"], 0)

    def test_hand_schema_is_normalized_and_limited_to_two_hands(self):
        landmarks = [
            types.SimpleNamespace(x=index / 20, y=1 - index / 20, z=-0.01 * index)
            for index in range(21)
        ]
        prediction = types.SimpleNamespace(
            hand_landmarks=[landmarks, landmarks, landmarks],
            handedness=[
                [types.SimpleNamespace(category_name="Left", score=0.98)],
                [types.SimpleNamespace(category_name="Right", score=0.97)],
                [types.SimpleNamespace(category_name="Left", score=0.96)],
            ],
        )
        fake_model = types.SimpleNamespace(detect=lambda _image: prediction)
        fake_mp = types.SimpleNamespace(
            Image=lambda **_kwargs: object(),
            ImageFormat=types.SimpleNamespace(SRGB=1),
        )
        image = service.np.zeros((480, 640, 3), dtype=service.np.uint8)
        with mock.patch.object(service, "_hand_model", fake_model), \
                mock.patch.object(service, "mp", fake_mp), \
                mock.patch.object(service.cv2, "imdecode", return_value=image), \
                mock.patch.object(service.cv2, "cvtColor", return_value=image, create=True), \
                mock.patch.object(service.cv2, "COLOR_BGR2RGB", 4, create=True):
            result = service._infer_hands(b"jpeg", 9)
        self.assertEqual(len(result["hands"]), 2)
        self.assertEqual(len(result["hands"][0]["landmarks"]), 21)
        self.assertEqual(result["hands"][0]["index_tip_norm"], [0.4, 0.6])
        self.assertEqual(result["hands"][0]["wrist_norm"], [0.0, 1.0])
        self.assertEqual(result["hands"][0]["bbox_norm"], [0.0, 0.0, 1.0, 1.0])

    def test_hand_model_creation_is_not_on_request_path(self):
        source = Path("yolo_service/app.py").read_text(encoding="utf-8")
        endpoint = source.split('@app.post("/v1/hands")', 1)[1]
        self.assertNotIn("create_from_options", endpoint)
        self.assertIn("run_in_executor", endpoint)

    def test_hand_model_loader_creates_one_two_hand_cpu_instance(self):
        model = types.SimpleNamespace(close=lambda: None)
        create = mock.Mock(return_value=model)
        base_options = mock.Mock(return_value="base-options")
        landmarker_options = mock.Mock(return_value="hand-options")
        fake_python = types.SimpleNamespace(
            BaseOptions=base_options,
        )
        fake_python.BaseOptions.Delegate = types.SimpleNamespace(CPU="cpu")
        fake_vision = types.SimpleNamespace(
            RunningMode=types.SimpleNamespace(IMAGE="image"),
            HandLandmarkerOptions=landmarker_options,
            HandLandmarker=types.SimpleNamespace(create_from_options=create),
        )
        with mock.patch.object(service, "HAND_MODEL_PATH", "hand_landmarker.task"), \
                mock.patch.object(service, "mp_python", fake_python), \
                mock.patch.object(service, "mp_vision", fake_vision):
            service._load_hand_model()
        base_options.assert_called_once_with(
            model_asset_path="hand_landmarker.task",
            delegate="cpu",
        )
        landmarker_options.assert_called_once_with(
            base_options="base-options",
            running_mode="image",
            num_hands=2,
        )
        create.assert_called_once_with("hand-options")
        self.assertIs(service._hand_model, model)


if __name__ == "__main__":
    unittest.main()
