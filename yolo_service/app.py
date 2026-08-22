"""Independent authenticated YOLOv8n HTTP inference service."""
from __future__ import annotations

import asyncio
import math
import os
import secrets
import statistics
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
import torch
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from ultralytics import YOLO

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception:  # Health stays available when the optional runtime is absent.
    mp = None
    mp_python = None
    mp_vision = None

try:
    from .core import clamp01, normalize_detection
except ImportError:  # Docker runs this directory as the application root.
    from core import clamp01, normalize_detection


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "/app/yolov8n.pt")
HAND_MODEL_PATH = os.getenv("HAND_MODEL_PATH", "/app/hand_landmarker.task")
DEVICE = os.getenv("YOLO_DEVICE", "cpu").strip() or "cpu"
DEFAULT_CONFIDENCE = _env_float("YOLO_CONFIDENCE", 0.25, 0.0, 1.0)
IMAGE_SIZE = _env_int("YOLO_IMAGE_SIZE", 640, 32, 4096)
TORCH_THREADS = _env_int("YOLO_TORCH_THREADS", 1, 1, 64)
MAX_JPEG_BYTES = _env_int("YOLO_MAX_JPEG_BYTES", 300 * 1024, 1024, 20 * 1024 * 1024)
SERVICE_TOKEN = os.getenv("YOLO_SERVICE_TOKEN", "")

_model = None
_model_loaded = False
_model_error: Optional[str] = None
_started_at = time.monotonic()
_admission = threading.Lock()
_metrics_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo-inference")
_hand_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hand-inference")
_latencies: deque[float] = deque(maxlen=200)
_hand_latencies: deque[float] = deque(maxlen=200)
_metrics = {
    "requests": 0,
    "successful_inference": 0,
    "failures": 0,
    "overload_rejections": 0,
    "latest_inference_ms": None,
    "detection_count": 0,
}
_hand_model = None
_hand_model_loaded = False
_hand_model_error: Optional[str] = None
_hand_admission = threading.Lock()
_hand_metrics = {
    "hand_inference_count": 0,
    "hand_inference_failures": 0,
    "hand_busy_rejections": 0,
    "latest_hand_inference_ms": None,
    "latest_hand_count": 0,
}


def _load_model() -> None:
    global _model, _model_loaded, _model_error
    try:
        model_file = Path(MODEL_PATH)
        if not model_file.is_file():
            raise FileNotFoundError(MODEL_PATH)
        torch.set_num_threads(TORCH_THREADS)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        if DEVICE.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("cuda_requested_but_unavailable")
        _model = YOLO(str(model_file), task="detect")
        _model.to(DEVICE)
        _model_loaded = True
        _model_error = None
        print(f"[YOLO-SERVICE] model ready device={DEVICE} path={model_file.name}", flush=True)
    except Exception as exc:
        _model = None
        _model_loaded = False
        _model_error = type(exc).__name__
        print(f"[YOLO-SERVICE] model unavailable reason={_model_error}", flush=True)


def _load_hand_model() -> None:
    """Create the MediaPipe Tasks graph once for the process lifetime."""
    global _hand_model, _hand_model_loaded, _hand_model_error
    try:
        if mp_python is None or mp_vision is None:
            raise RuntimeError("mediapipe_unavailable")
        model_file = Path(HAND_MODEL_PATH)
        if not model_file.is_file():
            raise FileNotFoundError(HAND_MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(model_file),
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
        )
        _hand_model = mp_vision.HandLandmarker.create_from_options(options)
        _hand_model_loaded = True
        _hand_model_error = None
        print(
            f"[HAND-SERVICE] model ready max_hands=2 path={model_file.name}",
            flush=True,
        )
    except Exception as exc:
        _hand_model = None
        _hand_model_loaded = False
        _hand_model_error = type(exc).__name__
        print(
            f"[HAND-SERVICE] model unavailable reason={_hand_model_error}",
            flush=True,
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    loop = asyncio.get_running_loop()
    await asyncio.gather(
        loop.run_in_executor(_executor, _load_model),
        loop.run_in_executor(_hand_executor, _load_hand_model),
    )
    yield
    if _hand_model is not None:
        try:
            _hand_model.close()
        except Exception:
            pass
    _executor.shutdown(wait=False, cancel_futures=True)
    _hand_executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="AI Glasses YOLO Shadow Service", lifespan=lifespan)


def _authorized(authorization: Optional[str]) -> None:
    if not SERVICE_TOKEN:
        raise HTTPException(status_code=503, detail="service authentication is not configured")
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="unauthorized")
    supplied = authorization[len(prefix):]
    if not secrets.compare_digest(supplied, SERVICE_TOKEN):
        raise HTTPException(status_code=401, detail="unauthorized")


def _rotate(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _infer(jpeg: bytes, frame_id: int, rotation: int, confidence: float) -> dict:
    if _model is None:
        raise RuntimeError("model_unavailable")
    started_ns = time.monotonic_ns()
    encoded = np.frombuffer(jpeg, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("malformed_jpeg")
    image = _rotate(image, rotation)
    height, width = image.shape[:2]
    prediction = _model.predict(
        source=image,
        device=DEVICE,
        conf=confidence,
        imgsz=IMAGE_SIZE,
        batch=1,
        save=False,
        verbose=False,
    )[0]
    raw_objects = []
    if prediction.boxes is not None:
        xyxy_values = prediction.boxes.xyxy.detach().cpu().tolist()
        class_values = prediction.boxes.cls.detach().cpu().tolist()
        confidence_values = prediction.boxes.conf.detach().cpu().tolist()
        for xyxy, class_value, score in zip(xyxy_values, class_values, confidence_values):
            class_id = int(class_value)
            raw_objects.append({
                "class_id": class_id,
                "label": prediction.names[class_id],
                "confidence": score,
                "bbox_xyxy": xyxy,
            })
    objects = [normalize_detection(item, width, height) for item in raw_objects]
    finished_ns = time.monotonic_ns()
    return {
        "frame_id": frame_id,
        "inference_ms": max(0.0, (finished_ns - started_ns) / 1_000_000),
        "image_width": width,
        "image_height": height,
        "objects": objects,
    }


def _infer_hands(jpeg: bytes, frame_id: int) -> dict:
    """Infer in the JPEG's exact source space; no rotation or mirroring."""
    if _hand_model is None or mp is None:
        raise RuntimeError("hand_model_unavailable")
    started_ns = time.monotonic_ns()
    encoded = np.frombuffer(jpeg, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("malformed_jpeg")
    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    media_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(image_rgb),
    )
    prediction = _hand_model.detect(media_image)
    hands = []
    raw_landmark_sets = list(prediction.hand_landmarks)[:2]
    raw_handedness = list(prediction.handedness)[:2]
    for hand_index, landmark_set in enumerate(raw_landmark_sets):
        if len(landmark_set) != 21:
            raise RuntimeError("invalid_landmark_count")
        landmarks = []
        for landmark_id, landmark in enumerate(landmark_set):
            z = float(landmark.z)
            if not math.isfinite(z):
                raise RuntimeError("non_finite_landmark")
            landmarks.append({
                "id": landmark_id,
                "x": clamp01(landmark.x),
                "y": clamp01(landmark.y),
                "z": z,
            })
        categories = raw_handedness[hand_index] if hand_index < len(raw_handedness) else []
        category = categories[0] if categories else None
        handedness = str(getattr(category, "category_name", "Unknown") or "Unknown")
        handedness_score = clamp01(getattr(category, "score", 0.0))
        xs = [item["x"] for item in landmarks]
        ys = [item["y"] for item in landmarks]
        hands.append({
            "hand_index": hand_index,
            # MediaPipe's category is returned verbatim. The service does not
            # mirror the image or silently swap Left/Right.
            "handedness": handedness,
            "handedness_score": handedness_score,
            "landmarks": landmarks,
            "index_tip_norm": [landmarks[8]["x"], landmarks[8]["y"]],
            "wrist_norm": [landmarks[0]["x"], landmarks[0]["y"]],
            "hand_center_norm": [sum(xs) / 21.0, sum(ys) / 21.0],
            "bbox_norm": [min(xs), min(ys), max(xs), max(ys)],
        })
    finished_ns = time.monotonic_ns()
    return {
        "frame_id": frame_id,
        "image_width": width,
        "image_height": height,
        "inference_ms": max(0.0, (finished_ns - started_ns) / 1_000_000),
        "hands": hands,
    }


@app.get("/healthz")
def healthz():
    with _metrics_lock:
        metrics = dict(_metrics)
        latencies = list(_latencies)
        hand_metrics = dict(_hand_metrics)
        hand_latencies = list(_hand_latencies)
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]
        average = statistics.fmean(latencies)
    else:
        p95 = average = None
    if hand_latencies:
        ordered_hand = sorted(hand_latencies)
        hand_p95 = ordered_hand[max(0, math.ceil(len(ordered_hand) * 0.95) - 1)]
        hand_average = statistics.fmean(hand_latencies)
    else:
        hand_p95 = hand_average = None
    body = {
        # Preserve the existing YOLO readiness contract. Hand readiness is
        # additive and must not make /v1/detect unhealthy.
        "ok": _model_loaded and bool(SERVICE_TOKEN),
        "model_loaded": _model_loaded,
        "model_error": _model_error,
        "hand_model_loaded": _hand_model_loaded,
        "hand_model_error": _hand_model_error,
        "authentication_configured": bool(SERVICE_TOKEN),
        "device": DEVICE,
        "uptime_sec": round(time.monotonic() - _started_at, 3),
        "process_rss_bytes": psutil.Process().memory_info().rss,
        **metrics,
        **hand_metrics,
        "average_inference_ms": None if average is None else round(average, 3),
        "p95_inference_ms": None if p95 is None else round(p95, 3),
        "average_hand_inference_ms": (
            None if hand_average is None else round(hand_average, 3)
        ),
        "p95_hand_inference_ms": None if hand_p95 is None else round(hand_p95, 3),
    }
    return JSONResponse(body, status_code=200 if body["ok"] else 503)


@app.post("/v1/detect")
async def detect(
    request: Request,
    confidence: float = Query(DEFAULT_CONFIDENCE, ge=0.0, le=1.0),
    authorization: Optional[str] = Header(None),
    x_frame_id: int = Header(..., ge=1),
    x_frame_received_monotonic_ns: Optional[int] = Header(None, ge=0),
    x_camera_rotation_deg: int = Header(0),
):
    del x_frame_received_monotonic_ns  # Accepted for traceability; never relabeled as capture time.
    _authorized(authorization)
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="model unavailable")
    if x_camera_rotation_deg not in (0, 90, 180, 270):
        raise HTTPException(status_code=400, detail="invalid camera rotation")
    if not _admission.acquire(blocking=False):
        with _metrics_lock:
            _metrics["overload_rejections"] += 1
        raise HTTPException(status_code=429, detail="inference busy")
    with _metrics_lock:
        _metrics["requests"] += 1
    try:
        jpeg_buffer = bytearray()
        async for chunk in request.stream():
            if len(jpeg_buffer) + len(chunk) > MAX_JPEG_BYTES:
                raise HTTPException(status_code=413, detail="invalid JPEG size")
            jpeg_buffer.extend(chunk)
        if not jpeg_buffer:
            raise HTTPException(status_code=413, detail="invalid JPEG size")
        inference_future = asyncio.get_running_loop().run_in_executor(
            _executor,
            _infer,
            bytes(jpeg_buffer),
            x_frame_id,
            x_camera_rotation_deg,
            confidence,
        )
        try:
            result = await asyncio.shield(inference_future)
        except asyncio.CancelledError:
            # Keep the single admission slot until the resident model actually
            # finishes; otherwise repeated client disconnects could enqueue
            # stale executor work behind an inference that is still running.
            try:
                await inference_future
            finally:
                raise
        with _metrics_lock:
            _metrics["successful_inference"] += 1
            _metrics["latest_inference_ms"] = result["inference_ms"]
            _metrics["detection_count"] = len(result["objects"])
            _latencies.append(result["inference_ms"])
        return result
    except HTTPException:
        with _metrics_lock:
            _metrics["failures"] += 1
        raise
    except Exception as exc:
        with _metrics_lock:
            _metrics["failures"] += 1
            failures = _metrics["failures"]
        if failures == 1 or failures % 10 == 0:
            print(f"[YOLO-SERVICE] inference failure count={failures} reason={type(exc).__name__}", flush=True)
        raise HTTPException(status_code=422, detail="inference failed") from None
    finally:
        _admission.release()


@app.post("/v1/hands")
async def hands(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_frame_id: int = Header(..., ge=1),
    x_frame_received_monotonic_ns: Optional[int] = Header(None, ge=0),
):
    # Accepted for traceability only. The backend receive timestamp is not a
    # camera capture timestamp and is never relabeled as one.
    del x_frame_received_monotonic_ns
    _authorized(authorization)
    if not _hand_model_loaded:
        raise HTTPException(status_code=503, detail="hand model unavailable")
    if not _hand_admission.acquire(blocking=False):
        with _metrics_lock:
            _hand_metrics["hand_busy_rejections"] += 1
        raise HTTPException(status_code=429, detail="hand inference busy")
    try:
        jpeg_buffer = bytearray()
        async for chunk in request.stream():
            if len(jpeg_buffer) + len(chunk) > MAX_JPEG_BYTES:
                raise HTTPException(status_code=413, detail="invalid JPEG size")
            jpeg_buffer.extend(chunk)
        if not jpeg_buffer:
            raise HTTPException(status_code=413, detail="invalid JPEG size")
        inference_future = asyncio.get_running_loop().run_in_executor(
            _hand_executor,
            _infer_hands,
            bytes(jpeg_buffer),
            x_frame_id,
        )
        try:
            result = await asyncio.shield(inference_future)
        except asyncio.CancelledError:
            # Keep admission until native inference actually exits. This
            # prevents disconnected callers from building an executor queue.
            try:
                await inference_future
            finally:
                raise
        with _metrics_lock:
            _hand_metrics["hand_inference_count"] += 1
            _hand_metrics["latest_hand_inference_ms"] = result["inference_ms"]
            _hand_metrics["latest_hand_count"] = len(result["hands"])
            _hand_latencies.append(result["inference_ms"])
        return result
    except HTTPException:
        with _metrics_lock:
            _hand_metrics["hand_inference_failures"] += 1
        raise
    except Exception as exc:
        with _metrics_lock:
            _hand_metrics["hand_inference_failures"] += 1
            failures = _hand_metrics["hand_inference_failures"]
        if failures == 1 or failures % 10 == 0:
            print(
                f"[HAND-SERVICE] inference failure count={failures} "
                f"reason={type(exc).__name__}",
                flush=True,
            )
        raise HTTPException(status_code=422, detail="hand inference failed") from None
    finally:
        _hand_admission.release()
