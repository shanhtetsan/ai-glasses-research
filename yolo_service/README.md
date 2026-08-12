# Perception inference service

This service is deployed independently from the glasses/Gemini backend. Build
from the repository root so the validated `yolov8n.pt` is available:

```sh
docker build -f yolo_service/Dockerfile -t ai-glasses-yolo-service .
```

Required secret:

```sh
flyctl secrets set YOLO_SERVICE_TOKEN=<shared-secret> -c yolo_service/fly.toml
```

Deploying the service does not enable the client in the main backend. Configure
`YOLO_SERVICE_URL` and the matching secret there, then explicitly set
`ENABLE_YOLO=true`. Keep the service token out of TOML and logs.

Hand tracking is independently enabled with `ENABLE_HAND_TRACKING=true`.
`HAND_SERVICE_URL` and `HAND_SERVICE_TOKEN` may be set explicitly; when omitted,
the client reuses `YOLO_SERVICE_URL` and `YOLO_SERVICE_TOKEN` because both
endpoints live in this authenticated service. `HAND_MIN_INTERVAL_SEC` defaults
to `0.33` and `HAND_CACHE_STALE_SEC` defaults to `1.25`.

The default image uses CPU-only PyTorch. `PYTORCH_INDEX_URL` is a build argument
so a compatible GPU wheel index and `YOLO_DEVICE` can be selected for a future
GPU deployment without changing the API contract.

The same process loads the repository's existing `hand_landmarker.task` once
at startup. Hand inference has its own single admission slot and executor, so
a concurrent hand request receives HTTP 429 instead of entering a queue.

## HTTP contract

`GET /healthz` is unauthenticated so the container platform can probe it. It
returns HTTP 200 only when both the model and service authentication are ready;
otherwise it returns HTTP 503. The JSON body includes aggregate health and
latency telemetry, never detections or credentials.

`POST /v1/detect?confidence=0.25` accepts the JPEG as the raw request body with:

```text
Authorization: Bearer <YOLO_SERVICE_TOKEN>
Content-Type: image/jpeg
X-Frame-ID: 1234
X-Frame-Received-Monotonic-Ns: 1234567890
X-Camera-Rotation-Deg: 0
```

The receive timestamp is optional; the frame ID is required and is echoed in
the response. Rotation must be 0, 90, 180, or 270 degrees. A successful response
has this shape:

```json
{
  "frame_id": 1234,
  "inference_ms": 42.5,
  "image_width": 320,
  "image_height": 240,
  "objects": [
    {
      "class_id": 0,
      "label": "person",
      "confidence": 0.93,
      "bbox_norm": [0.1, 0.05, 0.7, 0.95],
      "center_norm": [0.4, 0.5]
    }
  ]
}
```

Normalized values are clamped to `[0, 1]`. The service returns 401 for failed
authentication, 429 when its single inference slot is busy, 413 for an invalid
body size, 422 for malformed JPEG/inference failure, and 503 when the model is
unavailable.

`POST /v1/hands` uses the same bearer token and raw JPEG body. It requires
`X-Frame-ID` and accepts `X-Frame-Received-Monotonic-Ns`. It intentionally has
no rotation or mirror parameter: the caller sends the already-canonical RGB
JPEG and every normalized coordinate is in that exact source space. A response
contains at most two hands, each with 21 landmarks plus `index_tip_norm`,
`wrist_norm`, `hand_center_norm`, and `bbox_norm`.
