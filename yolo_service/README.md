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

Segmentation (`road_crossing`/`blind_path`) is independently enabled with
`ENABLE_YOLO_SEG=true`; the client reuses `YOLO_SERVICE_URL`/`YOLO_SERVICE_TOKEN`
unless `YOLO_SEG_SERVICE_URL`/`YOLO_SEG_SERVICE_TOKEN` are set explicitly.
`YOLO_SEG_MIN_INTERVAL_SEC` defaults to `4.0`, `YOLO_SEG_REQUEST_TIMEOUT_SEC`
to `8.0`, and `YOLO_SEG_CACHE_STALE_SEC` to `6.0` — calibrated to this
model's server-measured deployment latency (avg 3690ms, p95 5323ms on the
production CPU class), not its much faster ~220-280ms local latency. Don't
reset these to the small defaults the other detectors use without re-checking
`/healthz`'s `average_seg_inference_ms`/`p95_seg_inference_ms` first.

Open-vocabulary obstacle detection is independently enabled with
`ENABLE_YOLOE_OBSTACLES=true`, with the same `YOLO_SERVICE_URL`/`YOLO_SERVICE_TOKEN`
fallback via `YOLOE_SERVICE_URL`/`YOLOE_SERVICE_TOKEN`. The service prompts
YOLOE with a fixed obstacle whitelist (`OBSTACLE_WHITELIST` in `app.py`,
copied from `obstacle_detector_client.py`) once at startup, using CLIP text
embeddings precomputed offline — see "Precomputing YOLOE embeddings" below.
It never imports CLIP or reaches the network at runtime.
`YOLOE_MIN_INTERVAL_SEC` defaults to `2.5`, `YOLOE_REQUEST_TIMEOUT_SEC` to
`20.0` (server-measured avg 1956ms with one 14661ms outlier — no p95 yet;
revisit once one exists), and `YOLOE_CACHE_STALE_SEC` to `4.0`, deliberately
well under that outlier so a detection built from a 14-second-old frame
reads as stale rather than current.

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

`POST /v1/segment?confidence=0.25` (`yolo-seg.pt`) and
`POST /v1/obstacles?confidence=0.25` (`yoloe-11l-seg.pt`, prompted with the
fixed obstacle whitelist) use the same header contract, status codes, and
per-endpoint single-slot admission as `/v1/detect`, including
`X-Camera-Rotation-Deg`. Their objects carry an additional
`mask_coverage_norm` (fraction of the frame the instance mask covers,
resolution-independent):

```json
{
  "frame_id": 1234,
  "inference_ms": 210.3,
  "image_width": 320,
  "image_height": 240,
  "objects": [
    {
      "class_id": 1,
      "label": "blind_path",
      "confidence": 0.83,
      "bbox_norm": [0.0, 0.4, 1.0, 1.0],
      "center_norm": [0.5, 0.7],
      "mask_coverage_norm": 0.32
    }
  ]
}
```

`/v1/segment`'s checkpoint contract is `{0: "road_crossing", 1: "blind_path"}`;
the service refuses to serve a checkpoint whose class names don't match this
exactly (fails at load, not at inference).

## Precomputing YOLOE embeddings

`/v1/obstacles`'s whitelist is static, so its CLIP text embeddings are
computed once, offline, with `scripts/precompute_yoloe_embeddings.py`, rather
than in the service itself — calling `get_text_pe()` in the service would
import OpenAI's CLIP (an unpinned `pip install git+...` on first use) and
download the ~338MB ViT-B/32 checkpoint from OpenAI's CDN at container
startup. Run this once wherever CLIP already resolves (e.g. after running
`obstacle_detector_client.py` locally) and commit the resulting
`model/yoloe-whitelist-embeddings.pt`:

```sh
python3 scripts/precompute_yoloe_embeddings.py
```

Re-run it only if the whitelist in `obstacle_detector_client.py`/`app.py`'s
`OBSTACLE_WHITELIST` ever changes — the two must stay in sync.
