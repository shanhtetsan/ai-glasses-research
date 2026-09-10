# AI Smart Glasses for Blind and Low-Vision Navigation

Wearable glasses that describe surroundings and give spoken navigation guidance to blind and low-vision (BLV) users, using on-device sensors and a multimodal AI model.

**BMCC undergraduate research** · Advisor: Prof. Hao Tang · Participant testing in partnership with **VISIONS/Services for the Blind and Visually Impaired**, NYC.

> **New here? Read [`ONBOARDING.md`](./ONBOARDING.md) first, then [`PROJECT_CONTEXT.md`](./PROJECT_CONTEXT.md).**
> `PROJECT_CONTEXT.md` holds the design decisions and known failure modes. The code will not tell you why it's shaped this way.

---

## How it works

```
┌─────────────────────────────┐
│  GLASSES  (XIAO ESP32-S3)   │
│  camera · thermal · IMU     │
│  mic · speaker              │
└──────────────┬──────────────┘
               │  2 persistent WebSockets
               │  wsCamThermal  ·  wsAud
               ▼
┌─────────────────────────────┐         ┌──────────────────┐
│  BACKEND  (FastAPI, Fly.io) │ ◄─────► │  Gemini Live API │
│  stateful singleton         │         │  scene guidance  │
└──────────────┬──────────────┘         └──────────────────┘
               │
               ▼
┌─────────────────────────────┐
│  RESEARCHER DASHBOARD       │
│  Session · Diagnostics      │
└─────────────────────────────┘
```

The device streams camera, thermal, and audio to the backend. The backend forwards them to Google's Gemini Live API, which returns spoken guidance streamed back to the device speaker. A browser dashboard lets a researcher monitor and control a live session.

**The backend is a stateful singleton.** It holds live sockets and session state in memory. It cannot be horizontally scaled — one machine only.

---

## Hardware

| Component | Part | Role |
|---|---|---|
| MCU | XIAO ESP32-S3 Sense | Dual-core 240 MHz, WiFi, 8 MB PSRAM |
| Camera | OV2640 | Scene capture |
| Thermal | MLX90640 (24×32) | Thermal array |
| IMU | MPU-6050 (GY-521 breakout, I2C) | Orientation / motion — replaced ICM42688/SPI; `compile.ino`'s file header comment still says ICM42688, code does not |
| Mic | PDM | User speech |
| Speaker | MAX98357A (I2S) | Guidance playback |

Wiring: [`wiring_diagram.png`](./wiring_diagram.png) — **not yet committed.** The lead has this file locally and will add it; link is left in place for when it lands.

---

## Repository layout

All paths below confirmed against the working tree.

### Firmware
| File | Role |
|---|---|
| `compile/compile.ino` | Entire device firmware — capture loop, both WebSocket clients, I2S playback, sensor reads |

### Backend (Python / FastAPI)
| File | Role |
|---|---|
| `app_main.py` | Main application (4,140 lines). All HTTP endpoints and WebSocket handlers |
| `gemini_live_client.py` | Google Gemini Live API session management |
| `stability_runtime.py` | Dependency-free wire-protocol / message-parsing primitives shared by the backend and its tests (mic-loss control frames, camera/thermal/IMU/status struct layouts) — not strictly "stability-mode flags," but does back the STABILITY_MODE-era hardening |
| `perception_orientation.py` | Orientation / spatial reasoning helpers |
| `perception_fusion.py` | Fuses MediaPipe hand-landmark tracking with YOLO object detections: extracts a requested object from an utterance (matched against the validated `yolov8n.pt` label vocabulary + a small alias table), classifies hand gestures, tracks which detected object a hand is sticky-directed at (IoU + persistence across frames, per-axis hysteresis), and builds the compact "guidance facts" the Gemini prompt consumes. Imported by `app_main.py`, has its own test file (`tests/test_perception_fusion.py`). **Omitted from the original draft.** |
| `vision_backend.py` | **Hard dependency — imported unconditionally** at `app_main.py:253`, unlike the torch/mediapipe/dashscope imports a few lines above it, which are wrapped in `try/except` and degrade gracefully when absent. `vision_backend.py` has no such guard: if it or its dependencies are missing, the server fails to start. It's a switch point that reads `MODEL_BACKEND` (`gemini` → `gemini_client.py`, or `qwen` → loads a local Qwen2.5-Omni-3B onto MPS) and re-exports `stream_chat`/`OmniStreamPiece`. Note this is a **separate env var from `AI_BACKEND`** (see "Running locally") — `AI_BACKEND` picks the primary conversational pipeline (`gemini_live`/`gemini_regular`/`qwen`), `MODEL_BACKEND` picks what `vision_backend.py` re-exports for the non-`gemini_live` code paths. `requirements-cloud.txt` pins Pillow specifically because `gemini_client.py` (the default backend this module imports) decodes images via `PIL.Image`. **Omitted from the original draft**, and the draft's framing that all heavy imports degrade gracefully was wrong for this one. |
| `yolo_client.py` | YOLO object detection **client** — talks to the standalone `yolo_service/` microservice below over HTTP; not deployed to cloud *as part of the main backend* (it has no torch/ultralytics dependency itself, just an HTTP client) |
| `research_exporter.py` | Research data capture. Session gate persisted to disk (atomic write via `os.replace`), 6-hour auto-expiry (`DEFAULT_SESSION_TTL_SEC`), `publish_event()` explicitly guaranteed never to raise into hot paths (camera/audio/Gemini/turn code) |

### Standalone YOLO + hand-tracking microservice — `yolo_service/`

Deployed **independently** from the main backend, as its own Fly app (`ai-glasses-yolo-perception`, own `Dockerfile` + `fly.toml`, `internal_port = 8080`). `yolo_service/README.md` is explicit that deploying this service does **not** enable the client in the main backend — that requires setting `YOLO_SERVICE_URL` + `ENABLE_YOLO=true` in the main backend's own config.

It's an authenticated FastAPI app (`yolo_service/app.py`) running YOLOv8n (`ultralytics`, CPU by default) for object detection and, in the same process, MediaPipe hand-landmark tracking (`hand_landmarker.task`) behind a separate opt-in flag (`ENABLE_HAND_TRACKING`). `GET /healthz` is unauthenticated for the platform health probe; everything else requires the shared `YOLO_SERVICE_TOKEN`.

**Was it running during P01/P02?** The main backend's `fly.toml` already had `ENABLE_YOLO='true'` and `YOLO_SERVICE_URL` pointed at `ai-glasses-yolo-perception.fly.dev` as of commit `a33eab8` (the build both P01 and P02 likely ran, per `PROJECT_CONTEXT.md` §7) — the commit that added `yolo_service/` (`7a84056`, 2026-08-10) predates it. So the config was in place. **[VERIFY]** whether the separate `ai-glasses-yolo-perception` Fly app was actually deployed and reachable at session time is operational/deployment history this audit can't confirm from source alone — worth a lead check if it matters for the writeup.

### Frontend (researcher dashboard)
| File | Role |
|---|---|
| `templates/index.html` | **Live** dashboard markup |
| `static/main.js` | **Live** dashboard logic |

> ⚠️ Root-level `index.html` and `main.js` are **orphaned copies**. Editing them does nothing. The live files are under `templates/` and `static/`.

### Research platform
Standalone **Next.js** app — 11-table Prisma schema on Neon (Postgres), `/api/ingest` endpoint, session handshake to the realtime backend. Dark-themed dashboard.

---

## Two dependency files — read this before installing

| File | What it is | Deploys? |
|---|---|---|
| `requirements-cloud.txt` | FastAPI, google-genai, opencv-headless, numpy, Pillow, dotenv | ✅ **This is what runs in production** |
| `requirements.txt` | torch, ultralytics, mediapipe, pyaudio, pygame, dashscope | ❌ Local workstation build only |

`app_main.py` wraps *most* heavy imports (torch/ultralytics/mediapipe/dashscope-path) in `try/except`, so those missing packages **disable features silently** rather than crashing. **`vision_backend.py` is the exception** — it's imported unconditionally (see the Repository layout table above) and its default backend needs Pillow, which is why `requirements-cloud.txt` pins it even though it looks like an odd inclusion in an otherwise torch-free list.

**Consequence:** a significant portion of this repo did not run during P01 or P02. Confirm a feature was actually active before debugging it.

---

## Running locally

**There is no `.env.example` in this repo** — `.env` is gitignored and no template is committed. Env vars below were read directly from `os.getenv`/`os.environ` calls across `app_main.py`, `gemini_live_client.py`, `research_exporter.py`, and `yolo_client.py`.

```bash
pip install -r requirements-cloud.txt
# required: GEMINI_API_KEY
# optional (all have defaults): AI_BACKEND, AUDIO_STALE_AFTER_MS, GEMINI_VIDEO_INTERVAL_SEC,
#   LATENCY_LOG_CSV, RESEARCH_FRAME_SAMPLE_INTERVAL_SEC, RESEARCH_SESSION_DEFAULT_TTL_SEC,
#   RESEARCH_SESSION_SHARED_SECRET, RESEARCH_SESSION_STATE_PATH, STABILITY_TEST_TOKEN,
#   THERMAL_FACT_MAX_AGE_SEC, VISION_FRAME_MAX_AGE_SEC, VISION_MIN_INTERVAL_SEC,
#   RESEARCH_INGEST_PATH, RESEARCH_PLATFORM_URL, YOLO_SERVICE_URL, YOLO_SERVICE_TOKEN,
#   DASHSCOPE_API_KEY, GENERAL_DET_CONF, STABILITY_MODE (defaults true)
python3 app_main.py
```

Dashboard: `http://localhost:8081/` (`app_main.py`'s `if __name__ == "__main__"` block hardcodes `port=8081`; `fly.toml`'s `internal_port` and the `Dockerfile`'s `EXPOSE` agree)

### Firmware
Flash `compile.ino` via Arduino IDE with the XIAO ESP32-S3 board package. WiFi credentials (`WIFI_SSID`, `WIFI_PASS`) and the backend host (`SERVER_HOST`) are hardcoded plaintext constants near the top of `compile/compile.ino` (~line 42-44) — edit them there before flashing.

Note `#define STABILITY_MODE 1` (line 24). **Correction:** this does *not* make `/api/camera` a no-op. `handle_researcher_camera_cmd()` (compile.ino ~L779) and the `/api/camera` handler (`app_main.py` ~L2386) both explicitly carve `SET:FRAMESIZE=` and `SET:FPS=` out of the STABILITY_MODE block — those two are "compiled in regardless of STABILITY_MODE" per the firmware's own comment. Only the legacy sensor-tuning fields (`quality`, `exposure_auto`, `exposure_value`, `gain_ceiling`, `aec2`, `ae_level`) are blocked (HTTP 409) under STABILITY_MODE. See `PROJECT_CONTEXT.md` §3.2 — this is the Camera Researcher Control feature, implemented but not yet flash-tested.

---

## Deployment

Fly.io — `ai-glasses-for-research.fly.dev`
Object storage: Fly Tigris (frame durability)
Database (research platform): Neon Postgres via Prisma

Single machine, no horizontal scaling. See above.

---

## Key API surface

Extracted from `@app.get/post/websocket` decorators in `app_main.py` — confirmed complete as of this audit.

**WebSocket:** `/ws/camera_thermal` · `/ws_audio` · `/ws/camera` · `/ws/viewer` · `/ws/thermal` · `/ws/thermal_viewer` · `/ws_ui` · `/ws`

**HTTP:** `/api/health` · `/api/camera-freshness` · `/api/audio-freshness` · `/latency/metrics` · `/api/perception/latest` · `/api/perception/hands/latest` · `/api/recording` · `/api/vision` · `/internal/research-session` · `/api/settings` · `/api/camera` · `/api/restart` · `/api/command` · `/api/imu-validation` · `/api/thermal-display` · `/api/yolo/detections`

**Corrected:** the original list omitted `/api/perception/hands/latest` and `/internal/research-session`.

`/api/camera` is **not** a no-op under `STABILITY_MODE` — see "Running locally" above.

---

## Current status

**Working:** end-to-end capture → AI → speech pipeline; two participant sessions completed and analyzed (P01, P02); researcher dashboard; research data export.

**Known issues — see `PROJECT_CONTEXT.md` §4 for full detail:**

| Issue | Impact |
|---|---|
| Stale grounding | Model describes old frames. **Core failure mode.** |
| Camera mirroring (`set_hmirror`) | Left/right guidance may be **inverted**. Confirmed still set to `1` (on) in `compile.ino` line 842 as of this audit — **unconfirmed fix, still live.** |
| Turn-finalization race | 163/251 "interrupted" turns were false positives |
| No depth validation | Movement guidance isn't backed by obstacle sensing. **Safety-relevant.** |
| ESP32 heap fragmentation | Device-side instability; backend fixes can't touch it |

---

## Research framing

Feasibility pilot, small n. Findings support **within-participant contrasts and rate estimates only** — not population inference. Target: CSUN Assistive Technology 2027.

Prior work: UNav (arXiv 2209.11336) · AnyLoc (arXiv 2308.00688) · LightGlue (arXiv 2306.13643) · OpenEQA (CVPR 2024) · AlanaVLM

---

## Team

| Role | Person |
|---|---|
| Advisor | Prof. Hao Tang |
| Research mentor | Prof. Jiawei Liu |
| Team lead | Shan |
| Contributors | Ian, Alicia |

Task ownership and current assignments: [`TEAM_OPS.md`](./TEAM_OPS.md)
Development log: [`DEVLOG.md`](./DEVLOG.md)
