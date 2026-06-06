---
name: project-mode
description: Project is being adapted from outdoor blind navigation to indoor cooking assistant
metadata:
  type: project
---

This repo started as an outdoor blind-navigation assistant (blind path, crosswalk, traffic light detection via YOLO) running on ESP32 glasses. The user is adapting it to an indoor cooking assistant.

**Why:** New use-case — webcam voice conversation with Qwen-Omni about what the camera sees, indoors.

**How to apply:** Keep the Qwen-Omni conversation pipeline and Whisper ASR pipeline intact. Navigation features (blind path, crosswalk, traffic light, item search via yolomedia) are disabled — all related imports are wrapped in try/except so the app starts without YOLO model files. The orchestrator/navigator objects will be None at runtime; all code already guards on `if orchestrator:`.
