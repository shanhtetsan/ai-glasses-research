# PROJECT_CONTEXT.md

**AI Smart Glasses for BLV Navigation** — BMCC × VISIONS · Advisor: Prof. Hao Tang
**Purpose of this file:** the reasoning that isn't in the code. Why things are the way they are, what we already tried, what failed and why. Read this before changing anything.

> Maintained by: Shan (team lead). Anyone can append to the Decision Log — don't rewrite history, add a new dated entry.

---

## 1. What this project is

A wearable device that helps blind and low-vision users understand and navigate their surroundings. Glasses capture camera, thermal, and audio; a cloud backend streams that to a multimodal AI model; the model speaks guidance back through the device.

Two goals, and they pull in different directions — be aware of which one you're serving:

1. **Research** — produce findings good enough to publish (CSUN Assistive Technology 2027 abstract).
2. **Product** — a device that actually works for a person walking through a room.

This is a **feasibility pilot with small n**. Our findings support within-participant contrasts and rate estimates. They do **not** support population-level claims. Don't write "users prefer X" anywhere.

---

## 2. System shape (one paragraph)

The glasses (XIAO ESP32-S3) hold two persistent WebSocket connections to the backend — one for camera+thermal, one for audio. The backend (FastAPI/Python, deployed on Fly.io) receives those streams, forwards frames and speech to the Google Gemini Live API, and pipes synthesized speech back down to the device's speaker. A browser dashboard lets a researcher watch the session live and control settings. The backend is a **stateful singleton** — it holds live sockets and session state in memory, so it **cannot be horizontally scaled**. One machine, always.

---

## 3. Decision Log

Newest first. Each entry: what we decided, why, and what it rules out.

### 3.1 Architecture: continuous streaming vs. triggered capture — **OPEN**
**Status:** actively debated, not decided.
**The question:** keep streaming video continuously, or switch to wake-word-triggered burst capture matched against a stored set of reference views?

**Why it's on the table:** stale grounding (§4.1). Continuous streaming floods the model with frames and a growing conversation history; the model ends up describing something it saw a while ago.

**What triggered-capture would mean:** on wake word, grab a short burst (3–5 frames), match it against stored reference images of the space, and generate guidance relative to an estimated location. This is **topological** spatial memory (which place am I near?), **not metric SLAM** (exact coordinates) — the hardware can't support metric SLAM.

**Also relevant:** both P01 and P02 independently asked for wake-word activation, unprompted, after noticing the always-on mic picked up ambient conversation. That's user-driven evidence pointing the same way.

**Decision rule:** don't pick based on preference. Phase 1 (§6) tests it cheaply and offline first.

---

### 3.2 Camera Researcher Control — implemented, **not yet flashed**
**Spec file not in version control.** `CAMERA_RESEARCHER_CONTROL_SPEC.md` is referenced by name in earlier notes but was never committed to this repo or found in git history — the implementation landed, the spec didn't. Context is captured inline below instead of relying on that file.

A dashboard panel to change camera resolution and stream FPS at runtime, without reflashing mid-session.

Decisions made and locked:
- Implement firmware + backend + frontend, but **do not flash, deploy, or commit** until the first flash test is scheduled.
- Carve out **only** `SET:FRAMESIZE` and `SET:FPS` from the `STABILITY_MODE` block. Everything else in the `SET:` parser stays blocked.
- **Send-pacing only** — pace what we transmit, not what the camera captures.
- Terminology is mandatory: **"Stream FPS"**. It means target transmission rate. It is not capture FPS and not actual transmitted FPS. Use all three names precisely.
- 4 FPS is the explicit baseline condition.

**Still open:**
- XGA: fall through, or retire it from the legacy parser?
- I2C/SCCB bus-sharing check (read-only): does the camera's SCCB bus share pins with the MLX90640 and IMU, and does `apply_framesize()` take the shared mutex?
  **[VERIFY partial]** `apply_framesize()` (compile.ino ~L750) does **not** take `i2cMutex`. On the XIAO ESP32-S3 board block in `camera_pins.h`, the camera's SCCB pins are `SIOD=GPIO40`/`SIOC=GPIO39`, while the thermal+IMU `Wire` bus (guarded by `i2cMutex`) runs on `IMU_I2C_SDA=D4`/`IMU_I2C_SCL=D5`. These read as physically separate buses from the pin assignments alone, which would mean the missing mutex is not currently a collision risk — but this was checked from source only, not the schematic, so treat as a lead to confirm rather than a closed answer.
  Also note: the IMU is **MPU-6050** (I2C, GY-521 breakout), not ICM42688 — `compile.ino`'s own top-of-file comment (line 1) still says "IMU (ICM42688 SPI)" but the implementation (§~2347 onward) is MPU-6050 over I2C. `ICM42688.cpp/.h` still exist in `compile/` but are not `#include`d anywhere in `compile.ino`.

---

### 3.3 Primary latency metric changed
**Old:** overall backend latency.
**New:** `speech_end_to_first_i2s_ms` — measured **on the device**, from when the user stops speaking to when audio starts playing.

**Why:** the backend number overstated the user's actual perceived wait by roughly **5×**. It was convenient to collect, not representative. Don't optimize the old number.

---

### 3.4 YOLO deployment — deferred
Excluded from the `STABILITY_MODE` cloud deployment. `requirements.txt` (torch, ultralytics, mediapipe) describes a **local workstation build**. `requirements-cloud.txt` (FastAPI, google-genai, opencv-headless, numpy, Pillow, dotenv) is what **actually deploys**.

**Consequence, and this trips people up:** a large fraction of this repo did **not run** during P01 or P02. `app_main.py` guards those imports in try/except, so missing packages disable features silently instead of crashing. Before you debug something, confirm it was actually running.

---

## 4. Known failures and their real causes

These are hard-won. Don't re-derive them.

### 4.1 Stale grounding — *the core model failure*
The model describes old frames while fresh ones are confirmed arriving. Clearest evidence: a **27-second episode describing the floor** while the camera was pointed at a wall and ceiling.

It's a **context problem, not a transport problem**. The session accumulates a long history and the model attends to a stale frame buried in it. This is what drives the architecture debate in §3.1.

### 4.2 Transport improvements ≠ experience improvements — *the false lead*
Between P01 and P02, transport metrics improved (latency, dropped audio, frame rate) and the participant's experience got **worse**. Both sessions likely ran the **same build (`a33eab8`)**, so the metric changes weren't even from a code change.

**Rule that came out of this:** system metrics and participant experience are tracked **independently**. A subsystem's numbers improving doesn't mean the user's outcome improved — the failure can just move to a layer you're not watching.

### 4.3 Safety: movement guidance without depth validation
The system gave movement guidance with **no real obstacle or depth validation**. Given that participants are BLV and walking, this is a research-integrity concern, not just a bug. Flag it in any writeup. Don't quietly ship guidance we can't back.

### 4.4 Camera mirroring
`set_hmirror` should be `0`. It has been inverting left/right — meaning spatial guidance was **backwards**. **Status: not confirmed fixed. Verify this before the next session.**

### 4.5 Turn-finalization race
**163 of 251** turns logged as "user interrupted" were false positives from a firmware race in how turns get finalized. Any analysis using interruption counts before this is fixed is wrong.

### 4.6 Code/prompt contradiction
The system prompt tells the model no fixed command phrase is needed. **Corrected:** the code (`_VISION_PHRASES` in `app_main.py`, used by `is_explicit_vision_request()`) currently binds a camera frame to a question on **nine hardcoded strings**, not seven:
`"what is in front"`, `"what's in front"`, `"describe this scene"`, `"describe the scene"`, `"read this sign"`, `"what am i holding"`, `"is there a chair"`, `"look at this"`, `"what do you see"`.
(`is_explicit_vision_request()` also has two non-string-literal paths — `is_hand_perception_request()` and `is_target_directed_request()` + `extract_requested_target()` — so the binding isn't purely a fixed-string match even beyond these nine.) These still disagree with the system prompt. Pick one and make them match.

### 4.7 Firmware is often the real instability source
Backend Python fixes cannot touch ESP32 heap fragmentation (largest free block dropping to ~572–1268 bytes), I2C mutex timeouts, or the device tearing down its own sockets (`camera-send-unhealthy`, `thermal-send-unhealthy`).

A Codex-generated firmware fix once produced hours of stable streaming — the real fix lived in `compile.ino`. An audio regression was traced to an over-aggressive reconnect backoff (250ms–5s replacing 1–30s) causing memory exhaustion on weak WiFi; confirmed by A/B on commits `454cc8e` vs `577cd29`.

**First question when something is unstable: is this a backend bug or a device bug?**

### 4.8 `SET:FPS=` was vestigial — *resolved*
**Historical, pre-Camera-Researcher-Control.** `SET:FPS=` previously wrote `g_target_fps`, which nothing read. Real pacing was governed by a fixed send interval.

**Now:** `SET:FPS=` drives actual send pacing via `g_cam_send_interval_ms` (`compile.ino` ~L1092, "read volatile once"). `handle_researcher_camera_cmd()` calls `set_target_stream_fps()`, which sets both `g_target_fps` (log-only) and `g_cam_send_interval_ms`. The baseline constant is `constexpr uint32_t CAMERA_BASELINE_SEND_INTERVAL_MS = 250` (`compile.ino` line 54; 250ms = 4 FPS, matching the "4 FPS baseline" in §3.2). The symbol `CAMERA_MIN_FRAME_INTERVAL_MS` referenced in earlier notes does not exist in the current source.

Kept as a record of what changed, not as a live issue.

### 4.9 Stale IMU comment + dead files — documentation drift already caused a wrong citation
The hardware is **MPU-6050** (I2C, GY-521 breakout) — see §2 / hardware table. But `compile.ino`'s own file-header comment (line 1) still reads `"IMU (ICM42688 SPI)"`, and `ICM42688.cpp`/`ICM42688.h` still exist under `compile/` even though nothing in `compile.ino` `#include`s them (dead files, not dead-but-reachable code). This stale comment has already caused the wrong part to be cited outside the code — worth flagging so it doesn't happen again. Not fixing here: correcting the comment or removing the dead files is a source change, out of scope for this doc pass.

### 4.10 Hardcoded WiFi + backend host credentials in committed source — **must fix before the repo is shared or made public**
`compile.ino` lines 42-44 hardcode `WIFI_SSID`, `WIFI_PASS`, and `SERVER_HOST` as plaintext constants, committed to source. This is fine for a private working repo but is a real credential leak the moment this repo (or this branch) is pushed anywhere shared, made public, or given to anyone outside the team. Flagging as a known issue, not fixing — moving these to a gitignored config or build-time secret is a source change, out of scope for this doc pass.

---

## 5. Working rules

- **Audit before writing.** Inspect the actual code before implementing or specifying anything. Multiple spec errors were caught this way (compiled-out parser, vestigial FPS variable, wrong frontend file). The live frontend is `templates/index.html` + `static/main.js` — **not** the orphaned root-level copies.
- **Never interpret logs without knowing what the person was doing during that capture window.** A "deterministic 153-second reconnection cycle" turned out to be self-inflicted manual restarts. Ask first, hypothesize second.
- **Flag ambiguity, don't silently resolve it.**
- **Don't expand scope.** Do the assigned thing.

---

## 6. Research plan (Phases)

**Phase 1 — offline Gemini localization test.** *Approved by Prof. Tang; he proposed it.* Using existing `recordings/` footage (591 files from P01/P02, in the actual test rooms), test whether Gemini can match a query image to a reference view well enough to localize. Requires no new hardware, firmware, or capture. Cheapest possible test of the whole triggered-capture idea. **Can start immediately, in parallel with engineering.**

**Phase 2 — live prototype comparison.** Add wake-word detection (ESP-SR or microWakeWord), capture 3–5 frame bursts on wake, run matching live, compare head-to-head against the continuous-streaming baseline in the same space. Metrics: guidance consistency (same question, same position, same answer?), localization accuracy, bytes per query, latency, task success.

**Phase 3 — add visual place recognition, only if 1/2 aren't accurate enough.** Embed references (DINOv2), retrieve top-K, let Gemini reason over the short list. Same architecture as UNav.

**Prior work verified:** UNav (arXiv 2209.11336), AnyLoc (arXiv 2308.00688), LightGlue (arXiv 2306.13643), OpenEQA (CVPR 2024), the Nature LLM indoor navigation paper, AlanaVLM.

---

## 7. Sessions run

| Session | Site | Date | Build |
|---|---|---|---|
| P01 | VISIONS | 8/12 | `a33eab8` (likely) |
| P02 | BMCC | 8/13 | `a33eab8` (likely) |

72 logged cases were compressed into 15 recurring patterns across four layers: Model, System, Evidence, Hardware.

---

## 8. Open questions

- [ ] Architecture: continuous vs. triggered — pending Phase 1 results
- [ ] XGA fall-through or retire?
- [ ] I2C/SCCB bus-sharing + mutex check on `apply_framesize()`
- [ ] Is `set_hmirror` actually fixed?
- [ ] Firmware build hash + SNTP time sync — needed before P03 so we can tell which build produced which data
- [ ] Resolve the seven-hardcoded-strings vs. system-prompt contradiction
