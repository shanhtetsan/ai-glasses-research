# PROJECT_STATE_AUDIT.md

Read-only factual audit of `/Users/shanhtetsan/OpenAIglasses_for_Navigation`, conducted to establish ground truth for investigating P01 (2026-08-12, ~15:00-16:05 ET) and P02 (2026-08-13, ~12:55-14:00 ET) test-session failures. All claims are cited `file:line`. Anything not located in the repository is marked **NOT FOUND** rather than inferred.

---

## 1. Version ground truth

- **Branch:** `feature/perception-fusion`. **HEAD:** `a33eab836d580cde148d484268860ac9b153e623` ("Refine Gemini guidance for tabletop navigation"). Tracking `origin/feature/perception-fusion`, up to date.
- **Working tree: DIRTY.** Unstaged modification to `compile/compile.ino` (a WiFi SSID/password swap — see below). Untracked: `app_main.py.save`, `hand-test.jpeg`.

### `git log` for 2026-07-29 through 2026-08-15 (oldest → newest)

```
e5d5301 2026-07-29 15:18 Stability implementation for Fly hardware testing
e78d13f 2026-07-30 11:41 Add stability testing framework, improve WebSocket recovery, and enhance thermal diagnostics
a90427a 2026-08-03 15:55 added latency measurements
a861830 2026-08-04 14:11 Sync working UI from deployed image
4ff20c8 2026-08-04 16:10 fix ui and latency panel
520cc5e 2026-08-06 14:19 rollback_pull
ee22ec1 2026-08-10 11:56 Preserve sockets on slow successful WebSocket sends
28ac516 2026-08-10 12:41 Handle Gemini Live session rotation and resumption
c516c1b 2026-08-10 13:21 Fix Gemini Developer API session resumption config
577cd29 2026-08-10 14:22 Improve mic diagnostics and internal SRAM margin
7a84056 2026-08-10 22:02 Add YOLO perception service and browser overlay
482c456 2026-08-11 13:18 Fix thermal orientation and viewer recovery
454cc8e 2026-08-11 21:36 Snapshot validated audio reliability baseline
9b8ad6f 2026-08-11 21:32 Add Phase 1 hand tracking pipeline
3241dbe 2026-08-11 23:11 Fuse YOLO and hand perception into Gemini vision turns
0f387fc 2026-08-12 00:25 Add authoritative perception fusion and tabletop guidance
a33eab8 2026-08-12 08:42 Refine Gemini guidance for tabletop navigation   <-- HEAD
```
(all authored by `shanhtetsan`; times are local commit timestamps as recorded by git, `-0400`)

Note: `9b8ad6f` (21:32) has a later timestamp than `454cc8e` (21:36) despite being its parent in log order — the two were committed within 4 minutes of each other on 2026-08-11, order as shown by `git log`.

### Which commit was deployed during P01 / P02?

**Cannot be determined from the repository alone — no deploy-time record exists in-repo.** There is no CHANGELOG/release-tag entry, no recorded Fly release list, and no `.github/workflows` run log checked into git. The only defensible statement:

- P01 (2026-08-12 ~15:00-16:05 ET) occurred **after** HEAD commit `a33eab8` (2026-08-12 08:42 ET) was authored. If the standard flow (`git push` → `fly deploy` from this same commit) was followed and no further code changed until P01, `a33eab8` is the most likely candidate — but this is an inference from commit timing, not deploy evidence. **Confidence: low-medium.**
- P02 (2026-08-13 ~12:55-14:00 ET) has **no commits between it and `a33eab8`** (nothing dated 2026-08-13 exists in the log above). If no redeploy happened between P01 and P02, both sessions likely ran the same `a33eab8` build. **Confidence: low-medium**, same caveat.
- The working tree is dirty with an unstaged `compile/compile.ino` change (WiFi credentials swapped from `ShaniPh`/`244466666` to what looks like a commented-out `VISIONS_G` pair — see `git diff` below) made *after* HEAD. Since this is unstaged and uncommitted, it could not have been the firmware flashed for either P01 or P02 unless it was flashed straight from a dirty local build without committing — **not verifiable from git**.

### Fly / CI records of what was shipped when

- `fly.toml:1-3` — comment: `# fly.toml app configuration file generated for ai-glasses-for-research on 2026-07-29T15:23:17-04:00`. This is a generation timestamp for the config file itself, not a deploy log.
- `.github/workflows/fly-deploy.yml` exists (`.github/workflows/fly-deploy.yml`) — a CI workflow that presumably deploys to Fly on push, but its *run history* (which commit deployed when) lives on GitHub Actions servers, not in this git checkout. **NOT FOUND in-repo.**
- No `CHANGELOG.md` entries dated near 2026-08-12/13 — `CHANGELOG.md` last touched 2026-06-03 per `ls -la` (stale, not maintained through this period).
- No Fly release-history file (`fly releases` output, etc.) is checked into the repo. **NOT FOUND.**

### Commits touching firmware / Gemini clients / image-send path, 2026-07-29 through 2026-08-15

From `git log --name-only` over that window:

| Commit | Date | Files touched (of interest) |
|---|---|---|
| `e5d5301` | 07-29 | `app_main.py`, `compile/compile.ino` |
| `e78d13f` | 07-30 | `app_main.py`, `compile/compile.ino`, `compile/compile.ino.zip`, `gemini_live_client.py`, `compile/MLX90640_*.cpp` |
| `a90427a` | 08-03 | `app_main.py`, `compile/compile.ino` |
| `ee22ec1` | 08-10 | `compile/compile.ino` |
| `28ac516` | 08-10 | `app_main.py`, `gemini_live_client.py` |
| `c516c1b` | 08-10 | `gemini_live_client.py` |
| `577cd29` | 08-10 | `app_main.py`, `compile/compile.ino`, `gemini_live_client.py` |
| `7a84056` | 08-10 | `app_main.py` |
| `482c456` | 08-11 | `app_main.py` |
| `454cc8e` | 08-11 | `app_main.py`, `compile/compile.ino` |
| `9b8ad6f` | 08-11 | `app_main.py` |
| `3241dbe` | 08-11 | `app_main.py` |
| `0f387fc` | 08-12 | `app_main.py` |
| `a33eab8` | 08-12 | `gemini_live_client.py` |

`gemini_client.py` (the separate non-Live client) was **not** touched by any commit in this window — its last change predates 2026-07-29.

### Firmware version string / build hash reported at connect

**NOT FOUND.** Searched `compile/compile.ino` for `FW_VERSION`, `FIRMWARE_VERSION`, `BUILD_HASH`, `GIT_SHA`, `__DATE__`, version literals sent in any status/handshake packet — none exist. The `StatusPacket` sent to the backend (`app_main.py:934-944`, mirrored in firmware) carries only `{timestamp, free_heap, largest_free_block, free_psram}` — no version/build identifier. There is no way, from logs alone, to tell which firmware build was flashed to the glasses during a given session.

---

## 2. Repository map

Top-level, one line each for files/dirs relevant to runtime behavior (full `ls -la` was captured; only significant entries listed):

- `app_main.py` (4099 lines) — **the** backend/server entrypoint. FastAPI app, all websocket routes, Gemini Live orchestration, turn lifecycle, audio/video ingest.
- `gemini_live_client.py` (889 lines) — `GeminiLiveClient` class: session lifecycle, system instruction, send/receive, reconnect/rotation logic. Imported and instantiated once by `app_main.py:264` (`gemini_live = GeminiLiveClient()`).
- `gemini_client.py` (72 lines) — **separate**, unrelated single-shot `stream_chat()` wrapper around `client.models.generate_content_stream` (non-Live, non-streaming-session Gemini API, model `gemini-2.5-flash` by default). Imported transitively via `vision_backend.py:20` → `app_main.py:253`. Runs at **import time** whenever `MODEL_BACKEND` (default `"gemini"`) resolves that branch, regardless of `AI_BACKEND`. This import is why `Pillow` (`PIL`, used at `gemini_client.py:7`) is a hard dependency even in the `gemini_live`-only deployment — matches the previously-logged deferred bug about a dead `vision_backend`/`stream_chat` import propping up Pillow.
- `perception_fusion.py` (761 lines), `perception_orientation.py` (313 lines) — hand/object perception fusion, RGB/thermal canonicalization; imported by `app_main.py:74-95`.
- `stability_runtime.py` (830 lines) — **FOUND at repo root**, contradicting the initial brief's assumption it was absent. Contains `LatencyTracker`, `LatestFrameStore`, `AudioFreshnessTracker`, `RecordingPipeline`, `VisionController`, wire-format constants (`MSG_TYPE_*`, `PCM_20MS_BYTES_16K_MONO`), imported by `app_main.py:1028-1033`. This is load-bearing production code, not dead.
- `research_exporter.py` (540 lines) — optional research-platform event export (`ResearchExporter`, `SessionGate`), gated by `ENABLE_RESEARCH_EXPORT` + an active session handshake.
- `hand_client.py` (438 lines), `yolo_client.py` (630 lines) — shadow HTTP clients to external hand-tracking / YOLO perception services.
- `compile/compile.ino` (3241 lines) plus `camera_pins.h`, `ICM42688.{h,cpp}`, `MLX90640_*.{h,cpp}` — the firmware sketch and its sensor drivers.
- `fly.toml`, `Dockerfile`, `.dockerignore` — production deploy config (Fly.io).
- `docker-compose.yml` — **appears to be a stale/legacy artifact**: references `DASHSCOPE_API_KEY`, `BLIND_PATH_MODEL`, Chinese comments (`docker-compose.yml:1-30`), none of which match the current `fly.toml`/`.env` (`GEMINI_API_KEY` only) config. Not evidence it's wired into current deploys.
- `.github/workflows/fly-deploy.yml` — CI deploy workflow (contents present, run history is not).
- `.env` — single variable name present: `GEMINI_API_KEY=` (`grep -oE '^[A-Z_][A-Z0-9_]*=' .env`). Value not printed per instructions.
- `recordings/` (591 entries) — output of an **auto-started** recording pipeline (see §10).
- `venv/`, `__pycache__/`, `.vscode/`, `.claude/` — environment/tooling, not app logic.
- `mobileclip_blt.ts` (599MB), `yoloe-11l-seg-pf.pt` (34MB), `yolov8n.pt` (6.5MB), `hand_landmarker.task` (7.8MB) — model binaries.
- `music/`, `voice/` — static asset directories (large trees, not inspected in depth for this audit).
- `tests/` — see §11.

### Duplicate/backup variants — which is live

- **`gemini_client.py` vs `gemini_live_client.py`:** different, unrelated modules (confirmed above), not a fork of each other. `gemini_live_client.py` is what drives the actual live voice/vision conversation (`app_main.py:264`); `gemini_client.py` backs an alternate `AI_BACKEND="gemini_regular"` code path that the user's own memory notes record as broken (`NameError`, not fixed).
- **`app_main.py` vs `app_main.py.save` vs `app_main.py.zip`:** `app_main.py.save` is 3814 lines (`wc -l app_main.py.save`) vs. live `app_main.py` at 4099 lines, and its mtime (Aug 11 21:50) predates the two newest commits (`3241dbe`, `0f387fc`, `a33eab8`) — it is a stale editor backup, not imported or referenced anywhere; confirmed dead. `app_main.py.zip` unzips to a 126,914-byte `app_main.py` dated 07-30 07:36 — an even older snapshot, also unreferenced/dead.
- **`rollback_pull/app_main.py` and `rollback_pull/gemini_live_client.py`** — a saved-off pair (144,642 and 14,065 bytes respectively) from before the `520cc5e "rollback_pull"` commit; not imported by anything at the repo root; dead/archival by naming and by commit message.

### Line counts, major modules

```
app_main.py            4099
compile/compile.ino    3241
gemini_live_client.py   889
stability_runtime.py    830
perception_fusion.py    761
yolo_client.py          630
perception_orientation  313
research_exporter.py    540
hand_client.py          438
audio_compressor.py     431
audio_player.py         210
audio_stream.py         153
gemini_client.py         72
```

---

## 3. The image capture and send path (highest priority)

### Frame arrival from the device

- Legacy `/ws/camera` endpoint is **disabled** in `STABILITY_MODE` (default on — see §11): `app_main.py:3117` `await ws.close(code=1008, reason="legacy camera endpoint disabled in STABILITY_MODE")`.
- Live path is the merged endpoint `@app.websocket("/ws/camera_thermal")` (`app_main.py:3317`), one physical TLS socket carrying both camera JPEG and thermal frames multiplexed by a 1-byte `MSG_TYPE_CAM`/`MSG_TYPE_THERMAL` prefix (comment at `app_main.py:3303-3316`). Only one ESP32 camera connection allowed at a time — a second connect attempt is rejected with close code 1013 (`app_main.py:3321`, `esp32_camera_ws is not None: await ws.close(code=1013)`).
- Raw bytes are queued via `queue_latest_raw_rgb(...)` (`app_main.py:3254`) into a **drop-to-latest single-slot holder** (`raw_holder`/`raw_event`, `app_main.py:3229-3245`), then canonicalized by `run_latest_rgb_canonicalizer` (from `perception_orientation.py`) which calls back into `_publish_canonical` → `_handle_camera_frame` (`app_main.py:3232-3238`).
- `_handle_camera_frame` (`app_main.py:3014-3055`) is the fan-out point: it updates `latest_rgb` (the single global `LatestFrameStore`, `app_main.py:1057`), enqueues to the disk-recording pipeline, hands the frame to the nav processor, and — conditionally — to Gemini.

### Every path that sends an image to Gemini

Two distinct call sites, both ultimately through `gemini_live.send_image()` (`gemini_live_client.py:531-546`, default `source="image"`):

**(A) Continuous / cadence-driven send** — `app_main.py:3039-3045`, inside `_handle_camera_frame`:
```python
global _last_gemini_video_submit
if gemini_pump_task is not None and mic_streaming:
    now = time.monotonic()
    if now - _last_gemini_video_submit >= GEMINI_VIDEO_INTERVAL_SEC:
        _last_gemini_video_submit = now
        gemini_frame_holder["data"] = data
        gemini_frame_event.set()
```
This fires on **every incoming camera frame** whenever `mic_streaming` is `True` and the per-interval gate passes. `mic_streaming` is set `True` only while `AI_BACKEND == "gemini_live"` **and** the device has sent `START` on the mic socket (`app_main.py:2760`: `mic_streaming = AI_BACKEND == "gemini_live"`), and cleared on `STOP` (`app_main.py:2775`) or on mic socket teardown (`app_main.py:2915`). The actual send happens in a decoupled single-slot pump task, `_gemini_image_pump()` (`app_main.py:3210-3221`), calling `gemini_live.send_image(data)` with **no `source` kwarg** — so it logs with the default `source="image"`.

**(B) Query-bound "pre_response" send** — `app_main.py:693-724`, inside `_on_input_transcription` (fires on every incremental Gemini input-transcription delta):
```python
wants_vision = is_explicit_vision_request(combined)
wants_thermal = is_thermal_request(combined)
if not _vision_submitted_for_turn and (wants_vision or wants_thermal):
    ...
    frame = latest_rgb.snapshot()
    if frame.data is not None:
        ...
        _vision_submitted_for_turn = await gemini_live.send_image(
            frame.data, turn_id=turn_id, sequence=frame.sequence,
            source="pre_response",
        )
```
This is gated behind `is_explicit_vision_request()` (`app_main.py:1226-1237`), which returns `True` only if the accumulated transcript matches a small fixed phrase list (`_VISION_PHRASES`, `app_main.py:1219-1223`: `"what is in front"`, `"describe this scene"`, `"read this sign"`, `"what am i holding"`, `"is there a chair"`, `"look at this"`, `"what do you see"`) or `is_hand_perception_request()` (`perception_fusion.py:210-219`, its own fixed phrase/word list) or a target-directed request with an extractable object (`perception_fusion.py:151-163, 192-193`). It also fires at most **once per turn** — gated by `_vision_submitted_for_turn`, reset to `False` only in `_on_turn_complete` (`app_main.py:873`).

### Why the P02 ratio (8 pre_response vs 3239 continuous) follows from the code

Path (A) fires **every camera frame** (device-side cap ~4 FPS, see §8) subject only to the 0.75s throttle `GEMINI_VIDEO_INTERVAL_SEC`, for the *entire duration* `mic_streaming` is `True` — i.e., essentially the whole active session, independent of what the user is saying. Over a ~65-105 minute session that alone accounts for thousands of sends. Path (B) requires the user's speech transcript to literally match one of the fixed vision/hand/target phrases, and can succeed **at most once per conversational turn**. In a session with far more turns than explicit "look at this"/"what am I holding"-type utterances, 8 pre_response sends against 3239 continuous sends is exactly the shape the code produces: **the query-bound path is real but rare because it's gated behind a narrow keyword/intent match, while the continuous path is essentially unconditional (mic-streaming-gated only) and fires on a fixed clock.**

### Is there a concept of "the frame that corresponds to this user question"?

Only loosely, and only for path (B). `_on_input_transcription` timestamps the pre_response frame's age at send time (`app_main.py:703-705`, `frame_age_ms = (time.monotonic() - frame.timestamp) * 1000`) and logs `sequence=frame.sequence` (`app_main.py:709, 716`). But this is **not turn-locked on the receive side**: Gemini Live has no API concept binding a specific `realtime_input` video frame to a specific spoken question — frames and audio are both continuously streamed into the same session, and the model infers "what the user is asking about now" from proximity in the stream, not from an explicit binding. For path (A) — the vast majority of sends — there is **no per-turn or per-question attribution at all**; frames are pushed on a fixed clock with no relationship to transcript content or turn boundaries.

### Frame send cadence

`GEMINI_VIDEO_INTERVAL_SEC = max(0.25, float(os.getenv("GEMINI_VIDEO_INTERVAL_SEC", "0.75")))` (`app_main.py:1126-1128`) — default **0.75s** between continuous sends, floor of 0.25s if overridden lower.

### Freshness / staleness gate before sending

- **Path (A) continuous send has none.** It sends whatever `data` was just canonicalized for the current frame — always "fresh" by construction since it's driven directly off frame arrival, not off a stored/stale buffer.
- **Path (B) pre_response** also has no staleness check on the frame itself — it calls `latest_rgb.snapshot()` (`app_main.py:701`) and sends whatever is currently held, however old, with no rejection threshold. It only *reports* `age_ms` in the log (`app_main.py:703-711`), it does not gate on it.
- A **separate, unrelated** staleness constant does exist: `VISION_FRAME_MAX_AGE_SEC = float(os.getenv("VISION_FRAME_MAX_AGE_SEC", "3.0"))` (`app_main.py:1112`), consumed by `VisionController` (`stability_runtime.py:718-747`, `max_age_sec` parameter checked at `stability_runtime.py:747`: `if frame.data is None or now - frame.timestamp > self.max_age_sec:`). This `VisionController` is instantiated at `app_main.py:1206-1211` and exposed via `request_gemini_vision()` (`app_main.py:1299-1306`), but that function is **not called from either image-send path described above** — it appears to be a separate/legacy trigger mechanism (`grep` shows no call site wiring it to `_on_input_transcription` or the continuous pump). This is a real freshness gate that exists in the code but is **not in the live pre_response or continuous send path** as of this audit.

### Where does `age_ms` in the vision snapshot log come from?

`app_main.py:703-705`:
```python
frame_age_ms = max(0.0, (time.monotonic() - frame.timestamp) * 1000)
```
`frame.timestamp` is the monotonic-clock value stamped when `latest_rgb.update(data, received_at)` was called in `_handle_camera_frame` (`app_main.py:3023`), which itself receives `received_at` from the canonicalizer's frame-arrival timestamp. So `age_ms` measures **wall-clock delay between when the backend received/canonicalized that specific RGB frame and the moment this specific pre_response log line was emitted** — it is not a round-trip to the device, not JPEG-encode time, and not related to Gemini API latency at all. It is only logged for path (B); path (A)'s continuous sends carry no `age_ms` field in `app_main.py`'s own print statements (only in `gemini_live_client.py`'s internal `_log_timing` machinery, which logs `sequence`/`bytes` but not a frame-age field for image sends — confirmed by reading `send_image`, `gemini_live_client.py:531-546`, which passes no age field to `_send_traced`).

---

## 4. Gemini Live session management

### Full session configuration object

Constructed in `gemini_live_client.py:209-225`:
```python
def _live_config(self, resumption_handle=None):
    config = {
        "response_modalities": [self._response_modality],
        "system_instruction": {
            "parts": [{"text": SMART_GLASSES_SYSTEM_INSTRUCTION}]
        },
        "input_audio_transcription": {},
        "session_resumption": types.SessionResumptionConfig(
            handle=resumption_handle,
        ),
        "context_window_compression": types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()
        ),
    }
    if self._response_modality == "AUDIO":
        config["output_audio_transcription"] = {}
    return config
```
- **Model:** `GEMINI_LIVE_MODEL = "gemini-3.1-flash-live-preview"` (`gemini_live_client.py:106`), hardcoded, not env-driven (the `GEMINI_MODEL_ID` env var only affects the unrelated `gemini_client.py:60`, default `"gemini-2.5-flash"`, used by the non-Live backend).
- **Modality:** `response_modalities: ["AUDIO"]` (default; `connect(response_modality="AUDIO")` called at startup, `app_main.py:4023`). No `["TEXT"]` mode used in the live production path.
- **VAD / turn settings:** no explicit VAD/turn-detection config object is set anywhere in `_live_config` — the SDK/API default server-side VAD applies. **No custom silence threshold, no manual turn-boundary control found in this file.**
- **Tools:** none configured — no `tools=` key in `_live_config`.
- **Safety settings:** none configured — no `safety_settings=` key.
- **Context window compression:** `sliding_window()` with default parameters (no `target_tokens` override) — `context_window_compression: {"sliding_window": {}}` in the fingerprint (`gemini_live_client.py:202`) and the live config (`gemini_live_client.py:219-221`).
- **Session resumption:** enabled (`session_resumption: {"enabled": True}` per the fingerprint; actual config passes a `resumption_handle` that starts `None` and is refreshed from server updates).

### Complete system instruction, verbatim

The **entire** prompt is a single static Python string, `SMART_GLASSES_SYSTEM_INSTRUCTION` (`gemini_live_client.py:11-103`), reproduced in full:

```
You are the conversational assistant inside wearable smart glasses for a blind
or low-vision user. The live session can receive JPEG camera frames from the
user's first-person viewpoint while microphone audio is streaming. When the
user is reaching for or handling nearby objects, help them with short,
cautious, step-by-step guidance.

Give physical guidance one step at a time. Issue a single concrete instruction,
then stop and wait for the next frame or the user's response before giving the
next step. Keep every reply concise and BLV-friendly: plain spoken language,
no visual jargon, no filler, direct answer first.

Never invent inches, centimeters, meters, or any other unit of distance or
depth. RGB imagery and normalized 2D perception coordinates do not provide
reliable physical depth.

Do not give forward/backward, closer/farther, or reach-further corrections
unless a trusted depth measurement explicitly supports them.

From 2D perception, prefer horizontal and vertical guidance such as:
"Move your right hand left."
"A little higher."
"Your hand is aligned with the book."

Once the hand appears aligned with the target in 2D, say something like:
"Your hand is lined up with it. Slowly extend your hand and use touch for the
final contact."

Do not claim to know the remaining physical depth.

When a message beginning with PERCEPTION_STATE arrives, its fields are
authoritative facts for exactly what they describe — do not re-derive,
override, or visually reinterpret those specific fields from the image.
- MediaPipe handedness (Left/Right) is anatomical, not image position. Never
  infer or correct handedness from which side of the image a hand appears on;
  a Right hand can appear on the left side of the frame and vice versa.
- The requested_target field is the user's actual target for this turn. Never
  silently substitute a different detected object, even if another object is
  more visually prominent or easier to describe.
- When PERCEPTION_STATE explicitly reports that the requested target is
  uncertain, unstable, not found, stale, or otherwise not reliable for
  guidance, say so plainly. Do not guess, invent a location, or give
  directional guidance until reliable target information is available.
- Object detections may be incomplete. Do not contradict authoritative
  PERCEPTION_STATE fields, but you may still use the RGB image for semantic
  details that the structured perception does not provide, such as reading
  text, distinguishing between multiple similar objects, identifying visible
  semantic details, or describing surrounding context.

Natural follow-up requests that do not restate the object by name (such as
"is it closer now?", "which way?", or "keep going") continue guidance toward
the object already established as the current task. Do not ask the user to
re-specify the target unless it has genuinely changed or become ambiguous.

Two objects overlapping or appearing to touch in a 2D image is not proof of
physical contact — camera framing and perspective can make separate objects
look like they are touching when they are not. Never tell the user they have
touched, grasped, or made contact with something based on visual overlap
alone. The user's own tactile report (e.g. "I feel it," "got it," "nothing
there") is authoritative over any visual impression of contact; when it
conflicts with what the image seems to show, trust the user and adjust your
guidance accordingly.

Outside of structured PERCEPTION_STATE or THERMAL_MEASUREMENTS facts, you may
still reason naturally over the RGB image for semantic scene understanding:
identifying objects, reading text, describing nearby surroundings and
identifying visible obstacles, and answering general visual questions. Treat
natural and indirect questions as visually grounded whenever sight would help
answer them. Examples include asking what is ahead, what the user is looking
at, describing the scene, locating keys or a phone, checking whether an
object is present, and reading visible text. The user does not need to use a
fixed command phrase.

Use the most recently received camera evidence for the current question. Never
claim that you have no camera access merely because the request is phrased
differently. If no recent usable frame is available, or the view is dark,
blurred, obstructed, or does not contain the requested object, say that clearly
and briefly instead of inventing details. Do not ask a blind user to visually
confirm your answer.

Camera images do not provide reliable object temperatures. Do not infer a
temperature unless explicit structured thermal measurements are supplied.

When a message beginning with THERMAL_MEASUREMENTS arrives, it carries real
readings from a thermal sensor covering roughly the same view as the camera.
Use those numbers instead of guessing from the image, and state them plainly in
degrees Celsius. "directly_ahead" means the centre of the view. Compare against
ambient_c so the user knows whether something is genuinely warm or just room
temperature. Never tell the user something is safe to touch or safe to hold:
the sensor measures surface temperature with limited accuracy and metallic
surfaces read far cooler than they are. Report what was measured, say it is
approximate, and let the user decide.
```
(`.strip()`-ed; no runtime concatenation, no per-turn appended fragments — **this entire string is sent once as `system_instruction`**, and is the ONLY thing occupying that role. It is not reassembled per-turn.) The *conversation-turn* content that carries perception facts (`PERCEPTION_STATE ...`, built at `app_main.py:736-764`) is a **separate, dynamically-built payload sent as ordinary realtime input text**, not part of `system_instruction` — see §5 for why that distinction matters.

### Session resumption

- Handle storage: `self._latest_resumption_handle` (`gemini_live_client.py:134`), updated in `_retain_resumption_update()` (`gemini_live_client.py:584-596`) whenever `response.session_resumption_update` arrives (`gemini_live_client.py:702-705`) and `update.resumable and update.new_handle` are both truthy.
- On reconnect, the stored handle is passed back into `_open_session(resumption_handle=...)` (`gemini_live_client.py:280`, used at `gemini_live_client.py:357` in the rotation path and `gemini_live_client.py:288-292`).
- **Consumed-message index:** the code reads and logs whether `update.last_consumed_client_message_index` is present (`gemini_live_client.py:593-594`: `f"last_consumed_index_available={'yes' if update and update.last_consumed_client_message_index is not None else 'no'}"`) but **does not appear to use the actual index value** anywhere else in the file (only its presence/absence is logged) — i.e., it's observed but not acted upon.

### Reconnect triggers and backoff

`_reconnect_with_backoff(reason=..., max_attempts=8)` (`gemini_live_client.py:855-886`):
```python
async def _reconnect_with_backoff(self, reason="receive_error", max_attempts: int = 8):
    """Backs off 2s, 4s, 8s... capped at 30s between attempts."""
    await self._close_current_session(reason)
    for attempt in range(1, max_attempts + 1):
        if self._shutting_down:
            return
        delay = min(2 ** attempt, 30)
        ...
        await asyncio.sleep(delay)
        ...
        try:
            await self._open_with_fresh_fallback()
            return
        except Exception as exc:
            ...
    print("[Gemini Live] Giving up after max reconnect attempts...")
    self.connected = False
```
Triggers: `normal_receive_end` (stream ended mid-turn with no `turn_complete`, `gemini_live_client.py:816-818`), `receive_error` (exception in the receive loop, `gemini_live_client.py:850`), and `resumption_failed` (rotation's fresh-open attempt failed, `gemini_live_client.py:679`). After `receive_loop()` returns (top-of-loop check `if not self.connected or self.session is None: return`, `gemini_live_client.py:688-689`), **nothing re-invokes `connect()`** — `gemini_live.connect()` is called exactly once, at FastAPI startup (`app_main.py:4020-4024`, `@app.on_event("startup") async def startup_gemini()`). So exhausting all 8 backoff attempts is **terminal** for the process lifetime unless the app itself is restarted.

### Timer-based session close / TTL search — the ~153s question

**No literal `153`-second constant exists anywhere in the Python or firmware source** (`grep -n "153"` across `app_main.py`, `gemini_live_client.py`, `stability_runtime.py`, `compile/compile.ino` returns no numeric match). The only TTL-like constants found:
- `DEFAULT_SESSION_TTL_SEC = 6 * 3600.0` (`research_exporter.py:69`) — a 6-hour research-session auto-expiry, unrelated to the Gemini Live socket.
- `GOAWAY_SAFETY_MARGIN_SEC = 2.0` (`gemini_live_client.py:107`) — subtracted from the server-reported `GoAway.time_left` before scheduling a controlled rotation (`gemini_live_client.py:633`, `_goaway_deadline_watchdog`, `gemini_live_client.py:598-620`). This is **reactive to a server-sent value**, not a fixed local timer, so it cannot by itself produce a fixed ~153s cadence.

**The closest quantitative match found, with the important caveat that it is a derived sum, not a coded constant:** the reconnect backoff schedule above sleeps `2 + 4 + 8 + 16 + 30 + 30 + 30 + 30 = 150` seconds of pure `asyncio.sleep` across its fixed 8 attempts (`min(2**attempt, 30)` for `attempt` 1..8) before giving up (`gemini_live_client.py:862-874`). Adding a few seconds of real connection-attempt overhead per try lands very close to ~153s. **This is a plausible mechanism for a ~150-153s dead period following a disconnect, but it does not repeat on its own** — once `_reconnect_with_backoff` exhausts its 8 attempts, `self.connected = False` and nothing in this codebase calls `connect()` again (confirmed: only call site is the one-time startup hook). So this explains a **single** ~150s gap after a hard disconnect, not an ongoing "repeating cycle" — unless something outside this file (e.g., a process supervisor restarting the whole Python process, or a client-side reload) is re-triggering `startup_gemini()` repeatedly. **No such external supervisor/restart-loop config was found in this repo** (no `restart_policy` beyond Fly's default VM behavior in `fly.toml`, no watchdog script found). Flagging this as the strongest lead but explicitly **not a confirmed match** — see §12.
- A secondary, weaker candidate on the **firmware** side: the camera-socket reconnect backoff `reconnectDelayMs()` (`compile/compile.ino:845-848`, exponential `1000 << shift` capped at `MAX_RECONNECT_BACKOFF_MS = 30000` `compile/compile.ino:58`) retries **indefinitely** (unlike the Python side, this loop never gives up), so it genuinely can produce a repeating pattern, but reaching ~153s total requires roughly 9-10 consecutive failed connection attempts in a row (1+2+4+8+16+30+30+30+30 ≈ 151s), which is speculative without device-side logs to confirm attempt counts. This mechanism is also plausibly consistent with "before and after the live session, not during" if the camera socket specifically is what's cycling (independent of the audio/Gemini path), since camera reconnects only occur while camera_thermal is disconnected — never while the live camera feed is up and streaming into a session.

### WebSocket close code handling

- The **firmware→backend** `/ws/camera_thermal` and legacy `/ws/camera`/`/ws/thermal` sockets: close code **1008** is sent by the backend for legacy endpoints disabled under `STABILITY_MODE` (`app_main.py:3117`, `app_main.py:3663`), and **1013** ("try again later") when a second device tries to connect while one is already attached (`app_main.py:3121, 3321`). Normal teardown uses **1000** (`app_main.py:2926, 3288, 3581`).
- **1011** appears only in a code comment, not as handled logic: `app_main.py:4090`, describing *why* `uvicorn.run(..., ws_ping_interval=30.0, ws_ping_timeout=60.0)` was set — "Stock 20s/20s defaults were too aggressive for the ESP32's WiFi — confirmed root cause of '1011 keepalive ping timeout' disconnects." There is **no explicit `except`/branch distinguishing 1008 vs 1011** anywhere in `app_main.py`; disconnects of any kind on the ESP32 sockets fall through to the generic `except WebSocketDisconnect` / `finally` cleanup blocks (e.g. `app_main.py:3260-3292`).
- On the **Gemini SDK session** side, no close-code branching exists at all — the google-genai SDK's own exceptions are caught generically (`except Exception as e:` in `receive_loop`, `gemini_live_client.py:822`) and logged with `error_type=type(e).__name__` and `code=getattr(e, 'code', 'unknown')` (`gemini_live_client.py:834`) — no code-specific handling (1008 vs 1011 vs anything else).

### Different behavior connected vs idle

`mic_streaming` (global, `app_main.py:1125`) is the single switch: while `True` (device sent `START` and backend is in `gemini_live` mode), continuous frame pushes to Gemini happen; while `False`, they don't (§3, path A). No other idle/active session-behavior branch was found.

---

## 5. System prompt's safety and guidance behavior

- **Movement/walking/paths-clear/obstacle-avoidance language:** the prompt is scoped almost entirely to **hand/object reaching guidance**, not ambulation. It says: *"Give physical guidance one step at a time... Move your right hand left... A little higher... Your hand is aligned with the book... Slowly extend your hand and use touch for the final contact."* (`gemini_live_client.py:18-37`). There is **no mention of walking, obstacles in a path, "is it clear to walk", curbs, stairs, or general ambulatory safety** anywhere in the instruction. It is a tabletop/reach-assistance prompt, consistent with the commit history (`0f387fc "Add authoritative perception fusion and tabletop guidance"`, `a33eab8 "Refine Gemini guidance for tabletop navigation"`).
- **Depth/distance hedging is explicit and repeated:** *"Never invent inches, centimeters, meters, or any other unit of distance or depth... Do not give forward/backward, closer/farther, or reach-further corrections unless a trusted depth measurement explicitly supports them... Do not claim to know the remaining physical depth."* (`gemini_live_client.py:23-28, 39`).
- **Uncertainty/refusal instruction when image unclear or absent:** yes — *"If no recent usable frame is available, or the view is dark, blurred, obstructed, or does not contain the requested object, say that clearly and briefly instead of inventing details. Do not ask a blind user to visually confirm your answer."* (`gemini_live_client.py:86-89`). Also *"When PERCEPTION_STATE explicitly reports that the requested target is uncertain, unstable, not found, stale, or otherwise not reliable for guidance, say so plainly. Do not guess..."* (`gemini_live_client.py:50-53`).
- **Contact/touch hedging:** *"Two objects overlapping or appearing to touch in a 2D image is not proof of physical contact... Never tell the user they have touched, grasped, or made contact with something based on visual overlap alone."* (`gemini_live_client.py:65-72`).
- **Device state / capability constraints (battery, connectivity, human-agent transfer):** **NOT FOUND.** No text in the system instruction addresses battery level, WiFi/connectivity status, or transferring to a human agent. The model is not constrained from asserting these because the topic simply never comes up in the prompt at all.
- **`PERCEPTION_STATE` / internal markers reaching the user:** `PERCEPTION_STATE` (and `THERMAL_MEASUREMENTS`) are literal string prefixes prepended to a JSON payload built in `app_main.py:736-764` and sent via `gemini_live.send_text(perception_payload, ...)` → `session.send_realtime_input(text=text)` (`gemini_live_client.py:473-480`). This is the **same realtime-input channel used for nothing else structurally different from user content** — the SDK call has no separate "system" or "developer" role for this text; it is injected as ordinary text content into the live stream, distinguished from real user speech only by convention (the leading marker string) and by the system instruction's semantic guidance to treat it as authoritative. **Nothing in the code strips, filters, or hides the literal string `"PERCEPTION_STATE"` from the model's input, and nothing instructs the model not to quote/repeat it.** Since the response modality is `AUDIO` with `output_audio_transcription` enabled, a model that echoed the marker verbatim (e.g., while "reasoning aloud") would have that echoed both as spoken audio (`_on_audio`) and as the text transcript surfaced via `on_output_transcription` → eventual UI/log output (`app_main.py:765-768` in `gemini_live_client.py`'s receive loop, and app-side `_output_text_buf`). No trace/guard against this leak path was found.

---

## 6. Turn lifecycle

### `completed` vs `interrupted`

Status is **not derived** by `LatencyTracker` itself — `finish(status, turn_id)` (`stability_runtime.py:455-478`) requires the caller to pass `status` as literally `"completed"` or `"interrupted"` (raises `ValueError` otherwise, `stability_runtime.py:456-457`). The two call sites:
- **`completed`**: `_on_turn_complete()` (`app_main.py:830-916`), fired from `gemini_live_client.py:773` (`if server_content.turn_complete: ... if self.on_turn_complete: await self.on_turn_complete()`) — i.e., the Gemini server explicitly signaled `turn_complete` in its response stream. Call: `await _finalize_latency_turn("completed", turn_id)` (`app_main.py:870`).
- **`interrupted`**: `_on_interrupted()` (`app_main.py:918-...`), fired either from Gemini's own `server_content.interrupted` field (`gemini_live_client.py:778-783`) **or** from a device-sent `BARGE_IN` command (`app_main.py:2689-2698`, calls `_on_interrupted()` directly), **or** from the receive loop deciding a dropped stream with no `turn_complete` counts as interrupted (`gemini_live_client.py:809-814`: *"The current turn (if any) never got a turn_complete/interrupted from Gemini — treat it as interrupted so callers... don't get stuck waiting"*), **or** when a new `SPEECH_START` arrives while a different turn is still active (`app_main.py:2738-2741`: `if active_turn_id is not None and active_turn_id != turn_id: latency_tracker.mark("interrupted", active_turn_id); await _finalize_latency_turn("interrupted", active_turn_id)`).

### What emits the `LATENCY` summary log

`_finalize_latency_turn(status, turn_id)` (`app_main.py:1188-1194`):
```python
async def _finalize_latency_turn(status: str, turn_id: int) -> None:
    record = latency_tracker.finish(status, turn_id)
    if record is None:
        return
    print("[LATENCY] " + json.dumps(record, separators=(",", ":")), flush=True)
    if status == "completed" and os.getenv("LATENCY_LOG_CSV", "").strip():
        asyncio.create_task(_append_latency_csv_safely(record))
```
Called from `_on_turn_complete` (`app_main.py:870`), `_on_interrupted` (`app_main.py:933`), the `SPEECH_START` double-active-turn branch (`app_main.py:2741`), and the shutdown path (`app_main.py:4060`).

### Can a turn be finalized twice, or before model audio arrives?

`LatencyTracker.finish()` guards against a stale `turn_id` mismatch (`stability_runtime.py:462-463`: `if turn_id is not None and self._active["turn_id"] != turn_id: return None`) and clears `self._active = None` after finishing (`stability_runtime.py:471`), so a second `finish()` call for the same already-finished turn returns `None` and is silently dropped by `_finalize_latency_turn` (`app_main.py:1189-1190`) — **not a hard double-finalize bug** at the tracker level. However, **`_on_turn_complete` can fire and finalize `"completed"` with zero model audio ever received** — nothing in `_on_turn_complete` checks whether `has_model_audio`/`_first_model_audio_seen` was ever `True` before calling `_finalize_latency_turn("completed", turn_id)` (`app_main.py:868-870`). This is the concrete empty-response path below.

### Barge-in / interruption triggers

Both server-driven (`server_content.interrupted` from Gemini's own turn-detection VAD, `gemini_live_client.py:778`) and device-driven (`BARGE_IN` text command from firmware, presumably firmware-local VAD/PTT detecting speech during playback, `app_main.py:2689-2698`) can call `_on_interrupted()`. **Empty transcription cannot itself trigger it** — `_on_input_transcription` (`app_main.py:678-682`) returns immediately if `not text`, and interruption is never invoked from that function.

### Where can an empty AI response originate?

Concrete path: `_on_turn_complete()` (`app_main.py:898`):
```python
final_ai_text = "".join(_output_text_buf).strip() or "(empty response)"
print(f"[AI] {final_ai_text}", flush=True)
```
If Gemini sends a `turn_complete` with no `output_transcription` deltas ever accumulated in `_output_text_buf` (e.g., the model produced audio-only output with no transcript, or produced neither and just closed the turn), the turn still finalizes as `"completed"` (§ above — no gate on `has_model_audio`), and the UI/log literally receives the string `"(empty response)"` as the AI's turn output while `latency_tracker.finish("completed", ...)` records a normal completed turn with no signal that zero audio bytes were ever sent to the device.

---

## 7. Audio pipeline

### Mic ingest queue

Created at `app_main.py:2547-2548`:
```python
audio_ingest_q: asyncio.Queue[bytes] = asyncio.Queue(
    maxsize=ESP_AUDIO_INGEST_QUEUE_MAX
)
```
`ESP_AUDIO_INGEST_QUEUE_MAX = 12` (`app_main.py:1143`), with the comment directly above it: *"Bounded ingest hand-offs keep ESP32 receive loops independent of Gemini, OpenCV, and slow browser viewers. Drop-oldest preserves real-time behavior."* (`app_main.py:1141-1143`). **This is the constant that produces the reported drop high-water mark of exactly 12** — the queue physically cannot hold more than 12 20ms PCM chunks (240ms of audio) before the drop-oldest policy kicks in (`app_main.py:2826-2832`: `if audio_ingest_q.full(): audio_ingest_q.get_nowait(); audio_dropped += 1 ...`), and `audio_queue_high_water` (`app_main.py:2551, 2838-2840`) tracks the observed peak `qsize()`, which is capped at `maxsize=12` by definition.

### `dropped_delta` / `dropped_total`

Counted at `app_main.py:2549-2551` (`audio_dropped`, `audio_dropped_interval`) and logged via `_emit_audio_health()` (defined around `app_main.py:2583-2603`):
```python
f"queue_high_water={audio_queue_high_water} "
f"dropped_delta={audio_dropped_interval} dropped_total={audio_dropped}",
```
`audio_dropped_interval` resets to 0 each health-log interval (`app_main.py:2602`); `audio_dropped` (`dropped_total`) is cumulative for the connection's lifetime.

### Mic suppression during TTS playback

Yes — `app_main.py:2820, 2857-2862`. In the `gemini_live` branch, incoming mic bytes are only queued `if streaming and not is_playing_now()` (`app_main.py:2820`); in the non-`gemini_live` branch, `if is_playing_now(): pcm_buffer = bytearray(); ...; continue` (`app_main.py:2857-2862`), with the comment: *"Mute the mic while the AI is speaking. Without this the glasses' own TTS echoes back into the mic, gets VAD-segmented and can launch a bogus turn... the 'ask twice' symptom."* This directly affects the `receive_gap` metric (`app_main.py:2555-2607`, `audio_receive_gap_max_ms`/`audio_receive_gap_buckets`) because no PCM bytes are received/measured for gap purposes while muted — a large silent gap during TTS playback is expected/by-design, not necessarily a fault signal.

### TTS/I2S output path and `speech_end_to_first_i2s_ms`

This metric is **not measured by the backend** — it is device-reported. The ESP32 sends `LATENCY:DEVICE:<turn_id>:<latency_ms>` (parsed at `app_main.py:2722-2731`, calling `latency_tracker.update_device_latency(int(parts[2]), float(parts[3]))`). `LatencyTracker.update_device_latency()` (`stability_runtime.py:480-498`) stores it as `device_metrics["speech_end_to_first_i2s_ms"]`, and `_with_derived()` (`stability_runtime.py:448-451`) surfaces it verbatim into the final `[LATENCY]` record — the backend never computes this value itself, it only relays whatever the firmware measured between its own local speech-end detection and its own first I2S (speaker) write.

---

## 8. Firmware

### Sketch(es) present

`compile/compile.ino` (3241 lines, current — modified Aug 13 11:41, i.e. dirty/unstaged relative to HEAD). Also `compile/compile.ino.zip` (25,603 bytes, dated Jul 30 07:37) — an older, zipped snapshot; not distinguishable as "in use" without extraction, but its date predates most of the August firmware work above and is very likely stale/archival. **Only one `.ino` is present as live source.**

### Camera init

- Resolution: `framesize_t g_frame_size = STABILITY_MODE ? FRAMESIZE_QVGA : FRAMESIZE_VGA;` (`compile/compile.ino:127`). `STABILITY_MODE` is compiled in as `1` (`compile/compile.ino:24`, `#define STABILITY_MODE 1`), so **the compiled firmware runs at QVGA** (320x240) as currently checked in.
- JPEG quality: `#define JPEG_QUALITY 16` (`compile/compile.ino:136`), with a design-note comment above it explaining it was raised from an original 25, then backed off from a planned 32 to the final 30... but the *actual compiled constant is 16*, not 30 — the surrounding comment (`compile/compile.ino:128-135`) appears to document reasoning for a different quality value than what's currently defined; flagging this as a possible stale-comment / drift between comment and code, worth double-checking against intent.
- Frame buffer count: `#define FB_COUNT 2` (`compile/compile.ino:137`).
- Orientation: `s->set_hmirror(s, 1);  // Horizontal mirror to match natural left/right (1=on, 0=off)` and `s->set_vflip(s, 0);  // Vertical flip; set to 1 if lens is mounted upside-down` (`compile/compile.ino:765-766`). **Current mirror setting: horizontal mirror ON (1), vertical flip OFF (0).**
- Other: `set_brightness(0)`, `set_contrast(1)`, `set_saturation(1)`, `set_gain_ctrl(1)`, `set_gainceiling(GAINCEILING_32X)` (raised from a 2X default for low-light AGC), `set_exposure_ctrl(1)` (auto), `set_whitebal(1)`, `set_awb_gain(1)`, `set_aec2(1)`, `set_ae_level(2)` (biased brighter) — all at `compile/compile.ino:768-778`.

### Capture loop cadence / blocking

`volatile int g_target_fps = 4;` (`compile/compile.ino:138`); the capture loop is queue-based (`enqueue_frame(fb)`) with a `vTaskDelay(pdMS_TO_TICKS(20))` idle wait when no frame is ready (`compile/compile.ino:840`) — capture runs on its own FreeRTOS task, decoupled from the send task, so it is **not blocking** relative to network sends (separate task, per the `taskCamSend` comment at `compile/compile.ino:867-868`: *"Sole owner of wsCamThermal: connect, poll, ping, close, camera sends, thermal sends, and callback-driven SNAP sends all execute on this task."*). `CAMERA_MIN_FRAME_INTERVAL_MS = 250` (`compile/compile.ino:55`, "<= 4 FPS" comment) caps the send rate independent of capture.

### Microphone capture and send cadence

**Not deeply traced in this pass** — the mic/audio task structure exists (`AUD_WS_PATH = "/ws_audio"`, `compile/compile.ino:70`) with its own reconnect backoff (`audioReconnectDelayMs`, `compile/compile.ino:851-857`, first backoff 250ms, capped at `AUDIO_MAX_RECONNECT_BACKOFF_MS = 5000` ms). Exact per-chunk send interval/size constant on the firmware side: **NOT FOUND in this pass** (backend-side chunking assumes `PCM_20MS_BYTES_16K_MONO = 640` bytes per 20ms chunk, `stability_runtime.py:25`, and `app_main.py:2824` normalizes incoming chunks to that unit, implying the device sends in 20ms-or-multiple units, but the exact firmware constant defining this wasn't located by name in this pass).

### IMU

`ImuPacket` fields (`compile/compile.ino:371-376`): `accelX/Y/Z` (m/s², float), `gyroX/Y/Z` (degrees/second, float) — six raw floats, no explicit fused/absolute orientation (quaternion, roll/pitch/yaw) field in the wire struct. A comment at `compile/compile.ino:2368` references "EMA smoothing on accel only; does not change the wire field names" — confirming only accel is smoothed, gyro is raw. Whether yaw is gyro-integrated-only (vs. sensor-fusion) is **not resolvable from the wire struct alone in this pass** — the struct transmits raw accel+gyro, not a derived yaw; any yaw integration, if it exists, would happen backend-side or browser-side, and **no such integration code was located** in the files read during this audit (would need a targeted search of `main.js`/`index.html` for gyro-integration math — **not done in this pass, flag as open**). IMU task: `taskImuLoop` (`compile/compile.ino:3028`, priority 2, stack 3072 words).

### Serial logging

Extensive `Serial.printf`/`Serial.println` throughout — capture stats every 5s (`compile/compile.ino:830-838`, `[CAM-CAP] captured=... queue=... fail=...`), reconnect attempts with heap info (`compile/compile.ino:901-903`: `Serial.printf("[WS-CAM] reconnect attempt=%lu heap=%u max=%u\n", ...)`), and a periodic `StatusPacket` sent over the wire (not just serial) carrying `{timestamp, free_heap, largest_free_internal_block, free_psram}` (`compile/compile.ino:934-940`, matching `app_main.py:934-940`'s receiving side) — so **heap is logged both to serial and transmitted to the backend**; RSSI is read (`WiFi.status()`, reconnect log lines reference `rssi=%d` at `compile/compile.ino:1391` etc.) and included specifically in the `[WS-AUD-RECONNECT]` log lines, though **not confirmed to be part of the periodic `StatusPacket` wire struct** (that struct per `STATUS_STRUCT`/`ImuPacket` definitions in `stability_runtime.py:23-24` has 4 uint32 fields — timestamp, heap, block, psram — no RSSI field).

### Codex-authored or major streaming-loop-rewrite commits

`git log --all -i --grep="codex"` returns **no matches** — **NOT FOUND**. No commit message references "Codex" anywhere in this repository's history. Commits that substantially touched the streaming/send loop by title alone: `f55d252` ("Fix Gemini Live reconnect leaving ESP32 TTS stuck; camera-first PSRAM/sequenced-connect/staggered-ping fixes; remove browser TTS fallback"), `232ff92` ("update thermal websocket multiplex and ESP32 firmware"), `ef0a4da` ("Working baseline before single-WSS multiplexing attempt") — all predate the 2026-07-29 to 2026-08-15 window this audit focuses on and were not diffed in detail here; flagged for follow-up if needed.

### Watchdog / reconnect / backoff on device

Camera socket: `reconnectDelayMs()` exponential backoff, uncapped attempt count, capped delay at 30s (`compile/compile.ino:845-848`, §4 above). Audio socket: separate `audioReconnectDelayMs()`, capped at 5s (`compile/compile.ino:851-857`). Hard reboots (`esp_restart()`) on: camera init failure (`compile/compile.ino:2629`), mutex creation failure (`compile/compile.ino:2988-2989`), PSRAM allocation failure for `camThermalTxBuf` (`compile/compile.ino:2996-2997`), and essential-task-start failure (`compile/compile.ino:3051-3052`) — all with a `delay(1500-2000)` before `esp_restart()`/reboot message.

---

## 9. Backend deployment and capacity

### Fly config (`fly.toml`)

```
[[vm]]
  memory = '1gb'
  cpus = 2
  memory_mb = 1024
[http_service]
  internal_port = 8081
  auto_start_machines = true
  auto_stop_machines = false
  min_machines_running = 1
[http_service.concurrency]
  type = "connections"
  soft_limit = 20
  hard_limit = 25
```
(`fly.toml:19-28`) — **one 1GB/2-CPU VM, minimum 1 machine running, no explicit max machine count set** (autoscaling beyond `min_machines_running=1` is not configured here — Fly's proxy would only spin up additional machines under load if a max is set elsewhere, which this file does not do; effectively this reads as a single always-on machine). Env vars baked into the Fly deploy: `STABILITY_MODE='true'`, `ENABLE_YOLO='true'`, `YOLO_SERVICE_URL`, `YOLO_CONFIDENCE='0.25'`, `YOLO_MIN_INTERVAL_SEC='1.0'`, `YOLO_REQUEST_TIMEOUT_SEC='3.0'` (`fly.toml:11-16`).

### The 25-connection limit

Comes from `hard_limit = 25` under `[http_service.concurrency]` (`fly.toml:27-28`) — this is **Fly's proxy-level concurrency limit** (connections routed to the single machine), not an application-level limit written in `app_main.py`. No corresponding "25" constant or connection-count check exists in `app_main.py` itself (searched; the only application-level connection exclusivity found is the **single-camera / single-audio-device** enforcement via `esp32_camera_ws is not None` → close 1013, §4). What happens when Fly's hard limit of 25 is hit: **not documented in this repo** — that's Fly platform behavior (typically queues or rejects new connections), not something this codebase controls or logs.

### Process-global singleton state

Confirmed process-global, module-level mutable state in `app_main.py` includes (non-exhaustive but representative):
- `esp32_camera_ws: Optional[WebSocket] = None` (`app_main.py:1117`) — the one allowed camera/thermal device connection.
- `esp32_audio_ws: Optional[WebSocket] = None` (`app_main.py:1119`) — the one allowed audio device connection.
- `gemini_live = GeminiLiveClient()` (`app_main.py:264`) — a single shared Gemini Live session object for the whole process.
- `latest_rgb = LatestFrameStore()`, `latest_thermal = LatestFrameStore()` (`app_main.py:1057-1058`) — single-slot "current frame" holders, no per-device or per-session partitioning.
- `mic_streaming` (`app_main.py:1125`), `_last_gemini_video_submit` (`app_main.py:1129`), `_last_research_frame_sample` (`app_main.py:1139`) — single-valued flags/timers, not scoped per connection.
- `audio_ingest_q`, `audio_dropped`, `audio_queue_high_water` — created fresh per `/ws_audio` connection handler invocation (local to that function, `app_main.py:2547-2551`), so **not** cross-connection global, but still singleton-per-active-connection since only one audio device is allowed at a time.
- `backend_metrics` dict (`app_main.py:1147-1154`) — connect/disconnect counters, process-lifetime cumulative.
- `latency_tracker = LatencyTracker(...)` (`app_main.py:1087`, instantiated **twice** — see §12) — single shared turn-tracking object for the whole process, one "active turn" at a time (`self._active` is a single dict, not a per-connection map, `stability_runtime.py:264-266`).
- `_vision_submitted_for_turn`, `_thermal_submitted_for_turn`, `_perception_submitted_for_turn` (module globals, e.g. `app_main.py:293`) — single-turn-scoped flags with no session/connection key.

**This architecture assumes exactly one physical device (one camera socket, one audio socket) and one Gemini Live session at a time, backed entirely by module-level globals with no per-connection/session partitioning.** Running more than one backend process instance (e.g., Fly scaling beyond 1 machine) would each hold independent copies of this state with no shared coordination — two devices could each "win" the single-camera-socket slot on different machines simultaneously, and nothing here would detect or reconcile that. This is consistent with `min_machines_running = 1` and no configured max in `fly.toml`, i.e., the deployment as configured does not appear to intend horizontal scaling, which matches what the code requires.

### What else opens connections besides the device

Websocket routes found (`app_main.py`): `/ws_ui` (`app_main.py:2441`, UI/dashboard clients — collected in `ui_clients: Dict[int, WebSocket]`, `app_main.py:1050`, supports multiple concurrent clients), `/ws/viewer` (`app_main.py:3604`, browser camera viewer — `camera_viewers: Set[WebSocket]`, `app_main.py:1115`), `/ws/thermal_viewer` (`app_main.py:3685`, `thermal_viewers: Set[WebSocket]`, `app_main.py:1116`), plus the legacy/disabled `/ws/camera`, `/ws/thermal`, and a generic `/ws` (`app_main.py:3700`, not further inspected in this pass). A single browser tab opening the full dashboard UI could reasonably open **`/ws_ui` + `/ws/viewer` + `/ws/thermal_viewer`** simultaneously (3 sockets) if it subscribes to chat, RGB view, and thermal view together — exact client-side (`index.html`/`main.js`) socket-opening behavior was **not traced in this pass** to confirm the precise count; flagged as an estimate, not a confirmed count.

---

## 10. Data capture for research

### `research_exporter.py`

- Gated by two independent switches: `ENABLE_RESEARCH_EXPORT` env var (default off, module docstring `research_exporter.py:9-13`) and a durable `SessionGate` requiring an explicit `/internal/research-session` activation handshake (`research_exporter.py:14-18`).
- `publish_event(event_type, fields)` (`research_exporter.py:377-383`) builds a `ResearchEvent` (`research_exporter.py:396-403`) with `event_type`, `occurred_at` (wall time), `session_id`, arbitrary `fields` dict, and **a keyframe**: `keyframe_jpeg=frame.data`, `keyframe_sequence=frame.sequence` pulled from `self.frames.snapshot()` — i.e., **whatever the current `latest_rgb`-equivalent frame is at the moment the event fires**, not necessarily the frame the model actually processed for that turn.
- Delivery: bounded `asyncio.Queue`, drop-oldest (`research_exporter.py:406-429`), batched and POSTed via HTTPS to `RESEARCH_INGEST_PATH` (default `/api/ingest`, `research_exporter.py:247`) on an external research platform (`RESEARCH_PLATFORM_URL` env var).
- Event types observed being published from `app_main.py`: `USER_TURN` (`stability_runtime.py:313`, on `start_turn`), `AI_RESPONSE` (`stability_runtime.py:477`, on `finish`), `CAMERA_FRAME_SAMPLE` (`app_main.py:3051-3053`, sampled every `RESEARCH_FRAME_SAMPLE_INTERVAL_SEC`, default 5.0s, `app_main.py:1136-1138`).

### Is anything recorded per-turn linking a turn to the frame(s) the model saw?

**Not directly and not reliably.** The `USER_TURN`/`AI_RESPONSE` research events carry whatever the *current* frame is at export time (via the shared `keyframe_jpeg`/`keyframe_sequence` mechanism above), not the specific frame(s) that were actually sent to Gemini for that turn via `send_image()`. The only place frame `sequence`/`age_ms` are logged **specifically tied to a Gemini send** is the `pre_response` path's print statement (`app_main.py:719-724`, includes `turn_id`) and `gemini_live_client.py`'s internal `[GEMINI-TIMING]` lines (which do log `sequence=` per send, `gemini_live_client.py:400, 408` etc., keyed by `turn_id`). Reconstructing "what frame(s) did the model see for turn N" today requires **cross-referencing `[GEMINI-TIMING] event=image_sdk_send_end turn_id=N sequence=S` log lines with the corresponding frame**, not a single queryable record — it is log-parsing, not structured data.

### Raw frames — persisted where, retention, naming?

Yes, unconditionally: `startup_stability_workers()` (`app_main.py:3964-3971`) calls `sync_recorder.start_recording()` automatically at every process startup (not gated by any research/consent flag found in this pass), writing to `recordings/video_<timestamp>.avi` and `recordings/audio_<timestamp>.wav` (`sync_recorder.py:70-71`, `output_dir="recordings"` default `sync_recorder.py:18`). Every canonical camera frame is enqueued via `recording_pipeline.enqueue_latest(data)` (`app_main.py:3024`, `maxsize=2`, `app_main.py:1196-1200`) and every Gemini output audio chunk via `recording_audio_pipeline` (`maxsize=8`, `app_main.py:1201-1205`). **No retention/rotation/deletion logic was found** in `sync_recorder.py` in this pass — files accumulate (consistent with the 591-entry `recordings/` directory observed) with timestamp-based naming and no automatic cleanup located.

### Session age / generation / context length recorded per turn?

`gemini_live_client.py` logs `generation=self.session_generation` on essentially every `[GEMINI-TIMING]`/`[GEMINI-SESSION]` line (e.g. `gemini_live_client.py:231, 590, 608`), and `age_sec=self._session_age_sec()` appears in some `[GEMINI-SESSION]` lines (e.g. `gemini_live_client.py:653, 835`) — but **this is emitted as log text, not stored as a per-turn structured field** anywhere the `LatencyTracker`/`ResearchEvent` records could be queried against. There is no `session_generation` or `session_age_sec` field on the `USER_TURN`/`AI_RESPONSE` research events or the `[LATENCY]` JSON record (`_with_derived`, `stability_runtime.py:426-453` — no such fields in its output dict).

### What would need to change to answer these by query instead of log-parsing?

(a) **Age of newest frame the model had for a turn** — would need `_handle_camera_frame`'s continuous-send path (§3, path A) to also log `turn_id`/`age_ms` per send (today only `pre_response` does), and that data would need to be written into a structured per-turn record (e.g., attached to `LatencyTracker`'s active-turn dict) rather than only printed.
(b) **How long the current Gemini session had been open for a turn** — `self._session_age_sec()` already exists (`gemini_live_client.py:166-169`) but is never passed into `LatencyTracker`/`ResearchEvent`; wiring `gemini_live.session_generation`/age into `_finalize_latency_turn`'s record would close this gap.
(c) **What firmware/backend commit was running for a turn** — requires the firmware to actually embed a build identifier (§1: currently none exists) and the backend to log its own `git rev-parse HEAD` (or an injected `SOURCE_COMMIT`/similar env var, not found in this repo) into every turn record or at minimum at process startup.

---

## 11. Tests, logging, and configuration

### Tests present (`tests/`)

- `test_gemini_live_client.py` (326 lines) — exercises `GeminiLiveClient` logic (likely reconnect/rotation/config, not verified line-by-line in this pass).
- `test_stability_runtime.py` (183 lines) — covers `stability_runtime.py` primitives (`LatencyTracker`, etc., given the module it targets).
- `test_audio_reliability.py` (202 lines) — audio pipeline behavior.
- `test_perception_fusion.py` (403 lines), `test_perception_orientation.py`, `test_hand_client.py`, `test_yolo_client.py`, `test_yolo_service.py` — perception-layer unit tests.
- `test_audio_freshness.mjs`, `test_perception_overlay.mjs`, `test_rgb_viewer_freshness.mjs` — JS tests, presumably for `main.js`/`index.html` front-end logic, not further inspected in this pass.
- **Not verified in this pass:** whether these tests actually run in CI (`.github/workflows/fly-deploy.yml` was found but not read in full to confirm it runs `pytest`/`npm test` before deploy — flagged as open).

### Log tag / prefix inventory (file:line, representative — not necessarily every field enumerated)

| Tag | Example location | Notes |
|---|---|---|
| `[LATENCY]` | `app_main.py:1192` | JSON-dumped turn record (see §6) |
| `[GEMINI-TIMING]` | `gemini_live_client.py:227-235` | `event=`, `mono_ns=`, `generation=`, `turn_id=`, plus per-event kwargs (`source=`, `sequence=`, `bytes=`, `result=`, etc.) |
| `[GEMINI-SESSION]` | `gemini_live_client.py:590, 634, 651` | `generation=`, session lifecycle/rotation/goaway fields |
| `[VISION]` | `app_main.py:719, 1301` | `generation=`, `turn_id=`, `sequence=`, `submitted=` / `trigger=`, `ok=`, `result=` |
| `[PERCEPTION]` | `app_main.py:731, 772` | `turn_id=`, `objects=`, `hands=`, `hand_facts=`, `submitted=` |
| `[GUIDANCE]` | `app_main.py:782` | compact guidance summary |
| `[WS-INGEST]` | `app_main.py:2801, 2843, 3328, 3384` | `device=`, `socket=`, `event=`/`type=`, byte/queue counters |
| `[AUDIO-HEALTH]` | around `app_main.py:2595-2598` | `queue_high_water=`, `dropped_delta=`, `dropped_total=`, `receive_gap_max_ms=`, `receive_gap_buckets=` |
| `[AUDIO-INTEGRITY]` | `app_main.py:856` | `turn_id=`, `status=degraded`, `epoch=`, `chunks=`, `duration_ms=`, `reason=` |
| `[BARGE-IN]` | `app_main.py:2691` | `device=`, `event=received`, `generation=` |
| `[RGB-CANONICAL]` / `[RGB-CANONICAL-HEALTH]` | `perception_orientation.py:181, 191, 205` | frame-failure counts, health snapshot |
| `[RESEARCH-EXPORTER]` / `[RESEARCH-EXPORTER-HEALTH]` / `[RESEARCH-GATE]` | `research_exporter.py` | export/gate lifecycle |
| `[TTS]` / `[TTS-WS]` | `app_main.py` (multiple) | TTS send lifecycle |
| `[MIC]` / `[MIC-LOSS]` | `app_main.py` (multiple) | VAD/mic state, mic-loss packets |
| `[NAV MASTER]` / `[NAVIGATION]` / `[CROSS_STREET]` | `app_main.py` | navigation state machine (largely inert under `STABILITY_MODE`, since `general_detector`/`yolomedia` are disabled — `app_main.py:107-120`) |
| `[STABILITY]` | `app_main.py:3985` | stability-mode notices |
| `[SHUTDOWN]` | `app_main.py:4072` | graceful shutdown |

(Full field-by-field enumeration of every tag was not exhaustively completed for every log line in a 4099-line file within this pass; the above is representative and cites real, verified locations, not a complete index.)

### Environment variables read at runtime (names only; defaults as coded, no values printed)

From `os.getenv(...)` calls across `*.py` at repo root:

`AI_BACKEND`, `AIGLASS_AMP`, `AIGLASS_BLINDPATH_INTERVAL`, `AIGLASS_COMPRESS_TYPE`, `AIGLASS_CROSSWALK_INTERVAL`, `AIGLASS_DEBUG_TRAFFIC_LIGHT`, `AIGLASS_DEVICE`, `AIGLASS_DIRECTION_INTERVAL`, `AIGLASS_GPU_SLOTS`, `AIGLASS_MASK_MIN_AREA`, `AIGLASS_MASK_MORPH`, `AIGLASS_OBS_AUTO`, `AIGLASS_OBS_CACHE_FRAMES`, `AIGLASS_OBS_CONF`, `AIGLASS_OBS_INTERVAL`, `AIGLASS_OBS_MODEL`, `AIGLASS_PANEL_SCALE`, `AIGLASS_SEG_BP_ID`, `AIGLASS_SEG_CW_ID`, `AIGLASS_SIMULATE_TRAFFIC_LIGHT`, `AIGLASS_STRAIGHT_CONTINUOUS`, `AIGLASS_STRAIGHT_INTERVAL`, `AIGLASS_STRAIGHT_LIMIT`, `ASR_DEBUG_RAW`, `AUDIO_STALE_AFTER_MS`, `BLIND_MIN_CONF`, `CROSSWALK_ANGLE_THRESH_DEG`, `CROSSWALK_MIN_AREA`, `CROSSWALK_MIN_CONF`, `CROSSWALK_OFFSET_THRESH`, `DASHSCOPE_API_KEY`, `DASHSCOPE_COMPAT_BASE`, `GEMINI_API_KEY`, `GEMINI_MODEL_ID` (default `"gemini-2.5-flash"`, only affects `gemini_client.py`, not the live model), `GEMINI_VIDEO_INTERVAL_SEC` (default `"0.75"`), `GENERAL_DET_CONF`, `GENERAL_DET_NO_MPS`, `HAND_LANDMARKER`, `HAND_SERVICE_TOKEN`, `HAND_SERVICE_URL`, `INTERRUPT_KEYWORDS`, `LATENCY_LOG_CSV` (default off/empty), `MODEL_BACKEND` (default `"gemini"`), `QWEN_MAX_NEW_TOKENS`, `QWEN_MODEL`, `QWEN_OMNI_MODEL`, `RESEARCH_FRAME_SAMPLE_INTERVAL_SEC` (default `"5.0"`), `RESEARCH_INGEST_PATH` (default `"/api/ingest"`), `RESEARCH_PLATFORM_URL`, `RESEARCH_SESSION_DEFAULT_TTL_SEC` (falls back to `DEFAULT_SESSION_TTL_SEC` = 21600s), `RESEARCH_SESSION_SHARED_SECRET`, `RESEARCH_SESSION_STATE_PATH`, `SHOPPING_MODEL`, `STABILITY_TEST_TOKEN`, `THERMAL_FACT_MAX_AGE_SEC` (default `"3.0"`), `VISION_FRAME_MAX_AGE_SEC` (default `"3.0"`), `VISION_MIN_INTERVAL_SEC` (default `"2.0"`), `YOLO_SERVICE_TOKEN`, `YOLO_SERVICE_URL`, `YOLO_TRACKER_YAML`.

Plus, read via `env_bool()`/direct comparison rather than `os.getenv(...)` literal grep: `STABILITY_MODE` (`app_main.py:17`, default `True`), `ENABLE_YOLO`, `YOLO_CONFIDENCE`, `YOLO_MIN_INTERVAL_SEC`, `YOLO_REQUEST_TIMEOUT_SEC` (all set via `fly.toml`'s `[env]` block, consumed presumably in `yolo_client.py` — not individually re-verified by grep in this pass), `ENABLE_RESEARCH_EXPORT` (`research_exporter.py`, default off).

### Feature flags / dead toggles

- `STABILITY_MODE` (`app_main.py:17`, default `True`) is the single biggest behavior switch: it disables legacy `/ws/camera`/`/ws/thermal` endpoints (§3), disables `general_detector`/`yolomedia` imports (`app_main.py:107-120`), and gates `bridge_io.push_raw_jpeg` (`app_main.py:3031-3032`).
- `DEBUG = False` and `DEBUG_VAD = False` (`app_main.py:121-122`) — hardcoded (not env-driven) verbosity toggles for navigation/recorder/YOLO logs and VAD RMS debug printing respectively.
- `MODEL_BACKEND` / `AI_BACKEND` — two separate, similarly-named env switches: `MODEL_BACKEND` picks between `gemini_client`/`omni_client` for the `vision_backend.py` `stream_chat` re-export (§2), while `AI_BACKEND` picks the overall conversational backend (`"gemini_live"` vs others). Per the user's own prior memory notes, the non-`gemini_live` (`gemini_regular`/`qwen`) paths have known unresolved `NameError` bugs — **not independently re-verified in this pass**, but the import-time coupling to `gemini_client.py`/Pillow (§2) is newly confirmed here.

---

## 12. Honest assessment

### Three things most likely to produce a model answer grounded in an old frame while fresh frames are arriving

1. **The continuous send path has no staleness/freshness gate at all** (§3) — while this actually means frames sent this way are *always current* (sent as they're canonicalized), it also means **there is no mechanism ensuring the frame Gemini is currently "looking at" internally corresponds to the moment the user asked their question** — the model integrates whatever frames arrived in its sliding context window, with no hard binding between a specific frame and a specific spoken question. If Gemini's own internal buffering/inference lags behind the 0.75s send cadence under load, the most recently *processed* frame could meaningfully trail the most recently *sent* one, and nothing in this code would detect or report that gap.
2. **`VisionController`'s freshness gate (`VISION_FRAME_MAX_AGE_SEC=3.0`) exists but is not wired into either live send path** (§3) — a real staleness check was built (`stability_runtime.py:718-747`) and is completely bypassed by both the continuous pump and the `pre_response` sender, which both pull straight from `latest_rgb.snapshot()`/the raw canonicalized buffer with no age rejection. If this was intended to gate vision sends, it currently does not.
3. **`context_window_compression: sliding_window()` with no explicit token target** (`gemini_live_client.py:219-221`) — the session lets Gemini's own default sliding-window compression decide what old frames/turns get dropped from context as the session runs long, with zero visibility or control from this codebase into what specifically gets compressed away. A long session (matching P01/P02's ~1hr durations) could have its earliest frames evicted from the model's working context well before the conversation ends, and there's no code-side signal of when/what that happens.

### Contradictions found within this audit

- The task brief stated `stability_runtime.py` was **not found at the repo root**; it is present, 830 lines, actively imported by `app_main.py:1028-1033` and load-bearing (`LatencyTracker`, `LatestFrameStore`, etc.) — this is the most significant correction to the initial framing.
- Firmware comment at `compile/compile.ino:128-135` documents a JPEG-quality tuning rationale that references values (25 → 32 → 30) not matching the actual compiled constant (`#define JPEG_QUALITY 16`, `compile/compile.ino:136`) — comment and code appear to have drifted apart.
- `latency_tracker` is instantiated twice in immediate succession (`app_main.py:1086-1087`) — the first instantiation (without `on_event=research_exporter.publish_event`) is dead, immediately overwritten by the second. Harmless but sloppy — worth a one-line cleanup, and worth knowing that any code path that might have run between those two lines (none currently does) would have used the un-wired tracker.

### Five questions this repository alone cannot answer

1. **Which exact commit/build was flashed to the glasses and running on the backend during P01 and P02** — no version telemetry exists anywhere (§1); would need Fly's deploy/release history (`fly releases` output or GitHub Actions run log for `.github/workflows/fly-deploy.yml`) and the firmware flash log/build artifact from whatever machine compiled and uploaded the `.ino`.
2. **Whether the ~153-second repeating cycle is the Python Gemini-reconnect backoff, the firmware camera-socket backoff, or something else entirely (e.g., WiFi AP-side behavior, Fly edge/proxy timeout)** — the ~150s Python backoff sum is suggestive but the code as written does not auto-repeat that cycle (§4); would need the actual `[GEMINI-SESSION]`/`[WS-CAM]` log lines from the P01/P02 sessions to see real attempt counts and timestamps.
3. **What the firmware mic send cadence/chunk-size constant actually is** — not located by name in this pass (§8); would need a more targeted read of the audio task in `compile/compile.ino` (only partially covered here).
4. **Whether the tests in `tests/` actually run in CI before a Fly deploy** — `.github/workflows/fly-deploy.yml` exists but its contents were not read in this pass; would need to open that file (a five-minute follow-up, simply not done here given scope).
5. **Whether yaw is gyro-integrated-only versus sensor-fused, and where (if anywhere) that integration happens** — the wire struct only carries raw accel+gyro (§8); would need to search `main.js`/`index.html` or any browser-side code for orientation-integration math, not done in this pass.

---

*Audit conducted via git history, static code reading, and grep/search only. No code was executed, modified, or committed. All file:line citations verified against the working tree as of HEAD `a33eab836d580cde148d484268860ac9b153e623` plus the noted uncommitted `compile/compile.ino` diff.*
