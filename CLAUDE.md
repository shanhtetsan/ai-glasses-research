# CLAUDE.md — ai-glasses-research

Facts and gotchas. Check the code before acting on any of this; fix this file if it is wrong.

## Hardware / firmware (`compile/compile.ino`)

- IMU is an MPU-6050 on I2C at `0x68` (`MPU_ADDR`), read with bare `Wire`.
  Units are converted on-device: m/s² and deg/s. The backend must not rescale.
  The "ICM42688 SPI" in the file's first-line comment is stale; no ICM42688 code remains.
- `ENABLE_ESP32_AUDIO 0`: I2S mic and speaker tasks are compiled out.
  Speech and reply audio go through the iOS app (`ai-glasses-ios`) on `/ws_audio`.
- `camera_config_t` must be zero-initialised: `camera_config_t config = {};`.
  Arduino ESP32 board package 3.3.12 added `jpeg_buffer_size` to the struct.
  Stack garbage there sets the JPEG frame-buffer size, which gives
  "frame buffer malloc failed" while PSRAM is free, and later `FB-OVF` with `captured=0`.
- `s->set_hmirror(s, 1)` stays. A team hardware test on 2026-10-25 found it does not
  invert left/right in answers. Do not "fix" it.
- Nothing allocates PSRAM before `init_camera()`. It is the first call in `setup()`.
  Queues, the 250 KB `camThermalTxBuf` and the thermal buffers all come later.

## Orientation (`perception_orientation.py`)

- Frames are canonicalised once, at ingest:
  - RGB: `cv2.rotate(..., ROTATE_90_COUNTERCLOCKWISE)`.
  - Thermal: `canonicalize_thermal_payload` = `rot90(k=1)` then `fliplr`.
  Never rotate or flip again downstream.
- `CAMERA_ROTATION_DEG = 0` (`app_main.py`). `POST /api/settings` returns 400
  for any nonzero `camera_rotation_deg`.
- Phone frames are the exception: `/ws/camera_thermal?source=phone` frames arrive upright and
  go through `canonicalize_phone_rgb_jpeg` (same decode + quality-75 re-encode, no rotation).
  No param (the ESP32) means `glasses`; any other value is refused with 1008.
  A phone's IMU must be sent in glasses units and sign (deg/s; m/s² reading +g up at rest).
  iOS gives rad/s and g with the opposite sign; the backend does not convert.
- Frames are stored and shown exactly as Gemini received them.
  Do not add a display rotation to the trial viewer.

## Backend (`app_main.py`, Fly app `ai-glasses-for-research`)

- `/data` is the mounted Fly volume (`fly.toml` `[[mounts]]`) and persists.
  `/app` is rebuilt on every deploy. Nothing durable goes in `/app`.
  Gotcha: `SyncRecorder` defaults to `output_dir="recordings"`, i.e. `/app/recordings`,
  so those recordings are lost on deploy.
- Image sends to Gemini are tagged `source=`. Know which one a log line is before
  drawing conclusions:
  - `image`: the continuous pump, every `GEMINI_VIDEO_INTERVAL_SEC` (0.75 s) while the mic streams.
    It logs as `pump` when `TRIAL_RECORDER=1`.
  - `pre_response`: the frame pinned to a turn, at most once per turn.
  - `keepalive`: the idle-session frame. A rising `consecutive_repeats` means the
    camera stopped and one frozen frame is being resent.
  - `POST /api/vision` (`VisionController`) also sends with the default `source=image`,
    so it is indistinguishable from the pump in the log.
  - Each send writes about four `[GEMINI-TIMING]` lines. Count
    `event=image_sdk_send_end ... result=sent`, or the `[VISION] ... pre-response` line.
- `PIN_FRAME_EVERY_TURN=1` (default on) is what grounds a turn. It sends the latest frame
  if it is younger than `VISION_FRAME_MAX_AGE_SEC` (3 s).
  - Do not add phrases to `_VISION_PHRASES`. `is_explicit_vision_request()` is only the
    logged `vision_intent` signal.
  - Set the env var to 0 only for the ablation, and prefer not to: the arm is a
    per-trial setting now. `POST /internal/trial {"config": {"pin_frame_every_turn": false}}`
    (or the pinning control on the /trials start form) overrides the env default for that
    recorder session only, so both arms run on one deploy and one machine.
  - Nothing reads `PIN_FRAME_EVERY_TURN` directly except `trial_config.SessionConfig`.
    Resolve it with `session_config.get("pin_frame_every_turn")`, once per turn, in
    `_on_input_transcription`. Register new knobs in `_build_session_config()` and add
    their names to `trial_config.RECORDED_CONFIG_KEYS` (a test asserts the two match).
  - The override is held in memory and never persisted. A crash, a restart or a deploy
    comes back on the env default; a resumed session's `session_resume` event carries the
    config actually in force, which is what `resume_drift()` compares with `meta.json`.
- Primary latency metric: `speech_end_to_first_i2s_ms`, not `backend_turn_total_ms`.
  Despite the name, with ESP32 audio off it comes from the iOS app
  (`LATENCY:DEVICE:<turn>:<ms>`). It measures the app's speech end to the first reply
  audio chunk received, not I2S output.
- `STABILITY_MODE` defaults to true (`fly.toml` sets `'true'`). It disables navigation,
  YOLO/YOLOE (`general_detector`, `yolomedia`), local Whisper and torch imports.
  `requirements-cloud.txt` excludes torch and ultralytics.
- Gemini often files a reply under a server-opened turn (`answer_arrived_under`).
  Turn ids alone do not pair questions with answers.

## Repo hygiene

- `weights/` is gitignored. Never commit model weights.
  A 337 MB CLIP checkpoint once blocked every push (GitHub's limit is 100 MB).
- The repo is public. `compile.ino` holds WiFi credentials, so never commit new ones.
- Run tests with `PYTHONPATH=. venv/bin/python tests/<file>.py` (no pytest).
  `test_app_strict_guards_continuous_vision_and_models` is a known, obsolete failure,
  and `test_yolo_service` needs `psutil`, which `requirements-cloud.txt` excludes.
  Importing `app_main` needs `GEMINI_API_KEY` set to anything non-empty.

## Trial archive (`trial_recorder.py`, `trial_scorer.py`, `trial_storage.py`, `trial_archive.py`)

- `TRIAL_RECORDER=1` turns recording on, under `TRIAL_ROOT` (default `/data/trials`).
  When unset, `POST /internal/trial` returns 409 and nothing is recorded.
- `meta.json` is written once at session open and carries the runtime config observed
  at that moment: `pin_frame_every_turn`, `gemini_video_interval_sec`,
  `vision_frame_max_age_sec`, `stability_mode`, `model`, plus `config_overridden`
  naming whichever of them this trial set for itself. These come from
  `session_config.observed()`, never from the request body, so the typed `cond` label
  and the arm that ran can be checked against each other (`condition_mismatch()`; the
  trial page warns on a disagreement).
- `sensor_source` (`phone`/`glasses`, from `camera_source` in `app_main.py`) is a static field in
  `meta.json` and `session_start`, not a `session_config` knob. It is `null` until a device first
  connects. A phone<->glasses switch in an open trial writes `sensor_source_change`; the first
  connect after `null`, and a same-device reconnect, do not.
- `TRIAL_TOKEN` gates `/trials`, `/api/trials*`, frames and ZIPs. They fail closed with 401
  when it is unset. It also gates `/internal/trial*` when set, carried either as the
  `X-Trial-Token` header or as the `trial_token` cookie the viewer already holds
  (`_trial_auth_failed`), so the control block on the page works without pasting a token.
- The untitled fallback session (`untitled_<ISO8601>`) is opened lazily, at the first
  event that is evidence: a `speech_start`, or a `turn_end` carrying a question or a
  reply (`UNTITLED_TRIGGERS` in `trial_recorder.py`). A `START` on the mic socket,
  pump or keepalive frames, IMU and session rotations do NOT open one — they all
  happen on an idle box after a deploy, which used to write a 0-turn folder per deploy.
  `POST /internal/trial` still opens immediately, and `current.json` resume is unchanged.
- `DELETE /internal/trial/{id}` removes a local trial folder, and refuses with 409
  unless it has zero scored windows AND zero recorded turns, or if it is the open
  session. It never touches an R2 copy. The `/trials` list shows a delete control
  only on rows that pass the same test.
- R2 upload needs four secrets: `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`,
  `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`. `R2_ENDPOINT_URL` is optional.
  The judge needs `GEMINI_API_KEY`.
- Deploy with `fly deploy --build-arg GIT_SHA=$(git rev-parse HEAD)`.
  The image has no `.git`, so without it `meta.json` says `git_sha: "unknown"`
  and the trial cannot be tied to a build.
- The scorer groups by question window (`speech_start` to the next `speech_start`).
  Distance units come from a deterministic regex; the judge never decides them.
- IMU streams to the recorder only while a server turn is open, so heading has gaps.
  The viewer says "no IMU samples at this frame". This is a known limitation.
