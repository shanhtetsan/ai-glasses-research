# Strict Stability Acceptance Test

These steps validate the `STABILITY_MODE=1` firmware and `STABILITY_MODE=true`
backend. Do not treat a successful compile as hardware validation.

## Build, flash, and monitor

```bash
cd /Users/shanhtetsan/OpenAIglasses_for_Navigation
CLI="/Applications/Arduino IDE.app/Contents/Resources/app/lib/backend/resources/arduino-cli"
"$CLI" compile --fqbn esp32:esp32:XIAO_ESP32S3 \
  --build-path /private/tmp/openaiglasses-build compile
"$CLI" board list
PORT=/dev/cu.usbmodemXXXX
"$CLI" upload --fqbn esp32:esp32:XIAO_ESP32S3 \
  --port "$PORT" --build-path /private/tmp/openaiglasses-build compile
"$CLI" monitor --port "$PORT" --fqbn esp32:esp32:XIAO_ESP32S3 \
  --config baudrate=115200 --timestamp
```

Start the backend:

```bash
STABILITY_MODE=true STABILITY_TEST_TOKEN=replace-me \
  venv/bin/python -m uvicorn app_main:app --host 0.0.0.0 --port 8081
```

Run automated tests:

```bash
venv/bin/python -m unittest discover -s tests -v
```

## Boot test

Confirm all task results are `pdPASS`, followed by:

```text
[TASKS] all essential tasks pdPASS; network owners released
[I2S IN] ... ready
[I2S OUT] ... ready
[THERMAL] Ready
[IMU] MPU-6050 init OK (I2C)
[WS-AUD] open
[WS-CAM-THERMAL] open
```

Reject the build if any queue, mutex, PSRAM allocation, I2S initialization, or
essential task reports `[FATAL]`.

## Five-minute idle test

Leave the glasses stationary for five minutes. Save every `[HEALTH]` line.
Verify that:

- reconnect counters do not climb continuously;
- `minHeap` and `internalLargest` settle rather than decline every interval;
- every active task stack watermark stays above zero;
- IMU sequence advances at about 10 Hz and the UI does not show `stale`.

## RGB + thermal + IMU test

Run all three for 30 minutes. Open the web UI and alternate RGB/Thermal views.
Change palette, auto/manual range, hotspot, and interpolation. These controls
must change rendering without restarting the sensor or device.

Verify camera/thermal/IMU queues remain at depth zero or one, RGB remains live,
thermal min/max remain plausible, IMU axes show m/s² and deg/s, and sequence
numbers continue advancing.

Use the validation endpoint while performing each motion:

```bash
for mode in stationary tilt_forward tilt_backward tilt_left tilt_right rotate; do
  curl -s "http://127.0.0.1:8081/api/imu-validation?mode=$mode"
  sleep 3
done
```

## Audio test

Perform at least ten microphone interactions and hear ten complete speaker
responses. Confirm microphone send counters advance during RGB/thermal/IMU
traffic, TTS queue depth remains bounded, and every response reaches playback
completion without I2S errors.

## Gemini vision test

Before speaking a visual request, confirm `vision_successes` is zero or
unchanged:

```bash
curl -s http://127.0.0.1:8081/api/health
```

Perform at least five explicit requests such as “What is in front of me?” or
“Read this sign.” Confirm one recent cached frame is submitted per accepted
request and RGB streaming continues. Test the authenticated path:

```bash
curl -s -X POST http://127.0.0.1:8081/api/vision \
  -H 'Content-Type: application/json' \
  -H 'X-Stability-Token: replace-me' \
  -d '{"reason":"manual acceptance"}'
```

Repeat immediately to verify rate limiting. Stop RGB for over three seconds and
verify a controlled `no_recent_frame` result.

## Recording test

Start and stop recording three times:

```bash
curl -s -X POST http://127.0.0.1:8081/api/recording \
  -H 'Content-Type: application/json' -d '{"active":false}'
curl -s -X POST http://127.0.0.1:8081/api/recording \
  -H 'Content-Type: application/json' -d '{"active":true}'
```

Confirm produced video/audio files are playable, queue depth never exceeds two
RGB frames, drops are reported under load, and recording never disconnects the
ESP32.

RGB and audio are written by separate bounded workers. Synchronization is based
on the recorder's video timeline and silence padding; thermal is not embedded
in the recording. Under overload, dropped RGB frames can reduce temporal
precision, so frame-perfect RGB/audio/thermal synchronization is not promised.

## Network recovery test

Disable Wi-Fi/hotspot or stop the backend for approximately 20 seconds, then
restore it. Confirm exponential/capped retries, audio recovery first, then the
camera/sensor socket, with no permanent rapid reconnect loop.

## Weak-network test

Throttle or weaken the link for at least five minutes. Confirm camera frames,
thermal samples, and IMU samples are replaced/dropped instead of accumulated.
Audio must continue or recover independently.

The development simulator can exercise dispatch, slow sends, malformed packets,
and reconnects:

```bash
venv/bin/python dev_sensor_client.py --seconds 60 --delay 0.5 \
  --disconnect-every 10 --malformed
```
