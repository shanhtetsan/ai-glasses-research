---
name: esp32-network-setup
description: Current ESP32/XIAO S3 network, upload, and server integration state
metadata:
  type: project
---

Current ESP32 state before Mac reboot:

- Board: Seeed XIAO ESP32S3 Sense, serial port has been `/dev/cu.usbmodem101`.
- Arduino CLI installed via Homebrew. ESP32 core and `ArduinoWebsockets` library are installed.
- Compile target that works: `esp32:esp32:XIAO_ESP32S3:PSRAM=opi`.
- Sketch compiles successfully with PSRAM enabled.
- Upload command:
  `arduino-cli upload -p /dev/cu.usbmodem101 --fqbn esp32:esp32:XIAO_ESP32S3:PSRAM=opi compile`
- Monitor command:
  `arduino-cli monitor -p /dev/cu.usbmodem101 --config baudrate=115200`

Firmware state:

- `compile/compile.ino` has Wi-Fi diagnostics, 30s retry loop, runtime network status, PSRAM camera compile target, and WebSocket retry target logging.
- Current intended Wi-Fi for Mac Internet Sharing: `WIFI_SSID = "PRST"`, `WIFI_PASS = "phone12345"`.
- Current intended server mode: `USE_GATEWAY_AS_SERVER = true`, so ESP32 connects to `WiFi.gatewayIP():8081`.
- ESP32 memory follow-up after reboot: reduced `TTS_QUEUE_DEPTH` from 48 to 12 (~98 KB to ~25 KB queue storage), added boot/after-init heap + PSRAM diagnostics, and added queue/task allocation checks that restart with explicit `[MEM]` logs on failure.
- This is correct only when ESP32 joins the Mac-created Internet Sharing Wi-Fi. Expected ESP32 log:
  `[WiFi] OK ip=192.168.x.y gateway=192.168.x.1`
  `[MEM] after-tasks heap_free=... internal_free=... psram_free=...`
  `[NET] resolved server=192.168.x.1:8081`
  `[WS-CAM] connected`
  `[WS-AUD] connected`

What was working:

- Laptop/server voice path works with Whisper.
- Omni now loads successfully after upgrading Torch.
- ESP32 Wi-Fi can join phone hotspot `GTHE`, camera initializes when compiled with `PSRAM=opi`, IMU works.

Main blocker:

- Phone hotspot put ESP32 on `172.20.10.6` with gateway `172.20.10.1`, while Mac was `192.0.0.2`; they could not reach each other. WebSockets retried forever.
- Mac Internet Sharing was not active yet. `ifconfig bridge100` did not exist, and `ifconfig | grep -A5 bridge` only showed `bridge0` without an IPv4 `inet`.
- User was about to reboot Mac to get iPhone USB / Internet Sharing working.

After reboot, resume checklist:

1. Plug iPhone into Mac by USB and trust computer.
2. Confirm iPhone USB interface appears, likely `en5`, with `ifconfig`.
3. In System Settings -> General -> Sharing -> Internet Sharing:
   - Share from iPhone USB / `en5`
   - To devices using Wi-Fi
   - Wi-Fi name `PRST`
   - Password `phone12345`
   - Turn Internet Sharing ON and confirm any Start dialog.
4. Confirm Mac sharing bridge:
   `ifconfig | grep "192.168"` should show `192.168.x.1`.
5. Start server:
   `cd /Users/minerva/Desktop/ai-glasses-research && source venv/bin/activate && python app_main.py`
6. Upload/monitor ESP32 if needed using commands above.
7. Success condition: ESP32 gets `192.168.x.y`, gateway `192.168.x.1`, resolved server `192.168.x.1:8081`, and both WebSockets connect.

Post-reboot check:

- `ifconfig` shows iPhone USB `en5` active at `192.0.0.2`.
- Internet Sharing bridge is not active yet: no `bridge100` / `192.168.x.1` address was visible.
- `arduino-cli board list` did not show `/dev/cu.usbmodem101`; only Bluetooth/debug serial ports were listed.
- Firmware compile with `esp32:esp32:XIAO_ESP32S3:PSRAM=opi` succeeds:
  program storage 1,051,122 bytes (31%); global variables 75,924 bytes (23%), leaving 251,756 bytes.
