#!/usr/bin/env bash
# Print the values you need in compile/compile.ino before flashing the ESP32.
# The ESP32 must be on the same WiFi as this Mac, and reach this Mac at the
# printed IP on port 8081.
#
# Usage:
#   ./scripts/esp32_config.sh

set -eu

echo "=== ESP32 flash-time config ==="
echo ""

ssid=$(networksetup -getairportnetwork en0 2>/dev/null | awk -F': ' '{print $2}' || true)
if [[ -z "${ssid:-}" || "$ssid" == *"not associated"* ]]; then
  echo "WiFi:   (not connected to an AirPort network — connect to the same WiFi the ESP32 will use)"
else
  echo "WiFi:   $ssid"
fi

ip=$(ipconfig getifaddr en0 2>/dev/null || true)
if [[ -z "${ip:-}" ]]; then
  echo "Mac IP: (no IP on en0 — check 'ifconfig')"
else
  echo "Mac IP: $ip"
  # Sanity-check: 100.x is often Tailscale/CGNAT and not reachable by the ESP32.
  case "$ip" in
    100.*) echo "        (warning: 100.x looks like Tailscale/CGNAT — the ESP32 probably can't reach this.)" ;;
  esac
fi

echo ""
echo "Open compile/compile.ino and set:"
echo "  WIFI_SSID   = \"${ssid:-<your-wifi>}\";"
echo "  WIFI_PASS   = \"<your-wifi-password>\";"
echo "  SERVER_HOST = \"${ip:-<your-mac-ip>}\";"
echo "  SERVER_PORT = 8081;"
echo ""
echo "Then in a separate terminal:"
echo "  python app_main.py"
echo ""
echo "And flash the firmware. Watch the ESP32 serial monitor for:"
echo "  [WS-CAM] open"
echo "  [WS-AUD] open"
echo "  [WS-THRM] open"
echo "Server side should print [CONNECTED] Camera / Mic."
