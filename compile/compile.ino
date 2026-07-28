// ===== all_in_one_merged.ino — XIAO ESP32S3 Sense: Camera + Mic (PDM) + IMU (ICM42688 SPI) =====


#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_camera.h>
#include <ArduinoWebsockets.h>
#include "ESP_I2S.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
struct WavFmt;
#include <cstring>      // memcmp
#include <WiFiClient.h>
#include <WiFiClientSecure.h>
#include <Wire.h>
using namespace websockets;

// ===== Thermal enable/disable switch =====
// Flip to 0 and reflash to fully disable the thermal subsystem (no MLX90640
// init, no thermal frames sent) without touching anything else below, if it
// causes instability during testing. As of the single-socket multiplex
// rewrite there is only one TLS connection (wsMain) total — this flag no
// longer gates a separate connection, only whether FRAME_TYPE_THERMAL
// frames are ever sent over it.
#define THERMAL_ENABLED 0

// ===== IMU frame enable/disable switch =====
// Same as THERMAL_ENABLED above: gates whether FRAME_TYPE_IMU frames are
// ever sent over wsMain, not a separate connection (there is only the one
// now). The one and only time thermal alone was tested tonight (back when
// this was still a 4th concurrent TLS connection), it crashed the VM — kept
// as a distinct flag from THERMAL_ENABLED so each frame type can still be
// disabled independently while testing against openaiglasses-thermal-test.
#define IMU_WS_ENABLED 0

// ===== WiFi / Server =====
const char* WIFI_SSID   = "IanLeeiPhone";
const char* WIFI_PASS   = "ianleeiphone1";
const char* SERVER_HOST = "https://openaiglasses-thermal-test.fly.dev";
const uint16_t SERVER_PORT = 443;  // HTTPS/WSS port

// Single multiplexed endpoint replacing the old /ws/camera, /ws_audio,
// /ws/thermal, and /ws paths — see the framing helpers below.
static const char* MULTIPLEX_WS_PATH = "/ws/multiplex";

// TLS CA for Fly.io: "ISRG Root X2", trusted directly rather than the true
// self-signed root ("ISRG Root X1") one hop further up. Verified 2026-07-21
// by connecting directly to SERVER_HOST:443 and walking the served chain —
// actual path is leaf -> YE2 -> "Root YE" -> this cert (NOT "leaf -> YE2 ->
// X2" directly; there's a "Root YE" hop in between). This cert is itself
// issued by ISRG Root X1 (not self-signed) — that's expected for trusting
// an intermediate directly: mbedTLS just needs this exact cert to appear
// somewhere in the chain the server presents, and stops validating there.
// Byte-identical match confirmed via SHA-256 fingerprint against the live
// chain, not just visual comparison.
//
// Tradeoff vs. the X1 root this replaces: shorter validation chain, but
// shorter-lived — this cert expires 2032-09-02 (X1 runs to 2035-06-04).
// Update this constant before then.
//
// NOTE: ArduinoWebsockets 0.5.4's WebsocketsClient::setInsecure() on ESP32
// only clears its own cached cert fields and never reaches the underlying
// WiFiClientSecure — the ESP32 core's start_ssl_client() then has neither a
// CA cert nor an insecure flag set and refuses to connect at all (returns
// -1 immediately). There is no working "skip verification for now" shortcut
// in this library version on ESP32; setCACert() is the only path that
// actually connects.
static const char FLY_ROOT_CA[] PROGMEM = R"EOF(
-----BEGIN CERTIFICATE-----
MIIEcDCCAligAwIBAgIQbI8dxyfHEX97r4U6yYD5zTANBgkqhkiG9w0BAQsFADBP
MQswCQYDVQQGEwJVUzEpMCcGA1UEChMgSW50ZXJuZXQgU2VjdXJpdHkgUmVzZWFy
Y2ggR3JvdXAxFTATBgNVBAMTDElTUkcgUm9vdCBYMTAeFw0yNjA1MTMwMDAwMDBa
Fw0zMjA5MDIyMzU5NTlaME8xCzAJBgNVBAYTAlVTMSkwJwYDVQQKEyBJbnRlcm5l
dCBTZWN1cml0eSBSZXNlYXJjaCBHcm91cDEVMBMGA1UEAxMMSVNSRyBSb290IFgy
MHYwEAYHKoZIzj0CAQYFK4EEACIDYgAEzZvVn4CDCuwJSvMWSj5cz3es3mcFDR0H
ttwW+1qLFNvicWDEukWVEYmO6gbf9yoWHKS5xcUy4APgHoIYOIvXRdgKam7mAHf7
AlF9ItgKbppbd9/w+kHsOdx1ymgHDB/qo4H1MIHyMA4GA1UdDwEB/wQEAwIBBjAd
BgNVHSUEFjAUBggrBgEFBQcDAQYIKwYBBQUHAwIwDwYDVR0TAQH/BAUwAwEB/zAd
BgNVHQ4EFgQUfEKWrt5LSDv6kviejM9ti6lyN5UwHwYDVR0jBBgwFoAUebRZ5nu2
5eQBc4AIiMgaWPbpm24wMgYIKwYBBQUHAQEEJjAkMCIGCCsGAQUFBzAChhZodHRw
Oi8veDEuaS5sZW5jci5vcmcvMBMGA1UdIAQMMAowCAYGZ4EMAQIBMCcGA1UdHwQg
MB4wHKAaoBiGFmh0dHA6Ly94MS5jLmxlbmNyLm9yZy8wDQYJKoZIhvcNAQELBQAD
ggIBAD2/e9frmMxNpCV03qUHegg+MV2wz9644YoXdqtH8RyWYcBO7xfjjGEXdU1e
/o0OkEFiynUCOSIk/vLLo7ttz6CPAeNlWfC0XNkoGeWgK6jjXvozBaGuGH5n0Ufo
shMeWTuURqNN5G00sSXDTBrpp2+mgvdZQjb8K11TYMA25QA+YHNfbIEL0BniAhKS
2gsnJjSzrdZLI+EZ7SEyqdR2rkjd1KutLDU+n3TFyxjniZVGur4YlhMP3mY/dV95
IruAkkjOZier6hGBdEgZXXvaCz9u9iVEadsIE75pAGL8oHV5vxdARDiotRpul1IN
/UZwzAbrfUFcw1HkAcYD/mlZfnQ2ieCF2MS7j3Vhv7JPDKp45fmykmzYNSrumRW0
upFFKDBOoF7hsOb7oLyHS+Uft6jOUfOrogj8YUx38hKb2K20r42OgsSdDdxdeYWc
MS3Sb6mwJeSZEYxJ2gaXnDSPaKhhrNkYwljyVQyr4Nq+MEJytXNTnHqaAcrNwZlV
pcJL1KBnMrMjP7eanvUwL3FYj3cF17jtboLt7gLoi4+2rWZFvn+w54jmd/FIuhhZ
cEaU/wvU6BUNMtcVquVGHp7itQeDth5j+XL3j4WJ2SABwzUl6OeYdgpIt/ITZa+p
TT0mQ/r5XyA4MEAiabn7XJjvCERlF2dcn2wqJw+CreTkkQ2R
-----END CERTIFICATE-----
)EOF";

// ===== Camera config =====
#define CAMERA_MODEL_XIAO_ESP32S3
#include "camera_pins.h"

framesize_t g_frame_size = FRAMESIZE_VGA;
#define JPEG_QUALITY  17
#define FB_COUNT      2
volatile int g_target_fps = 0;


volatile unsigned long frame_captured_count = 0;
volatile unsigned long frame_sent_count = 0;
volatile unsigned long frame_dropped_count = 0;
volatile unsigned long last_stats_time = 0;
volatile unsigned long ws_send_fail_count = 0;

// ===== Thermal (MLX90640) =====
// Ported from feature/thermal-stability-fix as pure sensor driver code.
// Transport is NOT ported from that branch (it used HTTP POST) — instead
// frames are sent as FRAME_TYPE_THERMAL frames over the shared wsMain
// multiplex connection (see sendFramed() below), dispatched server-side by
// the /ws/multiplex handler in app_main.py.
#if THERMAL_ENABLED
#include "MLX90640_API.h"
#include "MLX90640_I2C_Driver.h"
#endif

#define THERMAL_ADDR 0x33
#define THERMAL_EMISSIVITY 0.95
#define THERMAL_TA_SHIFT 8
// Sensor refresh-rate register code (MLX90640 datasheet): 0x04 = 8 Hz.
// Matched to THERMAL_READ_INTERVAL_MS below so the software read loop never
// asks for frames faster than the sensor itself refreshes them.
#define THERMAL_REFRESH_RATE_CODE 0x04
#define THERMAL_READ_INTERVAL_MS 150  // ~6.7 Hz, within the 4-8 Hz target

// Guards the physical I2C bus, shared between the IMU (MPU-6050) and the
// MLX90640 — both are plain Wire peripherals on the same pins, and without
// this mutex their two FreeRTOS tasks can interleave I2C transactions and
// corrupt each other (the class of bug feature/thermal-stability-fix's own
// name refers to). Declared unconditionally (the IMU helpers below always
// use it, regardless of THERMAL_ENABLED); created once in setup().
SemaphoreHandle_t i2cMutex;

#if THERMAL_ENABLED
paramsMLX90640 mlx90640;
static float thermalPixels[32 * 24];         // 768 floats = 3072 bytes; row-major (24 rows x 32 cols)
// No wire-format buffer needed here anymore — sendFramed() below builds the
// header+payload itself. The old "THRM" 4-byte magic prefix is gone too:
// that existed only so the server could identify the message type without
// a socket of its own; the multiplex frame's Type byte (0x03) already does
// that job, so the payload is now exactly the 3072 raw float32 bytes.
bool thermalReady = false;

bool initThermal() {
  Serial.println("[THERMAL] Initializing...");

  if (!xSemaphoreTake(i2cMutex, portMAX_DELAY)) return false;

  Wire.beginTransmission(THERMAL_ADDR);
  if (Wire.endTransmission() != 0) {
    xSemaphoreGive(i2cMutex);
    Serial.println("[THERMAL] MLX90640 not found");
    return false;
  }

  uint16_t eeMLX90640[832];
  if (MLX90640_DumpEE(THERMAL_ADDR, eeMLX90640) != 0) {
    xSemaphoreGive(i2cMutex);
    return false;
  }

  if (MLX90640_ExtractParameters(eeMLX90640, &mlx90640) != 0) {
    xSemaphoreGive(i2cMutex);
    return false;
  }

  MLX90640_SetRefreshRate(THERMAL_ADDR, THERMAL_REFRESH_RATE_CODE);

  xSemaphoreGive(i2cMutex);

  Serial.println("[THERMAL] Ready");
  return true;
}
#endif  // THERMAL_ENABLED

// ===== Mic (PDM RX) =====
#define I2S_MIC_CLOCK_PIN 42
#define I2S_MIC_DATA_PIN  41
const int SAMPLE_RATE     = 16000; 
const int CHUNK_MS        = 20;
const int BYTES_PER_CHUNK = SAMPLE_RATE * CHUNK_MS / 1000 * 2;
const int AUDIO_QUEUE_DEPTH = 10;

// ===== Speaker (I2S TX → MAX98357A) =====
#define I2S_SPK_BCLK D7
#define I2S_SPK_LRC D8
#define I2S_SPK_DIN  D9
const int TTS_RATE = 8000;

// ===== IMU (MPU-6050 over I2C) =====

// Default I2C pins on XIAO ESP32S3: SDA=D4(GPIO5), SCL=D5(GPIO6)
// Change these if you wired the GY-521 to different pins.
#define IMU_I2C_SDA   D4   // D4
#define IMU_I2C_SCL   D5   // D5

// ===== WS / Queues / I2S =====
// Single multiplexed connection replacing wsCam/wsAud/wsThermal/wsImu.
// Camera, mic, thermal, and IMU frames — and control messages that used to
// be plain-text sends on whichever socket cared (SET:*, TTS:START/END,
// RESTART, SNAP:*) — all now travel over this one TLS connection, tagged by
// the Type byte in each frame's header (see sendFramed()/sendControl()
// below and the matching parser in wsMain's onMessage handler further
// down). This is the whole point of the change: four concurrent TLS
// sessions (four ~16KB+ mbedTLS buffers fighting the same fragmented heap)
// become one.
WebsocketsClient wsMain;
volatile bool main_ws_ready = false;
// Set when ConnectionClosed fires for wsMain; cleared the moment loop() acts
// on it. See the guard in loop() — crash evidence (symbolicated backtrace,
// tonight's flash test) showed wsCam.available() itself crashing
// (LoadProhibited, mbedTLS ssl_parse_record_header) when called right after
// a close, because NetworkClientSecure::write()'s own internal error path
// already tore down (freed) the mbedTLS session without going through
// WebsocketsClient's close()/event machinery — the wrapper's available()
// then dereferences that freed session. This flag skips calling the real
// available() on a client known to be in that state and goes straight to a
// fresh connectSecure(), which constructs a brand-new underlying client
// object (upgradeToSecuredConnection() unconditionally `new`s one) rather
// than touching the stale one. Same reasoning, now just one flag instead of
// four, since there's only one client left to apply it to.
volatile bool main_ws_closed_pending_reconnect = false;
volatile bool snapshot_in_progress = false; // Pause live capture during a high-res snapshot

// Guards every access to wsMain's internals: connectSecure() (setup and
// reconnect), available(), ping(), poll(), sendBinary() (via sendFramed()),
// and close(). ArduinoWebsockets/tiny_websockets has zero internal thread
// safety, and wsMain is now touched concurrently from loop() (core 1) and
// four independent sender tasks — taskCamSend/taskMicUpload on core 1,
// taskThermalLoop/taskImuLoop on core 0 — where the old 4-socket design had
// each WebsocketsClient object touched by essentially one sender task.
// connectSecure() unconditionally rebuilds the underlying TCP/TLS client
// object (upgradeToSecuredConnection() `new`s one) and rewires wsMain's
// internal endpoint on every single call, including reconnects — a
// concurrent sendBinary()/available() call from another core during that
// window is an unsynchronized data race on non-reentrant library internals.
//
// Recursive, not plain like i2cMutex: wsMain.poll() synchronously invokes
// the onMessage/onEvent callbacks while still holding the lock, and those
// callbacks themselves call sendFramed()/sendControl() (e.g. the SNAP:HQ
// handler, the "RESTART" handler's sendControl("START")) — a plain mutex
// would deadlock the same task trying to re-enter it from inside poll();
// a recursive mutex lets the same task recurse while still blocking every
// other task/core.
SemaphoreHandle_t wsMainMutex;

// ---------------------------------------------------------------------
// Multiplex framing: [1B Type][2B Seq BE][2B Length BE][4B Timestamp BE]
// followed by N bytes of payload. 9-byte header total.
//
// Type values:
//   0x01 Camera JPEG frame          0x04 IMU data frame (JSON text)
//   0x02 Audio PCM frame (bidi)     0x05 Control/keepalive (text payload)
//   0x03 Thermal data frame (3072B float32)
//
// Length-field ceiling: it's 2 bytes, so a single frame's payload is capped
// at 65535 bytes. VGA JPEG frames run close to that already (see the
// DMA-overflow investigation: the esp32-camera library itself budgets
// ~61440 bytes for VGA JPEG), and the SNAP:HQ high-res path targets SXGA,
// which will exceed it outright. sendFramed() below refuses to send
// (logs + returns false) rather than silently truncating/wrapping when a
// payload is too big for the field, so an oversized frame is dropped
// loudly instead of corrupting the stream. This is a real ceiling in the
// spec as given, not a bug — flagging it here since camera frames are the
// one payload type actually at risk of hitting it.
//
// Because ArduinoWebsockets delivers one complete WS message per onMessage
// call (it does its own message framing under the hood), the Length field
// here is redundant with — not load-bearing for — finding the payload's
// end: the receiver always knows the true payload size from the WS
// message itself (message size minus the 9-byte header). A wrong Length
// value can never desync a *later* frame the way it would in a raw
// byte-stream protocol; it can only make that one frame's declared size
// wrong. The parser below treats Length as an integrity check (logs a
// mismatch) and slices by the WS message's actual size, not by Length.
//
// Timestamp is device millis() here and server wall-clock ms on the other
// end — NOT a shared clock/epoch. It's informational per-frame metadata
// only; don't use it for cross-side latency math without adding real clock
// sync first.
#define FRAME_TYPE_CAMERA  0x01
#define FRAME_TYPE_AUDIO   0x02
#define FRAME_TYPE_THERMAL 0x03
#define FRAME_TYPE_IMU     0x04
#define FRAME_TYPE_CONTROL 0x05
#define FRAME_HDR_LEN      9
#define FRAME_MAX_PAYLOAD  0xFFFF

static uint16_t g_frame_seq = 0;

// Builds the 9-byte header and sends header+payload as one WS binary
// message. ArduinoWebsockets' sendBinary(const char*, size_t) takes a
// single contiguous buffer (no scatter/gather send), so header and payload
// have to be concatenated before the call — this static buffer (sized to
// the protocol's own 65535B payload ceiling, see above) avoids a per-frame
// heap alloc/free on an already heap-contention-sensitive device. It costs
// ~64KB of static RAM, permanently reserved, which is a genuinely new fixed
// cost on this board — worth watching during openaiglasses-thermal-test
// testing given how much heap fragmentation has already bitten this
// project tonight (mbedTLS, DMA buffers). Shrinkable later if it's a
// problem (e.g. by capping camera frame size well below 65535 and sizing
// this to match).
static uint8_t g_frame_combined_buf[FRAME_HDR_LEN + FRAME_MAX_PAYLOAD];

bool sendFramed(uint8_t type, const uint8_t* payload, size_t len) {
  if (len > FRAME_MAX_PAYLOAD) {
    Serial.printf("[MUX] refusing to send type=0x%02X: %u bytes exceeds %u-byte length-field limit\n",
                  type, (unsigned)len, (unsigned)FRAME_MAX_PAYLOAD);
    return false;
  }

  // g_frame_combined_buf is shared, static scratch space — the header
  // build below and the sendBinary() call both have to be inside the same
  // critical section, not just the send: two tasks racing here could
  // otherwise interleave writes into the same buffer before either one
  // calls sendBinary(). See wsMainMutex's declaration comment for why this
  // needs a recursive mutex.
  xSemaphoreTakeRecursive(wsMainMutex, portMAX_DELAY);

  uint16_t seq = g_frame_seq++;
  uint16_t length = (uint16_t)len;
  uint32_t ts = (uint32_t)millis();

  uint8_t* hdr = g_frame_combined_buf;
  hdr[0] = type;
  hdr[1] = (uint8_t)(seq >> 8);
  hdr[2] = (uint8_t)(seq & 0xFF);
  hdr[3] = (uint8_t)(length >> 8);
  hdr[4] = (uint8_t)(length & 0xFF);
  hdr[5] = (uint8_t)(ts >> 24);
  hdr[6] = (uint8_t)(ts >> 16);
  hdr[7] = (uint8_t)(ts >> 8);
  hdr[8] = (uint8_t)(ts & 0xFF);
  if (len) memcpy(g_frame_combined_buf + FRAME_HDR_LEN, payload, len);

  bool ok = wsMain.sendBinary((const char*)g_frame_combined_buf, FRAME_HDR_LEN + len);

  xSemaphoreGiveRecursive(wsMainMutex);
  return ok;
}

// Control/keepalive frame (Type 0x05) carrying the exact same command
// strings the old text-message sends used ("TTS:START", "SET:QUALITY=17",
// "RESTART", "SNAP:HQ", ...) — just wrapped in the shared frame format
// instead of sent as a raw WS text message. Preserves every existing
// string-matching handler on both ends verbatim; only the transport
// changed.
bool sendControl(const char* text) {
  return sendFramed(FRAME_TYPE_CONTROL, (const uint8_t*)text, strlen(text));
}

typedef camera_fb_t* fb_ptr_t;
QueueHandle_t qFrames;

typedef struct {
  size_t n;
  uint8_t data[BYTES_PER_CHUNK];
} AudioChunk;
QueueHandle_t qAudio;

#define TTS_QUEUE_DEPTH 16
typedef struct { uint16_t n; uint8_t data[2048]; } TTSChunk;
QueueHandle_t qTTS;
volatile bool tts_playing = false;

I2SClass i2sIn;   // PDM RX (Mic)
I2SClass i2sOut;  // STD TX (Speaker)
volatile bool run_audio_stream = false;

// ====================================================================
// Camera
// ====================================================================
bool apply_framesize(framesize_t fs) {
  sensor_t* s = esp_camera_sensor_get();
  if (!s) return false;
  int r = s->set_framesize(s, fs);
  if (r == 0) { g_frame_size = fs; return true; }
  return false;
}

bool init_camera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM; config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn  = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = g_frame_size;
  config.jpeg_quality = JPEG_QUALITY;
  config.fb_count     = FB_COUNT;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) { Serial.printf("[CAM] init failed: 0x%x\n", err); return false; }

  sensor_t * s = esp_camera_sensor_get();
  if (s) {

    s->set_hmirror(s, 1);  // ★ Horizontal mirror to match natural left/right (1=on, 0=off)
    s->set_vflip(s, 0);    // ★ Vertical flip; set to 1 if lens is mounted upside-down

    s->set_brightness(s, 0);
    s->set_contrast(s, 1);
    s->set_saturation(s, 1);
    s->set_gain_ctrl(s, 1);
    s->set_gainceiling(s, (gainceiling_t)GAINCEILING_32X);  // let AGC amplify dark scenes (default cap is 2X)
    s->set_exposure_ctrl(s, 1);   // auto exposure ON by default (UI can toggle via SET:AE_AUTO)
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_aec2(s, 1);            // extended AEC: allows longer integration in low light
    s->set_ae_level(s, 2);        // bias AE brighter (-2..+2); counters bright-lamp-in-frame metering
    // s->set_aec_value(s, 40);   // manual exposure only applies when AE is off (SET:AEC=<v> via UI)
  }
  return true;
}

inline void enqueue_frame(camera_fb_t* fb) {
  if (!fb) return;
  if (xQueueSend(qFrames, &fb, 0) != pdPASS) {

    fb_ptr_t drop = nullptr;
    if (xQueueReceive(qFrames, &drop, 0) == pdPASS) {
      if (drop) {
        esp_camera_fb_return(drop);
        frame_dropped_count++;  
      }
    }
    xQueueSend(qFrames, &fb, 0);
  }
}

void taskCamCapture(void*) {
  unsigned long last_log = 0;
  unsigned long capture_fail_count = 0;
  
  for(;;){
    if (snapshot_in_progress) { vTaskDelay(pdMS_TO_TICKS(5)); continue; }

    if (main_ws_ready) {
      camera_fb_t* fb = esp_camera_fb_get();
      if (fb) {
        frame_captured_count++;
        if (fb->format != PIXFORMAT_JPEG) { 
          esp_camera_fb_return(fb);
          capture_fail_count++;
        }
        else { 
          enqueue_frame(fb);
        }
      } else {
        capture_fail_count++;
        vTaskDelay(pdMS_TO_TICKS(2));
      }
      
      // Print capture stats every 5s
      unsigned long now = millis();
      if (now - last_log > 5000) {
        int queue_waiting = uxQueueMessagesWaiting(qFrames);
        Serial.printf("[CAM-CAP] captured=%lu, queue=%d, fail=%lu\n",
                      frame_captured_count, queue_waiting, capture_fail_count);
        last_log = now;
        capture_fail_count = 0;  // Reset failure count
      }
    } else {
      vTaskDelay(pdMS_TO_TICKS(20));
    }
  }
}

void taskCamSend(void*) {
  static TickType_t lastTick = 0;
  unsigned long last_log = 0;
  unsigned long send_timeout_count = 0;
  unsigned long last_sent_time = 0;
  
  for(;;){
    fb_ptr_t fb = nullptr;
    if (xQueueReceive(qFrames, &fb, pdMS_TO_TICKS(100)) == pdPASS) {
      if (fb && main_ws_ready) {
        // Frame-rate throttle: if target FPS is set, pace sends accordingly; extra frames discarded by qFrames
        if (g_target_fps > 0) {
          const int period_ms = 1000 / g_target_fps;
          TickType_t now = xTaskGetTickCount();
          int elapsed = (now - lastTick) * portTICK_PERIOD_MS;
          if (elapsed < period_ms) vTaskDelay(pdMS_TO_TICKS(period_ms - elapsed));
          lastTick = xTaskGetTickCount();
        }

        // A frame over FRAME_MAX_PAYLOAD isn't a transport failure — sendFramed()
        // refuses it outright (see its comment) — so it's handled separately
        // from a genuine send failure below: drop this one frame and keep
        // the connection, don't tear down wsMain over it.
        if (fb->len > FRAME_MAX_PAYLOAD) {
          Serial.printf("[CAM-SEND] dropping frame: %u bytes exceeds %u-byte frame limit\n",
                        (unsigned)fb->len, (unsigned)FRAME_MAX_PAYLOAD);
          esp_camera_fb_return(fb);
          continue;
        }

        unsigned long send_start = millis();
        bool ok = sendFramed(FRAME_TYPE_CAMERA, fb->buf, fb->len);
        unsigned long send_time = millis() - send_start;

        if (ok) {
          frame_sent_count++;
          last_sent_time = millis();


          if (send_time > 100) {
            Serial.printf("[CAM-SEND] WARNING: send took %lu ms (size=%u)\n", send_time, fb->len);
          }
        } else {
          ws_send_fail_count++;
          Serial.println("[CAM-SEND] ERROR: WebSocket send failed, closing...");
          esp_camera_fb_return(fb);
          xSemaphoreTakeRecursive(wsMainMutex, portMAX_DELAY);
          wsMain.close();
          xSemaphoreGiveRecursive(wsMainMutex);
          main_ws_ready = false;
          continue;
        }
        
        esp_camera_fb_return(fb);
        
        // Print send stats every 5s
        unsigned long now = millis();
        if (now - last_log > 5000) {
          unsigned long gap = now - last_sent_time;
          Serial.printf("[CAM-SEND] sent=%lu, dropped=%lu, ws_fail=%lu, last_gap=%lu ms\n", 
                        frame_sent_count, frame_dropped_count, ws_send_fail_count, gap);
          last_log = now;
        }
        
      } else if (fb) { 
        esp_camera_fb_return(fb); 
      }
    } else {

      unsigned long now = millis();
      if (main_ws_ready && last_sent_time > 0 && (now - last_sent_time) > 3000) {
        Serial.printf("[CAM-SEND] WARNING: No frame sent for %lu ms\n", now - last_sent_time);
        send_timeout_count++;
      }
    }
  }
}
// ====================================================================
// Mic (PDM RX)
// ====================================================================
void init_i2s_in(){
  i2sIn.setPinsPdmRx(I2S_MIC_CLOCK_PIN, I2S_MIC_DATA_PIN);
  if (!i2sIn.begin(I2S_MODE_PDM_RX, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("[I2S IN] init failed");
    while(1) { delay(1000); }
  }
  Serial.println("[I2S IN] PDM RX @16kHz 16bit MONO ready");
}

void taskMicCapture(void*){
  const int samples_per_chunk = BYTES_PER_CHUNK / 2; // int16
  for(;;){
    if (run_audio_stream && main_ws_ready) {
      AudioChunk ch; ch.n = BYTES_PER_CHUNK;
      int16_t* out = reinterpret_cast<int16_t*>(ch.data);
      int i = 0;
      while (i < samples_per_chunk){
        int v = i2sIn.read();
        if (v == -1) { delay(1); continue; }
        out[i++] = (int16_t)v;
      }
      if (xQueueSend(qAudio, &ch, 0) != pdPASS){
        AudioChunk dump;
        xQueueReceive(qAudio, &dump, 0);
        xQueueSend(qAudio, &ch, 0);
      }
    } else {
      vTaskDelay(pdMS_TO_TICKS(5));
    }
  }
}

void taskMicUpload(void*){
  for(;;){
    if (run_audio_stream && main_ws_ready){
      AudioChunk ch;
      if (xQueueReceive(qAudio, &ch, pdMS_TO_TICKS(100)) == pdPASS){
        sendFramed(FRAME_TYPE_AUDIO, ch.data, ch.n);
      }
    } else {
      vTaskDelay(pdMS_TO_TICKS(10));
    }
  }
}

// ====================================================================
// Speaker (I2S TX) + HTTP /stream.wav (chunked-safe)
// ====================================================================
void init_i2s_out(){
  i2sOut.setPins(I2S_SPK_BCLK, I2S_SPK_LRC, I2S_SPK_DIN);
  if (!i2sOut.begin(I2S_MODE_STD, TTS_RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO)) {
    Serial.println("[I2S OUT] init failed");
    while(1){ delay(1000); }
  }
  Serial.println("[I2S OUT] STD TX @16kHz 32bit STEREO ready");
}

struct WavFmt {
  uint16_t audioFormat;   // 1=PCM
  uint16_t numChannels;   // 1=mono
  uint32_t sampleRate;    // 16000
  uint32_t byteRate;
  uint16_t blockAlign;
  uint16_t bitsPerSample; // 16
};

static inline void mono16_to_stereo32_msb(const int16_t* in, size_t nSamp, int32_t* outLR, float gain = 0.7f) {
  for (size_t i = 0; i < nSamp; ++i) {
    int32_t s = (int32_t)((float)in[i] * gain);
    int32_t v32 = s << 16;
    outLR[i*2 + 0] = v32;
    outLR[i*2 + 1] = v32;
  }
}

// === chunked ===
static bool read_line(WiFiClient& cli, String& line, uint32_t timeout_ms=3000){
  line = "";
  uint32_t t0 = millis();
  while (millis() - t0 < timeout_ms){
    while (cli.available()){
      char ch = (char)cli.read();
      if (ch == '\n'){
        if (line.endsWith("\r")) line.remove(line.length()-1);
        return true;
      }
      line += ch;
    }
    delay(1);
  }
  return false;
}

static bool readN_http_body(WiFiClient& cli, uint8_t* buf, size_t n, bool chunked, size_t& chunk_left, uint32_t timeout_ms=3000){
  size_t got = 0;
  uint32_t t0 = millis();

  while (got < n){
    if (!cli.connected()) return false;
    if (!chunked){
      int avail = cli.available();
      if (avail > 0){
        int toread = (int)min((size_t)avail, n - got);
        int r = cli.read(buf + got, toread);
        if (r > 0) got += r;
      } else {
        if (millis() - t0 > timeout_ms) return false;
        delay(1);
      }
    } else {
      if (chunk_left == 0){
        String szline;
        if (!read_line(cli, szline, timeout_ms)) return false;
        int sc = szline.indexOf(';');
        if (sc >= 0) szline = szline.substring(0, sc);
        szline.trim();
        unsigned long sz = strtoul(szline.c_str(), nullptr, 16);
        if (sz == 0){
          String dummy;
          read_line(cli, dummy, 500);
          return false;
        }
        chunk_left = (size_t)sz;
      }
      int avail = cli.available();
      if (avail > 0){
        size_t want = min(n - got, chunk_left);
        int toread = (int)min((size_t)avail, want);
        int r = cli.read(buf + got, toread);
        if (r > 0){
          got += r;
          chunk_left -= (size_t)r;
          if (chunk_left == 0){
            while (cli.available() < 2) { if (millis() - t0 > timeout_ms) return false; delay(1); }
            cli.read(); cli.read();
          }
        }
      } else {
        if (millis() - t0 > timeout_ms) return false;
        delay(1);
      }
    }
  }
  return true;
}

static bool parse_wav_header(WiFiClient& cli, WavFmt& fmt, uint32_t& dataRemaining, bool chunked, size_t& chunk_left){
  uint8_t hdr12[12];
  if (!readN_http_body(cli, hdr12, 12, chunked, chunk_left)) return false;
  if (memcmp(hdr12, "RIFF", 4) != 0 || memcmp(hdr12 + 8, "WAVE", 4) != 0) return false;

  bool gotFmt = false;
  dataRemaining = 0;

  while (true) {
    uint8_t chdr[8];
    if (!readN_http_body(cli, chdr, 8, chunked, chunk_left)) return false;
    uint32_t sz = (uint32_t)chdr[4] | ((uint32_t)chdr[5] << 8) | ((uint32_t)chdr[6] << 16) | ((uint32_t)chdr[7] << 24);

    if (memcmp(chdr, "fmt ", 4) == 0) {
      if (sz < 16) return false;
      uint8_t fmtbuf[32];
      size_t toread = min(sz, (uint32_t)sizeof(fmtbuf));
      if (!readN_http_body(cli, fmtbuf, toread, chunked, chunk_left)) return false;
      uint32_t left = sz - (uint32_t)toread;
      while (left){
        uint8_t dump[64];
        size_t d = min((uint32_t)sizeof(dump), left);
        if (!readN_http_body(cli, dump, d, chunked, chunk_left)) return false;
        left -= d;
      }
      fmt.audioFormat   = (uint16_t) (fmtbuf[0] | (fmtbuf[1] << 8));
      fmt.numChannels   = (uint16_t) (fmtbuf[2] | (fmtbuf[3] << 8));
      fmt.sampleRate    = (uint32_t) (fmtbuf[4] | (fmtbuf[5] << 8) | (fmtbuf[6] << 16) | (fmtbuf[7] << 24));
      fmt.byteRate      = (uint32_t) (fmtbuf[8] | (fmtbuf[9] << 8) | (fmtbuf[10] << 16) | (fmtbuf[11] << 24));
      fmt.blockAlign    = (uint16_t) (fmtbuf[12] | (fmtbuf[13] << 8));
      fmt.bitsPerSample = (uint16_t) (fmtbuf[14] | (fmtbuf[15] << 8));
      gotFmt = true;
    }
    else if (memcmp(chdr, "data", 4) == 0) {
      if (!gotFmt) return false;
      dataRemaining = sz;
      return true;
    }
    else {
      uint32_t left = sz;
      while (left){
        uint8_t dump[128];
        size_t d = min((uint32_t)sizeof(dump), left);
        if (!readN_http_body(cli, dump, d, chunked, chunk_left)) return false;
        left -= d;
      }
    }
  }
}

// ---- HTTP
static TaskHandle_t taskHttpPlayHandle = nullptr;
static volatile bool http_play_running = false;

void taskHttpPlay(void*){
  http_play_running = true;
  // Not asked for directly, but required by the same SERVER_PORT=443 change:
  // this is a plain HTTP GET of /stream.wav, unrelated to wsMain/
  // ArduinoWebsockets — a bare WiFiClient can't complete a TLS handshake, so
  // without this it would silently fail to connect every time the server
  // only serves 443 (Fly.io doesn't listen on plain HTTP at all).
  WiFiClientSecure cli;
  cli.setCACert(FLY_ROOT_CA);

  auto readLine = [&](String& out, uint32_t timeout_ms)->bool {
    out = "";
    uint32_t t0 = millis();
    while (millis() - t0 < timeout_ms) {
      while (cli.available()) {
        char c = (char)cli.read();
        if (c == '\r') continue;
        if (c == '\n') return true;
        out += c;
        if (out.length() > 1024) return false;
      }
      delay(1);
    }
    return false;
  };

  auto readNRaw = [&](uint8_t* dst, size_t n, uint32_t timeout_ms)->bool {
    size_t got = 0;
    uint32_t t0 = millis();
    while (got < n) {
      if (!cli.connected()) return false;
      int avail = cli.available();
      if (avail > 0) {
        int take = (int)min((size_t)avail, n - got);
        int r = cli.read(dst + got, take);
        if (r > 0) { got += r; continue; }
      }
      if (millis() - t0 > timeout_ms) return false;
      delay(1);
    }
    return true;
  };

  auto makeBodyReader = [&](bool& is_chunked, uint32_t& chunk_left){
    return [&](uint8_t* dst, size_t n, uint32_t timeout_ms)->bool {
      size_t filled = 0;
      uint32_t t0 = millis();
      while (filled < n) {
        if (!cli.connected()) return false;
        if (is_chunked) {
          if (chunk_left == 0) {
            String szLine;
            if (!readLine(szLine, timeout_ms)) return false;
            int sc = szLine.indexOf(';');
            if (sc >= 0) szLine = szLine.substring(0, sc);
            szLine.trim();
            uint32_t sz = 0;
            if (sscanf(szLine.c_str(), "%x", &sz) != 1) return false;
            if (sz == 0) { String dummy; readLine(dummy, 200); return false; }
            chunk_left = sz;
          }
          size_t need = (size_t)min<uint32_t>(chunk_left, (uint32_t)(n - filled));
          while (cli.available() < (int)need) {
            if (millis() - t0 > timeout_ms) return false;
            if (!cli.connected()) return false;
            delay(1);
          }
          int r = cli.read(dst + filled, need);
          if (r <= 0) {
            if (millis() - t0 > timeout_ms) return false;
            delay(1); continue;
          }
          filled     += r;
          chunk_left -= r;
          if (chunk_left == 0) {
            char crlf[2];
            if (!readNRaw((uint8_t*)crlf, 2, 200)) return false;
          }
        } else {
          if (!readNRaw(dst + filled, n - filled, timeout_ms)) return false;
          filled = n;
        }
      }
      return true;
    };
  };

  static int32_t outLR[1024 * 2];
  const uint32_t BODY_TIMEOUT_MS = 1500;

  while (http_play_running) {
    if (!cli.connected()) {
      Serial.println("[AUDIO] HTTP connect...");
      Serial.printf("[AUDIO] pre-connect heap: free=%d max_alloc=%d\n",
        ESP.getFreeHeap(), ESP.getMaxAllocHeap());
      if (!cli.connect(SERVER_HOST, SERVER_PORT)) { delay(500); continue; }
      String req =
        String("GET /stream.wav HTTP/1.1\r\n") +
        "Host: " + SERVER_HOST + ":" + String(SERVER_PORT) + "\r\n" +
        "Connection: keep-alive\r\n\r\n";
      cli.print(req);
    }

    bool header_ok  = false;
    bool is_chunked = false;
    uint32_t content_len = 0;
    {
      String line; uint32_t t0 = millis();
      while (millis() - t0 < 3000) {
        if (!readLine(line, 1000)) { if (!cli.connected()) break; continue; }
        String u = line; u.toLowerCase();
        if (u.startsWith("transfer-encoding:")) { if (u.indexOf("chunked") >= 0) is_chunked = true; }
        else if (u.startsWith("content-length:")) { content_len = (uint32_t) strtoul(u.substring(strlen("content-length:")).c_str(), nullptr, 10); }
        if (line.length() == 0) { header_ok = true; break; }
      }
    }
    if (!header_ok) { cli.stop(); delay(300); continue; }

    uint32_t chunk_left = 0;
    auto readBody = makeBodyReader(is_chunked, chunk_left);

    uint8_t hdr12[12];
    if (!readBody(hdr12, 12, 1000)) { cli.stop(); delay(300); continue; }
    if (memcmp(hdr12, "RIFF", 4) != 0 || memcmp(hdr12 + 8, "WAVE", 4) != 0) { cli.stop(); delay(300); continue; }

    bool  gotFmt = false, gotData = false;
    uint8_t chdr[8];
    uint16_t audioFormat=0, numChannels=0, bitsPerSample=0;
    uint32_t sampleRate=0;

    while (!gotData) {
      if (!readBody(chdr, 8, 1000)) { cli.stop(); delay(300); goto reconnect; }
      uint32_t sz = (uint32_t)chdr[4] | ((uint32_t)chdr[5]<<8) | ((uint32_t)chdr[6]<<16) | ((uint32_t)chdr[7]<<24);

      if (memcmp(chdr, "fmt ", 4) == 0) {
        if (sz < 16) { cli.stop(); delay(300); goto reconnect; }
        uint8_t fmtbuf[32];
        size_t toread = min(sz, (uint32_t)sizeof(fmtbuf));
        if (!readBody(fmtbuf, toread, 1000)) { cli.stop(); delay(300); goto reconnect; }
        if (sz > toread) {
          size_t left = sz - toread;
          while (left) { uint8_t dump[128]; size_t d = min(left, sizeof(dump));
            if (!readBody(dump, d, 1000)) { cli.stop(); delay(300); goto reconnect; }
            left -= d;
          }
        }
        audioFormat   = (uint16_t)(fmtbuf[0] | (fmtbuf[1] << 8));
        numChannels   = (uint16_t)(fmtbuf[2] | (fmtbuf[3] << 8));
        sampleRate    = (uint32_t)(fmtbuf[4] | (fmtbuf[5] << 8) | (fmtbuf[6] << 16) | (fmtbuf[7] << 24));
        bitsPerSample = (uint16_t)(fmtbuf[14] | (fmtbuf[15] << 8));
        gotFmt = true;
      }
      else if (memcmp(chdr, "data", 4) == 0) {
        if (!gotFmt) { cli.stop(); delay(300); goto reconnect; }
        gotData = true;
      }
      else {
        size_t left = sz;
        while (left) { uint8_t dump[128]; size_t d = min(left, sizeof(dump));
          if (!readBody(dump, d, 1000)) { cli.stop(); delay(300); goto reconnect; }
          left -= d;
        }
      }
    }

    if (!(audioFormat==1 && numChannels==1 && bitsPerSample==16 && (sampleRate==8000 || sampleRate==12000 || sampleRate==16000))) {
      Serial.printf("[AUDIO] unsupported fmt: ch=%u bits=%u sr=%u af=%u\n",
                    numChannels, bitsPerSample, sampleRate, audioFormat);
      cli.stop(); delay(300); continue;
    }
    Serial.printf("[AUDIO] WAV ok: %u/16bit/mono (chunked=%d)\n", sampleRate, is_chunked ? 1 : 0);

    static uint32_t current_out_rate = 0;
    if (current_out_rate != sampleRate) {

      i2sOut.begin(I2S_MODE_STD, (int)sampleRate, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO);
      current_out_rate = sampleRate;
      Serial.printf("[I2S OUT] reconfig to %u Hz\n", sampleRate);
    }

    while (http_play_running) {
      uint8_t inbuf[2048];
      size_t  filled = 0;

      // Compute 20ms byte count based on sample rate (mono, 16-bit)
      uint32_t bytes20 = (sampleRate * 2 * 20) / 1000; // 16k=640,12k=480,8k=320
      if (bytes20 < 2) bytes20 = 2;

      if (!readBody(inbuf, bytes20, BODY_TIMEOUT_MS)) { break; }
      filled = bytes20;

      while (filled + bytes20 <= sizeof(inbuf)) {
        if (!readBody(inbuf + filled, bytes20, 2)) { break; }
        filled += bytes20;
      }

      if (filled & 1) filled -= 1;
      if (filled == 0) { vTaskDelay(pdMS_TO_TICKS(1)); continue; }

      if (tts_playing) continue;  // WebSocket TTS owns i2sOut right now; discard HTTP audio

      size_t samp = filled / 2;
      mono16_to_stereo32_msb((const int16_t*)inbuf, samp, outLR, 0.8f);

      size_t bytes = samp * 2 * sizeof(int32_t);
      size_t off = 0;
      while (off < bytes && http_play_running) {
        size_t wrote = i2sOut.write((uint8_t*)outLR + off, bytes - off);
        if (wrote == 0) vTaskDelay(pdMS_TO_TICKS(1));
        else off += wrote;
      }
    }

  reconnect:
    cli.stop();
    delay(200);
  }

  cli.stop();
  vTaskDelete(nullptr);
}

void startStreamWav(){
  if (taskHttpPlayHandle) return;
  xTaskCreatePinnedToCore(taskHttpPlay, "http_wav", 8192, nullptr, 2, &taskHttpPlayHandle, 0);
  Serial.println("[AUDIO] http_wav task started");
}
void stopStreamWav(){
  if (!taskHttpPlayHandle) return;
  http_play_running = false;
  vTaskDelay(pdMS_TO_TICKS(50));
  taskHttpPlayHandle = nullptr;
  Serial.println("[AUDIO] http_wav task stopped");
}

// ====================================================================
// TTS
// ====================================================================
void taskTTSPlay(void*){
  static int32_t stereo32Buf[1024*2];
  static bool first_chunk_pending = true;
  for(;;){
    if (!tts_playing){ vTaskDelay(pdMS_TO_TICKS(5)); continue; }
    TTSChunk ch;
    if (xQueueReceive(qTTS, &ch, pdMS_TO_TICKS(50)) == pdPASS){
      if (ch.n == 0) {                     // TTS:END sentinel from server
        tts_playing = false;
        run_audio_stream = true;           // playback truly finished — un-mute the mic
        first_chunk_pending = true;        // next session should log its first chunk again
        continue;
      }
      if (first_chunk_pending) {
        Serial.printf("[TTS-PLAY] chunk n=%u, i2s write starting\n", ch.n);
        first_chunk_pending = false;
      }
      size_t inSamp  = ch.n / 2;
      int16_t* inPtr = (int16_t*)ch.data;
      size_t outPairs = 0;
      for (size_t i = 0; i < inSamp; ++i){
        int32_t s = (int32_t)inPtr[i];
        s = (s * 29491) / 32768;
        int32_t v32 = s << 16;
        stereo32Buf[outPairs*2 + 0] = v32;
        stereo32Buf[outPairs*2 + 1] = v32;
        outPairs++;
        if (outPairs >= 1024){
          size_t bytes = outPairs * 2 * sizeof(int32_t);
          size_t off = 0;
          while (off < bytes){
            size_t wrote = i2sOut.write((uint8_t*)stereo32Buf + off, bytes - off);
            if (wrote == 0) vTaskDelay(pdMS_TO_TICKS(1)); else off += wrote;
          }
          outPairs = 0;
        }
      }
      if (outPairs){
        size_t bytes = outPairs * 2 * sizeof(int32_t);
        size_t off = 0;
        while (off < bytes){
          size_t wrote = i2sOut.write((uint8_t*)stereo32Buf + off, bytes - off);
          if (wrote == 0) vTaskDelay(pdMS_TO_TICKS(1)); else off += wrote;
        }
      }
    }
  }
}

inline void tts_reset_queue(){ if (qTTS) xQueueReset(qTTS); }

// ====================================================================
// IMU (MPU-6050 over I2C, bare Wire) 50 Hz via UDP
// ====================================================================
// No third-party library — avoids the sensor_t typedef collision with
// esp_camera.h that Adafruit_Sensor.h causes.

#define MPU_ADDR          0x68  // I2C address when ADO=LOW (GY-521 default)
#define MPU_REG_WHO_AM_I  0x75  // read-only ID register; MPU-6050 returns 0x68
#define MPU_REG_PWR_MGMT1 0x6B  // bit6=SLEEP; write 0x00 to wake the chip
#define MPU_REG_GYRO_CFG  0x1B  // bits[4:3]=FS_SEL; 0x18 → ±2000 dps
#define MPU_REG_ACCEL_CFG 0x1C  // bits[4:3]=AFS_SEL; 0x18 → ±16 g
#define MPU_REG_ACCEL_OUT 0x3B  // first of 14 burst bytes: AX AY AZ TEMP GX GY GZ

// At ±16 g: 2048 LSB per g.  Multiply by (9.80665 / 2048) to get m/s².
// At ±2000 dps: 16.4 LSB per dps.  Divide by 16.4 to get deg/s.
static const float MPU_ACCEL_SCALE = 9.80665f / 2048.0f;
static const float MPU_GYRO_SCALE  = 1.0f / 16.4f;

// Brings up the shared I2C bus exactly once, before the IMU (and, if
// enabled, thermal) tasks start — so neither task races to call Wire.begin()
// first. Called once from setup(). Necessary as soon as a second I2C
// consumer (MLX90640) exists on this bus; harmless when THERMAL_ENABLED=0.
void initI2cBus() {
  Wire.begin(IMU_I2C_SDA, IMU_I2C_SCL);
  Wire.setClock(400000);
}

static void mpu_write(uint8_t reg, uint8_t val) {
  if (!xSemaphoreTake(i2cMutex, portMAX_DELAY)) return;
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
  xSemaphoreGive(i2cMutex);
}

static uint8_t mpu_read1(uint8_t reg) {
  if (!xSemaphoreTake(i2cMutex, portMAX_DELAY)) return 0xFF;
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);  // repeated-START keeps bus active for the read
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1);
  uint8_t v = Wire.available() ? Wire.read() : 0xFF;
  xSemaphoreGive(i2cMutex);
  return v;
}

static void mpu_read14(uint8_t* dst) {
  if (!xSemaphoreTake(i2cMutex, portMAX_DELAY)) { memset(dst, 0, 14); return; }
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(MPU_REG_ACCEL_OUT);
  Wire.endTransmission(false);  // repeated-START — do not release bus
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14);
  for (uint8_t i = 0; i < 14; i++)
    dst[i] = Wire.available() ? Wire.read() : 0;
  xSemaphoreGive(i2cMutex);
}

bool imu_init_i2c() {
  delay(5);  // bus is already up via initI2cBus(), called once from setup()

  uint8_t who = mpu_read1(MPU_REG_WHO_AM_I);
  Serial.printf("[IMU] WHO_AM_I=0x%02X (expect 0x68)\n", who);
  if (who != 0x68) return false;

  mpu_write(MPU_REG_PWR_MGMT1, 0x00);  // clear SLEEP bit — chip starts sampling
  delay(10);
  mpu_write(MPU_REG_GYRO_CFG,  0x18);  // FS_SEL=3  → ±2000 dps
  mpu_write(MPU_REG_ACCEL_CFG, 0x18);  // AFS_SEL=3 → ±16 g
  Serial.println("[IMU] MPU-6050 init OK (I2C)");
  return true;
}

bool imu_read_once(float& tempC, float& ax, float& ay, float& az,
                   float& gx,   float& gy, float& gz) {
  uint8_t raw[14];
  mpu_read14(raw);

  // All values are 16-bit signed big-endian (high byte first).
  auto s16 = [](uint8_t hi, uint8_t lo) -> int16_t {
    return (int16_t)((uint16_t)hi << 8 | lo);
  };

  ax = s16(raw[0],  raw[1])  * MPU_ACCEL_SCALE;  // m/s²
  ay = s16(raw[2],  raw[3])  * MPU_ACCEL_SCALE;
  az = s16(raw[4],  raw[5])  * MPU_ACCEL_SCALE;
  // raw[6..7] = raw temperature — MPU-6050 datasheet formula:
  tempC = s16(raw[6], raw[7]) / 340.0f + 36.53f;
  gx = s16(raw[8],  raw[9])  * MPU_GYRO_SCALE;   // deg/s
  gy = s16(raw[10], raw[11]) * MPU_GYRO_SCALE;
  gz = s16(raw[12], raw[13]) * MPU_GYRO_SCALE;
  return true;
}

// EMA smoothing on accel only; does not change the wire field names.
static const float EMA_ALPHA = 0.20f;
bool  ema_inited = false;
float ax_f=0, ay_f=0, az_f=0;

void taskImuLoop(void*){
  for(;;){
    static bool inited = false;
    if (!inited){
      inited = imu_init_i2c();
      if (!inited){ vTaskDelay(pdMS_TO_TICKS(500)); continue; }
    }

    float tempC, ax, ay, az, gx, gy, gz;
    if (!imu_read_once(tempC, ax, ay, az, gx, gy, gz)){
      inited = false; vTaskDelay(pdMS_TO_TICKS(50)); continue;
    }

    if (!ema_inited){ ax_f=ax; ay_f=ay; az_f=az; ema_inited=true; }
    else {
      ax_f = EMA_ALPHA*ax + (1-EMA_ALPHA)*ax_f;
      ay_f = EMA_ALPHA*ay + (1-EMA_ALPHA)*ay_f;
      az_f = EMA_ALPHA*az + (1-EMA_ALPHA)*az_f;
    }

    char buf[256];
    unsigned long ts = millis();
    int n = snprintf(buf, sizeof(buf),
      "{\"ts\":%lu,\"temp_c\":%.2f,"
      "\"accel\":{\"x\":%.3f,\"y\":%.3f,\"z\":%.3f},"
      "\"gyro\":{\"x\":%.3f,\"y\":%.3f,\"z\":%.3f}}",
      ts, tempC, ax_f, ay_f, az_f, gx, gy, gz);

#if IMU_WS_ENABLED
    if (n > 0 && main_ws_ready) {
      sendFramed(FRAME_TYPE_IMU, (const uint8_t*)buf, (size_t)n);
    }
#endif
    vTaskDelay(pdMS_TO_TICKS(20)); // 50 Hz
  }
}

#if THERMAL_ENABLED
// ====================================================================
// Thermal (MLX90640) — sensor read + multiplexed send
// ====================================================================
// Read cadence/refresh rate ported from thermal-stability-fix's
// taskThermalLoop() (init sequence, MLX90640_GetFrameData/GetTa/CalculateTo
// calls), but re-throttled to THERMAL_READ_INTERVAL_MS (~6.7 Hz, was 3000ms
// there) and re-wired to send as a FRAME_TYPE_THERMAL frame over the shared
// wsMain connection instead of an HTTP POST (or, previously, its own socket).
void taskThermalLoop(void* pv) {
  for (;;) {
    if (!thermalReady) {
      thermalReady = initThermal();
      if (!thermalReady) { vTaskDelay(pdMS_TO_TICKS(2000)); continue; }
    }

    uint16_t frame[834];
    if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(100))) {
      int status = MLX90640_GetFrameData(THERMAL_ADDR, frame);
      xSemaphoreGive(i2cMutex);

      if (status < 0) {
        thermalReady = false;
        vTaskDelay(pdMS_TO_TICKS(1000));
        continue;
      }
    } else {
      Serial.println("[THERMAL] I2C busy, skipping frame");
      vTaskDelay(pdMS_TO_TICKS(THERMAL_READ_INTERVAL_MS));
      continue;
    }

    float Ta = MLX90640_GetTa(frame, &mlx90640);
    float tr = Ta - THERMAL_TA_SHIFT;
    MLX90640_CalculateTo(frame, &mlx90640, THERMAL_EMISSIVITY, tr, thermalPixels);

    if (main_ws_ready) {
      bool ok = sendFramed(FRAME_TYPE_THERMAL, (const uint8_t*)thermalPixels, sizeof(thermalPixels));
      if (!ok) {
        Serial.println("[THERMAL-SEND] ERROR: WebSocket send failed, closing...");
        xSemaphoreTakeRecursive(wsMainMutex, portMAX_DELAY);
        wsMain.close();
        xSemaphoreGiveRecursive(wsMainMutex);
        main_ws_ready = false;
      }
    }

    vTaskDelay(pdMS_TO_TICKS(THERMAL_READ_INTERVAL_MS));
  }
}
#endif  // THERMAL_ENABLED

// ====================================================================
// Setup / Loop
// ====================================================================

// Runs a single client's first connectSecure() attempt to completion
// (success or failure) before returning, so setup()'s initial connection
// sequence is genuinely one-at-a-time rather than four calls fired
// back-to-back. connectSecure() is already synchronous internally (TCP +
// TLS handshake + HTTP upgrade inline), so this doesn't change *that* —
// what it adds is the explicit delay(100) below, giving the heap a moment
// to settle between one TLS teardown/handshake and the next rather than
// chaining four ~16KB+ contiguous mbedTLS allocations with zero gap.
bool connectWsSequential(WebsocketsClient& client, const char* path, const char* objName, const char* tag) {
  Serial.printf("[DEBUG] WiFi status: %d, Free heap: %d, Max alloc heap: %d\n", WiFi.status(), ESP.getFreeHeap(), ESP.getMaxAllocHeap());
  unsigned long t0 = millis();
  bool connected = client.connectSecure(SERVER_HOST, SERVER_PORT, path);
  Serial.printf("[DEBUG] ws%s.connectSecure() (setup, first attempt) took %lu ms, result: %d\n", objName, millis() - t0, connected);
  if (connected) Serial.printf("[WS-%s] connected\n", tag);
  delay(100);
  return connected;
}

void setup() {
  Serial.begin(115200);
  delay(300);

  // Camera claims its memory first, on a clean/unfragmented heap — before
  // WiFi.begin() brings up the WiFi stack's own internal-RAM allocations,
  // and before any connectSecure() call claims a ~16KB+ contiguous mbedTLS
  // buffer (see the connectWsSequential() comment below for why that
  // matters). Camera stays in its default internal-RAM DMA mode (no
  // esp_camera_set_psram_mode(true) call) — PSRAM-DMA mode was tried and
  // reverted: the esp32-camera library's own frame-size guard
  // (recv_size = width*height/5) is under-provisioned for this sensor/
  // quality combo and floods "cam_hal: DMA overflow"/truncated JPEGs, a
  // known-acknowledged issue in this library version with no app-level
  // knob to raise it (the real fix is an ESP-IDF Kconfig value not
  // reachable from Arduino sketch code). The TLS/heap contention that
  // originally motivated PSRAM-DMA mode is instead being addressed via
  // single-socket multiplexing (one connection instead of four) — if that
  // works, PSRAM-DMA mode won't be needed, and stacking both would add
  // risk to an already-large change.
  if (!init_camera()) { Serial.println("[CAM] init failed, reboot..."); delay(1500); esp_restart(); }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);
  esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[WiFi] connecting");
  while (WiFi.status()!=WL_CONNECTED){ delay(300); Serial.print("."); }
  Serial.println(" OK " + WiFi.localIP().toString());
  Serial.printf("[DEBUG] Free heap: %d, Max alloc heap: %d\n", ESP.getFreeHeap(), ESP.getMaxAllocHeap());

  IPAddress resolvedIP;
  if (WiFi.hostByName(SERVER_HOST, resolvedIP)) {
    Serial.printf("[DEBUG] DNS resolved %s -> %s\n", SERVER_HOST, resolvedIP.toString().c_str());
  } else {
    Serial.println("[DEBUG] DNS resolution FAILED for SERVER_HOST");
  }

  // Temporary diagnostic, not the real connection: bypasses TLS entirely to
  // isolate whether a raw socket to SERVER_HOST:SERVER_PORT is even
  // reachable, before blaming the TLS/cert layer for a connect failure.
  WiFiClient testClient;
  Serial.println("[DEBUG] Testing plain TCP connect to port 443...");
  unsigned long tcpStart = millis();
  bool tcpOk = testClient.connect(SERVER_HOST, SERVER_PORT);
  Serial.printf("[DEBUG] Plain TCP connect result: %d, took %lu ms\n", tcpOk, millis() - tcpStart);
  if (tcpOk) testClient.stop();

  // Fly.io serves WSS only (port 443) — both clients need the root CA before
  // their first connectSecure() call. Set once here; upgradeToSecuredConnection()
  // re-applies it from this cached copy on every reconnect attempt in loop().
  //
  // setCACert() has no success/failure indicator to check here — confirmed
  // by reading the source at every layer of this call: WebsocketsClient::
  // setCACert() (ArduinoWebsockets 0.5.4) returns void and just stores the
  // pointer; SecuredEsp32TcpClient::setCACert() (esp32_tcp.hpp) forwards it
  // and also returns void; NetworkClientSecure::setCACert() (the real
  // WiFiClientSecure, esp32 core 3.3.10) is void too — it only stores the
  // pointer and clears _use_insecure, with zero PEM parsing at this point.
  // mbedTLS doesn't actually parse the cert until the TLS handshake inside
  // connectSecure() itself, so a malformed cert can only ever show up there
  // — which is exactly what the [DEBUG] timing/status prints around
  // connectSecure() below are already positioned to catch, not here.
  wsMain.setCACert(FLY_ROOT_CA);
  Serial.printf("[DEBUG] FLY_ROOT_CA length: %d bytes (sanity check the PROGMEM string is intact — not a parse/verify result, setCACert() has none)\n", strlen(FLY_ROOT_CA));

  wsMain.onEvent([](WebsocketsEvent ev, String){
    if (ev == WebsocketsEvent::ConnectionOpened)  {
      main_ws_ready = true;
      Serial.println("[WS-MAIN] open");
      // Reset statistics (previously wsCam's onEvent side effect)
      frame_sent_count = 0;
      frame_dropped_count = 0;
      ws_send_fail_count = 0;
      last_stats_time = millis();
    }
    if (ev == WebsocketsEvent::ConnectionClosed)  {
      main_ws_ready = false;
      main_ws_closed_pending_reconnect = true;
      Serial.printf("[WS-MAIN] closed (sent=%lu, dropped=%lu, fail=%lu)\n",
                    frame_sent_count, frame_dropped_count, ws_send_fail_count);
      stopStreamWav();  // previously wsAud's onEvent side effect
    }
  });

  // Single dispatcher for everything that used to be spread across
  // wsCam.onMessage (SET:*/SNAP:HQ text) and wsAud.onMessage (RESTART/
  // TTS:START/TTS:END text, plus binary TTS PCM chunks) — reads the 9-byte
  // header off every incoming binary WS message and routes by Type byte.
  // wsThermal and wsImu never had onMessage handlers (the server never
  // sent anything back on those sockets), so there's nothing to fold in
  // from them.
  wsMain.onMessage([](WebsocketsMessage msg){
    if (!msg.isBinary()) {
      // The server always sends framed binary messages now (even control
      // commands go out as Type 0x05) — a stray text frame shouldn't
      // happen. Logged instead of silently dropped so a protocol mismatch
      // is visible on the Serial console rather than just failing quietly.
      if (msg.isText()) {
        String txt = msg.data();
        Serial.printf("[MUX] unexpected text frame (ignored): %s\n", txt.c_str());
      }
      return;
    }

    const std::string& raw = msg.rawData();
    if (raw.size() < FRAME_HDR_LEN) {
      Serial.printf("[MUX] frame too short (%u bytes), dropping\n", (unsigned)raw.size());
      return;
    }
    const uint8_t* b = (const uint8_t*)raw.data();
    uint8_t  type         = b[0];
    uint16_t seq          = ((uint16_t)b[1] << 8) | b[2];
    uint16_t declared_len = ((uint16_t)b[3] << 8) | b[4];
    // bytes [5..8] are the sender's timestamp — not currently used on receive.
    size_t actual_len = raw.size() - FRAME_HDR_LEN;
    // See the framing comment above wsMain's declaration: actual_len (from
    // the WS message itself) is authoritative, not declared_len — a
    // mismatch is logged but never blocks processing.
    if (declared_len != actual_len) {
      Serial.printf("[MUX] type=0x%02X seq=%u: declared_len=%u != actual_len=%u (using actual)\n",
                    type, seq, declared_len, (unsigned)actual_len);
    }
    const uint8_t* payload = b + FRAME_HDR_LEN;

    switch (type) {
      case FRAME_TYPE_AUDIO: {  // 0x02 — TTS PCM chunk from server (was wsAud's binary branch)
        if (!tts_playing) return;
        if (!qTTS) {
          static bool warned = false;
          if (!warned) { Serial.println("[TTS] qTTS is NULL, dropping TTS audio"); warned = true; }
          return;
        }
        TTSChunk ch = {};
        size_t n = min(actual_len, sizeof(ch.data));
        ch.n = (uint16_t)n;
        memcpy(ch.data, payload, n);  // payload is a slice of raw (std::string) — safe for null bytes in PCM
        bool queued_ok = xQueueSend(qTTS, &ch, 0) == pdPASS;  // non-blocking; drop if queue full
        Serial.printf("[TTS] binary frame: %u bytes, tts_playing=%d, queued=%d\n", (unsigned)n, tts_playing, queued_ok);
        break;
      }

      case FRAME_TYPE_CONTROL: {  // 0x05 — text command, same strings/logic as before
        // Safe to build a String from payload via the single-arg (null-
        // terminated) constructor here specifically because payload always
        // runs to the exact end of `raw`, and std::string::data() is
        // null-terminated at its true end since C++11 — there's no
        // embedded/trailing garbage between payload's logical end and that
        // terminator for this to over-read into.
        String cmd((const char*)payload);
        cmd.trim();

        // ---- formerly wsAud's text commands ----
        if (cmd == "RESTART") {
          run_audio_stream = false; xQueueReset(qAudio); delay(50);
          sendControl("START"); run_audio_stream = true;
        } else if (cmd == "TTS:START") {
          run_audio_stream = false;   // mute mic during playback: no echo, no contention on wsMain
          xQueueReset(qAudio);        // drop any mic frames already captured
          tts_reset_queue();
          tts_playing = true;
          Serial.println("[TTS] START received, tts_playing=true");
        } else if (cmd == "TTS:END") {
          Serial.println("[TTS] END received, sentinel queued");
          if (!qTTS) {
            Serial.println("[TTS] qTTS is NULL, cannot send end-sentinel");
            tts_playing = false;      // fallback: force idle so the mic recovers
            run_audio_stream = true;
          } else {
            TTSChunk sentinel = {};  // ch.n == 0 tells taskTTSPlay the stream is done
            // Ensure the end-sentinel actually lands. If qTTS is briefly full the
            // sentinel could be dropped, leaving tts_playing stuck true and the mic
            // muted forever. Retry, then hard-recover if it still won't queue.
            int tries = 0;
            while (xQueueSend(qTTS, &sentinel, pdMS_TO_TICKS(20)) != pdPASS && tries < 10) {
              tries++;
            }
            if (tries >= 10) {
              tts_playing = false;      // fallback: force idle so the mic recovers
              run_audio_stream = true;
            }
          }
        }

        // ---- formerly wsCam's text commands ----
        else if (cmd.startsWith("SET:FRAMESIZE=")) {
          String v = cmd.substring(strlen("SET:FRAMESIZE="));
          v.toUpperCase();
          framesize_t fs = g_frame_size;
          if (v == "SVGA") fs = FRAMESIZE_SVGA;
          else if (v == "XGA") fs = FRAMESIZE_XGA;
          else if (v == "VGA") fs = FRAMESIZE_VGA;
          if (apply_framesize(fs)) Serial.printf("[CAM] framesize set to %s\n", v.c_str());
          else Serial.printf("[CAM] framesize set failed: %s\n", v.c_str());
        }
        else if (cmd.startsWith("SET:QUALITY=")) {     // Dynamic JPEG quality
          int q = cmd.substring(strlen("SET:QUALITY=")).toInt();
          q = constrain(q, 5, 40);
          sensor_t* s = esp_camera_sensor_get();
          if (s) { s->set_quality(s, q); Serial.printf("[CAM] quality=%d\n", q); }
        }
        else if (cmd.startsWith("SET:FPS=")) {         // Send throttle FPS
          int f = cmd.substring(strlen("SET:FPS=")).toInt();
          g_target_fps = (f <= 0 ? 0 : constrain(f, 5, 60));
          Serial.printf("[CAM] target_fps=%d\n", g_target_fps);
        }
        else if (cmd.startsWith("SET:AE_AUTO=")) {     // Auto-exposure on/off
          int on = cmd.substring(strlen("SET:AE_AUTO=")).toInt();
          sensor_t* s = esp_camera_sensor_get();
          if (s) { s->set_exposure_ctrl(s, on ? 1 : 0); Serial.printf("[CAM] ae_auto=%d\n", on ? 1 : 0); }
        }
        else if (cmd.startsWith("SET:AEC=")) {         // Manual exposure value (0-1200)
          int v = cmd.substring(strlen("SET:AEC=")).toInt();
          v = constrain(v, 0, 1200);
          sensor_t* s = esp_camera_sensor_get();
          if (s) { s->set_aec_value(s, v); Serial.printf("[CAM] aec=%d\n", v); }
        }
        else if (cmd.startsWith("SET:GAINCEIL=")) {    // AGC ceiling: 0=2X .. 6=128X
          int v = cmd.substring(strlen("SET:GAINCEIL=")).toInt();
          v = constrain(v, 0, 6);
          sensor_t* s = esp_camera_sensor_get();
          if (s) { s->set_gainceiling(s, (gainceiling_t)v); Serial.printf("[CAM] gainceil=%d\n", v); }
        }
        else if (cmd.startsWith("SET:AEC2=")) {        // Extended AEC (night mode) on/off
          int on = cmd.substring(strlen("SET:AEC2=")).toInt();
          sensor_t* s = esp_camera_sensor_get();
          if (s) { s->set_aec2(s, on ? 1 : 0); Serial.printf("[CAM] aec2=%d\n", on ? 1 : 0); }
        }
        else if (cmd.startsWith("SET:AE_LEVEL=")) {    // AE target bias (-2..+2)
          int v = cmd.substring(strlen("SET:AE_LEVEL=")).toInt();
          v = constrain(v, -2, 2);
          sensor_t* s = esp_camera_sensor_get();
          if (s) { s->set_ae_level(s, v); Serial.printf("[CAM] ae_level=%d\n", v); }
        }
        else if (cmd == "SNAP:HQ") {
          // Dead/orphaned feature carried over as-is: nothing in app_main.py
          // (or any other client in this repo) has ever sent "SNAP:HQ", and
          // the server never parsed the SNAP:BEGIN/SNAP:END markers below
          // either — this branch has been unreachable since it was written.
          // Preserved verbatim rather than removed, since dropping it wasn't
          // asked for. NOTE: SXGA-quality-18 snapshots routinely exceed the
          // 65535-byte frame limit (see FRAME_MAX_PAYLOAD above) — if this
          // is ever wired up for real, sendFramed() will refuse to send the
          // snapshot payload and log it rather than corrupt anything.
          Serial.println("[CAM] SNAP:HQ request");
          if (snapshot_in_progress) return;
          snapshot_in_progress = true;
          sensor_t* s = esp_camera_sensor_get();
          framesize_t old_fs = g_frame_size;
          int old_q = JPEG_QUALITY;
          // Target resolution: SXGA (increase to UXGA if PSRAM stability allows)
          framesize_t target_fs = FRAMESIZE_SXGA;
          if (s) {
            s->set_framesize(s, target_fs);
            s->set_quality(s, 18); // Lower value = higher quality
          }
          vTaskDelay(pdMS_TO_TICKS(500));
          camera_fb_t* fb = esp_camera_fb_get();
          if (fb && fb->format == PIXFORMAT_JPEG) {
            sendControl("SNAP:BEGIN");
            bool ok = sendFramed(FRAME_TYPE_CAMERA, fb->buf, fb->len);
            sendControl("SNAP:END");
            if (!ok) { Serial.println("[CAM] SNAP send failed"); }
            esp_camera_fb_return(fb);
          } else {
            if (fb) esp_camera_fb_return(fb);
            Serial.println("[CAM] SNAP: capture failed");
          }
          if (s) {
            s->set_framesize(s, old_fs);
            s->set_quality(s, old_q);
          }
          snapshot_in_progress = false;
        }
        break;
      }

      default:
        Serial.printf("[MUX] unexpected incoming type=0x%02X (seq=%u, %u bytes), ignoring\n",
                      type, seq, (unsigned)actual_len);
        break;
    }
  });

  // Initial connect sequence: each client's connectSecure() call runs to
  // completion (success or failure), plus a short heap-settle delay, before
  // the next one starts — see connectWsSequential()'s comment above. This
  // replaces four back-to-back connectSecure() calls that had no gap
  // between them.
  //
  // Note this now runs after camera init (moved to the very top of
  // setup(), see the comment there) rather than before it — so the
  // "freshest, least-fragmented heap" this block wants belongs to camera
  // init first, then this connect. loop()'s existing retry logic still
  // runs afterward as the fallback/reconnect path — if this succeeds,
  // wsMain.available() will already be true there and it simply won't
  // re-attempt.
  //
  // There's only one client to sequence now (this used to be four
  // back-to-back connectSecure() calls — wsCam/wsAud/wsThermal/wsImu, each
  // its own ~16KB+ mbedTLS buffer fighting the same fragmented heap; that
  // contention is exactly what single-socket multiplexing is meant to
  // remove), so the "sequential" part of connectWsSequential() no longer
  // has anything to sequence against — kept as-is since the function
  // itself (and the heap-settle delay it adds after the one call) is still
  // useful independent of that.
  //
  // Blocking note: connectSecure() is synchronous (TCP + TLS handshake +
  // HTTP upgrade inline) — if it hangs, setup() (and therefore I2S/queue/
  // task creation below) is delayed by however long that takes, up to the
  // 30s TCP / 120s handshake timeouts. That tradeoff is inherent to
  // attempting the connection this early; not otherwise mitigated here.
  bool main_connected = connectWsSequential(wsMain, MULTIPLEX_WS_PATH, "Main", "MAIN");
  if (main_connected) {
    delay(50);
    run_audio_stream = true;
    sendControl("START");
  }

  init_i2s_in();
  init_i2s_out();

  qFrames = xQueueCreate(3, sizeof(fb_ptr_t));  // 3 buffers to reduce frame drops
  qAudio  = xQueueCreate(AUDIO_QUEUE_DEPTH, sizeof(AudioChunk));
  qTTS    = xQueueCreate(TTS_QUEUE_DEPTH, sizeof(TTSChunk));
  if (!qTTS) {
    Serial.println("[FATAL] qTTS allocation failed - insufficient heap, retrying...");
    delay(200);
    qTTS = xQueueCreate(TTS_QUEUE_DEPTH, sizeof(TTSChunk));
    if (!qTTS) {
      Serial.println("[FATAL] qTTS allocation failed twice - insufficient heap, reboot...");
      delay(1500);
      esp_restart();
    }
  }

  i2cMutex = xSemaphoreCreateMutex();
  initI2cBus();  // bring up the shared I2C bus once, before the IMU/thermal tasks start

  xTaskCreatePinnedToCore(taskCamCapture, "cam_cap", 10240, NULL, 4, NULL, 1);
  xTaskCreatePinnedToCore(taskCamSend,    "cam_snd",  8192, NULL, 3, NULL, 1);
  xTaskCreatePinnedToCore(taskMicCapture, "mic_cap",   4096, NULL, 2, NULL, 0);
  xTaskCreatePinnedToCore(taskMicUpload,  "mic_upl",   4096, NULL, 2, NULL, 1);
  xTaskCreatePinnedToCore(taskImuLoop,    "imu_loop",  4096, NULL, 2, NULL, 0);
#if THERMAL_ENABLED
  xTaskCreatePinnedToCore(taskThermalLoop, "thermal",  8192, NULL, 1, NULL, 0);
#endif
  xTaskCreatePinnedToCore(taskTTSPlay,    "tts_play",  4096, NULL, 2, NULL, 0);
}

void loop() {
  // Guard: skip calling the real wsMain.available() when we know the client
  // was just closed — see main_ws_closed_pending_reconnect's declaration
  // comment for the crash this avoids (confirmed via a symbolicated
  // backtrace: this exact available() call reading a freed mbedTLS session,
  // right after a ConnectionClosed event). Short-circuit (||) means
  // wsMain.available() is never invoked at all on this pass when the flag
  // is set — we go straight to a fresh connectSecure() instead, which
  // constructs a new underlying client object rather than touching the
  // stale one.
  //
  // Non-blocking, cooldown-gated reconnect (rather than the old wsCam
  // path's blocking delay(1000) on failure): with four sockets, a blocking
  // retry on one client's reconnect used to stall the other three clients'
  // poll() on the same loop() pass — that's exactly why wsAud/wsThermal/
  // wsImu used this cooldown-gated style instead. Now that wsMain is the
  // only client, nothing else on this connection needs protecting from a
  // blocking retry, but there's no upside to blocking loop() either, so
  // this keeps the never-blocks version.
  static unsigned long last_main_retry = 0;
  unsigned long now_main = millis();
  // Preserves the exact short-circuit from the comment above: wsMain.available()
  // must never be called at all when main_ws_closed_pending_reconnect is
  // already true (that's the whole point of the guard) — so the mutex only
  // wraps the actual available() call, in the branch where it still runs.
  bool need_reconnect;
  if (main_ws_closed_pending_reconnect) {
    need_reconnect = true;
  } else {
    xSemaphoreTakeRecursive(wsMainMutex, portMAX_DELAY);
    need_reconnect = !wsMain.available();
    xSemaphoreGiveRecursive(wsMainMutex);
  }
  if (need_reconnect && (now_main - last_main_retry >= 2000)) {
    main_ws_closed_pending_reconnect = false;
    last_main_retry = now_main;
    Serial.printf("[DEBUG] WiFi status: %d, Free heap: %d, Max alloc heap: %d\n", WiFi.status(), ESP.getFreeHeap(), ESP.getMaxAllocHeap());
    unsigned long main_connect_t0 = millis();
    xSemaphoreTakeRecursive(wsMainMutex, portMAX_DELAY);
    bool main_connected = wsMain.connectSecure(SERVER_HOST, SERVER_PORT, MULTIPLEX_WS_PATH);
    xSemaphoreGiveRecursive(wsMainMutex);
    Serial.printf("[DEBUG] wsMain.connectSecure() took %lu ms\n", millis() - main_connect_t0);
    // NOTE: WebsocketsClient (ArduinoWebsockets 0.5.4) keeps its underlying
    // TCP/TLS client in a private std::shared_ptr<network::TcpClient> with
    // no accessor anywhere in the public API (checked src/tiny_websockets/
    // client.hpp in full) — there is no way to reach the WiFiClientSecure
    // instance through wsMain to call its lastError(char*, size_t). That
    // method genuinely exists on WiFiClientSecure/NetworkClientSecure
    // itself (checked the installed esp32 core 3.3.10 source), it's just
    // unreachable from here.
    if (main_connected) {
      Serial.println("[WS-MAIN] connected");
      delay(50);
      run_audio_stream = true;
      sendControl("START");
    } else {
      Serial.println("[WS-MAIN] retry in 2s...");
    }
  }

  // ---- Keepalive ping ----
  // ArduinoWebsockets has no built-in keepalive/ping timer. Believed cause
  // of the ~30s disconnect cycle: the phone hotspot's cellular NAT gateway
  // silently drops idle connections. This used to stagger one ping per 5s
  // tick across four clients specifically to avoid bursting four pings in
  // the same loop() iteration; with a single client there's nothing left to
  // stagger against, so it's just one ping every 5s now.
  //
  // Gated on main_ws_ready rather than calling wsMain.available() directly
  // — see main_ws_closed_pending_reconnect's declaration comment above for
  // the confirmed use-after-free crash (LoadProhibited, mbedTLS
  // ssl_parse_record_header) from calling .available() on a client whose
  // ConnectionClosed event hasn't been handled yet. main_ws_ready is
  // updated synchronously in the onEvent handler and costs nothing to
  // read, so it's the safe way to know "is this client actually up" here.
  static unsigned long last_keepalive_tick = 0;
  unsigned long now_keepalive = millis();
  if (now_keepalive - last_keepalive_tick >= 5000) {
    last_keepalive_tick = now_keepalive;
    if (main_ws_ready) {
      xSemaphoreTakeRecursive(wsMainMutex, portMAX_DELAY);
      wsMain.ping("");
      xSemaphoreGiveRecursive(wsMainMutex);
    }
  }

  // poll() synchronously invokes the onMessage/onEvent callbacks while
  // still holding this lock — they call sendFramed()/sendControl(), which
  // re-take the same mutex from the same task. That's exactly why this is
  // a recursive mutex and not a plain one (see wsMainMutex's declaration
  // comment) — a plain mutex would deadlock right here.
  xSemaphoreTakeRecursive(wsMainMutex, portMAX_DELAY);
  wsMain.poll();
  xSemaphoreGiveRecursive(wsMainMutex);
  delay(2);
}
