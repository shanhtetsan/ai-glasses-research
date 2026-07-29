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
#include "esp_heap_caps.h"  // MALLOC_CAP_SPIRAM for camThermalTxBuf
using namespace websockets;

// Strict hardware-integration mode: keep required sensor/audio paths and
// disable runtime camera mutation and experimental networking.
#define STABILITY_MODE 1

// ===== Thermal enable/disable switch =====
// Flip to 0 and reflash to fully disable the thermal subsystem (no MLX90640
// init, no thermal frames sent, no task) without touching anything else
// below, if it causes instability during testing. Camera keeps working on
// the shared wsCamThermal socket either way — see the single-socket merge
// comment near wsCamThermal's declaration below.
#define THERMAL_ENABLED 1

// IMU is always multiplexed on wsCamThermal; no third TLS client exists.

// ===== WiFi / Server =====
const char* WIFI_SSID   = "RashedAiPhone";
const char* WIFI_PASS   = "RashedA206";
const char* SERVER_HOST = "openaiglasses-thermal-test.fly.dev";
const uint16_t SERVER_PORT = 443;  // HTTPS/WSS port

// ===== Stability configuration =====
constexpr uint32_t HEALTH_LOG_INTERVAL_MS = 10000;
constexpr uint32_t HEARTBEAT_INTERVAL_MS = 15000;
constexpr uint32_t SOCKET_STALE_TIMEOUT_MS = 45000;
constexpr uint32_t CAMERA_MIN_FRAME_INTERVAL_MS = 250;  // <= 4 FPS
constexpr uint32_t THERMAL_MIN_FRAME_INTERVAL_MS = 300; // <= 3.3 FPS
constexpr uint32_t CAMERA_SEND_UNHEALTHY_MS = 2000;
constexpr uint32_t MAX_RECONNECT_BACKOFF_MS = 30000;
constexpr uint32_t CONNECTION_STABLE_RESET_MS = 30000;
constexpr size_t CRITICAL_INTERNAL_BLOCK_BYTES = 16 * 1024;
constexpr uint8_t CRITICAL_MEMORY_INTERVALS = 3;

// Camera and thermal share one connection (see wsCamThermal below) instead
// of separate /ws/camera and /ws/thermal sockets — merged to avoid the
// DMA/heap contention crashes seen running two TLS sockets' worth of
// camera+thermal traffic concurrently.
static const char* CAM_THERMAL_WS_PATH = "/ws/camera_thermal";
static const char* AUD_WS_PATH     = "/ws_audio";

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

framesize_t g_frame_size = STABILITY_MODE ? FRAMESIZE_QVGA : FRAMESIZE_VGA;
// Raised from 25: creates memory/bandwidth headroom for thermal's added
// traffic on the merged camera+thermal socket (see /ws/camera_thermal below).
// Backed off from the originally-planned 32 to 30 after testing against real
// recorded frames with the production YOLOE obstacle model: detection
// confidence/count held flat down to ~10KB/frame (VGA) but started degrading
// measurably below ~9KB, and 32 sits close enough to the upper edge of this
// codebase's accepted quality range (SET:QUALITY clamps to 40 max) that it
// risked landing in that degraded zone without hardware-level confirmation.
#define JPEG_QUALITY  16
#define FB_COUNT      2
volatile int g_target_fps = 4;


volatile unsigned long frame_captured_count = 0;
volatile unsigned long frame_sent_count = 0;
volatile unsigned long frame_dropped_count = 0;
volatile unsigned long last_stats_time = 0;
volatile unsigned long ws_send_fail_count = 0;

// ===== Thermal (MLX90640) =====
// Ported from feature/thermal-stability-fix as pure sensor driver code.
// Transport is NOT ported from that branch (it used HTTP POST) — instead
// frames are sent over the shared wsCamThermal socket below (1-byte
// MSG_TYPE_THERMAL prefix), matching app_main.py's merged /ws/camera_thermal
// contract.
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
#define THERMAL_RESOLUTION_CODE 0x02  // fixed 18-bit ADC resolution
#define THERMAL_READ_INTERVAL_MS THERMAL_MIN_FRAME_INTERVAL_MS

// Guards the physical I2C bus, shared between the IMU (MPU-6050) and the
// MLX90640 — both are plain Wire peripherals on the same pins, and without
// this mutex their two FreeRTOS tasks can interleave I2C transactions and
// corrupt each other (the class of bug feature/thermal-stability-fix's own
// name refers to). Declared unconditionally (the IMU helpers below always
// use it, regardless of THERMAL_ENABLED); created once in setup().
SemaphoreHandle_t i2cMutex;

#if THERMAL_ENABLED
paramsMLX90640 mlx90640;
constexpr size_t THERMAL_PIXEL_COUNT = 32 * 24;
constexpr size_t THERMAL_PIXEL_BYTES = THERMAL_PIXEL_COUNT * sizeof(float);
static float* thermalPixels = nullptr; // PSRAM, allocated and validated in setup()
bool thermalReady = false;

bool initThermal() {
  Serial.println("[THERMAL] Initializing...");

  if (!xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(500))) {
    Serial.println("[THERMAL] I2C mutex timeout during init");
    return false;
  }

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

  int refreshStatus = MLX90640_SetRefreshRate(THERMAL_ADDR, THERMAL_REFRESH_RATE_CODE);
  int resolutionStatus = MLX90640_SetResolution(THERMAL_ADDR, THERMAL_RESOLUTION_CODE);
  int modeStatus = MLX90640_SetChessMode(THERMAL_ADDR);

  xSemaphoreGive(i2cMutex);

  if (refreshStatus != 0 || resolutionStatus != 0 || modeStatus != 0) {
    Serial.printf("[THERMAL] fixed configuration failed refresh=%d resolution=%d mode=%d\n",
                  refreshStatus, resolutionStatus, modeStatus);
    return false;
  }

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
// Camera and thermal share this one socket (merged from formerly-separate
// wsCam/wsThermal) — see CAM_THERMAL_WS_PATH above. Every message sent on it
// starts with a 1-byte type prefix so the server can dispatch without a
// second socket: MSG_TYPE_CAM for a JPEG frame, MSG_TYPE_THERMAL for a raw
// float32 thermal block.
#define MSG_TYPE_CAM      0x01
#define MSG_TYPE_THERMAL  0x02
#define MSG_TYPE_IMU      0x03
#define MSG_TYPE_STATUS   0x04

WebsocketsClient wsCamThermal;
WebsocketsClient wsAud;
volatile bool cam_thermal_ws_ready = false;
volatile bool aud_ws_ready = false;
// Set when ConnectionClosed fires for wsCamThermal; cleared the moment loop()
// acts on it. See the guard in loop() — crash evidence (symbolicated
// backtrace, tonight's flash test) showed wsCam.available() itself crashing
// (LoadProhibited, mbedTLS ssl_parse_record_header) when called right after
// a close, because NetworkClientSecure::write()'s own internal error path
// already tore down (freed) the mbedTLS session without going through
// WebsocketsClient's close()/event machinery — the wrapper's available()
// then dereferences that freed session. This flag skips calling the real
// available() on a client known to be in that state and goes straight to a
// fresh connectSecure(), which constructs a brand-new underlying client
// object (upgradeToSecuredConnection() unconditionally `new`s one) rather
// than touching the stale one.
volatile bool cam_thermal_ws_closed_pending_reconnect = false;
volatile bool aud_ws_closed_pending_reconnect = false;
volatile bool snapshot_in_progress = false; // Pause live capture during a high-res snapshot

// taskCamSend is the sole wsCamThermal owner. The thermal and camera tasks
// only produce into bounded queues; callbacks execute synchronously from
// taskCamSend's poll(), so no cross-core WebSocket access is possible.
// Scratch buffer for building "1-byte type prefix + payload" messages before
// handing them to sendBinary(), which needs one contiguous buffer per WS
// frame. Sized to comfortably cover both a regular VGA preview JPEG
// (typically well under 50KB at JPEG_QUALITY=30) and a SNAP:HQ high-res
// (SXGA) capture, which can run well over 100KB. Allocated from PSRAM in
// setup() — trivial relative to the several MB available there.
#define CAM_TX_BUF_MAX (250 * 1024)
static uint8_t* camThermalTxBuf = nullptr;


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
#if THERMAL_ENABLED
typedef struct { uint8_t data[1 + THERMAL_PIXEL_BYTES]; } ThermalChunk;
QueueHandle_t qThermal;
#endif
typedef struct __attribute__((packed)) {
  uint32_t sequence;
  uint32_t uptimeMs;
  float accelX;  // m/s^2
  float accelY;
  float accelZ;
  float gyroX;   // degrees/second
  float gyroY;
  float gyroZ;
} ImuPacket;
static_assert(sizeof(ImuPacket) == 32, "IMU wire packet must remain 32 bytes");
typedef struct __attribute__((packed)) {
  uint32_t uptimeMs;
  uint32_t freeHeap;
  uint32_t largestInternal;
  uint32_t freePsram;
} StatusPacket;
static_assert(sizeof(StatusPacket) == 16, "status wire packet must remain 16 bytes");
QueueHandle_t qImu;
volatile bool tts_playing = false;

I2SClass i2sIn;   // PDM RX (Mic)
I2SClass i2sOut;  // STD TX (Speaker)
volatile bool run_audio_stream = false;

TaskHandle_t camCaptureTaskHandle = nullptr;
TaskHandle_t camNetworkTaskHandle = nullptr;
TaskHandle_t micCaptureTaskHandle = nullptr;
TaskHandle_t audioNetworkTaskHandle = nullptr;
TaskHandle_t thermalTaskHandle = nullptr;
TaskHandle_t ttsPlaybackTaskHandle = nullptr;
TaskHandle_t imuTaskHandle = nullptr;

volatile uint32_t camReconnectAttempts = 0;
volatile uint32_t audReconnectAttempts = 0;
volatile uint32_t lastCameraSendMs = 0;
volatile uint32_t lastMicSendMs = 0;
volatile uint32_t lastThermalSendMs = 0;
volatile uint32_t lastSpeakerPacketMs = 0;
volatile uint32_t camLastTrafficMs = 0;
volatile uint32_t audLastTrafficMs = 0;
volatile uint32_t micDroppedChunks = 0;
volatile uint32_t ttsDroppedChunks = 0;
volatile uint32_t thermalDroppedFrames = 0;
volatile uint32_t thermalSentFrames = 0;
volatile uint32_t imuSentPackets = 0;
volatile uint32_t imuDroppedPackets = 0;
volatile uint32_t micCapturedChunks = 0;
volatile uint32_t micSentChunks = 0;
volatile uint32_t speakerQueuedChunks = 0;
volatile uint32_t speakerPlayedChunks = 0;
volatile bool audioStartPending = false;
volatile bool controlledRestartRequested = false;
volatile bool setupComplete = false;

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
    if (xQueueSend(qFrames, &fb, 0) != pdPASS) {
      // Ownership never becomes ambiguous: if the newest frame cannot be
      // queued after evicting the old one, return it here exactly once.
      esp_camera_fb_return(fb);
      frame_dropped_count++;
      Serial.println("[CAM] latest-frame enqueue failed; returned new framebuffer");
    }
  }
}

void taskCamCapture(void*) {
  unsigned long last_log = 0;
  unsigned long capture_fail_count = 0;
  
  for(;;){
    if (snapshot_in_progress) { vTaskDelay(pdMS_TO_TICKS(5)); continue; }

    if (cam_thermal_ws_ready) {
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

static uint32_t reconnectDelayMs(uint32_t attempt) {
  uint32_t shift = min(attempt, (uint32_t)5);
  uint32_t base = min(1000UL << shift, MAX_RECONNECT_BACKOFF_MS);
  return min(base + (uint32_t)random(0, 251), MAX_RECONNECT_BACKOFF_MS);
}

static void closeCamSocket(const char* reason) {
  Serial.printf("[WS-CAM] closing reason=%s\n", reason);
  cam_thermal_ws_ready = false;
  wsCamThermal.close();
  cam_thermal_ws_closed_pending_reconnect = true;
}

// Sole owner of wsCamThermal: connect, poll, ping, close, camera sends,
// thermal sends, and callback-driven SNAP sends all execute on this task.
void taskCamSend(void*) {
  uint32_t nextReconnectMs = 0;
  uint32_t connectedSinceMs = 0;
  uint32_t nextPingMs = 0;
  uint32_t lastStatusMs = 0;
  uint32_t lastCamAttemptMs = 0;
#if THERMAL_ENABLED
  static ThermalChunk thermal;
#endif
  ImuPacket imuPacket;
  uint8_t imuWire[1 + sizeof(ImuPacket)];
  uint8_t statusWire[1 + sizeof(StatusPacket)];

  for (;;) {
    if (!setupComplete) { vTaskDelay(pdMS_TO_TICKS(20)); continue; }
    if (controlledRestartRequested) {
      if (cam_thermal_ws_ready) closeCamSocket("controlled-restart");
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }
    uint32_t now = millis();
    if (!cam_thermal_ws_ready) {
      fb_ptr_t stale = nullptr;
      while (xQueueReceive(qFrames, &stale, 0) == pdPASS)
        if (stale) { esp_camera_fb_return(stale); frame_dropped_count++; }
#if THERMAL_ENABLED
      xQueueReset(qThermal);
#endif
      if (cam_thermal_ws_closed_pending_reconnect) {
        cam_thermal_ws_closed_pending_reconnect = false;
        camReconnectAttempts++;
        nextReconnectMs = now + reconnectDelayMs(camReconnectAttempts - 1);
      }
      // Audio reconnect and uplink are intentionally allowed to recover first.
      if (aud_ws_ready && (int32_t)(now - nextReconnectMs) >= 0) {
        Serial.printf("[WS-CAM] reconnect attempt=%lu heap=%u max=%u\n",
                      (unsigned long)(camReconnectAttempts + 1),
                      ESP.getFreeHeap(), ESP.getMaxAllocHeap());
        uint32_t started = millis();
        bool ok = wsCamThermal.connectSecure(SERVER_HOST, SERVER_PORT, CAM_THERMAL_WS_PATH);
        if (ok) {
          connectedSinceMs = millis();
          camLastTrafficMs = connectedSinceMs;
          nextPingMs = connectedSinceMs + HEARTBEAT_INTERVAL_MS + 4000;
          Serial.printf("[WS-CAM] connected in %lu ms\n", millis() - started);
        } else {
          camReconnectAttempts++;
          uint32_t waitMs = reconnectDelayMs(camReconnectAttempts - 1);
          nextReconnectMs = millis() + waitMs;
          Serial.printf("[WS-CAM] reconnect failed; retry_ms=%lu\n", (unsigned long)waitMs);
        }
      }
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }

    wsCamThermal.poll();
    now = millis();
    if (connectedSinceMs && now - connectedSinceMs >= CONNECTION_STABLE_RESET_MS)
      camReconnectAttempts = 0;
    if ((int32_t)(now - nextPingMs) >= 0) {
      if (!wsCamThermal.ping("")) { closeCamSocket("heartbeat-send-failed"); continue; }
      nextPingMs = now + HEARTBEAT_INTERVAL_MS;
    }
    if (camLastTrafficMs && now - camLastTrafficMs >= SOCKET_STALE_TIMEOUT_MS) {
      closeCamSocket("stale");
      continue;
    }
    if (now - lastStatusMs >= HEALTH_LOG_INTERVAL_MS) {
      StatusPacket status = {
        now,
        ESP.getFreeHeap(),
        heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
        ESP.getFreePsram(),
      };
      statusWire[0] = MSG_TYPE_STATUS;
      memcpy(statusWire + 1, &status, sizeof(status));
      uint32_t statusStarted = millis();
      bool statusOk = wsCamThermal.sendBinary((const char*)statusWire, sizeof(statusWire));
      uint32_t statusElapsed = millis() - statusStarted;
      if (!statusOk || statusElapsed >= CAMERA_SEND_UNHEALTHY_MS) {
        closeCamSocket(statusOk ? "status-send-unhealthy" : "status-send-failed");
        continue;
      }
      lastStatusMs = now;
      camLastTrafficMs = millis();
    }

#if THERMAL_ENABLED
    // Audio remains higher priority; thermal goes before the larger camera frame.
    if (xQueueReceive(qThermal, &thermal, 0) == pdPASS) {
      uint32_t thermalStarted = millis();
      bool ok = wsCamThermal.sendBinary((const char*)thermal.data, sizeof(thermal.data));
      uint32_t thermalElapsed = millis() - thermalStarted;
      if (!ok || thermalElapsed >= CAMERA_SEND_UNHEALTHY_MS) {
        thermalDroppedFrames++;
        closeCamSocket(ok ? "thermal-send-unhealthy" : "thermal-send-failed");
        continue;
      }
      lastThermalSendMs = camLastTrafficMs = millis();
      thermalSentFrames++;
    }
#endif

    if (xQueueReceive(qImu, &imuPacket, 0) == pdPASS) {
      imuWire[0] = MSG_TYPE_IMU;
      memcpy(imuWire + 1, &imuPacket, sizeof(imuPacket));
      uint32_t started = millis();
      bool ok = wsCamThermal.sendBinary((const char*)imuWire, sizeof(imuWire));
      uint32_t elapsed = millis() - started;
      if (!ok || elapsed >= CAMERA_SEND_UNHEALTHY_MS) {
        imuDroppedPackets++;
        closeCamSocket(ok ? "imu-send-unhealthy" : "imu-send-failed");
        continue;
      }
      imuSentPackets++;
      camLastTrafficMs = millis();
    }

    fb_ptr_t fb = nullptr;
    if (now - lastCamAttemptMs >= CAMERA_MIN_FRAME_INTERVAL_MS &&
        xQueueReceive(qFrames, &fb, 0) == pdPASS) {
      lastCamAttemptMs = now;
      bool ok = false;
      uint32_t sendStarted = millis();
      if (fb && fb->len + 1 <= CAM_TX_BUF_MAX) {
        camThermalTxBuf[0] = MSG_TYPE_CAM;
        memcpy(camThermalTxBuf + 1, fb->buf, fb->len);
        size_t wireLen = fb->len + 1;
        esp_camera_fb_return(fb);
        fb = nullptr; // TLS may block, but the camera driver no longer owns this wait.
        ok = wsCamThermal.sendBinary((const char*)camThermalTxBuf, wireLen);
      } else if (fb) {
        Serial.printf("[CAM] frame too large bytes=%u; dropping\n", fb->len);
      }
      uint32_t elapsed = millis() - sendStarted;
      if (fb) esp_camera_fb_return(fb);
      if (ok) {
        frame_sent_count++;
        lastCameraSendMs = camLastTrafficMs = millis();
      } else {
        ws_send_fail_count++;
      }
      // ArduinoWebsockets 0.5.4 exposes no socket/write-timeout setter.
      // This detects a blocked TLS write only after it returns; it cannot
      // interrupt the call. The owner closes and drops the stale frame then.
      if (elapsed >= CAMERA_SEND_UNHEALTHY_MS) {
        Serial.printf("[CAM] unhealthy send elapsed_ms=%lu; reconnecting\n", elapsed);
        closeCamSocket("camera-send-unhealthy");
        continue;
      }
      if (!ok) { closeCamSocket("camera-send-failed"); continue; }
    }
    vTaskDelay(pdMS_TO_TICKS(2));
  }
}
// ====================================================================
// Mic (PDM RX)
// ====================================================================
bool init_i2s_in(){
  i2sIn.setPinsPdmRx(I2S_MIC_CLOCK_PIN, I2S_MIC_DATA_PIN);
  if (!i2sIn.begin(I2S_MODE_PDM_RX, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("[I2S IN] init failed");
    return false;
  }
  Serial.println("[I2S IN] PDM RX @16kHz 16bit MONO ready");
  return true;
}

void taskMicCapture(void*){
  const int samples_per_chunk = BYTES_PER_CHUNK / 2; // int16
  for(;;){
    if (run_audio_stream) {
      AudioChunk ch; ch.n = BYTES_PER_CHUNK;
      int16_t* out = reinterpret_cast<int16_t*>(ch.data);
      int i = 0;
      while (i < samples_per_chunk){
        int v = i2sIn.read();
        if (v == -1) { delay(1); continue; }
        out[i++] = (int16_t)v;
      }
      micCapturedChunks++;
      if (xQueueSend(qAudio, &ch, 0) != pdPASS){
        AudioChunk dump;
        if (xQueueReceive(qAudio, &dump, 0) == pdPASS) micDroppedChunks++;
        if (xQueueSend(qAudio, &ch, 0) != pdPASS) micDroppedChunks++;
        if ((micDroppedChunks % 50) == 1)
          Serial.printf("[MIC] queue full; dropped_oldest total=%lu\n", micDroppedChunks);
      }
    } else {
      vTaskDelay(pdMS_TO_TICKS(5));
    }
  }
}

void taskMicUpload(void*){
  uint32_t nextReconnectMs = 0;
  uint32_t connectedSinceMs = 0;
  uint32_t lastPingMs = 0;
  for(;;){
    if (!setupComplete) { vTaskDelay(pdMS_TO_TICKS(20)); continue; }
    if (controlledRestartRequested) {
      if (aud_ws_ready) { aud_ws_ready = false; wsAud.close(); }
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }
    uint32_t now = millis();
    if (!aud_ws_ready) {
      if (aud_ws_closed_pending_reconnect) {
        aud_ws_closed_pending_reconnect = false;
        audReconnectAttempts++;
        nextReconnectMs = now + reconnectDelayMs(audReconnectAttempts - 1);
      }
      if ((int32_t)(now - nextReconnectMs) >= 0) {
        Serial.printf("[WS-AUD] reconnect attempt=%lu heap=%u max=%u\n",
                      (unsigned long)(audReconnectAttempts + 1),
                      ESP.getFreeHeap(), ESP.getMaxAllocHeap());
        uint32_t started = millis();
        bool ok = wsAud.connectSecure(SERVER_HOST, SERVER_PORT, AUD_WS_PATH);
        if (ok) {
          connectedSinceMs = millis();
          audLastTrafficMs = connectedSinceMs;
          lastPingMs = connectedSinceMs;
          audioStartPending = true;
          Serial.printf("[WS-AUD] connected in %lu ms\n", millis() - started);
        } else {
          audReconnectAttempts++;
          uint32_t waitMs = reconnectDelayMs(audReconnectAttempts - 1);
          nextReconnectMs = millis() + waitMs;
          Serial.printf("[WS-AUD] reconnect failed; retry_ms=%lu\n", (unsigned long)waitMs);
        }
      }
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }

    wsAud.poll();
    now = millis();
    if (connectedSinceMs && now - connectedSinceMs >= CONNECTION_STABLE_RESET_MS)
      audReconnectAttempts = 0;
    if (audioStartPending) {
      audioStartPending = false;
      uint32_t startStarted = millis();
      bool startOk = wsAud.send("START");
      uint32_t startElapsed = millis() - startStarted;
      if (!startOk || startElapsed >= CAMERA_SEND_UNHEALTHY_MS) {
        Serial.println("[WS-AUD] START send failed");
        aud_ws_ready = false;
        wsAud.close();
        continue;
      }
      run_audio_stream = true;
      audLastTrafficMs = millis();
    }
    if (now - lastPingMs >= HEARTBEAT_INTERVAL_MS) {
      if (!wsAud.ping("")) {
        Serial.println("[WS-AUD] heartbeat send failed; reconnecting");
        aud_ws_ready = false; wsAud.close(); continue;
      }
      lastPingMs = now;
    }
    if (audLastTrafficMs && now - audLastTrafficMs >= SOCKET_STALE_TIMEOUT_MS) {
      Serial.println("[WS-AUD] stale; reconnecting");
      aud_ws_ready = false; wsAud.close(); continue;
    }
    if (run_audio_stream && !tts_playing) {
      AudioChunk ch;
      if (xQueueReceive(qAudio, &ch, 0) == pdPASS) {
        uint32_t audioStarted = millis();
        bool audioOk = wsAud.sendBinary((const char*)ch.data, ch.n);
        uint32_t audioElapsed = millis() - audioStarted;
        if (!audioOk || audioElapsed >= CAMERA_SEND_UNHEALTHY_MS) {
          Serial.printf("[MIC] upload failed bytes=%u; reconnecting\n", (unsigned)ch.n);
          micDroppedChunks++;
          aud_ws_ready = false; wsAud.close(); continue;
        }
        lastMicSendMs = audLastTrafficMs = millis();
        micSentChunks++;
      }
    }
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

// ====================================================================
// Speaker (I2S TX) + HTTP /stream.wav (chunked-safe)
// ====================================================================
bool init_i2s_out(){
  i2sOut.setPins(I2S_SPK_BCLK, I2S_SPK_LRC, I2S_SPK_DIN);
  if (!i2sOut.begin(I2S_MODE_STD, TTS_RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO)) {
    Serial.println("[I2S OUT] init failed");
    return false;
  }
  Serial.printf("[I2S OUT] STD TX @%dHz 32bit STEREO ready\n", TTS_RATE);
  return true;
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
  // this is a plain HTTP GET of /stream.wav, unrelated to wsCamThermal/wsAud/
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
  BaseType_t result = xTaskCreatePinnedToCore(
      taskHttpPlay, "http_wav", 8192, nullptr, 2, &taskHttpPlayHandle, 0);
  if (result != pdPASS || !taskHttpPlayHandle) {
    Serial.printf("[FATAL] component=HTTP-AUDIO task create result=%ld handle=%p heap=%u max=%u\n",
                  (long)result, taskHttpPlayHandle, ESP.getFreeHeap(), ESP.getMaxAllocHeap());
    delay(500);
    esp_restart();
  }
  Serial.printf("[AUDIO] http_wav task started result=%ld\n", (long)result);
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
        Serial.println("[TTS] playback complete");
        tts_playing = false;
        run_audio_stream = true;           // playback truly finished — un-mute the mic
        first_chunk_pending = true;        // next session should log its first chunk again
        continue;
      }
      speakerPlayedChunks++;
      if (first_chunk_pending) {
        Serial.printf("[TTS] playback start bytes=%u\n", ch.n);
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
            if (wrote == 0) {
              Serial.println("[TTS] I2S write failure wrote=0");
              vTaskDelay(pdMS_TO_TICKS(1));
            } else off += wrote;
          }
          outPairs = 0;
        }
      }
      if (outPairs){
        size_t bytes = outPairs * 2 * sizeof(int32_t);
        size_t off = 0;
        while (off < bytes){
          size_t wrote = i2sOut.write((uint8_t*)stereo32Buf + off, bytes - off);
          if (wrote == 0) {
            Serial.println("[TTS] I2S write failure wrote=0");
            vTaskDelay(pdMS_TO_TICKS(1));
          } else off += wrote;
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
  if (!xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(50))) {
    Serial.println("[IMU] I2C mutex timeout write");
    return;
  }
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
  xSemaphoreGive(i2cMutex);
}

static uint8_t mpu_read1(uint8_t reg) {
  if (!xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(50))) {
    Serial.println("[IMU] I2C mutex timeout read1");
    return 0xFF;
  }
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);  // repeated-START keeps bus active for the read
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1);
  uint8_t v = Wire.available() ? Wire.read() : 0xFF;
  xSemaphoreGive(i2cMutex);
  return v;
}

static void mpu_read14(uint8_t* dst) {
  if (!xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(50))) {
    Serial.println("[IMU] I2C mutex timeout read14");
    memset(dst, 0, 14);
    return;
  }
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
  uint32_t sequence = 0;
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

    ImuPacket packet = {
      ++sequence, millis(), ax_f, ay_f, az_f, gx, gy, gz
    };
    if (xQueueOverwrite(qImu, &packet) != pdPASS) imuDroppedPackets++;
    vTaskDelay(pdMS_TO_TICKS(100)); // Fixed 10 Hz stability cadence.
  }
}

#if THERMAL_ENABLED
// ====================================================================
// Thermal (MLX90640) — sensor read + wsCamThermal send
// ====================================================================
// Read cadence/refresh rate ported from thermal-stability-fix's
// taskThermalLoop() (init sequence, MLX90640_GetFrameData/GetTa/CalculateTo
// calls), but re-throttled to THERMAL_READ_INTERVAL_MS (~6.7 Hz, was 3000ms
// there) and re-wired to send over the shared wsCamThermal socket (1-byte
// MSG_TYPE_THERMAL prefix, mutex-protected — see wsCamThermalMutex's
// declaration comment for why: this task runs pinned to core 0 while
// taskCamSend runs pinned to core 1, both writing the same socket) instead
// of an HTTP POST.
void taskThermalLoop(void* pv) {
  uint32_t consecutiveReadFailures = 0;
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
        consecutiveReadFailures++;
        Serial.printf("[THERMAL] frame read failed status=%d consecutive=%lu\n",
                      status, (unsigned long)consecutiveReadFailures);
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
    consecutiveReadFailures = 0;

    static ThermalChunk chunk;
    chunk.data[0] = MSG_TYPE_THERMAL;
    memcpy(chunk.data + 1, thermalPixels, THERMAL_PIXEL_BYTES);
    if (xQueueOverwrite(qThermal, &chunk) != pdPASS) {
      thermalDroppedFrames++;
      Serial.printf("[THERMAL] latest-frame queue failed dropped=%lu\n",
                    (unsigned long)thermalDroppedFrames);
    }

    vTaskDelay(pdMS_TO_TICKS(THERMAL_MIN_FRAME_INTERVAL_MS));
  }
}
#endif  // THERMAL_ENABLED

// ====================================================================
// Setup / Loop
// ====================================================================

QueueHandle_t createPsramQueue(UBaseType_t depth, UBaseType_t itemSize, const char* name) {
  uint8_t* storage = (uint8_t*)heap_caps_calloc(depth, itemSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  StaticQueue_t* control = (StaticQueue_t*)heap_caps_calloc(
      1, sizeof(StaticQueue_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!storage || !control) {
    Serial.printf("[FATAL] queue backing allocation failed name=%s storage=%p control=%p\n",
                  name, storage, control);
    return nullptr;
  }
  QueueHandle_t q = xQueueCreateStatic(depth, itemSize, storage, control);
  if (!q) Serial.printf("[FATAL] xQueueCreateStatic failed name=%s\n", name);
  return q;
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
  wsCamThermal.setCACert(FLY_ROOT_CA);
  wsAud.setCACert(FLY_ROOT_CA);
  Serial.printf("[DEBUG] FLY_ROOT_CA length: %d bytes (sanity check the PROGMEM string is intact — not a parse/verify result, setCACert() has none)\n", strlen(FLY_ROOT_CA));

  wsCamThermal.onEvent([](WebsocketsEvent ev, String){
    if (ev == WebsocketsEvent::ConnectionOpened)  {
      cam_thermal_ws_ready = true;
      camLastTrafficMs = millis();
      Serial.println("[WS-CAM-THERMAL] open");
      // Reset statistics
      frame_sent_count = 0;
      frame_dropped_count = 0;
      ws_send_fail_count = 0;
      last_stats_time = millis();
    }
    if (ev == WebsocketsEvent::ConnectionClosed)  {
      cam_thermal_ws_ready = false;
      cam_thermal_ws_closed_pending_reconnect = true;
      Serial.printf("[WS-CAM-THERMAL] closed (sent=%lu, dropped=%lu, fail=%lu)\n",
                    frame_sent_count, frame_dropped_count, ws_send_fail_count);
    }
    if (ev == WebsocketsEvent::GotPing || ev == WebsocketsEvent::GotPong)
      camLastTrafficMs = millis();
  });

  wsCamThermal.onMessage([](WebsocketsMessage msg){
    camLastTrafficMs = millis();
#if STABILITY_MODE
    if (msg.isText()) {
      Serial.println("[CAM] runtime command ignored in STABILITY_MODE");
    }
    return;
#else
    if (msg.isText()){
      String cmd = msg.data(); cmd.trim();
      if (cmd.startsWith("SET:FRAMESIZE=")) {
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
          bool ok = false;
          if (fb->len + 1 > CAM_TX_BUF_MAX) {
            Serial.printf("[CAM] SNAP too large for tx buffer (%u bytes)\n", fb->len);
          } else {
            wsCamThermal.send("SNAP:BEGIN");
            camThermalTxBuf[0] = MSG_TYPE_CAM;
            memcpy(camThermalTxBuf + 1, fb->buf, fb->len);
            ok = wsCamThermal.sendBinary((const char*)camThermalTxBuf, fb->len + 1);
            wsCamThermal.send("SNAP:END");
          }
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
    }
#endif
  });

  wsAud.onEvent([](WebsocketsEvent ev, String){
    if (ev == WebsocketsEvent::ConnectionOpened)  {
      aud_ws_ready = true;
      audLastTrafficMs = millis();
      Serial.println("[WS-AUD] open");
    }
    if (ev == WebsocketsEvent::ConnectionClosed)  {
      aud_ws_ready = false;
      aud_ws_closed_pending_reconnect = true;
      Serial.println("[WS-AUD] closed");
      stopStreamWav();
    }
    if (ev == WebsocketsEvent::GotPing || ev == WebsocketsEvent::GotPong)
      audLastTrafficMs = millis();
  });

  wsAud.onMessage([](WebsocketsMessage msg){
    audLastTrafficMs = millis();
    if (msg.isText()){
      String s = msg.data(); s.trim();
      if (s == "RESTART"){
        run_audio_stream = false;
        xQueueReset(qAudio);
        audioStartPending = true; // owner task sends START after callback returns
      } else if (s == "TTS:START") {
        run_audio_stream = false;   // mute mic during playback: no echo, no wsAud contention
        xQueueReset(qAudio);        // drop any mic frames already captured
        tts_reset_queue();
        tts_playing = true;
        Serial.println("[TTS] START received, tts_playing=true");
      } else if (s == "TTS:END") {
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
          if (xQueueSend(qTTS, &sentinel, 0) != pdPASS) {
            TTSChunk dropped;
            xQueueReceive(qTTS, &dropped, 0);
            if (xQueueSend(qTTS, &sentinel, 0) != pdPASS) {
              Serial.println("[TTS] end sentinel queue failure; forcing recovery");
              ttsDroppedChunks++;
            }
            tts_playing = false;      // fallback: force idle so the mic recovers
            run_audio_stream = true;
          }
        }
      }
    } else if (msg.isBinary()) {
      if (!tts_playing) return;
      if (!qTTS) {
        static bool warned = false;
        if (!warned) { Serial.println("[TTS] qTTS is NULL, dropping TTS audio"); warned = true; }
        return;
      }
      static TTSChunk ch;
      memset(&ch, 0, sizeof(ch));
      size_t n = min((size_t)msg.length(), sizeof(ch.data));
      ch.n = (uint16_t)n;
      memcpy(ch.data, msg.rawData().c_str(), n);  // rawData() is std::string — safe for null bytes in PCM
      bool queued_ok = xQueueSend(qTTS, &ch, 0) == pdPASS;  // non-blocking; drop if queue full
      lastSpeakerPacketMs = millis();
      if (queued_ok) {
        speakerQueuedChunks++;
        if ((speakerQueuedChunks % 25) == 1)
          Serial.printf("[TTS] queued=%lu depth=%u\n",
                        (unsigned long)speakerQueuedChunks,
                        (unsigned)uxQueueMessagesWaiting(qTTS));
      } else {
        ttsDroppedChunks++;
        Serial.printf("[TTS] queue overflow dropped=%lu depth=%u\n",
                      (unsigned long)ttsDroppedChunks,
                      (unsigned)uxQueueMessagesWaiting(qTTS));
      }
    }
  });

// Thermal has no separate onEvent registration — it shares wsCamThermal's
// connection lifecycle (open/close) with the camera, handled by the single
// wsCamThermal.onEvent() above. It also has no onMessage handler: the server
// (dispatched by MSG_TYPE from the merged /ws/camera_thermal handler) never
// sends anything back for thermal frames specifically, only camera SET:*
// commands, which wsCamThermal.onMessage() above already handles.

  // Network owner tasks below perform all initial connections and reconnects.
  // Starting audio first preserves heap and service priority; camera waits
  // until the audio socket is ready before attempting its TLS handshake.

  if (!init_i2s_in() || !init_i2s_out()) {
    Serial.printf("[FATAL] I2S initialization failed heap=%u max=%u\n",
                  ESP.getFreeHeap(), ESP.getMaxAllocHeap());
    delay(1500);
    esp_restart();
  }

  // Live video should retain only the newest frame. Keeping three camera
  // framebuffer pointers allows stale frames to occupy scarce resources while
  // the TLS sender is blocked, so use a single-slot latest-frame queue.
  qFrames = xQueueCreate(1, sizeof(fb_ptr_t));
  qImu    = xQueueCreate(1, sizeof(ImuPacket));
  qAudio  = createPsramQueue(AUDIO_QUEUE_DEPTH, sizeof(AudioChunk), "audio");
  qTTS    = createPsramQueue(TTS_QUEUE_DEPTH, sizeof(TTSChunk), "tts");
#if THERMAL_ENABLED
  qThermal = createPsramQueue(1, sizeof(ThermalChunk), "thermal");
#endif

  if (!qFrames || !qImu || !qAudio || !qTTS
#if THERMAL_ENABLED
      || !qThermal
#endif
  ) {
    Serial.printf("[FATAL] Queue creation failed qFrames=%p qImu=%p qAudio=%p qTTS=%p"
#if THERMAL_ENABLED
                  " qThermal=%p"
#endif
                  "\n", qFrames, qImu, qAudio, qTTS
#if THERMAL_ENABLED
                  , qThermal
#endif
    );
    Serial.printf("[FATAL] heap=%u max=%u internalLargest=%u psram=%u\n",
                  ESP.getFreeHeap(), ESP.getMaxAllocHeap(),
                  heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                  ESP.getFreePsram());
    delay(1500);
    esp_restart();
  }

  i2cMutex = xSemaphoreCreateMutex();
  if (!i2cMutex) {
    Serial.println("[FATAL] i2cMutex creation failed, rebooting...");
    delay(1500);
    esp_restart();
  }
  initI2cBus();  // bring up the shared I2C bus once, before IMU/thermal tasks

  camThermalTxBuf = (uint8_t*)heap_caps_malloc(CAM_TX_BUF_MAX, MALLOC_CAP_SPIRAM);
  if (!camThermalTxBuf) {
    Serial.println("[FATAL] camThermalTxBuf PSRAM allocation failed, rebooting...");
    delay(1500);
    esp_restart();
  }
#if THERMAL_ENABLED
  thermalPixels = (float*)heap_caps_malloc(THERMAL_PIXEL_BYTES, MALLOC_CAP_SPIRAM);
  if (!thermalPixels) {
    Serial.printf("[FATAL] thermal pixel PSRAM allocation failed bytes=%u heap=%u psram=%u\n",
                  (unsigned)THERMAL_PIXEL_BYTES, ESP.getFreeHeap(), ESP.getFreePsram());
    delay(1500);
    esp_restart();
  }
#endif

  Serial.printf("[HEAP] before task creation: free=%u max_alloc=%u internal_largest=%u\n",
                ESP.getFreeHeap(),
                ESP.getMaxAllocHeap(),
                heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));

  // Reduced stacks preserve contiguous internal SRAM for WiFi/mbedTLS while
  // remaining conservative for the work performed by each task.
  BaseType_t cam_cap_task_ok = xTaskCreatePinnedToCore(
      taskCamCapture, "cam_cap", 6144, NULL, 4, &camCaptureTaskHandle, 1);
  BaseType_t cam_send_task_ok = xTaskCreatePinnedToCore(
      taskCamSend, "cam_net", 6144, NULL, 3, &camNetworkTaskHandle, 1);
  BaseType_t mic_cap_task_ok = xTaskCreatePinnedToCore(
      taskMicCapture, "mic_cap", 3072, NULL, 4, &micCaptureTaskHandle, 0);
  BaseType_t mic_upload_task_ok = xTaskCreatePinnedToCore(
      taskMicUpload, "aud_net", 4096, NULL, 5, &audioNetworkTaskHandle, 1);

  // IMU is required and produces latest-only packets for the sensor owner.
  BaseType_t imu_task_ok = xTaskCreatePinnedToCore(
      taskImuLoop, "imu_loop", 3072, NULL, 2, &imuTaskHandle, 0);

  BaseType_t thermal_task_ok = pdPASS;
#if THERMAL_ENABLED
  Serial.printf("[HEAP] before thermal task: free=%u max_alloc=%u internal_largest=%u\n",
                ESP.getFreeHeap(),
                ESP.getMaxAllocHeap(),
                heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  thermal_task_ok = xTaskCreatePinnedToCore(
      taskThermalLoop, "thermal", 4096, NULL, 2, &thermalTaskHandle, 0);
#endif

  BaseType_t tts_task_ok = xTaskCreatePinnedToCore(
      taskTTSPlay, "tts_play", 4096, NULL, 3, &ttsPlaybackTaskHandle, 0);

  Serial.printf("[TASKS] camCap=%d camSend=%d micCap=%d micUpload=%d imu=%d thermal=%d tts=%d\n",
                cam_cap_task_ok,
                cam_send_task_ok,
                mic_cap_task_ok,
                mic_upload_task_ok,
                imu_task_ok,
                thermal_task_ok,
                tts_task_ok);

  Serial.printf("[HEAP] after task creation: free=%u max_alloc=%u internal_largest=%u\n",
                ESP.getFreeHeap(),
                ESP.getMaxAllocHeap(),
                heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));

  bool essential_task_failed =
      cam_cap_task_ok != pdPASS ||
      cam_send_task_ok != pdPASS ||
      mic_cap_task_ok != pdPASS ||
      mic_upload_task_ok != pdPASS ||
      tts_task_ok != pdPASS;

#if THERMAL_ENABLED
  essential_task_failed = essential_task_failed || thermal_task_ok != pdPASS;
#endif
  essential_task_failed = essential_task_failed || imu_task_ok != pdPASS;

  if (essential_task_failed) {
    Serial.println("[FATAL] One or more essential tasks failed to start; rebooting...");
    delay(2000);
    esp_restart();
  }
  setupComplete = true;
  Serial.println("[TASKS] all essential tasks pdPASS; network owners released");
}


void loop() {
  static uint32_t lastHealthMs = 0;
  static uint8_t criticalMemoryCount = 0;
  uint32_t now = millis();
  if (now - lastHealthMs < HEALTH_LOG_INTERVAL_MS) {
    delay(20);
    return;
  }
  lastHealthMs = now;

  size_t internalLargest =
      heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  UBaseType_t camQ = qFrames ? uxQueueMessagesWaiting(qFrames) : 0;
  UBaseType_t micQ = qAudio ? uxQueueMessagesWaiting(qAudio) : 0;
  UBaseType_t ttsQ = qTTS ? uxQueueMessagesWaiting(qTTS) : 0;
  UBaseType_t imuQ = qImu ? uxQueueMessagesWaiting(qImu) : 0;
#if THERMAL_ENABLED
  UBaseType_t thermalQ = qThermal ? uxQueueMessagesWaiting(qThermal) : 0;
#else
  UBaseType_t thermalQ = 0;
#endif
  UBaseType_t swCamCap = camCaptureTaskHandle ? uxTaskGetStackHighWaterMark(camCaptureTaskHandle) : 0;
  UBaseType_t swCamNet = camNetworkTaskHandle ? uxTaskGetStackHighWaterMark(camNetworkTaskHandle) : 0;
  UBaseType_t swMicCap = micCaptureTaskHandle ? uxTaskGetStackHighWaterMark(micCaptureTaskHandle) : 0;
  UBaseType_t swAudNet = audioNetworkTaskHandle ? uxTaskGetStackHighWaterMark(audioNetworkTaskHandle) : 0;
  UBaseType_t swThermal = thermalTaskHandle ? uxTaskGetStackHighWaterMark(thermalTaskHandle) : 0;
  UBaseType_t swTts = ttsPlaybackTaskHandle ? uxTaskGetStackHighWaterMark(ttsPlaybackTaskHandle) : 0;
  UBaseType_t swImu = imuTaskHandle ? uxTaskGetStackHighWaterMark(imuTaskHandle) : 0;
  UBaseType_t swHttp = taskHttpPlayHandle ? uxTaskGetStackHighWaterMark(taskHttpPlayHandle) : 0;

  Serial.printf(
      "[HEALTH] uptime=%lu heap=%u minHeap=%u maxAlloc=%u internalLargest=%u psram=%u "
      "camQ=%u thermalQ=%u imuQ=%u micQ=%u ttsQ=%u wsCam=%d wsAud=%d "
      "cam=%lu/%lu/%lu thermal=%lu/%lu imu=%lu/%lu mic=%lu/%lu/%lu "
      "speaker=%lu/%lu/%lu reconnect=%lu/%lu "
      "ageCam=%lu ageThermal=%lu ageMic=%lu ageSpeaker=%lu activityCam=%lu activityAud=%lu "
      "stackCamCap=%u stackCamNet=%u stackMicCap=%u stackAudNet=%u "
      "stackThermal=%u stackTts=%u stackImu=%u stackHttp=%u\n",
      (unsigned long)now, ESP.getFreeHeap(), ESP.getMinFreeHeap(),
      ESP.getMaxAllocHeap(), internalLargest, ESP.getFreePsram(),
      (unsigned)camQ, (unsigned)thermalQ, (unsigned)imuQ, (unsigned)micQ, (unsigned)ttsQ,
      cam_thermal_ws_ready, aud_ws_ready,
      frame_captured_count, frame_sent_count, frame_dropped_count,
      (unsigned long)thermalSentFrames, (unsigned long)thermalDroppedFrames,
      (unsigned long)imuSentPackets, (unsigned long)imuDroppedPackets,
      (unsigned long)micCapturedChunks, (unsigned long)micSentChunks,
      (unsigned long)micDroppedChunks,
      (unsigned long)speakerQueuedChunks, (unsigned long)speakerPlayedChunks,
      (unsigned long)ttsDroppedChunks,
      (unsigned long)camReconnectAttempts, (unsigned long)audReconnectAttempts,
      lastCameraSendMs ? (unsigned long)(now - lastCameraSendMs) : 0UL,
      lastThermalSendMs ? (unsigned long)(now - lastThermalSendMs) : 0UL,
      lastMicSendMs ? (unsigned long)(now - lastMicSendMs) : 0UL,
      lastSpeakerPacketMs ? (unsigned long)(now - lastSpeakerPacketMs) : 0UL,
      camLastTrafficMs ? (unsigned long)(now - camLastTrafficMs) : 0UL,
      audLastTrafficMs ? (unsigned long)(now - audLastTrafficMs) : 0UL,
      (unsigned)swCamCap, (unsigned)swCamNet, (unsigned)swMicCap, (unsigned)swAudNet,
      (unsigned)swThermal, (unsigned)swTts, (unsigned)swImu, (unsigned)swHttp);

  if (internalLargest < CRITICAL_INTERNAL_BLOCK_BYTES) criticalMemoryCount++;
  else criticalMemoryCount = 0;
  if (criticalMemoryCount >= CRITICAL_MEMORY_INTERVALS) {
    Serial.printf("[FATAL] persistent critical internal heap largest=%u intervals=%u; controlled restart\n",
                  internalLargest, criticalMemoryCount);
    controlledRestartRequested = true;
    delay(300);
    esp_restart();
  }
}
