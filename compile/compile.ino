// ===== all_in_one_merged.ino — XIAO ESP32S3 Sense: Camera + Mic (PDM) + IMU (ICM42688 SPI) =====


#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_camera.h>
#include <ArduinoWebsockets.h>
#include "ESP_I2S.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "esp_heap_caps.h"
struct WavFmt;
#include <cstring>      // memcmp
#include <WiFiUdp.h>
#include <WiFiClient.h> 
#include <Wire.h>
using namespace websockets;

// ===== WiFi / Server =====
const char* WIFI_SSID   = "PRST";
const char* WIFI_PASS   = "phone12345";
const char* SERVER_HOST = "192.0.0.2";
const uint16_t SERVER_PORT = 8081;
const uint32_t WIFI_CONNECT_TIMEOUT_MS = 30000;
const uint32_t WIFI_RETRY_DELAY_MS = 3000;
const bool USE_GATEWAY_AS_SERVER = true;  // Mac Internet Sharing: server runs on Wi-Fi gateway.

static const char* CAM_WS_PATH = "/ws/camera";
static const char* AUD_WS_PATH = "/ws_audio";

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

// ===== Mic (PDM RX) =====
#define I2S_MIC_CLOCK_PIN 42
#define I2S_MIC_DATA_PIN  41
const int SAMPLE_RATE     = 16000; 
const int CHUNK_MS        = 20;
const int BYTES_PER_CHUNK = SAMPLE_RATE * CHUNK_MS / 1000 * 2;
const int AUDIO_QUEUE_DEPTH = 10;

// ===== Speaker (I2S TX → MAX98357A) =====
#define I2S_SPK_BCLK D1
#define I2S_SPK_LRCK D2
#define I2S_SPK_DIN  D3
const int TTS_RATE = 16000;
const bool SPEAKER_TEST_ON_BOOT = true;

// ===== IMU (MPU-6050 over I2C) / UDP =====
// Default I2C pins on XIAO ESP32S3: SDA=D4(GPIO5), SCL=D5(GPIO6)
// Change these if you wired the GY-521 to different pins.
#define IMU_I2C_SDA   5   // D4
#define IMU_I2C_SCL   6   // D5
const char* UDP_HOST  = SERVER_HOST;
const int   UDP_PORT  = 12345;

WiFiUDP udp;
String resolved_server_host = SERVER_HOST;

const char* wifi_status_name(wl_status_t status) {
  switch (status) {
    case WL_IDLE_STATUS: return "IDLE";
    case WL_NO_SSID_AVAIL: return "NO_SSID_AVAILABLE";
    case WL_SCAN_COMPLETED: return "SCAN_COMPLETED";
    case WL_CONNECTED: return "CONNECTED";
    case WL_CONNECT_FAILED: return "CONNECT_FAILED";
    case WL_CONNECTION_LOST: return "CONNECTION_LOST";
    case WL_DISCONNECTED: return "DISCONNECTED";
    default: return "UNKNOWN";
  }
}

void print_wifi_scan_for_target() {
  Serial.printf("[WiFi] scanning for SSID '%s'...\n", WIFI_SSID);
  int n = WiFi.scanNetworks(false, true);
  if (n <= 0) {
    Serial.println("[WiFi] scan found no networks");
    return;
  }

  bool found = false;
  for (int i = 0; i < n; i++) {
    String ssid = WiFi.SSID(i);
    if (ssid == WIFI_SSID) {
      found = true;
      Serial.printf(
        "[WiFi] found target SSID, RSSI=%d dBm, channel=%d, encryption=%d\n",
        WiFi.RSSI(i),
        WiFi.channel(i),
        WiFi.encryptionType(i)
      );
    }
  }
  if (!found) {
    Serial.println("[WiFi] target SSID not found. Check hotspot name and 2.4 GHz compatibility.");
  }
  WiFi.scanDelete();
}

void connect_wifi_forever() {
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);
  esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);

  Serial.printf("[WiFi] MAC=%s\n", WiFi.macAddress().c_str());
  Serial.printf("[WiFi] target SSID='%s'\n", WIFI_SSID);
  Serial.printf("[NET] configured server=%s:%u camera=%s audio=%s udp=%s:%d gateway_mode=%d\n",
                SERVER_HOST, SERVER_PORT, CAM_WS_PATH, AUD_WS_PATH, UDP_HOST, UDP_PORT,
                USE_GATEWAY_AS_SERVER ? 1 : 0);

  for (;;) {
    print_wifi_scan_for_target();

    WiFi.disconnect(true, true);
    delay(300);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    Serial.print("[WiFi] connecting");
    uint32_t start = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - start < WIFI_CONNECT_TIMEOUT_MS) {
      delay(300);
      Serial.print(".");
    }

    wl_status_t status = WiFi.status();
    if (status == WL_CONNECTED) {
      Serial.printf("\n[WiFi] OK ip=%s gateway=%s rssi=%d dBm\n",
                    WiFi.localIP().toString().c_str(),
                    WiFi.gatewayIP().toString().c_str(),
                    WiFi.RSSI());
      resolved_server_host = USE_GATEWAY_AS_SERVER
        ? WiFi.gatewayIP().toString()
        : String(SERVER_HOST);
      Serial.printf("[NET] resolved server=%s:%u\n",
                    resolved_server_host.c_str(),
                    SERVER_PORT);
      return;
    }

    Serial.printf("\n[WiFi] failed after %lu ms, status=%s (%d). Retrying in %lu ms...\n",
                  WIFI_CONNECT_TIMEOUT_MS,
                  wifi_status_name(status),
                  (int)status,
                  WIFI_RETRY_DELAY_MS);
    delay(WIFI_RETRY_DELAY_MS);
  }
}

// ===== WS / Queues / I2S =====
WebsocketsClient wsCam;
WebsocketsClient wsAud;
volatile bool cam_ws_ready = false;
volatile bool aud_ws_ready = false;
volatile bool snapshot_in_progress = false; // Pause live capture during a high-res snapshot

typedef camera_fb_t* fb_ptr_t;
QueueHandle_t qFrames;

typedef struct {
  size_t n;
  uint8_t data[BYTES_PER_CHUNK];
} AudioChunk;
QueueHandle_t qAudio;

#define TTS_QUEUE_DEPTH 12
typedef struct { uint16_t n; uint8_t data[2048]; } TTSChunk;
QueueHandle_t qTTS;
volatile bool tts_playing = false;

I2SClass i2sIn;   // PDM RX (Mic)
I2SClass i2sOut;  // STD TX (Speaker)
volatile bool run_audio_stream = false;

void print_network_runtime_status(const char* reason) {
  Serial.printf(
    "[NET] %s wifi=%s ip=%s gateway=%s target=%s:%u rssi=%d\n",
    reason,
    wifi_status_name(WiFi.status()),
    WiFi.localIP().toString().c_str(),
    WiFi.gatewayIP().toString().c_str(),
    resolved_server_host.c_str(),
    SERVER_PORT,
    WiFi.RSSI()
  );
}

void print_memory_status(const char* reason) {
  Serial.printf(
    "[MEM] %s heap_free=%u heap_min=%u internal_free=%u psram_free=%u psram_size=%u\n",
    reason,
    ESP.getFreeHeap(),
    ESP.getMinFreeHeap(),
    heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
    ESP.getFreePsram(),
    ESP.getPsramSize()
  );
}

bool create_task_checked(TaskFunction_t task,
                         const char* name,
                         uint32_t stack_words,
                         UBaseType_t priority,
                         BaseType_t core) {
  BaseType_t ok = xTaskCreatePinnedToCore(task, name, stack_words, NULL, priority, NULL, core);
  if (ok != pdPASS) {
    Serial.printf("[MEM] task create failed name=%s stack_words=%u core=%d\n",
                  name,
                  stack_words,
                  (int)core);
    print_memory_status("task-create-failed");
    return false;
  }
  return true;
}

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
    s->set_exposure_ctrl(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_aec2(s, 0);
    s->set_aec_value(s, 40);
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
    
    if (cam_ws_ready) {
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
      if (fb && cam_ws_ready) {
        // Frame-rate throttle: if target FPS is set, pace sends accordingly; extra frames discarded by qFrames
        if (g_target_fps > 0) {
          const int period_ms = 1000 / g_target_fps;
          TickType_t now = xTaskGetTickCount();
          int elapsed = (now - lastTick) * portTICK_PERIOD_MS;
          if (elapsed < period_ms) vTaskDelay(pdMS_TO_TICKS(period_ms - elapsed));
          lastTick = xTaskGetTickCount();
        }
        
        unsigned long send_start = millis();
        bool ok = wsCam.sendBinary((const char*)fb->buf, fb->len);
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
          wsCam.close(); 
          cam_ws_ready = false;
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
      if (cam_ws_ready && last_sent_time > 0 && (now - last_sent_time) > 3000) {
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
    if (run_audio_stream && aud_ws_ready) {
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
    if (run_audio_stream && aud_ws_ready){
      AudioChunk ch;
      if (xQueueReceive(qAudio, &ch, pdMS_TO_TICKS(100)) == pdPASS){
        wsAud.sendBinary((const char*)ch.data, ch.n);
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
  i2sOut.setPins(I2S_SPK_BCLK, I2S_SPK_LRCK, I2S_SPK_DIN);
  if (!i2sOut.begin(I2S_MODE_STD, TTS_RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO)) {
    Serial.println("[I2S OUT] init failed");
    while(1){ delay(1000); }
  }
  Serial.println("[I2S OUT] STD TX @16kHz 32bit STEREO ready");
}

void play_speaker_test_tone(uint16_t freq_hz = 880, uint16_t duration_ms = 900) {
  Serial.printf("[I2S OUT] speaker test tone %u Hz for %u ms\n", freq_hz, duration_ms);
  const uint32_t sample_rate = TTS_RATE;
  const size_t chunk_samples = 256;
  int32_t outLR[chunk_samples * 2];
  uint32_t total_samples = (sample_rate * duration_ms) / 1000;
  uint32_t phase = 0;
  uint32_t phase_step = ((uint64_t)freq_hz << 32) / sample_rate;

  while (total_samples > 0) {
    size_t n = min((uint32_t)chunk_samples, total_samples);
    for (size_t i = 0; i < n; i++) {
      phase += phase_step;
      int32_t s = (phase & 0x80000000UL) ? 9000 : -9000;
      int32_t v32 = s << 16;
      outLR[i * 2 + 0] = v32;
      outLR[i * 2 + 1] = v32;
    }
    size_t bytes = n * 2 * sizeof(int32_t);
    size_t off = 0;
    while (off < bytes) {
      size_t wrote = i2sOut.write((uint8_t*)outLR + off, bytes - off);
      if (wrote == 0) vTaskDelay(pdMS_TO_TICKS(1));
      else off += wrote;
    }
    total_samples -= n;
  }
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
  WiFiClient cli;

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
      if (!cli.connect(resolved_server_host.c_str(), SERVER_PORT)) { delay(500); continue; }
      String req =
        String("GET /stream.wav HTTP/1.1\r\n") +
        "Host: " + resolved_server_host + ":" + String(SERVER_PORT) + "\r\n" +
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
  for(;;){
    if (!tts_playing){ vTaskDelay(pdMS_TO_TICKS(5)); continue; }
    TTSChunk ch;
    if (xQueueReceive(qTTS, &ch, pdMS_TO_TICKS(50)) == pdPASS){
      size_t inSamp  = ch.n / 2;
      int16_t* inPtr = (int16_t*)ch.data;
      size_t outPairs = 0;
      for (size_t i = 0; i < inSamp; ++i){
        int32_t s = (int32_t)inPtr[i];
        s = (s * 19660) / 32768;
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

static void mpu_write(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

static uint8_t mpu_read1(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);  // repeated-START keeps bus active for the read
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0xFF;
}

static void mpu_read14(uint8_t* dst) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(MPU_REG_ACCEL_OUT);
  Wire.endTransmission(false);  // repeated-START — do not release bus
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14);
  for (uint8_t i = 0; i < 14; i++)
    dst[i] = Wire.available() ? Wire.read() : 0;
}

bool imu_init_i2c() {
  Wire.begin(IMU_I2C_SDA, IMU_I2C_SCL);
  delay(5);

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

// EMA smoothing on accel only; does not change the UDP field names.
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

    if (n > 0) {
      udp.beginPacket(resolved_server_host.c_str(), UDP_PORT);
      udp.write((const uint8_t*)buf, n);
      udp.endPacket();
    }
    vTaskDelay(pdMS_TO_TICKS(20)); // 50 Hz
  }
}

// ====================================================================
// Setup / Loop
// ====================================================================
void setup() {
  Serial.begin(115200);
  delay(1000);
  print_memory_status("boot");
  if (!psramFound()) {
    Serial.println("[MEM] PSRAM not found. Re-upload with FQBN esp32:esp32:XIAO_ESP32S3:PSRAM=opi");
  }

  connect_wifi_forever();
  print_memory_status("after-wifi");

  if (!init_camera()) { Serial.println("[CAM] init failed, reboot..."); delay(1500); esp_restart(); }
  print_memory_status("after-camera");

  udp.begin(0);

  init_i2s_in();
  init_i2s_out();
  if (SPEAKER_TEST_ON_BOOT) {
    play_speaker_test_tone();
  }
  print_memory_status("after-i2s");

  qFrames = xQueueCreate(3, sizeof(fb_ptr_t));  // 3 buffers to reduce frame drops
  qAudio  = xQueueCreate(AUDIO_QUEUE_DEPTH, sizeof(AudioChunk));
  qTTS    = xQueueCreate(TTS_QUEUE_DEPTH, sizeof(TTSChunk));
  if (!qFrames || !qAudio || !qTTS) {
    Serial.printf("[MEM] queue create failed qFrames=%p qAudio=%p qTTS=%p\n", qFrames, qAudio, qTTS);
    print_memory_status("queue-create-failed");
    delay(1500);
    esp_restart();
  }
  print_memory_status("after-queues");

  if (!create_task_checked(taskCamCapture, "cam_cap", 10240, 4, 1) ||
      !create_task_checked(taskCamSend,    "cam_snd",  8192, 3, 1) ||
      !create_task_checked(taskMicCapture, "mic_cap",  4096, 2, 0) ||
      !create_task_checked(taskMicUpload,  "mic_upl",  4096, 2, 1) ||
      !create_task_checked(taskImuLoop,    "imu_loop", 4096, 2, 0) ||
      !create_task_checked(taskTTSPlay,    "tts_play", 4096, 2, 0)) {
    delay(1500);
    esp_restart();
  }
  print_memory_status("after-tasks");

  wsCam.onEvent([](WebsocketsEvent ev, String){
    if (ev == WebsocketsEvent::ConnectionOpened)  { 
      cam_ws_ready = true;  
      Serial.println("[WS-CAM] open");
      // Reset statistics
      frame_sent_count = 0;
      frame_dropped_count = 0;
      ws_send_fail_count = 0;
      last_stats_time = millis();
    }
    if (ev == WebsocketsEvent::ConnectionClosed)  { 
      cam_ws_ready = false; 
      Serial.printf("[WS-CAM] closed (sent=%lu, dropped=%lu, fail=%lu)\n", 
                    frame_sent_count, frame_dropped_count, ws_send_fail_count);
    }
  });

  wsCam.onMessage([](WebsocketsMessage msg){
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
          wsCam.send("SNAP:BEGIN");
          bool ok = wsCam.sendBinary((const char*)fb->buf, fb->len);
          wsCam.send("SNAP:END");
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
  });

  wsAud.onEvent([](WebsocketsEvent ev, String){
    if (ev == WebsocketsEvent::ConnectionOpened)  { aud_ws_ready = true;  Serial.println("[WS-AUD] open"); }
    if (ev == WebsocketsEvent::ConnectionClosed)  { 
      aud_ws_ready = false; 
      Serial.println("[WS-AUD] closed"); 
      stopStreamWav();
    }
  });

  wsAud.onMessage([](WebsocketsMessage msg){
    if (msg.isText()){
      String s = msg.data(); s.trim();
      if (s == "RESTART"){
        run_audio_stream = false; xQueueReset(qAudio); delay(50);
        wsAud.send("START"); run_audio_stream = true;
      }
    }
  });
}

void loop() {
  static unsigned long last_net_status = 0;
  static unsigned long cam_retry_count = 0;
  static unsigned long aud_retry_count = 0;

  if (WiFi.status() != WL_CONNECTED) {
    Serial.printf("[WiFi] lost connection, status=%s (%d). Reconnecting...\n",
                  wifi_status_name(WiFi.status()),
                  (int)WiFi.status());
    cam_ws_ready = false;
    aud_ws_ready = false;
    run_audio_stream = false;
    stopStreamWav();
    wsCam.close();
    wsAud.close();
    connect_wifi_forever();
  }

  unsigned long now = millis();
  if (now - last_net_status > 10000) {
    print_network_runtime_status("runtime");
    last_net_status = now;
  }

  if (!wsCam.available()) {
    if (wsCam.connect(resolved_server_host.c_str(), SERVER_PORT, CAM_WS_PATH)) {
      Serial.println("[WS-CAM] connected");
      cam_retry_count = 0;
    } else {
      cam_retry_count++;
      Serial.printf("[WS-CAM] retry #%lu target=%s:%u in 1s...\n",
                    cam_retry_count,
                    resolved_server_host.c_str(),
                    SERVER_PORT);
      delay(1000);
    }
  }

  if (!wsAud.available()) {
    if (wsAud.connect(resolved_server_host.c_str(), SERVER_PORT, AUD_WS_PATH)) {
      Serial.println("[WS-AUD] connected");
      aud_retry_count = 0;
      delay(50);
      run_audio_stream = true;
      wsAud.send("START");
      startStreamWav();   // /stream.wav (chunked)
    } else {
      aud_retry_count++;
      Serial.printf("[WS-AUD] retry #%lu target=%s:%u in 2s...\n",
                    aud_retry_count,
                    resolved_server_host.c_str(),
                    SERVER_PORT);
      delay(2000);
    }
  }

  wsCam.poll();
  wsAud.poll();
  delay(2);
}
