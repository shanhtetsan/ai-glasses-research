# AI-Powered Smart Glasses for Blind Navigation 🤖👓

<div align="center">

An intelligent navigation and assistance system for visually impaired individuals, integrating tactile paving navigation, pedestrian crossing assistance, object recognition, and real-time voice interaction.  
**This project is for educational and research purposes only. Do not distribute directly to visually impaired individuals.**  
This repository contains code only. Model files can be found at: https://www.modelscope.cn/models/archifancy/AIGlasses_for_navigation — download and place in the `/model` folder.

[Features](#features) • [Quick Start](#quick-start) • [System Architecture](#system-architecture) • [Usage](#usage) • [Development Docs](#development-docs)

</div>

---

## 📋 Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Usage](#usage)
- [Configuration](#configuration)
- [Development Docs](#development-docs)

---

## ✨ Features

### 🚶 Tactile Paving Navigation
- **Real-time detection**: YOLO segmentation model detects tactile paving in real time
- **Voice guidance**: Precise directional instructions (turn left, turn right, go straight, etc.)
- **Obstacle detection**: Automatically identifies obstacles ahead and plans avoidance routes
- **Turn detection**: Automatically identifies sharp turns and provides advance warnings
- **Optical flow stabilization**: Lucas-Kanade optical flow algorithm stabilizes the mask and reduces jitter

### 🚦 Pedestrian Crossing Assistance
- **Zebra crossing recognition**: Real-time detection of crosswalk position and direction
- **Traffic light recognition**: Color and shape-based traffic light state detection
- **Alignment guidance**: Guides the user to align with the center of the crosswalk
- **Safety alerts**: Voice prompt when the light turns green

### 🔍 Object Recognition and Search
- **Smart object search**: Voice commands to find items (e.g., "Find me a Red Bull")
- **Real-time object tracking**: YOLO-E open-vocabulary detection + ByteTrack tracking
- **Hand guidance**: MediaPipe hand detection guides the user's hand toward the object
- **Grasp detection**: Detects gripping motion to confirm the item has been picked up
- **Multimodal feedback**: Visual annotation + voice guidance + centering prompts

### 🎙️ Real-Time Voice Interaction
- **ASR (Automatic Speech Recognition)**: Real-time speech recognition via Alibaba Cloud DashScope Paraformer
- **Multimodal dialogue**: Qwen-Omni-Turbo supports image + text input with voice output
- **Smart command parsing**: Automatically recognizes navigation, search, and conversational commands
- **Context awareness**: Intelligently filters irrelevant commands depending on the current mode

### 📹 Video and Audio Processing
- **Real-time video streaming**: WebSocket push, supports multiple simultaneous viewers
- **Synchronized A/V recording**: Automatically saves timestamped video and audio files
- **IMU data fusion**: Receives IMU data from ESP32 for attitude estimation
- **Multi-channel audio mixing**: Supports simultaneous playback of system voice, AI responses, and ambient audio

### 🎨 Visualization and Interaction
- **Web monitoring**: View the processed video stream in real time via browser
- **IMU 3D visualization**: Real-time device pose rendering using Three.js
- **Status panel**: Displays navigation state, detection info, FPS, etc.

---

## 💻 System Requirements

### Hardware
**Server/Development machine:**
- CPU: Intel i5 or higher (i7/i9 recommended)
- GPU: NVIDIA GPU (CUDA 11.8+, RTX 3060 or higher recommended)
- RAM: 8GB (16GB recommended)
- Storage: 10GB free space

**Client device (optional):**
- ESP32-CAM or other WebSocket-capable camera
- Microphone (for voice input)
- Speaker or headphones (for voice output)

### Software
- OS: Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.15+
- Python: 3.9–3.11
- CUDA: 11.8 or later (required for GPU acceleration)
- Browser: Chrome 90+, Firefox 88+, Edge 90+ (for web monitoring)

### API Keys
- **Alibaba Cloud DashScope API Key** (required):
  - Used for ASR and Qwen-Omni dialogue
  - Sign up at: https://dashscope.console.aliyun.com/

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/aiglass.git
cd aiglass/rebuild1002
```

### 2. Install dependencies

Create a virtual environment (recommended):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

Install Python packages:
```bash
pip install -r requirements.txt
```

### 3. Download model files

Place the following model files in the `model/` directory:

| File | Purpose | Size | Download |
|------|---------|------|----------|
| `yolo-seg.pt` | Tactile paving segmentation | ~50MB | [TBD] |
| `yoloe-11l-seg.pt` | Open-vocabulary detection | ~80MB | [TBD] |
| `shoppingbest5.pt` | Object recognition | ~30MB | [TBD] |
| `trafficlight.pt` | Traffic light detection | ~20MB | [TBD] |
| `hand_landmarker.task` | Hand detection | ~15MB | [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models) |

### 4. Configure API key

Create a `.env` file:
```bash
DASHSCOPE_API_KEY=your_api_key_here
```

### 5. Start the system
```bash
python app_main.py
```

The system starts at `http://0.0.0.0:8081`. Open your browser to see the live monitoring interface.

### 6. Connect device (optional)

If using ESP32-CAM:
1. Flash `compile/compile.ino` to the ESP32
2. Update WiFi credentials to match your network
3. The ESP32 will automatically connect to the WebSocket endpoint

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  ESP32-CAM   │  │   Browser    │  │    Mobile    │      │
│  │ (video/audio)│  │ (monitor UI) │  │(voice control)│      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │ WebSocket        │ HTTP/WS          │ WebSocket
┌─────────▼──────────────────▼──────────────────▼─────────────┐
│    ┌─────────────────────────────────────────────────────┐   │
│    │         FastAPI Main Service (app_main.py)          │   │
│    │  - WebSocket routing                                │   │
│    │  - Audio/video stream distribution                  │   │
│    │  - State management and coordination                │   │
│    └────┬────────────────┬────────────────┬──────────────┘   │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐         │
│  │  ASR Module │  │Omni Dialogue│  │Audio Playback│         │
│  │ (asr_core)  │  │(omni_client)│  │(audio_player)│         │
│  └─────────────┘  └─────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼─────────────┐
│                   Navigation Master Layer                     │
│    ┌─────────────────────────────────────────────────┐       │
│    │  NavigationMaster (navigation_master.py)         │       │
│    │  - State machine: IDLE / CHAT / BLINDPATH_NAV /  │       │
│    │    CROSSING / TRAFFIC_LIGHT / ITEM_SEARCH        │       │
│    └───┬─────────────────┬───────────────────┬────────┘       │
│   ┌────▼────────┐  ┌─────▼───────────┐  ┌───▼────────┐      │
│   │Blind Path   │  │ Cross Street    │  │Item Search │      │
│   │(blindpath)  │  │ (crossstreet)   │  │(yolomedia) │      │
│   └─────────────┘  └─────────────────┘  └────────────┘      │
└─────────────────────────────────────────────────────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼─────────────┐
│                      Model Inference Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │YOLO Segment  │  │  YOLO-E Det  │  │  MediaPipe   │       │
│  │(paving/zebra)│  │(open vocab)  │  │(hand detect) │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │Traffic Light │  │Optical Flow  │                         │
│  │(HSV + YOLO)  │  │(Lucas-Kanade)│                         │
│  └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
          │
┌─────────▼─────────────────────────────────────────────────────┐
│                     External Services Layer                    │
│  ┌──────────────────────────────────────────────┐            │
│  │  Alibaba Cloud DashScope API                  │            │
│  │  - Paraformer ASR (real-time speech recognition)│          │
│  │  - Qwen-Omni-Turbo (multimodal dialogue)      │            │
│  │  - Qwen-Turbo (tag extraction)                │            │
│  └──────────────────────────────────────────────┘            │
└───────────────────────────────────────────────────────────────┘
```

### Core Modules

| Module | File | Function |
|--------|------|----------|
| Main App | `app_main.py` | FastAPI service, WebSocket management, state coordination |
| Navigation Master | `navigation_master.py` | State machine, mode switching, voice throttling |
| Blind Path Nav | `workflow_blindpath.py` | Tactile paving detection, obstacle avoidance, turn guidance |
| Cross Street Nav | `workflow_crossstreet.py` | Zebra crossing detection, traffic light recognition, alignment |
| Item Search | `yolomedia.py` | Object detection, hand guidance, grasp confirmation |
| Speech Recognition | `asr_core.py` | Real-time ASR, VAD, command parsing |
| Speech Synthesis | `omni_client.py` | Qwen-Omni streaming voice generation |
| Audio Playback | `audio_player.py` | Multi-channel mixing, TTS playback, volume control |
| Video Recording | `sync_recorder.py` | Synchronized A/V recording |
| Bridge I/O | `bridge_io.py` | Thread-safe frame buffering and distribution |

---

## 📖 Usage

### Voice Commands

The system responds to voice commands without a wake word.

#### Navigation
```
"Start navigation" / "Tactile paving navigation"  → Start blind path navigation
"Stop navigation" / "End navigation"               → Stop blind path navigation
"Start crossing" / "Help me cross"                 → Start crosswalk mode
"Done crossing" / "End crossing"                   → Stop crosswalk mode
```

#### Traffic Light Detection
```
"Detect traffic light" / "Check traffic light"    → Start traffic light detection
"Stop detection" / "Stop traffic light"            → Stop detection
```

#### Object Search
```
"Help me find [item name]"    → Start object search
  Examples:
  - "Help me find a Red Bull"
  - "Find me a water bottle"
"Found it" / "Got it"         → Confirm item retrieved
```

#### Conversational AI
```
"What is this?"               → Take photo and identify
"Can I eat this?"             → Item consultation
Any other question            → General AI dialogue
```

### Navigation States

| State | Description |
|-------|-------------|
| IDLE | Waiting for user commands, showing raw video |
| CHAT | Multimodal AI dialogue, navigation paused |
| BLINDPATH_NAV | Following tactile paving with real-time correction and obstacle detection |
| CROSSING | Crosswalk mode — finding zebra crossing, waiting for green light, crossing |
| ITEM_SEARCH | Detecting target item, guiding hand, confirming grasp |
| TRAFFIC_LIGHT_DETECTION | Monitoring traffic light state, voice announcing changes |

### Web Monitoring Interface

Open `http://localhost:8081` in your browser to see:
- Live processed video stream with navigation overlays
- Status panel: current mode, detection info, FPS
- IMU 3D visualization of device pose
- Speech recognition results and AI responses

### WebSocket Endpoints

| Endpoint | Purpose | Format |
|----------|---------|--------|
| `/ws/camera` | ESP32 camera stream | Binary (JPEG) |
| `/ws/viewer` | Browser video subscription | Binary (JPEG) |
| `/ws_audio` | ESP32 audio upload | Binary (PCM16) |
| `/ws_ui` | UI status push | JSON |
| `/ws` | IMU data reception | JSON |
| `/stream.wav` | Audio download stream | Binary (WAV) |

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```bash
# Alibaba Cloud API
DASHSCOPE_API_KEY=sk-xxxxx

# Model paths (optional — defaults used if not set)
BLIND_PATH_MODEL=model/yolo-seg.pt
OBSTACLE_MODEL=model/yoloe-11l-seg.pt
YOLOE_MODEL_PATH=model/yoloe-11l-seg.pt

# Navigation parameters
AIGLASS_MASK_MIN_AREA=1500      # Minimum mask area
AIGLASS_MASK_MORPH=3            # Morphological kernel size
AIGLASS_MASK_MISS_TTL=6         # Mask loss tolerance frames
AIGLASS_PANEL_SCALE=0.65        # Status panel scale

# Audio
TTS_INTERVAL_SEC=1.0            # Voice broadcast interval
ENABLE_TTS=true                 # Enable TTS
```

### Adjust Performance Parameters

```python
# yolomedia.py
HAND_DOWNSCALE = 0.8    # Hand detection downscale (lower = faster, less accurate)
HAND_FPS_DIV = 1        # Frame skip for hand detection (2 = every other frame)

# workflow_blindpath.py
FEATURE_PARAMS = dict(
    maxCorners=600,      # Optical flow feature points (fewer = faster)
    qualityLevel=0.001,
    minDistance=5
)
```

---

## 🛠️ Development Docs

### Adding New Voice Commands

In `app_main.py`, inside `start_ai_with_text_custom()`:

```python
if "your keyword" in user_text:
    print("[CUSTOM] New command triggered")
    await ui_broadcast_final("[System] New feature started")
    return
```

To modify command filtering:
```python
allowed_keywords = ["help me see", "help me find", "your new keyword"]
```

### Extending Navigation

Add a new state in `workflow_blindpath.py`:
```python
# In BlindPathNavigator.__init__()
self.your_new_state_var = False

# In process_frame()
def process_frame(self, image):
    if self.your_new_state_var:
        guidance_text = "New state guidance"
```

Add to state machine in `navigation_master.py`:
```python
class NavigationMaster:
    def start_your_new_mode(self):
        self.state = "YOUR_NEW_MODE"
```

### Integrating New Models

```python
# your_model_wrapper.py
class YourModelWrapper:
    def __init__(self, model_path):
        self.model = load_your_model(model_path)
    
    def detect(self, image):
        return results
```

Load in `app_main.py`:
```python
your_model = YourModelWrapper("model/your_model.pt")
```

### Debugging

Enable detailed logging:
```python
# Top of app_main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check FPS bottlenecks:
```python
# yolomedia.py
PERF_DEBUG = True
```

Test individual modules:
```bash
python test_cross_street_blindpath.py
python test_traffic_light.py
python test_recorder.py
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
