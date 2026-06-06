# app_main.py
# -*- coding: utf-8 -*-
import os, sys, time, json, asyncio, base64, audioop
from typing import Any, Dict, Optional, Tuple, List, Callable, Set, Deque
from collections import deque
from dataclasses import dataclass
import re
# Navigation modules — optional, disabled for indoor/cooking mode
try:
    from navigation_master import NavigationMaster, OrchestratorResult
    from workflow_blindpath import BlindPathNavigator
    from workflow_crossstreet import CrossStreetNavigator
    _NAV_AVAILABLE = True
except Exception as _nav_err:
    NavigationMaster = OrchestratorResult = BlindPathNavigator = CrossStreetNavigator = None  # type: ignore
    _NAV_AVAILABLE = False
    print(f"[NAV] Navigation modules disabled: {_nav_err}")

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
import uvicorn
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore

try:
    from obstacle_detector_client import ObstacleDetectorClient
except ImportError:
    ObstacleDetectorClient = None  # type: ignore

try:
    import mediapipe as mp
except ImportError:
    mp = None  # type: ignore

import bridge_io
import threading

try:
    import yolomedia
    _YOLOMEDIA_AVAILABLE = True
except Exception as _ym_err:
    yolomedia = None  # type: ignore
    _YOLOMEDIA_AVAILABLE = False
    print(f"[YOLOMEDIA] Disabled: {_ym_err}")

# ---- Windows 事件循环策略 ----
# ---- Windows event loop policy ----
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ---- .env ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- Whisper ASR ----
import whisper as _whisper_module

SAMPLE_RATE  = 16000
WHISPER_LANG = os.getenv("WHISPER_LANG", "zh")
print("[WHISPER] Loading model...")
_whisper_model = _whisper_module.load_model(os.getenv("WHISPER_MODEL", "base"))
print(f"[WHISPER] Model loaded: {os.getenv('WHISPER_MODEL', 'base')}")

# ---- 引入我们的模块 ----
# ---- Import our modules ----
from audio_stream import (
    register_stream_route,         # 挂 /stream.wav
    # mount /stream.wav route
    broadcast_pcm16_realtime,      # 实时向连接分发 16k PCM
    # distribute 16k PCM to connections in real time
    hard_reset_audio,              # 音频+AI 播放总闸
    # master switch for audio + AI playback
    BYTES_PER_20MS_16K,
    is_playing_now,
    current_ai_task,
)
from omni_client import stream_chat, OmniStreamPiece
from asr_core import (
    ASRCallback,
    set_current_recognition,
    stop_current_recognition,
    has_hotword,
)
from audio_player import initialize_audio_system, play_voice_text

# ---- 同步录制器 ----
# ---- Synchronous recorder ----
import sync_recorder
import signal
import atexit

# ---- IMU UDP ----
UDP_IP   = "0.0.0.0"
UDP_PORT = 12345

app = FastAPI()

# ====== 状态与容器 ======
# ====== State and containers ======
app.mount("/static", StaticFiles(directory="static"), name="static")

ui_clients: Dict[int, WebSocket] = {}
current_partial: str = ""
recent_finals: List[str] = []
RECENT_MAX = 50
last_frames: Deque[Tuple[float, bytes]] = deque(maxlen=10)

camera_viewers: Set[WebSocket] = set()
esp32_camera_ws: Optional[WebSocket] = None
imu_ws_clients: Set[WebSocket] = set()
esp32_audio_ws: Optional[WebSocket] = None

# 【新增】盲道导航相关全局变量
# [New] Global variables for blind-path navigation
blind_path_navigator = None
navigation_active = False
yolo_seg_model = None
obstacle_detector = None

# 【新增】过马路导航相关全局变量
# [New] Global variables for cross-street navigation
cross_street_navigator = None
cross_street_active = False
orchestrator = None  # 新增
# new addition

# 【新增】omni对话状态标志
# [New] Omni conversation state flags
omni_conversation_active = False  # 标记omni对话是否正在进行
# flag indicating whether omni conversation is in progress
omni_previous_nav_state = None  # 保存omni激活前的导航状态，用于恢复
# saves the navigation state before omni activation, for restoring later

# 【新增】模型加载函数
# [New] Model loading function
def load_navigation_models():
    """加载盲道导航所需的模型"""
    # Load models required for blind-path navigation
    global yolo_seg_model, obstacle_detector

    try:
        seg_model_path = os.getenv("BLIND_PATH_MODEL", r"C:\Users\Administrator\Desktop\rebuild1002\model\yolo-seg.pt")

        if os.path.exists(seg_model_path):
            print(f"[NAVIGATION] Model file found, loading...")
            # 模型文件存在，开始加载
            yolo_seg_model = YOLO(seg_model_path)

            # 强制放到 GPU
            # Force model onto GPU
            if torch.cuda.is_available():
                yolo_seg_model.to("cuda")
                print(f"[NAVIGATION] Blind-path segmentation model loaded on GPU: {yolo_seg_model.device}")
            else:
                print("[NAVIGATION] CUDA not available, model stays on CPU")
                # CUDA不可用，模型仍在CPU

            # 测试模型是否能正常运行
            # Test whether the model runs correctly
            try:
                test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                results = yolo_seg_model.predict(
                    test_img,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    verbose=False
                )
                print(f"[NAVIGATION] Model test successful, class count: {len(yolo_seg_model.names) if hasattr(yolo_seg_model, 'names') else 'unknown'}")
                # 模型测试成功，支持的类别数
                if hasattr(yolo_seg_model, 'names'):
                    print(f"[NAVIGATION] Model classes: {yolo_seg_model.names}")
                    # 模型类别
            except Exception as e:
                print(f"[NAVIGATION] Model test failed: {e}")
                # 模型测试失败
        else:
            print(f"[NAVIGATION] Error: model file not found: {seg_model_path}")
            # 错误：找不到模型文件
            print(f"[NAVIGATION] Current working directory: {os.getcwd()}")
            # 当前工作目录
            print(f"[NAVIGATION] Please check the file path")
            # 请检查文件路径是否正确
            
        # 【修改开始】使用 ObstacleDetectorClient 替代直接的 YOLO
        # [Edit] Use ObstacleDetectorClient instead of direct YOLO
        obstacle_model_path = os.getenv("OBSTACLE_MODEL", r"C:\Users\Administrator\Desktop\rebuild1002\model\yoloe-11l-seg.pt")
        print(f"[NAVIGATION] Loading obstacle detection model: {obstacle_model_path}")
        # 尝试加载障碍物检测模型
        
        if os.path.exists(obstacle_model_path):
            print(f"[NAVIGATION] Obstacle model file found, loading...")
            # 障碍物检测模型文件存在，开始加载
            try:
                # 使用 ObstacleDetectorClient 封装的 YOLO-E
                # Use YOLO-E wrapped in ObstacleDetectorClient
                obstacle_detector = ObstacleDetectorClient(model_path=obstacle_model_path)
                print(f"[NAVIGATION] ========== YOLO-E obstacle detector loaded successfully ==========")
                # YOLO-E 障碍物检测器加载成功
                
                # 检查模型是否成功加载
                # Check whether the model was loaded successfully
                if hasattr(obstacle_detector, 'model') and obstacle_detector.model is not None:
                    print(f"[NAVIGATION] YOLO-E model initialized")
                    # YOLO-E 模型已初始化
                    print(f"[NAVIGATION] Model device: {next(obstacle_detector.model.parameters()).device}")
                    # 模型设备
                else:
                    print(f"[NAVIGATION] Warning: YOLO-E model initialization abnormal")
                    # 警告：YOLO-E 模型初始化异常
                
                # 检查白名单是否成功加载
                # Check whether the whitelist was loaded successfully
                if hasattr(obstacle_detector, 'WHITELIST_CLASSES'):
                    print(f"[NAVIGATION] Whitelist class count: {len(obstacle_detector.WHITELIST_CLASSES)}")
                    # 白名单类别数
                    print(f"[NAVIGATION] First 10 whitelist classes: {', '.join(obstacle_detector.WHITELIST_CLASSES[:10])}")
                    # 白名单前10个类别
                else:
                    print(f"[NAVIGATION] Warning: whitelist classes not defined")
                    # 警告：白名单类别未定义
                
                # 检查文本特征是否成功预计算
                # Check whether text features were pre-computed successfully
                if hasattr(obstacle_detector, 'whitelist_embeddings') and obstacle_detector.whitelist_embeddings is not None:
                    print(f"[NAVIGATION] YOLO-E text features pre-computed")
                    # YOLO-E 文本特征已预计算
                    print(f"[NAVIGATION] Text feature tensor shape: {obstacle_detector.whitelist_embeddings.shape if hasattr(obstacle_detector.whitelist_embeddings, 'shape') else 'unknown'}")
                    # 文本特征张量形状
                else:
                    print(f"[NAVIGATION] Warning: YOLO-E text features not pre-computed")
                    # 警告：YOLO-E 文本特征未预计算
                
                # 测试障碍物检测功能
                # Test obstacle detection functionality
                print(f"[NAVIGATION] Testing YOLO-E detection...")
                # 开始测试 YOLO-E 检测功能
                try:
                    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                    # 在测试图像中画一个白色矩形，模拟一个物体
                    # Draw a white rectangle in the test image to simulate an object
                    cv2.rectangle(test_img, (200, 200), (400, 400), (255, 255, 255), -1)
                    
                    # 测试检测（不提供 path_mask）
                    # Test detection (without providing path_mask)
                    test_results = obstacle_detector.detect(test_img)
                    print(f"[NAVIGATION] YOLO-E detection test successful!")
                    # YOLO-E 检测测试成功
                    print(f"[NAVIGATION] Test detection result count: {len(test_results)}")
                    # 测试检测结果数
                    
                    if len(test_results) > 0:
                        print(f"[NAVIGATION] Objects detected in test:")
                        # 测试检测到的物体
                        for i, obj in enumerate(test_results):
                            print(f"  - Object {i+1}: {obj.get('name', 'unknown')}, "
                                  f"area ratio: {obj.get('area_ratio', 0):.3f}, "
                                  f"position: ({obj.get('center_x', 0):.0f}, {obj.get('center_y', 0):.0f})")
                except Exception as e:
                    print(f"[NAVIGATION] YOLO-E detection test failed: {e}")
                    # YOLO-E 检测测试失败
                    import traceback
                    traceback.print_exc()
                
                print(f"[NAVIGATION] ========== YOLO-E obstacle detector load complete ==========")
                # YOLO-E 障碍物检测器加载完成
                
            except Exception as e:
                print(f"[NAVIGATION] Obstacle detector load failed: {e}")
                # 障碍物检测器加载失败
                import traceback
                traceback.print_exc()
                obstacle_detector = None
        else:
            print(f"[NAVIGATION] Warning: obstacle model file not found: {obstacle_model_path}")
            # 警告：找不到障碍物检测模型文件
        
    except Exception as e:
        print(f"[NAVIGATION] Model loading failed: {e}")
        # 模型加载失败
        import traceback
        traceback.print_exc()

# 在程序启动时加载模型
# Load models at program startup
print("[NAVIGATION] Starting navigation model load...")
# 开始加载导航模型
# load_navigation_models()  # No need for cooking demo
print(f"[NAVIGATION] Model load complete - yolo_seg_model: {yolo_seg_model is not None}")
# 模型加载完成

# 【新增】启动同步录制
# [New] Start synchronous recording
print("[RECORDER] Starting synchronous recording system...")
# 启动同步录制系统
sync_recorder.start_recording()
print("[RECORDER] Recording system started, will auto-save video and audio")
# 录制系统已启动，将自动保存视频和音频

# 【新增】注册退出处理器，确保Ctrl+C时保存录制文件
# [New] Register exit handler to ensure recording is saved on Ctrl+C
def cleanup_on_exit():
    """程序退出时的清理工作"""
    # Cleanup work when the program exits
    print("\n[SYSTEM] Shutting down recorder...")
    # 正在关闭录制器
    try:
        sync_recorder.stop_recording()
        print("[SYSTEM] Recording file saved")
        # 录制文件已保存
    except Exception as e:
        print(f"[SYSTEM] Error shutting down recorder: {e}")
        # 关闭录制器时出错

def signal_handler(sig, frame):
    """处理Ctrl+C信号"""
    # Handle Ctrl+C signal
    print("\n[SYSTEM] Interrupt signal received, shutting down safely...")
    # 收到中断信号，正在安全退出
    cleanup_on_exit()
    import sys
    sys.exit(0)

# 注册信号处理器
# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # 终止信号 / termination signal
atexit.register(cleanup_on_exit)               # 正常退出时也调用 / also called on normal exit

print("[RECORDER] Exit handler registered - recording will auto-save on Ctrl+C")
# 已注册退出处理器 - Ctrl+C时会自动保存录制文件


# 【新增】预加载红绿灯检测模型（避免进入WAIT_TRAFFIC_LIGHT状态时卡顿）
# [New] Pre-load traffic light detection model (avoid stutter when entering WAIT_TRAFFIC_LIGHT state)
try:
    import trafficlight_detection
    print("[TRAFFIC_LIGHT] Pre-loading traffic light detection model...")
    # 开始预加载红绿灯检测模型
    if trafficlight_detection.init_model():
        print("[TRAFFIC_LIGHT] Traffic light detection model pre-loaded successfully")
        # 红绿灯检测模型预加载成功
        # 执行一次测试推理，完全预热模型
        # Run one test inference to fully warm up the model
        try:
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = trafficlight_detection.process_single_frame(test_img)
            print("[TRAFFIC_LIGHT] Model warmup complete")
            # 模型预热完成
        except Exception as e:
            print(f"[TRAFFIC_LIGHT] Model warmup failed: {e}")
            # 模型预热失败
    else:
        print("[TRAFFIC_LIGHT] Traffic light detection model pre-load failed")
        # 红绿灯检测模型预加载失败
except Exception as e:
    print(f"[TRAFFIC_LIGHT] Traffic light model pre-load error: {e}")
    # 红绿灯模型预加载出错

# ============== 关键：系统级"硬重置"总闸 =================
# ============== Key: system-level "hard reset" master switch =================
interrupt_lock = asyncio.Lock()

# ============== YOLO媒体线程管理 =================
# ============== YOLO media thread management =================
yolomedia_thread: Optional[threading.Thread] = None
yolomedia_stop_event = threading.Event()
yolomedia_running = False
yolomedia_sending_frames = False  # 新增：标记YOLO是否已经开始发送处理后的帧
# flag indicating whether YOLO has started sending processed frames

# 物品名称到YOLO类别的映射
# Mapping from item names to YOLO class names
ITEM_TO_CLASS_MAP = {
    "红牛": "Red_Bull",
    "AD钙奶": "AD_milk",
    "ad钙奶": "AD_milk",
    "钙奶": "AD_milk",
}

async def ui_broadcast_raw(msg: str):
    dead = []
    for k, ws in list(ui_clients.items()):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(k)
    for k in dead:
        ui_clients.pop(k, None)


async def ui_broadcast_partial(text: str):
    global current_partial
    current_partial = text
    await ui_broadcast_raw("PARTIAL:" + text)

async def ui_broadcast_final(text: str):
    global current_partial, recent_finals
    current_partial = ""
    recent_finals.append(text)
    if len(recent_finals) > RECENT_MAX:
        recent_finals = recent_finals[-RECENT_MAX:]
    await ui_broadcast_raw("FINAL:" + text)
    print(f"[ASR/AI FINAL] {text}", flush=True)

async def full_system_reset(reason: str = ""):
    """
    Return to the state right after startup:
    1) Stop playback + cancel AI task + cut all /stream.wav (hard_reset_audio)
    2) Stop ASR real-time recognition stream (critical)
    3) Clear UI state
    4) Clear recent camera frames (avoid splicing old frames into the next round)
    5) Notify ESP32: RESET (optional)
    """
    # 1) 音频&AI / Audio & AI
    await hard_reset_audio(reason or "full_system_reset")

    # 2) ASR
    await stop_current_recognition()

    # 3) UI
    global current_partial, recent_finals
    current_partial = ""
    recent_finals = []

    # 4) 相机帧 / Camera frames
    try:
        last_frames.clear()
    except Exception:
        pass

    # 5) 通知 ESP32 / Notify ESP32
    try:
        if esp32_audio_ws and (esp32_audio_ws.client_state == WebSocketState.CONNECTED):
            await esp32_audio_ws.send_text("RESET")
    except Exception:
        pass

    print("[SYSTEM] full reset done.", flush=True)

# ========= 启动/停止 YOLO 媒体处理 =========
# ========= Start/stop YOLO media processing =========
def start_yolomedia_with_target(target_name: str):
    """启动yolomedia线程，搜索指定物品"""
    # Start yolomedia thread to search for the specified item
    global yolomedia_thread, yolomedia_stop_event, yolomedia_running, yolomedia_sending_frames
    
    # 如果已经在运行，先停止
    # If already running, stop first
    if yolomedia_running:
        stop_yolomedia()
    
    # 查找对应的YOLO类别
    # Look up the corresponding YOLO class
    yolo_class = ITEM_TO_CLASS_MAP.get(target_name, target_name)
    print(f"[YOLOMEDIA] Starting with target: {target_name} -> YOLO class: {yolo_class}", flush=True)
    print(f"[YOLOMEDIA] Available mappings: {ITEM_TO_CLASS_MAP}", flush=True)
    
    yolomedia_stop_event.clear()
    yolomedia_running = True
    yolomedia_sending_frames = False  # 重置发送帧状态 / reset frame-sending state
    
    def _run():
        try:
            # 传递目标类别名和停止事件
            # Pass the target class name and stop event
            yolomedia.main(headless=True, prompt_name=yolo_class, stop_event=yolomedia_stop_event)
        except Exception as e:
            print(f"[YOLOMEDIA] worker stopped: {e}", flush=True)
        finally:
            global yolomedia_running, yolomedia_sending_frames
            yolomedia_running = False
            yolomedia_sending_frames = False
    
    yolomedia_thread = threading.Thread(target=_run, daemon=True)
    yolomedia_thread.start()
    print(f"[YOLOMEDIA] background worker started for: {yolo_class} (initializing, showing raw feed temporarily)", flush=True)
    # 正在初始化，暂时显示原始画面

def stop_yolomedia():
    """停止yolomedia线程"""
    # Stop the yolomedia thread
    global yolomedia_thread, yolomedia_stop_event, yolomedia_running, yolomedia_sending_frames
    
    if yolomedia_running:
        print("[YOLOMEDIA] Stopping worker...", flush=True)
        yolomedia_stop_event.set()
        
        # 等待线程结束（最多等5秒）
        # Wait for thread to end (up to 5 seconds)
        if yolomedia_thread and yolomedia_thread.is_alive():
            yolomedia_thread.join(timeout=5.0)
        
        yolomedia_running = False
        yolomedia_sending_frames = False
        
        # 【新增】如果orchestrator在找物品模式，结束时不自动恢复（由命令控制）
        # [New] If orchestrator is in item search mode, do not auto-restore on exit (controlled by command)
        # 只清理标志位即可 / just clear the flags
        print("[YOLOMEDIA] Worker stopped, waiting for state switch.", flush=True)
        # 等待状态切换

# ========= 自定义的 start_ai_with_text，支持识别特殊命令 =========
# ========= Custom start_ai_with_text, supports special command recognition =========
async def start_ai_with_text_custom(user_text: str):
    """扩展版的AI启动函数，支持识别特殊命令"""
    # Extended AI launch function with special command recognition
    global navigation_active, blind_path_navigator, cross_street_active, cross_street_navigator, orchestrator
    
    # 【修改】在导航模式和红绿灯检测模式下，只有特定词才进入omni对话
    # [Edit] In navigation and traffic light detection modes, only specific keywords trigger omni dialogue
    if orchestrator:
        current_state = orchestrator.get_state()
        # 如果在导航模式或红绿灯检测模式（非CHAT模式）
        # If in navigation or traffic light detection mode (not CHAT mode)
        if current_state not in ["CHAT", "IDLE"]:
            # 检查是否是允许的对话触发词
            # Check if it's an allowed dialogue trigger keyword
            allowed_keywords = ["帮我看", "帮我看下", "帮我找", "找一下", "看看", "识别一下"]
            is_allowed_query = any(keyword in user_text for keyword in allowed_keywords)
            
            # 检查是否是导航控制命令
            # Check if it's a navigation control command
            nav_control_keywords = ["开始过马路", "过马路结束", "开始导航", "盲道导航", "停止导航", "结束导航", 
                                   "检测红绿灯", "看红绿灯", "停止检测", "停止红绿灯"]
            is_nav_control = any(keyword in user_text for keyword in nav_control_keywords)
            
            # 如果既不是允许的查询，也不是导航控制命令，则丢弃
            # If neither an allowed query nor a navigation control command, discard
            if not is_allowed_query and not is_nav_control:
                mode_name = "Traffic light detection" if current_state == "TRAFFIC_LIGHT_DETECTION" else "Navigation"
                print(f"[{mode_name} mode] Discarding non-dialogue voice: {user_text}")
                return  # 直接丢弃，不进入omni / discard directly, do not enter omni
    
    # 【修改】检查是否是过马路相关命令 - 使用orchestrator控制
    # [Edit] Check for cross-street commands - controlled via orchestrator
    if "开始过马路" in user_text or "帮我过马路" in user_text:
        # 【新增】如果正在找物品，先停止
        # [New] If currently searching for an item, stop first
        if yolomedia_running:
            stop_yolomedia()
            print("[ITEM_SEARCH] Switching from item search to cross-street mode")
            # 从找物品模式切换到过马路
        
        if orchestrator:
            orchestrator.start_crossing()
            print(f"[CROSS_STREET] Cross-street mode started, state: {orchestrator.get_state()}")
            # 过马路模式已启动，状态
            play_voice_text("Cross-street mode started.")
            # 过马路模式已启动
            await ui_broadcast_final("[System] Cross-street mode started")
        else:
            print("[CROSS_STREET] Warning: navigation orchestrator not initialized!")
            # 警告：导航统领器未初始化！
            play_voice_text("Failed to start cross-street mode, please try again later.")
            await ui_broadcast_final("[System] Navigation system not ready")
        return
    
    if "过马路结束" in user_text or "结束过马路" in user_text:
        if orchestrator:
            orchestrator.stop_navigation()
            print(f"[CROSS_STREET] Navigation stopped, state: {orchestrator.get_state()}")
            # 导航已停止，状态
            play_voice_text("Navigation stopped.")
            # 已停止导航
            await ui_broadcast_final("[System] Cross-street mode stopped")
        else:
            await ui_broadcast_final("[System] Navigation system not running")
        return
    
    # 【修改】检查是否是红绿灯检测命令 - 实现与盲道导航互斥
    # [Edit] Check for traffic light detection command - mutually exclusive with blind-path navigation
    if "检测红绿灯" in user_text or "看红绿灯" in user_text:
        try:
            import trafficlight_detection
            
            # 切换orchestrator到红绿灯检测模式（暂停盲道导航）
            # Switch orchestrator to traffic light detection mode (pause blind-path navigation)
            if orchestrator:
                orchestrator.start_traffic_light_detection()
                print(f"[TRAFFIC] Switched to traffic light detection mode, state: {orchestrator.get_state()}")
                # 切换到红绿灯检测模式，状态
            
            # 【改进】使用主线程模式而不是独立线程，避免掉帧
            # [Improvement] Use main-thread mode instead of a separate thread to avoid dropped frames
            success = trafficlight_detection.init_model()
            trafficlight_detection.reset_detection_state()
            
            if success:
                await ui_broadcast_final("[System] Traffic light detection started")
            else:
                await ui_broadcast_final("[System] Traffic light model load failed")
        except Exception as e:
            print(f"[TRAFFIC] Failed to start traffic light detection: {e}")
            # 启动红绿灯检测失败
            await ui_broadcast_final(f"[System] Start failed: {e}")
        return
    
    if "停止检测" in user_text or "停止红绿灯" in user_text:
        try:
            # 恢复到对话模式
            # Restore to dialogue mode
            if orchestrator:
                orchestrator.stop_navigation()
                print(f"[TRAFFIC] Traffic light detection stopped, restored to {orchestrator.get_state()} mode")
                # 红绿灯检测停止，恢复到...模式
            
            await ui_broadcast_final("[System] Traffic light detection stopped")
        except Exception as e:
            print(f"[TRAFFIC] Failed to stop traffic light detection: {e}")
            # 停止红绿灯检测失败
            await ui_broadcast_final(f"[System] Stop failed: {e}")
        return
    
    # 【修改】检查是否是导航相关命令 - 使用orchestrator控制
    # [Edit] Check for navigation commands - controlled via orchestrator
    if "开始导航" in user_text or "盲道导航" in user_text or "帮我导航" in user_text:
        # 【新增】如果正在找物品，先停止
        # [New] If currently searching for an item, stop first
        if yolomedia_running:
            stop_yolomedia()
            print("[ITEM_SEARCH] Switching from item search to blind-path navigation")
            # 从找物品模式切换到盲道导航
        
        if orchestrator:
            orchestrator.start_blind_path_navigation()
            print(f"[NAVIGATION] Blind-path navigation started, state: {orchestrator.get_state()}")
            # 盲道导航已启动，状态
            await ui_broadcast_final("[System] Blind-path navigation started")
        else:
            print("[NAVIGATION] Warning: navigation orchestrator not initialized!")
            # 警告：导航统领器未初始化！
            await ui_broadcast_final("[System] Navigation system not ready")
        return
    
    if "停止导航" in user_text or "结束导航" in user_text:
        if orchestrator:
            orchestrator.stop_navigation()
            print(f"[NAVIGATION] Navigation stopped, state: {orchestrator.get_state()}")
            # 导航已停止，状态
            await ui_broadcast_final("[System] Blind-path navigation stopped")
        else:
            await ui_broadcast_final("[System] Navigation system not running")
        return

    nav_cmd_keywords = ["开始过马路", "过马路结束", "开始导航", "盲道导航", "停止导航", "结束导航", "立即通过", "现在通过", "继续"]
    if any(k in user_text for k in nav_cmd_keywords):
        if orchestrator:
            orchestrator.on_voice_command(user_text)
            await ui_broadcast_final("[System] Navigation mode updated")
            # 导航模式已更新
        else:
            await ui_broadcast_final("[System] Navigation orchestrator not initialized")
            # 导航统领器未初始化
        return    

    # 检查是否是"帮我找/识别一下xxx"的命令
    # Check if it's a "help me find/identify xxx" command
    # 扩展正则表达式，支持更多关键词
    # Extended regex to support more keywords
    find_pattern = r"(?:^\s*帮我)?\s*找一下\s*(.+?)(?:。|！|？|$)"
    match = re.search(find_pattern, user_text)
        
    if match and _YOLOMEDIA_AVAILABLE:
        # 提取中文物品名称
        # Extract Chinese item name
        item_cn = match.group(1).strip()
        if item_cn:
            try:
                from qwen_extractor import extract_english_label
                label_en, src = extract_english_label(item_cn)
            except Exception:
                label_en, src = item_cn, "fallback"
            print(f"[COMMAND] Finder request: '{item_cn}' -> '{label_en}' (src={src})", flush=True)

            if orchestrator:
                orchestrator.start_item_search()
                print(f"[ITEM_SEARCH] Switched to item search mode, state: {orchestrator.get_state()}")

            start_yolomedia_with_target(label_en)

            try:
                await ui_broadcast_final(f"[Item Search] Searching for {item_cn}...")
            except Exception:
                pass

            return
    
    # 检查是否是"找到了"的命令
    # Check for "found it" command
    if "找到了" in user_text or "拿到了" in user_text:
        print("[COMMAND] Found command detected", flush=True)
        stop_yolomedia()
        
        # 【新增】停止找物品模式，恢复之前的导航状态
        # [New] Stop item search mode, restore previous navigation state
        if orchestrator:
            orchestrator.stop_item_search(restore_nav=True)
            current_state = orchestrator.get_state()
            print(f"[ITEM_SEARCH] Item search ended, current state: {current_state}")
            # 找物品结束，当前状态
            
            # 根据恢复的状态给出反馈
            # Give feedback based on restored state
            if current_state in ["BLINDPATH_NAV", "SEEKING_CROSSWALK", "WAIT_TRAFFIC_LIGHT", "CROSSING", "SEEKING_NEXT_BLINDPATH"]:
                await ui_broadcast_final("[Item Search] Item found, resuming navigation.")
                # 已找到物品，继续导航
            else:
                await ui_broadcast_final("[Item Search] Item found.")
                # 已找到物品
        else:
            await ui_broadcast_final("[Item Search] Item found.")
        
        return
    
    # 【修改】omni对话开始时，切换到CHAT模式
    # [Edit] When omni dialogue starts, switch to CHAT mode
    global omni_conversation_active, omni_previous_nav_state
    omni_conversation_active = True
    
    # 保存当前导航状态并切换到CHAT模式
    # Save current navigation state and switch to CHAT mode
    if orchestrator:
        current_state = orchestrator.get_state()
        # 只有在导航模式下才需要保存和切换
        # Only save and switch if currently in navigation mode
        if current_state not in ["CHAT", "IDLE"]:
            omni_previous_nav_state = current_state
            orchestrator.force_state("CHAT")
            print(f"[OMNI] Dialogue started, switching from {current_state} to CHAT mode")
            # 对话开始，从...切换到CHAT模式
        else:
            omni_previous_nav_state = None
            print(f"[OMNI] Dialogue started (already in {current_state} mode)")
            # 对话开始（当前已在...模式）
    
    # 如果不是特殊命令，执行原有的AI对话逻辑
    # If not a special command, run the original AI dialogue logic
    # 但如果yolomedia正在运行，暂时不处理普通对话
    # But if yolomedia is running, skip normal dialogue for now
    if yolomedia_running:
        print("[AI] YOLO media is running, skipping normal AI response", flush=True)
        return
    
    await start_ai_with_text(user_text)

# ========= Omni 播放启动 =========
# ========= Omni playback launch =========
async def start_ai_with_text(user_text: str):
    """硬重置后，开启新的 AI 语音输出。"""
    # After a hard reset, start new AI voice output
    async def _runner():
        txt_buf: List[str] = []
        rate_state = None

        # 组装（图像+文本）
        # Assemble (image + text) content
        content_list = []
        if last_frames:
            try:
                _, jpeg_bytes = last_frames[-1]
                img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            except Exception:
                pass
        content_list.append({"type": "text", "text": user_text})

        try:
            async for piece in stream_chat(content_list, audio_format="wav"):
                # 文本增量（仅 UI）
                # Text delta (UI only)
                if piece.text_delta:
                    txt_buf.append(piece.text_delta)
                    try:
                        await ui_broadcast_partial("[AI] " + "".join(txt_buf))
                    except Exception:
                        pass

                # 音频分片：Omni 返回 24k (PCM16) 的 wav audio.data（Base64）；下行需要 8k PCM16
                # Audio chunk: Omni returns 24k (PCM16) wav audio.data (Base64); downlink needs 8k PCM16
                if piece.audio_b64:
                    try:
                        pcm24 = base64.b64decode(piece.audio_b64)
                    except Exception:
                        pcm24 = b""
                    if pcm24:
                        # 24k → 8k (使用ratecv保证音调和速度不变)
                        # 24k → 8k (use ratecv to preserve pitch and speed)
                        pcm8k, rate_state = audioop.ratecv(pcm24, 2, 1, 24000, 8000, rate_state)
                        pcm8k = audioop.mul(pcm8k, 2, 0.60)
                        if pcm8k:
                            await broadcast_pcm16_realtime(pcm8k)

        except asyncio.CancelledError:
            # 被新一轮打断 / interrupted by a new round
            raise
        except Exception as e:
            try:
                await ui_broadcast_final(f"[AI] Error occurred: {e}")
                # 发生错误
            except Exception:
                pass
        finally:
            # 【修改】标记omni对话结束，恢复之前的导航模式
            # [Edit] Mark omni dialogue as ended, restore previous navigation mode
            global omni_conversation_active, omni_previous_nav_state
            omni_conversation_active = False
            
            # 恢复之前的导航状态
            # Restore previous navigation state
            if orchestrator and omni_previous_nav_state:
                orchestrator.force_state(omni_previous_nav_state)
                print(f"[OMNI] Dialogue ended, restored to {omni_previous_nav_state} mode")
                # 对话结束，恢复到...模式
                omni_previous_nav_state = None
            else:
                print(f"[OMNI] Dialogue ended (no navigation state to restore)")
                # 对话结束（无需恢复导航状态）
            
            # 自然结束时，给当前连接一个 "完结" 信号
            # On natural end, send a "finished" signal to current connections
            from audio_stream import stream_clients
            for sc in list(stream_clients):
                if not sc.abort_event.is_set():
                    try: sc.q.put_nowait(b"\x00"*BYTES_PER_20MS_16K)  # 一帧静音 / one silent frame
                    except Exception: pass
                    try: sc.q.put_nowait(None)
                    except Exception: pass

            final_text = ("".join(txt_buf)).strip() or "（Empty response）"
            # 空响应
            try:
                await ui_broadcast_final("[AI] " + final_text)
            except Exception:
                pass

    # 真正启动前先硬重置，保证**绝无**旧音频残留
    # Hard reset before actually starting — guarantees no old audio residue
    await hard_reset_audio("start_ai_with_text")
    loop = asyncio.get_running_loop()
    from audio_stream import current_ai_task as _task_holder
    from audio_stream import __dict__ as _as_dict
    # 设置模块内的 current_ai_task / set module-level current_ai_task
    task = loop.create_task(_runner())
    _as_dict["current_ai_task"] = task

# ---------- 页面 / 健康 ----------
# ---------- Page / health ----------
@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/api/health", response_class=PlainTextResponse)
def health():
    return "OK"

# 注册 /stream.wav / Register /stream.wav
register_stream_route(app)

# ---------- WebSocket：WebUI 文本（ASR/AI 状态推送） ----------
# ---------- WebSocket: WebUI text (ASR/AI status push) ----------
@app.websocket("/ws_ui")
async def ws_ui(ws: WebSocket):
    await ws.accept()
    ui_clients[id(ws)] = ws
    try:
        init = {"partial": current_partial, "finals": recent_finals[-10:]}
        await ws.send_text("INIT:" + json.dumps(init, ensure_ascii=False))
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        ui_clients.pop(id(ws), None)

# ---------- WebSocket：ESP32 音频入口（ASR 上行） ----------
# ---------- WebSocket: ESP32 audio input (ASR uplink) ----------
@app.websocket("/ws_audio")
async def ws_audio(ws: WebSocket):
    global esp32_audio_ws
    esp32_audio_ws = ws
    await ws.accept()
    print("\n[AUDIO] client connected")
    pcm_buffer = bytearray()
    recording = False

    try:
        while True:
            if WebSocketState and ws.client_state != WebSocketState.CONNECTED:
                break
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                if "Cannot call \"receive\"" in str(e):
                    break
                raise

            if "text" in msg and msg["text"] is not None:
                raw = (msg["text"] or "").strip()
                cmd = raw.upper()

                if cmd == "START":
                    print("[AUDIO] START received")
                    pcm_buffer = bytearray()
                    recording = True
                    await set_current_recognition(None)
                    await ui_broadcast_partial("(Audio receiving started...)")
                    await ws.send_text("OK:STARTED")

                elif cmd == "STOP":
                    print("[AUDIO] STOP received")
                    recording = False
                    await ws.send_text("OK:STOPPED")

                    if pcm_buffer:
                        audio_f32 = np.frombuffer(bytes(pcm_buffer), dtype=np.int16).astype(np.float32) / 32768.0
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(
                            None,
                            lambda: _whisper_model.transcribe(audio_f32, language=WHISPER_LANG)
                        )
                        text = result.get("text", "").strip()

                        if text:
                            print(f"[WHISPER] {text}", flush=True)
                            await ui_broadcast_final(text)

                            if has_hotword(text):
                                async with interrupt_lock:
                                    print(f"[ASR HOTWORD] '{text}' -> FULL RESET", flush=True)
                                    await full_system_reset("Hotword interrupt")
                            elif not is_playing_now():
                                async with interrupt_lock:
                                    print(f"[LLM INPUT TEXT] {text}", flush=True)
                                    await start_ai_with_text_custom(text)

                    pcm_buffer = bytearray()

                elif raw.startswith("PROMPT:"):
                    # 设备端主动发起一轮：同样使用"先硬重置后播放"的强语义
                    # Device-side initiated round: also uses "hard reset first, then play" semantics
                    text = raw[len("PROMPT:"):].strip()
                    if text:
                        async with interrupt_lock:
                            await start_ai_with_text_custom(text)
                        await ws.send_text("OK:PROMPT_ACCEPTED")
                    else:
                        await ws.send_text("ERR:EMPTY_PROMPT")

            elif "bytes" in msg and msg["bytes"] is not None:
                if recording:
                    pcm_buffer += msg["bytes"]

    except Exception as e:
        print(f"\n[WS ERROR] {e}")
    finally:
        await set_current_recognition(None)
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        if esp32_audio_ws is ws:
            esp32_audio_ws = None
        print("[WS] connection closed")

# ---------- WebSocket：ESP32 相机入口（JPEG 二进制） ----------
# ---------- WebSocket: ESP32 camera input (JPEG binary) ----------
@app.websocket("/ws/camera")
async def ws_camera_esp(ws: WebSocket):
    global esp32_camera_ws, blind_path_navigator, cross_street_navigator, cross_street_active, navigation_active, orchestrator
    if esp32_camera_ws is not None:
        await ws.close(code=1013)
        return
    esp32_camera_ws = ws
    await ws.accept()
    print("[CAMERA] ESP32 connected")
    
    # 【新增】初始化盲道导航器
    # [New] Initialize blind-path navigator
    if blind_path_navigator is None and yolo_seg_model is not None:
        blind_path_navigator = BlindPathNavigator(yolo_seg_model, obstacle_detector)
        print("[NAVIGATION] Blind-path navigator initialized")
        # 盲道导航器已初始化
    else:
        if blind_path_navigator is not None:
            print("[NAVIGATION] Navigator already exists, no re-initialization needed")
            # 导航器已存在，无需重新初始化
        elif yolo_seg_model is None:
            print("[NAVIGATION] Warning: YOLO model not loaded, cannot initialize navigator")
            # 警告：YOLO模型未加载，无法初始化导航器
    
    # 【新增】初始化过马路导航器
    # [New] Initialize cross-street navigator
    if cross_street_navigator is None:
        if yolo_seg_model:
            cross_street_navigator = CrossStreetNavigator(
                seg_model=yolo_seg_model,
                coco_model=None,
                obs_model=None
            )
            print("[CROSS_STREET] Cross-street navigator initialized (simplified - zebra crossing detection only)")
            # 过马路导航器已初始化（简化版 - 仅斑马线检测）
        else:
            print("[CROSS_STREET] Error: missing segmentation model, cannot initialize cross-street navigator")
            # 错误：缺少分割模型，无法初始化过马路导航器
            if not yolo_seg_model:
                print("[CROSS_STREET] - Missing segmentation model (yolo_seg_model)")
                # 缺少分割模型
            if not obstacle_detector:
                print("[CROSS_STREET] - Missing obstacle detector (obstacle_detector)")
                # 缺少障碍物检测器
    
    if orchestrator is None and blind_path_navigator is not None and cross_street_navigator is not None:
        orchestrator = NavigationMaster(blind_path_navigator, cross_street_navigator)
        print("[NAV MASTER] Navigation orchestrator state machine initialized (managed mode)")
        # 统领状态机已初始化（托管模式）
    frame_counter = 0  # 添加帧计数器 / add frame counter
    
    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                frame_counter += 1
                
                # 【新增】录制原始帧
                # [New] Record raw frame
                try:
                    sync_recorder.record_frame(data)
                except Exception as e:
                    if frame_counter % 100 == 0:  # 避免日志刷屏 / avoid log flooding
                        print(f"[RECORDER] Frame recording failed: {e}")
                        # 录制帧失败
                
                try:
                    last_frames.append((time.time(), data))
                except Exception:
                    pass
                
                # 推送到bridge_io（供yolomedia使用）
                # Push to bridge_io (for yolomedia use)
                bridge_io.push_raw_jpeg(data)
                
                # 【调试】检查导航条件
                # [Debug] Check navigation conditions
                if frame_counter % 30 == 0:  # 每30帧输出一次 / output every 30 frames
                    state_dbg = orchestrator.get_state() if orchestrator else "N/A"
                    print(f"[NAVIGATION DEBUG] Frame:{frame_counter}, state={state_dbg}, yolomedia_running={yolomedia_running}")
                
                # 统一解码（添加更严格的异常处理）
                # Unified decode (with stricter exception handling)
                try:
                    arr = np.frombuffer(data, dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    # 验证解码结果 / Validate decode result
                    if bgr is None or bgr.size == 0:
                        if frame_counter % 30 == 0:
                            print(f"[JPEG] Decode failed: data length={len(data)}")
                            # 解码失败：数据长度
                        bgr = None
                except Exception as e:
                    if frame_counter % 30 == 0:
                        print(f"[JPEG] Decode exception: {e}")
                        # 解码异常
                    bgr = None

                # 【托管】优先交给统领状态机（寻物未占用画面时）
                # [Managed] Give priority to navigation orchestrator (when item search is not occupying the screen)
                # 【修改】找物品模式时不执行导航处理，让yolomedia接管画面
                # [Edit] In item search mode, skip navigation processing and let yolomedia take over the display
                if orchestrator and not yolomedia_running and bgr is not None:
                    current_state = orchestrator.get_state()
                    
                    # 【新增】找物品模式：不处理画面，等待yolomedia发送处理后的帧
                    # [New] Item search mode: don't process frame, wait for yolomedia to send processed frames
                    if current_state == "ITEM_SEARCH":
                        # 找物品模式下，如果yolomedia还没开始发送帧，先显示原始画面
                        # In item search mode, show raw feed if yolomedia hasn't started sending frames yet
                        if not yolomedia_sending_frames and camera_viewers:
                            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            if ok:
                                jpeg_data = enc.tobytes()
                                dead = []
                                for viewer_ws in list(camera_viewers):
                                    try:
                                        await viewer_ws.send_bytes(jpeg_data)
                                    except Exception:
                                        dead.append(viewer_ws)
                                for d in dead:
                                    camera_viewers.discard(d)
                        continue  # 跳过后续的导航处理 / skip subsequent navigation processing
                    
                    out_img = bgr
                    try:
                        # 【新增】检查是否在红绿灯检测模式
                        # [New] Check if in traffic light detection mode
                        if current_state == "TRAFFIC_LIGHT_DETECTION":
                            # 红绿灯检测模式：在主线程中直接处理，避免掉帧
                            # Traffic light detection mode: process directly in main thread to avoid dropped frames
                            import trafficlight_detection
                            result = trafficlight_detection.process_single_frame(bgr, ui_broadcast_callback=ui_broadcast_final)
                            out_img = result['vis_image'] if result['vis_image'] is not None else bgr
                        else:
                            # 其他模式：正常的导航处理
                            # Other modes: normal navigation processing
                            res = orchestrator.process_frame(bgr)

                            # 语音引导（内部已节流）
                            # Voice guidance (internally throttled)
                            # 注：omni对话时已切换到CHAT模式，不会生成导航语音
                            # Note: during omni dialogue, already switched to CHAT mode, no navigation voice generated
                            if res.guidance_text:
                                try:
                                    play_voice_text(res.guidance_text)
                                    await ui_broadcast_final(f"[Navigation] {res.guidance_text}")
                                    # 导航
                                except Exception:
                                    pass

                            # 输出图像 / Output image
                            out_img = res.annotated_image if res.annotated_image is not None else bgr
                    except Exception as e:
                        if frame_counter % 100 == 0:
                            print(f"[NAV MASTER] Error processing frame: {e}")
                            # 处理帧时出错

                    # 广播图像 / Broadcast image
                    if camera_viewers and out_img is not None:
                        ok, enc = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            jpeg_data = enc.tobytes()
                            dead = []
                            for viewer_ws in list(camera_viewers):
                                try:
                                    await viewer_ws.send_bytes(jpeg_data)
                                except Exception:
                                    dead.append(viewer_ws)
                            for d in dead:
                                camera_viewers.discard(d)
                    continue

                # 【回退】寻物占用或者未解码成功，按原始画面回传
                # [Fallback] Item search active or decode failed, return raw feed
                if not yolomedia_sending_frames and camera_viewers:
                    try:
                        if bgr is None:
                            arr = np.frombuffer(data, dtype=np.uint8)
                            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if bgr is not None:
                            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            if ok:
                                jpeg_data = enc.tobytes()
                                dead = []
                                for viewer_ws in list(camera_viewers):
                                    try:
                                        await viewer_ws.send_bytes(jpeg_data)
                                    except Exception:
                                        dead.append(viewer_ws)
                                for ws in dead:
                                    camera_viewers.discard(ws)
                    except Exception as e:
                        print(f"[CAMERA] Broadcast error: {e}")

            elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[CAMERA ERROR] {e}")
    finally:
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        esp32_camera_ws = None
        print("[CAMERA] ESP32 disconnected")
        
        # 【新增】清理导航状态
        # [New] Clean up navigation state
        if blind_path_navigator:
            blind_path_navigator.reset()
        if cross_street_navigator:
            cross_street_navigator.reset()
        if orchestrator:
            orchestrator.reset()
            print("[NAV MASTER] Orchestrator reset")
            # 统领器已重置

# ---------- WebSocket：浏览器订阅相机帧 ----------
# ---------- WebSocket: browser camera frame subscription ----------
@app.websocket("/ws/viewer")
async def ws_viewer(ws: WebSocket):
    await ws.accept()
    camera_viewers.add(ws)
    print(f"[VIEWER] Browser connected. Total viewers: {len(camera_viewers)}", flush=True)
    try:
        while True:
            # 保持连接活跃 / Keep connection alive
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        print("[VIEWER] Browser disconnected", flush=True)
    finally:
        try: 
            camera_viewers.remove(ws)
        except Exception: 
            pass
        print(f"[VIEWER] Removed. Total viewers: {len(camera_viewers)}", flush=True)

# ---------- WebSocket：浏览器订阅 IMU ----------
# ---------- WebSocket: browser IMU subscription ----------
@app.websocket("/ws")
async def ws_imu(ws: WebSocket):
    await ws.accept()
    imu_ws_clients.add(ws)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        imu_ws_clients.discard(ws)

async def imu_broadcast(msg: str):
    if not imu_ws_clients: return
    dead = []
    for ws in list(imu_ws_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        imu_ws_clients.discard(ws)

# ---------- 服务端 IMU 估计（原样保留） ----------
# ---------- Server-side IMU estimation (kept as-is) ----------
from math import atan2, hypot, pi
GRAV_BETA   = 0.98
STILL_W     = 0.4
YAW_DB      = 0.08
YAW_LEAK    = 0.2
ANG_EMA     = 0.15
AUTO_REZERO = True
USE_PROJ    = True
FREEZE_STILL= True
G     = 9.807
A_TOL = 0.08 * G
gLP = {"x":0.0, "y":0.0, "z":0.0}
gOff= {"x":0.0, "y":0.0, "z":0.0}
BIAS_ALPHA = 0.002
yaw  = 0.0
Rf = Pf = Yf = 0.0
ref = {"roll":0.0, "pitch":0.0, "yaw":0.0}
holdStart = 0.0
isStill   = False
last_ts_imu = 0.0
last_wall = 0.0
imu_store: List[Dict[str, Any]] = []

def _wrap180(a: float) -> float:
    a = a % 360.0
    if a >= 180.0: a -= 360.0
    if a < -180.0: a += 360.0
    return a

def process_imu_and_maybe_store(d: Dict[str, Any]):
    global gLP, gOff, yaw, Rf, Pf, Yf, ref, holdStart, isStill, last_ts_imu, last_wall

    t_ms = float(d.get("ts", 0.0))
    now_wall = time.monotonic()
    if t_ms <= 0.0:
        t_ms = (now_wall * 1000.0)
    if last_ts_imu <= 0.0 or t_ms <= last_ts_imu or (t_ms - last_ts_imu) > 3000.0:
        dt = 0.02
    else:
        dt = (t_ms - last_ts_imu) / 1000.0
    last_ts_imu = t_ms

    ax = float(((d.get("accel") or {}).get("x", 0.0)))
    ay = float(((d.get("accel") or {}).get("y", 0.0)))
    az = float(((d.get("accel") or {}).get("z", 0.0)))
    wx = float(((d.get("gyro")  or {}).get("x", 0.0)))
    wy = float(((d.get("gyro")  or {}).get("y", 0.0)))
    wz = float(((d.get("gyro")  or {}).get("z", 0.0)))

    gLP["x"] = GRAV_BETA * gLP["x"] + (1.0 - GRAV_BETA) * ax
    gLP["y"] = GRAV_BETA * gLP["y"] + (1.0 - GRAV_BETA) * ay
    gLP["z"] = GRAV_BETA * gLP["z"] + (1.0 - GRAV_BETA) * az
    gmag = hypot(gLP["x"], gLP["y"], gLP["z"]) or 1.0
    gHat = {"x": gLP["x"]/gmag, "y": gLP["y"]/gmag, "z": gLP["z"]/gmag}

    roll  = (atan2(az, ay)   * 180.0 / pi)
    pitch = (atan2(-ax, ay)  * 180.0 / pi)

    aNorm = hypot(ax, ay, az); wNorm = hypot(wx, wy, wz)
    nearFlat = (abs(roll) < 2.0 and abs(pitch) < 2.0)
    stillCond = (abs(aNorm - G) < A_TOL) and (wNorm < STILL_W)

    if stillCond:
        if holdStart <= 0.0: holdStart = t_ms
        if not isStill and (t_ms - holdStart) > 350.0: isStill = True
        gOff["x"] = (1.0 - BIAS_ALPHA)*gOff["x"] + BIAS_ALPHA*wx
        gOff["y"] = (1.0 - BIAS_ALPHA)*gOff["y"] + BIAS_ALPHA*wy
        gOff["z"] = (1.0 - BIAS_ALPHA)*gOff["z"] + BIAS_ALPHA*wz
    else:
        holdStart = 0.0; isStill = False

    if USE_PROJ:
        yawdot = ((wx - gOff["x"])*gHat["x"] + (wy - gOff["y"])*gHat["y"] + (wz - gOff["z"])*gHat["z"])
    else:
        yawdot = (wy - gOff["y"])

    if abs(yawdot) < YAW_DB: yawdot = 0.0
    if FREEZE_STILL and stillCond: yawdot = 0.0

    yaw = _wrap180(yaw + yawdot * dt)

    if (YAW_LEAK > 0.0) and nearFlat and stillCond and abs(yaw) > 0.0:
        step = YAW_LEAK * dt * (-1.0 if yaw > 0 else (1.0 if yaw < 0 else 0.0))
        if abs(yaw) <= abs(step): yaw = 0.0
        else: yaw += step

    global Rf, Pf, Yf, ref, last_wall
    Rf = ANG_EMA * roll  + (1.0 - ANG_EMA) * Rf
    Pf = ANG_EMA * pitch + (1.0 - ANG_EMA) * Pf
    Yf = ANG_EMA * yaw   + (1.0 - ANG_EMA) * Yf

    if AUTO_REZERO and nearFlat and (wNorm < STILL_W):
        if holdStart <= 0.0: holdStart = t_ms
        if not isStill and (t_ms - holdStart) > 350.0:
            ref.update({"roll": Rf, "pitch": Pf, "yaw": Yf})
            isStill = True

    R = _wrap180(Rf - ref["roll"])
    P = _wrap180(Pf - ref["pitch"])
    Y = _wrap180(Yf - ref["yaw"])

    now_wall = time.monotonic()
    if last_wall <= 0.0 or (now_wall - last_wall) >= 0.100:
        last_wall = now_wall
        item = {
            "ts": t_ms/1000.0,
            "angles": {"roll": R, "pitch": P, "yaw": Y},
            "accel":  {"x": ax, "y": ay, "z": az},
            "gyro":   {"x": wx, "y": wy, "z": wz},
        }
        imu_store.append(item)

# ---------- UDP 接收 IMU 并转发 ----------
# ---------- Receive IMU via UDP and forward ----------
class UDPProto(asyncio.DatagramProtocol):
    def connection_made(self, transport):
        print(f"[UDP] listening on {UDP_IP}:{UDP_PORT}")
    def datagram_received(self, data, addr):
        try:
            s = data.decode('utf-8', errors='ignore').strip()
            d = json.loads(s)
            if 'ts' not in d and 'timestamp_ms' in d:
                d['ts'] = d.pop('timestamp_ms')
            process_imu_and_maybe_store(d)
            asyncio.create_task(imu_broadcast(json.dumps(d)))
        except Exception:
            pass


# === 新增：注册给 bridge_io 的发送回调（把 JPEG 广播给 /ws/viewer） ===
# === New: register send callback for bridge_io (broadcast JPEG to /ws/viewer) ===
@app.on_event("startup")
async def on_startup_register_bridge_sender():
    # 保存主线程的事件循环
    # Save the main thread's event loop
    main_loop = asyncio.get_event_loop()
    
    def _sender(jpeg_bytes: bytes):
        # 注意：这个函数可能在非协程线程里被调用，需要切回主事件循环
        # Note: this function may be called from a non-coroutine thread; must switch back to main event loop
        try:
            # 检查事件循环状态，避免在关闭时发送
            # Check event loop state to avoid sending during shutdown
            if main_loop.is_closed():
                return
            
            # 标记YOLO已经开始发送处理后的帧
            # Mark that YOLO has started sending processed frames
            global yolomedia_sending_frames
            if not yolomedia_sending_frames:
                yolomedia_sending_frames = True
                print("[YOLOMEDIA] Started sending processed frames, switching to YOLO feed", flush=True)
                # 开始发送处理后的帧，切换到YOLO画面
            
            async def _broadcast():
                if not camera_viewers:
                    return
                dead = []
                for ws in list(camera_viewers):
                    try:
                        await ws.send_bytes(jpeg_bytes)
                    except Exception as e:
                        dead.append(ws)
                for ws in dead:
                    try:
                        camera_viewers.remove(ws)
                    except Exception:
                        pass
            
            # 使用保存的主线程事件循环
            # Use the saved main thread event loop
            future = asyncio.run_coroutine_threadsafe(_broadcast(), main_loop)
            # 不等待结果，避免阻塞生产线程
            # Don't wait for result, avoid blocking producer thread
        except Exception as e:
            # 只在非预期错误时打印日志
            # Only log unexpected errors
            if "Event loop is closed" not in str(e):
                print(f"[DEBUG] _sender error: {e}", flush=True)

    bridge_io.set_sender(_sender)

@app.on_event("startup")
async def on_startup_init_audio():
    """启动时初始化音频系统"""
    # Initialize audio system on startup
    def _init():
        try:
            initialize_audio_system()
        except Exception as e:
            print(f"[AUDIO] Initialization failed: {e}")
            # 初始化失败
    
    threading.Thread(target=_init, daemon=True).start()

@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(lambda: UDPProto(), local_addr=(UDP_IP, UDP_PORT))

@app.on_event("shutdown")
async def on_shutdown():
    """应用关闭时的清理工作"""
    # Cleanup work when the application shuts down
    print("[SHUTDOWN] Starting resource cleanup...")
    # 开始清理资源
    
    stop_yolomedia()
    await hard_reset_audio("shutdown")
    
    print("[SHUTDOWN] Resource cleanup complete")
    # 资源清理完成

# --- 导出接口（可选） ---
# --- Export interfaces (optional) ---
def get_last_frames():
    return last_frames

def get_camera_ws():
    return esp32_camera_ws

if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=8081,
        log_level="warning", access_log=False,
        loop="asyncio", workers=1, reload=False
    )