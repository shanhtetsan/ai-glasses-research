# bridge_io.py
# Minimal bridge: receives raw JPEG → provides BGR frames to external algorithms; external algorithms produce BGR → broadcast to frontend
import threading
from collections import deque
import time
import cv2
import numpy as np

# Raw JPEG frame buffer (keeps only the latest N frames)
_MAX_BUF = 4
_frames = deque(maxlen=_MAX_BUF)
_cond = threading.Condition()

# Callback for sending JPEG to the frontend, registered by app_main.py at startup
_sender_lock = threading.Lock()
_sender_cb = None

# Callback for sending UI text to the frontend (registered by app_main.py at startup)
_ui_sender_lock = threading.Lock()
_ui_sender_cb = None

def set_sender(cb):
    """Called by app_main.py to register a callback: cb(jpeg_bytes)->None"""
    global _sender_cb
    with _sender_lock:
        _sender_cb = cb

def set_ui_sender(cb):
    """Called by app_main.py to register a callback: cb(text:str)->None"""
    global _ui_sender_cb
    with _ui_sender_lock:
        _ui_sender_cb = cb

def push_raw_jpeg(jpeg_bytes: bytes):
    """Called by app_main.py when a /ws/camera frame is received"""
    if not jpeg_bytes:
        return
    with _cond:
        _frames.append((time.time(), jpeg_bytes))
        _cond.notify_all()

def wait_raw_bgr(timeout_sec: float = 0.5):
    """Called by YOLO/MediaPipe scripts: wait and return the latest BGR frame; returns None on timeout"""
    t_end = time.time() + timeout_sec
    last = None
    while time.time() < t_end:
        with _cond:
            if _frames:
                last = _frames[-1]
        if last is None:
            time.sleep(0.01)
            continue
        # Decode JPEG to BGR
        ts, jpeg = last
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            # Mirror at the source if needed
            #bgr = cv2.flip(bgr, 1)
            return bgr
        # Decode failed, wait and retry
        time.sleep(0.01)
    return None

def send_vis_bgr(bgr, quality: int = 80):
    """Called by YOLO/MediaPipe scripts: push the processed frame to the frontend viewer"""
    if bgr is None:
        return
    
    # Encode directly with no additional enhancement
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return
    with _sender_lock:
        cb = _sender_cb
    if cb:
        try:
            cb(enc.tobytes())
        except Exception:
            pass

def send_ui_final(text: str):
    """Push a UI message to the frontend as a final answer (thread-safe callback)"""
    if not text:
        return
    with _ui_sender_lock:
        cb = _ui_sender_cb
    if cb:
        try:
            cb(str(text))
        except Exception:
            pass
