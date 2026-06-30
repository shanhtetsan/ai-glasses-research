# -*- coding: utf-8 -*-
"""
Traffic light detection module — standalone workflow version.
Detects traffic light state in real time using a YOLO model with voice feedback.
Controlled via voice commands "detect traffic light" / "stop detection".
"""

import os
import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO
import bridge_io
from audio_player import play_voice_text  # Uses the unified voice playback interface
import logging

logger = logging.getLogger(__name__)

# ========= Configuration parameters =========
# Override with env var TRAFFIC_LIGHT_MODEL; default is ./model/ relative to this file.
YOLO_MODEL_PATH = os.getenv(
    "TRAFFIC_LIGHT_MODEL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "trafficlight.pt"),
)

# ========= Display parameters =========
CONF_THRESHOLD = 0.25  # Confidence threshold
FONT_SIZE = 20
STROKE_WIDTH = 3

# ========= TTS parameters =========
TTS_INTERVAL_SEC = 2.0  # TTS announcement interval (prevents excessive repetition)
ENABLE_TTS = False  # [Disabled] Traffic-light module does not announce; workflow_crossstreet.py handles all voice output

# ========= Thread control =========
_detection_thread = None
_stop_event = None
_detection_running = False

# ========= Single-frame processing mode =========
_model = None  # Global model instance
_last_tts_ts = 0.0
_last_detected_light = None
_detection_history = []

# ========= Frontend colors (BGR) =========
FRONTEND_COLORS = {
    "text": (230, 237, 243),   # white text
    "red": (0, 0, 255),        # red
    "yellow": (0, 255, 255),   # yellow
    "green": (0, 255, 0),      # green
    "muted": (159, 176, 195),  # gray
}

# Traffic light state to color mapping
LIGHT_COLORS = {
    "stop": FRONTEND_COLORS["red"],
    "countdown_go": FRONTEND_COLORS["yellow"],
    "go": FRONTEND_COLORS["green"],
}

# Traffic light state to label mapping
# Includes only true traffic light classes; excludes crosswalk (crossing) and blank states
LIGHT_NAMES = {
    "stop": "Red",              # Vehicle red light
    "go": "Green",              # Vehicle green light
    "countdown_go": "Yellow",   # Green countdown (shown as yellow)
    "countdown_stop": "Red",    # Red countdown
}

# Traffic light state to voice-file mapping
LIGHT_VOICE_MAP = {
    "stop": "红灯",              # → voice/红灯.WAV
    "go": "绿灯",                # → voice/绿灯.WAV
    "countdown_go": "黄灯",      # → voice/黄灯.WAV (green countdown shown as yellow)
    "countdown_stop": "红灯",    # → voice/红灯.WAV
}

# Classes to filter out (not detected or displayed)
FILTERED_CLASSES = {
    "crossing",          # crosswalk — not needed
    "blank",             # blank
    "countdown_blank"    # countdown blank
}

# UI text management
_UI_LINE = 0
_UI_H = 0
_UI_TR_LINE = 0
_UI_TOP_MARGIN = 12
_UI_RIGHT_MARGIN = 12
UNIFIED_FONT_PX = 12

def ui_reset_overlay(img_h: int):
    """Call once per frame to reset overlay line counter."""
    global _UI_LINE, _UI_H, _UI_TR_LINE
    _UI_LINE = 0
    _UI_TR_LINE = 0
    _UI_H = int(img_h)

def _ui_next_y_top(font_size: int) -> int:
    """Return the y-coordinate for the next line in the top-right corner."""
    global _UI_TR_LINE
    line_gap = max(4, int(font_size * 0.25))
    y_top = _UI_TOP_MARGIN + (_UI_TR_LINE * (font_size + line_gap))
    _UI_TR_LINE += 1
    return y_top

# ======== Text rendering ========
_PIL_OK = False
_FONT_PATH = None

def _init_font():
    global _PIL_OK, _FONT_PATH
    try:
        from PIL import ImageFont
        _PIL_OK = True
    except Exception:
        _PIL_OK = False
        return
    candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\msyh.ttf",
        r"C:\\Windows\\Fonts\\simhei.ttf",
        r"C:\\Windows\\Fonts\\simfang.ttf",
        r"C:\\Windows\\Fonts\\simsun.ttc",
        r"C:\\Windows\\Fonts\\simsunb.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            _FONT_PATH = p
            return
    _PIL_OK = False

_init_font()

def draw_text_cn(img_bgr, text, xy, font_size=20, color=(255,255,255), ui_hint=True):
    """Unified text rendering."""
    color = (255, 255, 255)
    font_size = int(UNIFIED_FONT_PX)

    H, W = img_bgr.shape[:2]
    y_top = _ui_next_y_top(font_size) if ui_hint else xy[1]
    tw = th = 0
    font_obj = None

    if _PIL_OK and _FONT_PATH:
        try:
            from PIL import Image, ImageDraw, ImageFont
            font_obj = ImageFont.truetype(_FONT_PATH, font_size)
            bbox = ImageDraw.Draw(Image.new('RGB', (1,1))).textbbox((0,0), text, font=font_obj)
            tw = max(1, bbox[2] - bbox[0])
            th = max(1, bbox[3] - bbox[1])
        except Exception:
            pass
    
    if _PIL_OK and _FONT_PATH and font_obj is not None:
        try:
            from PIL import Image, ImageDraw
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)
            if ui_hint:
                x = max(8, W - _UI_RIGHT_MARGIN - tw)
                y = y_top
            else:
                x = xy[0]
                y = xy[1]
            draw.text((x, y), text, fill=color, font=font_obj)
            img_bgr[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
            return
        except Exception:
            pass
    
    # OpenCV fallback
    if tw <= 0 or th <= 0:
        scale = font_size/24.0
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    if ui_hint:
        x = max(8, W - _UI_RIGHT_MARGIN - int(tw))
        y_baseline = int(y_top + th)
    else:
        x = xy[0]
        y_baseline = xy[1] + int(th)
    cv2.putText(img_bgr, text, (x, y_baseline), cv2.FONT_HERSHEY_SIMPLEX, font_size/24.0, color, 2, cv2.LINE_AA)

def main(headless: bool = True, stop_event=None):
    """
    Traffic light detection main function.

    Args:
        headless: Whether to run without an OpenCV window.
        stop_event: threading.Event used to stop detection.
    """

    print("[TRAFFIC] Loading YOLO traffic light detection model...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"[TRAFFIC] Model loaded: {YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"[TRAFFIC] Model load failed: {e}")
        return

    # Get class names
    class_names = model.names if hasattr(model, 'names') else {}
    print(f"[TRAFFIC] Model classes: {class_names}")

    # State tracking
    last_tts_ts = 0.0
    last_detected_light = None
    fps_hist = []

    # Stability check: use majority vote rather than consecutive frames
    detection_history = []  # keep last N frames of detections
    HISTORY_SIZE = 5        # keep last 5 frames
    MAJORITY_THRESHOLD = 3  # at least 3 of 5 frames must agree for a stable state

    # Frame statistics
    frame_count = 0
    frame_received_count = 0
    frame_none_count = 0
    last_frame_log_time = time.time()

    print("[TRAFFIC] Waiting for ESP32 frames...")

    try:
        while True:
            # Check stop event
            if stop_event and stop_event.is_set():
                print("[TRAFFIC] Stop event triggered, exiting")
                break

            # Get raw BGR frame from bridge_io (longer timeout for reliability)
            frame = bridge_io.wait_raw_bgr(timeout_sec=2.0)

            frame_count += 1

            if frame is None:
                frame_none_count += 1
                # Print frame stats every 3s
                current_time = time.time()
                if current_time - last_frame_log_time > 3.0:
                    print(f"[TRAFFIC] Frame stats: total={frame_count}, received={frame_received_count}, "
                          f"dropped={frame_none_count}, drop_rate={frame_none_count/frame_count*100:.1f}%")
                    last_frame_log_time = current_time
                
                if headless:
                    cv2.waitKey(1)
                continue
            
            frame_received_count += 1

            # reset UI overlay
            H, W = frame.shape[:2]
            ui_reset_overlay(H)

            vis = frame.copy()
            t_now = time.time()

            # YOLO inference with timing
            inference_start = time.time()
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            inference_time = (time.time() - inference_start) * 1000

            # Monitor inference time
            if inference_time > 100:
                print(f"[TRAFFIC] WARNING: inference took {inference_time:.0f}ms")

            # Process detection results
            detected_light = None
            max_conf = 0.0

            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    # Find the highest-confidence traffic light detection (exclude crosswalk etc.)
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = class_names.get(cls_id, f"class_{cls_id}")
                        class_name_lower = class_name.lower()

                        # Skip unwanted classes
                        if class_name_lower in FILTERED_CLASSES:
                            continue

                        if conf > max_conf:
                            max_conf = conf
                            detected_light = class_name_lower

                    # Draw detection boxes (traffic lights only)
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = class_names.get(cls_id, f"class_{cls_id}")
                        class_name_lower = class_name.lower()
                        
                        # Skip unwanted classes
                        if class_name_lower in FILTERED_CLASSES:
                            continue

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Determine color
                        color = LIGHT_COLORS.get(class_name_lower, FRONTEND_COLORS["text"])

                        # Draw bounding box
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, STROKE_WIDTH)

                        # Draw label using PIL
                        label = f"{LIGHT_NAMES.get(class_name.lower(), class_name)}: {conf:.2f}"

                        if _PIL_OK and _FONT_PATH:
                            try:
                                from PIL import Image, ImageDraw, ImageFont
                                # Draw label with a larger font
                                font_obj = ImageFont.truetype(_FONT_PATH, 20)
                                # Convert to PIL image
                                img_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(img_rgb)
                                draw = ImageDraw.Draw(pil_img)

                                # Compute text dimensions
                                bbox = draw.textbbox((0, 0), label, font=font_obj)
                                text_w = bbox[2] - bbox[0]
                                text_h = bbox[3] - bbox[1]

                                # Label position
                                label_y = max(y1 - text_h - 8, text_h)

                                # Draw background rectangle
                                bg_x1 = x1
                                bg_y1 = label_y - text_h - 4
                                bg_x2 = x1 + text_w + 8
                                bg_y2 = label_y + 4
                                cv2.rectangle(vis, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

                                # Re-convert (rectangle was drawn with OpenCV)
                                img_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(img_rgb)
                                draw = ImageDraw.Draw(pil_img)

                                # [Removed] draw text label
                                # draw.text((x1 + 4, label_y - text_h), label, fill=(0, 0, 0), font=font_obj)

                                # Convert back to OpenCV format
                                vis[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
                            except Exception as e:
                                # [Removed] text label on PIL failure
                                pass
                        else:
                            # [Removed] text label
                            pass

            # Stability check: majority vote rather than consecutive frames
            detection_history.append(detected_light)
            if len(detection_history) > HISTORY_SIZE:
                detection_history.pop(0)

            # Determine whether state is stable (majority vote)
            stable_light = None
            if len(detection_history) >= MAJORITY_THRESHOLD:
                # Count occurrences of each state in the recent N frames
                valid_detections = [d for d in detection_history if d and d in LIGHT_NAMES]
                if len(valid_detections) >= MAJORITY_THRESHOLD:
                    # Find the most common state
                    from collections import Counter
                    counter = Counter(valid_detections)
                    most_common = counter.most_common(1)
                    if most_common and most_common[0][1] >= MAJORITY_THRESHOLD:
                        stable_light = most_common[0][0]
                        # Debug output
                        if frame_received_count % 30 == 0:
                            print(f"[TRAFFIC] History: {detection_history[-5:]}, stable: {stable_light}")

            # [TTS disabled] Detection only — voice handled by workflow_crossstreet.py
            if stable_light:
                # Log state changes (no announcement)
                if stable_light != last_detected_light:
                    last_detected_light = stable_light
                    print(f"[TRAFFIC] Stable state changed: {LIGHT_NAMES[stable_light]} (not announced)")
                    last_tts_ts = t_now
                # Interval elapsed — update timestamp (no announcement)
                elif (t_now - last_tts_ts) > TTS_INTERVAL_SEC:
                    print(f"[TRAFFIC] Stable state: {LIGHT_NAMES[stable_light]} (not announced)")
                    last_tts_ts = t_now

            # [removed] show current detection status
            # if detected_light and detected_light in LIGHT_NAMES:
            #     status_text = f"Detected: {LIGHT_NAMES[detected_light]} ({max_conf:.2f})"
            #     color = LIGHT_COLORS[detected_light]
            # else:
            #     status_text = "Detected: none"
            #     color = FRONTEND_COLORS["muted"]
            # draw_text_cn(vis, status_text, (10, 40), font_size=18, color=color)

            # [removed] show stable state
            # if stable_light:
            #     stable_text = f"Stable state: {LIGHT_NAMES[stable_light]}"
            #     stable_color = LIGHT_COLORS[stable_light]
            # else:
            #     stable_text = f"Stable state: waiting ({len(detection_history)}/{HISTORY_SIZE})"
            #     stable_color = FRONTEND_COLORS["muted"]
            # draw_text_cn(vis, stable_text, (10, 60), font_size=18, color=stable_color)

            # [removed] FPS calculation and display
            # fps_hist.append(t_now)
            # if len(fps_hist) > 30:
            #     fps_hist.pop(0)
            # fps = 0.0 if len(fps_hist) < 2 else (len(fps_hist)-1)/(fps_hist[-1]-fps_hist[0])
            # draw_text_cn(vis, f"FPS: {fps:.1f}", (10, 20), font_size=16, color=FRONTEND_COLORS["text"])

            # send visualisation result to frontend
            bridge_io.send_vis_bgr(vis)

            # display window in non-headless mode
            if not headless:
                cv2.imshow("Traffic Light Detection", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
            else:
                cv2.waitKey(1)

    except Exception as e:
        print(f"[TRAFFIC] Detection error: {e}")
    finally:
        if not headless:
            cv2.destroyAllWindows()
        print("[TRAFFIC] Traffic light detection stopped")


def start_detection():
    """Start traffic light detection (runs in a background thread)."""
    global _detection_thread, _stop_event, _detection_running

    if _detection_running:
        print("[TRAFFIC] Traffic light detection already running")
        return False

    _stop_event = threading.Event()
    _detection_thread = threading.Thread(
        target=main,
        args=(True, _stop_event),  # headless=True, stop_event
        daemon=True,
        name="TrafficLightDetection"
    )
    _detection_thread.start()
    _detection_running = True
    print("[TRAFFIC] Traffic light detection started (background thread)")
    return True

def stop_detection():
    """Stop traffic light detection."""
    global _detection_thread, _stop_event, _detection_running

    if not _detection_running:
        print("[TRAFFIC] Traffic light detection not running")
        return False

    print("[TRAFFIC] Stopping traffic light detection...")
    if _stop_event:
        _stop_event.set()

    if _detection_thread:
        _detection_thread.join(timeout=2.0)
        _detection_thread = None

    _stop_event = None
    _detection_running = False
    print("[TRAFFIC] Traffic light detection stopped")
    return True

def is_detection_running():
    """Check if traffic light detection is running."""
    return _detection_running

def init_model():
    """Initialize the YOLO model (single-frame processing mode)."""
    global _model
    if _model is not None:
        print("[TRAFFIC] Model already loaded")
        return True

    try:
        print("[TRAFFIC] Loading YOLO traffic light detection model...")
        _model = YOLO(YOLO_MODEL_PATH)
        print(f"[TRAFFIC] Model loaded: {YOLO_MODEL_PATH}")
        class_names = _model.names if hasattr(_model, 'names') else {}
        print(f"[TRAFFIC] Model classes: {class_names}")
        return True
    except Exception as e:
        print(f"[TRAFFIC] Model load failed: {e}")
        _model = None
        return False

def process_single_frame(image: np.ndarray, ui_broadcast_callback=None) -> dict:
    """
    Process a single frame (main-thread mode to avoid dropped frames).

    Args:
        image: Input image.
        ui_broadcast_callback: Frontend broadcast callback (to display traffic light state).
    Returns:
        {'vis_image': visualized image, 'detected_light': detected light, 'stable_light': stable state}
    """
    global _model, _last_tts_ts, _last_detected_light, _detection_history

    if _model is None:
        if not init_model():
            return {'vis_image': image, 'detected_light': None, 'stable_light': None}

    vis = image.copy()
    t_now = time.time()

    # YOLO inference
    results = _model(image, conf=CONF_THRESHOLD, verbose=False)

    # Process detection results
    detected_light = None
    max_conf = 0.0
    class_names = _model.names if hasattr(_model, 'names') else {}

    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            # Find highest-confidence traffic light (filter out crosswalk etc.)
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names.get(cls_id, f"class_{cls_id}")
                class_name_lower = class_name.lower()

                # Skip unwanted classes (crosswalk, blank, etc.)
                if class_name_lower in FILTERED_CLASSES:
                    continue

                if conf > max_conf:
                    max_conf = conf
                    detected_light = class_name_lower

            # Draw bounding boxes (traffic lights only, not crosswalk)
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names.get(cls_id, f"class_{cls_id}")
                class_name_lower = class_name.lower()

                # Skip unwanted classes
                if class_name_lower in FILTERED_CLASSES:
                    continue

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Determine color
                color = LIGHT_COLORS.get(class_name_lower, FRONTEND_COLORS["text"])

                # Draw bounding box
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, STROKE_WIDTH)

    # Stability check (relaxed majority vote)
    _detection_history.append(detected_light)
    if len(_detection_history) > 5:
        _detection_history.pop(0)

    stable_light = None
    if len(_detection_history) >= 2:  # Lowered from 3 to 2 frames
        from collections import Counter
        valid_detections = [d for d in _detection_history if d and d in LIGHT_NAMES]
        if len(valid_detections) >= 2:  # Lowered from 3 to 2 frames
            counter = Counter(valid_detections)
            most_common = counter.most_common(1)
            if most_common and most_common[0][1] >= 2:  # Lowered from 3 to 2 occurrences
                stable_light = most_common[0][0]

    # [Debug disabled] print(f"[TRAFFIC-DEBUG] detected={detected_light}, stable={stable_light}, history={_detection_history}")

    # [TTS disabled] Detection only — voice handled by workflow_crossstreet.py
    if stable_light:
        # Log state change (no announcement)
        if stable_light != _last_detected_light:
            _last_detected_light = stable_light
            print(f"[TRAFFIC] Stable state changed: {LIGHT_NAMES[stable_light]} (not announced)")
            _last_tts_ts = t_now
        elif (t_now - _last_tts_ts) > TTS_INTERVAL_SEC:
            # Interval elapsed — update timestamp (no announcement)
            print(f"[TRAFFIC] Stable state: {LIGHT_NAMES[stable_light]} (not announced)")
            _last_tts_ts = t_now

    # [Removed] status text overlay
    # if detected_light and detected_light in LIGHT_NAMES:
    #     status_text = f"{LIGHT_NAMES[detected_light]} ({max_conf:.2f})"
    # else:
    #     status_text = "No detection"
    #
    # if stable_light:
    #     stable_text = f"Stable: {LIGHT_NAMES[stable_light]}"
    # else:
    #     stable_text = f"Waiting for stable ({len(_detection_history)}/5)"
    #
    # cv2.putText(vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    # cv2.putText(vis, stable_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return {
        'vis_image': vis,
        'detected_light': detected_light,
        'stable_light': stable_light
    }

def reset_detection_state():
    """Reset detection state."""
    global _last_tts_ts, _last_detected_light, _detection_history
    _last_tts_ts = 0.0
    _last_detected_light = None
    _detection_history = []
    print("[TRAFFIC] Detection state reset")

if __name__ == "__main__":
    main(headless=False)



