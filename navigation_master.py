# navigation_master.py
# -*- coding: utf-8 -*-
import time
import math
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Deque, List, Tuple
from collections import deque

# Workflow imports (decoupled from existing files)
from workflow_blindpath import BlindPathNavigator, ProcessingResult as BlindResult
from workflow_crossstreet import CrossStreetNavigator, CrossStreetResult as CrossResult

# ========== State constants ==========
IDLE = "IDLE"                          # Idle / disabled
CHAT = "CHAT"                          # Chat mode (no navigation, returns raw frame only)
BLINDPATH_NAV = "BLINDPATH_NAV"        # Walking tactile path (uses BlindPathNavigator)
SEEKING_CROSSWALK = "SEEKING_CROSSWALK"# Crosswalk spotted during blind-path phase, aligning/approaching
WAIT_TRAFFIC_LIGHT = "WAIT_TRAFFIC_LIGHT" # Reached crosswalk, waiting for traffic light (placeholder)
CROSSING = "CROSSING"                  # Crossing the street (uses CrossStreetNavigator)
SEEKING_NEXT_BLINDPATH = "SEEKING_NEXT_BLINDPATH" # After crossing, seeking next tactile path entry
RECOVERY = "RECOVERY"                  # Fallback/recovery (when perception is temporarily lost)
TRAFFIC_LIGHT_DETECTION = "TRAFFIC_LIGHT_DETECTION"  # Traffic light detection mode
ITEM_SEARCH = "ITEM_SEARCH"            # Item-search mode (navigation paused, yolomedia handles frames)

# ========== Return type ==========
@dataclass
class OrchestratorResult:
    annotated_image: Optional[np.ndarray]
    guidance_text: str
    state: str
    extras: Dict[str, Any]

# ========== Utilities: signal smoothing / majority vote ==========
class MajorityFilter:
    def __init__(self, size: int = 8):
        self.buf: Deque[str] = deque(maxlen=size)

    def push(self, v: str):
        self.buf.append(v)

    def majority(self) -> str:
        if not self.buf:
            return "unknown"
        cnt = {}
        for v in self.buf:
            cnt[v] = cnt.get(v, 0) + 1
        # Robust sort: give 'unknown' the lowest weight
        items = sorted(cnt.items(), key=lambda x: (0 if x[0]=="unknown" else 1, x[1]), reverse=True)
        return items[0][0]

    def history(self) -> List[str]:
        return list(self.buf)

    def clear(self):
        self.buf.clear()

# ========== Traffic light detection ==========
class TrafficLightDetector:
    “””
    Traffic light detector:
    1) Prefers yoloe_backend-style detection (if available).
    2) Fallback: uses HSV colour heuristics on the upper half of the frame to find bright red/yellow/green blobs.
    Output: ('red'|'green'|'yellow'|'unknown', meta)
    “””
    def __init__(self):
        self.has_backend = False
        self.backend = None
        try:
            # Dynamic import (adjust to match your local yoloe_backend interface)
            import yoloe_backend as _yeb  # noqa
            self.backend = _yeb
            self.has_backend = True
        except Exception:
            self.has_backend = False
            self.backend = None

    def _try_backend(self, bgr: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        “””
        Attempt to call the yoloe_backend-style interface with lenient dispatch:
        - First tries backend.detect(image, target_classes=['traffic light'])
        - Falls back to backend.infer_image(image) and filters for 'traffic light'
        - Returns 'unknown' if both fail.
        Expected result entries should contain a bbox or mask; extend colour classification logic as needed.
        “””
        if not self.has_backend or self.backend is None:
            return "unknown", {"reason": "backend_not_available"}

        res = None
        try:
            if hasattr(self.backend, "detect"):
                # expected: detect returns [{'name': 'traffic light', 'box':[x1,y1,x2,y2], ...}, ...]
                res = self.backend.detect(bgr, target_classes=["traffic light"])
            elif hasattr(self.backend, "infer_image"):
                # expected: infer_image returns [{'label': 'traffic light', 'bbox': [x1,y1,x2,y2], ...}, ...]
                res = self.backend.infer_image(bgr)
            else:
                return "unknown", {"reason": "backend_no_suitable_api"}
        except Exception as e:
            return "unknown", {"reason": f"backend_failed:{e}"}

        if not res or len(res) == 0:
            return "unknown", {"reason": "no_detection"}

        # Use the largest box as the primary light, then run HSV color classification
        H, W = bgr.shape[:2]
        best = None
        best_area = 0
        boxes = []
        for item in res:
            # Normalize box field names
            if "box" in item and isinstance(item["box"], (list, tuple)) and len(item["box"]) == 4:
                x1, y1, x2, y2 = item["box"]
            elif "bbox" in item and isinstance(item["bbox"], (list, tuple)) and len(item["bbox"]) == 4:
                x1, y1, x2, y2 = item["bbox"]
            else:
                continue
            x1 = int(max(0, min(W-1, x1))); x2 = int(max(0, min(W-1, x2)))
            y1 = int(max(0, min(H-1, y1))); y2 = int(max(0, min(H-1, y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            boxes.append((x1, y1, x2, y2, area))
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)

        if best is None:
            return "unknown", {"reason": "no_valid_bbox", "raw": len(res)}

        x1, y1, x2, y2 = best
        roi = bgr[y1:y2, x1:x2]
        color = self._classify_color_hsv(roi)
        return color, {"bbox": best, "count": len(res), "boxes": boxes}

    def _classify_color_hsv(self, roi_bgr: np.ndarray) -> str:
        “””Simple HSV threshold-based red/yellow/green classification on the ROI; picks the dominant colour by area.”””
        if roi_bgr is None or roi_bgr.size == 0:
            return “unknown”
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # Red range (two segments)
        lower_red1 = np.array([0, 80, 120]); upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 80, 120]); upper_red2 = np.array([180, 255, 255])
        mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)

        # Green
        lower_green = np.array([40, 60, 120]); upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Yellow
        lower_yellow = np.array([18, 80, 150]); upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Area thresholds (relative to ROI)
        total = roi_bgr.shape[0] * roi_bgr.shape[1] + 1e-6
        r_ratio = float(np.count_nonzero(mask_red)) / total
        g_ratio = float(np.count_nonzero(mask_green)) / total
        y_ratio = float(np.count_nonzero(mask_yellow)) / total

        # Suppress weak responses from noisy backgrounds
        thr = 0.03
        candidates = []
        if r_ratio > thr: candidates.append(("red", r_ratio))
        if g_ratio > thr: candidates.append(("green", g_ratio))
        if y_ratio > thr: candidates.append(("yellow", y_ratio))
        if not candidates:
            return "unknown"
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def detect(self, bgr: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        “””
        Main entry: try backend first; fall back to upper-half HSV blob detection (no bounding box required).
        “””
        # 1) Try backend
        if self.has_backend:
            color, meta = self._try_backend(bgr)
            if color != “unknown”:
                return color, {“method”: “backend”, **meta}

        # 2) Fallback: upper half HSV clustering + connected components, find largest blob
        H, W = bgr.shape[:2]
        roi = bgr[:int(H * 0.5), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Brightness threshold (suppress dark areas / car headlights)
        v = hsv[:, :, 2]
        bright = (v > 140).astype(np.uint8) * 255

        # Rough color classification
        col = self._classify_color_hsv(roi)
        return col, {“method”: “fallback”, “note”: “no_backend”, “bright_ratio”: float(np.mean(bright > 0))}

# ========== Visual utilities ==========
def _color_bgr(name: str) -> Tuple[int, int, int]:
    if name == "red": return (0, 0, 255)
    if name == "green": return (0, 255, 0)
    if name == "yellow": return (0, 255, 255)
    if name == "blue": return (255, 0, 0)
    if name == "orange": return (0, 165, 255)
    if name == "cyan": return (255, 255, 0)
    if name == "magenta": return (255, 0, 255)
    if name == "gray": return (128, 128, 128)
    if name == "white": return (255, 255, 255)
    return (200, 200, 200)

def _put_text(img, text, org, color=(255,255,255), scale=0.7, thick=2, outline=True):
    if outline:
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx==0 and dy==0: continue
                cv2.putText(img, text, (org[0]+dx, org[1]+dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

def _draw_badge(img, text, pos=(10, 28), fg="white", bg="blue"):
    color_fg = _color_bgr(fg); color_bg = _color_bgr(bg)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x, y = pos
    pad = 6
    cv2.rectangle(img, (x-4, y-th-pad), (x+tw+8, y+pad//2), color_bg, -1)
    _put_text(img, text, (x, y), color=color_fg, scale=0.6, thick=2, outline=False)

def _draw_state_panel(img, kv: Dict[str, Any], pos=(10, 60)):
    x, y = pos
    line_h = 22
    for i, (k, v) in enumerate(kv.items()):
        _put_text(img, f"{k}: {v}", (x, y + i*line_h), color=(255,255,255), scale=0.6, thick=2)

def _draw_frame_border(img, color=(0,255,0), thickness=3):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0,0), (w-1, h-1), color, thickness)

def _draw_progress_bar(img, ratio: float, pos=(10, 90), size=(180, 10), color="cyan"):
    ratio = max(0.0, min(1.0, float(ratio)))
    x, y = pos
    w, h = size
    cv2.rectangle(img, (x, y), (x+w, y+h), (80,80,80), 1)
    cv2.rectangle(img, (x+1, y+1), (x+1+int((w-2)*ratio), y+h-1), _color_bgr(color), -1)

# ========== Orchestrator ==========
class NavigationMaster:
    def __init__(self,
                 blind_nav: BlindPathNavigator,
                 cross_nav: CrossStreetNavigator,
                 *,
                 min_tts_interval: float = 1.2):
        self.blind = blind_nav
        self.cross = cross_nav
        self.state = IDLE
        self.last_guidance_ts = 0.0
        self.min_tts_interval = min_tts_interval

        # Debounce / stability counters
        self.cnt_crosswalk_seen = 0         # Crosswalk seen from blind-path side (approaching/ready)
        self.cnt_align_ready = 0            # Crosswalk ready + alignment achieved
        self.cnt_cross_end = 0              # End-of-crossing condition count
        self.cnt_lost = 0                   # Perception-lost count (triggers RECOVERY)

        # Cooldown period to prevent state jitter
        self.cooldown_until = 0.0

        # Emergency recovery target state
        self.prev_target_state = BLINDPATH_NAV

        # Traffic light
        self.tld = TrafficLightDetector()
        self.tl_major = MajorityFilter(size=8)
        self.tl_last_color = "unknown"

        # Parameters (can be tuned on-site)
        self.FRAMES_CROSS_SEEN = 8
        self.FRAMES_ALIGN_READY = 12
        self.FRAMES_CROSS_END = 12
        self.FRAMES_NEXT_BLIND_OK = 8
        self.FRAMES_LOST_MAX = 45

        self.ANGLE_ALIGN_THR_DEG = 12.0
        self.OFFSET_ALIGN_THR = 0.15

        self.COOLDOWN_SEC = 0.6
        
        # Item-search state management
        self.prev_nav_state_before_search = None  # Previous nav state before entering item-search (for restoration)

    # ----- External interface -----
    def get_state(self) -> str:
        return self.state

    def start_blind_path_navigation(self):
        """Start blind-path navigation mode."""
        self.state = BLINDPATH_NAV
        self.cooldown_until = time.time() + self.COOLDOWN_SEC
        if self.blind:
            self.blind.reset()

    def stop_navigation(self):
        """Stop navigation and return to chat mode."""
        self.state = CHAT
        self.cooldown_until = time.time() + self.COOLDOWN_SEC
        if self.blind:
            self.blind.reset()

    def start_crossing(self):
        """Start crosswalk mode."""
        self.state = CROSSING
        self.cooldown_until = time.time() + self.COOLDOWN_SEC
        if self.cross:
            self.cross.reset()

    def start_traffic_light_detection(self):
        """Start traffic light detection mode."""
        self.state = TRAFFIC_LIGHT_DETECTION
        self.cooldown_until = time.time() + self.COOLDOWN_SEC

    def is_in_navigation_mode(self):
        """Check whether in navigation mode (not chat mode)."""
        return self.state not in ["CHAT", "IDLE", "TRAFFIC_LIGHT_DETECTION", "ITEM_SEARCH"]

    def start_item_search(self):
        """Start item-search mode, pausing current navigation."""
        # Save current navigation state (if navigating)
        if self.state in [BLINDPATH_NAV, SEEKING_CROSSWALK, WAIT_TRAFFIC_LIGHT, CROSSING, SEEKING_NEXT_BLINDPATH]:
            self.prev_nav_state_before_search = self.state
            print(f"[NAV MASTER] Pausing navigation state {self.state}, switching to item-search mode")
        else:
            self.prev_nav_state_before_search = None

        self.state = ITEM_SEARCH
        self.cooldown_until = time.time() + self.COOLDOWN_SEC

    def stop_item_search(self, restore_nav: bool = True):
        """Stop item-search mode."""
        # Restore previous navigation state if requested
        if restore_nav and self.prev_nav_state_before_search:
            self.state = self.prev_nav_state_before_search
            print(f"[NAV MASTER] Item search ended, restoring navigation state {self.state}")
            self.prev_nav_state_before_search = None
        else:
            # Otherwise return to chat mode
            self.state = CHAT
            print(f"[NAV MASTER] Item search ended, returning to chat mode")

        self.cooldown_until = time.time() + self.COOLDOWN_SEC

    def force_state(self, s: str):
        self.state = s
        self.cooldown_until = time.time() + self.COOLDOWN_SEC

    def on_voice_command(self, text: str):
        t = (text or "").strip()
        if "开始过马路" in t:
            # enter wait state or cross directly (low-traffic environments)
            if self.state in (BLINDPATH_NAV, SEEKING_CROSSWALK, WAIT_TRAFFIC_LIGHT, IDLE, RECOVERY, SEEKING_NEXT_BLINDPATH):
                self.state = WAIT_TRAFFIC_LIGHT
                self.cooldown_until = time.time() + self.COOLDOWN_SEC
        elif "立即通过" in t or "现在通过" in t:
            self.state = CROSSING
            self.cooldown_until = time.time() + self.COOLDOWN_SEC
        elif "停止" in t or "结束" in t:
            self.state = IDLE
        elif "继续" in t:
            if self.state == IDLE:
                self.state = BLINDPATH_NAV

    def reset(self):
        self.state = IDLE
        self.cnt_crosswalk_seen = 0
        self.cnt_align_ready = 0
        self.cnt_cross_end = 0
        self.cnt_lost = 0
        self.tl_major.clear()
        self.tl_last_color = "unknown"
        self.prev_target_state = BLINDPATH_NAV
        self._last_wait_light_announce = 0  # Reset wait-for-green-light announcement timestamp
        try:
            self.blind.reset()
        except Exception:
            pass
        try:
            self.cross.reset()
        except Exception:
            pass

    # ----- Internal utilities -----
    def _say(self, now: float, text: str) -> str:
        if not text:
            return ""
        if now - self.last_guidance_ts >= self.min_tts_interval:
            self.last_guidance_ts = now
            return text
        return ""

    def _draw_tl_status(self, img: np.ndarray, color: str, meta: Dict[str, Any]):
        if img is None:
            return
        color_bgr = _color_bgr(color)
        cv2.circle(img, (24, 24), 10, color_bgr, -1)
        _put_text(img, f"Light: {color}", (40, 30), color=color_bgr, scale=0.6, thick=2, outline=False)
        if meta and "bbox" in meta:
            x1, y1, x2, y2 = meta["bbox"]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, 2)

        hist = self.tl_major.history()
        if hist:
            x0, y0 = 10, 50
            r = 6
            gap = 16
            for i, hcol in enumerate(hist[-12:]):
                cv2.circle(img, (x0 + i*gap, y0), r, _color_bgr(hcol), -1)
            _put_text(img, "Light history", (x0, y0+20), color=(255,255,255), scale=0.5, thick=1)

    # ----- Main loop -----
    def process_frame(self, bgr: np.ndarray) -> OrchestratorResult:
        now = time.time()

        # IDLE state defaults to CHAT mode
        if self.state == IDLE:
            self.state = CHAT
            self.cooldown_until = now + self.COOLDOWN_SEC

        # CHAT mode: return raw frame only, no navigation
        if self.state == CHAT:
            return OrchestratorResult(
                annotated_image=bgr,
                guidance_text="",
                state="CHAT",
                extras={"mode": "Chat mode"}
            )

        # Traffic light detection mode: return raw frame only
        if self.state == TRAFFIC_LIGHT_DETECTION:
            return OrchestratorResult(
                annotated_image=bgr,
                guidance_text="",
                state="TRAFFIC_LIGHT_DETECTION",
                extras={"mode": "Traffic light detection mode"}
            )

        # Item-search mode: return raw frame only, handled by yolomedia
        if self.state == ITEM_SEARCH:
            return OrchestratorResult(
                annotated_image=bgr,
                guidance_text="",
                state="ITEM_SEARCH",
                extras={"mode": "Item search mode", "prev_nav_state": self.prev_nav_state_before_search}
            )

        # During cooldown: continue outputting frames but avoid instant state switches
        in_cooldown = now < self.cooldown_until

        # Per-state processing
        if self.state in (BLINDPATH_NAV, SEEKING_CROSSWALK, SEEKING_NEXT_BLINDPATH, RECOVERY):
            # --- Blind-path side --- always call blind-path navigator
            try:
                bres: BlindResult = self.blind.process_frame(bgr)
            except Exception as e:
                # Exception → enter recovery mode
                self.state = RECOVERY
                self.cnt_lost += 5
                ann_err = bgr.copy()
                return OrchestratorResult(ann_err, self._say(now, ""), self.state, {"error": str(e)})

            ann = bres.annotated_image if bres.annotated_image is not None else bgr.copy()
            say = bres.guidance_text or ""

            state_info = bres.state_info or {}
            cross_stage = state_info.get("crosswalk_stage", "not_detected")
            blind_state = state_info.get("state", "UNKNOWN")
            angle = float(state_info.get("last_angle", 0.0))
            center_x_ratio = float(state_info.get("last_center_x_ratio", 0.5))

            # --- Blind-path → crosswalk detected (approaching/ready)
            if self.state == BLINDPATH_NAV:
                if cross_stage in ("approaching", "ready"):
                    self.cnt_crosswalk_seen += 1
                else:
                    self.cnt_crosswalk_seen = max(0, self.cnt_crosswalk_seen - 1)

                if self.cnt_crosswalk_seen >= self.FRAMES_CROSS_SEEN and not in_cooldown:
                    self.state = SEEKING_CROSSWALK
                    self.cooldown_until = now + self.COOLDOWN_SEC
                    say = "Approaching crosswalk, aligning your direction."

            # --- Alignment stage: using angle and offset from blind's internal crosswalk_tracker
            elif self.state == SEEKING_CROSSWALK:
                aligned = (abs(angle) <= self.ANGLE_ALIGN_THR_DEG and abs(center_x_ratio - 0.5) <= self.OFFSET_ALIGN_THR)
                if cross_stage == "ready" and aligned:
                    self.cnt_align_ready += 1
                else:
                    self.cnt_align_ready = max(0, self.cnt_align_ready - 1)

                if self.cnt_align_ready >= self.FRAMES_ALIGN_READY and not in_cooldown:
                    self.state = WAIT_TRAFFIC_LIGHT
                    self.cooldown_until = now + self.COOLDOWN_SEC
                    say = "At the crosswalk, please wait for the traffic light."

                # _draw_frame_border(ann, color=_color_bgr("orange"), thickness=3)

            # --- Post-crossing: seeking next tactile path (boarding flow)
            elif self.state == SEEKING_NEXT_BLINDPATH:
                if blind_state == "NAVIGATING":
                    self.cnt_cross_end += 1
                else:
                    self.cnt_cross_end = max(0, self.cnt_cross_end - 1)
                if self.cnt_cross_end >= self.FRAMES_NEXT_BLIND_OK and not in_cooldown:
                    self.state = BLINDPATH_NAV
                    self.cooldown_until = now + self.COOLDOWN_SEC
                    say = "Direction correct, please continue forward."

            # --- Recovery: return to blind-path once perception is restored
            elif self.state == RECOVERY:
                if blind_state in ("ONBOARDING", "NAVIGATING"):
                    self.state = BLINDPATH_NAV
                    self.cooldown_until = now + self.COOLDOWN_SEC
                    say = ""
                else:
                    say = ""

            # Lost count (fallback)
            if blind_state == "UNKNOWN" and cross_stage == "not_detected":
                self.cnt_lost += 1
            else:
                self.cnt_lost = max(0, self.cnt_lost - 2)
            if self.cnt_lost >= self.FRAMES_LOST_MAX and self.state != RECOVERY:
                self.prev_target_state = self.state
                self.state = RECOVERY
                self.cooldown_until = now + self.COOLDOWN_SEC
                say = "Complex surroundings, entering recovery mode."

            return OrchestratorResult(ann, self._say(now, say), self.state, {"source": "blind", "cross_stage": cross_stage, "blind_state": blind_state})

        if self.state == WAIT_TRAFFIC_LIGHT:
            ann = bgr.copy()
            # Traffic light detection (majority vote + cooldown)
            color, meta = self.tld.detect(bgr)
            self.tl_major.push(color)
            major = self.tl_major.majority()
            self.tl_last_color = major

            say = ""
            if major == "green" and not in_cooldown:
                self.state = CROSSING
                self.cooldown_until = now + self.COOLDOWN_SEC
                say = "Green light stable, start crossing."
            else:
                # Only announce when first entering state or at intervals
                if not hasattr(self, '_last_wait_light_announce'):
                    self._last_wait_light_announce = 0
                if now - self._last_wait_light_announce > 5.0:  # Announce every 5s
                    say = "Waiting for green light..."
                    self._last_wait_light_announce = now



            return OrchestratorResult(ann, self._say(now, say), self.state, {"traffic_light": major})

        if self.state == CROSSING:
            try:
                cres: CrossResult = self.cross.process_frame(bgr)
            except Exception as e:
                self.state = RECOVERY
                ann_err = bgr.copy()
                return OrchestratorResult(ann_err, self._say(now, ""), self.state, {"error": str(e)})

            ann = cres.annotated_image if cres.annotated_image is not None else bgr.copy()
            say = cres.guidance_text or ""

            # Check whether a tactile path was detected
            blind_path_detected = getattr(cres, 'blind_path_detected', False)
            blind_path_guidance = getattr(cres, 'blind_path_guidance', "")

            # If tactile path detected, prioritize path guidance
            if blind_path_detected and blind_path_guidance:
                # If should_switch_to_blindpath (path is close), switch immediately
                if hasattr(cres, "should_switch_to_blindpath") and cres.should_switch_to_blindpath:
                    if not in_cooldown:
                        self.state = BLINDPATH_NAV
                        self.cooldown_until = now + self.COOLDOWN_SEC
                        say = "Reached the tactile path, switching to path navigation."
                        self.cnt_cross_end = 0  # Reset counter
                        # Reset blind-path navigator state
                        if hasattr(self.blind, 'reset'):
                            self.blind.reset()
                else:
                    # Path is still far — continue crossing but give path guidance
                    # guidance is already in cres.guidance_text
                    pass

            # Original end condition: multiple consecutive "seeking crosswalk" frames
            end_hint = False
            if "寻找斑马线" in (say or ""):
                end_hint = True
            # Note: no longer ending crossing solely because of should_switch_to_blindpath
            # if hasattr(cres, "should_switch_to_blindpath") and cres.should_switch_to_blindpath:
            #     end_hint = True

            self.cnt_cross_end = self.cnt_cross_end + 1 if end_hint else max(0, self.cnt_cross_end - 1)

            if self.cnt_cross_end >= self.FRAMES_CROSS_END and not in_cooldown:
                self.state = SEEKING_NEXT_BLINDPATH
                self.cooldown_until = now + self.COOLDOWN_SEC
                say = "Finished crossing, prepare to step onto the pavement."

            return OrchestratorResult(ann, self._say(now, say), self.state, {"source": "cross", "end_cnt": self.cnt_cross_end})

        # Fallback
        ann = bgr.copy()
        return OrchestratorResult(ann, "", self.state, {})


