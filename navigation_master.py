# navigation_master.py
# -*- coding: utf-8 -*-
import time
import math
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Deque, List, Tuple
from collections import deque

# 工作流导入（与现有文件解耦）
# Workflow imports (decoupled from existing files)
from workflow_blindpath import BlindPathNavigator, ProcessingResult as BlindResult
from workflow_crossstreet import CrossStreetNavigator, CrossStreetResult as CrossResult

# ========== 状态常量 ==========
# ========== State constants ==========
IDLE = "IDLE"                          # 空闲/未启用 / Idle / not active
CHAT = "CHAT"                          # 对话模式（不进行导航，只返回原始画面）/ Dialogue mode (no navigation, return raw feed)
BLINDPATH_NAV = "BLINDPATH_NAV"        # 正在走盲道（复用 BlindPathNavigator）/ Walking on tactile paving (uses BlindPathNavigator)
SEEKING_CROSSWALK = "SEEKING_CROSSWALK"# 盲道阶段发现斑马线，正对准/靠近 / Crosswalk detected during blind-path, aligning/approaching
WAIT_TRAFFIC_LIGHT = "WAIT_TRAFFIC_LIGHT" # 到达斑马线后等待交通灯（可选/占位）/ Waiting for traffic light after reaching crosswalk (optional/placeholder)
CROSSING = "CROSSING"                  # 正在过马路（复用 CrossStreetNavigator）/ Crossing the street (uses CrossStreetNavigator)
SEEKING_NEXT_BLINDPATH = "SEEKING_NEXT_BLINDPATH" # 过完马路后寻找下一段盲道入口（上盲道）/ Finding next tactile paving entry after crossing
RECOVERY = "RECOVERY"                  # 兜底/恢复（感知暂时丢失时）/ Fallback/recovery (when perception is temporarily lost)
TRAFFIC_LIGHT_DETECTION = "TRAFFIC_LIGHT_DETECTION"  # 红绿灯检测模式 / Traffic light detection mode
ITEM_SEARCH = "ITEM_SEARCH"            # 找物品模式（暂停导航，由yolomedia处理画面）/ Item search mode (navigation paused, yolomedia handles display)

# ========== 返回结构 ==========
# ========== Return structure ==========
@dataclass
class OrchestratorResult:
    annotated_image: Optional[np.ndarray]
    guidance_text: str
    state: str
    extras: Dict[str, Any]

# ========== 实用：信号平滑/多数表决 ==========
# ========== Utility: signal smoothing / majority vote ==========
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
        # 稳健排序：unknown 权重最低
        # Robust sort: "unknown" has lowest weight
        items = sorted(cnt.items(), key=lambda x: (0 if x[0]=="unknown" else 1, x[1]), reverse=True)
        return items[0][0]

    def history(self) -> List[str]:
        return list(self.buf)

    def clear(self):
        self.buf.clear()

# ========== 红绿灯识别 ==========
# ========== Traffic light detection ==========
class TrafficLightDetector:
    """
    红绿灯识别器：
    Traffic light detector:
    1) 优先尝试 yoloe_backend 风格的检测（如可用）；
    1) First tries yoloe_backend-style detection (if available);
    2) 回退：无模型时，使用 HSV 颜色启发式在上半屏寻找亮红/黄/绿的"灯团"。
    2) Fallback: without a model, uses HSV color heuristics on the top half of the screen to find bright red/yellow/green "light blobs".
    输出：('red'|'green'|'yellow'|'unknown', meta)
    Output: ('red'|'green'|'yellow'|'unknown', meta)
    """
    def __init__(self):
        self.has_backend = False
        self.backend = None
        try:
            # 尝试动态导入（根据你本地 yoloe_backend 的接口调整）
            # Try dynamic import (adjust based on your local yoloe_backend interface)
            import yoloe_backend as _yeb  # noqa
            self.backend = _yeb
            self.has_backend = True
        except Exception:
            self.has_backend = False
            self.backend = None

    def _try_backend(self, bgr: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """
        尝试调用 yoloe_backend 风格的接口。由于各项目实现不同，这里做"宽容地调用"：
        Try calling the yoloe_backend-style interface. Since implementations vary, this is a "lenient call":
        - 优先尝试 backend.detect(image, target_classes=['traffic light'])
        - First try backend.detect(image, target_classes=['traffic light'])
        - 次选 backend.infer_image(image) 后在结果中过滤 'traffic light'
        - Fallback: backend.infer_image(image), then filter for 'traffic light'
        - 以上都失败则返回 unknown
        - If both fail, return unknown
        预期结果条目应含 bbox 或 mask，可自行扩展"颜色判定"逻辑（ROI 取样 HSV）
        Expected result entries should contain bbox or mask; color classification logic (ROI HSV sampling) can be extended
        """
        if not self.has_backend or self.backend is None:
            return "unknown", {"reason": "backend_not_available"}

        res = None
        try:
            if hasattr(self.backend, "detect"):
                # 假定 detect 返回 [{'name': 'traffic light', 'box':[x1,y1,x2,y2], ...}, ...]
                # Assumes detect returns [{'name': 'traffic light', 'box':[x1,y1,x2,y2], ...}, ...]
                res = self.backend.detect(bgr, target_classes=["traffic light"])
            elif hasattr(self.backend, "infer_image"):
                # 假定 infer_image 返回 [{'label': 'traffic light', 'bbox': [x1,y1,x2,y2], ...}, ...]
                # Assumes infer_image returns [{'label': 'traffic light', 'bbox': [x1,y1,x2,y2], ...}, ...]
                res = self.backend.infer_image(bgr)
            else:
                return "unknown", {"reason": "backend_no_suitable_api"}
        except Exception as e:
            return "unknown", {"reason": f"backend_failed:{e}"}

        if not res or len(res) == 0:
            return "unknown", {"reason": "no_detection"}

        # 拿到最大框作为主灯，做 HSV 颜色判断
        # Take the largest bounding box as the main light, perform HSV color classification
        H, W = bgr.shape[:2]
        best = None
        best_area = 0
        boxes = []
        for item in res:
            # 统一盒字段
            # Normalize bounding box field
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
        """对 ROI 做 HSV 基于阈值的红/黄/绿简单判定；取面积最大的主色。"""
        # HSV threshold-based red/yellow/green classification on ROI; pick the dominant color by area
        if roi_bgr is None or roi_bgr.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # 红色范围（两段）
        # Red range (two segments)
        lower_red1 = np.array([0, 80, 120]); upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 80, 120]); upper_red2 = np.array([180, 255, 255])
        mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)

        # 绿色 / Green
        lower_green = np.array([40, 60, 120]); upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # 黄色 / Yellow
        lower_yellow = np.array([18, 80, 150]); upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 面积阈值（相对 ROI）
        # Area threshold (relative to ROI)
        total = roi_bgr.shape[0] * roi_bgr.shape[1] + 1e-6
        r_ratio = float(np.count_nonzero(mask_red)) / total
        g_ratio = float(np.count_nonzero(mask_green)) / total
        y_ratio = float(np.count_nonzero(mask_yellow)) / total

        # 简单抑制"脏背景导致的弱响应"
        # Suppress weak responses caused by noisy backgrounds
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
        """
        总入口：先尝试后端；失败则在上半屏自行找"亮色灯团"（无需框）。
        Main entry: try backend first; if it fails, find bright color blobs in the top half of the screen (no bounding box needed).
        """
        # 1) 尝试后端 / Try backend
        if self.has_backend:
            color, meta = self._try_backend(bgr)
            if color != "unknown":
                return color, {"method": "backend", **meta}

        # 2) 回退：上半屏 HSV 聚类 + 连通域，选最大"灯团"判色
        # 2) Fallback: top-half HSV clustering + connected components, pick largest blob for color
        H, W = bgr.shape[:2]
        roi = bgr[:int(H * 0.5), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 高亮阈值（抑制暗部/车灯）
        # Brightness threshold (suppress dark areas / headlights)
        v = hsv[:, :, 2]
        bright = (v > 140).astype(np.uint8) * 255

        # 粗分颜色 / Rough color classification
        col = self._classify_color_hsv(roi)
        return col, {"method": "fallback", "note": "no_backend", "bright_ratio": float(np.mean(bright > 0))}

# ========== 视觉辅助工具 ==========
# ========== Visual utility tools ==========
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

# ========== 统领器 ==========
# ========== Navigation Orchestrator ==========
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

        # 防抖/稳定计数
        # Debounce / stability counters
        self.cnt_crosswalk_seen = 0         # 盲道侧看见斑马线（approaching/ready）/ crosswalk seen during blind-path (approaching/ready)
        self.cnt_align_ready = 0            # 斑马线 ready + 对准达标 / crosswalk ready + alignment satisfied
        self.cnt_cross_end = 0              # 过马路结束条件累计 / accumulated crossing-end condition count
        self.cnt_lost = 0                   # 感知丢失累计（进入 RECOVERY）/ accumulated perception loss count (triggers RECOVERY)

        # 冷却期避免状态抖动
        # Cooldown period to avoid state oscillation
        self.cooldown_until = 0.0

        # 紧急恢复目标
        # Emergency recovery target state
        self.prev_target_state = BLINDPATH_NAV

        # 交通灯 / Traffic light
        self.tld = TrafficLightDetector()
        self.tl_major = MajorityFilter(size=8)
        self.tl_last_color = "unknown"

        # 参数（可按现场再调）
        # Parameters (can be tuned on-site)
        self.FRAMES_CROSS_SEEN = 8
        self.FRAMES_ALIGN_READY = 12
        self.FRAMES_CROSS_END = 12
        self.FRAMES_NEXT_BLIND_OK = 8
        self.FRAMES_LOST_MAX = 45

        self.ANGLE_ALIGN_THR_DEG = 12.0
        self.OFFSET_ALIGN_THR = 0.15

        self.COOLDOWN_SEC = 0.6
        
        # 找物品状态管理
        # Item search state management
        self.prev_nav_state_before_search = None  # 找物品前的导航状态，用于恢复 / navigation state before item search, for restoration

    # ----- 外部交互 -----
    # ----- External interface -----
    def get_state(self) -> str:
        return self.state
    
    def start_blind_path_navigation(self):
        """启动盲道导航模式"""
        # Start blind-path navigation mode
        self.state = BLINDPATH_NAV
        self.cooldown_until = time.time() + self.COOLDOWN_SEC
        if self.blind:
            self.blind.reset()
    
    def stop_navigation(self):
        """停止导航，回到对话模式"""
        # Stop navigation, return to dialogue mode
        self.state = CHAT
        self.cooldown_until = time.time() + self.COOLDOWN_SEC
        if self.blind:
            self.blind.reset()
    
    def start_crossing(self):
        """启动过马路模式"""
        # Start cross-street mode
        self.state = CROSSING
        self.cooldown_until = time.time() + self.COOLDOWN_SEC
        if self.cross:
            self.cross.reset()
    
    def start_traffic_light_detection(self):
        """启动红绿灯检测模式"""
        # Start traffic light detection mode
        self.state = TRAFFIC_LIGHT_DETECTION
        self.cooldown_until = time.time() + self.COOLDOWN_SEC
    
    def is_in_navigation_mode(self):
        """检查是否在导航模式（非对话模式）"""
        # Check whether currently in navigation mode (not dialogue mode)
        return self.state not in ["CHAT", "IDLE", "TRAFFIC_LIGHT_DETECTION", "ITEM_SEARCH"]
    
    def start_item_search(self):
        """启动找物品模式，暂停当前导航"""
        # Start item search mode, pause current navigation
        # 保存当前导航状态（如果在导航中）
        # Save current navigation state (if navigating)
        if self.state in [BLINDPATH_NAV, SEEKING_CROSSWALK, WAIT_TRAFFIC_LIGHT, CROSSING, SEEKING_NEXT_BLINDPATH]:
            self.prev_nav_state_before_search = self.state
            print(f"[NAV MASTER] Pausing navigation state {self.state}, switching to item search mode")
            # 暂停导航状态，切换到找物品模式
        else:
            self.prev_nav_state_before_search = None
        
        self.state = ITEM_SEARCH
        self.cooldown_until = time.time() + self.COOLDOWN_SEC
    
    def stop_item_search(self, restore_nav: bool = True):
        """停止找物品模式"""
        # Stop item search mode
        # 如果需要恢复之前的导航状态
        # If previous navigation state should be restored
        if restore_nav and self.prev_nav_state_before_search:
            self.state = self.prev_nav_state_before_search
            print(f"[NAV MASTER] Item search ended, restored to navigation state {self.state}")
            # 找物品结束，恢复到导航状态
            self.prev_nav_state_before_search = None
        else:
            # 否则回到对话模式 / Otherwise return to dialogue mode
            self.state = CHAT
            print(f"[NAV MASTER] Item search ended, returning to dialogue mode")
            # 找物品结束，回到对话模式
        
        self.cooldown_until = time.time() + self.COOLDOWN_SEC

    def force_state(self, s: str):
        self.state = s
        self.cooldown_until = time.time() + self.COOLDOWN_SEC

    def on_voice_command(self, text: str):
        t = (text or "").strip()
        if "开始过马路" in t:
            # 直接进入等待/或立即过马路（低速环境可直过）
            # Directly enter wait state / or cross immediately (can cross directly in slow-speed environments)
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
        self._last_wait_light_announce = 0  # 重置等待绿灯播报时间 / reset traffic light wait announcement time
        try:
            self.blind.reset()
        except Exception:
            pass
        try:
            self.cross.reset()
        except Exception:
            pass

    # ----- 内部工具 -----
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
        # 角标与文本 / Badge and text
        cv2.circle(img, (24, 24), 10, color_bgr, -1)
        _put_text(img, f"Traffic light: {color}", (40, 30), color=color_bgr, scale=0.6, thick=2, outline=False)
        # 画 bbox（若有）/ Draw bbox (if available)
        if meta and "bbox" in meta:
            x1, y1, x2, y2 = meta["bbox"]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, 2)

        # 多数表决历史（最近8帧）
        # Majority vote history (last 8 frames)
        hist = self.tl_major.history()
        if hist:
            x0, y0 = 10, 50
            r = 6
            gap = 16
            for i, hcol in enumerate(hist[-12:]):
                cv2.circle(img, (x0 + i*gap, y0), r, _color_bgr(hcol), -1)
            _put_text(img, "Light history", (x0, y0+20), color=(255,255,255), scale=0.5, thick=1)
            # 信号历史

    # ----- 主循环 -----
    # ----- Main loop -----
    def process_frame(self, bgr: np.ndarray) -> OrchestratorResult:
        now = time.time()
        
        # 【修改】IDLE状态默认进入CHAT模式，而不是自动开始导航
        # [Edit] IDLE state defaults to CHAT mode instead of auto-starting navigation
        if self.state == IDLE:
            self.state = CHAT
            self.cooldown_until = now + self.COOLDOWN_SEC
        
        # 【新增】CHAT模式：只返回原始画面，不进行导航
        # [New] CHAT mode: return raw feed only, no navigation processing
        if self.state == CHAT:
            return OrchestratorResult(
                annotated_image=bgr,
                guidance_text="",
                state="CHAT",
                extras={"mode": "Dialogue mode"}
                # 对话模式
            )
        
        # 【新增】红绿灯检测模式：只返回原始画面，由红绿灯模块处理
        # [New] Traffic light detection mode: return raw feed, handled by traffic light module
        if self.state == TRAFFIC_LIGHT_DETECTION:
            return OrchestratorResult(
                annotated_image=bgr,
                guidance_text="",
                state="TRAFFIC_LIGHT_DETECTION",
                extras={"mode": "Traffic light detection mode"}
                # 红绿灯检测模式
            )
        
        # 【新增】找物品模式：只返回原始画面，由yolomedia处理
        # [New] Item search mode: return raw feed, handled by yolomedia
        if self.state == ITEM_SEARCH:
            return OrchestratorResult(
                annotated_image=bgr,
                guidance_text="",
                state="ITEM_SEARCH",
                extras={"mode": "Item search mode", "prev_nav_state": self.prev_nav_state_before_search}
                # 找物品模式
            )

        # 冷却期内允许继续输出画面，但避免"瞬时切换"
        # During cooldown, continue outputting frames but avoid instant state switching
        in_cooldown = now < self.cooldown_until

        # 各状态处理
        # Per-state processing
        if self.state in (BLINDPATH_NAV, SEEKING_CROSSWALK, SEEKING_NEXT_BLINDPATH, RECOVERY):
            # —— 盲道侧 —— 统一调用盲道导航器
            # —— Blind-path side —— unified call to blind-path navigator
            try:
                bres: BlindResult = self.blind.process_frame(bgr)
            except Exception as e:
                # 异常 → 进入恢复态 / Exception → enter recovery state
                self.state = RECOVERY
                self.cnt_lost += 5
                ann_err = bgr.copy()
                return OrchestratorResult(ann_err, self._say(now, ""), self.state, {"error": str(e)})

            ann = bres.annotated_image if bres.annotated_image is not None else bgr.copy()
            say = bres.guidance_text or ""

            state_info = bres.state_info or {}
            cross_stage = state_info.get("crosswalk_stage", "not_detected")
            blind_state = state_info.get("state", "UNKNOWN")
            # 可选字段（若工作流未来补充）
            # Optional fields (if workflow provides them in the future)
            angle = float(state_info.get("last_angle", 0.0))
            center_x_ratio = float(state_info.get("last_center_x_ratio", 0.5))

            # —— 盲道 → 发现斑马线（approaching/ready）
            # —— Blind-path → crosswalk detected (approaching/ready)
            if self.state == BLINDPATH_NAV:
                if cross_stage in ("approaching", "ready"):
                    self.cnt_crosswalk_seen += 1
                else:
                    self.cnt_crosswalk_seen = max(0, self.cnt_crosswalk_seen - 1)

                if self.cnt_crosswalk_seen >= self.FRAMES_CROSS_SEEN and not in_cooldown:
                    self.state = SEEKING_CROSSWALK
                    self.cooldown_until = now + self.COOLDOWN_SEC
                    say = "Approaching crosswalk, aligning direction for you."
                    # 正在接近斑马线，为您对准方向。

            # —— 对准阶段：同时利用 blind 内部 crosswalk_tracker 的角度与偏移（若提供）
            # —— Alignment phase: uses angle and offset from blind's internal crosswalk_tracker (if available)
            elif self.state == SEEKING_CROSSWALK:
                aligned = (abs(angle) <= self.ANGLE_ALIGN_THR_DEG and abs(center_x_ratio - 0.5) <= self.OFFSET_ALIGN_THR)
                if cross_stage == "ready" and aligned:
                    self.cnt_align_ready += 1
                else:
                    self.cnt_align_ready = max(0, self.cnt_align_ready - 1)

                if self.cnt_align_ready >= self.FRAMES_ALIGN_READY and not in_cooldown:
                    self.state = WAIT_TRAFFIC_LIGHT
                    self.cooldown_until = now + self.COOLDOWN_SEC
                    say = "Reached crosswalk, please wait for the traffic light."
                    # 已到达斑马线，请等待红绿灯。

            # —— 过马路后寻找下一段盲道（上盲道流程）
            # —— After crossing, find next tactile paving entry (boarding process)
            elif self.state == SEEKING_NEXT_BLINDPATH:
                if blind_state == "NAVIGATING":
                    self.cnt_cross_end += 1
                else:
                    self.cnt_cross_end = max(0, self.cnt_cross_end - 1)
                if self.cnt_cross_end >= self.FRAMES_NEXT_BLIND_OK and not in_cooldown:
                    self.state = BLINDPATH_NAV
                    self.cooldown_until = now + self.COOLDOWN_SEC
                    say = "Direction correct, please continue forward."
                    # 方向正确，请继续前进。

            # —— 恢复态：一旦盲道恢复可用则回盲道
            # —— Recovery state: return to blind-path navigation once it becomes available again
            elif self.state == RECOVERY:
                if blind_state in ("ONBOARDING", "NAVIGATING"):
                    self.state = BLINDPATH_NAV
                    self.cooldown_until = now + self.COOLDOWN_SEC
                    say = ""
                else:
                    say = ""

            # 丢失计数（兜底）
            # Loss counter (fallback)
            if blind_state == "UNKNOWN" and cross_stage == "not_detected":
                self.cnt_lost += 1
            else:
                self.cnt_lost = max(0, self.cnt_lost - 2)
            if self.cnt_lost >= self.FRAMES_LOST_MAX and self.state != RECOVERY:
                self.prev_target_state = self.state
                self.state = RECOVERY
                self.cooldown_until = now + self.COOLDOWN_SEC
                say = "Complex environment, entering recovery mode."
                # 环境复杂，进入恢复模式。

            return OrchestratorResult(ann, self._say(now, say), self.state, {"source": "blind", "cross_stage": cross_stage, "blind_state": blind_state})

        if self.state == WAIT_TRAFFIC_LIGHT:
            ann = bgr.copy()
            # 红绿灯识别（多数表决+冷却）
            # Traffic light detection (majority vote + cooldown)
            color, meta = self.tld.detect(bgr)
            self.tl_major.push(color)
            major = self.tl_major.majority()
            self.tl_last_color = major

            say = ""
            if major == "green" and not in_cooldown:
                self.state = CROSSING
                self.cooldown_until = now + self.COOLDOWN_SEC
                say = "Green light confirmed, start crossing."
                # 绿灯稳定，开始通行。
            else:
                # 只在刚进入状态或每隔一段时间才播报
                # Announce only when first entering state or after a time interval
                if not hasattr(self, '_last_wait_light_announce'):
                    self._last_wait_light_announce = 0
                if now - self._last_wait_light_announce > 5.0:  # 5秒播报一次 / announce every 5 seconds
                    say = "Waiting for green light..."
                    # 正在等待绿灯…
                    self._last_wait_light_announce = now

            return OrchestratorResult(ann, self._say(now, say), self.state, {"traffic_light": major})

        if self.state == CROSSING:
            try:
                cres: CrossResult = self.cross.process_frame(bgr)
            except Exception as e:
                # 异常 → 恢复 / Exception → recovery
                self.state = RECOVERY
                ann_err = bgr.copy()
                return OrchestratorResult(ann_err, self._say(now, ""), self.state, {"error": str(e)})

            ann = cres.annotated_image if cres.annotated_image is not None else bgr.copy()
            say = cres.guidance_text or ""

            # 新增：检查是否检测到盲道
            # New: check if tactile paving is detected
            blind_path_detected = getattr(cres, 'blind_path_detected', False)
            blind_path_guidance = getattr(cres, 'blind_path_guidance', "")
            
            # 如果检测到盲道且需要引导，优先处理盲道引导
            # If tactile paving is detected and guidance is needed, prioritize blind-path guidance
            if blind_path_detected and blind_path_guidance:
                # 如果应该切换到盲道导航（盲道很近），直接切换状态
                # If should switch to blind-path navigation (paving is very close), switch state directly
                if hasattr(cres, "should_switch_to_blindpath") and cres.should_switch_to_blindpath:
                    if not in_cooldown:
                        self.state = BLINDPATH_NAV
                        self.cooldown_until = now + self.COOLDOWN_SEC
                        say = "Reached tactile paving, switching to blind-path navigation."
                        # 已到盲道跟前，切换到盲道导航。
                        self.cnt_cross_end = 0  # 重置计数器 / reset counter
                        # 重置盲道导航器状态 / reset blind-path navigator state
                        if hasattr(self.blind, 'reset'):
                            self.blind.reset()
                else:
                    # 盲道较远，继续过马路但给出盲道引导
                    # Paving is far, continue crossing but provide blind-path guidance
                    # say is already included in cres.guidance_text
                    pass

            # 原有的结束条件：连续多帧"寻找斑马线"
            # Original end condition: consecutive frames of "seeking crosswalk"
            end_hint = False
            if "寻找斑马线" in (say or ""):
                end_hint = True

            self.cnt_cross_end = self.cnt_cross_end + 1 if end_hint else max(0, self.cnt_cross_end - 1)

            if self.cnt_cross_end >= self.FRAMES_CROSS_END and not in_cooldown:
                self.state = SEEKING_NEXT_BLINDPATH
                self.cooldown_until = now + self.COOLDOWN_SEC
                say = "Crossing complete, preparing to board the sidewalk."
                # 过马路结束，准备上人行道。

            return OrchestratorResult(ann, self._say(now, say), self.state, {"source": "cross", "end_cnt": self.cnt_cross_end})

        # 兜底 / Fallback
        ann = bgr.copy()
        return OrchestratorResult(ann, "", self.state, {})