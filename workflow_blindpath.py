# workflow_blindpath.py
# -*- coding: utf-8 -*-
"""
Tactile path navigation workflow - clean version
All Redis and Celery dependencies removed; can be integrated directly into any Python application.
"""
import os
import time
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import torch
from obstacle_detector_client import ObstacleDetectorClient
from audio_player import play_voice_text
from crosswalk_awareness import CrosswalkAwarenessMonitor, split_combined_voice  # crosswalk awareness
# Try to import Pillow for text rendering
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image, ImageDraw, ImageFont = None, None, None

logger = logging.getLogger(__name__)

# ========== State constants ==========
STATE_ONBOARDING = "ONBOARDING"
STATE_NAVIGATING = "NAVIGATING"
STATE_MANEUVERING_TURN = "MANEUVERING_TURN"
STATE_AVOIDING_OBSTACLE = "AVOIDING_OBSTACLE"
STATE_LOCKING_ON = "LOCKING_ON"

# ONBOARDING sub-steps
ONBOARDING_STEP_ROTATION = "ROTATION"
ONBOARDING_STEP_TRANSLATION = "TRANSLATION"

# Turn maneuver sub-steps
MANEUVER_STEP_1_ISSUE_COMMAND = "ISSUE_COMMAND"
MANEUVER_STEP_2_WAIT_FOR_SHIFT = "WAIT_FOR_SHIFT"
MANEUVER_STEP_3_ALIGN_ON_NEW_PATH = "ALIGN_ON_NEW_PATH"

# Color definitions (BGR format)
VIS_COLORS = {
    "blind_path": (0, 255, 0),      # green
    "obstacle": (0, 0, 255),        # red
    "crosswalk": (0, 165, 255),     # orange
    "centerline": (0, 255, 255),    # yellow
    "target_point": (255, 0, 0),    # blue
    "turn_point": (128, 0, 128),    # purple
    "pulse_effect": (100, 100, 255) # light red
}

# Obstacle name mapping (English labels)
_OBSTACLE_NAME_CN = {
    'person': 'person',
    'bicycle': 'bicycle',
    'car': 'car',
    'motorcycle': 'motorcycle',
    'bus': 'bus',
    'truck': 'truck',
    'animal': 'animal',
    'scooter': 'scooter',
    'stroller': 'stroller',
    'dog': 'dog',
}

# Dynamic class name list
DYNAMIC_CLASS_NAMES = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'animal', 'dog'}

@dataclass
class ProcessingResult:
    """Processing result data class"""
    guidance_text: str  # voice guidance text
    visualizations: List[Dict[str, Any]]  # visualization element list
    annotated_image: Optional[np.ndarray] = None  # annotated image
    state_info: Dict[str, Any] = None  # state info
    
    def __post_init__(self):
        if self.state_info is None:
            self.state_info = {}


class BlindPathNavigator:
    """Tactile path navigator - no external dependencies"""

    def __init__(self, yolo_model=None, obstacle_detector=None):
        """
        Initialize the navigator.
        :param yolo_model: YOLO segmentation model (optional)
        :param obstacle_detector: obstacle detector (optional)
        """
        self.yolo_model = yolo_model
        self.obstacle_detector = obstacle_detector
        
        # State variables
        self.current_state = STATE_ONBOARDING
        self.onboarding_step = ONBOARDING_STEP_ROTATION
        self.maneuver_step = MANEUVER_STEP_1_ISSUE_COMMAND
        self.maneuver_target_info = None
        

        # Optical flow tracking parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.05,
            minDistance=10,
            blockSize=7,
            useHarrisDetector=False,
            k=0.04
        )
        
        # Optical flow point cache
        self.flow_points = {}  # {mask_type: points}
        self.flow_grace = {}   # {mask_type: grace_count}
        self.FLOW_GRACE_MAX = 3  # Reduced from 8 to 3 frames for fast optical-flow cleanup

        # Centerline smoothing cache
        self.centerline_history = []  # historical centerline data
        self.centerline_history_max = 5  # keep the most recent 5 frames for smoothing

        # Polynomial coefficient smoothing cache
        self.poly_coeffs_history = []  # historical polynomial coefficients
        self.poly_coeffs_history_max = 8  # keep the most recent 8 frames of coefficients for smoothing

        # Turn detection tracker
        self.turn_detection_tracker = {
            'direction': None,
            'consecutive_hits': 0,
            'last_seen_frame': 0,
            'corner_info': None
        }
        
        # Turn cooldown
        self.turn_cooldown_frames = 0
        self.TURN_COOLDOWN_DURATION = 50
        
        # Obstacle avoidance state
        self.avoidance_plan = None
        self.avoidance_step_index = 0
        self.lock_on_data = None
        
        # Crosswalk tracker
        self.crosswalk_tracker = {
            'stage': 'not_detected',
            'consecutive_frames': 0,
            'last_area_ratio': 0.0,
            'last_bottom_y_ratio': 0.0,
            'last_center_x_ratio': 0.5,
            'position_announced': False,
            'alignment_status': 'not_aligned',
            'last_seen_frame': 0,
            'last_angle': 0.0
        }
        
        # Frame counter
        self.frame_counter = 0
        
        # Straight-ahead prompt configuration - supports env vars
        self.guide_interval = float(os.getenv("AIGLASS_STRAIGHT_INTERVAL", "4.0"))  # announcement interval (seconds)
        self.last_guide_time = 0.0
        self.straight_continuous_mode = os.getenv("AIGLASS_STRAIGHT_CONTINUOUS", "1") == "1"  # continuous mode
        self.straight_repeat_limit = int(os.getenv("AIGLASS_STRAIGHT_LIMIT", "2"))  # max repeats in limit mode
        self.straight_repeat_count = 0

        # Direction command repeat configuration
        self.direction_interval = float(os.getenv("AIGLASS_DIRECTION_INTERVAL", "3.0"))  # direction command interval (seconds)
        self.last_direction_time = 0.0
        self.last_direction_message = ""

        logger.info(f"[BlindPath] Straight-ahead config: interval={self.guide_interval}s, "
                   f"continuous={self.straight_continuous_mode}, "
                   f"repeat_limit={self.straight_repeat_limit}")
        logger.info(f"[BlindPath] Direction config: interval={self.direction_interval}s")

        # Cache variables
        self.prev_gray = None
        self.prev_blind_path_mask = None
        self.prev_crosswalk_mask = None
        self.prev_obstacle_cache = []
        self.last_guidance_message = ""
        self.last_detected_obstacles = []
        self.last_obstacle_detection_frame = 0
        self.last_any_speech_time = 0
        
        # Crosswalk ready state flags
        self.crosswalk_ready_announced = False
        self.crosswalk_ready_time = 0
        
        # Pending obstacle voice announcement
        self.pending_obstacle_voice = None
        
        # Traffic light detection
        self.traffic_light_detector = None
        self.init_traffic_light_detector()
        self.traffic_light_history = deque(maxlen=8)  # for majority voting
        self.last_traffic_light_state = "unknown"
        self.green_light_announced = False
        
        # Threshold settings
        self.CLASS_CONF_THRESHOLDS = {
            1: 0.20,  # blind_path
            0: 0.30   # crosswalk
        }
        
        # Navigation thresholds
        self.ONBOARDING_ALIGN_THRESHOLD_RATIO = 0.1
        self.VP_FIT_ERROR_THRESHOLD = 8.0

        self.ONBOARDING_ORIENTATION_THRESHOLD_RAD = np.deg2rad(10)
        self.ONBOARDING_CENTER_OFFSET_THRESHOLD_RATIO = 0.15
        self.NAV_ORIENTATION_THRESHOLD_RAD = np.deg2rad(10)
        self.NAV_CENTER_OFFSET_THRESHOLD_RATIO = 0.15
        self.CURVATURE_PROXY_THRESHOLD = 5e-5
        
        # Crosswalk switch thresholds
        self.CROSSWALK_SWITCH_AREA_RATIO = 0.22
        self.CROSSWALK_SWITCH_BOTTOM_RATIO = 0.9
        self.CROSSWALK_SWITCH_CONSECUTIVE_FRAMES = 10
        
        # Obstacle detection interval - read from env vars for performance tuning
        self.OBSTACLE_DETECTION_INTERVAL = int(os.getenv("AIGLASS_OBS_INTERVAL", "15"))  # detect every N frames
        self.OBSTACLE_CACHE_DURATION_FRAMES = int(os.getenv("AIGLASS_OBS_CACHE_FRAMES", "10"))  # cache 10 frames

        # Obstacle announcement management
        self.last_obstacle_speech = ""
        self.last_obstacle_speech_time = 0
        self.obstacle_speech_cooldown = 5.0  # same obstacle: no repeat within 5 seconds

        # Mask stabilization parameters (optical-flow extrapolation disabled; these are unused)
        self.MASK_STAB_MIN_AREA = int(os.getenv("AIGLASS_MASK_MIN_AREA", "1500"))
        self.MASK_STAB_KERNEL = int(os.getenv("AIGLASS_MASK_MORPH", "3"))
        self.MASK_MISS_TTL = 0  # set to 0: optical-flow extrapolation disabled, fully real-time
        self.blind_miss_ttl = 0
        self.cross_miss_ttl = 0

        # Optical flow tracking parameters
        self.flow_iou_threshold = 0.3  # reinitialize flow points when IoU drops below this

        # Tactile path YOLO detection interval
        self.BLINDPATH_DETECTION_INTERVAL = int(os.getenv("AIGLASS_BLINDPATH_INTERVAL", "8"))  # detect every N frames
        self.last_blindpath_detection_frame = 0
        self.last_blindpath_mask = None
        self.last_crosswalk_mask = None

        # Crosswalk awareness monitor
        self.crosswalk_monitor = CrosswalkAwarenessMonitor()
        logger.info("[BlindPath] Crosswalk awareness monitor initialized")
        logger.info(f"[BlindPath] Tactile path detection interval: every {self.BLINDPATH_DETECTION_INTERVAL} frames")
    
    def init_traffic_light_detector(self):
        """Initialize the traffic light detector."""
        try:
            self.traffic_light_yolo = None
            # Load a dedicated traffic-light model here if available:
            # self.traffic_light_yolo = YOLO('path/to/traffic_light_model.pt')
        except Exception as e:
            logger.info(f"Traffic light YOLO model not loaded: {e}")
    
    def detect_traffic_light(self, image: np.ndarray) -> str:
        """Detect traffic light state.
        Returns: 'red', 'green', 'yellow', or 'unknown'
        """
        # Simulation mode (for testing)
        if os.getenv("AIGLASS_SIMULATE_TRAFFIC_LIGHT", "0") == "1":
            # Simulate traffic light changes based on frame count
            cycle = (self.frame_counter // 100) % 3
            if cycle == 0:
                return "red"
            elif cycle == 1:
                return "yellow"
            else:
                return "green"
        
        # Use YOLO model if available
        if self.traffic_light_yolo:
            try:
                results = self.traffic_light_yolo.predict(image, verbose=False, conf=0.3)
                # TODO: parse YOLO results to determine traffic light color
                pass
            except:
                pass

        # Fall back to HSV color detection
        return self._detect_traffic_light_by_color(image)
    
    def _detect_traffic_light_by_color(self, image: np.ndarray) -> str:
        """Detect traffic light color using HSV color space."""
        h, w = image.shape[:2]
        # Scan the upper 70% of the image (traffic lights may appear at various heights)
        roi = image[:int(h * 0.7), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Brightened image aids detection of dim traffic lights
        hsv_bright = hsv.copy()
        hsv_bright[:, :, 2] = cv2.add(hsv_bright[:, :, 2], 30)  # boost brightness

        # Color ranges (optimized parameters)
        # Red uses two ranges because it straddles 0° in HSV
        lower_red1 = np.array([0, 120, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Green (wider range to accommodate different lighting conditions)
        lower_green = np.array([40, 60, 60])
        upper_green = np.array([90, 255, 255])
        
        # Yellow
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        
        # Create masks on both original and brightened images
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red1_bright = cv2.inRange(hsv_bright, lower_red1, upper_red1)
        mask_red2_bright = cv2.inRange(hsv_bright, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(cv2.bitwise_or(mask_red1, mask_red2), 
                                 cv2.bitwise_or(mask_red1_bright, mask_red2_bright))
        
        mask_green = cv2.bitwise_or(cv2.inRange(hsv, lower_green, upper_green),
                                   cv2.inRange(hsv_bright, lower_green, upper_green))
        mask_yellow = cv2.bitwise_or(cv2.inRange(hsv, lower_yellow, upper_yellow),
                                    cv2.inRange(hsv_bright, lower_yellow, upper_yellow))
        
        # Morphological denoising
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        
        # Compute area for each color
        area_red = cv2.countNonZero(mask_red)
        area_green = cv2.countNonZero(mask_green)
        area_yellow = cv2.countNonZero(mask_yellow)
        
        # Minimum area threshold (lower = more sensitive)
        min_area = 30

        if hasattr(self, 'frame_counter') and self.frame_counter % 30 == 0:
            logger.info(f"[HSV detection] red:{area_red}, green:{area_green}, yellow:{area_yellow}")
            # Save debug images
            if os.getenv("AIGLASS_DEBUG_TRAFFIC_LIGHT", "0") == "1":
                debug_dir = "traffic_light_debug"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(f"{debug_dir}/frame_{self.frame_counter}_roi.jpg", roi)
                cv2.imwrite(f"{debug_dir}/frame_{self.frame_counter}_red.jpg", mask_red)
                cv2.imwrite(f"{debug_dir}/frame_{self.frame_counter}_green.jpg", mask_green)
                cv2.imwrite(f"{debug_dir}/frame_{self.frame_counter}_yellow.jpg", mask_yellow)
        
        # Determine color (priority: green > red > yellow)
        if area_green > min_area and area_green > area_red * 0.8:  # green has priority
            return "green"
        elif area_red > min_area and area_red > area_green:
            return "red"
        elif area_yellow > min_area:
            return "yellow"
        else:
            return "unknown"
    
    def _get_voice_priority(self, guidance_text):
        """Get the priority of a voice command.
        Priority: obstacle(100) > turn/shift(50) > go straight(10)
        """
        if not guidance_text:
            return 0

        # Obstacle announcement - highest priority
        obstacle_keywords = ['watch out', 'ahead, stop', 'Obstacle']
        for keyword in obstacle_keywords:
            if keyword in guidance_text:
                return 100
        
        # Turn and shift - medium priority
        direction_keywords = ['Turn', 'turn', 'Shift', 'shift', 'shifting', 'Fine-tune', 'move slightly']
        for keyword in direction_keywords:
            if keyword in guidance_text:
                return 50
        
        # Go straight - lowest priority
        if 'Go straight' in guidance_text:
            return 10

        # Other commands - default medium priority
        return 30

    def process_frame(self, image: np.ndarray) -> ProcessingResult:
        """
        Process a single frame.
        :param image: BGR image
        :return: ProcessingResult
        """
        self.frame_counter += 1
        
        # Update turn cooldown
        if self.turn_cooldown_frames > 0:
            self.turn_cooldown_frames -= 1
        
        image_height, image_width = image.shape[:2]
        image_center_x = image_width / 2
        
        # Convert to grayscale
        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Visualization element list
        frame_visualizations = []
        guidance_text = ""
        
        # 1. Run YOLO detection every frame (no caching)
        blind_path_mask, crosswalk_mask = self._detect_path_and_crosswalk(image)
        
        # Debug: log YOLO detection results every 30 frames
        if self.frame_counter % 30 == 0:
            has_blind = blind_path_mask is not None and np.sum(blind_path_mask > 0) > 0
            has_cross = crosswalk_mask is not None and np.sum(crosswalk_mask > 0) > 0
            logger.info(f"[YOLO] Frame={self.frame_counter}, blind_path={'yes' if has_blind else 'no'}, "
                       f"crosswalk={'yes' if has_cross else 'no'}")
            if has_cross:
                cross_area = np.sum(crosswalk_mask > 0) / crosswalk_mask.size
                logger.info(f"[YOLO] crosswalk raw area: {cross_area*100:.2f}%")

        # 2. Mask stabilization and optical-flow extrapolation are disabled.
        # Use real-time detection results directly.
        crosswalk_mask_before_stabilize = crosswalk_mask
        
        if self.frame_counter % 30 == 0 and crosswalk_mask_before_stabilize is not None:
            after_stab = crosswalk_mask is not None and np.sum(crosswalk_mask > 0) > 0
            logger.info(f"[Mask stabilize] crosswalk after stabilize: {'yes' if after_stab else 'no (filtered)'}")
        
        # 3. Obstacle detection on every frame regardless of state
        logger.info(f"[Frame {self.frame_counter}] Starting obstacle detection...")

        # Cache strategy: re-detect at interval, reuse cache otherwise
        if self.frame_counter % self.OBSTACLE_DETECTION_INTERVAL == 0:
            detected_obstacles = self._detect_obstacles(image, blind_path_mask)
            self.last_detected_obstacles = detected_obstacles
            self.last_obstacle_detection_frame = self.frame_counter
            logger.info(f"[Frame {self.frame_counter}] New detection: {len(detected_obstacles)} obstacles found")
        else:
            if self.frame_counter - self.last_obstacle_detection_frame < self.OBSTACLE_CACHE_DURATION_FRAMES:
                detected_obstacles = self.last_detected_obstacles
                logger.info(f"[Frame {self.frame_counter}] Using cached obstacle data: {len(detected_obstacles)} obstacles")
            else:
                detected_obstacles = []
                logger.info(f"[Frame {self.frame_counter}] Cache expired, no obstacle data")
        
        # Visualize all detected obstacles (not only near ones)
        for i, obs in enumerate(detected_obstacles):
            logger.info(f"  Obstacle {i+1}: {obs.get('name', 'unknown')}, "
                    f"bottom_y_ratio={obs.get('bottom_y_ratio', 0):.2f}, "
                    f"area_ratio={obs.get('area_ratio', 0):.3f}, "
                    f"pos=({obs.get('center_x', 0):.0f}, {obs.get('center_y', 0):.0f})")
            self._add_obstacle_visualization(obs, frame_visualizations)
        
        # Check for near obstacles and queue voice announcement
        self._check_and_set_obstacle_voice(detected_obstacles)
        
        # Crosswalk awareness processing
        if crosswalk_mask is not None:
            cross_pixels = np.sum(crosswalk_mask > 0)
            if cross_pixels > 0:
                logger.info(f"[Crosswalk] passing to monitor: pixels={cross_pixels}, area={cross_pixels/crosswalk_mask.size*100:.2f}%")
            else:
                logger.info(f"[Crosswalk] crosswalk_mask all zeros, no crosswalk")
        else:
            if self.frame_counter % 30 == 0:
                logger.info(f"[Crosswalk] crosswalk_mask is None")
        
        crosswalk_guidance = self.crosswalk_monitor.process_frame(crosswalk_mask, blind_path_mask)
        if crosswalk_guidance:
            logger.info(f"[Crosswalk awareness] result: area={crosswalk_guidance.get('area', 0):.3f}, "
                       f"should_broadcast={crosswalk_guidance.get('should_broadcast', False)}, "
                       f"voice={crosswalk_guidance.get('voice_text', 'None')}")
        if crosswalk_guidance and crosswalk_guidance['should_broadcast']:
            # Queue crosswalk voice via pending mechanism
            if not hasattr(self, 'pending_crosswalk_voice'):
                self.pending_crosswalk_voice = None
            self.pending_crosswalk_voice = crosswalk_guidance
            logger.info(f"[Crosswalk voice] queued: {crosswalk_guidance['voice_text']}, priority={crosswalk_guidance['priority']}")
        
        # Add crosswalk visualization
        if crosswalk_mask is not None:
            # Compute visualization data
            total_pixels = crosswalk_mask.size
            crosswalk_pixels = np.sum(crosswalk_mask > 0)
            area_ratio = crosswalk_pixels / total_pixels
            
            y_coords, x_coords = np.where(crosswalk_mask > 0)
            if len(y_coords) > 0:
                center_x_ratio = np.mean(x_coords) / crosswalk_mask.shape[1]
                center_y_ratio = np.mean(y_coords) / crosswalk_mask.shape[0]
                has_occlusion = self.crosswalk_monitor._check_occlusion(crosswalk_mask, blind_path_mask)
                
                # Get visualization data
                viz_data = self.crosswalk_monitor.get_visualization_data(
                    crosswalk_mask, area_ratio, center_x_ratio, center_y_ratio, has_occlusion
                )
                
                # Add crosswalk mask visualization
                self._add_mask_visualization(crosswalk_mask, frame_visualizations, 
                                            "crosswalk_mask", viz_data['stage_color'])
                
                # Add crosswalk detection info visualization
                self._add_crosswalk_info_visualization(viz_data, image_height, image_width, 
                                                      frame_visualizations)
        
        # 4. Crosswalk tracker update is disabled - blind-path mode no longer transitions to crosswalk mode
        # self._update_crosswalk_tracker(crosswalk_mask, image_height, image_width)

        # 5. Path visualization
        self._add_mask_visualization(blind_path_mask, frame_visualizations, "blind_path_mask", "rgba(0, 255, 0, 0.4)")
        # Crosswalk visualization is handled by crosswalk_monitor; not added here

        # 5b. Always run blind-path navigation; crosswalk state logic is disabled
        current_stage = 'not_detected'  # fixed: crosswalk detection not handled here
        # current_stage = self.crosswalk_tracker['stage']  # disabled

        # Run blind-path navigation directly, ignoring crosswalk state
        if False:  # current_stage == 'ready':
            # Check if the ready prompt has already been announced
            if not hasattr(self, 'crosswalk_ready_announced'):
                self.crosswalk_ready_announced = False
                self.crosswalk_ready_time = 0
            
            current_time = time.time()
            
            # Detect traffic light
            traffic_light_color = self.detect_traffic_light(image)
            self.traffic_light_history.append(traffic_light_color)

            if self.frame_counter % 30 == 0:
                logger.info(f"[Traffic light] current: {traffic_light_color}, history: {list(self.traffic_light_history)}")

            # Majority vote for a stable traffic light state
            if len(self.traffic_light_history) >= 3:
                color_counts = {}
                for color in self.traffic_light_history:
                    color_counts[color] = color_counts.get(color, 0) + 1
                # Pick the most frequent color
                stable_color = max(color_counts.items(), key=lambda x: x[1])[0]
            else:
                stable_color = "unknown"
            
            # Add traffic light visualization
            self._add_traffic_light_visualization(
                stable_color, frame_visualizations, image_height, image_width
            )
            
            # Decide voice announcement
            if not self.crosswalk_ready_announced:
                guidance_text = "Aligned, ready to switch to crossing mode."
                self.crosswalk_ready_announced = True
                self.crosswalk_ready_time = current_time
            elif stable_color == "green" and not self.green_light_announced:
                guidance_text = "Green light stable, start crossing."
                self.green_light_announced = True
            elif stable_color == "red":
                # Periodic reminder while red light is on
                if current_time - self.crosswalk_ready_time > 5.0:
                    guidance_text = "Waiting for green light..."
                    self.crosswalk_ready_time = current_time
                else:
                    guidance_text = ""
            else:
                guidance_text = ""
            
            frame_visualizations.append({
                "type": "data_panel",
                "data": {
                    "Status": "Waiting to cross",
                    "Traffic Light": stable_color,
                    "History": len(self.traffic_light_history)
                },
                "position": (25, image_height - 120)
            })
            
        elif False:  # current_stage == 'approaching':
            guidance_text = self._handle_crosswalk_approaching(
                frame_visualizations, image_height, image_width, image
            )
            
        # elif current_stage in ['far', 'not_detected']:
        else:  # always run blind-path navigation
            # Crosswalk far-detection prompt is disabled:
            # if current_stage == 'far' and not self.crosswalk_tracker['position_announced']:
            #     guidance_text = "Crosswalk detected ahead, continue straight."
            #     self.crosswalk_tracker['position_announced'] = True

            if blind_path_mask is None:
                guidance_text = ""
                frame_visualizations.append({
                    "type": "data_panel",
                    "data": {
                        "Status": "Searching for tactile path"
                    },
                    "position": (image_width - 180, 20)
                })
            else:
                guidance_text = self._execute_state_machine(
                    blind_path_mask, image, frame_visualizations,
                    image_height, image_width, curr_gray
                )
        
        # 6. Update cache
        self.prev_gray = curr_gray
        if blind_path_mask is not None:
            self.prev_blind_path_mask = blind_path_mask.copy()
        if crosswalk_mask is not None:
            self.prev_crosswalk_mask = crosswalk_mask.copy()
        
        # Voice priority management system
        current_time = time.time()

        # Collect all candidate voice commands
        voice_candidates = []

        # 1. Add main navigation voice
        if guidance_text:
            voice_candidates.append({
                'text': guidance_text,
                'priority': self._get_voice_priority(guidance_text),
                'source': 'navigation'
            })
        
        # 2. Check for pending obstacle voice (always highest priority)
        if hasattr(self, 'pending_obstacle_voice'):
            if self.pending_obstacle_voice:
                voice_candidates.append({
                    'text': self.pending_obstacle_voice,
                    'priority': 100,  # obstacle always highest priority
                    'source': 'obstacle'
                })
                self.pending_obstacle_voice = None

        # Check for pending crosswalk voice
        if hasattr(self, 'pending_crosswalk_voice'):
            if self.pending_crosswalk_voice:
                voice_candidates.append({
                    'text': self.pending_crosswalk_voice['voice_text'],
                    'priority': self.pending_crosswalk_voice['priority'],
                    'source': 'crosswalk'
                })
                self.pending_crosswalk_voice = None

        # 3. Select the highest-priority voice
        if voice_candidates:
            # Sort by priority descending, pick the top
            voice_candidates.sort(key=lambda x: x['priority'], reverse=True)
            selected_voice = voice_candidates[0]
            final_guidance_text = selected_voice['text']
            
            # Global speech cooldown: enforce minimum gap between any two announcements
            MIN_SPEECH_INTERVAL = 1.2  # at least 1.2 s between any two speech outputs
            if hasattr(self, 'last_any_speech_time'):
                if current_time - self.last_any_speech_time < MIN_SPEECH_INTERVAL:
                    final_guidance_text = ""  # too soon, skip this announcement

            # Throttle "Go straight" announcements
            if final_guidance_text == "Go straight":
                if self.straight_continuous_mode:
                    # Continuous mode: check time interval only
                    if current_time - self.last_guide_time >= self.guide_interval:
                        self.last_guide_time = current_time
                        self.straight_repeat_count += 1
                        self.last_any_speech_time = current_time
                    else:
                        final_guidance_text = ""
                else:
                    # Limit mode: check time interval and repeat count
                    if (current_time - self.last_guide_time >= self.guide_interval) and \
                       (self.straight_repeat_count < self.straight_repeat_limit):
                        self.last_guide_time = current_time
                        self.straight_repeat_count += 1
                        self.last_any_speech_time = current_time
                    else:
                        final_guidance_text = ""
            elif final_guidance_text and selected_voice['source'] != 'obstacle':
                # Non-straight, non-obstacle command - direction commands support continuous repeat
                # Check if it's a direction command
                direction_keywords = ["Turn", "turn", "Shift", "shift", "shifting", "Fine-tune", "move slightly"]
                is_direction = any(keyword in final_guidance_text for keyword in direction_keywords)
                
                if is_direction:
                    # Direction command: supports continuous repeat
                    if final_guidance_text == self.last_direction_message:
                        # Same direction command: check time interval
                        if current_time - self.last_direction_time >= self.direction_interval:
                            self.last_direction_time = current_time
                            self.last_any_speech_time = current_time
                            self.straight_repeat_count = 0
                        else:
                            final_guidance_text = ""  # interval not met, skip
                    else:
                        # New direction command: announce immediately
                        self.last_direction_message = final_guidance_text
                        self.last_direction_time = current_time
                        self.last_any_speech_time = current_time
                        self.straight_repeat_count = 0
                else:
                    # Other commands: announce only once
                    if final_guidance_text != self.last_guidance_message:
                        self.last_guidance_message = final_guidance_text
                        self.straight_repeat_count = 0
                        self.last_any_speech_time = current_time
                    else:
                        final_guidance_text = ""
            elif final_guidance_text and selected_voice['source'] == 'obstacle':
                # Obstacle voice always plays
                self.last_any_speech_time = current_time
            elif final_guidance_text and selected_voice['source'] == 'crosswalk':
                # Crosswalk voice always plays (not subject to duplicate check)
                self.last_any_speech_time = current_time

            # Play the selected voice
            if final_guidance_text:
                try:
                    # Lazy import: app_main imports this module at startup, before
                    # AI_BACKEND is defined, so importing at module scope would be
                    # circular. By call time app_main has finished loading.
                    from app_main import AI_BACKEND
                    # For combined crosswalk voice: play only the first part to stay real-time
                    if selected_voice.get('source') == 'crosswalk' and ',' in final_guidance_text:
                        voice_parts = split_combined_voice(final_guidance_text)
                        logger.info(f"[Crosswalk voice] combined: {len(voice_parts)} parts, playing only first part")
                        if voice_parts and AI_BACKEND != "gemini_live":
                            play_voice_text(voice_parts[0])
                            logger.info(f"[Voice] priority={selected_voice['priority']}: {voice_parts[0]}")
                    elif AI_BACKEND != "gemini_live":
                        play_voice_text(final_guidance_text)
                        logger.info(f"[Voice] priority={selected_voice['priority']}: {final_guidance_text}")
                except Exception as e:
                    logger.error(f"[Voice] playback failed: {e}")
        else:
            final_guidance_text = ""
        
        # 7. Generate annotated image
        annotated_image = None

        if frame_visualizations:
            annotated_image = self._draw_visualizations(image.copy(), frame_visualizations)
        else:
            annotated_image = image.copy()
        
        # Add bottom command button (shows the currently announced voice text)
        current_instruction = final_guidance_text if final_guidance_text else "Waiting..."
        annotated_image = self._draw_command_button(annotated_image, current_instruction)
        
        # 8. Return result
        return ProcessingResult(
            guidance_text=guidance_text,
            visualizations=frame_visualizations,
            annotated_image=annotated_image,
            state_info={
                "state": self.current_state,
                "crosswalk_stage": current_stage,
                "frame_count": self.frame_counter
            }
        )
    
    def _detect_path_and_crosswalk(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect tactile path and crosswalk masks."""
        if self.yolo_model is None:
            # No model loaded: return simulated data for testing
            logger.warning("YOLO model not loaded, returning simulated data")
            h, w = image.shape[:2]
            # Simulate a tactile path: a vertical strip centered in the image (20% width)
            blind_path_mask = np.zeros((h, w), dtype=np.uint8)
            strip_width = int(w * 0.2)
            strip_left = (w - strip_width) // 2
            blind_path_mask[int(h*0.3):, strip_left:strip_left+strip_width] = 255
            return blind_path_mask, None
        
        blind_path_mask = None
        crosswalk_mask = None
        
        try:
            min_conf = min(self.CLASS_CONF_THRESHOLDS.values())
            results = self.yolo_model.predict(image, verbose=False, conf=min_conf, classes=[0, 1])
            
            if (results and results[0] and results[0].masks is not None and 
                results[0].boxes is not None and len(results[0].masks.data) > 0):
                
                for mask_tensor, conf_tensor, cls_tensor in zip(
                    results[0].masks.data, results[0].boxes.conf, results[0].boxes.cls
                ):
                    class_id = int(cls_tensor.item())
                    confidence = float(conf_tensor.item())
                    threshold = self.CLASS_CONF_THRESHOLDS.get(class_id, 1.0)
                    
                    if confidence >= threshold:
                        current_mask = self._tensor_to_mask(mask_tensor, image.shape[1], image.shape[0])
                        
                        if class_id == 1:  # tactile path
                            if blind_path_mask is None:
                                blind_path_mask = current_mask
                            else:
                                blind_path_mask = cv2.bitwise_or(blind_path_mask, current_mask)
                        elif class_id == 0:  # crosswalk
                            if crosswalk_mask is None:
                                crosswalk_mask = current_mask
                            else:
                                crosswalk_mask = cv2.bitwise_or(crosswalk_mask, current_mask)
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            # Return simulated data on detection failure
            h, w = image.shape[:2]
            blind_path_mask = np.zeros((h, w), dtype=np.uint8)
            strip_width = int(w * 0.2)
            strip_left = (w - strip_width) // 2
            blind_path_mask[int(h*0.3):, strip_left:strip_left+strip_width] = 255
        
        return blind_path_mask, crosswalk_mask
    
    def _tensor_to_mask(self, mask_tensor, out_w: int, out_h: int, binarize: bool = True) -> np.ndarray:
        """Convert a tensor mask to a numpy array."""
        try:
            import torch
            
            if not isinstance(mask_tensor, torch.Tensor):
                arr = np.asarray(mask_tensor)
                if arr.dtype != np.uint8:
                    arr = (arr > 0.5).astype(np.uint8) * 255 if binarize else (arr * 255.0).astype(np.uint8)
                mask_u8 = arr
            else:
                if mask_tensor.dtype in (torch.bfloat16, torch.float16):
                    mask_tensor = mask_tensor.to(torch.float32)
                
                if mask_tensor.ndim > 2:
                    mask_tensor = mask_tensor.squeeze()
                
                if binarize:
                    mask_tensor = (mask_tensor > 0.5).to(torch.uint8).mul_(255)
                    mask_u8 = mask_tensor.cpu().numpy()
                else:
                    mask_u8 = (mask_tensor.mul(255).clamp_(0, 255).to(torch.uint8)).cpu().numpy()
            
            if mask_u8.ndim == 3:
                mask_u8 = mask_u8.squeeze(-1)
            
            if mask_u8.shape[1] != out_w or mask_u8.shape[0] != out_h:
                mask_u8 = cv2.resize(mask_u8, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            
            return mask_u8
        except ImportError:
            # Return empty mask when torch is not available
            return np.zeros((out_h, out_w), dtype=np.uint8)
    
    def _stabilize_mask(self, prev_gray, curr_gray, raw_mask, prev_stable_mask, mask_type):
        """Stabilize a mask using Lucas-Kanade optical flow."""
        if mask_type == 'blind_path':
            ttl = self.blind_miss_ttl
            min_area = self.MASK_STAB_MIN_AREA
        else:  # crosswalk
            ttl = self.cross_miss_ttl
            min_area = self.MASK_STAB_MIN_AREA
        
        # Delegate to the LK-flow stabilization implementation
        stable_mask = self._stabilize_seg_mask(
            prev_gray, curr_gray, raw_mask, prev_stable_mask,
            (curr_gray.shape[1], curr_gray.shape[0]) if curr_gray is not None else (640, 480),
            min_area_px=min_area,
            morph_kernel=self.MASK_STAB_KERNEL,
            mask_type=mask_type
        )
        
        if stable_mask is not None:
            # Reset TTL
            if mask_type == 'blind_path':
                self.blind_miss_ttl = self.MASK_MISS_TTL
            else:
                self.cross_miss_ttl = self.MASK_MISS_TTL
            return stable_mask
        else:
            # Decrement TTL
            if mask_type == 'blind_path':
                self.blind_miss_ttl = max(0, self.blind_miss_ttl - 1)
            else:
                self.cross_miss_ttl = max(0, self.cross_miss_ttl - 1)
            return None
    
    def _stabilize_seg_mask(self, prev_gray, curr_gray, curr_mask, prev_stable_mask,
                          image_wh, min_area_px=1500, morph_kernel=3, iou_high_thr=0.4, mask_type='',
                          fast_clear=True):
        """Mask stabilization implementation using Lucas-Kanade optical flow."""
        W, H = image_wh
        
        def _binarize(mask):
            if mask is None:
                return None
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask
        
        def _morph_smooth(mask, kernel_size):
            if mask is None:
                return None
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (max(1, kernel_size), max(1, kernel_size)))
            sm = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
            sm = cv2.morphologyEx(sm, cv2.MORPH_OPEN, k, iterations=1)
            return sm
        
        curr_mask_b = _binarize(curr_mask)
        prev_mask_b = _binarize(prev_stable_mask)
        
        # No previous data: return current mask directly
        if prev_mask_b is None or prev_gray is None or curr_gray is None:
            return _morph_smooth(curr_mask_b, morph_kernel) if curr_mask_b is not None else None
        
        # Current frame has a detection result
        if curr_mask_b is not None and np.sum(curr_mask_b > 0) >= min_area_px:
            # Compute IoU with previous frame
            if prev_mask_b is not None:
                inter = np.logical_and(curr_mask_b > 0, prev_mask_b > 0).sum()
                union = np.logical_or(curr_mask_b > 0, prev_mask_b > 0).sum()
                iou = float(inter) / float(union) if union > 0 else 0.0
                
                # IoU is high enough: detection is stable, use current result
                if iou >= iou_high_thr:
                    return _morph_smooth(curr_mask_b, morph_kernel)

                # IoU is low but overlapping: fuse via weighted average
                elif iou > 0.1:
                    # Predict mask position via optical flow
                    flow_mask = self._predict_mask_with_flow(prev_mask_b, prev_gray, curr_gray)
                    if flow_mask is not None:
                        # Dynamically adjust weights based on IoU:
                        # lower IoU → rely more on flow; higher IoU → rely more on current detection
                        w_curr = min(0.9, 0.4 + iou)  # IoU=0.1→w_curr=0.5, IoU=0.5→w_curr=0.9
                        w_flow = 1.0 - w_curr
                        
                        fused = (w_curr * curr_mask_b.astype(np.float32) + 
                                w_flow * flow_mask.astype(np.float32))
                        fused_bin = (fused >= 128).astype(np.uint8) * 255
                        
                        # Reinitialize flow points when IoU is too low
                        if iou < self.flow_iou_threshold:
                            self.flow_points['blind_path'] = None
                        
                        return _morph_smooth(fused_bin, morph_kernel)
            
            # No history or IoU too low: use current detection
            return _morph_smooth(curr_mask_b, morph_kernel)

        # Current frame has no detection: try optical-flow extrapolation
        else:
            # Get TTL for this mask type
            if mask_type == 'blind_path':
                ttl = self.blind_miss_ttl
            else:
                ttl = self.cross_miss_ttl
            
            # No detection in current frame: clear quickly when fast_clear is set
            if fast_clear and ttl <= 1:
                # TTL exhausted: return None immediately, skip optical flow
                return None

            if prev_mask_b is not None and np.sum(prev_mask_b > 0) >= min_area_px and ttl > 0:
                # Use optical flow to predict
                flow_mask = self._predict_mask_with_flow(prev_mask_b, prev_gray, curr_gray)
                if flow_mask is not None and np.sum(flow_mask > 0) >= min_area_px * 0.5:
                    return _morph_smooth(flow_mask, morph_kernel)
            
            # Optical flow failed or TTL exceeded
            return None
    
    def _predict_mask_with_flow(self, prev_mask, prev_gray, curr_gray):
        """Predict mask position using Lucas-Kanade optical flow (improved version)."""
        try:
            # Method 1: convex-hull approach
            if hasattr(self, 'flow_points') and 'blind_path' in self.flow_points:
                p0 = self.flow_points['blind_path']
                if p0 is not None and len(p0) >= 5:
                    # Compute optical flow
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
                    
                    if p1 is not None and st is not None:
                        good_new = p1[st == 1]
                        if len(good_new) >= 5:
                            # Update flow points
                            self.flow_points['blind_path'] = good_new.reshape(-1, 1, 2)

                            # Generate convex-hull mask
                            hull = cv2.convexHull(good_new.reshape(-1, 1, 2))
                            poly = hull.reshape(-1, 2)
                            
                            if len(poly) >= 3:
                                H, W = curr_gray.shape[:2]
                                flow_mask = np.zeros((H, W), dtype=np.uint8)
                                cv2.fillPoly(flow_mask, [poly.astype(np.int32)], 255)
                                return flow_mask
            
            # Method 2: edge feature-point approach (fallback)
            edge_mask = self._get_edge_mask(prev_mask, offset=10)

            # Detect feature points
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=edge_mask, **self.feature_params)
            if p0 is None or len(p0) < 8:
                return None
            
            # Save feature points for next frame
            self.flow_points['blind_path'] = p0

            # Compute optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
            
            if p1 is None or st is None:
                return None
            
            # Keep only successfully tracked points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) < 5:
                return None

            # Estimate transformation matrix (RANSAC for robustness)
            M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            
            if M is None:
                return None
            
            # Apply transformation
            H, W = curr_gray.shape[:2]
            flow_mask = cv2.warpAffine(prev_mask, M, (W, H), 
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
            
            return flow_mask
            
        except Exception as e:
            logger.debug(f"Optical flow prediction failed: {e}")
            return None
            
    
    def _get_edge_mask(self, mask, offset=10):
        """Get the inner-edge region of a mask for feature point detection."""
        if mask is None:
            return None

        # Erode to get the interior mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (offset*2, offset*2))
        inner = cv2.erode(mask, kernel, iterations=1)

        # Edge = original - interior
        edge = cv2.subtract(mask, inner)

        # Slightly dilate the edge region
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge = cv2.dilate(edge, kernel_small, iterations=1)
        
        return edge

    def _smooth_centerline(self, centerline_data):
        """Smooth centerline data to reduce jitter."""
        if centerline_data is None or len(centerline_data) < 5:
            return centerline_data

        # Save to history
        self.centerline_history.append(centerline_data.copy())
        if len(self.centerline_history) > self.centerline_history_max:
            self.centerline_history.pop(0)
        
        # Insufficient history: return lightly smoothed current-frame data
        if len(self.centerline_history) < 3:
            # Apply spatial smoothing on current frame via sliding window average
            smoothed_data = centerline_data.copy()
            window_size = 5
            for i in range(len(smoothed_data)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(smoothed_data), i + window_size // 2 + 1)
                window = smoothed_data[start_idx:end_idx]
                if len(window) > 0:
                    smoothed_data[i, 1] = np.mean(window[:, 1])  # smooth x coordinate
                    smoothed_data[i, 2] = np.mean(window[:, 2])  # smooth width
            return smoothed_data
        
        # Temporal smoothing: weighted average over historical frames
        smoothed_data = centerline_data.copy()

        # For each y coordinate, find corresponding data in historical frames
        for i, (y, x, width) in enumerate(centerline_data):
            x_values = [x]
            width_values = [width]
            weights = [1.0]  # current frame has the highest weight

            # Look up nearby y coordinates in the last 2 historical frames
            for hist_idx, hist_data in enumerate(self.centerline_history[-3:-1]):
                # Find the closest y coordinate
                y_diffs = np.abs(hist_data[:, 0] - y)
                if len(y_diffs) > 0:
                    closest_idx = np.argmin(y_diffs)
                    if y_diffs[closest_idx] < 10:  # y difference < 10 pixels
                        x_values.append(hist_data[closest_idx, 1])
                        width_values.append(hist_data[closest_idx, 2])
                        # Historical frame weight decreases with age
                        weights.append(0.5 ** (len(self.centerline_history) - hist_idx - 1))
            
            # Weighted average
            if len(x_values) > 1:
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                smoothed_data[i, 1] = np.sum(np.array(x_values) * weights)
                smoothed_data[i, 2] = np.sum(np.array(width_values) * weights)
        
        # Spatial smoothing: second sliding-window pass on the result
        window_size = 3
        final_data = smoothed_data.copy()
        for i in range(len(final_data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(final_data), i + window_size // 2 + 1)
            window = smoothed_data[start_idx:end_idx]
            if len(window) > 0:
                final_data[i, 1] = np.mean(window[:, 1])
                final_data[i, 2] = np.mean(window[:, 2])
        
        return final_data

    def _estimate_affine(self, prev_gray, curr_gray, mask=None):
        """Estimate affine transform using optical flow (fallback method)."""
        try:
            # Extract feature points
            if mask is not None:
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **self.feature_params)
            else:
                p0 = cv2.goodFeaturesToTrack(prev_gray, **self.feature_params)
            
            if p0 is None or len(p0) < 4:
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            
            # Compute optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)

            if p1 is None or st is None:
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

            # Keep only successfully tracked points
            good_new = p1[st == 1].reshape(-1, 2)
            good_old = p0[st == 1].reshape(-1, 2)

            if len(good_new) < 4:
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

            # Estimate affine transform
            M, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
            
            if M is None:
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            
            return M
            
        except Exception as e:
            logger.debug(f"Affine estimation failed: {e}")
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    
    def _warp_mask(self, mask, M, output_shape):
        """Apply an affine transformation to a mask."""
        try:
            W, H = output_shape
            warped = cv2.warpAffine(mask, M, (W, H), 
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
            return warped
        except:
            return None
    
    def _add_mask_visualization(self, mask, visualizations, viz_type, color, add_outline=True):
        """Add mask visualization with optional outline."""
        if mask is None:
            return

        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                points = main_contour.squeeze(1)[::5].tolist()

                # Add fill
                visualizations.append({
                    "type": viz_type,
                    "points": points,
                    "color": color
                })

                # Add outline (tactile path mask has no outline)
                if add_outline and viz_type != "blind_path_mask":
                    visualizations.append({
                        "type": "outline",
                        "points": points,
                        "color": "rgba(255, 255, 255, 0.8)",  # white outline
                        "thickness": 3
                    })
        except:
            pass

    
    def _update_crosswalk_tracker(self, crosswalk_mask, image_height, image_width):
        """Update the crosswalk tracker."""
        if crosswalk_mask is not None:
            self.crosswalk_tracker['consecutive_frames'] += 1
            self.crosswalk_tracker['last_seen_frame'] = self.frame_counter

            # Compute key metrics
            total_area = image_height * image_width
            area_ratio = np.sum(crosswalk_mask > 0) / total_area
            y_coords, x_coords = np.where(crosswalk_mask > 0)
            
            if len(y_coords) > 0:
                bottom_y_ratio = np.max(y_coords) / image_height
                center_x_ratio = np.mean(x_coords) / image_width
                
                self.crosswalk_tracker['last_area_ratio'] = area_ratio
                self.crosswalk_tracker['last_bottom_y_ratio'] = bottom_y_ratio
                self.crosswalk_tracker['last_center_x_ratio'] = center_x_ratio
                
                # Compute angle
                try:
                    contours, _ = cv2.findContours(crosswalk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        main_contour = max(contours, key=cv2.contourArea)
                        rect = cv2.minAreaRect(main_contour)
                        angle = rect[-1]
                        w, h = rect[1]
                        if w < h:
                            angle += 90
                        self.crosswalk_tracker['last_angle'] = angle
                except:
                    self.crosswalk_tracker['last_angle'] = 0.0
                
                # State transition
                is_ready_to_switch = (
                    area_ratio >= self.CROSSWALK_SWITCH_AREA_RATIO and
                    bottom_y_ratio >= self.CROSSWALK_SWITCH_BOTTOM_RATIO or
                    (self.crosswalk_tracker['consecutive_frames'] >= self.CROSSWALK_SWITCH_CONSECUTIVE_FRAMES 
                     and area_ratio > 0.18)
                )
                
                if is_ready_to_switch and self.crosswalk_tracker['alignment_status'] == 'aligned':
                    if self.crosswalk_tracker['stage'] != 'ready':
                        self.crosswalk_tracker['stage'] = 'ready'
                elif area_ratio > 0.07 or bottom_y_ratio > 0.75:
                    if self.crosswalk_tracker['stage'] in ['far', 'not_detected']:
                        self.crosswalk_tracker['stage'] = 'approaching'
                elif area_ratio > 0.01:
                    if self.crosswalk_tracker['stage'] == 'not_detected':
                        self.crosswalk_tracker['stage'] = 'far'
        else:
            # Detection lost
            if self.frame_counter - self.crosswalk_tracker['last_seen_frame'] > 15:
                self.crosswalk_tracker['stage'] = 'not_detected'
                self.crosswalk_tracker['consecutive_frames'] = 0
                self.crosswalk_tracker['position_announced'] = False
                self.crosswalk_tracker['alignment_status'] = 'not_aligned'
                # Reset ready state flags
                if hasattr(self, 'crosswalk_ready_announced'):
                    self.crosswalk_ready_announced = False
                    self.crosswalk_ready_time = 0
                if hasattr(self, 'traffic_light_history'):
                    self.traffic_light_history.clear()
                    self.green_light_announced = False
    
    def _handle_crosswalk_approaching(self, frame_visualizations, image_height, image_width, image):
        """Handle the crosswalk-approaching state."""
        # Obstacle detection
        if self.obstacle_detector and self.frame_counter % self.OBSTACLE_DETECTION_INTERVAL == 0:
            detected_obstacles = self._detect_obstacles(image)
            self.last_detected_obstacles = detected_obstacles
            self.last_obstacle_detection_frame = self.frame_counter
        
        # Add obstacle visualization
        for obs in self.last_detected_obstacles:
            self._add_obstacle_visualization(obs, frame_visualizations)

        # Check for near obstacles (high thresholds: only very close objects trigger an alert)
        NEAR_DISTANCE_Y_THRESHOLD = 0.75
        NEAR_DISTANCE_AREA_THRESHOLD = 0.12
        near_obstacles = [
            obs for obs in self.last_detected_obstacles
            if (obs.get('bottom_y_ratio', 0) > NEAR_DISTANCE_Y_THRESHOLD or
                obs.get('area_ratio', 0) > NEAR_DISTANCE_AREA_THRESHOLD)
        ]
        
        # If near obstacles exist, apply announcement logic
        if near_obstacles:
            main_obstacle = near_obstacles[0]
            obstacle_name = main_obstacle.get('name', '')
            current_time = time.time()
            
            # Check whether to announce (avoid duplicate)
            should_announce = False
            if obstacle_name != self.last_obstacle_speech:
                should_announce = True
                self.last_obstacle_speech = obstacle_name
                self.last_obstacle_speech_time = current_time
            elif current_time - self.last_obstacle_speech_time > self.obstacle_speech_cooldown:
                should_announce = True
                self.last_obstacle_speech_time = current_time

            if should_announce:
                return self._speech_for_obstacle(obstacle_name)
        else:
            # No near obstacles: clear record
            self.last_obstacle_speech = ""

        # Alignment logic
        if self.crosswalk_tracker['alignment_status'] == 'not_aligned':
            guidance_text = "Approaching crosswalk, aligning your direction."
            self.crosswalk_tracker['alignment_status'] = 'aligning'
        else:
            angle = self.crosswalk_tracker['last_angle']
            center_x_ratio = self.crosswalk_tracker['last_center_x_ratio']
            
            ANGLE_ALIGN_THRESHOLD = 15
            POSITION_ALIGN_THRESHOLD = 0.25
            
            if abs(angle) > ANGLE_ALIGN_THRESHOLD:
                guidance_text = "Turn right" if angle < 0 else "Turn left"
            elif abs(center_x_ratio - 0.5) > (POSITION_ALIGN_THRESHOLD / 2):
                guidance_text = "Shift right" if center_x_ratio < 0.5 else "Shift left"
            else:
                self.crosswalk_tracker['alignment_status'] = 'aligned'
                guidance_text = "Crosswalk aligned, continue forward."
        
        data_for_panel = {
            "Status": "Aligning to crosswalk",
            "Guidance": guidance_text,
            "Angle": f"{self.crosswalk_tracker['last_angle']:.1f}°",
            "Offset": f"{(self.crosswalk_tracker['last_center_x_ratio'] - 0.5):.2f}"
        }
        frame_visualizations.append({
            "type": "data_panel",
            "data": data_for_panel,
            "position": (25, image_height - 75)
        })
        
        return guidance_text
    
    def _execute_state_machine(self, mask, image, frame_visualizations,
                              image_height, image_width, curr_gray):
        """Execute the navigation state machine."""
        if self.current_state == STATE_ONBOARDING:
            return self._handle_onboarding(mask, image, frame_visualizations, 
                                         image_height, image_width)
        elif self.current_state == STATE_NAVIGATING:
            return self._handle_navigating(mask, image, frame_visualizations,
                                         image_height, image_width, curr_gray)
        elif self.current_state == STATE_MANEUVERING_TURN:
            return self._handle_maneuvering_turn(mask, image, frame_visualizations,
                                               image_height, image_width)
        elif self.current_state == STATE_LOCKING_ON:
            return self._handle_locking_on(frame_visualizations)
        elif self.current_state == STATE_AVOIDING_OBSTACLE:
            return self._handle_avoiding_obstacle(mask, image, frame_visualizations,
                                                image_height, image_width)
        
        return ""
    
    def _handle_onboarding(self, mask, image, frame_visualizations, image_height, image_width):
        """Handle the onboarding (step-onto-path) state."""
        image_center_x = image_width / 2
        vp_features = self._get_vanishing_point_features(mask)

        if vp_features and vp_features['fit_error'] < self.VP_FIT_ERROR_THRESHOLD:
            # Use vanishing-point method
            VP, L_center = vp_features["VP"], vp_features["L_center"]
            
            if self.onboarding_step == ONBOARDING_STEP_ROTATION:
                if abs(VP[0] - image_center_x) < (image_width * self.ONBOARDING_ALIGN_THRESHOLD_RATIO):
                    guidance_text = "Direction aligned! Now calibrating position."
                    self.onboarding_step = ONBOARDING_STEP_TRANSLATION
                else:
                    guidance_text = "Please turn left." if VP[0] < image_center_x else "Please turn right."
                
                angle_error_px = VP[0] - image_center_x
                self._add_data_panel(frame_visualizations, {
                    "Status": "Onboarding (direction)",
                    "Guidance": guidance_text,
                    "Angle": f"{angle_error_px:.1f}px",
                    "Offset": "pending calibration"
                }, (25, image_height - 75))
                
            elif self.onboarding_step == ONBOARDING_STEP_TRANSLATION:
                L_center_bottom_x = self._calculate_line_x_at_y(L_center, image_height - 1)
                
                if L_center_bottom_x:
                    center_offset_pixels = L_center_bottom_x - image_center_x
                    center_offset_ratio = abs(center_offset_pixels) / image_width
                    
                    if center_offset_ratio < self.ONBOARDING_CENTER_OFFSET_THRESHOLD_RATIO:
                        guidance_text = "Calibration complete! You are on the path, start walking."
                        self.current_state = STATE_NAVIGATING
                    else:
                        guidance_text = "Please shift left." if L_center_bottom_x < image_center_x else "Please shift right."
                    
                    self._add_data_panel(frame_visualizations, {
                        "Status": "Onboarding (position)",
                        "Guidance": guidance_text,
                        "Angle": "aligned",
                        "Offset": f"{center_offset_ratio * 100:.1f}%"
                    }, (25, image_height - 75))
                else:
                    guidance_text = "Please move forward for a clearer view of the path."
        else:
            # Use pixel-domain method
            pixel_features = self._get_pixel_domain_features(mask, image.shape)
            if not pixel_features:
                return ""
            self._add_navigation_info_visualization(pixel_features, image_height, image_width, frame_visualizations)
            guidance_text = self._handle_pixel_domain_onboarding(
                pixel_features, image_height, image_width, frame_visualizations
            )
        
        return guidance_text
    
    def _handle_navigating(self, mask, image, frame_visualizations,
                          image_height, image_width, curr_gray):
        """Handle the regular navigation state."""
        image_center_x = image_width / 2

        # Extract path features
        features = self._get_pixel_domain_features(mask, image.shape)
        if not features:
            return "Path feature extraction failed"
        self._add_navigation_info_visualization(features, image_height, image_width, frame_visualizations)
        
        # Turn detection
        if self.turn_cooldown_frames == 0:
            corner_info = self._detect_sharp_corner(features['centerline_data'])
            if corner_info:
                self._update_turn_tracker(corner_info)
                
                if self.turn_detection_tracker['consecutive_hits'] >= 3:
                    stable_corner_info = self.turn_detection_tracker['corner_info']
                    corner_y = stable_corner_info['corner_point_pixel'][1]
                    turn_trigger_y_threshold = image_height * 0.65
                    
                    if corner_y > turn_trigger_y_threshold:
                        # Trigger turn maneuver
                        self.current_state = STATE_MANEUVERING_TURN
                        self.maneuver_target_info = stable_corner_info
                        self.maneuver_step = MANEUVER_STEP_1_ISSUE_COMMAND
                        self._reset_turn_tracker()
                        # Do not announce "reaching turn" — let subsequent logic handle it
                        return ""
                    else:
                        # Turn preview disabled: continue regular navigation
                        pass
        
        # Priority 1: obstacle detection (highest priority)
        obstacles = self._check_obstacles(image, mask, frame_visualizations)
        if obstacles:
            main_obstacle = obstacles[0]
            obstacle_name = main_obstacle.get('name', '')
            current_time = time.time()

            # Check whether to announce (avoid duplicate)
            should_announce = False
            if obstacle_name != self.last_obstacle_speech:
                # Different obstacle: announce immediately
                should_announce = True
                self.last_obstacle_speech = obstacle_name
                self.last_obstacle_speech_time = current_time
            elif current_time - self.last_obstacle_speech_time > self.obstacle_speech_cooldown:
                # Same obstacle but cooldown elapsed: announce again
                should_announce = True
                self.last_obstacle_speech_time = current_time

            if should_announce:
                # Queue the obstacle voice instead of returning immediately
                self.pending_obstacle_voice = self._speech_for_obstacle(obstacle_name)
            # If no announcement needed, continue normal navigation
        else:
            # No obstacles: clear record
            self.last_obstacle_speech = ""
            self.pending_obstacle_voice = None

        # Priority 2: regular navigation (shift/turn > go straight)
        return self._generate_navigation_guidance(
            features, image_height, image_width, frame_visualizations
        )
    
    def _handle_maneuvering_turn(self, mask, image, frame_visualizations,
                                image_height, image_width):
        """Handle the turn-maneuvering state."""
        features = self._get_pixel_domain_features(mask, image.shape)
        if not features:
            return "Path lost, searching again."
        self._add_navigation_info_visualization(features, image_height, image_width, frame_visualizations)
        if self.maneuver_step == MANEUVER_STEP_1_ISSUE_COMMAND:
            direction_text = 'right' if self.maneuver_target_info['direction'] == 'right' else 'left'
            guidance_text = f"Please shift {direction_text}."
            
            poly_func = features['poly_func']
            y_check = image_height * 0.7
            self.maneuver_target_info['old_path_center_x'] = poly_func(y_check)
            
            self.maneuver_step = MANEUVER_STEP_2_WAIT_FOR_SHIFT
            
            self._add_data_panel(frame_visualizations, {
                "Status": "Maneuvering turn",
                "Guidance": guidance_text,
                "Step": "issuing command",
                "Direction": direction_text
            }, (25, image_height - 75))
            
            return guidance_text
            
        elif self.maneuver_step == MANEUVER_STEP_2_WAIT_FOR_SHIFT:
            old_path_x = self.maneuver_target_info.get('old_path_center_x')
            if old_path_x is None:
                self.maneuver_step = MANEUVER_STEP_1_ISSUE_COMMAND
                return ""
            
            poly_func = features['poly_func']
            y_check = image_height * 0.7
            current_path_x = poly_func(y_check)
            shift_distance = abs(current_path_x - old_path_x)
            
            centerline_data = features['centerline_data']
            width_at_check_y = self._get_width_at_y(centerline_data, y_check)
            
            if shift_distance > (width_at_check_y * 0.5):
                guidance_text = "Movement detected, aligning to new direction."
                self.maneuver_step = MANEUVER_STEP_3_ALIGN_ON_NEW_PATH
            else:
                direction_text = 'right' if self.maneuver_target_info['direction'] == 'right' else 'left'
                guidance_text = f"Please keep shifting {direction_text}."
            
            self._add_data_panel(frame_visualizations, {
                "Status": "Maneuvering turn",
                "Guidance": guidance_text,
                "Step": "waiting for shift",
                "Shift": f"{shift_distance:.1f}px"
            }, (25, image_height - 75))
            
            return guidance_text
            
        elif self.maneuver_step == MANEUVER_STEP_3_ALIGN_ON_NEW_PATH:
            poly_func = features['poly_func']
            y_check = image_height * 0.5
            current_path_x_at_center = poly_func(y_check)
            
            pixel_error = current_path_x_at_center - image_width / 2
            center_offset_ratio = abs(pixel_error) / image_width
            
            if center_offset_ratio < self.NAV_CENTER_OFFSET_THRESHOLD_RATIO:
                guidance_text = "Aligned to new path, please walk straight ahead."
                self.current_state = STATE_NAVIGATING
                self.maneuver_target_info = None
                self.turn_cooldown_frames = self.TURN_COOLDOWN_DURATION
            else:
                move_direction = "right" if pixel_error > 0 else "left"
                guidance_text = f"Fine-tune {move_direction} to align with path."
            
            self._add_data_panel(frame_visualizations, {
                "Status": "Maneuvering turn",
                "Guidance": guidance_text,
                "Step": "aligning to new path",
                "Error": f"{center_offset_ratio * 100:.1f}%"
            }, (25, image_height - 75))
            
            return guidance_text
    
    def _handle_locking_on(self, frame_visualizations):
        """Handle the lock-on state."""
        if not self.lock_on_data:
            self.current_state = STATE_NAVIGATING
            return ""

        main_obstacle = self.lock_on_data['main_obstacle']

        # Add pulse effect
        self._add_obstacle_visualization(main_obstacle, frame_visualizations, pulse_effect=True)

        # Check elapsed time
        if time.time() - self.lock_on_data['start_time'] > 0.7:
            self.avoidance_plan = self.lock_on_data['avoidance_plan']
            self.avoidance_step_index = 0
            self.current_state = STATE_AVOIDING_OBSTACLE
            self.lock_on_data = None
        
        return ""
    
    def _handle_avoiding_obstacle(self, mask, image, frame_visualizations,
                                 image_height, image_width):
        """Handle the obstacle-avoidance state."""
        if not self.avoidance_plan or self.avoidance_step_index >= len(self.avoidance_plan):
            self.current_state = STATE_NAVIGATING
            self.avoidance_plan = None
            return "Avoidance complete, back on the tactile path."
        
        step = self.avoidance_plan[self.avoidance_step_index]
        
        if step['type'] == 'sidestep_clear':
            direction = step['direction']
            
            if self.obstacle_detector:
                final_obstacles = self._detect_obstacles(image, mask)
            else:
                final_obstacles = []
            
            if final_obstacles:
                guidance_text = f"Path blocked, please shift {'right' if direction == 'right' else 'left'}."
            else:
                guidance_text = "Okay, stop moving sideways."
                self.avoidance_step_index += 1
            
            self._add_data_panel(frame_visualizations, {
                "Status": "Avoiding obstacle",
                "Guidance": guidance_text,
                "Step": "sidestep out",
                "Direction": direction
            }, (25, image_height - 75))
            
            return guidance_text
            
        elif step['type'] == 'forward_pass':
            # Simplified: advance to next step immediately
            self.avoidance_step_index += 1
            return "Walk forward a few steps past the obstacle, then say 'done'."
            
        elif step['type'] == 'sidestep_return':
            direction = step['direction']
            features = self._get_pixel_domain_features(mask, image.shape)
            
            if not features:
                return f"Tactile path not visible, move slightly {'right' if direction == 'right' else 'left'}."
            
            poly_func = features['poly_func']
            y_target = image_height * 0.5
            x_target = poly_func(y_target)
            
            center_offset_pixels = x_target - image_width / 2
            center_offset_ratio = abs(center_offset_pixels) / image_width
            
            if center_offset_ratio < self.NAV_CENTER_OFFSET_THRESHOLD_RATIO:
                guidance_text = "Back on the tactile path."
                self.avoidance_step_index += 1
            else:
                guidance_text = "Shift right to align with path" if center_offset_pixels > 0 else "Shift left to align with path"
            
            self._add_data_panel(frame_visualizations, {
                "Status": "Avoiding obstacle",
                "Guidance": guidance_text,
                "Step": "return to path",
                "Offset": f"{center_offset_ratio * 100:.1f}%"
            }, (25, image_height - 75))
            
            return guidance_text
    
    # ========== Helper methods ==========

    def _get_vanishing_point_features(self, mask):
        """Extract vanishing-point features."""
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: 
                return None
            main_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(main_contour) < 5000: 
                return None
            
            rect = cv2.minAreaRect(main_contour)
            center, _, angle = rect
            angle_rad = np.deg2rad(angle)
            R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], 
                          [np.sin(angle_rad), np.cos(angle_rad)]])
            points_transformed = np.dot(main_contour.squeeze(1) - center, R)
            left_points = main_contour.squeeze(1)[points_transformed[:, 0] < 0]
            right_points = main_contour.squeeze(1)[points_transformed[:, 0] >= 0]
            
            if len(left_points) < 20 or len(right_points) < 20: 
                return None
            
            [vx_l, vy_l, x_l, y_l] = cv2.fitLine(left_points, cv2.DIST_L2, 0, 0.01, 0.01)
            [vx_r, vy_r, x_r, y_r] = cv2.fitLine(right_points, cv2.DIST_L2, 0, 0.01, 0.01)
            
            a1, b1, c1 = vy_l, -vx_l, vx_l * y_l - vy_l * x_l
            a2, b2, c2 = vy_r, -vx_r, vx_r * y_r - vy_r * x_r
            determinant = a1 * b2 - a2 * b1
            
            if abs(determinant) < 1e-6: 
                return None
            
            vp_x = (b1 * c2 - b2 * c1) / determinant
            vp_y = (a2 * c1 - a1 * c2) / determinant
            L_center = ((vx_l + vx_r) / 2, (vy_l + vy_r) / 2, (x_l + x_r) / 2, (y_l + y_r) / 2)
            
            total_dist = 0
            for pt in left_points: 
                total_dist += abs((pt[0] - x_l) * vy_l - (pt[1] - y_l) * vx_l)
            for pt in right_points: 
                total_dist += abs((pt[0] - x_r) * vx_r - (pt[1] - y_r) * vy_r)
            fit_error = total_dist / (len(left_points) + len(right_points))
            
            return {"VP": (vp_x, vp_y), "L_center": L_center, "fit_error": fit_error}
        except:
            return None
    
    def _get_pixel_domain_features(self, mask, image_shape):
        """Extract pixel-domain path features."""
        try:
            height, width = image_shape[:2]
            
            centerline_data = []
            for y in range(height - 1, int(height * 0.3), -5):
                row = mask[y, :]
                x_pixels = np.where(row > 0)[0]
                if x_pixels.size > 10:
                    x_min, x_max = x_pixels[0], x_pixels[-1]
                    path_width = x_max - x_min
                    center_x = (x_min + x_max) / 2
                    centerline_data.append([y, center_x, path_width])
            
            if len(centerline_data) < 20: 
                return None
            
            data = np.array(centerline_data)
            
            # Apply centerline smoothing
            data = self._smooth_centerline(data)

            # Detect sharp turns
            sharp_turn_index = self._find_sharp_turn(data)
            if sharp_turn_index is not None:
                cutoff_index = int(sharp_turn_index * 0.6)
                if cutoff_index >= 10:
                    data = data[:cutoff_index]
            
            y_coords, x_coords, widths = data[:, 0], data[:, 1], data[:, 2]
            weights = widths
            
            # Raw polynomial fit
            coeffs_raw = np.polyfit(y_coords, x_coords, 2, w=weights)

            # Temporally smooth the polynomial coefficients
            self.poly_coeffs_history.append(coeffs_raw.copy())
            if len(self.poly_coeffs_history) > self.poly_coeffs_history_max:
                self.poly_coeffs_history.pop(0)
            
            # Exponential weighted moving average of coefficients
            if len(self.poly_coeffs_history) >= 3:
                # More recent frames get higher weight
                weights_time = np.array([0.7 ** (len(self.poly_coeffs_history) - i - 1) 
                                        for i in range(len(self.poly_coeffs_history))])
                weights_time = weights_time / np.sum(weights_time)
                
                # Weighted-average coefficients
                coeffs = np.zeros_like(coeffs_raw)
                for i, hist_coeffs in enumerate(self.poly_coeffs_history):
                    coeffs += hist_coeffs * weights_time[i]
            else:
                coeffs = coeffs_raw
            
            poly_func = np.poly1d(coeffs)
            
            curvature_proxy = abs(coeffs[0])
            tangent_slope = 2 * coeffs[0] * height + coeffs[1]
            tangent_angle_rad = np.arctan(tangent_slope)
            
            return {
                "poly_func": poly_func,
                "curvature_proxy": curvature_proxy,
                "tangent_angle_rad": tangent_angle_rad,
                "centerline_data": np.array(centerline_data)
            }
        except Exception as e:
            logger.warning(f"Pixel domain feature calculation failed: {e}")
            return None
    
    def _find_sharp_turn(self, data):
        """Find the index of a sharp turn in centerline data."""
        window_size = 5
        angle_threshold = 30
        
        for i in range(len(data) - 2 * window_size):
            front_window = data[i:i + window_size]
            back_window = data[i + window_size:i + 2 * window_size]
            
            front_dir = [front_window[-1, 1] - front_window[0, 1],
                        front_window[-1, 0] - front_window[0, 0]]
            back_dir = [back_window[-1, 1] - back_window[0, 1],
                       back_window[-1, 0] - back_window[0, 0]]
            
            angle1 = np.arctan2(front_dir[1], front_dir[0])
            angle2 = np.arctan2(back_dir[1], back_dir[0])
            angle_diff = abs(np.degrees(angle2 - angle1))
            
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff > angle_threshold:
                return i + window_size
        
        return None
    
    def _detect_sharp_corner(self, centerline_data, angle_threshold_deg=45):
        """Detect a sharp corner in the centerline data."""
        try:
            if len(centerline_data) < 15: 
                return None
            points_in_range = np.array(centerline_data)
            num_points = len(points_in_range)
            
            window_size = max(5, int(num_points * 0.15))
            best_turn_info = None
            max_angle_diff = 0
            
            for i in range(0, num_points - 2 * window_size, 2):
                front_segment = points_in_range[i:i + window_size]
                back_segment = points_in_range[i + window_size:i + 2 * window_size]
                
                if len(front_segment) < 3 or len(back_segment) < 3:
                    continue
                
                front_y = front_segment[:, 0]
                front_x = front_segment[:, 1]
                front_coeffs = np.polyfit(front_y, front_x, 1)
                front_slope = front_coeffs[0]
                
                back_y = back_segment[:, 0]
                back_x = back_segment[:, 1]
                back_coeffs = np.polyfit(back_y, back_x, 1)
                back_slope = back_coeffs[0]
                
                front_angle = np.arctan(front_slope)
                back_angle = np.arctan(back_slope)
                
                angle_diff_rad = back_angle - front_angle
                angle_diff_deg = abs(np.degrees(angle_diff_rad))
                
                if angle_diff_deg > max_angle_diff and angle_diff_deg > angle_threshold_deg:
                    max_angle_diff = angle_diff_deg
                    corner_point_idx = i + window_size
                    corner_point = points_in_range[corner_point_idx]
                    
                    direction = "right" if angle_diff_rad > 0 else "left"
                    
                    post_turn_segment = points_in_range[
                        corner_point_idx:min(corner_point_idx + window_size * 2, num_points)]
                    if len(post_turn_segment) > 0:
                        post_turn_center_x = np.mean(post_turn_segment[:, 1])
                    else:
                        post_turn_center_x = corner_point[1]
                    
                    best_turn_info = {
                        "corner_point_pixel": (corner_point[1], corner_point[0]),
                        "turn_angle": max_angle_diff,
                        "direction": direction,
                        "post_turn_center_x": post_turn_center_x,
                        "corner_point_idx": corner_point_idx
                    }
            
            return best_turn_info
        
        except Exception as e:
            logger.warning(f"Corner detection error: {e}")
            return None
    
    def _update_turn_tracker(self, corner_info):
        """Update the turn detection tracker."""
        detected_direction = corner_info['direction']
        
        if detected_direction == self.turn_detection_tracker['direction']:
            self.turn_detection_tracker['consecutive_hits'] += 1
        else:
            self.turn_detection_tracker['direction'] = detected_direction
            self.turn_detection_tracker['consecutive_hits'] = 1
        
        self.turn_detection_tracker['last_seen_frame'] = self.frame_counter
        self.turn_detection_tracker['corner_info'] = corner_info
    
    def _reset_turn_tracker(self):
        """Reset the turn detection tracker."""
        self.turn_detection_tracker = {
            'direction': None,
            'consecutive_hits': 0,
            'last_seen_frame': 0,
            'corner_info': None
        }
    
    def _calculate_line_x_at_y(self, line_params, y_target):
        """Calculate the x value of a line at a given y coordinate."""
        vx, vy, x0, y0 = line_params
        if abs(vy) < 1e-6:
            return None
        t = (y_target - y0) / vy
        x = x0 + t * vx
        return x
    
    def _get_width_at_y(self, centerline_data, y_target):
        """Get the path width at a given y coordinate."""
        ys = centerline_data[:, 0]
        ws = centerline_data[:, 2]
        idx = np.abs(ys - y_target).argmin()
        return ws[idx]
    
    def _detect_obstacles(self, image, path_mask=None):
        """Detect obstacles in the image."""
        logger.info(f"[_detect_obstacles] start, Frame={self.frame_counter}, obstacle_detector={'loaded' if self.obstacle_detector else 'not loaded'}")

        if self.obstacle_detector is None:
            logger.warning("[_detect_obstacles] Obstacle detector not loaded!")
            return []

        # Print whitelist classes on the first call only
        if not hasattr(self, '_classes_printed'):
            self._classes_printed = True
            if hasattr(self.obstacle_detector, 'WHITELIST_CLASSES'):
                logger.info("[_detect_obstacles] ===== Obstacle detection whitelist =====")
                for idx, name in enumerate(self.obstacle_detector.WHITELIST_CLASSES):
                    logger.info(f"  - class {idx}: {name}")
                logger.info(f"[_detect_obstacles] total {len(self.obstacle_detector.WHITELIST_CLASSES)} classes")

        try:
            # Use ObstacleDetectorClient.detect() which handles YOLO-E prompt setup,
            # detection, and basic filtering automatically.
            logger.info(f"[_detect_obstacles] calling ObstacleDetectorClient.detect()... image.shape={image.shape}")
            detected_obstacles = self.obstacle_detector.detect(image, path_mask=path_mask)

            logger.info(f"[_detect_obstacles] ObstacleDetectorClient returned {len(detected_obstacles)} objects")

            # ObstacleDetectorClient already returns data in the correct format:
            # - name, mask, area, area_ratio, center_x, center_y, bottom_y_ratio

            # Supplement any fields that may be missing but are needed downstream
            H, W = image.shape[:2]
            for i, obj in enumerate(detected_obstacles):
                # Add bounding box for visualization
                if 'mask' in obj and obj['mask'] is not None:
                    y_coords, x_coords = np.where(obj['mask'] > 0)
                    if len(y_coords) > 0 and len(x_coords) > 0:
                        x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                        x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
                        obj['box_coords'] = (x1, y1, x2, y2)

                        # Fill in any missing fields
                        if 'y_position_ratio' not in obj:
                            obj['y_position_ratio'] = obj.get('center_y', 0) / H
                        if 'label' not in obj:
                            obj['label'] = obj.get('name', 'unknown')
                        if 'center' not in obj:
                            obj['center'] = (obj.get('center_x', 0), obj.get('center_y', 0))
                        # Dummy confidence (ObstacleDetectorClient already filters low-confidence)
                        if 'confidence' not in obj:
                            obj['confidence'] = 0.5

                # Detailed log output
                logger.info(f"[_detect_obstacles] object {i+1}/{len(detected_obstacles)}: ")
                logger.info(f"  - class: {obj.get('name', 'unknown')}")
                logger.info(f"  - area: {obj.get('area', 0)} pixels ({obj.get('area_ratio', 0):.3f} of image)")
                logger.info(f"  - center: ({obj.get('center_x', 0):.1f}, {obj.get('center_y', 0):.1f})")
                logger.info(f"  - bottom_y_ratio: {obj.get('bottom_y_ratio', 0):.3f} (near: {'yes' if obj.get('bottom_y_ratio', 0) > 0.7 else 'no'})")
                logger.info(f"  - area_ratio: {obj.get('area_ratio', 0):.3f} (too large: {'yes' if obj.get('area_ratio', 0) > 0.1 else 'no'})")

            # ObstacleDetectorClient has already applied:
            # 1. Size filter (objects covering > 70% of the image are removed)
            # 2. Confidence filter (AIGLASS_OBS_CONF env var, default 0.25)
            # 3. If path_mask is provided: spatial filter requiring >= 100px intersection
            #    and intersection >= 1% of the object area
            # No additional filtering is needed here.

            logger.info(f"[_detect_obstacles] ===== Final result: returning {len(detected_obstacles)} obstacles =====")
            for idx, obj in enumerate(detected_obstacles):
                logger.info(f"  {idx+1}. {obj.get('name', 'unknown')} - pos:({obj.get('center_x', 0):.0f},{obj.get('center_y', 0):.0f}) "
                        f"bottom_y_ratio:{obj.get('bottom_y_ratio', 0):.2f} area_ratio:{obj.get('area_ratio', 0):.3f}")

            return detected_obstacles

        except Exception as e:
            logger.error(f"[_detect_obstacles] Obstacle detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _check_and_set_obstacle_voice(self, obstacles):
        """Check obstacles and queue a voice announcement if needed."""
        if not obstacles:
            self.last_obstacle_speech = ""
            self.pending_obstacle_voice = None
            return

        # Near-obstacle thresholds (high values: only very close objects trigger an alert)
        NEAR_DISTANCE_Y_THRESHOLD = 0.75   # bottom of obstacle must be below 75% of frame height
        NEAR_DISTANCE_AREA_THRESHOLD = 0.12  # obstacle must cover at least 12% of the frame
        
        near_obstacles = []
        for obs in obstacles:
            if (obs.get('bottom_y_ratio', 0) > NEAR_DISTANCE_Y_THRESHOLD or
                obs.get('area_ratio', 0) > NEAR_DISTANCE_AREA_THRESHOLD):
                near_obstacles.append(obs)
        
        if near_obstacles:
            # Pick the most prominent obstacle (largest area)
            main_obstacle = max(near_obstacles, key=lambda x: x.get('area_ratio', 0))
            obstacle_name = main_obstacle.get('name', '')
            current_time = time.time()

            # Check whether to announce (avoid duplicates)
            should_announce = False
            if obstacle_name != self.last_obstacle_speech:
                # Different obstacle: announce immediately
                should_announce = True
                self.last_obstacle_speech = obstacle_name
                self.last_obstacle_speech_time = current_time
            elif current_time - self.last_obstacle_speech_time > self.obstacle_speech_cooldown:
                # Same obstacle but cooldown elapsed: announce again
                should_announce = True
                self.last_obstacle_speech_time = current_time

            if should_announce:
                self.pending_obstacle_voice = self._speech_for_obstacle(obstacle_name)
        else:
            # No near obstacles
            self.last_obstacle_speech = ""
            self.pending_obstacle_voice = None

    def _check_obstacles(self, image, mask, frame_visualizations):
        """Check and handle obstacles."""
        # Cache strategy
        if self.frame_counter % self.OBSTACLE_DETECTION_INTERVAL == 0:
            final_obstacles = self._detect_obstacles(image, mask)
            # Stabilize obstacle list to avoid duplicate overlap
            if hasattr(self, 'prev_gray') and self.prev_gray is not None:
                curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                final_obstacles = self._stabilize_obstacle_list(
                    final_obstacles, 
                    self.last_detected_obstacles,
                    self.prev_gray,
                    curr_gray,
                    image.shape[:2]
                )
            self.last_detected_obstacles = final_obstacles
            self.last_obstacle_detection_frame = self.frame_counter
        else:
            if self.frame_counter - self.last_obstacle_detection_frame < self.OBSTACLE_CACHE_DURATION_FRAMES:
                final_obstacles = self.last_detected_obstacles
            else:
                final_obstacles = []
        
        # Add visualization
        for obs in final_obstacles:
            self._add_obstacle_visualization(obs, frame_visualizations)

        # Near-obstacle thresholds
        NEAR_DISTANCE_Y_THRESHOLD = 0.75
        NEAR_DISTANCE_AREA_THRESHOLD = 0.12
        
        near_obstacles = [
            obs for obs in final_obstacles
            if (obs.get('bottom_y_ratio', 0) > NEAR_DISTANCE_Y_THRESHOLD or
                obs.get('area_ratio', 0) > NEAR_DISTANCE_AREA_THRESHOLD)
        ]
        
        return near_obstacles
    
    def _plan_avoidance(self, obstacle_info, image_width):
        """Plan an obstacle-avoidance path."""
        obstacle_center_x = obstacle_info['center_x']
        image_center_x = image_width / 2
        
        if obstacle_center_x < image_center_x:
            turn_direction = 'right'
        else:
            turn_direction = 'left'
        
        plan = [
            {'type': 'sidestep_clear', 'direction': turn_direction},
            {'type': 'forward_pass'},
            {'type': 'sidestep_return', 'direction': 'left' if turn_direction == 'right' else 'right'}
        ]
        return plan
    
    def _generate_navigation_guidance(self, features, image_height, image_width, frame_visualizations):
        """Generate navigation guidance."""
        poly_func = features['poly_func']
        is_curve = features['curvature_proxy'] > self.CURVATURE_PROXY_THRESHOLD
        lookahead_ratio = 0.6 if is_curve else 0.4
        y_target = image_height * lookahead_ratio
        x_target = poly_func(y_target)
        
        # Add centerline visualization
        plot_y = np.arange(int(image_height * 0.3), image_height, 5).astype(int)
        plot_x = poly_func(plot_y).astype(int)
        centerline_points = np.vstack((plot_x, plot_y)).T.tolist()
        frame_visualizations.append({
            "type": "polyline",
            "points": centerline_points,
            "color": "yellow",
            "width": 2
        })
        
        # Add target point
        frame_visualizations.append({
            "type": "circle",
            "center": [int(x_target), int(y_target)],
            "radius": 10,
            "color": "red"
        })
        
        # Compute navigation command (priority: turn/shift > straight)
        center_offset_pixels = x_target - image_width / 2
        center_offset_ratio = abs(center_offset_pixels) / image_width
        orientation_error_rad = features['tangent_angle_rad']
        
        # Check if a turn is needed first
        if orientation_error_rad > self.NAV_ORIENTATION_THRESHOLD_RAD:
            guidance_text = "Turn left"
        elif orientation_error_rad < -self.NAV_ORIENTATION_THRESHOLD_RAD:
            guidance_text = "Turn right"
        elif center_offset_ratio > self.NAV_CENTER_OFFSET_THRESHOLD_RATIO:
            guidance_text = "Shift right" if center_offset_pixels > 0 else "Shift left"
        else:
            guidance_text = "Go straight"
        
        self._add_data_panel(frame_visualizations, {
            "Status": "Navigating",
            "Guidance": guidance_text,
            "Heading": f"{np.degrees(orientation_error_rad):.1f}°",
            "Offset": f"{center_offset_ratio * 100:.1f}%"
        }, (25, image_height - 75))
        
        return guidance_text
    
    def _handle_pixel_domain_onboarding(self, pixel_features, image_height, image_width, frame_visualizations):
        """Handle pixel-domain onboarding guidance."""
        image_center_x = image_width / 2
        orientation_error_rad = pixel_features['tangent_angle_rad']
        poly_func = pixel_features['poly_func']
        
        y_bottom = image_height - 1
        x_target_bottom = poly_func(y_bottom)
        center_offset_pixels = x_target_bottom - image_center_x
        center_offset_ratio = abs(center_offset_pixels) / image_width
        
        if self.onboarding_step == ONBOARDING_STEP_ROTATION:
            if abs(orientation_error_rad) < self.ONBOARDING_ORIENTATION_THRESHOLD_RAD:
                guidance_text = "Direction aligned! Now calibrating position."
                self.onboarding_step = ONBOARDING_STEP_TRANSLATION
            else:
                guidance_text = "Please turn left." if orientation_error_rad > 0.1 else "Please turn right."
            
            self._add_data_panel(frame_visualizations, {
                "Status": "Onboarding (direction)",
                "Guidance": guidance_text,
                "Angle": f"{np.degrees(orientation_error_rad):.1f}°",
                "Offset": "pending calibration"
            }, (25, image_height - 75))
            self._add_navigation_info_visualization(pixel_features, image_height, image_width, frame_visualizations)
    
            return guidance_text
            
        elif self.onboarding_step == ONBOARDING_STEP_TRANSLATION:
            if center_offset_ratio < self.ONBOARDING_CENTER_OFFSET_THRESHOLD_RATIO:
                guidance_text = "Calibration complete! You are on the path, start walking."
                self.current_state = STATE_NAVIGATING
            else:
                guidance_text = "Please shift right." if center_offset_pixels > 0 else "Please shift left."
            
            self._add_data_panel(frame_visualizations, {
                "Status": "Onboarding (position)",
                "Guidance": guidance_text,
                "Angle": "aligned",
                "Offset": f"{center_offset_ratio * 100:.1f}%"
            }, (25, image_height - 75))
        
        return guidance_text
    
    def _add_obstacle_visualization(self, obstacle, visualizations, pulse_effect=False):
        """Add obstacle visualization (outline only: red for near, yellow for far)."""
        try:
            bottom_y_ratio = obstacle.get('bottom_y_ratio', 0)
            area_ratio = obstacle.get('area_ratio', 0)
            
            is_near = bottom_y_ratio > 0.7 or area_ratio > 0.1

            # Add mask outline visualization
            if 'mask' in obstacle and obstacle['mask'] is not None:
                mask = obstacle['mask']
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Pick the largest contour
                    max_contour = max(contours, key=cv2.contourArea)
                    points = max_contour.squeeze(1)[::5].tolist()
                    
                    # Color by distance: red for near, yellow for far
                    if is_near:
                        outline_color = "rgba(255, 0, 0, 1.0)"  # red
                        thickness = 3
                    else:
                        outline_color = "rgba(255, 255, 0, 0.8)"  # yellow
                        thickness = 2

                    # Outline only (no fill, no text)
                    visualizations.append({
                        "type": "outline",
                        "points": points,
                        "color": outline_color,
                        "thickness": thickness
                    })
        except Exception as e:
            logger.error(f"[_add_obstacle_visualization] Failed to add obstacle visualization: {e}")

    def _add_navigation_info_visualization(self, features, image_height, image_width, frame_visualizations):
        """Add visualization for navigation computation info."""
        if not features:
            return

        try:
            # Extract computed values
            poly_func = features.get('poly_func')
            curvature_proxy = features.get('curvature_proxy', 0)
            tangent_angle_rad = features.get('tangent_angle_rad', 0)
            tangent_angle_deg = np.degrees(tangent_angle_rad)
            
            # Draw tangent direction
            if poly_func:
                # Compute tangent at the bottom of the frame
                y_bottom = image_height - 50
                x_bottom = poly_func(y_bottom)

                # Compute the tangent endpoint
                tangent_length = 100
                dx = tangent_length * np.cos(tangent_angle_rad)
                dy = tangent_length * np.sin(tangent_angle_rad)
                
                # Draw vertical baseline (dashed, pointing up)
                baseline_length = 80
                frame_visualizations.append({
                    "type": "dashed_line",
                    "start": [int(x_bottom), int(y_bottom)],
                    "end": [int(x_bottom), int(y_bottom - baseline_length)],
                    "color": "rgba(255, 255, 255, 0.6)",  # white dashed line
                    "thickness": 2
                })
                
                # Add tangent visualization
                frame_visualizations.append({
                    "type": "arrow",
                    "start": [int(x_bottom), int(y_bottom)],
                    "end": [int(x_bottom + dx), int(y_bottom - dy)],  # note: Y axis is inverted
                    "color": "rgba(0, 255, 255, 0.8)",  # cyan
                    "thickness": 3,
                    "tip_length": 0.3
                })
                
                # Draw angle arc between baseline and tangent
                arc_radius = 40
                # Baseline angle is -90° (straight up); tangent_angle_deg is relative to horizontal
                # OpenCV angles are measured counter-clockwise from the right horizontal
                start_angle = -90  # baseline (straight up)
                end_angle = -90 + tangent_angle_deg  # tangent angle
                frame_visualizations.append({
                    "type": "angle_arc",
                    "center": [int(x_bottom), int(y_bottom)],
                    "radius": arc_radius,
                    "start_angle": start_angle,
                    "end_angle": end_angle,
                    "color": "rgba(255, 200, 0, 0.8)",  # orange-yellow
                    "thickness": 2
                })
                
                # Add angle label (half-size font)
                frame_visualizations.append({
                    "type": "text_with_bg",
                    "text": f"Angle: {tangent_angle_deg:.1f}°",
                    "position": [int(x_bottom + 10), int(y_bottom - 30)],
                    "font_scale": 0.3,  # halved from 0.6
                    "color": "rgba(255, 255, 255, 1.0)",
                    "bg_color": "rgba(0, 0, 0, 0.7)"
                })
            
            # Add curvature info (half-size font)
            if curvature_proxy > 0.00001:
                curve_text = "Curve" if curvature_proxy > 0.00005 else "Gentle curve"
                frame_visualizations.append({
                    "type": "text_with_bg",
                    "text": f"{curve_text}: {curvature_proxy:.2e}",
                    "position": [20, 100],
                    "font_scale": 0.25,  # halved from 0.5
                    "color": "rgba(255, 255, 0, 1.0)",
                    "bg_color": "rgba(0, 0, 0, 0.7)"
                })
                
            # Show centerline data points and path width
            if 'centerline_data' in features:
                centerline_data = features['centerline_data']
                mid_idx = len(centerline_data) // 2
                if mid_idx < len(centerline_data):
                    y, x, width = centerline_data[mid_idx]
                    # Draw width indicator as a double-headed arrow
                    frame_visualizations.append({
                        "type": "double_arrow",  # double-headed arrow type
                        "start": [int(x - width/2), int(y)],
                        "end": [int(x + width/2), int(y)],
                        "color": "rgba(0, 255, 0, 0.8)",
                        "thickness": 2,
                        "tip_length": 0.15
                    })
                    # Add width label (half-size font)
                    frame_visualizations.append({
                        "type": "text_with_bg",
                        "text": f"Width: {width:.0f}px",
                        "position": [int(x - 30), int(y - 10)],
                        "font_scale": 0.25,  # halved from 0.5
                        "color": "rgba(255, 255, 255, 1.0)",
                        "bg_color": "rgba(0, 0, 0, 0.7)"
                    })
        except Exception as e:
            logger.error(f"Failed to add navigation info visualization: {e}")

    def _add_data_panel(self, visualizations, data, position):
        """Append a data panel element to the visualization list."""
        visualizations.append({
            "type": "data_panel",
            "data": data,
            "position": position
        })
    
    def _add_crosswalk_info_visualization(self, viz_data, image_height, image_width, visualizations):
        """Add crosswalk detection info visualization."""
        try:
            # 1. Draw crosswalk center marker (large cross)
            center_x = int(viz_data['center_x_ratio'] * image_width)
            center_y = int(viz_data['center_y_ratio'] * image_height)
            
            cross_size = 20 if viz_data['in_arrival'] else 15
            cross_color = "rgba(255, 100, 0, 1.0)" if viz_data['in_arrival'] else "rgba(0, 200, 255, 0.8)"
            
            # Horizontal line
            visualizations.append({
                "type": "line",
                "start": [center_x - cross_size, center_y],
                "end": [center_x + cross_size, center_y],
                "color": cross_color,
                "thickness": 2
            })
            # Vertical line
            visualizations.append({
                "type": "line",
                "start": [center_x, center_y - cross_size],
                "end": [center_x, center_y + cross_size],
                "color": cross_color,
                "thickness": 2
            })

            # 2. Draw arrow from screen center to crosswalk center
            screen_center_x = image_width // 2
            screen_center_y = image_height // 2
            
            # Only draw arrow when crosswalk is not near screen center
            distance = np.sqrt((center_x - screen_center_x)**2 + (center_y - screen_center_y)**2)
            if distance > 80:  # threshold: 80px offset before drawing arrow
                visualizations.append({
                    "type": "arrow",
                    "start": [screen_center_x, screen_center_y],
                    "end": [center_x, center_y],
                    "color": "rgba(255, 150, 0, 0.6)",
                    "thickness": 2,
                    "tip_length": 0.15
                })
            
            # 3. Info panel (top-right corner)
            panel_x = image_width - 180
            panel_y = 20

            panel_data = {
                "Crosswalk": viz_data['stage'],
                "Area": f"{viz_data['area_ratio']*100:.1f}%",
                "Position": viz_data['position'],
            }

            if viz_data['has_occlusion']:
                panel_data["Status"] = "occluded"
            elif viz_data['in_arrival']:
                panel_data["Status"] = "ready to cross"
            
            visualizations.append({
                "type": "data_panel",
                "data": panel_data,
                "position": (panel_x, panel_y)
            })
            
            # 4. Add area progress bar
            bar_width = 150
            bar_height = 20
            bar_x = image_width - bar_width - 20
            bar_y = panel_y + 90
            
            # Background box
            visualizations.append({
                "type": "rectangle",
                "top_left": (bar_x, bar_y),
                "bottom_right": (bar_x + bar_width, bar_y + bar_height),
                "color": "rgba(50, 50, 50, 0.7)",
                "filled": True
            })
            
            # Progress fill (capped at arrival threshold 0.25 → 100%)
            progress = min(viz_data['area_ratio'] / 0.25, 1.0)
            fill_width = int(bar_width * progress)
            
            # Color by stage
            if viz_data['in_arrival']:
                fill_color = "rgba(0, 255, 100, 0.8)"   # green (ready to cross)
            elif viz_data['area_ratio'] >= 0.18:
                fill_color = "rgba(255, 200, 0, 0.8)"   # yellow (approaching)
            elif viz_data['area_ratio'] >= 0.08:
                fill_color = "rgba(0, 200, 255, 0.8)"   # cyan (closing in)
            else:
                fill_color = "rgba(100, 150, 255, 0.8)" # blue (detected)
            
            visualizations.append({
                "type": "rectangle",
                "top_left": (bar_x + 2, bar_y + 2),
                "bottom_right": (bar_x + fill_width - 2, bar_y + bar_height - 2),
                "color": fill_color,
                "filled": True
            })
            
            # Progress bar label (small font)
            visualizations.append({
                "type": "text_with_bg",
                "text": f"Proximity: {int(progress * 100)}%",
                "position": [bar_x, bar_y - 18],
                "font_scale": 0.25,
                "color": "rgba(255, 255, 255, 1.0)",
                "bg_color": "rgba(0, 0, 0, 0.7)"
            })
            
        except Exception as e:
            logger.error(f"Failed to add crosswalk visualization: {e}")
    
    def _add_traffic_light_visualization(self, color, visualizations, image_height, image_width):
        """Add traffic light state visualization (top-right indicator)."""
        x = image_width - 100
        y = 50
        
        # Background box
        visualizations.append({
            "type": "rectangle",
            "top_left": (x - 40, y - 40),
            "bottom_right": (x + 40, y + 100),
            "color": "rgba(0, 0, 0, 0.5)",
            "filled": True
        })
        
        # Three light circles
        colors = {
            "red": [(255, 0, 0), (50, 0, 0), (50, 0, 0)],
            "yellow": [(50, 50, 0), (255, 255, 0), (50, 50, 0)],
            "green": [(0, 50, 0), (0, 50, 0), (0, 255, 0)],
            "unknown": [(50, 50, 50), (50, 50, 50), (50, 50, 50)]
        }
        
        light_colors = colors.get(color, colors["unknown"])
        positions = [y - 20, y + 20, y + 60]
        
        for i, (pos_y, light_color) in enumerate(zip(positions, light_colors)):
            # Outer ring
            visualizations.append({
                "type": "circle",
                "center": [x, pos_y],
                "radius": 18,
                "color": f"rgba(100, 100, 100, 1.0)",
                "thickness": 2
            })
            # Inner circle (light color)
            visualizations.append({
                "type": "circle",
                "center": [x, pos_y],
                "radius": 15,
                "color": f"rgba({light_color[0]}, {light_color[1]}, {light_color[2]}, 1.0)",
                "filled": True
            })
        
        # Label
        visualizations.append({
            "type": "text_with_bg",
            "text": f"Signal: {color}",
            "position": [x - 35, y + 90],
            "font_scale": 0.5,
            "color": "rgba(255, 255, 255, 1.0)",
            "bg_color": "rgba(0, 0, 0, 0.7)"
        })
    
    def _to_cn_obstacle(self, name: str) -> str:
        """Return the display name for an obstacle (falls back to generic label)."""
        try:
            key = (name or '').strip().lower()
            return _OBSTACLE_NAME_CN.get(key, 'obstacle')
        except:
            return 'obstacle'

    def _speech_for_obstacle(self, name: str) -> str:
        k = (name or '').strip().lower()
        if k == 'person': return "Person ahead, watch out."
        if k == 'car': return "Vehicle ahead, watch out."
        if k == 'bicycle': return "Bicycle ahead, stop."
        if k == 'motorcycle': return "Motorcycle ahead, stop."
        if k == 'bus': return "Bus ahead, stop."
        if k == 'truck': return "Truck ahead, stop."
        if k == 'scooter': return "Scooter ahead, stop."
        if k == 'stroller': return "Stroller ahead, stop."
        if k == 'dog': return "Dog ahead, stop."
        if k == 'animal': return "Animal ahead, stop."
        return "Obstacle ahead, watch out."

    def _draw_command_button(self, image, text):
        """Draw the bottom-center command button (unified with crosswalk mode)."""
        try:
            H, W = image.shape[:2]
            full_text = f"Current: {text if text else '—'}"

            # Button parameters
            font_px = 14
            pad_x, pad_y = 14, 8
            bottom_margin = 28

            # Compute text dimensions
            if PIL_AVAILABLE:
                try:
                    from PIL import Image as PILImage, ImageDraw, ImageFont
                    # Try to load a CJK-capable font
                    font = None
                    for font_path in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
                        if os.path.exists(font_path):
                            try:
                                font = ImageFont.truetype(font_path, font_px)
                                break
                            except:
                                continue
                    if font:
                        bbox = ImageDraw.Draw(PILImage.new('RGB', (1, 1))).textbbox((0, 0), full_text, font=font)
                        tw = max(1, bbox[2] - bbox[0])
                        th = max(1, bbox[3] - bbox[1])
                    else:
                        scale = font_px / 24.0
                        (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                except:
                    scale = font_px / 24.0
                    (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            else:
                scale = font_px / 24.0
                (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            
            # Button position (bottom-center)
            bw = tw + pad_x * 2
            bh = th + pad_y * 2
            radius = max(10, bh // 2)
            
            cx = W // 2
            left = max(8, cx - bw // 2)
            top = H - bottom_margin - bh
            right = min(W - 8, left + bw)
            bottom = top + bh
            
            # Semi-transparent rounded background
            overlay = image.copy()
            bg_color = (26, 32, 41)   # dark background
            border_color = (60, 76, 102)  # border

            # Rounded rect (center strip + two end circles)
            cv2.rectangle(overlay, (left + radius, top), (right - radius, bottom), bg_color, -1)
            cv2.circle(overlay, (left + radius, (top + bottom) // 2), radius, bg_color, -1)
            cv2.circle(overlay, (right - radius, (top + bottom) // 2), radius, bg_color, -1)

            # Blend semi-transparent
            cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

            # Draw border
            cv2.rectangle(image, (left + radius, top), (right - radius, bottom), border_color, 1)
            cv2.circle(image, (left + radius, (top + bottom) // 2), radius, border_color, 1)
            cv2.circle(image, (right - radius, (top + bottom) // 2), radius, border_color, 1)

            # Draw text
            text_x = left + pad_x
            text_y = top + pad_y + th

            if PIL_AVAILABLE and 'font' in locals() and font:
                # Use PIL for text rendering
                pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                draw.text((text_x, top + pad_y), full_text, font=font, fill=(255, 255, 255))
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            else:
                # Fall back to OpenCV
                cv2.putText(image, full_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)

            return image
        except Exception as e:
            logger.error(f"Failed to draw command button: {e}")
            return image
    
    def _parse_color(self, color_str):
        """Parse a color string and return it in BGR format."""
        try:
            if color_str.startswith('rgba('):
                values = color_str[5:-1].split(',')
                r, g, b = int(values[0]), int(values[1]), int(values[2])
                return (b, g, r)  # OpenCV uses BGR
            elif color_str == 'yellow':
                return (0, 255, 255)
            elif color_str == 'red':
                return (0, 0, 255)
            else:
                return (0, 0, 255)  # default red
        except:
            return (0, 0, 255)

    def _draw_data_panel_no_bg(self, image, data, position=(15, 15)):
        """Draw the data panel without a black background (stroke effect only)."""
        if not PIL_AVAILABLE:
            return image

        try:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img, "RGBA")

            env_scale = float(os.getenv("AIGLASS_PANEL_SCALE", "0.7"))
            base_font_size = max(10, int(round(14 * env_scale)))

            # Try multiple fonts to ensure CJK character display
            font = None
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei
                "C:/Windows/Fonts/simhei.ttf",    # SimHei
                "C:/Windows/Fonts/simsun.ttc",    # SimSun
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
                "/System/Library/Fonts/PingFang.ttc",  # macOS
            ]
            
            for font_path in font_paths:
                try:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, base_font_size)
                        break
                except:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
            
            # Draw text with stroke effect
            y_offset = position[1]
            for key, value in data.items():
                text = f"{key}: {value}"

                # Black stroke (8 directions)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text((position[0] + dx, y_offset + dy), text,
                                    font=font, fill=(0, 0, 0, 255))

                # White text
                draw.text((position[0], y_offset), text, font=font, fill=(255, 255, 255, 255))
                y_offset += base_font_size + 5

            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.warning(f"Failed to draw data panel: {e}")
            return image


    def _draw_visualizations(self, image, viz_elements):
        """Draw all visualization elements onto the frame."""
        if not viz_elements:
            return image

        # Current time for animation effects
        current_time = time.time()

        # Separate panel elements from everything else
        panel_elements = [v for v in viz_elements if v.get("type") == "data_panel"]
        standard_elements = [v for v in viz_elements if v.get("type") != "data_panel"]

        # First pass: draw fills (semi-transparent overlays)
        for element in standard_elements:
            elem_type = element.get("type")
            
            if elem_type in ['blind_path_mask', 'obstacle_mask', 'crosswalk_mask']:
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 255, 0.5)"))
                    
                    # Handle pulse effect
                    if element.get("effect") == "pulse":
                        pulse_speed = element.get("pulse_speed", 1.0)
                        alpha = 0.3 + 0.3 * np.sin(current_time * pulse_speed * 2 * np.pi)
                    else:
                        alpha = 0.4

                    # Bounding box of the polygon
                    x, y, w, h = cv2.boundingRect(points)

                    # Clamp to image bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, image.shape[1] - x)
                    h = min(h, image.shape[0] - y)

                    if w > 0 and h > 0:
                        # Binary mask: 1 inside the polygon, 0 outside
                        binary_mask = np.zeros((h, w), dtype=np.uint8)
                        local_points = points - np.array([x, y])
                        cv2.fillPoly(binary_mask, [local_points], 255)

                        # Blend only inside the polygon
                        local_region = image[y:y+h, x:x+w].copy()

                        # Color overlay
                        color_overlay = np.zeros((h, w, 3), dtype=np.uint8)
                        color_overlay[:] = color

                        # Apply binary mask blending
                        for c in range(3):
                            local_region[:, :, c] = np.where(
                                binary_mask > 0,
                                (1 - alpha) * local_region[:, :, c] + alpha * color_overlay[:, :, c],
                                local_region[:, :, c]
                            )

                        # Write blended region back
                        image[y:y+h, x:x+w] = local_region

        # Second pass: draw outlines and other elements
        for element in standard_elements:
            elem_type = element.get("type")
            
            # Draw line
            if elem_type == 'line':
                start = tuple(element.get("start", (0, 0)))
                end = tuple(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(255, 255, 255, 1.0)"))
                thickness = element.get("thickness", 2)
                cv2.line(image, start, end, color, thickness)
            
            # Draw outline/stroke
            elif elem_type == 'outline':
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 255, 1.0)"))
                    thickness = element.get("thickness", 3)
                    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
            
            # Draw polyline
            elif elem_type == 'polyline':
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 0, 1.0)"))
                    thickness = element.get("width", 2)
                    cv2.polylines(image, [points], isClosed=False, color=color, thickness=thickness)
            
            # Draw circle
            elif elem_type == 'circle':
                center = tuple(element.get("center", (0, 0)))
                radius = element.get("radius", 10)
                color = self._parse_color(element.get("color", "rgba(255, 0, 0, 1.0)"))
                thickness = element.get("thickness", -1 if element.get("filled", True) else 2)
                cv2.circle(image, center, radius, color, thickness)
            
            # Draw rectangle
            elif elem_type == 'rectangle':
                top_left = tuple(element.get("top_left", (0, 0)))
                bottom_right = tuple(element.get("bottom_right", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(0, 0, 0, 0.5)"))
                thickness = -1 if element.get("filled", True) else 2
                cv2.rectangle(image, top_left, bottom_right, color, thickness)
            
            # Draw arrow
            elif elem_type == 'arrow':
                start = tuple(element.get("start", (0, 0)))
                end = tuple(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(0, 255, 255, 1.0)"))
                thickness = element.get("thickness", 2)
                tip_length = element.get("tip_length", 0.3)
                cv2.arrowedLine(image, start, end, color, thickness, tipLength=tip_length)
            
            # Draw double-headed arrow
            elif elem_type == 'double_arrow':
                start = tuple(element.get("start", (0, 0)))
                end = tuple(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(0, 255, 0, 0.8)"))
                thickness = element.get("thickness", 2)
                tip_length = element.get("tip_length", 0.15)
                # Center line
                cv2.line(image, start, end, color, thickness)
                # Arrowheads at both ends
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = np.sqrt(dx*dx + dy*dy)
                if length > 0:
                    # Unit direction vector
                    ux, uy = dx/length, dy/length
                    # Arrowhead length
                    arrow_len = length * tip_length
                    # Left-end arrowhead
                    tip1_x = int(start[0] + arrow_len * ux)
                    tip1_y = int(start[1] + arrow_len * uy)
                    angle = np.arctan2(dy, dx)
                    arrow_angle = 30 * np.pi / 180  # arrowhead half-angle
                    p1 = (int(start[0] + arrow_len * np.cos(angle - arrow_angle)),
                          int(start[1] + arrow_len * np.sin(angle - arrow_angle)))
                    p2 = (int(start[0] + arrow_len * np.cos(angle + arrow_angle)),
                          int(start[1] + arrow_len * np.sin(angle + arrow_angle)))
                    cv2.line(image, start, p1, color, thickness)
                    cv2.line(image, start, p2, color, thickness)
                    # Right-end arrowhead
                    p3 = (int(end[0] - arrow_len * np.cos(angle - arrow_angle)),
                          int(end[1] - arrow_len * np.sin(angle - arrow_angle)))
                    p4 = (int(end[0] - arrow_len * np.cos(angle + arrow_angle)),
                          int(end[1] - arrow_len * np.sin(angle + arrow_angle)))
                    cv2.line(image, end, p3, color, thickness)
                    cv2.line(image, end, p4, color, thickness)
            
            # Draw dashed line
            elif elem_type == 'dashed_line':
                start = np.array(element.get("start", (0, 0)))
                end = np.array(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(255, 255, 255, 0.6)"))
                thickness = element.get("thickness", 2)
                dash_length = 10
                gap_length = 5
                # Total length and direction
                total_vec = end - start
                total_len = np.linalg.norm(total_vec)
                if total_len > 0:
                    unit_vec = total_vec / total_len
                    # Draw dash segments
                    current_len = 0
                    while current_len < total_len:
                        seg_start = start + unit_vec * current_len
                        seg_end = start + unit_vec * min(current_len + dash_length, total_len)
                        cv2.line(image, tuple(seg_start.astype(int)), tuple(seg_end.astype(int)), color, thickness)
                        current_len += dash_length + gap_length
            
            # Draw angle arc
            elif elem_type == 'angle_arc':
                center = tuple(element.get("center", (100, 100)))
                radius = element.get("radius", 40)
                start_angle = element.get("start_angle", -90)
                end_angle = element.get("end_angle", 0)
                color = self._parse_color(element.get("color", "rgba(255, 200, 0, 0.8)"))
                thickness = element.get("thickness", 2)
                # cv2.ellipse measures startAngle/endAngle clockwise from the right horizontal.
                # Our angles are counter-clockwise from the right horizontal (math convention),
                # so negate to convert.
                cv2_start = -end_angle
                cv2_end = -start_angle
                # Ensure correct angle ordering
                if cv2_start > cv2_end:
                    cv2_start, cv2_end = cv2_end, cv2_start
                cv2.ellipse(image, center, (radius, radius), 0, cv2_start, cv2_end, color, thickness)
            
            # Draw text with background
            elif elem_type == 'text_with_bg':
                text = element.get("text", "")
                pos = element.get("position", [10, 30])
                font_scale = element.get("font_scale", 0.6)
                color = self._parse_color(element.get("color", "rgba(255, 255, 255, 1.0)"))

                # Use the CJK-aware text drawing function
                image = self._draw_chinese_text(image, text, tuple(pos), 
                                              font_scale=font_scale, 
                                              color=color,
                                              stroke_color=(0, 0, 0),
                                              stroke_width=1)
            
            # Draw warning icon
            elif elem_type == 'warning_icon':
                pos = element.get("position", (100, 100))
                level = element.get("level", "info")
                text = element.get("text", "")
                flash = element.get("flash", False)

                # Icon color by severity level
                if level == "danger":
                    icon_color = (0, 0, 255)    # red
                    text_color = (255, 255, 255)
                elif level == "warning":
                    icon_color = (0, 165, 255)  # orange
                    text_color = (255, 255, 255)
                else:
                    icon_color = (0, 255, 255)  # yellow
                    text_color = (0, 0, 0)

                # Flashing effect
                if flash:
                    alpha = 0.5 + 0.5 * np.sin(current_time * 4 * np.pi)
                    icon_color = tuple(int(c * alpha) for c in icon_color)

                # Triangle warning icon
                triangle = np.array([
                    [pos[0], pos[1] - 20],
                    [pos[0] - 15, pos[1]],
                    [pos[0] + 15, pos[1]]
                ], np.int32)
                cv2.fillPoly(image, [triangle], icon_color)
                cv2.polylines(image, [triangle], True, (255, 255, 255), 2)

                # Exclamation mark
                cv2.putText(image, "!", (pos[0] - 5, pos[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Text label
                if text:
                    font_scale = 0.5
                    text_pos = (pos[0] - 20, pos[1] + 20)
                    image = self._draw_chinese_text(image, text, text_pos,
                                                  font_scale=font_scale,
                                                  color=text_color,
                                                  stroke_color=(0, 0, 0),
                                                  stroke_width=1)
            
            # Plain text
            elif elem_type == 'text':
                text = element.get("text", "")
                pos = tuple(element.get("pos", (10, 30)))
                image = self._draw_chinese_text(image, text, pos,
                                              font_scale=0.7,
                                              color=(255, 255, 255),
                                              stroke_color=(0, 0, 0),
                                              stroke_width=1)
        
        # Draw data panels (no-background version)
        if PIL_AVAILABLE:
            for panel in panel_elements:
                image = self._draw_data_panel_no_bg(image, panel["data"], panel["position"])
        else:
            # PIL not available — fall back to stroked OpenCV text
            for panel in panel_elements:
                y_offset = panel["position"][1]
                for key, value in panel["data"].items():
                    text = f"{key}: {value}"
                    # Stroke
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:
                                cv2.putText(image, text, (panel["position"][0] + dx, y_offset + dy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    # White text
                    cv2.putText(image, text, (panel["position"][0], y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += 25
        
        return image


    
    def _draw_chinese_text(self, image, text, position, font_scale=0.6, color=(255, 255, 255),
                         stroke_color=(0, 0, 0), stroke_width=1):
        """Draw text using a CJK-capable font with stroke effect (white text on black outline)."""
        if not PIL_AVAILABLE:
            # Fall back to cv2.putText (CJK glyphs will appear as "?")
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, color, 2)
            return image

        try:
            # Convert to PIL image
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            # Font size derived from font_scale (base 24px at scale 0.6)
            base_size = 24
            font_size = int(base_size * font_scale / 0.6)

            # Try several CJK-capable fonts in priority order
            font = None
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei
                "C:/Windows/Fonts/msyh.ttf",      # Microsoft YaHei (older)
                "C:/Windows/Fonts/simhei.ttf",    # SimHei
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux
                "/System/Library/Fonts/PingFang.ttc",  # macOS
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except:
                        continue
            
            if font is None:
                font = ImageFont.load_default()
            
            # Convert OpenCV BGR color to RGB for PIL
            rgb_color = (color[2], color[1], color[0])
            rgb_stroke = (stroke_color[2], stroke_color[1], stroke_color[0])

            # Draw text with stroke effect
            x, y = position
            # Stroke pass
            draw.text((x, y), text, font=font, fill=rgb_stroke,
                     stroke_width=stroke_width, stroke_fill=rgb_stroke)
            # Main text pass
            draw.text((x, y), text, font=font, fill=rgb_color)

            # Convert back to OpenCV format
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.warning(f"Failed to draw text: {e}")
            # Fall back to cv2.putText
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, 2)
            return image

    def _draw_data_panel(self, image, data, position=(15, 15)):
        """Draw a data panel overlay (requires Pillow)."""
        if not PIL_AVAILABLE:
            return image

        try:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img, "RGBA")

            env_scale = float(os.getenv("AIGLASS_PANEL_SCALE", "0.65"))
            base_font_size = max(8, int(round(16 * env_scale)))
            padding = max(4, int(round(8 * env_scale)))

            # Try to load a CJK-capable font
            font = None
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei
                "C:/Windows/Fonts/msyh.ttf",      # Microsoft YaHei (older)
                "C:/Windows/Fonts/simhei.ttf",    # SimHei
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, base_font_size)
                        break
                    except:
                        continue
            
            if font is None:
                font = ImageFont.load_default()
            
            text_lines = [f"{key}: {value}" for key, value in data.items()]
            text_to_draw = "\n".join(text_lines)
            
            bbox = draw.textbbox(position, text_to_draw, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            bg_rect = [
                (position[0] - padding, position[1] - padding),
                (position[0] + text_w + padding, position[1] + text_h + padding)
            ]
            draw.rectangle(bg_rect, fill=(0, 0, 0, 128))
            draw.text(position, text_to_draw, font=font, fill=(255, 255, 255, 255))
            
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        except Exception:
            return image
    
    def reset(self):
        """Reset navigator state to initial values."""
        self.current_state = STATE_ONBOARDING
        self.onboarding_step = ONBOARDING_STEP_ROTATION
        self.maneuver_step = MANEUVER_STEP_1_ISSUE_COMMAND
        self.maneuver_target_info = None
        self.turn_detection_tracker = {
            'direction': None,
            'consecutive_hits': 0,
            'last_seen_frame': 0,
            'corner_info': None
        }
        self.turn_cooldown_frames = 0
        self.avoidance_plan = None
        self.avoidance_step_index = 0
        self.lock_on_data = None
        
        # Reset optical-flow and smoothing state
        self.flow_points = {}
        self.flow_grace = {}
        self.centerline_history = []
        self.blind_miss_ttl = 0
        self.cross_miss_ttl = 0

        # Reset voice/speech state
        self.pending_obstacle_voice = None
        self.last_obstacle_speech = ""
        self.last_obstacle_speech_time = 0

        # Reset polynomial coefficient history
        self.poly_coeffs_history = []
        self.crosswalk_tracker = {
            'stage': 'not_detected',
            'consecutive_frames': 0,
            'last_area_ratio': 0.0,
            'last_bottom_y_ratio': 0.0,
            'last_center_x_ratio': 0.5,
            'position_announced': False,
            'alignment_status': 'not_aligned',
            'last_seen_frame': 0,
            'last_angle': 0.0
        }
        self.frame_counter = 0
        self.prev_gray = None
        self.prev_blind_path_mask = None
        self.prev_crosswalk_mask = None
        self.prev_obstacle_cache = []
        self.last_guidance_message = ""
        self.last_detected_obstacles = []
        self.last_obstacle_detection_frame = 0
        self.last_obstacle_speech = ""
        self.last_obstacle_speech_time = 0
        self.last_any_speech_time = 0
        self.crosswalk_ready_announced = False
        self.crosswalk_ready_time = 0
        self.traffic_light_history.clear()
        self.last_traffic_light_state = "unknown"
        self.green_light_announced = False
    
    def _stabilize_obstacle_list(self, obstacles, prev_obstacles, prev_gray, curr_gray,
                                image_shape, threshold=0.5):
        """Stabilize the obstacle list to avoid double-counting across frames."""
        if not obstacles or prev_gray is None or curr_gray is None:
            return obstacles

        H, W = image_shape
        stabilized = []
        used_prev = set()  # track which previous obstacles have been matched

        for curr_obs in obstacles:
            if 'mask' not in curr_obs or curr_obs['mask'] is None:
                stabilized.append(curr_obs)
                continue

            curr_mask = curr_obs['mask']
            best_match = None
            best_iou = 0
            best_idx = -1

            # Find the best-matching previous obstacle
            if prev_obstacles:
                for idx, prev_obs in enumerate(prev_obstacles):
                    if idx in used_prev or 'mask' not in prev_obs:
                        continue

                    # Warp the previous mask with optical flow
                    flow_mask = self._predict_mask_with_flow(prev_obs['mask'], prev_gray, curr_gray)
                    if flow_mask is None:
                        flow_mask = prev_obs['mask']

                    # Compute IoU
                    inter = np.logical_and(curr_mask > 0, flow_mask > 0).sum()
                    union = np.logical_or(curr_mask > 0, flow_mask > 0).sum()
                    iou = float(inter) / float(union) if union > 0 else 0.0

                    if iou > best_iou and iou > threshold:
                        best_iou = iou
                        best_match = flow_mask
                        best_idx = idx

            # Fuse current detection with flow-warped previous mask for stability
            if best_match is not None and best_idx >= 0:
                used_prev.add(best_idx)
                fused_mask = ((0.8 * curr_mask + 0.2 * best_match) > 128).astype(np.uint8) * 255
                curr_obs['mask'] = fused_mask
                self._update_obstacle_properties(curr_obs, H, W)
            
            stabilized.append(curr_obs)
        
        return stabilized
  
    def _speech_for_obstacle(self, name: str) -> str:
        k = (name or '').strip().lower()
        if k == 'person': return "Person ahead, watch out."
        if k == 'car': return "Vehicle ahead, watch out."
        if k == 'bicycle': return "Bicycle ahead, stop."
        if k == 'motorcycle': return "Motorcycle ahead, stop."
        if k == 'bus': return "Bus ahead, stop."
        if k == 'truck': return "Truck ahead, stop."
        if k == 'scooter': return "Scooter ahead, stop."
        if k == 'stroller': return "Stroller ahead, stop."
        if k == 'dog': return "Dog ahead, stop."
        if k == 'animal': return "Animal ahead, stop."
        return "Obstacle ahead, watch out."

    def _update_obstacle_properties(self, obs, H, W):
        """Recompute derived properties (centroid, area, bounding box) from the obstacle mask."""
        if 'mask' not in obs or obs['mask'] is None:
            return

        mask = obs['mask']
        y_coords, x_coords = np.where(mask > 0)

        if len(y_coords) > 0:
            obs['area'] = len(y_coords)
            obs['center_x'] = float(np.mean(x_coords))
            obs['center_y'] = float(np.mean(y_coords))
            obs['y_position_ratio'] = obs['center_y'] / H
            obs['area_ratio'] = obs['area'] / (H * W)
            obs['bottom_y_ratio'] = np.max(y_coords) / H

            # Update bounding box
            x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
            x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
            obs['box_coords'] = (x1, y1, x2, y2)