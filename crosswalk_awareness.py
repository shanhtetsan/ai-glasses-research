# -*- coding: utf-8 -*-
"""
Crosswalk awareness monitor.
Area-change-based crosswalk detection and voice guidance.
Does not trigger state transitions — provides voice guidance only.
"""
import time
import numpy as np
from collections import deque
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CrosswalkAwarenessMonitor:
    """Crosswalk awareness monitor — voice-only guidance module"""

    def __init__(self):
        # Area thresholds (fixed anchors)
        self.THRESHOLDS = {
            'discover': 0.01,      # 1%  - spotted
            'approaching': 0.08,   # 8%  - approaching
            'near': 0.18,          # 18% - close
            'arrival': 0.25,       # 25% - arrived (ready to cross)
        }

        # Broadcasted thresholds (to avoid repetition)
        self.broadcasted_thresholds = set()

        # Area history
        self.area_history = deque(maxlen=30)  # keep last 30 frames

        # Timestamp tracking
        self.last_broadcast_time = 0
        self.arrival_first_broadcast_time = 0

        # State flags
        self.in_arrival_state = False  # Whether we're in "ready to cross" state
        self.last_position_zone = None  # Last broadcasted direction zone

        # Announcement interval config (smaller = more frequent)
        # [Tuned] All intervals divided by 1.5 — increases announcement frequency 1.5×
        self.REPEAT_INTERVALS = {
            'approaching': 6.7,   # Approaching stage: repeat every 6.7s (original 10s ÷ 1.5)
            'near': 3.3,          # Close stage: repeat every 3.3s (original 5s ÷ 1.5)
            'arrival': 5.3,       # Arrived stage: repeat every 5.3s (original 8s ÷ 1.5)
        }
        # Tip: adjust the values above to change announcement frequency
        # - Smaller value = more frequent announcements
        # - Larger value = less frequent announcements

        # Occlusion detection threshold
        self.OCCLUSION_THRESHOLD = 0.30  # Overlap >30% is considered occluded

    def process_frame(self, crosswalk_mask, blind_path_mask=None) -> Optional[Dict[str, Any]]:
        """
        Process crosswalk detection for the current frame.

        Returns:
        {
            'voice_text': voice prompt text,
            'priority': priority level,
            'should_broadcast': whether to broadcast,
            'area': current area ratio,
            'position': position description,
            'visualization': visualization info (for external rendering)
        }
        or None (nothing to broadcast)
        """
        # Reset state when no crosswalk detected
        if crosswalk_mask is None:
            self._reset_if_needed()
            return None

        # 1. Compute area
        total_pixels = crosswalk_mask.size
        crosswalk_pixels = np.sum(crosswalk_mask > 0)
        area_ratio = crosswalk_pixels / total_pixels

        # 2. Compute center position
        y_coords, x_coords = np.where(crosswalk_mask > 0)
        if len(y_coords) == 0:
            return None

        center_x_ratio = np.mean(x_coords) / crosswalk_mask.shape[1]
        center_y_ratio = np.mean(y_coords) / crosswalk_mask.shape[0]

        # 3. Record history
        current_time = time.time()
        self.area_history.append({
            'area': area_ratio,
            'center_x': center_x_ratio,
            'center_y': center_y_ratio,
            'time': current_time
        })

        # 4. Check occlusion
        has_occlusion = self._check_occlusion(crosswalk_mask, blind_path_mask)

        # 5. Determine current stage and generate voice
        return self._generate_guidance(area_ratio, center_x_ratio, center_y_ratio,
                                       has_occlusion, current_time)

    def _check_occlusion(self, crosswalk_mask, blind_path_mask) -> bool:
        """Check whether the crosswalk is occluded by the tactile path."""
        if blind_path_mask is None:
            return False

        crosswalk_area = crosswalk_mask > 0
        blind_path_area = blind_path_mask > 0

        # Compute overlap
        overlap = np.logical_and(crosswalk_area, blind_path_area)
        overlap_ratio = np.sum(overlap) / max(np.sum(crosswalk_area), 1)

        # Overlap exceeding threshold counts as occluded
        return overlap_ratio > self.OCCLUSION_THRESHOLD

    def _get_position_description(self, center_x_ratio) -> str:
        """Get position description (three-zone split)."""
        if center_x_ratio < 0.40:
            return "on the left"
        elif center_x_ratio < 0.60:
            return "in the center"
        else:
            return "on the right"

    def _generate_guidance(self, area_ratio, center_x_ratio, center_y_ratio,
                          has_occlusion, current_time) -> Optional[Dict[str, Any]]:
        """Generate guidance voice output."""

        # Check if area is stable (avoid jitter)
        if not self._is_area_stable(area_ratio):
            return None

        position_desc = self._get_position_description(center_x_ratio)

        # Stage 1: Spotted (0.01–0.08)
        if area_ratio >= self.THRESHOLDS['discover'] and area_ratio < self.THRESHOLDS['approaching']:
            if self.THRESHOLDS['discover'] not in self.broadcasted_thresholds:
                self.broadcasted_thresholds.add(self.THRESHOLDS['discover'])
                return {
                    'voice_text': f"Crosswalk spotted in the distance,{position_desc}",
                    'priority': 55,  # Raised to 55 to exceed blind-path direction commands (50)
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': position_desc
                }

        # Stage 2: Approaching (0.08–0.18)
        elif area_ratio >= self.THRESHOLDS['approaching'] and area_ratio < self.THRESHOLDS['near']:
            # First announcement
            if self.THRESHOLDS['approaching'] not in self.broadcasted_thresholds:
                self.broadcasted_thresholds.add(self.THRESHOLDS['approaching'])
                self.last_broadcast_time = current_time
                self.last_position_zone = position_desc
                return {
                    'voice_text': f"Approaching crosswalk,{position_desc}",
                    'priority': 55,  # Raised to 55
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': position_desc
                }
            # Repeat announcement (every 10s or when position changes)
            elif (current_time - self.last_broadcast_time >= self.REPEAT_INTERVALS['approaching'] or
                  position_desc != self.last_position_zone):
                self.last_broadcast_time = current_time
                self.last_position_zone = position_desc
                return {
                    'voice_text': f"Approaching crosswalk,{position_desc}",
                    'priority': 55,  # Raised to 55
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': position_desc
                }

        # Stage 3: Getting close (0.18–0.25)
        elif area_ratio >= self.THRESHOLDS['near'] and area_ratio < self.THRESHOLDS['arrival']:
            # First announcement
            if self.THRESHOLDS['near'] not in self.broadcasted_thresholds:
                self.broadcasted_thresholds.add(self.THRESHOLDS['near'])
                self.last_broadcast_time = current_time
                self.last_position_zone = position_desc
                return {
                    'voice_text': f"Crosswalk getting close,{position_desc}",
                    'priority': 60,
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': position_desc
                }
            # Repeat announcement (every 5s or when position changes)
            elif (current_time - self.last_broadcast_time >= self.REPEAT_INTERVALS['near'] or
                  position_desc != self.last_position_zone):
                self.last_broadcast_time = current_time
                self.last_position_zone = position_desc
                return {
                    'voice_text': f"Crosswalk getting close,{position_desc}",
                    'priority': 60,
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': position_desc
                }

        # Stage 4: Arrived (area ≥ 0.25, not occluded)
        elif area_ratio >= self.THRESHOLDS['arrival']:
            # Occlusion must be clear before announcing crossing readiness
            if has_occlusion:
                # Occluded — do not announce crossing yet, remain in stage 3
                logger.info(f"[CROSSWALK] Area {area_ratio:.2f} reached but occluded — holding back crossing announcement")
                return None

            # First time reaching this stage
            if not self.in_arrival_state:
                self.in_arrival_state = True
                self.arrival_first_broadcast_time = current_time
                self.last_broadcast_time = current_time
                logger.info(f"[CROSSWALK] Arrived: area={area_ratio:.2f}, not occluded")
                return {
                    'voice_text': "Crosswalk reached, you can cross now.",
                    'priority': 80,
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': 'Arrived'
                }
            # Repeat announcement (every 8s)
            elif current_time - self.last_broadcast_time >= self.REPEAT_INTERVALS['arrival']:
                self.last_broadcast_time = current_time
                return {
                    'voice_text': "Crosswalk reached, you can cross now.",
                    'priority': 80,
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': 'Arrived'
                }
            # Timeout: auto-exit arrival state after 30s
            elif current_time - self.arrival_first_broadcast_time > 30.0:
                logger.info("[CROSSWALK] Arrived state timed out after 30s, auto-exit")
                self.in_arrival_state = False
                return None

        # Fallback: if area drops from the arrival state
        if self.in_arrival_state and area_ratio < 0.20:
            logger.info(f"[CROSSWALK] Area dropped to {area_ratio:.2f}, exiting arrival state")
            self.in_arrival_state = False
            # Clear some broadcasted markers to allow re-announcement
            self.broadcasted_thresholds.discard(self.THRESHOLDS['arrival'])

        return None

    def _is_area_stable(self, area_ratio, stability_frames=5) -> bool:
        """Check whether area is stable (to avoid jitter triggers)."""
        if len(self.area_history) < stability_frames:
            return True  # Early stage: assume stable

        recent_areas = [h['area'] for h in list(self.area_history)[-stability_frames:]]

        # Check that recent N frames are all within ±20% of current area
        for recent_area in recent_areas:
            if abs(recent_area - area_ratio) / max(area_ratio, 0.001) > 0.20:
                return False

        return True

    def _reset_if_needed(self):
        """Reset state when crosswalk disappears."""
        if len(self.area_history) > 0:
            logger.info("[CROSSWALK] Crosswalk disappeared, resetting state")

        self.broadcasted_thresholds.clear()
        self.area_history.clear()
        self.in_arrival_state = False
        self.last_position_zone = None

    def reset(self):
        """Full reset."""
        self.broadcasted_thresholds.clear()
        self.area_history.clear()
        self.in_arrival_state = False
        self.last_broadcast_time = 0
        self.arrival_first_broadcast_time = 0
        self.last_position_zone = None
        logger.info("[CROSSWALK] Awareness monitor reset")

    def is_in_arrival_state(self) -> bool:
        """Whether in arrival state (used externally to pause blind-path voice)."""
        return self.in_arrival_state

    def get_current_area(self) -> float:
        """Get current crosswalk area ratio."""
        if len(self.area_history) > 0:
            return self.area_history[-1]['area']
        return 0.0

    def get_visualization_data(self, crosswalk_mask, area_ratio, center_x_ratio, center_y_ratio, has_occlusion) -> Dict[str, Any]:
        """
        Get visualization data.
        Returns a dict with all visualization elements.
        """
        if crosswalk_mask is None:
            return {}

        # Determine current stage (all orange)
        if area_ratio >= self.THRESHOLDS['arrival']:
            stage = "Arrived"
            stage_color = "rgba(255, 165, 0, 0.5)"   # orange
        elif area_ratio >= self.THRESHOLDS['near']:
            stage = "Close"
            stage_color = "rgba(255, 165, 0, 0.45)"  # orange
        elif area_ratio >= self.THRESHOLDS['approaching']:
            stage = "Approaching"
            stage_color = "rgba(255, 165, 0, 0.40)"  # orange
        else:
            stage = "Spotted"
            stage_color = "rgba(255, 165, 0, 0.35)"  # orange

        # Position description
        position = self._get_position_description(center_x_ratio)

        return {
            'area_ratio': area_ratio,
            'stage': stage,
            'stage_color': stage_color,
            'position': position,
            'center_x_ratio': center_x_ratio,
            'center_y_ratio': center_y_ratio,
            'has_occlusion': has_occlusion,
            'in_arrival': self.in_arrival_state
        }


# Helper functions
def split_combined_voice(combined_text: str) -> list:
    """
    Split a combined voice string into individual prompts.
    Example: "Crosswalk spotted in the distance,on the left" → ["Crosswalk spotted in the distance", "on the left"]
    """
    if ',' in combined_text:
        parts = combined_text.split(',')
        return [p.strip() for p in parts if p.strip()]
    return [combined_text]
