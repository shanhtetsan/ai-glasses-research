"""Dependency-light intent and hand-fusion helpers.

MediaPipe handedness is anatomical. Image position is an independent fact and
must never be used to rename or swap the tracker label.
"""
from __future__ import annotations

import math
import re
import threading
from copy import deepcopy
from typing import Any, Optional


HAND_HANDEDNESS_AUTHORITATIVE_THRESHOLD = 0.80
GUIDANCE_TARGET_MIN_CONFIDENCE = 0.50
GUIDANCE_TARGET_PERSISTENCE_FRAMES = 3
GUIDANCE_TARGET_IOU_THRESHOLD = 0.30
GUIDANCE_TARGET_MAX_MISSED_FRAMES = 1
GUIDANCE_RELATION_DEADBAND = 0.06
GUIDANCE_RELATION_HYSTERESIS = 0.015

# Exact class vocabulary read from the repository's validated yolov8n.pt.
# Target extraction is intentionally closed-world: text that cannot be mapped
# to one of these labels is not treated as a requested target.
SUPPORTED_YOLO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

_EXPLICIT_TARGET_ALIASES = {
    "bike": "bicycle",
    "bikes": "bicycle",
    "motorbike": "motorcycle",
    "motorbikes": "motorcycle",
    "plane": "airplane",
    "planes": "airplane",
    "ball": "sports ball",
    "balls": "sports ball",
    "sofa": "couch",
    "sofas": "couch",
    "plant": "potted plant",
    "plants": "potted plant",
    "table": "dining table",
    "tables": "dining table",
    "television": "tv",
    "televisions": "tv",
    "phone": "cell phone",
    "phones": "cell phone",
    "cellphone": "cell phone",
    "cellphones": "cell phone",
    "mobile phone": "cell phone",
    "mobile phones": "cell phone",
    "fridge": "refrigerator",
    "fridges": "refrigerator",
    "people": "person",
    "mice": "mouse",
    "knives": "knife",
    "hair dryer": "hair drier",
    "hair dryers": "hair drier",
}

_HAND_PERCEPTION_PHRASES = (
    "which hand",
    "what hand",
    "hand am i",
    "holding up",
    "left hand",
    "right hand",
    "my hand",
    "open palm",
)
_HAND_PERCEPTION_WORDS = (
    "fist",
    "palm",
    "gesture",
    "pointing",
    "reach",
    "reaching",
    "grab",
    "grabbing",
    "grasp",
    "grasping",
)

_POINTING_INTENT_RE = re.compile(r"\bpoint(?:ing|ed)?\b")
_REACH_GRAB_INTENT_RE = re.compile(
    r"\b(?:reach(?:ing|ed)?|grab(?:bing|bed)?|grasp(?:ing|ed)?|pick(?:ing|ed)?\s+up)\b"
)
_TARGET_DIRECTED_INTENT_RE = re.compile(
    r"\b(?:grab(?:bing|bed)?|grasp(?:ing|ed)?|pick(?:ing|ed)?\s+up|"
    r"reach(?:ing|ed)?\s+(?:for|toward|towards)|point(?:ing|ed)?\s+at|"
    r"find|locate|where\s+is|where\s+are)\b"
)
_DEICTIC_TARGET_RE = re.compile(r"\b(?:it|this|that|there)\b")


def normalize_object_label(value: Any) -> str:
    """Normalize only spelling separators/case; never infer semantics."""
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold())
    return " ".join(normalized.split())


def _plural_alias(label: str) -> Optional[str]:
    words = label.split()
    if not words or words[-1].endswith("s"):
        return None
    word = words[-1]
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        words[-1] = word[:-1] + "ies"
    elif word.endswith(("ch", "sh", "x", "z")):
        words[-1] = word + "es"
    elif word.endswith("fe"):
        words[-1] = word[:-2] + "ves"
    elif word.endswith("f"):
        words[-1] = word[:-1] + "ves"
    else:
        words[-1] = word + "s"
    return " ".join(words)


def _target_aliases() -> dict[str, str]:
    aliases = {normalize_object_label(label): label for label in SUPPORTED_YOLO_LABELS}
    for label in SUPPORTED_YOLO_LABELS:
        plural = _plural_alias(normalize_object_label(label))
        if plural:
            aliases.setdefault(plural, label)
    aliases.update(_EXPLICIT_TARGET_ALIASES)
    return aliases


TARGET_LABEL_ALIASES = _target_aliases()


def canonical_yolo_label(value: Any) -> Optional[str]:
    """Map an exact normalized label or declared alias to a YOLO class."""
    return TARGET_LABEL_ALIASES.get(normalize_object_label(value))


def extract_requested_target(text: str) -> Optional[str]:
    """Extract one unambiguous supported object name from the current turn."""
    normalized = normalize_object_label(text)
    if not normalized:
        return None
    padded = f" {normalized} "
    matches = {
        canonical
        for alias, canonical in TARGET_LABEL_ALIASES.items()
        if f" {alias} " in padded
    }
    return next(iter(matches)) if len(matches) == 1 else None


def guidance_anchor_for_utterance(text: str) -> str:
    """Choose a 2D hand anchor from explicit task words.

    Pointing uses the index tip. Reach/grab/grasp/pick-up tasks use the hand
    center. Mixed or absent task language conservatively defaults to index tip.
    """
    normalized = normalize_object_label(text)
    pointing = _POINTING_INTENT_RE.search(normalized) is not None
    reach_grab = _REACH_GRAB_INTENT_RE.search(normalized) is not None
    if reach_grab and not pointing:
        return "hand_center"
    return "index_tip"


def guidance_anchor_reason(text: str) -> str:
    normalized = normalize_object_label(text)
    pointing = _POINTING_INTENT_RE.search(normalized) is not None
    reach_grab = _REACH_GRAB_INTENT_RE.search(normalized) is not None
    if pointing and reach_grab:
        return "ambiguous_default"
    if pointing:
        return "pointing_intent"
    if reach_grab:
        return "reach_grab_intent"
    return "conservative_default"


def is_target_directed_request(text: str) -> bool:
    return _TARGET_DIRECTED_INTENT_RE.search(normalize_object_label(text)) is not None


def should_wait_for_explicit_target(text: str) -> bool:
    """Avoid grounding an incomplete streamed command before its noun arrives."""
    normalized = normalize_object_label(text)
    command_starts_without_noun = re.match(
        r"^(?:please\s+)?(?:grab|grasp|reach|point|pick|find|locate)\b",
        normalized,
    ) is not None
    return bool(
        (is_target_directed_request(normalized) or command_starts_without_noun)
        and extract_requested_target(normalized) is None
        and _DEICTIC_TARGET_RE.search(normalized) is None
    )


def is_hand_perception_request(text: str) -> bool:
    """Recognize only explicit hand/gesture language, not every utterance."""
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return False
    if any(phrase in normalized for phrase in _HAND_PERCEPTION_PHRASES):
        return True
    return any(
        re.search(rf"\b{re.escape(word)}\b", normalized) is not None
        for word in _HAND_PERCEPTION_WORDS
    )


def image_horizontal_position(x_norm: Any) -> Optional[str]:
    """Classify canonical RGB x without inferring anatomical handedness."""
    try:
        x = float(x_norm)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    if x < 1.0 / 3.0:
        return "image_left"
    if x > 2.0 / 3.0:
        return "image_right"
    return "image_center"


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _joint_angle_degrees(
    a: tuple[float, float],
    vertex: tuple[float, float],
    c: tuple[float, float],
) -> Optional[float]:
    first = (a[0] - vertex[0], a[1] - vertex[1])
    second = (c[0] - vertex[0], c[1] - vertex[1])
    denominator = math.hypot(*first) * math.hypot(*second)
    if denominator <= 1e-9:
        return None
    cosine = max(-1.0, min(1.0, (first[0] * second[0] + first[1] * second[1]) / denominator))
    return math.degrees(math.acos(cosine))


def classify_hand_gesture(landmarks: Any) -> dict:
    """Conservatively classify four coarse gestures from 21 MediaPipe points.

    The rules use only normalized 2D geometry and palm-relative distances.
    Thumb geometry is intentionally excluded because its apparent flexion is
    especially sensitive to viewpoint. A result is authoritative only when all
    four non-thumb fingers satisfy a strict extended/folded pattern.
    """
    result = {
        "gesture": "unknown",
        "gesture_source": "landmark_geometry",
        "gesture_authoritative": False,
        "gesture_confidence": 0.0,
    }
    if not isinstance(landmarks, list) or len(landmarks) != 21:
        return result

    points: dict[int, tuple[float, float]] = {}
    try:
        for expected_id, item in enumerate(landmarks):
            landmark_id = int(item.get("id", expected_id))
            x = float(item["x"])
            y = float(item["y"])
            if landmark_id != expected_id or not math.isfinite(x) or not math.isfinite(y):
                return result
            points[landmark_id] = (x, y)
    except (AttributeError, KeyError, TypeError, ValueError):
        return result

    wrist = points[0]
    palm_scales = [_distance(wrist, points[index]) for index in (5, 9, 13, 17)]
    palm_scale = sum(palm_scales) / len(palm_scales)
    if palm_scale <= 1e-6:
        return result

    statuses: list[str] = []
    for mcp_id, pip_id, tip_id in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
        mcp = points[mcp_id]
        pip = points[pip_id]
        tip = points[tip_id]
        angle = _joint_angle_degrees(mcp, pip, tip)
        if angle is None:
            statuses.append("ambiguous")
            continue
        tip_from_wrist = _distance(wrist, tip)
        pip_from_wrist = _distance(wrist, pip)
        tip_from_mcp = _distance(mcp, tip)
        pip_from_mcp = _distance(mcp, pip)
        if (
            angle >= 155.0
            and tip_from_wrist >= pip_from_wrist + 0.16 * palm_scale
            and tip_from_mcp >= pip_from_mcp + 0.16 * palm_scale
        ):
            statuses.append("extended")
        elif (
            angle <= 125.0
            and tip_from_wrist <= pip_from_wrist + 0.12 * palm_scale
            and tip_from_mcp <= pip_from_mcp + 0.30 * palm_scale
        ):
            statuses.append("folded")
        else:
            statuses.append("ambiguous")

    if statuses == ["extended"] * 4:
        result.update(gesture="open_palm", gesture_authoritative=True, gesture_confidence=0.90)
    elif statuses == ["folded"] * 4:
        result.update(gesture="fist", gesture_authoritative=True, gesture_confidence=0.90)
    elif statuses == ["extended", "folded", "folded", "folded"]:
        result.update(gesture="pointing", gesture_authoritative=True, gesture_confidence=0.92)
    return result


def build_hand_fusion_fact(
    hand: Any,
    authoritative_threshold: float = HAND_HANDEDNESS_AUTHORITATIVE_THRESHOLD,
) -> Optional[dict]:
    """Compact one tracker result without mutating its handedness label."""
    if not isinstance(hand, dict):
        return None
    compact = {}
    for key in (
        "hand_index",
        "handedness",
        "handedness_score",
        "index_tip_norm",
        "wrist_norm",
        "hand_center_norm",
        "bbox_norm",
    ):
        if key in hand and hand[key] is not None:
            compact[key] = hand[key]

    handedness = str(hand.get("handedness", "Unknown") or "Unknown")
    try:
        score = float(hand.get("handedness_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    if not math.isfinite(score):
        score = 0.0
    center = hand.get("hand_center_norm")
    x_norm = center[0] if isinstance(center, (list, tuple)) and len(center) == 2 else None

    # Keep the original field for API compatibility and add explicit semantics
    # for Gemini. No x-dependent code changes the anatomical label.
    compact["handedness"] = handedness
    compact["anatomical_handedness"] = handedness
    compact["handedness_source"] = "mediapipe"
    compact["handedness_authoritative"] = bool(
        handedness in {"Left", "Right"} and score >= authoritative_threshold
    )
    compact["image_position"] = image_horizontal_position(x_norm)
    compact.update(classify_hand_gesture(hand.get("landmarks")))
    return compact


def bbox_iou(first: Any, second: Any) -> float:
    """Return normalized 2D intersection-over-union for two xyxy boxes."""
    try:
        ax1, ay1, ax2, ay2 = (float(value) for value in first)
        bx1, by1, bx2, by2 = (float(value) for value in second)
    except (TypeError, ValueError):
        return 0.0
    if not all(math.isfinite(value) for value in (ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)):
        return 0.0
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


class TargetStabilityTracker:
    """Select a sticky guidance target only after label+IoU persistence."""

    def __init__(
        self,
        min_confidence: float = GUIDANCE_TARGET_MIN_CONFIDENCE,
        persistence_frames: int = GUIDANCE_TARGET_PERSISTENCE_FRAMES,
        iou_threshold: float = GUIDANCE_TARGET_IOU_THRESHOLD,
        max_missed_frames: int = GUIDANCE_TARGET_MAX_MISSED_FRAMES,
    ) -> None:
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.persistence_frames = max(1, int(persistence_frames))
        self.iou_threshold = max(0.0, min(1.0, float(iou_threshold)))
        self.max_missed_frames = max(0, int(max_missed_frames))
        self._candidate: Optional[dict] = None
        self._candidate_count = 0
        self._stable: Optional[dict] = None
        self._stable_frame_id: Any = None
        self._missed_frames = 0
        self._latest = self._uncertain(None, 0)
        self._lock = threading.Lock()

    def _uncertain(self, candidate: Optional[dict], count: int, frame_id: Any = None) -> dict:
        state = {
            "target_state": "uncertain",
            "stable": False,
            "persistence_count": count,
            "persistence_required": self.persistence_frames,
            "minimum_confidence": self.min_confidence,
            "overlap_iou_threshold": self.iou_threshold,
            "missed_frames": self._missed_frames,
            "max_missed_frames": self.max_missed_frames,
            "held_through_miss": False,
            "frame_id": frame_id,
        }
        if candidate is not None:
            state.update(self._target_fields(candidate))
        return state

    @staticmethod
    def _target_fields(target: dict) -> dict:
        bbox = [float(value) for value in target["bbox_norm"]]
        return {
            "target": str(target["label"]),
            "target_label": str(target["label"]),
            "target_confidence": round(float(target["confidence"]), 4),
            "target_bbox": bbox,
            "target_center": [
                round((bbox[0] + bbox[2]) / 2.0, 6),
                round((bbox[1] + bbox[3]) / 2.0, 6),
            ],
        }

    def _eligible(self, objects: Any) -> list[dict]:
        eligible = []
        for item in objects if isinstance(objects, list) else []:
            try:
                confidence = float(item["confidence"])
                bbox = [float(value) for value in item["bbox_norm"]]
                label = str(item["label"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                confidence >= self.min_confidence
                and label
                and len(bbox) == 4
                and all(math.isfinite(value) for value in bbox)
                and bbox[0] <= bbox[2]
                and bbox[1] <= bbox[3]
            ):
                eligible.append({**item, "label": label, "confidence": confidence, "bbox_norm": bbox})
        return eligible

    def _best_match(self, reference: dict, objects: list[dict]) -> Optional[dict]:
        matches = [
            item for item in objects
            if item["label"].casefold() == reference["label"].casefold()
            and bbox_iou(item["bbox_norm"], reference["bbox_norm"]) >= self.iou_threshold
        ]
        if not matches:
            return None
        return max(
            matches,
            key=lambda item: (bbox_iou(item["bbox_norm"], reference["bbox_norm"]), item["confidence"]),
        )

    def _stable_state(
        self,
        target: dict,
        frame_id: Any,
        *,
        held_through_miss: bool,
    ) -> dict:
        return {
            "target_state": "stable",
            "stable": True,
            "persistence_count": self._candidate_count,
            "persistence_required": self.persistence_frames,
            "minimum_confidence": self.min_confidence,
            "overlap_iou_threshold": self.iou_threshold,
            "missed_frames": self._missed_frames,
            "max_missed_frames": self.max_missed_frames,
            "held_through_miss": held_through_miss,
            "target_observation": "held_miss" if held_through_miss else "current",
            "target_observation_frame_id": self._stable_frame_id,
            "frame_id": frame_id,
            **self._target_fields(target),
        }

    def update(self, objects: Any, frame_id: Any = None) -> dict:
        eligible = self._eligible(objects)
        with self._lock:
            stable_match = self._best_match(self._stable, eligible) if self._stable else None
            if stable_match is not None:
                self._stable = deepcopy(stable_match)
                self._stable_frame_id = frame_id
                self._missed_frames = 0
                self._candidate = deepcopy(stable_match)
                self._candidate_count = max(self.persistence_frames, self._candidate_count)
                self._latest = self._stable_state(
                    stable_match, frame_id, held_through_miss=False
                )
                return deepcopy(self._latest)

            if self._stable is not None and self._missed_frames < self.max_missed_frames:
                # Hold only the last validated bbox. The state explicitly says
                # it came from a miss; no competing or low-confidence box can
                # alter the position during this grace frame.
                self._missed_frames += 1
                self._latest = self._stable_state(
                    self._stable, frame_id, held_through_miss=True
                )
                return deepcopy(self._latest)

            # Miss tolerance is exhausted. Any new or spatially inconsistent
            # detection must qualify from persistence_count=1.
            self._stable = None
            self._stable_frame_id = None
            self._missed_frames = 0
            candidate_match = self._best_match(self._candidate, eligible) if self._candidate else None
            if candidate_match is not None:
                self._candidate = deepcopy(candidate_match)
                self._candidate_count += 1
            elif eligible:
                self._candidate = deepcopy(max(eligible, key=lambda item: item["confidence"]))
                self._candidate_count = 1
            else:
                self._candidate = None
                self._candidate_count = 0

            if self._candidate is not None and self._candidate_count >= self.persistence_frames:
                self._stable = deepcopy(self._candidate)
                self._stable_frame_id = frame_id
                self._latest = self._stable_state(
                    self._candidate, frame_id, held_through_miss=False
                )
            else:
                self._latest = self._uncertain(self._candidate, self._candidate_count, frame_id)
            return deepcopy(self._latest)

    def latest(self) -> dict:
        with self._lock:
            return deepcopy(self._latest)


def _axis_relation(
    offset: float,
    negative: str,
    positive: str,
    aligned: str,
    deadband: float,
    hysteresis: float,
    previous: Optional[str],
) -> str:
    if previous == negative and offset < -(deadband - hysteresis):
        return negative
    if previous == positive and offset > deadband - hysteresis:
        return positive
    if previous == aligned and abs(offset) <= deadband + hysteresis:
        return aligned
    if offset < -deadband:
        return negative
    if offset > deadband:
        return positive
    return aligned


class HandTargetGuidanceTracker:
    """Build canonical 2D hand-to-target facts with per-axis hysteresis."""

    def __init__(
        self,
        deadband: float = GUIDANCE_RELATION_DEADBAND,
        hysteresis: float = GUIDANCE_RELATION_HYSTERESIS,
    ) -> None:
        self.deadband = max(0.0, float(deadband))
        self.hysteresis = max(0.0, min(self.deadband, float(hysteresis)))
        self._relations: dict[tuple[str, str], tuple[str, str]] = {}
        self._lock = threading.Lock()

    def build(
        self,
        target_state: Any,
        hand: Any,
        *,
        guidance_anchor: str = "index_tip",
        anchor_reason: str = "conservative_default",
        requested_target: Optional[str] = None,
        objects_age_ms: Any = None,
        hands_age_ms: Any = None,
        object_frame_id: Any = None,
        hand_frame_id: Any = None,
    ) -> Optional[dict]:
        if not isinstance(target_state, dict) or not isinstance(hand, dict):
            return None
        anchor = guidance_anchor if guidance_anchor in {"index_tip", "hand_center"} else "index_tip"
        hand_label = str(hand.get("anatomical_handedness") or hand.get("handedness") or "Unknown")
        try:
            handedness_confidence = round(float(hand.get("handedness_score", 0.0)), 4)
        except (TypeError, ValueError):
            handedness_confidence = 0.0
        bound_target = requested_target or target_state.get("requested_target")
        if target_state.get("target_state") != "stable":
            return {
                "target_state": target_state.get("target_state", "uncertain"),
                "requested_target": bound_target,
                "target_binding": target_state.get("target_binding", "explicit" if bound_target else "generic"),
                "target": target_state.get("target"),
                "target_confidence": target_state.get("target_confidence"),
                "hand": hand_label,
                "anatomical_hand": hand_label,
                "handedness_confidence": handedness_confidence,
                "handedness_authoritative": bool(hand.get("handedness_authoritative")),
                "gesture": hand.get("gesture", "unknown"),
                "gesture_source": hand.get("gesture_source", "landmark_geometry"),
                "gesture_authoritative": bool(hand.get("gesture_authoritative")),
                "guidance_anchor": anchor,
                "guidance_anchor_reason": anchor_reason,
                "held_through_miss": bool(target_state.get("held_through_miss")),
                "missed_frames": int(target_state.get("missed_frames", 0)),
                "max_missed_frames": int(target_state.get("max_missed_frames", 0)),
                "target_observation": target_state.get("target_observation"),
                "target_observation_frame_id": target_state.get("target_observation_frame_id"),
                "object_frame_id": object_frame_id,
                "objects_age_ms": objects_age_ms,
            }
        try:
            bbox = [float(value) for value in target_state["target_bbox"]]
            index_tip = [float(value) for value in hand["index_tip_norm"]]
            hand_center = [float(value) for value in hand["hand_center_norm"]]
            anchor_point = index_tip if anchor == "index_tip" else hand_center
            if len(bbox) != 4 or len(index_tip) != 2 or len(hand_center) != 2:
                return None
        except (KeyError, TypeError, ValueError):
            return None
        center = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
        dx = center[0] - anchor_point[0]
        dy = center[1] - anchor_point[1]
        key = (
            str(target_state.get("target", "unknown")),
            f"{hand_label}:{hand.get('hand_index', 0)}:{anchor}",
        )
        with self._lock:
            previous_h, previous_v = self._relations.get(key, (None, None))
            horizontal = _axis_relation(
                dx, "target_left", "target_right", "horizontally_aligned",
                self.deadband, self.hysteresis, previous_h,
            )
            vertical = _axis_relation(
                dy, "target_above", "target_below", "vertically_aligned",
                self.deadband, self.hysteresis, previous_v,
            )
            self._relations[key] = (horizontal, vertical)
        return {
            "coordinate_space": "canonical_rgb_normalized_2d",
            "requested_target": bound_target,
            "target_binding": target_state.get("target_binding", "explicit" if bound_target else "generic"),
            "target": target_state["target"],
            "target_label": target_state["target"],
            "target_confidence": target_state["target_confidence"],
            "target_state": "stable",
            "target_bbox": bbox,
            "target_center": [round(center[0], 6), round(center[1], 6)],
            "hand": hand_label,
            "anatomical_hand": hand_label,
            "handedness_confidence": handedness_confidence,
            "handedness_authoritative": bool(hand.get("handedness_authoritative")),
            "gesture": hand.get("gesture", "unknown"),
            "gesture_source": hand.get("gesture_source", "landmark_geometry"),
            "gesture_authoritative": bool(hand.get("gesture_authoritative")),
            "guidance_anchor": anchor,
            "guidance_anchor_reason": anchor_reason,
            "guidance_anchor_norm": anchor_point,
            "index_tip": index_tip,
            "hand_center": hand_center,
            "dx_norm": round(dx, 6),
            "dy_norm": round(dy, 6),
            "horizontal_relation": horizontal,
            "vertical_relation": vertical,
            "index_inside_target_bbox": bool(
                bbox[0] <= index_tip[0] <= bbox[2]
                and bbox[1] <= index_tip[1] <= bbox[3]
            ),
            "guidance_anchor_inside_target_bbox": bool(
                bbox[0] <= anchor_point[0] <= bbox[2]
                and bbox[1] <= anchor_point[1] <= bbox[3]
            ),
            "held_through_miss": bool(target_state.get("held_through_miss")),
            "missed_frames": int(target_state.get("missed_frames", 0)),
            "max_missed_frames": int(target_state.get("max_missed_frames", 0)),
            "target_observation": target_state.get("target_observation", "current"),
            "target_observation_frame_id": target_state.get("target_observation_frame_id"),
            "deadband_norm": self.deadband,
            "hysteresis_norm": self.hysteresis,
            "objects_age_ms": objects_age_ms,
            "hands_age_ms": hands_age_ms,
            "object_frame_id": object_frame_id,
            "hand_frame_id": hand_frame_id,
            "physical_distance_available": False,
        }


def compact_guidance_for_log(guidance: Any) -> str:
    if not isinstance(guidance, dict):
        return "requested=none target=none stable=no hand=none gesture=unknown anchor=index_tip"
    requested = guidance.get("requested_target") or "none"
    target = guidance.get("target") or "none"
    confidence = guidance.get("target_confidence")
    try:
        target_fact = f"{target}:{float(confidence):.2f}"
    except (TypeError, ValueError):
        target_fact = str(target)
    stable = "yes" if guidance.get("target_state") == "stable" else "no"
    hand = guidance.get("anatomical_hand") or guidance.get("hand") or "none"
    try:
        hand_fact = f"{hand}:{float(guidance.get('handedness_confidence')):.2f}"
    except (TypeError, ValueError):
        hand_fact = str(hand)
    gesture = guidance.get("gesture") or "unknown"
    anchor = guidance.get("guidance_anchor") or "index_tip"
    try:
        dx = f"{float(guidance.get('dx_norm')):.2f}"
        dy = f"{float(guidance.get('dy_norm')):.2f}"
    except (TypeError, ValueError):
        dx = dy = "na"
    horizontal = guidance.get("horizontal_relation") or "unknown"
    vertical = guidance.get("vertical_relation") or "unknown"
    if horizontal == "horizontally_aligned":
        horizontal = "aligned"
    if vertical == "vertically_aligned":
        vertical = "aligned"
    inside = guidance.get("guidance_anchor_inside_target_bbox")
    if inside is None:
        inside = guidance.get("index_inside_target_bbox")
    inside_text = "yes" if inside is True else "no" if inside is False else "na"
    return (
        f"requested={requested} target={target_fact} stable={stable} "
        f"hand={hand_fact} gesture={gesture} anchor={anchor} dx={dx} dy={dy} "
        f"h={horizontal} v={vertical} inside={inside_text}"
    )


def compact_hand_facts_for_log(hands: list[dict]) -> str:
    facts = []
    for hand in hands:
        label = hand.get("anatomical_handedness") or hand.get("handedness") or "Unknown"
        try:
            score = float(hand.get("handedness_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        position = hand.get("image_position") or "image_unknown"
        facts.append(f"{label}:{score:.2f}@{position}")
    return ",".join(facts) or "none"
