"""Dependency-light detection normalization shared by the service and tests."""
from __future__ import annotations

import math
from typing import Any


def clamp01(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("coordinate must be finite")
    return max(0.0, min(1.0, number))


def normalize_detection(raw: dict, image_width: int, image_height: int) -> dict:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    xyxy = raw.get("bbox_xyxy")
    if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
        raise ValueError("bbox_xyxy must contain four coordinates")
    x1, y1, x2, y2 = (
        clamp01(xyxy[0] / image_width),
        clamp01(xyxy[1] / image_height),
        clamp01(xyxy[2] / image_width),
        clamp01(xyxy[3] / image_height),
    )
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return {
        "class_id": int(raw["class_id"]),
        "label": str(raw["label"]),
        "confidence": clamp01(raw["confidence"]),
        "bbox_norm": [x1, y1, x2, y2],
        "center_norm": [clamp01((x1 + x2) / 2), clamp01((y1 + y2) / 2)],
    }
