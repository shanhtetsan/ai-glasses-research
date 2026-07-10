# general_detector.py
# -*- coding: utf-8 -*-
"""Prompt-free general object detection using a YOLOE '*-seg-pf' model.

Unlike obstacle_detector_client / yoloe_backend (which drive YOLOE with text
prompts), this uses the prompt-free variant: it detects across the model's
built-in vocabulary (~4585 classes) with no prompt at all. Loaded lazily on
first use so it never slows startup on machines that don't use it, and it picks
CPU automatically when CUDA is unavailable (laptop testing).
"""
import os
import threading
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np

_MODEL = None
_LOCK = threading.Lock()

_MODEL_PATH = os.getenv(
    "GENERAL_DET_MODEL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "yoloe-11l-seg-pf.pt"),
)


def _device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Apple GPU (~2x faster than CPU on laptop). Opt out with GENERAL_DET_NO_MPS=1.
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() \
                and os.getenv("GENERAL_DET_NO_MPS", "0") != "1":
            return "mps"
    except Exception:
        pass
    return "cpu"


def load():
    """Load the prompt-free YOLOE model once (thread-safe)."""
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                from ultralytics import YOLOE
                # Prefer the local weights; if the file is missing OR unreadable
                # (e.g. a truncated download), fall back to the bare filename so
                # ultralytics auto-downloads a fresh copy.
                try:
                    path = _MODEL_PATH if os.path.exists(_MODEL_PATH) else os.path.basename(_MODEL_PATH)
                    m = YOLOE(path)
                except Exception:
                    m = YOLOE(os.path.basename(_MODEL_PATH))
                try:
                    m.to(_device())
                except Exception:
                    pass
                _MODEL = m
    return _MODEL


def detect(
    frame_bgr: np.ndarray,
    conf: float = 0.25,
    imgsz: int = 640,
    max_det: int = 30,
) -> Tuple[Optional[np.ndarray], List[Tuple[str, int]]]:
    """Run prompt-free detection on one BGR frame.

    Returns (annotated_bgr, counts) where counts is a list of
    (class_name, count) sorted by count descending. annotated_bgr is None if
    inference fails.
    """
    model = load()
    try:
        r = model.predict(
            frame_bgr, verbose=False, conf=conf, imgsz=imgsz,
            max_det=max_det, device=_device(),
        )[0]
    except Exception:
        return None, []

    annotated = r.plot(masks=False)  # boxes + labels only, no segmentation overlay

    counts: List[Tuple[str, int]] = []
    if r.boxes is not None and getattr(r.boxes, "cls", None) is not None:
        names = [r.names[int(c)] for c in r.boxes.cls.tolist()]
        counts = Counter(names).most_common()
    return annotated, counts
