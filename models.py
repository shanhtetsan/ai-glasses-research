# app/models.py
import os
import logging
import torch
from threading import Semaphore
from contextlib import contextmanager
from typing import List
from app.cloud.obstacle_detector_client import ObstacleDetectorClient
# ==========================================================
# 0. Import all model wrapper classes (Clients) and Ultralytics base class
# ==========================================================
# Wrapper class used by the crosswalk workflow
from app.cloud.crosswalk_detector_client import CrosswalkDetector
from app.cloud.coco_perception_client import COCOClient
from obstacle_detector_client import ObstacleDetectorClient

# Ultralytics class used directly by the blind-path workflow
from ultralytics import YOLO, YOLOE

logger = logging.getLogger(__name__)

# ==========================================================
# 1. Global device and concurrency control (unified)
# ==========================================================
DEVICE = os.getenv("AIGLASS_DEVICE", "cuda:0")
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    logger.warning(f"AIGLASS_DEVICE={DEVICE} but CUDA not detected — falling back to CPU")
    DEVICE = "cpu"
IS_CUDA = DEVICE.startswith("cuda")

# AMP (automatic mixed precision) configuration
AMP_POLICY = os.getenv("AIGLASS_AMP", "bf16").lower()
AMP_DTYPE = torch.bfloat16 if AMP_POLICY == "bf16" else (
    torch.float16 if AMP_POLICY == "fp16" else None) if IS_CUDA else None

# Core: single global GPU concurrency semaphore, shared across all workflows
GPU_SLOTS = int(os.getenv("AIGLASS_GPU_SLOTS", "2"))
gpu_semaphore = Semaphore(GPU_SLOTS)


# Unified inference context manager — all workflows should use this to call models
@contextmanager
def gpu_infer_slot():
    """
    Unified: GPU concurrency throttle + torch.inference_mode() + AMP autocast
    """
    with gpu_semaphore:
        if IS_CUDA and AMP_POLICY != "off" and AMP_DTYPE is not None:
            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                yield
        else:
            with torch.inference_mode():
                yield


# cuDNN acceleration optimization
try:
    if IS_CUDA:
        torch.backends.cudnn.benchmark = True
except Exception:
    pass

# ==========================================================
# 2. Global model instance definitions (all initialized to None)
# ==========================================================

# --- Crosswalk workflow models (wrapped in Client classes) ---
crosswalk_detector_client: CrosswalkDetector = None
coco_client: COCOClient = None
# ObstacleDetectorClient serves as the general-purpose obstacle detector for all scenes
obstacle_detector_client: ObstacleDetectorClient = None

# --- Blind-path workflow models (direct Ultralytics classes) ---
# Primarily used for segmentation and path planning; different detection logic from the crosswalk scene
blindpath_seg_model: YOLO = None
# Obstacle detection reuses obstacle_detector_client, but YOLOE text embeddings are kept separately
blindpath_whitelist_embeddings = None

# Global loading-state flag
models_are_loaded = False


# ==========================================================
# 3. Unified model loading function (called by celery.py at startup)
# ==========================================================
def init_all_models():
    """
    Called once when the Celery Worker process starts.
    Loads all models needed by each workflow into global variables.
    """
    global models_are_loaded
    if models_are_loaded:
        return

    logger.info(f"========= 🚀 Starting global model preload (target device: {DEVICE}) =========")

    try:
        # --- [1] Load general-purpose obstacle detector (ObstacleDetectorClient) ---
        global obstacle_detector_client
        logger.info("[1/4] Loading general-purpose obstacle detection model (ObstacleDetectorClient)...")
        obstacle_detector_client = ObstacleDetectorClient(model_path='models/yoloe-11l-seg.pt')

        # Move model to target device (critical fix — was missing)
        if hasattr(obstacle_detector_client, 'model') and obstacle_detector_client.model is not None:
            obstacle_detector_client.model.to(DEVICE)

        logger.info("...General-purpose obstacle detection model loaded successfully.")

        # --- [2] Load crosswalk-specific models (Clients) ---
        global crosswalk_detector_client, coco_client
        logger.info("[2/4] Loading crosswalk segmentation model (CrosswalkDetector)...")
        crosswalk_detector_client = CrosswalkDetector(model_path='models/yolo-seg.pt')
        # Move the internal YOLO model to the target device
        if hasattr(crosswalk_detector_client, 'model') and crosswalk_detector_client.model is not None:
            crosswalk_detector_client.model.to(DEVICE)
        logger.info("...Crosswalk segmentation model loaded successfully.")

        logger.info("[3/4] Loading general perception model (COCOClient)...")
        coco_client = COCOClient(model_path='models/yolov8l-world.pt')
        # Move the internal YOLO model to the target device
        if hasattr(coco_client, 'model') and coco_client.model is not None:
            coco_client.model.to(DEVICE)
        logger.info("...General perception model loaded successfully.")

        # --- [4] Load blind-path specific models ---
        global blindpath_seg_model, blindpath_whitelist_embeddings
        logger.info("[4/4] Loading blind-path segmentation model (YOLO)...")
        blindpath_seg_model = YOLO('models/yolo-seg.pt')
        blindpath_seg_model.to(DEVICE)
        blindpath_seg_model.fuse()
        logger.info("...Blind-path segmentation model loaded successfully.")

        # Link the obstacle model embeddings for use in the blind-path workflow
        if obstacle_detector_client:
            blindpath_whitelist_embeddings = obstacle_detector_client.whitelist_embeddings
            logger.info("...Obstacle model embeddings linked to blind-path workflow.")

        models_are_loaded = True
        logger.info("========= ✅ All models preloaded. Worker ready! =========")

    except Exception as e:
        logger.error(f"Fatal error during model preload: {e}", exc_info=True)
        # Propagate the exception — this will fail the Celery Worker at startup, which is intentional.
        # A worker without models is useless; surfacing the problem early is the right behavior.
        raise