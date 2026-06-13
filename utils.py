# utils.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Item name mapping
ITEM_TO_CLASS_MAP = {
    "红牛": "Red_Bull",
    "AD钙奶": "AD_milk",
    "ad钙奶": "AD_milk",
    "钙奶": "AD_milk",
}

# English class name to display label mapping
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

# Dynamic category name list
DYNAMIC_CLASS_NAMES = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'animal', 'dog'}

def extract_english_label(item_cn: str) -> tuple:
    """
    Look up the English label for a given item name.
    :param item_cn: item name (may be Chinese or English)
    :return: (English label, source)
    """
    # Check local mapping first
    if item_cn in ITEM_TO_CLASS_MAP:
        return ITEM_TO_CLASS_MAP[item_cn], "local"

    # If not found, return original name
    return item_cn, "direct"

def _to_cn_obstacle(name: str) -> str:
    """
    Map an obstacle class name to a display label.
    :param name: obstacle class name
    :return: display label
    """
    try:
        key = (name or '').strip().lower()
        return _OBSTACLE_NAME_CN.get(key, 'obstacle')
    except Exception:
        return 'obstacle'

def estimate_global_affine(prev_gray, curr_gray, mask=None):
    """
    Estimate the global affine transform between two frames.
    :param prev_gray: previous frame grayscale
    :param curr_gray: current frame grayscale
    :param mask: optional mask; features only computed inside it
    :return: (affine matrix, inlier count)
    """
    try:
        # Extract keypoints
        detector = cv2.ORB_create(nfeatures=500)
        kp1, des1 = detector.detectAndCompute(prev_gray, mask)
        kp2, des2 = detector.detectAndCompute(curr_gray, mask)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), 0

        # Match keypoints
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)

        if len(matches) < 4:
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), 0

        # Extract matched point pairs
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate affine transform with RANSAC
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                                 ransacReprojThreshold=3.0)
        
        if M is None:
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), 0
        
        inlier_count = np.sum(inliers) if inliers is not None else 0
        return M, inlier_count
        
    except Exception as e:
        logger.warning(f"estimate_global_affine failed: {e}")
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), 0

def warp_mask(mask, M, output_shape):
    """
    Apply an affine transform to a mask.
    :param mask: input mask
    :param M: 2x3 affine matrix
    :param output_shape: output shape (width, height)
    :return: transformed mask
    """
    try:
        if mask is None or M is None:
            return None
        
        W, H = output_shape
        warped = cv2.warpAffine(mask, M, (W, H), 
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
        return warped
        
    except Exception as e:
        logger.warning(f"warp_mask failed: {e}")
        return None

def estimate_translation_flow(prev_gray, curr_gray, mask=None):
    """
    Estimate translational optical flow between two frames.
    :param prev_gray: previous frame grayscale
    :param curr_gray: current frame grayscale
    :param mask: optional mask
    :return: (median flow magnitude, translation matrix)
    """
    try:
        # Compute sparse optical flow
        corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                         qualityLevel=0.3, minDistance=7,
                                         mask=mask)

        if corners is None or len(corners) < 10:
            return 0.0, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # Track points
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray,
                                                       corners, None)

        # Keep only valid points
        valid_old = corners[status == 1]
        valid_new = next_pts[status == 1]

        if len(valid_old) < 5:
            return 0.0, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # Compute displacement
        flow_vectors = valid_new - valid_old
        flow_magnitudes = np.linalg.norm(flow_vectors, axis=1)
        median_flow = np.median(flow_magnitudes)

        # Estimate mean translation
        mean_translation = np.mean(flow_vectors, axis=0)
        M = np.array([[1, 0, mean_translation[0]],
                      [0, 1, mean_translation[1]]], dtype=np.float32)
        
        return median_flow, M
        
    except Exception as e:
        logger.warning(f"estimate_translation_flow failed: {e}")
        return 0.0, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

def is_stationary_frame(prev_gray, curr_gray, mask=None, threshold=0.35):
    """
    Determine whether the user is stationary.
    :param prev_gray: previous frame grayscale
    :param curr_gray: current frame grayscale
    :param mask: optional mask
    :param threshold: stationary detection threshold
    :return: True if stationary, False if moving
    """
    try:
        median_flow, _ = estimate_translation_flow(prev_gray, curr_gray, mask)
        return median_flow < threshold
    except:
        return False

def compute_approach_metrics(prev_obstacles, curr_obstacles, M, H, W):
    """
    Compute approach metrics for obstacles.
    :param prev_obstacles: previous frame obstacle list
    :param curr_obstacles: current frame obstacle list
    :param M: affine transform matrix
    :param H: image height
    :param W: image width
    :return: list of approach metrics
    """
    metrics = []
    
    for curr_obs in curr_obstacles:
        # Find the best matching obstacle from the previous frame
        best_match = None
        best_iou = 0.0

        curr_mask = curr_obs.get('mask')
        if curr_mask is None:
            metrics.append(None)
            continue

        for prev_obs in prev_obstacles:
            prev_mask = prev_obs.get('mask')
            if prev_mask is None:
                continue

            # Warp previous frame mask to current frame
            warped_prev = warp_mask(prev_mask, M, (W, H))
            if warped_prev is None:
                continue

            # Compute IoU
            intersection = np.logical_and(curr_mask > 0, warped_prev > 0).sum()
            union = np.logical_or(curr_mask > 0, warped_prev > 0).sum()
            iou = intersection / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_match = prev_obs

        if best_match is None:
            metrics.append(None)
            continue

        # Compute metrics
        curr_area = curr_obs.get('area', 0)
        prev_area = best_match.get('area', 0)
        area_growth = (curr_area - prev_area) / prev_area if prev_area > 0 else 0.0

        curr_bottom_y = curr_obs.get('bottom_y_ratio', 0)
        prev_bottom_y = best_match.get('bottom_y_ratio', 0)
        v_forward = curr_bottom_y - prev_bottom_y
        
        metrics.append({
            'area_growth': area_growth,
            'v_forward': v_forward,
            'iou': best_iou
        })
    
    return metrics

def compute_risk_scores(obstacles, prev_obstacles, M, path_mask, image_shape,
                       stop_th=0.6, avoid_th=0.56):
    """
    Compute risk scores for obstacles.
    :param obstacles: current obstacle list
    :param prev_obstacles: previous frame obstacle list
    :param M: affine transform matrix
    :param path_mask: path mask
    :param image_shape: image shape
    :param stop_th: stop threshold
    :param avoid_th: avoidance threshold
    :return: (scored obstacle list, should_stop, should_avoid, visualization elements)
    """
    H, W = image_shape[:2]
    has_stop = False
    has_avoid = False
    risk_vis = []

    # Compute approach metrics
    metrics = compute_approach_metrics(prev_obstacles, obstacles, M, H, W)

    for obs, met in zip(obstacles, metrics):
        risk_score = 0.0

        if met is not None:
            # Risk based on approach speed and area growth
            if met['v_forward'] > 0.004:  # moving down (approaching)
                risk_score += 0.3
            if met['area_growth'] > 0.01:  # growing in area
                risk_score += 0.3

        # Risk based on proximity
        bottom_y = obs.get('bottom_y_ratio', 0)
        area_ratio = obs.get('area_ratio', 0)

        if bottom_y > 0.8 or area_ratio > 0.15:
            risk_score += 0.3

        # Extra risk for dynamic objects
        name_lower = str(obs.get('name', '')).lower()
        if name_lower in DYNAMIC_CLASS_NAMES:
            risk_score *= 1.2

        obs['risk_score'] = risk_score

        # Update flags
        if risk_score >= stop_th:
            has_stop = True
        elif risk_score >= avoid_th:
            has_avoid = True

        # Add risk visualization
        if risk_score > 0.3:
            risk_color = "rgba(255, 0, 0, 0.3)" if risk_score >= stop_th else "rgba(255, 165, 0, 0.3)"
            risk_vis.append({
                "type": "risk_indicator",
                "score": risk_score,
                "color": risk_color,
                "position": [int(obs.get('center_x', W/2)), int(obs.get('center_y', H/2))]
            })
    
    return obstacles, has_stop, has_avoid, risk_vis

