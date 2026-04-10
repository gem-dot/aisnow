#!/usr/bin/env python3
# 依赖安装：
#   conda activate ski_avatar
#   pip install -U ultralytics opencv-python tqdm torch numpy realesrgan basicsr rtmlib onnxruntime

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
except Exception:
    RRDBNet = None
    RealESRGANer = None

try:
    from rtmlib import RTMPose
except Exception:
    RTMPose = None

INPUT_VIDEO_PATH = 'input/test.mp4'
OUTPUT_VIDEO_PATH = 'output/ski_keypoints_overlay.mp4'
COMPARE_OUTPUT_VIDEO_PATH = 'output/ski_keypoints_side_by_side.mp4'
SLOW_OUTPUT_VIDEO_PATH = 'output/ski_keypoints_overlay_slow.mp4'
SLOW_COMPARE_OUTPUT_VIDEO_PATH = 'output/ski_keypoints_side_by_side_slow.mp4'
TRACK_CACHE_PATH = 'output/ski_bytetrack_boxes.pt'
KEYPOINT_CACHE_PATH = 'output/ski_rtmpose_keypoints.pt'
PRECOMPUTED_KEYPOINTS_PATH = 'model/preprocess/vitpose.pt'
LOCAL_BBX_PATH = 'model/preprocess/bbx.pt'
YOLO_DETECTOR_PATH = 'model/checkpoints/yolo/yolov8x.pt'
REAL_ESRGAN_PATH = 'model/checkpoints/realesrgan/RealESRGAN_x2plus.pth'
TRACKER_CONFIG_PATH = 'model/bytetrack_ski.yaml'
RTMPOSE_MODEL_URL = (
    'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
    'rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip'
)

PROJ_ROOT = Path(__file__).resolve().parent
os.environ.setdefault('TORCH_HOME', str(PROJ_ROOT / 'model' / 'rtmlib_cache'))

YOLO_IMGSZ = 1280
YOLO_CONF = 0.15
KEYPOINT_DRAW_CONF_THRESHOLD = 0.32
POSE_INPUT_SIZE = (288, 384)
SMALL_BOX_THRESHOLD = 120.0
SMALL_BOX_CONTEXT = 1.25
NORMAL_BOX_CONTEXT = 1.16
BACKLIGHT_TRIGGER_SCORE = 0.26
STRONG_BACKLIGHT_SCORE = 0.54
BACKLIGHT_RESCUE_CONF_SCALE = 0.72
BACKLIGHT_RESCUE_ROI_SCALE = 2.35

BONE_GLOW_COLOR = (255, 255, 0)
BONE_CORE_COLOR = (90, 255, 90)
JOINT_GLOW_COLOR = (255, 255, 0)
JOINT_CORE_COLOR = (0, 255, 180)
TEXT_BG_COLOR = (15, 22, 42)
TEXT_COLOR = (240, 250, 255)

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

POSE_SCORE_WEIGHTS = np.asarray([
    0.90, 0.75, 0.75, 0.60, 0.60,
    1.20, 1.20,
    0.95, 0.95,
    0.70, 0.70,
    1.35, 1.35,
    1.15, 1.15,
    0.90, 0.90,
], dtype=np.float32)

TEMPORAL_BASE_ALPHA = np.asarray([
    0.20, 0.18, 0.18, 0.23, 0.23,
    0.14, 0.14,
    0.18, 0.18,
    0.24, 0.24,
    0.12, 0.12,
    0.16, 0.16,
    0.21, 0.21,
], dtype=np.float32)

JOINT_JUMP_RATIO = np.asarray([
    0.17, 0.15, 0.15, 0.16, 0.16,
    0.12, 0.12,
    0.14, 0.14,
    0.18, 0.18,
    0.11, 0.11,
    0.14, 0.14,
    0.19, 0.19,
], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description='ByteTrack + RTMPose-X + Real-ESRGAN skiing keypoint overlay.')
    parser.add_argument('--input', type=str, default=INPUT_VIDEO_PATH)
    parser.add_argument('--output', type=str, default=OUTPUT_VIDEO_PATH)
    parser.add_argument('--compare-output', type=str, default=COMPARE_OUTPUT_VIDEO_PATH)
    parser.add_argument('--slow-output', type=str, default=SLOW_OUTPUT_VIDEO_PATH)
    parser.add_argument('--slow-compare-output', type=str, default=SLOW_COMPARE_OUTPUT_VIDEO_PATH)
    parser.add_argument('--slow-factor', type=float, default=0.30)
    parser.add_argument('--track-cache', type=str, default=TRACK_CACHE_PATH)
    parser.add_argument('--keypoint-cache', type=str, default=KEYPOINT_CACHE_PATH)
    parser.add_argument('--seed-kp', type=str, default=PRECOMPUTED_KEYPOINTS_PATH)
    parser.add_argument('--local-bbx', type=str, default=LOCAL_BBX_PATH)
    parser.add_argument('--detector-model', type=str, default=YOLO_DETECTOR_PATH)
    parser.add_argument('--realesrgan-model', type=str, default=REAL_ESRGAN_PATH)
    parser.add_argument('--tracker-config', type=str, default=TRACKER_CONFIG_PATH)
    parser.add_argument('--rtmpose-model', type=str, default=RTMPOSE_MODEL_URL)
    parser.add_argument('--imgsz', type=int, default=YOLO_IMGSZ)
    parser.add_argument('--conf', type=float, default=YOLO_CONF)
    parser.add_argument('--small-box-threshold', type=float, default=SMALL_BOX_THRESHOLD)
    parser.add_argument('--disable-sr', action='store_true')
    parser.add_argument('--force-redo-track', action='store_true')
    parser.add_argument('--force-redo-pose', action='store_true')
    parser.add_argument('--max-frames', type=int, default=0)
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJ_ROOT / path)


def load_precomputed_keypoints(path: Path):
    if not path.exists():
        return None
    obj = torch.load(path, map_location='cpu')
    arr = obj
    if isinstance(obj, dict):
        for key in ('keypoints', 'kp2d', 'vitpose', 'pred_keypoints'):
            if key in obj:
                arr = obj[key]
                break
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[1] != 17 or arr.shape[2] < 3:
        raise ValueError(f'关键点文件格式不支持: {path}')
    return arr[:, :, :3].astype(np.float32)


def load_local_bbox(path: Path):
    if not path.exists():
        return None
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and 'bbx_xys' in obj:
        return obj['bbx_xys'].clone().float().numpy()
    if isinstance(obj, torch.Tensor) and obj.ndim == 2 and obj.shape[1] == 3:
        return obj.clone().float().numpy()
    raise ValueError(f'本地 bbox 文件格式不支持: {path}')


def save_tensor_dict(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_tensor_dict(path: Path):
    if not path.exists():
        return None
    return torch.load(path, map_location='cpu')


def weighted_mean_conf(conf):
    conf = np.asarray(conf, dtype=np.float32)
    return float(np.average(np.clip(conf, 0.0, 1.0), weights=POSE_SCORE_WEIGHTS))


def bbox_xys_to_xyxy(bbx_xys):
    cx, cy, size = [float(v) for v in bbx_xys]
    half = max(size * 0.5, 1.0)
    return np.asarray([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)


def bbox_xyxy_to_xys(box_xyxy):
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    size = max(width, height)
    return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5, size], dtype=np.float32)


def box_size(box_xyxy):
    return max(float(box_xyxy[2] - box_xyxy[0]), float(box_xyxy[3] - box_xyxy[1]), 1.0)


def box_center(box_xyxy):
    return np.asarray([(box_xyxy[0] + box_xyxy[2]) * 0.5, (box_xyxy[1] + box_xyxy[3]) * 0.5], dtype=np.float32)


def box_center_distance_ratio(box_a, box_b):
    if box_a is None or box_b is None:
        return 1e9
    scale = max(box_size(box_a), box_size(box_b), 1.0)
    return float(np.linalg.norm(box_center(box_a) - box_center(box_b)) / scale)


def clamp_box_xyxy(box_xyxy, frame_shape):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = float(np.clip(x1, 0.0, width - 2.0))
    y1 = float(np.clip(y1, 0.0, height - 2.0))
    x2 = float(np.clip(x2, x1 + 2.0, width - 1.0))
    y2 = float(np.clip(y2, y1 + 2.0, height - 1.0))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def expand_box_xyxy(box_xyxy, scale, shift_y_ratio=0.0, frame_shape=None):
    center = box_center(box_xyxy)
    size = box_size(box_xyxy) * float(scale)
    center = center.astype(np.float32)
    center[1] += shift_y_ratio * size
    half = size * 0.5
    out = np.asarray([center[0] - half, center[1] - half, center[0] + half, center[1] + half], dtype=np.float32)
    if frame_shape is not None:
        out = clamp_box_xyxy(out, frame_shape)
    return out


def blend_boxes(primary, secondary, secondary_ratio=0.25):
    if primary is None:
        return None if secondary is None else np.asarray(secondary, dtype=np.float32)
    if secondary is None:
        return np.asarray(primary, dtype=np.float32)
    ratio = float(np.clip(secondary_ratio, 0.0, 1.0))
    out = np.asarray(primary, dtype=np.float32) * (1.0 - ratio) + np.asarray(secondary, dtype=np.float32) * ratio
    return out.astype(np.float32)


def box_size_ratio(box_a, box_b):
    if box_a is None or box_b is None:
        return 1.0
    return float(box_size(box_a) / max(box_size(box_b), 1.0))


def box_area(box_xyxy):
    if box_xyxy is None:
        return 1.0
    width = max(float(box_xyxy[2] - box_xyxy[0]), 1.0)
    height = max(float(box_xyxy[3] - box_xyxy[1]), 1.0)
    return float(width * height)


def box_dims(box_xyxy):
    if box_xyxy is None:
        return 1.0, 1.0
    return max(float(box_xyxy[2] - box_xyxy[0]), 1.0), max(float(box_xyxy[3] - box_xyxy[1]), 1.0)


def can_smooth_box_transition(cur_box, ref_box, cur_found, ref_found, crowded=False):
    if cur_box is None or ref_box is None:
        return False

    cur_size = box_size(cur_box)
    ref_size = box_size(ref_box)
    size_ratio = max(cur_size, ref_size) / max(min(cur_size, ref_size), 1.0)
    area_ratio = max(box_area(cur_box), box_area(ref_box)) / max(min(box_area(cur_box), box_area(ref_box)), 1.0)
    center_ratio = box_center_distance_ratio(cur_box, ref_box)
    iou = box_iou(cur_box, ref_box)

    if crowded:
        if size_ratio > 1.18 or area_ratio > 1.40:
            return False
        return bool(iou >= 0.42 or center_ratio <= 0.12)

    if (not cur_found) or (not ref_found):
        if size_ratio > 1.22 or area_ratio > 1.48:
            return False
        return bool(iou >= 0.34 or center_ratio <= 0.13)

    if size_ratio > 1.28 or area_ratio > 1.60:
        return False
    return bool(iou >= 0.32 or center_ratio <= 0.14)


def estimate_reference_box_size(prev_box, predicted_box, fallback_box):
    ref_sizes = [box_size(box) for box in (predicted_box, prev_box, fallback_box) if box is not None]
    if not ref_sizes:
        return 1.0
    return float(np.median(np.asarray(ref_sizes, dtype=np.float32)))


def is_relock_size_consistent(box_xyxy, prev_box, predicted_box, fallback_box, lost_lock_frames, occlusion_mode):
    if box_xyxy is None:
        return False
    ref_boxes = [box for box in (predicted_box, prev_box, fallback_box) if box is not None]
    ref_size = estimate_reference_box_size(prev_box, predicted_box, fallback_box)
    ref_area = float(np.median(np.asarray([box_area(box) for box in ref_boxes], dtype=np.float32))) if ref_boxes else 1.0
    if ref_size <= 1.0 or ref_area <= 1.0:
        return True

    size_ratio = float(box_size(box_xyxy) / max(ref_size, 1.0))
    area_ratio = float(box_area(box_xyxy) / max(ref_area, 1.0))
    if lost_lock_frames <= 4:
        side_low, side_high = ((0.70, 1.32) if occlusion_mode else (0.66, 1.38))
        area_low, area_high = ((0.62, 1.28) if occlusion_mode else (0.58, 1.34))
    elif lost_lock_frames <= 10:
        side_low, side_high = ((0.58, 1.50) if occlusion_mode else (0.54, 1.60))
        area_low, area_high = ((0.46, 1.50) if occlusion_mode else (0.42, 1.60))
    elif lost_lock_frames <= 18:
        side_low, side_high = ((0.46, 1.72) if occlusion_mode else (0.42, 1.84))
        area_low, area_high = ((0.32, 1.90) if occlusion_mode else (0.28, 2.00))
    else:
        side_low, side_high = (0.34, 2.10)
        area_low, area_high = (0.22, 2.35)
    return bool(side_low <= size_ratio <= side_high and area_low <= area_ratio <= area_high)


def is_relock_shape_consistent(box_xyxy, ref_box, lost_lock_frames, crowded=False):
    if box_xyxy is None or ref_box is None:
        return True
    cur_w, cur_h = box_dims(box_xyxy)
    ref_w, ref_h = box_dims(ref_box)
    w_ratio = float(cur_w / max(ref_w, 1.0))
    h_ratio = float(cur_h / max(ref_h, 1.0))
    if crowded:
        if lost_lock_frames <= 4:
            low, high = 0.72, 1.18
        elif lost_lock_frames <= 10:
            low, high = 0.60, 1.28
        else:
            low, high = 0.48, 1.42
    else:
        if lost_lock_frames <= 4:
            low, high = 0.66, 1.28
        elif lost_lock_frames <= 10:
            low, high = 0.54, 1.38
        else:
            low, high = 0.42, 1.56
    return bool(low <= w_ratio <= high and low <= h_ratio <= high)


def is_locked_track_anchor_consistent(
    box_xyxy,
    prev_box,
    predicted_box,
    fallback_box,
    lost_lock_frames,
    crowded=False,
    motion_score=0.0,
):
    if box_xyxy is None:
        return False
    if prev_box is None and predicted_box is None and fallback_box is None:
        return True

    affinity = compute_target_affinity(box_xyxy, prev_box, predicted_box, fallback_box)
    anchor_score = float(affinity['anchor_score'])
    anchor_center = float(affinity['anchor_center'])
    motion_score = float(np.clip(motion_score, 0.0, 1.0))

    if crowded:
        if lost_lock_frames <= 2:
            score_floor, center_limit, motion_floor = 0.46, 0.58, 0.06
        elif lost_lock_frames <= 6:
            score_floor, center_limit, motion_floor = 0.40, 0.66, 0.05
        else:
            score_floor, center_limit, motion_floor = 0.32, 0.76, 0.04
    else:
        if lost_lock_frames <= 0:
            score_floor, center_limit, motion_floor = 0.24, 0.76, 0.02
        elif lost_lock_frames <= 3:
            score_floor, center_limit, motion_floor = 0.30, 0.72, 0.03
        elif lost_lock_frames <= 8:
            score_floor, center_limit, motion_floor = 0.26, 0.82, 0.02
        else:
            score_floor, center_limit, motion_floor = 0.20, 0.94, 0.0

    return bool(anchor_score >= score_floor or (anchor_center <= center_limit and motion_score >= motion_floor))


def constrain_hold_box(hold_box, anchor_box, frame_shape, lost_lock_frames, occlusion_mode):
    if hold_box is None:
        return None
    if anchor_box is None:
        return clamp_box_xyxy(hold_box, frame_shape)

    anchor_size = box_size(anchor_box)
    hold_size = box_size(hold_box)
    if lost_lock_frames <= 4:
        max_size_ratio = 1.16 if occlusion_mode else 1.12
        max_shift_ratio = 0.34 if occlusion_mode else 0.28
    elif lost_lock_frames <= 10:
        max_size_ratio = 1.26 if occlusion_mode else 1.20
        max_shift_ratio = 0.48 if occlusion_mode else 0.40
    else:
        max_size_ratio = 1.36 if occlusion_mode else 1.28
        max_shift_ratio = 0.66 if occlusion_mode else 0.56

    size = min(hold_size, anchor_size * max_size_ratio)
    anchor_center = box_center(anchor_box)
    hold_center = box_center(hold_box)
    delta = hold_center - anchor_center
    max_shift = anchor_size * max_shift_ratio
    dist = float(np.linalg.norm(delta))
    if dist > max_shift and dist > 1e-6:
        delta = delta * (max_shift / dist)
    center = anchor_center + delta
    half = size * 0.5
    out = np.asarray([center[0] - half, center[1] - half, center[0] + half, center[1] + half], dtype=np.float32)
    return clamp_box_xyxy(out, frame_shape)

def box_iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(1.0, float(box_a[2] - box_a[0]) * float(box_a[3] - box_a[1]))
    area_b = max(1.0, float(box_b[2] - box_b[0]) * float(box_b[3] - box_b[1]))
    return float(inter / max(area_a + area_b - inter, 1.0))


def extract_target_appearance(frame: np.ndarray, box_xyxy: np.ndarray):
    if frame is None or box_xyxy is None or frame.size == 0:
        return None
    box_xyxy = clamp_box_xyxy(box_xyxy, frame.shape[:2])
    x1, y1, x2, y2 = np.round(box_xyxy).astype(int)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    height, width = crop.shape[:2]
    inner_x1 = max(int(width * 0.18), 0)
    inner_x2 = min(int(width * 0.82), width)
    inner_y1 = max(int(height * 0.10), 0)
    inner_y2 = min(int(height * 0.64), height)
    torso = crop[inner_y1:inner_y2, inner_x1:inner_x2]
    if torso.size == 0:
        torso = crop

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (((sat >= 28) | (val <= 218)) & (val >= 12)).astype(np.uint8) * 255
    if int(np.count_nonzero(mask)) < max(48, int(mask.size * 0.04)):
        mask = ((val <= 245) & (val >= 12)).astype(np.uint8) * 255
    if int(np.count_nonzero(mask)) < 24:
        return None

    hist_hs = cv2.calcHist([hsv], [0, 1], mask, [18, 12], [0, 180, 0, 256]).astype(np.float32).reshape(-1)
    hist_v = cv2.calcHist([hsv], [2], mask, [8], [0, 256]).astype(np.float32).reshape(-1)
    hist = np.concatenate([hist_hs, hist_v], axis=0)
    hist_sum = float(hist.sum())
    if hist_sum <= 1e-6:
        return None
    hist /= hist_sum
    mean_bgr = np.asarray(cv2.mean(torso, mask=mask)[:3], dtype=np.float32) / 255.0
    return hist.astype(np.float32), mean_bgr.astype(np.float32)


def blend_appearance_feature(base_feature, new_feature, ratio):
    if base_feature is None:
        return new_feature
    if new_feature is None:
        return base_feature

    alpha = float(np.clip(ratio, 0.0, 1.0))
    base_hist, base_mean = base_feature
    new_hist, new_mean = new_feature
    hist = base_hist * (1.0 - alpha) + new_hist * alpha
    hist_sum = float(hist.sum())
    if hist_sum > 1e-6:
        hist = hist / hist_sum
    mean = base_mean * (1.0 - alpha) + new_mean * alpha
    return hist.astype(np.float32), mean.astype(np.float32)


def appearance_similarity(feature_a, feature_b):
    if feature_a is None or feature_b is None:
        return 0.50

    hist_a, mean_a = feature_a
    hist_b, mean_b = feature_b
    hist_score = float(np.sum(np.sqrt(np.clip(hist_a, 0.0, 1.0) * np.clip(hist_b, 0.0, 1.0))))
    hist_score = float(np.clip(hist_score, 0.0, 1.0))
    mean_dist = float(np.linalg.norm(mean_a - mean_b))
    mean_score = float(np.clip(1.0 - mean_dist / 0.85, 0.0, 1.0))
    return float(0.84 * hist_score + 0.16 * mean_score)


def compute_appearance_score(frame, box_xyxy, appearance_anchor, appearance_recent):
    feature = extract_target_appearance(frame, box_xyxy)
    if feature is None:
        return 0.50, None

    weighted_scores = []
    if appearance_anchor is not None:
        weighted_scores.append((appearance_similarity(feature, appearance_anchor), 0.68))
    if appearance_recent is not None:
        weighted_scores.append((appearance_similarity(feature, appearance_recent), 0.32 if appearance_anchor is not None else 1.0))

    if not weighted_scores:
        return 0.50, feature

    total_weight = sum(weight for _, weight in weighted_scores)
    score = sum(score * weight for score, weight in weighted_scores) / max(total_weight, 1e-6)
    return float(np.clip(score, 0.0, 1.0)), feature


def compute_box_motion_score(prev_gray: np.ndarray | None, gray: np.ndarray | None, box_xyxy: np.ndarray | None):
    if prev_gray is None or gray is None or box_xyxy is None:
        return 0.0
    x1, y1, x2, y2 = np.round(box_xyxy).astype(int)
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, gray.shape[1])
    y2 = min(y2, gray.shape[0])
    if x2 - x1 < 12 or y2 - y1 < 12:
        return 0.0
    prev_roi = prev_gray[y1:y2, x1:x2]
    cur_roi = gray[y1:y2, x1:x2]
    if prev_roi.size == 0 or cur_roi.size == 0:
        return 0.0
    prev_small = cv2.resize(prev_roi, (32, 32), interpolation=cv2.INTER_AREA)
    cur_small = cv2.resize(cur_roi, (32, 32), interpolation=cv2.INTER_AREA)
    diff = float(np.mean(np.abs(cur_small.astype(np.float32) - prev_small.astype(np.float32))) / 255.0)
    return float(np.clip((diff - 0.025) / 0.18, 0.0, 1.0))


def estimate_box_from_keypoints(keypoints_xy, keypoints_conf, conf_thresh=0.20):
    valid = np.asarray(keypoints_conf, dtype=np.float32) >= conf_thresh
    if int(valid.sum()) < 4:
        return None
    x_valid = keypoints_xy[valid, 0]
    y_valid = keypoints_xy[valid, 1]
    return np.asarray([
        float(np.min(x_valid)),
        float(np.min(y_valid)),
        float(np.max(x_valid)),
        float(np.max(y_valid)),
    ], dtype=np.float32)


def estimate_seed_box(seed_kp):
    if seed_kp is None:
        return None
    box = estimate_box_from_keypoints(seed_kp[:, :2], seed_kp[:, 2], conf_thresh=0.18)
    if box is None:
        return None
    return box.astype(np.float32)


def choose_pose_device():
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_detector(model_path: Path):
    if YOLO is None:
        raise RuntimeError('ultralytics 未安装或导入失败')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    return YOLO(str(model_path)).to(device)


def load_sr_model(model_path: Path):
    # 性能优化：强制禁用超分以节省算力
    return None
    rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    return RealESRGANer(
        scale=2,
        model_path=str(model_path),
        model=rrdb,
        tile=0,
        tile_pad=8,
        pre_pad=0,
        half=False,
        device='cpu',
    )


def load_pose_model(model_ref: str):
    if RTMPose is None:
        raise RuntimeError('rtmlib / onnxruntime 未安装或导入失败')
    return RTMPose(model_ref, model_input_size=POSE_INPUT_SIZE, backend='onnxruntime', device=choose_pose_device())


def estimate_backlight_score(image: np.ndarray):
    if image is None or image.size == 0:
        return 0.0
    if image.ndim == 2:
        luma = image.astype(np.float32)
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        luma = lab[:, :, 0].astype(np.float32)

    height, width = luma.shape[:2]
    if height < 8 or width < 8:
        return 0.0

    inner_y1 = max(int(height * 0.18), 1)
    inner_y2 = min(int(height * 0.86), height - 1)
    inner_x1 = max(int(width * 0.18), 1)
    inner_x2 = min(int(width * 0.82), width - 1)
    center = luma[inner_y1:inner_y2, inner_x1:inner_x2]
    if center.size == 0:
        center = luma

    border_mask = np.ones_like(luma, dtype=bool)
    border_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False
    border = luma[border_mask]
    if border.size == 0:
        border = luma.reshape(-1)

    mean_luma = float(np.mean(luma))
    std_luma = float(np.std(luma))
    center_mean = float(np.mean(center))
    border_mean = float(np.mean(border))
    silhouette_score = float(np.clip((border_mean - center_mean - 18.0) / 85.0, 0.0, 1.0))
    border_bright = float(np.clip((border_mean - 150.0) / 72.0, 0.0, 1.0))
    center_dark = float(np.clip((118.0 - center_mean) / 70.0, 0.0, 1.0))
    flat_dark = float(np.clip((112.0 - mean_luma) / 84.0, 0.0, 1.0)) * float(np.clip((56.0 - std_luma) / 42.0, 0.0, 1.0))
    low_texture = float(np.clip((48.0 - std_luma) / 40.0, 0.0, 1.0)) * float(np.clip((145.0 - mean_luma) / 96.0, 0.0, 1.0))

    return float(np.clip(max(0.58 * silhouette_score + 0.26 * border_bright + 0.16 * center_dark, 0.62 * flat_dark + 0.38 * low_texture), 0.0, 1.0))


def enhance_backlit_image(image: np.ndarray, strong: bool = False):
    if image is None or image.size == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    tile_size = (6, 6) if min(image.shape[:2]) < 96 else (8, 8)
    clahe = cv2.createCLAHE(clipLimit=3.8 if strong else 2.8, tileGridSize=tile_size)
    l_channel = clahe.apply(l_channel)

    mean_luma = float(np.mean(l_channel))
    if mean_luma < 92.0:
        gamma = 0.72 if strong else 0.80
    elif mean_luma < 118.0:
        gamma = 0.80 if strong else 0.87
    else:
        gamma = 0.90 if strong else 0.95
    lut = np.asarray(((np.linspace(0.0, 1.0, 256) ** gamma) * 255.0).clip(0, 255), dtype=np.uint8)
    l_channel = cv2.LUT(l_channel, lut)

    merged = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(merged, (0, 0), 1.2 if strong else 0.9)
    sharpened = cv2.addWeighted(merged, 1.14 if strong else 1.10, blur, -0.14 if strong else -0.10, 0.0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def run_detector_predict(detector, image: np.ndarray, imgsz: int, conf: float):
    try:
        results = detector.predict(image, imgsz=imgsz, conf=conf, verbose=False, classes=[0])
    except Exception:
        results = detector(image, imgsz=imgsz, conf=conf, verbose=False, classes=[0])
    boxes = results[0].boxes if results else None
    boxes_xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32) if boxes is not None and boxes.xyxy is not None else np.zeros((0, 4), dtype=np.float32)
    scores = boxes.conf.detach().cpu().numpy().astype(np.float32) if boxes is not None and boxes.conf is not None else np.zeros((0,), dtype=np.float32)
    return boxes_xyxy, scores


def detect_rescue_candidates(detector, frame, search_anchor_box, predicted_box, fallback_box, imgsz, conf, lost_lock_frames):
    ref_box = search_anchor_box if search_anchor_box is not None else (predicted_box if predicted_box is not None else fallback_box)
    if ref_box is None:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0.0

    roi_scale = BACKLIGHT_RESCUE_ROI_SCALE
    ref_size = box_size(ref_box)
    if ref_size < 120.0:
        roi_scale = max(roi_scale, 2.55)
    if ref_size < 85.0:
        roi_scale = max(roi_scale, 2.85)
    if lost_lock_frames >= 4:
        roi_scale += 0.18

    roi_box = expand_box_xyxy(ref_box, roi_scale, shift_y_ratio=-0.04 if ref_size < 120.0 else -0.02, frame_shape=frame.shape[:2])
    roi_box = clamp_box_xyxy(roi_box, frame.shape[:2])
    x1, y1, x2, y2 = np.round(roi_box).astype(int)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0.0

    roi = frame[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0.0

    backlight_score = estimate_backlight_score(roi)
    if lost_lock_frames <= 0 and backlight_score < BACKLIGHT_TRIGGER_SCORE:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), backlight_score

    enhanced_roi = enhance_backlit_image(roi, strong=bool(backlight_score >= STRONG_BACKLIGHT_SCORE or lost_lock_frames >= 2))
    rescue_conf = max(conf * BACKLIGHT_RESCUE_CONF_SCALE, 0.08)
    rescue_imgsz = max(960, min(int(imgsz), 1280))
    rescue_boxes, rescue_scores = run_detector_predict(detector, enhanced_roi, imgsz=rescue_imgsz, conf=rescue_conf)
    if len(rescue_boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), backlight_score

    mapped_boxes = []
    mapped_scores = []
    for idx, box in enumerate(rescue_boxes):
        mapped_box = np.asarray([box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1], dtype=np.float32)
        mapped_box = clamp_box_xyxy(mapped_box, frame.shape[:2])
        affinity = compute_target_affinity(mapped_box, search_anchor_box, predicted_box, fallback_box)
        if affinity['anchor_score'] >= 0.18 or affinity['anchor_center'] <= 1.18 or box_iou(mapped_box, roi_box) >= 0.10:
            mapped_boxes.append(mapped_box)
            mapped_scores.append(float(rescue_scores[idx]) if len(rescue_scores) > idx else 0.0)

    if not mapped_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), backlight_score

    return np.asarray(mapped_boxes, dtype=np.float32), np.asarray(mapped_scores, dtype=np.float32), backlight_score


def smooth_boxes(boxes_xyxy: np.ndarray, found_mask: np.ndarray, crowd_mask: np.ndarray | None = None):
    out = boxes_xyxy.astype(np.float32).copy()
    if len(out) == 0:
        return out

    crowd = np.zeros((len(out),), dtype=bool) if crowd_mask is None else np.asarray(crowd_mask, dtype=bool)
    prev = out[0].copy()
    prev_found = bool(found_mask[0]) if len(found_mask) > 0 else True
    prev_crowded = bool(crowd[0]) if len(crowd) > 0 else False
    for idx in range(1, len(out)):
        cur_found = bool(found_mask[idx]) if idx < len(found_mask) else True
        cur_crowded = bool(crowd[idx]) if idx < len(crowd) else False
        if cur_crowded or prev_crowded or (not cur_found) or (not prev_found):
            prev = out[idx].copy()
            prev_found = cur_found
            prev_crowded = cur_crowded
            continue
        if not can_smooth_box_transition(out[idx], prev, cur_found, prev_found, crowded=False):
            prev = out[idx].copy()
            prev_found = cur_found
            prev_crowded = cur_crowded
            continue
        alpha = 0.84
        out[idx] = alpha * out[idx] + (1.0 - alpha) * prev
        prev = out[idx].copy()
        prev_found = cur_found
        prev_crowded = cur_crowded

    return out


def predict_next_box(prev_box, prev_prev_box, frame_shape=None, steps=1):
    if prev_box is None:
        return None
    out = np.asarray(prev_box, dtype=np.float32).copy()
    if prev_prev_box is not None:
        prev_center = box_center(prev_prev_box)
        cur_center = box_center(prev_box)
        velocity = cur_center - prev_center
        max_shift = max(box_size(prev_box) * 0.38, 10.0)
        velocity = np.clip(velocity, -max_shift, max_shift)
        step_scale = float(max(1, int(steps)))
        velocity = velocity * step_scale
        out += np.asarray([velocity[0], velocity[1], velocity[0], velocity[1]], dtype=np.float32)
    if frame_shape is not None:
        out = clamp_box_xyxy(out, frame_shape)
    return out.astype(np.float32)


def compute_target_affinity(box_xyxy, prev_box, predicted_box, fallback_box):
    prev_iou = box_iou(box_xyxy, prev_box) if prev_box is not None else 0.0
    pred_iou = box_iou(box_xyxy, predicted_box) if predicted_box is not None else 0.0
    fallback_iou = box_iou(box_xyxy, fallback_box) if fallback_box is not None else 0.0

    prev_center = box_center_distance_ratio(box_xyxy, prev_box) if prev_box is not None else 1e9
    pred_center = box_center_distance_ratio(box_xyxy, predicted_box) if predicted_box is not None else 1e9
    fallback_center = box_center_distance_ratio(box_xyxy, fallback_box) if fallback_box is not None else 1e9

    anchor_iou = max(prev_iou, pred_iou, fallback_iou)
    anchor_center = min(prev_center, pred_center, fallback_center)
    anchor_score = float(np.clip(max(anchor_iou, 1.0 - anchor_center / 0.82), 0.0, 1.0))
    return {
        'prev_iou': float(prev_iou),
        'pred_iou': float(pred_iou),
        'fallback_iou': float(fallback_iou),
        'prev_center': float(prev_center),
        'pred_center': float(pred_center),
        'fallback_center': float(fallback_center),
        'anchor_iou': float(anchor_iou),
        'anchor_center': float(anchor_center),
        'anchor_score': anchor_score,
    }


def is_consistent_single_target(box_xyxy, prev_box, predicted_box, fallback_box, lost_lock_frames):
    if prev_box is None and predicted_box is None and fallback_box is None:
        return True

    affinity = compute_target_affinity(box_xyxy, prev_box, predicted_box, fallback_box)
    anchor_iou = affinity['anchor_iou']
    anchor_center = affinity['anchor_center']

    if lost_lock_frames <= 1:
        return bool(anchor_iou >= 0.025 or anchor_center <= 0.55)
    if lost_lock_frames <= 3:
        return bool(anchor_iou >= 0.015 or anchor_center <= 0.65)
    if lost_lock_frames <= 6:
        return bool(anchor_iou >= 0.010 or anchor_center <= 0.78)
    return bool(anchor_iou >= 0.006 or anchor_center <= 0.92)


def repair_found_mask(found_mask: np.ndarray, boxes_xyxy: np.ndarray, max_gap: int = 9):
    out = found_mask.astype(bool).copy()
    if len(out) == 0:
        return out

    idx = 0
    while idx < len(out):
        if out[idx]:
            idx += 1
            continue
        gap_start = idx
        while idx < len(out) and not out[idx]:
            idx += 1
        gap_end = idx
        gap_len = gap_end - gap_start
        if gap_start == 0 or gap_end >= len(out) or gap_len > max_gap:
            continue

        left_box = boxes_xyxy[gap_start - 1]
        right_box = boxes_xyxy[gap_end]
        left_size = box_size(left_box)
        right_size = box_size(right_box)
        size_ratio = max(left_size, right_size) / max(min(left_size, right_size), 1.0)
        center_dist = float(np.linalg.norm(box_center(left_box) - box_center(right_box)) / max(left_size, right_size, 1.0))
        if center_dist > 1.35 or size_ratio > 1.50:
            continue

        left_center = box_center(left_box)
        right_center = box_center(right_box)
        ok = True
        for inner_idx in range(gap_start, gap_end):
            blend = float((inner_idx - gap_start + 1) / (gap_len + 1))
            interp_center = left_center * (1.0 - blend) + right_center * blend
            cur_center = box_center(boxes_xyxy[inner_idx])
            cur_size = box_size(boxes_xyxy[inner_idx])
            if float(np.linalg.norm(cur_center - interp_center) / max(cur_size, left_size, right_size, 1.0)) > 0.55:
                ok = False
                break
        if ok:
            out[gap_start:gap_end] = True

    return out


def detect_target_crowding(boxes_xyxy, scores, prev_box, predicted_box, fallback_box):
    candidate_infos = []
    for idx, box in enumerate(boxes_xyxy):
        affinity = compute_target_affinity(box, prev_box, predicted_box, fallback_box)
        anchor_score = float(affinity['anchor_score'])
        anchor_center = float(affinity['anchor_center'])
        anchor_iou = float(affinity['anchor_iou'])
        det_score = float(scores[idx]) if len(scores) > idx else 0.0
        if anchor_score >= 0.20 or anchor_iou >= 0.02 or anchor_center <= 1.05:
            candidate_infos.append({
                'idx': idx,
                'box': np.asarray(box, dtype=np.float32),
                'det_score': det_score,
                'anchor_score': anchor_score,
                'anchor_center': anchor_center,
            })

    candidate_infos.sort(key=lambda item: (-item['anchor_score'], item['anchor_center'], -item['det_score']))
    crowded_now = False
    crowd_margin = 1.0
    if len(candidate_infos) >= 2:
        top = candidate_infos[0]
        second = candidate_infos[1]
        crowd_margin = float(top['anchor_score'] - second['anchor_score'])
        pair_iou = box_iou(top['box'], second['box'])
        pair_center = box_center_distance_ratio(top['box'], second['box'])
        crowded_now = bool(
            (top['anchor_score'] >= 0.58 and second['anchor_score'] >= 0.40)
            and (pair_iou >= 0.05 or pair_center <= 1.15 or crowd_margin <= 0.16)
        )
    return crowded_now, crowd_margin, candidate_infos


def select_primary_detection(
    frame,
    boxes_xyxy,
    scores,
    track_ids,
    prev_box,
    predicted_box,
    locked_track_id,
    fallback_box,
    lost_lock_frames,
    appearance_anchor=None,
    appearance_recent=None,
    prev_gray=None,
    cur_gray=None,
):
    if len(boxes_xyxy) == 0:
        return None, None, -1e9, 0.0, 0.50

    height, width = frame.shape[:2]
    ref_center = np.asarray([width * 0.50, height * 0.58], dtype=np.float32)
    frame_diag = float(math.hypot(width, height))
    use_appearance = appearance_anchor is not None or appearance_recent is not None

    best_idx = None
    best_score = -1e9
    best_appearance_score = 0.50
    for idx, box in enumerate(boxes_xyxy):
        score = float(scores[idx]) if len(scores) > idx else 0.0
        track_id = None if track_ids is None or int(track_ids[idx]) < 0 else int(track_ids[idx])
        center = box_center(box)
        dist = float(np.linalg.norm(center - ref_center) / max(frame_diag, 1.0))
        center_score = float(np.clip(1.0 - dist / 0.33, 0.0, 1.0))
        area_ratio = float(((box[2] - box[0]) * (box[3] - box[1])) / max(width * height, 1.0))
        area_score = float(np.clip((area_ratio - 0.003) / 0.040, 0.0, 1.0))
        temporal_iou = box_iou(box, prev_box) if prev_box is not None else 0.45
        temporal_center = box_center_distance_ratio(box, prev_box) if prev_box is not None else 0.0
        temporal_score = float(np.clip(max(temporal_iou, 1.0 - temporal_center / 0.75), 0.0, 1.0))
        fallback_score = box_iou(box, fallback_box) if fallback_box is not None else 0.0
        appearance_score = 0.50
        if use_appearance:
            appearance_score, _ = compute_appearance_score(frame, box, appearance_anchor, appearance_recent)
        motion_score = compute_box_motion_score(prev_gray, cur_gray, box)

        if locked_track_id is None:
            total = (
                0.22 * center_score
                + 0.10 * area_score
                + 0.16 * float(np.clip(score, 0.0, 1.0))
                + 0.28 * temporal_score
                + 0.18 * fallback_score
                + 0.04 * appearance_score
                + 0.02 * motion_score
            )
        else:
            single_clear_scene = len(boxes_xyxy) == 1
            if not is_consistent_single_target(box, prev_box, predicted_box, fallback_box, lost_lock_frames):
                warmup_retarget = False
                if single_clear_scene and lost_lock_frames <= 3:
                    prev_size_ratio = box_size_ratio(box, prev_box) if prev_box is not None else 1.0
                    warmup_retarget = bool(
                        float(np.clip(score, 0.0, 1.0)) >= 0.16
                        and (area_score >= 0.22 or center_score >= 0.18 or prev_size_ratio >= 1.30)
                        and (prev_box is None or prev_size_ratio >= 1.18 or center_score >= 0.20)
                    )
                if not warmup_retarget:
                    continue
            affinity = compute_target_affinity(box, prev_box, predicted_box, fallback_box)
            anchor_score = affinity['anchor_score']
            single_clear_scene = len(boxes_xyxy) == 1
            if prev_gray is not None and lost_lock_frames <= 6 and motion_score < 0.06:
                low_motion_anchor_floor = 0.60 if single_clear_scene else 0.88
                if anchor_score < low_motion_anchor_floor:
                    continue
            ref_size = estimate_reference_box_size(prev_box, predicted_box, fallback_box)
            size_ratio = float(box_size(box) / max(ref_size, 1.0))
            size_score = float(np.clip(1.0 - abs(math.log(max(size_ratio, 1e-3))) / math.log(1.9), 0.0, 1.0))
            id_bonus = 0.0
            if track_id is not None and track_id == locked_track_id:
                same_track_ok = is_locked_track_anchor_consistent(
                    box,
                    prev_box,
                    predicted_box,
                    fallback_box,
                    lost_lock_frames,
                    crowded=bool(lost_lock_frames > 0),
                    motion_score=motion_score,
                )
                if same_track_ok:
                    id_bonus = 0.68
                elif anchor_score >= 0.22:
                    id_bonus = 0.08
                else:
                    id_bonus = -0.16
            total = (
                0.48 * anchor_score
                + 0.12 * float(np.clip(score, 0.0, 1.0))
                + 0.10 * size_score
                + 0.10 * appearance_score
                + 0.10 * fallback_score
                + 0.08 * motion_score
                + 0.02 * center_score
                + id_bonus
            )

        if total > best_score:
            best_score = total
            best_idx = idx
            best_appearance_score = appearance_score

    if best_idx is None:
        return None, None, -1e9, 0.0, 0.50

    best_box = np.asarray(boxes_xyxy[best_idx], dtype=np.float32)
    best_track_id = None if track_ids is None or int(track_ids[best_idx]) < 0 else int(track_ids[best_idx])
    best_det_score = float(scores[best_idx]) if len(scores) > best_idx else 0.0
    return best_box, best_track_id, float(best_score), best_det_score, float(best_appearance_score)


def track_target_bboxes(
    input_path: Path,
    detector_model_path: Path,
    tracker_config_path: Path,
    cache_path: Path,
    local_bbx_xys: np.ndarray | None,
    imgsz: int,
    conf: float,
    max_frames: int,
    force: bool,
):
    if cache_path.exists() and not force:
        cache = load_tensor_dict(cache_path)
        if isinstance(cache, dict) and 'boxes_xyxy' in cache:
            print(f'[Info] Reuse ByteTrack cache: {cache_path}')
            boxes_xyxy = np.asarray(cache['boxes_xyxy'], dtype=np.float32)
            found_mask = np.asarray(cache.get('found_mask', np.ones((len(boxes_xyxy),), dtype=np.bool_)), dtype=bool)
            crowd_mask = np.asarray(cache.get('crowd_mask', np.zeros((len(boxes_xyxy),), dtype=np.bool_)), dtype=bool)
            return boxes_xyxy, found_mask, crowd_mask

    detector = load_detector(detector_model_path)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频：{input_path}')

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = total_frames if total_frames > 0 else (len(local_bbx_xys) if local_bbx_xys is not None else 0)
    if local_bbx_xys is not None:
        frame_limit = min(frame_limit, len(local_bbx_xys))
    if max_frames and max_frames > 0:
        frame_limit = min(frame_limit, int(max_frames))

    tracked_boxes = []
    found_mask = []
    crowd_mask = []
    prev_box = None
    prev_prev_box = None
    confirmed_box = None
    confirmed_prev_box = None
    locked_track_id = None
    relock_candidate_id = None
    relock_candidate_hits = 0
    lost_lock_frames = 0
    crowd_hold_frames = 0
    appearance_anchor = None
    appearance_recent = None
    prev_gray = None

    with tqdm(total=frame_limit, desc='ByteTrack', unit='frame') as pbar:
        frame_idx = 0
        # 优化：YOLO 降频缓存
        yolo_interval = 3
        cached_boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        cached_scores = np.zeros((0,), dtype=np.float32)
        cached_track_ids = None

        while frame_idx < frame_limit:
            ok, frame = cap.read()
            if not ok:
                break

            # 性能优化：降分辨率（长边压缩至1080px）
            h, w = frame.shape[:2]
            if max(h, w) > 1080:
                scale = 1080.0 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            fallback_box = None
            if local_bbx_xys is not None and frame_idx < len(local_bbx_xys):
                fallback_box = bbox_xys_to_xyxy(local_bbx_xys[frame_idx])
                fallback_box = clamp_box_xyxy(fallback_box, frame.shape[:2])

            anchor_box = confirmed_box if confirmed_box is not None and lost_lock_frames > 0 else prev_box
            anchor_prev_box = confirmed_prev_box if confirmed_prev_box is not None and lost_lock_frames > 0 else prev_prev_box
            prediction_steps = 1 if lost_lock_frames <= 0 else min(lost_lock_frames + 1, 6)
            predicted_box = predict_next_box(anchor_box, anchor_prev_box, frame.shape[:2], steps=prediction_steps)
            search_anchor_box = predicted_box if lost_lock_frames > 0 and predicted_box is not None else anchor_box

            search_backlight_score = 0.0

            # 优化：每3帧才运行一次YOLO检测，中间帧复用缓存
            if frame_idx % yolo_interval == 0:
                results = detector.track(
                    frame,
                    imgsz=imgsz,
                    conf=conf,
                    verbose=False,
                    persist=True,
                    classes=[0],
                    tracker=str(tracker_config_path),
                )
                boxes = results[0].boxes
                cached_boxes_xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32) if boxes is not None and boxes.xyxy is not None else np.zeros((0, 4), dtype=np.float32)
                cached_scores = boxes.conf.detach().cpu().numpy().astype(np.float32) if boxes is not None and boxes.conf is not None else np.zeros((0,), dtype=np.float32)
                cached_track_ids = None
                if boxes is not None and boxes.id is not None:
                    cached_track_ids = boxes.id.detach().cpu().numpy().astype(np.int32)

            # 使用缓存的检测结果
            boxes_xyxy = cached_boxes_xyxy
            scores = cached_scores
            track_ids = cached_track_ids

            rescue_boxes = np.zeros((0, 4), dtype=np.float32)
            rescue_scores = np.zeros((0,), dtype=np.float32)
            should_try_rescue = (
                locked_track_id is None
                or lost_lock_frames > 0
            )
            if should_try_rescue:
                rescue_boxes, rescue_scores, search_backlight_score = detect_rescue_candidates(
                    detector,
                    frame,
                    search_anchor_box,
                    predicted_box,
                    fallback_box,
                    imgsz,
                    conf,
                    lost_lock_frames,
                )
            if len(rescue_boxes) > 0:
                if track_ids is None:
                    track_ids = np.full((len(boxes_xyxy),), -1, dtype=np.int32)
                rescue_ids = np.full((len(rescue_boxes),), -1, dtype=np.int32)
                boxes_xyxy = np.concatenate([boxes_xyxy, rescue_boxes], axis=0) if len(boxes_xyxy) > 0 else rescue_boxes.astype(np.float32)
                scores = np.concatenate([scores, rescue_scores], axis=0) if len(scores) > 0 else rescue_scores.astype(np.float32)
                track_ids = np.concatenate([track_ids, rescue_ids], axis=0)

            crowded_now, crowd_margin, _ = detect_target_crowding(boxes_xyxy, scores, search_anchor_box, predicted_box, fallback_box)
            if crowded_now:
                crowd_hold_frames = min(crowd_hold_frames + 1, 10)
            else:
                crowd_hold_frames = max(crowd_hold_frames - 1, 0)
            occlusion_mode = locked_track_id is not None and crowd_hold_frames > 0
            appearance_refs_exist = appearance_anchor is not None or appearance_recent is not None
            appearance_relax = 0.10 if search_backlight_score >= STRONG_BACKLIGHT_SCORE else (0.06 if search_backlight_score >= BACKLIGHT_TRIGGER_SCORE else 0.0)

            selected_box = None
            selected_track_id = None
            selected_score = -1e9
            selected_appearance_score = 0.50
            selected_from_detection = False
            found = False
            clear_locked_track = False

            if locked_track_id is not None and track_ids is not None and len(track_ids) > 0:
                locked_indices = np.where(track_ids == locked_track_id)[0]
                if len(locked_indices) > 0:
                    best_locked_idx = int(locked_indices[np.argmax(scores[locked_indices])])
                    locked_box = np.asarray(boxes_xyxy[best_locked_idx], dtype=np.float32)
                    locked_score = float(scores[best_locked_idx]) if len(scores) > best_locked_idx else 0.0
                    locked_motion_score = compute_box_motion_score(prev_gray, gray_frame, locked_box)
                    locked_affinity = compute_target_affinity(locked_box, search_anchor_box, predicted_box, fallback_box)
                    locked_consistent = is_consistent_single_target(
                        locked_box,
                        search_anchor_box,
                        predicted_box,
                        fallback_box,
                        lost_lock_frames,
                    )
                    locked_appearance_score, _ = compute_appearance_score(frame, locked_box, appearance_anchor, appearance_recent)
                    locked_appearance_threshold = max(0.18, (0.44 if occlusion_mode else 0.36) - appearance_relax)
                    locked_appearance_ok = (not appearance_refs_exist) or locked_appearance_score >= locked_appearance_threshold
                    locked_size_ok = True
                    locked_shape_ok = True
                    locked_anchor_ok = is_locked_track_anchor_consistent(
                        locked_box,
                        search_anchor_box,
                        predicted_box,
                        fallback_box,
                        lost_lock_frames,
                        crowded=bool(crowded_now or occlusion_mode),
                        motion_score=locked_motion_score,
                    )
                    if search_anchor_box is not None and (crowded_now or occlusion_mode or lost_lock_frames > 0):
                        locked_size_ok = is_relock_size_consistent(
                            locked_box,
                            search_anchor_box,
                            predicted_box,
                            fallback_box,
                            max(lost_lock_frames, 1),
                            True if (crowded_now or occlusion_mode) else occlusion_mode,
                        )
                        if crowded_now or occlusion_mode:
                            locked_shape_ok = is_relock_shape_consistent(locked_box, search_anchor_box, max(lost_lock_frames, 1), crowded=True)
                    if locked_consistent and locked_appearance_ok and locked_size_ok and locked_shape_ok and locked_anchor_ok and locked_score >= 0.12:
                        selected_box = locked_box
                        selected_track_id = int(locked_track_id)
                        selected_score = max(locked_score, locked_affinity['anchor_score'])
                        selected_appearance_score = locked_appearance_score
                        selected_from_detection = True
                        found = True
                        lost_lock_frames = 0
                        relock_candidate_id = None
                        relock_candidate_hits = 0

            candidate_box = None
            candidate_track_id = None
            candidate_score = -1e9
            candidate_det_score = 0.0
            candidate_appearance_score = 0.50
            if selected_box is None:
                search_lost_frames = lost_lock_frames if locked_track_id is None else (lost_lock_frames + 1)
                candidate_box, candidate_track_id, candidate_score, candidate_det_score, candidate_appearance_score = select_primary_detection(
                    frame,
                    boxes_xyxy,
                    scores,
                    track_ids,
                    search_anchor_box,
                    predicted_box,
                    locked_track_id,
                    fallback_box,
                    search_lost_frames,
                    appearance_anchor=appearance_anchor,
                    appearance_recent=appearance_recent,
                    prev_gray=prev_gray,
                    cur_gray=gray_frame,
                )

                if locked_track_id is None:
                    selected_box = candidate_box
                    selected_track_id = candidate_track_id
                    selected_score = candidate_score
                    selected_appearance_score = candidate_appearance_score
                    selected_from_detection = selected_box is not None
                    found = selected_box is not None
                    lost_lock_frames = 0 if found else lost_lock_frames
                else:
                    lost_lock_frames += 1
                    can_relock = False
                    rescue_accept = False
                    trackless_relock_accept = False
                    candidate_anchor_score = 0.0
                    if candidate_box is not None:
                        candidate_affinity = compute_target_affinity(candidate_box, search_anchor_box, predicted_box, fallback_box)
                        candidate_anchor_score = candidate_affinity['anchor_score']
                        margin_gate = (not occlusion_mode) or crowd_margin >= 0.22 or candidate_anchor_score >= 0.94
                        appearance_gate = (not appearance_refs_exist) or candidate_appearance_score >= max(0.22, (0.60 if occlusion_mode else 0.46) - appearance_relax)
                        size_gate = is_relock_size_consistent(
                            candidate_box,
                            search_anchor_box,
                            predicted_box,
                            fallback_box,
                            lost_lock_frames,
                            occlusion_mode,
                        )
                        shape_gate = True
                        if search_anchor_box is not None and (crowded_now or occlusion_mode):
                            shape_gate = is_relock_shape_consistent(candidate_box, search_anchor_box, lost_lock_frames, crowded=True)
                        if lost_lock_frames <= 4:
                            pred_center_gate = predicted_box is None or candidate_affinity['pred_center'] <= 0.36
                        elif lost_lock_frames <= 10:
                            pred_center_gate = predicted_box is None or candidate_affinity['pred_center'] <= 0.50
                        elif lost_lock_frames <= 18:
                            pred_center_gate = predicted_box is None or candidate_affinity['pred_center'] <= 0.66
                        else:
                            pred_center_gate = True
                        candidate_consistent = is_consistent_single_target(candidate_box, search_anchor_box, predicted_box, fallback_box, lost_lock_frames)
                        rescue_accept = bool(
                            candidate_track_id is None
                            and search_backlight_score >= BACKLIGHT_TRIGGER_SCORE
                            and candidate_anchor_score >= (0.62 if occlusion_mode else 0.56)
                            and candidate_det_score >= 0.10
                            and size_gate
                            and shape_gate
                            and pred_center_gate
                            and candidate_consistent
                        )
                        trackless_relock_accept = bool(
                            candidate_track_id is None
                            and candidate_consistent
                            and appearance_gate
                            and size_gate
                            and shape_gate
                            and pred_center_gate
                            and (
                                (
                                    lost_lock_frames >= 8
                                    and candidate_det_score >= 0.30
                                    and candidate_appearance_score >= max(0.28, (0.58 if occlusion_mode else 0.42) - appearance_relax)
                                    and (candidate_score >= 0.46 or candidate_anchor_score >= 0.40)
                                )
                                or (
                                    lost_lock_frames >= 18
                                    and candidate_det_score >= 0.18
                                    and candidate_appearance_score >= max(0.24, (0.54 if occlusion_mode else 0.38) - appearance_relax)
                                    and (candidate_score >= 0.42 or candidate_anchor_score >= 0.32)
                                )
                            )
                        )
                        can_relock = bool(
                            candidate_track_id is not None
                            and candidate_track_id != locked_track_id
                            and margin_gate
                            and appearance_gate
                            and size_gate
                            and shape_gate
                            and pred_center_gate
                            and candidate_consistent
                            and (
                                candidate_det_score >= 0.60
                                or candidate_score >= 0.40
                                or candidate_anchor_score >= 0.52
                            )
                        )
                    immediate_single_target_recover = bool(
                        candidate_box is not None
                        and candidate_track_id is not None
                        and len(boxes_xyxy) == 1
                        and (not crowded_now)
                        and candidate_consistent
                        and candidate_anchor_score >= 0.46
                        and candidate_det_score >= 0.10
                        and size_gate
                        and shape_gate
                        and pred_center_gate
                    ) if candidate_box is not None else False
                    warmup_trackless_accept = bool(
                        candidate_box is not None
                        and candidate_track_id is None
                        and len(boxes_xyxy) == 1
                        and (not crowded_now)
                        and lost_lock_frames <= 3
                        and candidate_det_score >= 0.16
                        and (
                            search_anchor_box is None
                            or box_size(candidate_box) >= box_size(search_anchor_box) * 1.18
                            or candidate_score >= 0.28
                        )
                    ) if candidate_box is not None else False
                    if immediate_single_target_recover:
                        selected_box = candidate_box
                        selected_track_id = int(candidate_track_id)
                        selected_score = max(candidate_score, candidate_det_score, candidate_anchor_score)
                        selected_appearance_score = candidate_appearance_score
                        selected_from_detection = True
                        found = True
                        lost_lock_frames = 0
                        relock_candidate_id = None
                        relock_candidate_hits = 0
                    elif warmup_trackless_accept:
                        selected_box = candidate_box
                        selected_track_id = None
                        selected_score = max(candidate_score, candidate_det_score, candidate_anchor_score)
                        selected_appearance_score = candidate_appearance_score
                        selected_from_detection = True
                        found = True
                        clear_locked_track = True
                        lost_lock_frames = 0
                        relock_candidate_id = None
                        relock_candidate_hits = 0
                    elif rescue_accept:
                        selected_box = candidate_box
                        selected_track_id = None
                        selected_score = max(candidate_score, candidate_det_score, candidate_anchor_score)
                        selected_appearance_score = candidate_appearance_score
                        selected_from_detection = True
                        found = True
                        clear_locked_track = True
                        lost_lock_frames = 0
                        relock_candidate_id = None
                        relock_candidate_hits = 0
                    elif trackless_relock_accept:
                        selected_box = candidate_box
                        selected_track_id = None
                        selected_score = max(candidate_score, candidate_det_score, candidate_anchor_score)
                        selected_appearance_score = candidate_appearance_score
                        selected_from_detection = True
                        found = True
                        clear_locked_track = True
                        lost_lock_frames = 0
                        relock_candidate_id = None
                        relock_candidate_hits = 0
                    elif can_relock:
                        if relock_candidate_id == int(candidate_track_id):
                            relock_candidate_hits += 1
                        else:
                            relock_candidate_id = int(candidate_track_id)
                            relock_candidate_hits = 1
                        crowd_relock_mode = bool(crowded_now or occlusion_mode)
                        accept_fast = (
                            (not crowd_relock_mode)
                            and candidate_det_score >= 0.88
                            and candidate_anchor_score >= 0.78
                            and candidate_appearance_score >= max(0.30, 0.60 - appearance_relax)
                            and lost_lock_frames >= 1
                            and size_gate
                            and shape_gate
                            and pred_center_gate
                        )
                        accept_close = (
                            (not crowd_relock_mode)
                            and candidate_anchor_score >= 0.90
                            and candidate_appearance_score >= max(0.34, 0.64 - appearance_relax)
                            and candidate_det_score >= 0.18
                            and lost_lock_frames >= 1
                            and size_gate
                            and shape_gate
                            and pred_center_gate
                        )
                        accept_single_clear = (
                            (not crowd_relock_mode)
                            and len(boxes_xyxy) == 1
                            and candidate_anchor_score >= 0.72
                            and ((not appearance_refs_exist) or candidate_appearance_score >= max(0.24, 0.50 - appearance_relax))
                            and candidate_det_score >= 0.12
                            and lost_lock_frames >= 1
                            and size_gate
                            and shape_gate
                            and pred_center_gate
                        )
                        accept_crowd_relock = (
                            crowd_relock_mode
                            and relock_candidate_hits >= 4
                            and candidate_anchor_score >= 0.90
                            and candidate_appearance_score >= max(0.42, 0.66 - appearance_relax)
                            and candidate_det_score >= 0.14
                            and lost_lock_frames >= 2
                            and size_gate
                            and shape_gate
                            and pred_center_gate
                        )
                        stable_hits = 7 if occlusion_mode else 3
                        stable_anchor = 0.84 if occlusion_mode else 0.62
                        stable_appearance = max(0.36, (0.64 if occlusion_mode else 0.48) - appearance_relax)
                        late_frames = 18 if occlusion_mode else 9
                        late_anchor = 0.74 if occlusion_mode else 0.54
                        late_appearance = max(0.34, (0.62 if occlusion_mode else 0.45) - appearance_relax)
                        accept_stable = (
                            relock_candidate_hits >= stable_hits
                            and candidate_anchor_score >= stable_anchor
                            and candidate_appearance_score >= stable_appearance
                            and size_gate
                            and shape_gate
                            and pred_center_gate
                        )
                        accept_late = (
                            lost_lock_frames >= late_frames
                            and candidate_anchor_score >= late_anchor
                            and candidate_appearance_score >= late_appearance
                            and ((size_gate and shape_gate) or lost_lock_frames >= 22)
                        )
                        if accept_fast or accept_close or accept_single_clear or accept_crowd_relock or accept_stable or accept_late:
                            selected_box = candidate_box
                            selected_track_id = int(candidate_track_id)
                            selected_score = max(candidate_score, candidate_det_score, candidate_anchor_score)
                            selected_appearance_score = candidate_appearance_score
                            selected_from_detection = True
                            found = True
                            lost_lock_frames = 0
                            relock_candidate_id = None
                            relock_candidate_hits = 0
                    else:
                        relock_candidate_id = None
                        relock_candidate_hits = 0

            if selected_box is None:
                if search_anchor_box is not None:
                    hold_box = predicted_box if predicted_box is not None else search_anchor_box
                    hold_scale = 1.012 if occlusion_mode else 1.006
                    hold_size = box_size(hold_box)
                    if hold_size < 90.0:
                        hold_scale = max(hold_scale, 1.14)
                    if hold_size < 60.0:
                        hold_scale = max(hold_scale, 1.22)
                    selected_box = expand_box_xyxy(hold_box, hold_scale, frame_shape=frame.shape[:2])
                    selected_box = constrain_hold_box(selected_box, search_anchor_box, frame.shape[:2], lost_lock_frames, occlusion_mode)
                    hold_found_limit = 12 if hold_size < 140.0 else 10
                    if occlusion_mode and lost_lock_frames <= hold_found_limit:
                        found = True
                        selected_track_id = int(locked_track_id)
                        selected_score = max(selected_score, 0.18)
                elif fallback_box is not None:
                    selected_box = fallback_box
                else:
                    height, width = frame.shape[:2]
                    default_size = min(width, height) * 0.28
                    selected_box = expand_box_xyxy(
                        np.asarray([width * 0.40, height * 0.20, width * 0.60, height * 0.20 + default_size], dtype=np.float32),
                        1.0,
                        frame_shape=frame.shape[:2],
                    )
            else:
                if fallback_box is not None and locked_track_id is None:
                    ratio = 0.20 if box_iou(selected_box, fallback_box) > 0.05 else 0.08
                    if selected_score < 0.60:
                        ratio = max(ratio, 0.26)
                    selected_box = blend_boxes(selected_box, fallback_box, ratio)
                elif fallback_box is not None and selected_score < 0.42:
                    fallback_ratio = 0.18 if (locked_track_id is not None and (occlusion_mode or lost_lock_frames > 0)) else 0.08
                    selected_box = blend_boxes(selected_box, fallback_box, fallback_ratio)
                if search_anchor_box is not None:
                    if found:
                        blend_ratio = 0.08
                        if occlusion_mode:
                            blend_ratio = min(blend_ratio, 0.10)
                    elif lost_lock_frames <= 4:
                        blend_ratio = 0.04
                    else:
                        blend_ratio = 0.0
                    if blend_ratio > 0.0:
                        selected_box = blend_boxes(selected_box, search_anchor_box, blend_ratio)

            selected_box = clamp_box_xyxy(selected_box, frame.shape[:2])
            if selected_from_detection and found:
                selected_feature_score, selected_feature = compute_appearance_score(frame, selected_box, appearance_anchor, appearance_recent)
                if selected_feature is not None:
                    observed_score = selected_appearance_score if appearance_refs_exist else selected_feature_score
                    if appearance_anchor is None:
                        appearance_anchor = selected_feature
                    else:
                        anchor_ratio = 0.0
                        if crowded_now or occlusion_mode:
                            if observed_score >= 0.62:
                                anchor_ratio = 0.03
                        elif observed_score >= 0.68 or selected_track_id == locked_track_id or locked_track_id is None:
                            anchor_ratio = 0.10
                        if anchor_ratio > 0.0:
                            appearance_anchor = blend_appearance_feature(appearance_anchor, selected_feature, anchor_ratio)

                    if appearance_recent is None:
                        appearance_recent = selected_feature
                    else:
                        recent_ratio = 0.28 if (not crowded_now and observed_score >= 0.60) else 0.10
                        appearance_recent = blend_appearance_feature(appearance_recent, selected_feature, recent_ratio)

            tracked_boxes.append(selected_box)
            found_mask.append(found)
            crowd_mask.append(bool(crowded_now or occlusion_mode))
            prev_prev_box = None if prev_box is None else prev_box.copy()
            prev_box = selected_box.copy()
            if selected_from_detection and found:
                confirmed_prev_box = None if confirmed_box is None else confirmed_box.copy()
                confirmed_box = selected_box.copy()
            if clear_locked_track:
                locked_track_id = None
            elif selected_track_id is not None:
                locked_track_id = int(selected_track_id)

            prev_gray = gray_frame
            frame_idx += 1
            pbar.update(1)

    cap.release()

    tracked_boxes = np.asarray(tracked_boxes, dtype=np.float32)
    found_mask = np.asarray(found_mask, dtype=bool)
    crowd_mask = np.asarray(crowd_mask, dtype=bool)
    found_mask = repair_found_mask(found_mask, tracked_boxes)
    tracked_boxes = smooth_boxes(tracked_boxes, found_mask, crowd_mask)
    save_tensor_dict(cache_path, {
        'boxes_xyxy': torch.from_numpy(tracked_boxes),
        'found_mask': torch.from_numpy(found_mask.astype(np.bool_)),
        'crowd_mask': torch.from_numpy(crowd_mask.astype(np.bool_)),
    })
    return tracked_boxes, found_mask, crowd_mask


def crop_from_box(frame: np.ndarray, box_xyxy: np.ndarray):
    box_xyxy = clamp_box_xyxy(box_xyxy, frame.shape[:2])
    x1, y1, x2, y2 = np.round(box_xyxy).astype(int)
    crop = frame[y1:y2, x1:x2].copy()
    return crop, (x1, y1)


def maybe_upscale_crop(crop: np.ndarray, sr_model, threshold: float):
    max_dim = max(crop.shape[:2])
    target_dim = max(int(threshold * (1.55 if max_dim < threshold * 0.55 else 1.35)), 176)
    if max_dim >= target_dim:
        return crop, 1.0

    processed = crop
    total_scale = 1.0

    if sr_model is not None and max_dim < threshold * 0.72:
        try:
            enhanced, _ = sr_model.enhance(processed, outscale=2)
            step_scale = float(enhanced.shape[1] / max(processed.shape[1], 1))
            if step_scale > 1.01:
                processed = enhanced
                total_scale *= step_scale
        except Exception:
            pass

    cur_dim = max(processed.shape[:2])
    if cur_dim < target_dim:
        resize_scale = min(float(target_dim / max(cur_dim, 1.0)), 6.0 / max(total_scale, 1e-6))
        if resize_scale > 1.03:
            new_w = max(int(round(processed.shape[1] * resize_scale)), 2)
            new_h = max(int(round(processed.shape[0] * resize_scale)), 2)
            interp = cv2.INTER_LANCZOS4 if resize_scale >= 2.2 else cv2.INTER_CUBIC
            processed = cv2.resize(processed, (new_w, new_h), interpolation=interp)
            total_scale = float(processed.shape[1] / max(crop.shape[1], 1))

    return processed, total_scale

def infer_pose_from_processed_crop(processed_crop: np.ndarray, origin, scale_factor: float, pose_model, ref_kp=None):
    bboxes = [[0, 0, processed_crop.shape[1], processed_crop.shape[0]]]
    keypoints, scores = pose_model(processed_crop, bboxes=bboxes)
    if len(keypoints) == 0:
        return None, -1e9, 0.0, 0

    crop_center = np.asarray([processed_crop.shape[1] * 0.5, processed_crop.shape[0] * 0.5], dtype=np.float32)
    crop_scale = max(float(max(processed_crop.shape[:2])), 1.0)
    best_kp = None
    best_score = -1e9
    best_mean_conf = 0.0
    best_visible_count = 0
    for person_idx in range(len(keypoints)):
        xy = np.asarray(keypoints[person_idx], dtype=np.float32)
        conf = np.asarray(scores[person_idx], dtype=np.float32).reshape(-1)
        local_xy = xy.copy()
        if scale_factor != 1.0:
            xy /= scale_factor
        xy[:, 0] += origin[0]
        xy[:, 1] += origin[1]
        kp = np.concatenate([xy, conf[:, None]], axis=1).astype(np.float32)
        mean_conf = weighted_mean_conf(conf)
        visible_count = int((conf >= 0.20).sum())
        temporal_score = joint_match_score(kp[:, :2], conf, ref_kp) if ref_kp is not None else 0.50
        pose_box_local = estimate_box_from_keypoints(local_xy, conf, conf_thresh=0.20)
        pose_center = crop_center if pose_box_local is None else box_center(pose_box_local)
        center_offset = float(np.linalg.norm(pose_center - crop_center) / crop_scale)
        center_score = float(np.clip(1.0 - center_offset / 0.28, 0.0, 1.0))
        if ref_kp is not None:
            total = (
                0.34 * mean_conf
                + 0.46 * temporal_score
                + 0.12 * center_score
                + 0.08 * (visible_count / 17.0)
            )
            if temporal_score < 0.40:
                total -= 0.20
            elif temporal_score >= 0.74 and visible_count >= 11:
                total += 0.04
        else:
            total = (
                0.52 * mean_conf
                + 0.26 * temporal_score
                + 0.14 * center_score
                + 0.08 * (visible_count / 17.0)
            )
        if total > best_score:
            best_score = total
            best_kp = kp
            best_mean_conf = float(mean_conf)
            best_visible_count = visible_count
    return best_kp, best_score, best_mean_conf, best_visible_count


def build_pose_crop_variants(crop: np.ndarray, sr_model, small_box_threshold: float):
    processed_crop, scale_factor = maybe_upscale_crop(crop, sr_model, small_box_threshold)
    variants = [(processed_crop, scale_factor, 0.0, False)]
    backlight_score = estimate_backlight_score(crop)
    max_dim = max(crop.shape[:2])
    need_extra = backlight_score >= BACKLIGHT_TRIGGER_SCORE or max_dim < small_box_threshold * 1.10
    if not need_extra:
        return variants, backlight_score

    enhanced_crop = enhance_backlit_image(crop, strong=False)
    enhanced_processed, enhanced_scale = maybe_upscale_crop(enhanced_crop, sr_model, small_box_threshold)
    variants.append((enhanced_processed, enhanced_scale, backlight_score, False))

    if backlight_score >= STRONG_BACKLIGHT_SCORE or max_dim < small_box_threshold * 0.78:
        strong_crop = enhance_backlit_image(crop, strong=True)
        strong_processed, strong_scale = maybe_upscale_crop(strong_crop, sr_model, small_box_threshold)
        variants.append((strong_processed, strong_scale, backlight_score, True))

    return variants, backlight_score


def pose_from_crop(frame: np.ndarray, box_xyxy: np.ndarray, pose_model, sr_model, small_box_threshold: float, ref_kp=None):
    crop, origin = crop_from_box(frame, box_xyxy)
    if crop.size == 0:
        return None

    variants, backlight_score = build_pose_crop_variants(crop, sr_model, small_box_threshold)
    base_processed, base_scale, _, _ = variants[0]
    best_kp, base_score, base_mean_conf, base_visible_count = infer_pose_from_processed_crop(base_processed, origin, base_scale, pose_model, ref_kp=ref_kp)
    best_total = base_score

    if len(variants) == 1:
        return best_kp

    base_temporal = joint_match_score(best_kp[:, :2], best_kp[:, 2], ref_kp) if (best_kp is not None and ref_kp is not None) else 0.50
    crop_small = max(crop.shape[:2]) < small_box_threshold * 0.82
    need_rescue = (
        best_kp is None
        or base_mean_conf < 0.60
        or base_visible_count < 10
        or base_temporal < 0.68
        or (crop_small and (base_mean_conf < 0.72 or base_visible_count < 12))
        or backlight_score >= STRONG_BACKLIGHT_SCORE
    )
    if not need_rescue:
        return best_kp

    for processed_crop, scale_factor, variant_backlight, strong_variant in variants[1:]:
        kp, score, mean_conf, visible_count = infer_pose_from_processed_crop(processed_crop, origin, scale_factor, pose_model, ref_kp=ref_kp)
        if kp is None:
            continue
        variant_bonus = 0.0
        if variant_backlight >= BACKLIGHT_TRIGGER_SCORE:
            variant_bonus += 0.02 * float(np.clip(variant_backlight, 0.0, 1.0))
            if strong_variant:
                variant_bonus += 0.015
            if mean_conf >= 0.48:
                variant_bonus += 0.012
            if visible_count >= 11:
                variant_bonus += 0.010
        total = score + variant_bonus
        if total > best_total:
            best_total = total
            best_kp = kp

    if best_kp is None and backlight_score >= BACKLIGHT_TRIGGER_SCORE:
        enhanced_crop = enhance_backlit_image(crop, strong=True)
        processed_crop, scale_factor = maybe_upscale_crop(enhanced_crop, sr_model, small_box_threshold)
        best_kp, _, _, _ = infer_pose_from_processed_crop(processed_crop, origin, scale_factor, pose_model, ref_kp=ref_kp)
    return best_kp


def build_pose_candidates(base_box, seed_box, prev_box, frame_shape, crowded=False):
    size = box_size(base_box)
    candidates = []
    seen = set()

    def add(box):
        box = clamp_box_xyxy(box, frame_shape)
        key = tuple(np.round(box, 1).tolist())
        if key not in seen:
            seen.add(key)
            candidates.append(box)

    context = SMALL_BOX_CONTEXT if size < SMALL_BOX_THRESHOLD else NORMAL_BOX_CONTEXT
    if crowded:
        context = min(context, 1.06)
    if size < 55.0:
        context = max(context, 1.90)
    elif size < 80.0:
        context = max(context, 1.62)
    elif size < 110.0:
        context = max(context, 1.38)

    if crowded and prev_box is not None:
        tight_scale = 1.04 if size < 110.0 else 0.98
        add(expand_box_xyxy(prev_box, tight_scale, shift_y_ratio=-0.02, frame_shape=frame_shape))
        add(expand_box_xyxy(prev_box, 0.92, shift_y_ratio=-0.04, frame_shape=frame_shape))
        add(blend_boxes(expand_box_xyxy(base_box, max(context, 1.00), frame_shape=frame_shape), expand_box_xyxy(prev_box, 0.98, frame_shape=frame_shape), 0.72))
    add(expand_box_xyxy(base_box, context, frame_shape=frame_shape))

    if size < 55.0:
        add(expand_box_xyxy(base_box, 2.20, shift_y_ratio=-0.05, frame_shape=frame_shape))
        add(expand_box_xyxy(base_box, 1.78, shift_y_ratio=-0.08, frame_shape=frame_shape))
    elif size < 85.0:
        add(expand_box_xyxy(base_box, 1.78, shift_y_ratio=-0.04, frame_shape=frame_shape))
        add(expand_box_xyxy(base_box, 1.46, shift_y_ratio=-0.06, frame_shape=frame_shape))
    elif size < 175.0:
        add(expand_box_xyxy(base_box, 0.96, shift_y_ratio=-0.03, frame_shape=frame_shape))
        add(expand_box_xyxy(base_box, 0.88, shift_y_ratio=-0.05, frame_shape=frame_shape))
    elif size < 260.0:
        add(expand_box_xyxy(base_box, 1.05, shift_y_ratio=-0.02, frame_shape=frame_shape))

    if seed_box is not None:
        seed_scale = 1.12 if size < 80.0 else 1.08
        seed_ratio = 0.58 if size < 80.0 else 0.45
        add(blend_boxes(expand_box_xyxy(base_box, context, frame_shape=frame_shape), expand_box_xyxy(seed_box, seed_scale, frame_shape=frame_shape), seed_ratio))
    if prev_box is not None and size < 240.0:
        prev_scale = 1.12 if size < 80.0 else 1.04
        prev_ratio = 0.48 if size < 80.0 else 0.25
        add(blend_boxes(expand_box_xyxy(base_box, context, frame_shape=frame_shape), expand_box_xyxy(prev_box, prev_scale, frame_shape=frame_shape), prev_ratio))
        if size < 90.0:
            add(expand_box_xyxy(prev_box, 1.14, frame_shape=frame_shape))
            add(blend_boxes(expand_box_xyxy(base_box, max(context, 1.55), frame_shape=frame_shape), expand_box_xyxy(prev_box, 1.08, frame_shape=frame_shape), 0.62))

    if crowded and prev_box is not None:
        add(expand_box_xyxy(prev_box, 1.02, frame_shape=frame_shape))
        add(expand_box_xyxy(prev_box, 0.94, shift_y_ratio=-0.02, frame_shape=frame_shape))
        add(blend_boxes(expand_box_xyxy(base_box, max(context, 1.02), frame_shape=frame_shape), expand_box_xyxy(prev_box, 1.00, frame_shape=frame_shape), 0.60))
        add(blend_boxes(expand_box_xyxy(base_box, 0.96, frame_shape=frame_shape), expand_box_xyxy(prev_box, 0.96, frame_shape=frame_shape), 0.72))

    return candidates

def joint_match_score(cur_xy, cur_conf, ref_kp):
    if ref_kp is None:
        return 0.50
    ref_xy = ref_kp[:, :2]
    ref_conf = np.clip(ref_kp[:, 2], 0.0, 1.0)
    valid = (cur_conf >= 0.20) & (ref_conf >= 0.20)
    if int(valid.sum()) < 4:
        return 0.45
    ref_box = estimate_box_from_keypoints(ref_xy, ref_conf, conf_thresh=0.20)
    scale = box_size(ref_box) if ref_box is not None else 180.0
    dist = np.linalg.norm(cur_xy[valid] - ref_xy[valid], axis=1)
    mean_dist = float(np.mean(dist) / max(scale, 1.0))
    return float(np.clip(1.0 - mean_dist / 0.18, 0.0, 1.0))


def remap_keypoints_to_box(ref_kp, target_box, conf_scale=0.88):
    if ref_kp is None or target_box is None:
        return None
    ref_box = estimate_box_from_keypoints(ref_kp[:, :2], ref_kp[:, 2], conf_thresh=0.18)
    if ref_box is None:
        return None
    ref_center = box_center(ref_box)
    target_center = box_center(target_box)
    scale = float(box_size(target_box) / max(box_size(ref_box), 1.0))
    out = ref_kp.copy().astype(np.float32)
    out[:, 0] = (out[:, 0] - ref_center[0]) * scale + target_center[0]
    out[:, 1] = (out[:, 1] - ref_center[1]) * scale + target_center[1]
    out[:, 2] = np.clip(out[:, 2] * conf_scale, 0.0, 0.97)
    return out


def pose_structure_stats(kp2d, conf_thresh=0.18):
    if kp2d is None:
        return 0, None, 0.0, 0.0
    xy = np.asarray(kp2d[:, :2], dtype=np.float32)
    conf = np.clip(np.asarray(kp2d[:, 2], dtype=np.float32), 0.0, 1.0)
    visible = conf >= conf_thresh
    visible_count = int(visible.sum())
    pose_box = estimate_box_from_keypoints(xy, conf, conf_thresh=conf_thresh)
    pose_size = 0.0 if pose_box is None else box_size(pose_box)
    if visible_count >= 2:
        xy_std = float(np.max(np.std(xy[visible], axis=0)))
    else:
        xy_std = 0.0
    return visible_count, pose_box, float(pose_size), xy_std


def is_degenerate_pose(kp2d, candidate_box):
    visible_count, _, pose_size, xy_std = pose_structure_stats(kp2d, conf_thresh=0.18)
    if candidate_box is None:
        return visible_count < 5
    cand_size = box_size(candidate_box)
    if visible_count < 4:
        return True
    spread_ratio = float(pose_size / max(cand_size, 1.0))
    std_ratio = float(xy_std / max(cand_size, 1.0))
    if cand_size < 55.0:
        return bool(spread_ratio < 0.30 or std_ratio < 0.10)
    if cand_size < 85.0:
        return bool(spread_ratio < 0.24 or std_ratio < 0.08)
    if cand_size < 120.0:
        return bool(spread_ratio < 0.18 or std_ratio < 0.06)
    return bool(spread_ratio < 0.12 or std_ratio < 0.045)

def score_pose_candidate(kp2d, candidate_box, prev_kp, seed_kp, crowded=False):
    xy = kp2d[:, :2].astype(np.float32)
    conf = np.clip(kp2d[:, 2].astype(np.float32), 0.0, 1.0)
    visible = conf >= 0.20
    visible_count = int(visible.sum())
    if visible_count < 5:
        return -1e9

    mean_conf = weighted_mean_conf(conf)
    torso_conf = float(np.mean(conf[[5, 6, 11, 12]]))
    leg_conf = float(np.mean(conf[[11, 12, 13, 14, 15, 16]]))
    cand_center = box_center(candidate_box)
    pose_box = estimate_box_from_keypoints(xy, conf, conf_thresh=0.20)
    pose_center = box_center(pose_box) if pose_box is not None else cand_center
    center_offset = float(np.linalg.norm(cand_center - pose_center) / max(box_size(candidate_box), 1.0))
    center_score = float(np.clip(1.0 - center_offset / 0.18, 0.0, 1.0))
    temporal_score = joint_match_score(xy, conf, prev_kp)
    seed_score = joint_match_score(xy, conf, seed_kp)

    temporal_penalty = 0.0
    if prev_kp is not None and temporal_score < 0.38:
        temporal_penalty = 0.22 if crowded else 0.10
    seed_penalty = 0.0
    if seed_kp is not None and seed_score < 0.34:
        seed_penalty = 0.14 if crowded else 0.06

    if crowded:
        return (
            0.24 * mean_conf
            + 0.06 * torso_conf
            + 0.04 * leg_conf
            + 0.36 * temporal_score
            + 0.22 * seed_score
            + 0.04 * center_score
            + 0.04 * (visible_count / 17.0)
            - temporal_penalty
            - seed_penalty
        )

    return (
        0.38 * mean_conf
        + 0.12 * torso_conf
        + 0.08 * leg_conf
        + 0.20 * temporal_score
        + 0.12 * seed_score
        + 0.06 * center_score
        + 0.04 * (visible_count / 17.0)
        - temporal_penalty
        - seed_penalty
    )


def merge_with_seed(best_kp, seed_kp):
    if seed_kp is None:
        return best_kp
    out = best_kp.copy().astype(np.float32)
    for joint_idx in range(out.shape[0]):
        cur_conf = float(out[joint_idx, 2])
        seed_conf = float(seed_kp[joint_idx, 2])
        if cur_conf < 0.18 and seed_conf > 0.45:
            out[joint_idx, :2] = seed_kp[joint_idx, :2]
            out[joint_idx, 2] = min(0.96, seed_conf * 0.92)
        elif cur_conf < 0.40 and seed_conf > cur_conf + 0.18:
            out[joint_idx, :2] = 0.60 * seed_kp[joint_idx, :2] + 0.40 * out[joint_idx, :2]
            out[joint_idx, 2] = min(0.95, 0.55 * seed_conf + 0.45 * cur_conf)
    return out


def generate_keypoints(
    input_path: Path,
    track_boxes: np.ndarray,
    crowd_mask: np.ndarray | None,
    keypoint_cache_path: Path,
    pose_model_ref: str,
    seed_arr: np.ndarray | None,
    sr_model,
    small_box_threshold: float,
    max_frames: int,
    force: bool,
):
    if keypoint_cache_path.exists() and not force:
        cache = load_tensor_dict(keypoint_cache_path)
        if isinstance(cache, dict) and 'raw_keypoints' in cache:
            print(f'[Info] Reuse raw keypoint cache: {keypoint_cache_path}')
            return np.asarray(cache['raw_keypoints'], dtype=np.float32)
        if isinstance(cache, dict) and 'keypoints' in cache:
            print(f'[Info] Existing cache lacks raw keypoints, regenerate pose: {keypoint_cache_path}')

    pose_model = load_pose_model(pose_model_ref)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频：{input_path}')

    total_frames = len(track_boxes)
    frame_limit = total_frames
    if seed_arr is not None:
        frame_limit = min(frame_limit, len(seed_arr))
    if crowd_mask is not None:
        frame_limit = min(frame_limit, len(crowd_mask))
    if max_frames and max_frames > 0:
        frame_limit = min(frame_limit, int(max_frames))

    results = np.zeros((frame_limit, 17, 3), dtype=np.float32)
    prev_kp = None
    prev_box = None

    with tqdm(total=frame_limit, desc='RTMPose-X', unit='frame') as pbar:
        for frame_idx in range(frame_limit):
            ok, frame = cap.read()
            if not ok:
                results = results[:frame_idx]
                break

            # 性能优化：降分辨率（长边压缩至1080px）
            h, w = frame.shape[:2]
            if max(h, w) > 1080:
                scale = 1080.0 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            base_box = clamp_box_xyxy(track_boxes[frame_idx], frame.shape[:2])
            crowded = bool(crowd_mask is not None and frame_idx < len(crowd_mask) and crowd_mask[frame_idx])
            seed_kp = None if seed_arr is None else seed_arr[frame_idx].astype(np.float32)
            seed_box = estimate_seed_box(seed_kp)
            candidates = build_pose_candidates(base_box, seed_box, prev_box, frame.shape[:2], crowded=crowded)

            best_kp = None
            best_box = None
            best_score = -1e9
            base_size = box_size(base_box)
            for candidate_idx, candidate_box in enumerate(candidates):
                kp2d = pose_from_crop(frame, candidate_box, pose_model, sr_model, small_box_threshold, ref_kp=prev_kp)
                if kp2d is None:
                    continue
                degenerate = is_degenerate_pose(kp2d, candidate_box)
                score = score_pose_candidate(kp2d, candidate_box, prev_kp, seed_kp, crowded=crowded)
                if degenerate:
                    score -= 0.34 if base_size < 90.0 else 0.20
                if score > best_score:
                    best_score = score
                    best_kp = kp2d
                    best_box = candidate_box

                if crowded:
                    continue

                cur_conf = np.clip(kp2d[:, 2], 0.0, 1.0)
                mean_conf = float(np.mean(cur_conf))
                temporal_ok = prev_kp is None or joint_match_score(kp2d[:, :2], cur_conf, prev_kp) >= 0.74
                seed_ok = seed_kp is None or joint_match_score(kp2d[:, :2], cur_conf, seed_kp) >= 0.68
                if candidate_idx == 0 and base_size >= max(110.0, small_box_threshold * 0.92):
                    if score >= 0.67 and mean_conf >= 0.63 and temporal_ok and seed_ok:
                        break
                elif candidate_idx == 1 and best_score >= 0.72 and mean_conf >= 0.60 and temporal_ok:
                    break

            fallback_box_for_pose = best_box if best_box is not None else base_box
            mapped_prev_kp = remap_keypoints_to_box(prev_kp, fallback_box_for_pose, conf_scale=0.90 if base_size < 90.0 else 0.86)
            mapped_seed_kp = remap_keypoints_to_box(seed_kp, fallback_box_for_pose, conf_scale=0.92 if base_size < 90.0 else 0.88)

            if best_kp is None:
                if mapped_seed_kp is not None:
                    best_kp = mapped_seed_kp
                elif mapped_prev_kp is not None:
                    best_kp = mapped_prev_kp
                elif seed_kp is not None:
                    best_kp = seed_kp.copy().astype(np.float32)
                elif prev_kp is not None:
                    best_kp = prev_kp.copy().astype(np.float32)
                else:
                    best_kp = np.zeros((17, 3), dtype=np.float32)
            else:
                if base_size < 120.0 and is_degenerate_pose(best_kp, fallback_box_for_pose):
                    if mapped_prev_kp is not None:
                        best_kp = mapped_prev_kp
                    elif mapped_seed_kp is not None:
                        best_kp = mapped_seed_kp
                elif seed_kp is not None:
                    best_kp = merge_with_seed(best_kp, seed_kp)
                best_kp[:, 2] = np.clip(best_kp[:, 2], 0.0, 1.0)

            results[frame_idx] = best_kp
            prev_kp = best_kp.copy()
            if best_box is None:
                prev_box = base_box.copy()
            else:
                pose_box = estimate_box_from_keypoints(best_kp[:, :2], best_kp[:, 2], conf_thresh=0.18)
                prev_box = best_box if pose_box is None else blend_boxes(best_box, pose_box, 0.35)
            pbar.update(1)

    cap.release()
    save_tensor_dict(keypoint_cache_path, {'raw_keypoints': torch.from_numpy(results.astype(np.float32))})
    return results


def interpolate_short_gaps(keypoints_arr: np.ndarray, max_gap: int = 2):
    out = keypoints_arr.copy().astype(np.float32)
    frame_count, joint_count = out.shape[:2]
    for joint_idx in range(joint_count):
        conf = out[:, joint_idx, 2]
        valid = conf >= 0.20
        frame_idx = 0
        while frame_idx < frame_count:
            if valid[frame_idx]:
                frame_idx += 1
                continue
            gap_start = frame_idx
            while frame_idx < frame_count and not valid[frame_idx]:
                frame_idx += 1
            gap_end = frame_idx
            gap = gap_end - gap_start
            if gap > max_gap or gap_start == 0 or gap_end >= frame_count:
                continue
            left = out[gap_start - 1, joint_idx]
            right = out[gap_end, joint_idx]
            if left[2] < 0.20 or right[2] < 0.20:
                continue
            for missing_idx in range(gap):
                ratio = float((missing_idx + 1) / (gap + 1))
                out[gap_start + missing_idx, joint_idx, :2] = left[:2] * (1.0 - ratio) + right[:2] * ratio
                out[gap_start + missing_idx, joint_idx, 2] = min(left[2], right[2]) * 0.92
    return out


def ema_temporal_pass(keypoints_arr: np.ndarray, track_boxes: np.ndarray, reverse: bool = False):
    out = keypoints_arr[::-1].copy().astype(np.float32) if reverse else keypoints_arr.copy().astype(np.float32)
    boxes = track_boxes[::-1].copy().astype(np.float32) if reverse else track_boxes.copy().astype(np.float32)
    prev_xy = None
    prev_conf = None

    for frame_idx in range(len(out)):
        xy = out[frame_idx, :, :2].copy()
        conf = np.clip(out[frame_idx, :, 2].copy(), 0.0, 1.0)
        if prev_xy is None:
            prev_xy = xy.copy()
            prev_conf = conf.copy()
            continue

        body_scale = box_size(boxes[frame_idx])
        refined_xy = xy.copy()
        refined_conf = conf.copy()
        for joint_idx in range(out.shape[1]):
            score = float(conf[joint_idx])
            if float(prev_conf[joint_idx]) < 0.05:
                continue
            if score < 0.20:
                refined_xy[joint_idx] = prev_xy[joint_idx]
                refined_conf[joint_idx] = max(score, prev_conf[joint_idx] * 0.92)
                continue
            alpha = float(np.clip(TEMPORAL_BASE_ALPHA[joint_idx] + 0.56 * score, 0.10, 0.88))
            delta = float(np.linalg.norm(xy[joint_idx] - prev_xy[joint_idx]))
            jump_limit = float(JOINT_JUMP_RATIO[joint_idx] * body_scale)
            if delta > jump_limit and score < 0.88:
                alpha *= 0.24
            refined_xy[joint_idx] = alpha * xy[joint_idx] + (1.0 - alpha) * prev_xy[joint_idx]

        out[frame_idx, :, :2] = refined_xy
        out[frame_idx, :, 2] = refined_conf
        prev_xy = refined_xy.copy()
        prev_conf = refined_conf.copy()

    return out[::-1] if reverse else out


def window_temporal_refine(keypoints_arr: np.ndarray):
    src = keypoints_arr.copy().astype(np.float32)
    out = src.copy()
    frame_count, joint_count = src.shape[:2]
    for frame_idx in range(frame_count):
        left = max(0, frame_idx - 2)
        right = min(frame_count, frame_idx + 3)
        window_idx = np.arange(left, right)
        kernel = 1.0 / (1.0 + np.abs(window_idx - frame_idx).astype(np.float32))
        kernel[window_idx == frame_idx] *= 1.45
        for joint_idx in range(joint_count):
            conf = np.clip(src[left:right, joint_idx, 2], 0.05, 1.0)
            weights = kernel * conf
            if float(weights.sum()) <= 0.0:
                continue
            out[frame_idx, joint_idx, :2] = np.sum(src[left:right, joint_idx, :2] * weights[:, None], axis=0) / weights.sum()
            out[frame_idx, joint_idx, 2] = max(src[frame_idx, joint_idx, 2], float(np.max(src[left:right, joint_idx, 2]) * 0.98))
    return out


def offline_temporal_refine(keypoints_arr: np.ndarray, track_boxes: np.ndarray):
    base = interpolate_short_gaps(keypoints_arr)
    forward = ema_temporal_pass(base, track_boxes, reverse=False)
    backward = ema_temporal_pass(base, track_boxes, reverse=True)
    merged = base.copy().astype(np.float32)

    for frame_idx in range(len(merged)):
        for joint_idx in range(merged.shape[1]):
            candidates_xy = np.stack([
                base[frame_idx, joint_idx, :2],
                forward[frame_idx, joint_idx, :2],
                backward[frame_idx, joint_idx, :2],
            ], axis=0)
            weights = np.asarray([
                max(float(base[frame_idx, joint_idx, 2]), 0.05),
                max(float(forward[frame_idx, joint_idx, 2]) * 0.85, 0.05),
                max(float(backward[frame_idx, joint_idx, 2]) * 0.85, 0.05),
            ], dtype=np.float32)
            merged[frame_idx, joint_idx, :2] = np.average(candidates_xy, axis=0, weights=weights)
            merged[frame_idx, joint_idx, 2] = float(np.clip(max(
                base[frame_idx, joint_idx, 2],
                forward[frame_idx, joint_idx, 2] * 0.97,
                backward[frame_idx, joint_idx, 2] * 0.97,
            ), 0.0, 1.0))

    return window_temporal_refine(merged)


def point_in_box(point_xy: np.ndarray, box_xyxy: np.ndarray) -> bool:
    return bool(
        box_xyxy[0] <= point_xy[0] <= box_xyxy[2]
        and box_xyxy[1] <= point_xy[1] <= box_xyxy[3]
    )


def expanded_track_box(track_box: np.ndarray) -> np.ndarray:
    size = box_size(track_box)
    scale = 1.18 if size < 180.0 else 1.24
    return expand_box_xyxy(track_box, scale)


def enforce_track_box_consistency(keypoints_arr: np.ndarray, track_boxes: np.ndarray) -> np.ndarray:
    out = keypoints_arr.copy().astype(np.float32)
    frame_count = min(len(out), len(track_boxes))
    joint_count = out.shape[1]

    for frame_idx in range(frame_count):
        cur_box = expanded_track_box(np.asarray(track_boxes[frame_idx], dtype=np.float32))
        for joint_idx in range(joint_count):
            if float(out[frame_idx, joint_idx, 2]) < 0.18:
                continue
            if point_in_box(out[frame_idx, joint_idx, :2], cur_box):
                continue

            candidates_xy = []
            candidates_w = []
            candidates_conf = []
            for delta in range(1, 4):
                for neighbor_idx in (frame_idx - delta, frame_idx + delta):
                    if neighbor_idx < 0 or neighbor_idx >= frame_count:
                        continue
                    cand = out[neighbor_idx, joint_idx]
                    if float(cand[2]) < 0.18:
                        continue
                    if point_in_box(cand[:2], cur_box):
                        candidates_xy.append(cand[:2].copy())
                        candidates_conf.append(float(cand[2]))
                        candidates_w.append(float(cand[2]) / delta)

            if candidates_xy:
                weights = np.asarray(candidates_w, dtype=np.float32)
                out[frame_idx, joint_idx, :2] = np.average(np.stack(candidates_xy, axis=0), axis=0, weights=weights)
                out[frame_idx, joint_idx, 2] = float(np.clip(np.average(np.asarray(candidates_conf, dtype=np.float32), weights=weights), 0.20, 0.95))
            else:
                out[frame_idx, joint_idx, 2] = min(float(out[frame_idx, joint_idx, 2]), 0.05)

    return window_temporal_refine(interpolate_short_gaps(out, max_gap=1))



def compute_pose_center(keypoints_xy: np.ndarray, keypoints_conf: np.ndarray):
    torso_ids = [5, 6, 11, 12]
    torso_valid = [idx for idx in torso_ids if float(keypoints_conf[idx]) >= 0.20]
    if len(torso_valid) >= 2:
        return np.mean(keypoints_xy[torso_valid], axis=0).astype(np.float32)
    valid = np.asarray(keypoints_conf, dtype=np.float32) >= 0.20
    if int(valid.sum()) >= 4:
        return np.mean(keypoints_xy[valid], axis=0).astype(np.float32)
    return None


def reduce_temporal_lag(refined_arr: np.ndarray, raw_arr: np.ndarray, track_boxes: np.ndarray):
    out = refined_arr.copy().astype(np.float32)
    frame_count = min(len(out), len(raw_arr), len(track_boxes))
    fast_response_joints = {7, 8, 9, 10, 13, 14, 15, 16}

    for frame_idx in range(frame_count):
        body_scale = max(box_size(track_boxes[frame_idx]), 1.0)
        raw_xy = raw_arr[frame_idx, :, :2].astype(np.float32)
        raw_conf = np.clip(raw_arr[frame_idx, :, 2].astype(np.float32), 0.0, 1.0)
        out_xy = out[frame_idx, :, :2]
        out_conf = np.clip(out[frame_idx, :, 2].astype(np.float32), 0.0, 1.0)

        track_motion = 0.0
        if frame_idx > 0:
            prev_center = box_center(np.asarray(track_boxes[frame_idx - 1], dtype=np.float32))
            cur_center = box_center(np.asarray(track_boxes[frame_idx], dtype=np.float32))
            track_motion = float(np.linalg.norm(cur_center - prev_center) / body_scale)

        raw_center = compute_pose_center(raw_xy, raw_conf)
        out_center = compute_pose_center(out_xy, out_conf)
        if raw_center is not None and out_center is not None:
            center_delta = raw_center - out_center
            center_delta_norm = float(np.linalg.norm(center_delta) / body_scale)
            if center_delta_norm > 0.020:
                center_ratio = float(np.clip((center_delta_norm - 0.020) / 0.09, 0.24, 0.88))
                if track_motion > 0.030:
                    center_ratio = min(0.92, center_ratio + 0.08)
                valid = out_conf >= 0.18
                out[frame_idx, valid, :2] += center_delta * center_ratio

        for joint_idx in range(out.shape[1]):
            if float(raw_conf[joint_idx]) < 0.28 or float(out_conf[joint_idx]) < 0.18:
                continue
            delta = float(np.linalg.norm(out[frame_idx, joint_idx, :2] - raw_xy[joint_idx]) / body_scale)
            if delta <= 0.014:
                continue
            joint_ratio = float(np.clip((delta - 0.014) / 0.08, 0.22, 0.90))
            if joint_idx in fast_response_joints:
                joint_ratio = min(0.94, joint_ratio + 0.08)
            if body_scale < 170.0:
                joint_ratio = min(0.95, joint_ratio + 0.05)
            if track_motion > 0.030:
                joint_ratio = min(0.96, joint_ratio + float(np.clip((track_motion - 0.030) / 0.06, 0.04, 0.14)))
            if frame_idx > 0 and float(raw_arr[frame_idx - 1, joint_idx, 2]) >= 0.28:
                raw_step = float(np.linalg.norm(raw_xy[joint_idx] - raw_arr[frame_idx - 1, joint_idx, :2]) / body_scale)
                out_step = float(np.linalg.norm(out[frame_idx, joint_idx, :2] - out[frame_idx - 1, joint_idx, :2]) / body_scale)
                if raw_step > out_step + 0.020:
                    joint_ratio = min(0.97, joint_ratio + float(np.clip((raw_step - out_step - 0.020) / 0.08, 0.05, 0.18)))
            if float(raw_conf[joint_idx]) > float(out_conf[joint_idx]) + 0.08:
                joint_ratio = max(joint_ratio, 0.50)
            out[frame_idx, joint_idx, :2] = (
                out[frame_idx, joint_idx, :2] * (1.0 - joint_ratio)
                + raw_xy[joint_idx] * joint_ratio
            )
            out[frame_idx, joint_idx, 2] = max(out[frame_idx, joint_idx, 2], raw_conf[joint_idx] * 0.99)

    return out.astype(np.float32)


def repair_brief_outliers(refined_arr: np.ndarray, raw_arr: np.ndarray, track_boxes: np.ndarray):
    out = refined_arr.copy().astype(np.float32)
    frame_count = min(len(out), len(raw_arr), len(track_boxes))
    fast_response_joints = {7, 8, 9, 10, 13, 14, 15, 16}

    for frame_idx in range(1, frame_count - 1):
        body_scale = max(box_size(track_boxes[frame_idx]), 1.0)
        for joint_idx in range(out.shape[1]):
            prev_conf = float(out[frame_idx - 1, joint_idx, 2])
            next_conf = float(out[frame_idx + 1, joint_idx, 2])
            cur_conf = float(out[frame_idx, joint_idx, 2])
            raw_conf = float(raw_arr[frame_idx, joint_idx, 2])
            if min(prev_conf, next_conf, raw_conf) < 0.24 or cur_conf < 0.18:
                continue

            neighbor_mid = (out[frame_idx - 1, joint_idx, :2] + out[frame_idx + 1, joint_idx, :2]) * 0.5
            cur_err = float(np.linalg.norm(out[frame_idx, joint_idx, :2] - neighbor_mid) / body_scale)
            raw_err = float(np.linalg.norm(raw_arr[frame_idx, joint_idx, :2] - neighbor_mid) / body_scale)
            neighbor_span = float(np.linalg.norm(out[frame_idx + 1, joint_idx, :2] - out[frame_idx - 1, joint_idx, :2]) / body_scale)
            if cur_err <= max(0.040, neighbor_span * 1.25):
                continue
            if raw_err + 0.010 >= cur_err:
                continue

            blend = 0.76 if joint_idx in fast_response_joints else 0.62
            out[frame_idx, joint_idx, :2] = out[frame_idx, joint_idx, :2] * (1.0 - blend) + raw_arr[frame_idx, joint_idx, :2] * blend
            out[frame_idx, joint_idx, 2] = max(out[frame_idx, joint_idx, 2], raw_conf * 0.99)

    return out.astype(np.float32)


def compress_pair_distance(points_xy: np.ndarray, idx_a: int, idx_b: int, target_dist: float):
    point_a = points_xy[idx_a].copy()
    point_b = points_xy[idx_b].copy()
    mid = (point_a + point_b) * 0.5
    delta = point_b - point_a
    dist = float(np.linalg.norm(delta))
    if dist > 1e-4:
        direction = delta / dist
    else:
        direction = np.asarray([1.0, 0.0], dtype=np.float32)
    half = direction * (target_dist * 0.5)
    points_xy[idx_a] = mid - half
    points_xy[idx_b] = mid + half


def prepare_render_keypoints(keypoints_xy: np.ndarray, keypoints_conf: np.ndarray, box_xyxy: np.ndarray | None):
    out_xy = keypoints_xy.astype(np.float32).copy()
    out_conf = keypoints_conf.astype(np.float32).copy()
    if box_xyxy is None:
        return out_xy, out_conf

    body_scale = max(box_size(box_xyxy), 1.0)
    pair_specs = [
        (11, 12, 0.15, 0.050, 0.020),
        (13, 14, 0.13, 0.036, 0.015),
        (15, 16, 0.11, 0.028, 0.012),
    ]

    close_count = 0
    lower_body_points = []
    for idx in (11, 12, 13, 14, 15, 16):
        if float(out_conf[idx]) >= KEYPOINT_DRAW_CONF_THRESHOLD:
            lower_body_points.append(out_xy[idx])
    lower_body_width = None
    if len(lower_body_points) >= 4:
        lower_body_points = np.stack(lower_body_points, axis=0)
        lower_body_width = float(lower_body_points[:, 0].max() - lower_body_points[:, 0].min())

    for left_idx, right_idx, trigger_ratio, _, _ in pair_specs:
        if float(out_conf[left_idx]) < KEYPOINT_DRAW_CONF_THRESHOLD or float(out_conf[right_idx]) < KEYPOINT_DRAW_CONF_THRESHOLD:
            continue
        pair_dist = float(np.linalg.norm(out_xy[left_idx] - out_xy[right_idx]))
        if pair_dist <= body_scale * trigger_ratio:
            close_count += 1

    overlap_mode = close_count >= 2 or (lower_body_width is not None and lower_body_width <= body_scale * 0.14)

    for left_idx, right_idx, trigger_ratio, max_ratio, min_ratio in pair_specs:
        if float(out_conf[left_idx]) < KEYPOINT_DRAW_CONF_THRESHOLD or float(out_conf[right_idx]) < KEYPOINT_DRAW_CONF_THRESHOLD:
            continue
        pair_dist = float(np.linalg.norm(out_xy[left_idx] - out_xy[right_idx]))
        if pair_dist > body_scale * trigger_ratio and not overlap_mode:
            continue

        target_dist = min(pair_dist, body_scale * max_ratio)
        if overlap_mode:
            target_dist = min(target_dist, body_scale * (max_ratio * 0.86))
        target_dist = max(target_dist, body_scale * min_ratio)
        compress_pair_distance(out_xy, left_idx, right_idx, float(target_dist))

    return out_xy, out_conf


def adaptive_style(box_xyxy):
    if box_xyxy is None:
        return 6, 3, 6, 4
    body_span = max(float(box_xyxy[2] - box_xyxy[0]), float(box_xyxy[3] - box_xyxy[1]), 60.0)
    scale = float(np.clip(body_span / 150.0, 0.80, 1.35))
    glow_thick = max(4, int(round(6 * scale)))
    core_thick = max(2, int(round(2.5 * scale)))
    glow_radius = max(5, int(round(6 * scale)))
    core_radius = max(3, int(round(3.5 * scale)))
    return glow_thick, core_thick, glow_radius, core_radius


def draw_hud(frame, text_left, text_right=None):
    overlay = frame.copy()
    cv2.rectangle(overlay, (16, 16), (860, 86 if text_right else 58), TEXT_BG_COLOR, -1)
    frame[:] = cv2.addWeighted(overlay, 0.32, frame, 0.68, 0.0)
    cv2.putText(frame, text_left, (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.85, TEXT_COLOR, 2, cv2.LINE_AA)
    if text_right:
        cv2.putText(frame, text_right, (28, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_COLOR, 2, cv2.LINE_AA)
    return frame


# 全局状态锁（防抖机制）
_teaching_state = {
    'center_of_gravity': {'text': '', 'advice': '', 'lock': 0},
    'knee_angle': {'text': '', 'advice': '', 'lock': 0},
    'body_rotation': {'text': '', 'advice': '', 'lock': 0},
    'forward_lean': {'text': '', 'advice': '', 'lock': 0},
    'gaze_direction': {'text': '', 'advice': '', 'lock': 0},
    'lock_duration': 50
}

def cv2_to_pil(cv2_img):
    """OpenCV BGR -> PIL RGB"""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_img):
    """PIL RGB -> OpenCV BGR"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_advanced_visuals(frame, trajectory_points, toe_frames, heel_frames, current_frame, keypoints_xy=None, keypoints_conf=None):
    """教学化重构：5大核心指标面板（仅Advanced模式）"""
    global _teaching_state
    h, w = frame.shape[:2]

    if keypoints_xy is None or keypoints_conf is None:
        return frame

    # 转换为PIL处理中文
    pil_img = cv2_to_pil(frame)
    draw = ImageDraw.Draw(pil_img, 'RGBA')

    try:
        font = ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', 24)
        font_small = ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', 20)
    except:
        font = ImageFont.load_default()
        font_small = font

    indicators = []

    # 1. 重心检测
    if keypoints_conf[11] > 0.3 and keypoints_conf[12] > 0.3 and keypoints_conf[15] > 0.3 and keypoints_conf[16] > 0.3:
        hip_x = (keypoints_xy[11, 0] + keypoints_xy[12, 0]) / 2
        ankle_x = (keypoints_xy[15, 0] + keypoints_xy[16, 0]) / 2
        diff = hip_x - ankle_x

        if diff < -20:
            status, advice = "重心偏向板尾", "尝试将体重均匀分布于双脚"
        elif diff > 20:
            status, advice = "重心偏向板头", "适度后移，保持平衡"
        else:
            status, advice = "重心居中稳定", "保持当前重心，平稳滑行"

        if _teaching_state['center_of_gravity']['lock'] > 0:
            _teaching_state['center_of_gravity']['lock'] -= 1
        elif status != _teaching_state['center_of_gravity']['text']:
            _teaching_state['center_of_gravity'] = {'text': status, 'advice': advice, 'lock': _teaching_state['lock_duration']}

        if _teaching_state['center_of_gravity']['text']:
            indicators.append(('center_of_gravity', _teaching_state['center_of_gravity']['text'], _teaching_state['center_of_gravity']['advice']))

    # 2. 膝盖角度
    if keypoints_conf[11] > 0.3 and keypoints_conf[13] > 0.3 and keypoints_conf[15] > 0.3:
        hip, knee, ankle = keypoints_xy[11], keypoints_xy[13], keypoints_xy[15]
        v1, v2 = hip - knee, ankle - knee
        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1)))

        if angle > 150:
            status, advice = "膝关节接近直立", "适度屈膝，为减震留出空间"
        elif angle < 100:
            status, advice = "下肢保持深蹲", "可适当放松，避免过度疲劳"
        else:
            status, advice = "下肢保持屈膝", "维持良好站姿，感受板刃抓地"

        if _teaching_state['knee_angle']['lock'] > 0:
            _teaching_state['knee_angle']['lock'] -= 1
        elif status != _teaching_state['knee_angle']['text']:
            _teaching_state['knee_angle'] = {'text': status, 'advice': advice, 'lock': _teaching_state['lock_duration']}

        if _teaching_state['knee_angle']['text']:
            indicators.append(('knee_angle', _teaching_state['knee_angle']['text'], _teaching_state['knee_angle']['advice']))

    # 3. 身体旋转
    if keypoints_conf[5] > 0.3 and keypoints_conf[6] > 0.3 and keypoints_conf[11] > 0.3 and keypoints_conf[12] > 0.3:
        shoulder_slope = (keypoints_xy[6, 1] - keypoints_xy[5, 1]) / (keypoints_xy[6, 0] - keypoints_xy[5, 0] + 1e-6)
        hip_slope = (keypoints_xy[12, 1] - keypoints_xy[11, 1]) / (keypoints_xy[12, 0] - keypoints_xy[11, 0] + 1e-6)

        if abs(shoulder_slope - hip_slope) > 0.3:
            status, advice = "肩部发生相对旋转", "保持上半身安静，随板身移动"
        else:
            status, advice = "上下身保持随动", "核心稳定，姿态控制良好"

        if _teaching_state['body_rotation']['lock'] > 0:
            _teaching_state['body_rotation']['lock'] -= 1
        elif status != _teaching_state['body_rotation']['text']:
            _teaching_state['body_rotation'] = {'text': status, 'advice': advice, 'lock': _teaching_state['lock_duration']}

        if _teaching_state['body_rotation']['text']:
            indicators.append(('body_rotation', _teaching_state['body_rotation']['text'], _teaching_state['body_rotation']['advice']))

    # 4. 弯腰检测
    if keypoints_conf[5] > 0.3 and keypoints_conf[6] > 0.3 and keypoints_conf[11] > 0.3 and keypoints_conf[12] > 0.3:
        shoulder_y = (keypoints_xy[5, 1] + keypoints_xy[6, 1]) / 2
        hip_y = (keypoints_xy[11, 1] + keypoints_xy[12, 1]) / 2
        torso_length = hip_y - shoulder_y

        if torso_length < 80:
            status, advice = "上半身过度前倾(折髋)", "挺胸收腹，通过屈膝来降低重心"
        else:
            status, advice = "上半身姿态挺拔", "核心收紧，保持优良的上半身框架"

        if _teaching_state['forward_lean']['lock'] > 0:
            _teaching_state['forward_lean']['lock'] -= 1
        elif status != _teaching_state['forward_lean']['text']:
            _teaching_state['forward_lean'] = {'text': status, 'advice': advice, 'lock': _teaching_state['lock_duration']}

        if _teaching_state['forward_lean']['text']:
            indicators.append(('forward_lean', _teaching_state['forward_lean']['text'], _teaching_state['forward_lean']['advice']))

    # 5. 视线检测
    if keypoints_conf[0] > 0.3 and keypoints_conf[5] > 0.3 and keypoints_conf[6] > 0.3:
        nose_y = keypoints_xy[0, 1]
        shoulder_y = (keypoints_xy[5, 1] + keypoints_xy[6, 1]) / 2

        if nose_y > shoulder_y + 30:
            status, advice = "视线向下注视雪板", "请抬起头，视线看向前方或出弯方向"
        else:
            status, advice = "视线朝向滑行轨迹", "保持视线引导，提前观察下一个弯道"

        if _teaching_state['gaze_direction']['lock'] > 0:
            _teaching_state['gaze_direction']['lock'] -= 1
        elif status != _teaching_state['gaze_direction']['text']:
            _teaching_state['gaze_direction'] = {'text': status, 'advice': advice, 'lock': _teaching_state['lock_duration']}

        if _teaching_state['gaze_direction']['text']:
            indicators.append(('gaze_direction', _teaching_state['gaze_direction']['text'], _teaching_state['gaze_direction']['advice']))

    # 绘制教学面板（左下角）
    if indicators:
        panel_x, panel_y = 30, h - 250
        panel_w, panel_h = 650, 220

        # 半透明深色底板
        draw.rectangle([(panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h)],
                      fill=(30, 30, 30, 128))

        # 绘制指标文字
        y_offset = panel_y + 20
        for idx, (key, status, advice) in enumerate(indicators[:3]):  # 最多显示3个
            # 状态（白色）
            draw.text((panel_x + 15, y_offset), f"[状态] {status}",
                     font=font, fill=(255, 255, 255, 255))
            # 建议（亮黄色）
            draw.text((panel_x + 15, y_offset + 35), f"[建议] {advice}",
                     font=font_small, fill=(255, 220, 100, 255))
            y_offset += 70

    return pil_to_cv2(pil_img)


def draw_pose_overlay(frame, keypoints_xy, keypoints_conf, box_xyxy=None):
    if keypoints_xy is None or keypoints_conf is None:
        return frame
    if box_xyxy is None:
        box_xyxy = estimate_box_from_keypoints(keypoints_xy, keypoints_conf)
    glow_thick, core_thick, glow_radius, core_radius = adaptive_style(box_xyxy)

    output = frame.copy()
    glow = frame.copy()

    for start, end in SKELETON:
        if keypoints_conf[start] >= KEYPOINT_DRAW_CONF_THRESHOLD and keypoints_conf[end] >= KEYPOINT_DRAW_CONF_THRESHOLD:
            p1 = tuple(np.round(keypoints_xy[start]).astype(int))
            p2 = tuple(np.round(keypoints_xy[end]).astype(int))
            cv2.line(glow, p1, p2, BONE_GLOW_COLOR, glow_thick, cv2.LINE_AA)

    for idx in range(len(keypoints_xy)):
        if keypoints_conf[idx] >= KEYPOINT_DRAW_CONF_THRESHOLD:
            center = tuple(np.round(keypoints_xy[idx]).astype(int))
            cv2.circle(glow, center, glow_radius, JOINT_GLOW_COLOR, -1, cv2.LINE_AA)

    output = cv2.addWeighted(glow, 0.28, output, 0.72, 0.0)

    for start, end in SKELETON:
        if keypoints_conf[start] >= KEYPOINT_DRAW_CONF_THRESHOLD and keypoints_conf[end] >= KEYPOINT_DRAW_CONF_THRESHOLD:
            p1 = tuple(np.round(keypoints_xy[start]).astype(int))
            p2 = tuple(np.round(keypoints_xy[end]).astype(int))
            cv2.line(output, p1, p2, BONE_CORE_COLOR, core_thick, cv2.LINE_AA)

    for idx in range(len(keypoints_xy)):
        if keypoints_conf[idx] >= KEYPOINT_DRAW_CONF_THRESHOLD:
            center = tuple(np.round(keypoints_xy[idx]).astype(int))
            cv2.circle(output, center, core_radius, JOINT_CORE_COLOR, -1, cv2.LINE_AA)

    return output


def render_from_keypoints(
    input_path: Path,
    output_path: Path,
    compare_output_path: Path,
    keypoints_arr: np.ndarray,
    model_label: str,
    track_boxes: np.ndarray | None = None,
    slow_overlay_path: Path | None = None,
    slow_compare_path: Path | None = None,
    slow_factor: float = 0.30,
):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频：{input_path}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 性能优化：输出尺寸匹配降维后的分辨率
    if max(width, height) > 1080:
        scale = 1080.0 / max(width, height)
        width = int(width * scale)
        height = int(height * scale)

    frame_limit = len(keypoints_arr)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    overlay_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    compare_writer = cv2.VideoWriter(str(compare_output_path), fourcc, fps, (width * 2, height))
    if not overlay_writer.isOpened() or not compare_writer.isOpened():
        cap.release()
        raise RuntimeError('无法初始化输出视频 writer')

    slow_overlay_writer = None
    slow_compare_writer = None
    if slow_overlay_path is not None and slow_compare_path is not None and 0.0 < slow_factor < 1.0:
        slow_fps = max(1.0, fps * slow_factor)
        slow_overlay_writer = cv2.VideoWriter(str(slow_overlay_path), fourcc, slow_fps, (width, height))
        slow_compare_writer = cv2.VideoWriter(str(slow_compare_path), fourcc, slow_fps, (width * 2, height))
        if not slow_overlay_writer.isOpened() or not slow_compare_writer.isOpened():
            cap.release()
            overlay_writer.release()
            compare_writer.release()
            raise RuntimeError('无法初始化慢速输出视频 writer')

    with tqdm(total=frame_limit, desc='Render Overlay', unit='frame') as pbar:
        frame_idx = 0
        # 视觉重构：状态追踪变量
        toe_frames = 0  # 前刃帧数
        heel_frames = 0  # 后刃帧数
        trajectory_points = []  # 重心轨迹点 [(x, y), ...]

        while frame_idx < frame_limit:
            ok, frame = cap.read()
            if not ok:
                break

            # 性能优化：降分辨率（长边压缩至1080px）
            h, w = frame.shape[:2]
            if max(h, w) > 1080:
                scale = 1080.0 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            original_frame = frame.copy()
            kp_frame = keypoints_arr[frame_idx]
            xy = kp_frame[:, :2].astype(np.float32)
            conf = kp_frame[:, 2].astype(np.float32)
            if track_boxes is not None and frame_idx < len(track_boxes):
                box = np.asarray(track_boxes[frame_idx], dtype=np.float32)
            else:
                box = estimate_box_from_keypoints(xy, conf, conf_thresh=0.18)

            if box is not None:
                draw_xy, draw_conf = prepare_render_keypoints(xy, conf, box)
                overlay_frame = draw_pose_overlay(frame, draw_xy, draw_conf, box)

                # 视觉重构：提取重心坐标（骨盆中点：关键点11和12的中点）
                if conf[11] > 0.3 and conf[12] > 0.3:
                    hip_center_x = (xy[11, 0] + xy[12, 0]) / 2
                    hip_center_y = (xy[11, 1] + xy[12, 1]) / 2
                    trajectory_points.append((hip_center_x, hip_center_y))

                    # 简单前后刃判断：基于脚踝Y坐标差异（左脚踝15，右脚踝16）
                    if conf[15] > 0.3 and conf[16] > 0.3:
                        ankle_diff = xy[15, 1] - xy[16, 1]
                        if abs(ankle_diff) > 5:
                            if ankle_diff > 0:
                                toe_frames += 1
                            else:
                                heel_frames += 1

                # 视觉重构：绘制三大功能
                overlay_frame = draw_advanced_visuals(overlay_frame, trajectory_points, toe_frames, heel_frames, frame_idx, keypoints_xy=xy, keypoints_conf=conf)
            else:
                overlay_frame = frame.copy()

            original_frame = draw_hud(original_frame, 'Original Skiing Video')
            overlay_frame = draw_hud(overlay_frame, 'ByteTrack + RTMPose-X Overlay', f'Pose: {model_label}')
            side_by_side = cv2.hconcat([original_frame, overlay_frame])
            overlay_writer.write(overlay_frame)
            compare_writer.write(side_by_side)
            if slow_overlay_writer is not None and slow_compare_writer is not None:
                slow_overlay_writer.write(overlay_frame)
                slow_compare_writer.write(side_by_side)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    overlay_writer.release()
    compare_writer.release()
    if slow_overlay_writer is not None:
        slow_overlay_writer.release()
    if slow_compare_writer is not None:
        slow_compare_writer.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output)
    compare_output_path = resolve_project_path(args.compare_output)
    slow_output_path = resolve_project_path(args.slow_output) if args.slow_output else None
    slow_compare_output_path = resolve_project_path(args.slow_compare_output) if args.slow_compare_output else None
    track_cache_path = resolve_project_path(args.track_cache)
    keypoint_cache_path = resolve_project_path(args.keypoint_cache)
    seed_path = resolve_project_path(args.seed_kp)
    local_bbx_path = resolve_project_path(args.local_bbx)
    detector_model_path = resolve_project_path(args.detector_model)
    realesrgan_model_path = resolve_project_path(args.realesrgan_model)
    tracker_config_path = resolve_project_path(args.tracker_config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compare_output_path.parent.mkdir(parents=True, exist_ok=True)
    if slow_output_path is not None:
        slow_output_path.parent.mkdir(parents=True, exist_ok=True)
    if slow_compare_output_path is not None:
        slow_compare_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f'输入视频不存在：{input_path.resolve()}')
    if not detector_model_path.exists():
        raise FileNotFoundError(f'检测模型不存在：{detector_model_path.resolve()}')
    if not tracker_config_path.exists():
        raise FileNotFoundError(f'ByteTrack 配置不存在：{tracker_config_path.resolve()}')

    seed_arr = load_precomputed_keypoints(seed_path) if seed_path.exists() else None
    local_bbx_xys = load_local_bbox(local_bbx_path) if local_bbx_path.exists() else None
    sr_model = None if args.disable_sr else load_sr_model(realesrgan_model_path)

    track_boxes, found_mask, crowd_mask = track_target_bboxes(
        input_path=input_path,
        detector_model_path=detector_model_path,
        tracker_config_path=tracker_config_path,
        cache_path=track_cache_path,
        local_bbx_xys=local_bbx_xys,
        imgsz=args.imgsz,
        conf=args.conf,
        max_frames=args.max_frames,
        force=args.force_redo_track,
    )

    raw_keypoints_arr = generate_keypoints(
        input_path=input_path,
        track_boxes=track_boxes,
        crowd_mask=crowd_mask,
        keypoint_cache_path=keypoint_cache_path,
        pose_model_ref=args.rtmpose_model,
        seed_arr=seed_arr,
        sr_model=sr_model,
        small_box_threshold=args.small_box_threshold,
        max_frames=args.max_frames,
        force=args.force_redo_pose,
    )

    if args.max_frames and args.max_frames > 0:
        raw_keypoints_arr = raw_keypoints_arr[: int(args.max_frames)]
        track_boxes = track_boxes[: int(args.max_frames)]
        found_mask = found_mask[: int(args.max_frames)]
        crowd_mask = crowd_mask[: int(args.max_frames)]

    keypoints_arr = offline_temporal_refine(raw_keypoints_arr.astype(np.float32), track_boxes[: len(raw_keypoints_arr)])
    keypoints_arr = reduce_temporal_lag(keypoints_arr.astype(np.float32), raw_keypoints_arr.astype(np.float32), track_boxes[: len(raw_keypoints_arr)])
    keypoints_arr = repair_brief_outliers(keypoints_arr.astype(np.float32), raw_keypoints_arr.astype(np.float32), track_boxes[: len(raw_keypoints_arr)])
    keypoints_arr = enforce_track_box_consistency(keypoints_arr.astype(np.float32), track_boxes[: len(raw_keypoints_arr)])
    save_tensor_dict(keypoint_cache_path, {
        'raw_keypoints': torch.from_numpy(raw_keypoints_arr.astype(np.float32)),
        'keypoints': torch.from_numpy(keypoints_arr.astype(np.float32)),
        'track_boxes': torch.from_numpy(track_boxes[: len(raw_keypoints_arr)].astype(np.float32)),
        'found_mask': torch.from_numpy(found_mask[: len(raw_keypoints_arr)].astype(np.bool_)),
        'crowd_mask': torch.from_numpy(crowd_mask[: len(raw_keypoints_arr)].astype(np.bool_)),
    })

    render_from_keypoints(
        input_path,
        output_path,
        compare_output_path,
        keypoints_arr,
        model_label='rtmpose-x + track-box constraint + temporal refine',
        track_boxes=track_boxes[: len(keypoints_arr)],
        slow_overlay_path=slow_output_path,
        slow_compare_path=slow_compare_output_path,
        slow_factor=args.slow_factor,
    )

    print('[Success] 高精度骨骼叠加视频已生成，请验收输出文件。')
    print(f'[Overlay] {output_path.resolve()}')
    print(f'[Compare] {compare_output_path.resolve()}')
    if slow_output_path is not None and 0.0 < args.slow_factor < 1.0:
        print(f'[Slow Overlay] {slow_output_path.resolve()}')
    if slow_compare_output_path is not None and 0.0 < args.slow_factor < 1.0:
        print(f'[Slow Compare] {slow_compare_output_path.resolve()}')


if __name__ == '__main__':
    main()
