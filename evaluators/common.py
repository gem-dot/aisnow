from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

JOINT_NAMES = (
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
)
JOINT_INDEX = {name: idx for idx, name in enumerate(JOINT_NAMES)}


def as_numpy(value, dtype=np.float32):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(dtype)
    return np.asarray(value, dtype=dtype)


def load_keypoint_payload(cache_path: Path) -> dict[str, np.ndarray | None]:
    payload = torch.load(cache_path, map_location='cpu')
    if not isinstance(payload, dict):
        raise RuntimeError(f'关键点缓存格式错误: {cache_path}')
    keypoints = as_numpy(payload.get('keypoints'))
    raw_keypoints = as_numpy(payload.get('raw_keypoints'))
    track_boxes = as_numpy(payload.get('track_boxes'))
    found_mask = as_numpy(payload.get('found_mask'), dtype=np.bool_)
    crowd_mask = as_numpy(payload.get('crowd_mask'), dtype=np.bool_)
    if keypoints is None and raw_keypoints is None:
        raise RuntimeError(f'缓存缺少 keypoints / raw_keypoints: {cache_path}')
    if keypoints is None:
        keypoints = raw_keypoints.copy()
    if raw_keypoints is None:
        raw_keypoints = keypoints.copy()
    frame_count = len(keypoints)
    if found_mask is None:
        found_mask = np.ones((frame_count,), dtype=bool)
    if crowd_mask is None:
        crowd_mask = np.zeros((frame_count,), dtype=bool)
    if track_boxes is not None:
        frame_count = min(frame_count, len(track_boxes))
    frame_count = min(frame_count, len(found_mask), len(crowd_mask), len(raw_keypoints))
    return {
        'keypoints': keypoints[:frame_count].astype(np.float32),
        'raw_keypoints': raw_keypoints[:frame_count].astype(np.float32),
        'track_boxes': None if track_boxes is None else track_boxes[:frame_count].astype(np.float32),
        'found_mask': found_mask[:frame_count].astype(bool),
        'crowd_mask': crowd_mask[:frame_count].astype(bool),
    }


def adapt_keypoints_to_dict(keypoints_row: np.ndarray) -> dict[str, dict[str, float]]:
    row = np.asarray(keypoints_row, dtype=np.float32)
    mapping: dict[str, dict[str, float]] = {}
    for name, idx in JOINT_INDEX.items():
        if idx < len(row):
            x = float(row[idx][0])
            y = float(row[idx][1])
            conf = float(row[idx][2]) if row.shape[1] >= 3 else 0.0
        else:
            x = float('nan')
            y = float('nan')
            conf = 0.0
        mapping[name] = {'x': x, 'y': y, 'conf': conf}
    return mapping


def weighted_mean_1d(values, weights, min_conf: float = 0.18) -> float:
    vals = np.asarray(values, dtype=np.float32)
    w = np.clip(np.asarray(weights, dtype=np.float32), 0.0, 1.0)
    valid = np.isfinite(vals) & (w >= min_conf)
    if int(valid.sum()) == 0:
        return float('nan')
    return float(np.average(vals[valid], weights=w[valid]))


def point_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    point_a = np.asarray(point_a, dtype=np.float32)
    point_b = np.asarray(point_b, dtype=np.float32)
    if not np.isfinite(point_a).all() or not np.isfinite(point_b).all():
        return float('nan')
    return float(np.linalg.norm(point_a - point_b))


def knee_angle_deg(hip_xy: np.ndarray, knee_xy: np.ndarray, ankle_xy: np.ndarray) -> float:
    hip_xy = np.asarray(hip_xy, dtype=np.float32)
    knee_xy = np.asarray(knee_xy, dtype=np.float32)
    ankle_xy = np.asarray(ankle_xy, dtype=np.float32)
    if not np.isfinite(hip_xy).all() or not np.isfinite(knee_xy).all() or not np.isfinite(ankle_xy).all():
        return float('nan')
    vector_a = hip_xy - knee_xy
    vector_b = ankle_xy - knee_xy
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a < 1e-5 or norm_b < 1e-5:
        return float('nan')
    cos_theta = float(np.dot(vector_a, vector_b) / max(norm_a * norm_b, 1e-6))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def interp_nan_1d(values: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    indices = np.arange(len(values), dtype=np.float32)
    valid = np.isfinite(values)
    if int(valid.sum()) == 0:
        return np.full_like(values, float(fill_value), dtype=np.float32)
    if int(valid.sum()) == 1:
        return np.full_like(values, float(values[valid][0]), dtype=np.float32)
    return np.interp(indices, indices[valid], values[valid]).astype(np.float32)


def interp_nan_2d(values_xy: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    values_xy = np.asarray(values_xy, dtype=np.float32)
    if values_xy.size == 0:
        return values_xy.astype(np.float32)
    out = values_xy.copy()
    out[:, 0] = interp_nan_1d(out[:, 0], fill_value=fill_value)
    out[:, 1] = interp_nan_1d(out[:, 1], fill_value=fill_value)
    return out.astype(np.float32)


def joint_xy(keypoints_dict: dict[str, dict[str, float]], name: str, conf_min: float = 0.18) -> np.ndarray:
    item = keypoints_dict.get(name, {})
    conf = float(item.get('conf', 0.0) or 0.0)
    if conf < conf_min:
        return np.asarray([np.nan, np.nan], dtype=np.float32)
    return np.asarray([float(item.get('x', float('nan'))), float(item.get('y', float('nan')))], dtype=np.float32)


def probe_video_metadata(video_path: Path) -> tuple[float, int]:
    fps = 30.0
    frame_count = 0
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        fps_value = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count_value = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps_value > 1e-6:
            fps = fps_value
        frame_count = max(frame_count_value, 0)
    cap.release()
    return fps, frame_count
