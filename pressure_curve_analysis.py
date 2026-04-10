#!/usr/bin/env python3
# pip install numpy scipy matplotlib torch opencv-python
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import find_peaks, savgol_filter

JOINT = {
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}

ANALYSIS_MIN_CONF = 0.18
HUD_FONT_PATH: str | None = None


def configure_plot_fonts() -> None:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [
        'PingFang HK',
        'Hiragino Sans GB',
        'Songti SC',
        'Arial Unicode MS',
        'STHeiti',
        'Heiti TC',
        'DejaVu Sans',
    ]
    plt.rcParams['axes.unicode_minus'] = False


def get_hud_font_path() -> str | None:
    global HUD_FONT_PATH
    if HUD_FONT_PATH is not None:
        return HUD_FONT_PATH
    preferred = {'PingFang HK', 'Hiragino Sans GB', 'Songti SC', 'Arial Unicode MS', 'STHeiti', 'Heiti TC'}
    for item in font_manager.fontManager.ttflist:
        if item.name in preferred:
            HUD_FONT_PATH = item.fname
            return HUD_FONT_PATH
    return None


def load_hud_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = get_hud_font_path()
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


@dataclass(frozen=True)
class TurnSegment:
    index: int
    start: int
    apex: int
    end: int
    direction: str


@dataclass(frozen=True)
class AnalysisWindow:
    start: int
    end: int


def as_numpy(value, dtype=np.float32):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(dtype)
    return np.asarray(value, dtype=dtype)


def load_payload(cache_path: Path) -> dict[str, np.ndarray | None]:
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


def weighted_center(points_xy: np.ndarray, conf: np.ndarray, min_conf: float = ANALYSIS_MIN_CONF) -> np.ndarray:
    weights = np.clip(np.asarray(conf, dtype=np.float32), 0.0, 1.0)
    valid = np.isfinite(points_xy).all(axis=1) & (weights >= min_conf)
    if int(valid.sum()) == 0:
        return np.asarray([np.nan, np.nan], dtype=np.float32)
    pts = np.asarray(points_xy[valid], dtype=np.float32)
    w = weights[valid]
    return (pts * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-6)


def point_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    if not np.isfinite(point_a).all() or not np.isfinite(point_b).all():
        return float('nan')
    return float(np.linalg.norm(point_a - point_b))


def knee_angle_deg(hip_xy: np.ndarray, knee_xy: np.ndarray, ankle_xy: np.ndarray) -> float:
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


def interp_nan_2d(values_xy: np.ndarray) -> np.ndarray:
    out = np.asarray(values_xy, dtype=np.float32).copy()
    out[:, 0] = interp_nan_1d(out[:, 0])
    out[:, 1] = interp_nan_1d(out[:, 1])
    return out.astype(np.float32)


def smooth_signal(values: np.ndarray, target_window: int = 21, polyorder: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    size = len(values)
    if size < 5:
        return values.copy()
    window = min(target_window, size if size % 2 == 1 else size - 1)
    if window < 5:
        return values.copy()
    poly = min(polyorder, window - 2)
    return savgol_filter(values, window_length=window, polyorder=poly, mode='interp').astype(np.float32)


def robust_normalize(values: np.ndarray, low_q: float = 0.05, high_q: float = 0.95) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo = float(np.nanquantile(values, low_q))
    hi = float(np.nanquantile(values, high_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-6:
        return np.full_like(values, 0.5, dtype=np.float32)
    out = (values - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def valid_count(keypoints: np.ndarray, threshold: float = ANALYSIS_MIN_CONF) -> np.ndarray:
    return (keypoints[:, :, 2] >= threshold).sum(axis=1).astype(np.int32)


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask.astype(bool)):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def bridge_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    bridged = mask.astype(bool).copy()
    if max_gap <= 0 or len(bridged) == 0:
        return bridged
    false_runs = contiguous_true_runs(~bridged)
    for start, end in false_runs:
        if start == 0 or end == len(bridged) - 1:
            continue
        if (end - start + 1) <= max_gap and bridged[start - 1] and bridged[end + 1]:
            bridged[start:end + 1] = True
    return bridged


def select_analysis_window(
    found_mask: np.ndarray,
    crowd_mask: np.ndarray,
    quality_mask: np.ndarray,
    min_frames: int,
) -> AnalysisWindow:
    usable_mask = found_mask.astype(bool) & quality_mask.astype(bool)
    usable_indices = np.flatnonzero(usable_mask)
    if usable_indices.size > 0:
        span_start = int(usable_indices[0])
        span_end = int(usable_indices[-1])
        span_length = span_end - span_start + 1
        span_coverage = float(usable_mask[span_start:span_end + 1].mean())
        if span_length >= min_frames and span_coverage >= 0.72:
            return AnalysisWindow(start=span_start, end=span_end)

    bridged_usable = bridge_short_gaps(usable_mask, max_gap=max(6, int(round(min_frames * 0.15))))
    usable_runs = contiguous_true_runs(bridged_usable)
    usable_runs = [run for run in usable_runs if run[1] - run[0] + 1 >= min_frames]
    if usable_runs:
        start, end = min(usable_runs, key=lambda run: (run[0], -(run[1] - run[0] + 1)))
        return AnalysisWindow(start=int(start), end=int(end))

    non_crowd_mask = usable_mask & (~crowd_mask.astype(bool))
    bridged_non_crowd = bridge_short_gaps(non_crowd_mask, max_gap=max(4, int(round(min_frames * 0.08))))
    runs = contiguous_true_runs(bridged_non_crowd)
    runs = [run for run in runs if run[1] - run[0] + 1 >= max(24, min_frames // 2)]
    if runs:
        start, end = min(runs, key=lambda run: (run[0], -(run[1] - run[0] + 1)))
        return AnalysisWindow(start=int(start), end=int(end))

    fallback_runs = contiguous_true_runs(usable_mask)
    fallback_runs = [run for run in fallback_runs if run[1] - run[0] + 1 >= max(24, min_frames // 2)]
    if fallback_runs:
        start, end = min(fallback_runs, key=lambda run: (run[0], -(run[1] - run[0] + 1)))
        return AnalysisWindow(start=int(start), end=int(end))

    return AnalysisWindow(start=0, end=len(found_mask) - 1)


def build_kinematic_signals(keypoints: np.ndarray, track_boxes: np.ndarray | None) -> dict[str, np.ndarray]:
    frame_count = len(keypoints)
    pelvis_xy = np.full((frame_count, 2), np.nan, dtype=np.float32)
    knee_center_xy = np.full((frame_count, 2), np.nan, dtype=np.float32)
    ankle_center_xy = np.full((frame_count, 2), np.nan, dtype=np.float32)
    leg_chain = np.full((frame_count,), np.nan, dtype=np.float32)
    support_extension = np.full((frame_count,), np.nan, dtype=np.float32)
    vertical_gap = np.full((frame_count,), np.nan, dtype=np.float32)
    pelvis_box_gap = np.full((frame_count,), np.nan, dtype=np.float32)
    knee_angle = np.full((frame_count,), np.nan, dtype=np.float32)
    lean_signal = np.full((frame_count,), np.nan, dtype=np.float32)
    body_scale = np.full((frame_count,), np.nan, dtype=np.float32)

    for frame_idx in range(frame_count):
        frame_kp = keypoints[frame_idx]
        xy = frame_kp[:, :2].astype(np.float32)
        conf = frame_kp[:, 2].astype(np.float32)

        left_hip_xy = xy[JOINT['left_hip']]
        right_hip_xy = xy[JOINT['right_hip']]
        left_knee_xy = xy[JOINT['left_knee']]
        right_knee_xy = xy[JOINT['right_knee']]
        left_ankle_xy = xy[JOINT['left_ankle']]
        right_ankle_xy = xy[JOINT['right_ankle']]

        pelvis_xy[frame_idx] = weighted_center(
            np.stack([left_hip_xy, right_hip_xy], axis=0),
            np.asarray([conf[JOINT['left_hip']], conf[JOINT['right_hip']]], dtype=np.float32),
        )
        knee_center_xy[frame_idx] = weighted_center(
            np.stack([left_knee_xy, right_knee_xy], axis=0),
            np.asarray([conf[JOINT['left_knee']], conf[JOINT['right_knee']]], dtype=np.float32),
        )
        ankle_center_xy[frame_idx] = weighted_center(
            np.stack([left_ankle_xy, right_ankle_xy], axis=0),
            np.asarray([conf[JOINT['left_ankle']], conf[JOINT['right_ankle']]], dtype=np.float32),
        )

        left_chain = point_distance(left_hip_xy, left_knee_xy) + point_distance(left_knee_xy, left_ankle_xy)
        right_chain = point_distance(right_hip_xy, right_knee_xy) + point_distance(right_knee_xy, right_ankle_xy)
        chain_values = [value for value in (left_chain, right_chain) if np.isfinite(value) and value > 1e-3]
        if chain_values:
            leg_chain[frame_idx] = float(np.mean(chain_values))

        left_angle = knee_angle_deg(left_hip_xy, left_knee_xy, left_ankle_xy)
        right_angle = knee_angle_deg(right_hip_xy, right_knee_xy, right_ankle_xy)
        angle_values = [value for value in (left_angle, right_angle) if np.isfinite(value)]
        if angle_values:
            knee_angle[frame_idx] = float(np.mean(angle_values))

        if np.isfinite(pelvis_xy[frame_idx]).all() and np.isfinite(ankle_center_xy[frame_idx]).all():
            support_extension[frame_idx] = point_distance(pelvis_xy[frame_idx], ankle_center_xy[frame_idx])
            vertical_gap[frame_idx] = max(float(ankle_center_xy[frame_idx][1] - pelvis_xy[frame_idx][1]), 0.0)

        if track_boxes is not None and frame_idx < len(track_boxes):
            box = np.asarray(track_boxes[frame_idx], dtype=np.float32)
            body_scale[frame_idx] = max(float(box[3] - box[1]), float(box[2] - box[0]), 1.0)
            if np.isfinite(pelvis_xy[frame_idx]).all():
                pelvis_box_gap[frame_idx] = max(float(box[3] - pelvis_xy[frame_idx][1]), 0.0)

    pelvis_xy = interp_nan_2d(pelvis_xy)
    knee_center_xy = interp_nan_2d(knee_center_xy)
    ankle_center_xy = interp_nan_2d(ankle_center_xy)
    leg_chain = interp_nan_1d(leg_chain, fill_value=float(np.nanmedian(leg_chain[np.isfinite(leg_chain)])) if np.isfinite(leg_chain).any() else 1.0)
    knee_angle = interp_nan_1d(knee_angle, fill_value=170.0)
    support_extension = interp_nan_1d(support_extension, fill_value=float(np.nanmedian(support_extension[np.isfinite(support_extension)])) if np.isfinite(support_extension).any() else 1.0)
    vertical_gap = interp_nan_1d(vertical_gap, fill_value=float(np.nanmedian(vertical_gap[np.isfinite(vertical_gap)])) if np.isfinite(vertical_gap).any() else 1.0)
    body_scale = interp_nan_1d(body_scale, fill_value=float(np.nanmedian(body_scale[np.isfinite(body_scale)])) if np.isfinite(body_scale).any() else 1.0)
    pelvis_box_gap = interp_nan_1d(pelvis_box_gap, fill_value=float(np.nanmedian(pelvis_box_gap[np.isfinite(pelvis_box_gap)])) if np.isfinite(pelvis_box_gap).any() else float(np.nanmedian(vertical_gap[np.isfinite(vertical_gap)])) if np.isfinite(vertical_gap).any() else 1.0)

    scale = np.maximum(leg_chain, body_scale * 0.32)
    scale = np.maximum(scale, 1.0)
    lean_signal = (pelvis_xy[:, 0] - ankle_center_xy[:, 0]) / scale
    support_extension_norm = support_extension / scale
    vertical_gap_norm = vertical_gap / scale
    pelvis_box_gap_norm = pelvis_box_gap / scale

    return {
        'pelvis_xy': pelvis_xy,
        'knee_center_xy': knee_center_xy,
        'ankle_center_xy': ankle_center_xy,
        'leg_chain': leg_chain.astype(np.float32),
        'body_scale': body_scale.astype(np.float32),
        'support_extension_norm': support_extension_norm.astype(np.float32),
        'vertical_gap_norm': vertical_gap_norm.astype(np.float32),
        'pelvis_box_gap_norm': pelvis_box_gap_norm.astype(np.float32),
        'knee_angle_deg': knee_angle.astype(np.float32),
        'lean_signal': lean_signal.astype(np.float32),
    }


def compute_pressure_curve(signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    vertical_gap_norm = smooth_signal(signals['vertical_gap_norm'], target_window=21)
    support_extension_norm = smooth_signal(signals['support_extension_norm'], target_window=25)
    pelvis_box_gap_norm = smooth_signal(signals.get('pelvis_box_gap_norm', signals['vertical_gap_norm']), target_window=25)
    knee_angle_deg = smooth_signal(signals['knee_angle_deg'], target_window=21)
    lean_signal = smooth_signal(signals['lean_signal'], target_window=31)

    vertical_compression_component = 1.0 - robust_normalize(vertical_gap_norm)
    support_compression_component = 1.0 - robust_normalize(support_extension_norm)
    box_compression_component = 1.0 - robust_normalize(pelvis_box_gap_norm)
    knee_flex_component = np.clip((170.0 - knee_angle_deg) / 95.0, 0.0, 1.0).astype(np.float32)
    knee_flex_component = robust_normalize(smooth_signal(knee_flex_component, target_window=21))

    actual_pressure = (
        0.60 * vertical_compression_component
        + 0.18 * support_compression_component
        + 0.14 * box_compression_component
        + 0.08 * knee_flex_component
    )
    actual_pressure = smooth_signal(actual_pressure.astype(np.float32), target_window=25)
    actual_pressure = robust_normalize(actual_pressure)

    lean_centered = lean_signal - float(np.median(lean_signal))
    lean_centered = smooth_signal(lean_centered.astype(np.float32), target_window=31)

    return {
        'actual_pressure': actual_pressure.astype(np.float32),
        'vertical_compression_component': vertical_compression_component.astype(np.float32),
        'compression_component': support_compression_component.astype(np.float32),
        'box_compression_component': box_compression_component.astype(np.float32),
        'knee_flex_component': knee_flex_component.astype(np.float32),
        'lean_signal': lean_centered.astype(np.float32),
    }


def filter_alternating_apexes(candidates: list[dict[str, float]], min_gap: int) -> list[dict[str, float]]:
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda item: item['frame'])
    filtered: list[dict[str, float]] = []
    for candidate in candidates:
        if not filtered:
            filtered.append(candidate)
            continue
        prev = filtered[-1]
        if candidate['sign'] == prev['sign']:
            if abs(candidate['amplitude']) > abs(prev['amplitude']):
                filtered[-1] = candidate
            continue
        if candidate['frame'] - prev['frame'] < min_gap:
            if abs(candidate['amplitude']) > abs(prev['amplitude']):
                filtered[-1] = candidate
            continue
        filtered.append(candidate)
    return filtered


def choose_boundary(signal: np.ndarray, start: int, end: int) -> int:
    if end <= start:
        return int(start)
    window = np.abs(signal[start:end + 1])
    return int(start + int(np.argmin(window)))


def refine_boundary_with_lean(boundary: int, lean_signal: np.ndarray, search_radius: int) -> int:
    start = max(0, int(boundary - search_radius))
    end = min(len(lean_signal) - 1, int(boundary + search_radius))
    if end <= start:
        return int(boundary)
    local_idx = choose_boundary(lean_signal, start, end)
    if abs(local_idx - int(boundary)) <= max(2, search_radius):
        return int(local_idx)
    return int(boundary)


def boundary_strength(pressure: np.ndarray, boundary: int, radius: int) -> float:
    start = max(0, int(boundary - radius))
    end = min(len(pressure) - 1, int(boundary + radius))
    if end <= start:
        return 0.0
    local = pressure[start:end + 1]
    return float(np.max(local) - np.min(local))


def boundary_lean_features(
    lean_signal: np.ndarray,
    boundary: int,
    side_radius: int,
    neutral_radius: int,
) -> dict[str, float | bool]:
    center_idx = int(np.clip(boundary, 0, len(lean_signal) - 1))
    left_start = max(0, center_idx - int(side_radius))
    right_end = min(len(lean_signal), center_idx + int(side_radius) + 1)
    left_slice = lean_signal[left_start:center_idx]
    right_slice = lean_signal[center_idx + 1:right_end]
    center_slice = np.abs(lean_signal[max(0, center_idx - int(neutral_radius)):min(len(lean_signal), center_idx + int(neutral_radius) + 1)])

    center_value = float(lean_signal[center_idx]) if len(lean_signal) > 0 else 0.0
    left_mean = float(np.median(left_slice)) if len(left_slice) > 0 else center_value
    right_mean = float(np.median(right_slice)) if len(right_slice) > 0 else center_value
    neutral_abs = float(np.min(center_slice)) if len(center_slice) > 0 else abs(center_value)

    left_sign = 0 if abs(left_mean) < 1e-6 else (1 if left_mean > 0.0 else -1)
    right_sign = 0 if abs(right_mean) < 1e-6 else (1 if right_mean > 0.0 else -1)
    sign_flip = left_sign != 0 and right_sign != 0 and left_sign != right_sign
    same_side = left_sign != 0 and right_sign != 0 and left_sign == right_sign

    return {
        'left_mean': left_mean,
        'right_mean': right_mean,
        'neutral_abs': neutral_abs,
        'sign_flip': bool(sign_flip),
        'same_side': bool(same_side),
    }


def fallback_apex_candidates(actual_pressure: np.ndarray, min_gap: int) -> list[dict[str, float]]:
    prominence = max(0.07, float(np.std(actual_pressure) * 0.15))
    peaks, props = find_peaks(actual_pressure, distance=min_gap, prominence=prominence)
    if len(peaks) == 0:
        peak = int(np.argmax(actual_pressure))
        return [{'frame': peak, 'sign': 1.0, 'amplitude': float(actual_pressure[peak]), 'prominence': float(actual_pressure[peak])}]
    out: list[dict[str, float]] = []
    for idx, peak in enumerate(peaks.tolist()):
        prom = float(props['prominences'][idx]) if 'prominences' in props else float(actual_pressure[peak])
        out.append({'frame': int(peak), 'sign': 1.0, 'amplitude': float(actual_pressure[peak]), 'prominence': prom})
    return out


def detect_turn_segments(lean_signal: np.ndarray, actual_pressure: np.ndarray, fps: float) -> list[TurnSegment]:
    frame_count = len(lean_signal)
    if frame_count <= 3:
        return [TurnSegment(index=1, start=0, apex=max(frame_count // 2, 0), end=max(frame_count - 1, 0), direction='unknown')]

    fps_eff = max(float(fps), 1.0)
    lean_smooth = smooth_signal(lean_signal, target_window=max(21, int(round(fps_eff * 0.70)) // 2 * 2 + 1))
    pressure_smooth = smooth_signal(actual_pressure, target_window=max(21, int(round(fps_eff * 0.65)) // 2 * 2 + 1))
    pressure_grad = smooth_signal(np.gradient(pressure_smooth).astype(np.float32), target_window=max(11, int(round(fps_eff * 0.18)) // 2 * 2 + 1))

    min_turn_frames = max(20, int(round(fps_eff * 0.40)))
    min_boundary_gap = max(18, int(round(fps_eff * 0.45)))
    boundary_refine_radius = max(4, int(round(fps_eff * 0.12)))
    boundary_strength_radius = max(6, int(round(fps_eff * 0.18)))
    boundary_side_radius = max(8, int(round(fps_eff * 0.28)))
    boundary_neutral_radius = max(3, int(round(fps_eff * 0.10)))
    peak_distance = max(18, int(round(fps_eff * 0.55)))
    valley_distance = max(18, int(round(fps_eff * 0.55)))

    lean_deadband = max(0.06, float(np.std(lean_smooth) * 0.12))
    transition_neutral_abs = max(0.045, lean_deadband * 0.78)
    sustained_same_side_abs = max(0.075, lean_deadband * 0.95)

    peak_prom = max(0.05, float(np.std(pressure_smooth) * 0.18))
    valley_prom = max(0.04, float(np.std(pressure_smooth) * 0.16))
    peaks, valley_peak_props = find_peaks(pressure_smooth, distance=peak_distance, prominence=peak_prom)
    valleys, valley_props = find_peaks(-pressure_smooth, distance=valley_distance, prominence=valley_prom)

    peak_candidates = [int(item) for item in peaks.tolist()]
    if not peak_candidates:
        peak_candidates = [int(np.argmax(pressure_smooth))]

    raw_boundaries = [0]
    for idx, valley in enumerate(valleys.tolist()):
        valley_idx = int(valley)
        prom = float(valley_props['prominences'][idx]) if 'prominences' in valley_props else 0.0
        left_peak = max([peak for peak in peak_candidates if peak < valley_idx], default=None)
        right_peak = min([peak for peak in peak_candidates if peak > valley_idx], default=None)
        has_two_sides = left_peak is not None and right_peak is not None
        if not has_two_sides and prom < 0.06:
            continue

        if has_two_sides:
            left_peak_pressure = float(pressure_smooth[int(left_peak)])
            right_peak_pressure = float(pressure_smooth[int(right_peak)])
            valley_pressure = float(pressure_smooth[valley_idx])
            ref_peak = max(min(left_peak_pressure, right_peak_pressure), 1e-6)
            unload_ratio = float((ref_peak - valley_pressure) / ref_peak)
            lean_info = boundary_lean_features(lean_smooth, valley_idx, boundary_side_radius, boundary_neutral_radius)
            no_real_transition = (not bool(lean_info['sign_flip'])) and float(lean_info['neutral_abs']) > transition_neutral_abs
            same_side_hold = (
                bool(lean_info['same_side'])
                and min(abs(float(lean_info['left_mean'])), abs(float(lean_info['right_mean']))) > sustained_same_side_abs
            )
            if same_side_hold and no_real_transition:
                continue
            if unload_ratio < 0.16 and no_real_transition:
                continue
            if no_real_transition and valley_pressure > max(0.24, ref_peak * 0.50) and unload_ratio < 0.48:
                continue

        refined = refine_boundary_with_lean(valley_idx, lean_smooth, boundary_refine_radius)
        if refined <= 0 or refined >= frame_count - 1:
            continue
        raw_boundaries.append(int(refined))
    raw_boundaries.append(frame_count - 1)

    boundaries: list[int] = []
    for boundary in sorted(set(raw_boundaries)):
        if not boundaries:
            boundaries.append(int(boundary))
            continue
        if int(boundary) - boundaries[-1] < min_boundary_gap:
            prev_strength = boundary_strength(pressure_smooth, boundaries[-1], boundary_strength_radius)
            curr_strength = boundary_strength(pressure_smooth, int(boundary), boundary_strength_radius)
            if curr_strength > prev_strength:
                boundaries[-1] = int(boundary)
            continue
        boundaries.append(int(boundary))
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    if boundaries[-1] != frame_count - 1:
        boundaries.append(frame_count - 1)

    def evaluate_segments(current_boundaries: list[int]) -> list[dict[str, float | int]]:
        segments: list[dict[str, float | int]] = []
        for start, end in zip(current_boundaries[:-1], current_boundaries[1:]):
            if end <= start:
                continue
            duration = int(end - start + 1)
            if duration < min_turn_frames:
                continue
            left_valley = float(pressure_smooth[start])
            right_valley = float(pressure_smooth[end])
            segment_pressure = pressure_smooth[start:end + 1]
            apex = int(start + int(np.argmax(segment_pressure)))
            peak_pressure = float(pressure_smooth[apex])
            prominence = peak_pressure - max(min(left_valley, right_valley), min(left_valley, right_valley) - 1e-6)
            valley_floor = min(left_valley, right_valley)
            baseline = max(peak_pressure - valley_floor, 1e-6)
            ascent_frames = max(apex - start, 1)
            release_frames = max(end - apex, 1)
            weights = segment_pressure - float(np.min(segment_pressure)) + 1e-3
            weighted_lean = float(np.sum(lean_smooth[start:end + 1] * weights) / max(float(np.sum(weights)), 1e-6))
            zero_crossings = int(np.count_nonzero(np.diff(np.signbit(lean_smooth[start:end + 1]))))
            peak_grad_before = float(np.max(np.abs(pressure_grad[max(start, apex - boundary_strength_radius):apex + 1]))) if apex >= start else 0.0
            peak_grad_after = float(np.max(np.abs(pressure_grad[apex:min(end + 1, apex + boundary_strength_radius + 1)]))) if end >= apex else 0.0
            segments.append({
                'start': int(start),
                'apex': int(apex),
                'end': int(end),
                'duration': int(duration),
                'peak_pressure': peak_pressure,
                'prominence': float(prominence),
                'weighted_lean': weighted_lean,
                'zero_crossings': zero_crossings,
                'ascent_ratio': float(ascent_frames / max(duration, 1)),
                'release_ratio': float(release_frames / max(duration, 1)),
                'gradient_energy': float(max(peak_grad_before, peak_grad_after)),
                'pressure_span': float(baseline),
            })
        return segments

    def should_merge_adjacent(
        left_segment: dict[str, float | int],
        right_segment: dict[str, float | int],
        boundary: int,
        median_duration: float,
    ) -> bool:
        boundary_idx = int(boundary)
        valley_pressure = float(pressure_smooth[boundary_idx])
        ref_peak = max(min(float(left_segment['peak_pressure']), float(right_segment['peak_pressure'])), 1e-6)
        unload_ratio = float((ref_peak - valley_pressure) / ref_peak)
        lean_info = boundary_lean_features(lean_smooth, boundary_idx, boundary_side_radius, boundary_neutral_radius)
        no_real_transition = (not bool(lean_info['sign_flip'])) and float(lean_info['neutral_abs']) > transition_neutral_abs
        same_side = (
            bool(lean_info['same_side'])
            and min(abs(float(left_segment['weighted_lean'])), abs(float(right_segment['weighted_lean']))) > sustained_same_side_abs
        )
        short_side = min(int(left_segment['duration']), int(right_segment['duration']))
        long_side = max(int(left_segment['duration']), int(right_segment['duration']))
        shallow_boundary = valley_pressure > max(0.20, ref_peak * 0.45)

        if same_side and no_real_transition:
            return True
        if no_real_transition and shallow_boundary and unload_ratio < 0.52:
            return True
        if short_side < max(min_turn_frames, int(round(median_duration * 0.62))) and no_real_transition and unload_ratio < 0.42:
            return True
        if long_side > max(int(round(median_duration * 1.55)), int(round(fps_eff * 1.45))) and same_side and unload_ratio < 0.62:
            return True
        return False

    for _ in range(24):
        segments = evaluate_segments(boundaries)
        if len(segments) <= 1:
            break
        durations = [int(item['duration']) for item in segments]
        prominences = [float(item['prominence']) for item in segments]
        median_duration = float(np.median(durations)) if durations else float(min_turn_frames)
        median_prominence = float(np.median(prominences)) if prominences else 0.08
        changed = False

        for seg_idx in range(len(segments) - 1):
            if should_merge_adjacent(segments[seg_idx], segments[seg_idx + 1], boundaries[seg_idx + 1], median_duration):
                del boundaries[seg_idx + 1]
                changed = True
                break
        if changed:
            continue

        for seg_idx, segment in enumerate(segments):
            duration = int(segment['duration'])
            prominence = float(segment['prominence'])
            ascent_ratio = float(segment['ascent_ratio'])
            release_ratio = float(segment['release_ratio'])
            weak_by_pressure = prominence < max(0.045, median_prominence * 0.46)
            weak_by_duration = duration < max(min_turn_frames, int(round(median_duration * 0.52)))
            implausible_shape = ascent_ratio < 0.10 or release_ratio < 0.10
            too_many_crossings = int(segment['zero_crossings']) >= 3 and prominence < max(0.055, median_prominence * 0.62)
            edge_relaxed = seg_idx in (0, len(segments) - 1) and prominence < max(0.040, median_prominence * 0.42)
            if not ((weak_by_pressure and weak_by_duration) or implausible_shape or too_many_crossings or edge_relaxed):
                continue
            candidate_boundary_indices: list[tuple[float, int]] = []
            left_boundary_idx = seg_idx
            right_boundary_idx = seg_idx + 1
            if left_boundary_idx > 0:
                candidate_boundary_indices.append((boundary_strength(pressure_smooth, boundaries[left_boundary_idx], boundary_strength_radius), left_boundary_idx))
            if right_boundary_idx < len(boundaries) - 1:
                candidate_boundary_indices.append((boundary_strength(pressure_smooth, boundaries[right_boundary_idx], boundary_strength_radius), right_boundary_idx))
            if not candidate_boundary_indices:
                continue
            remove_idx = min(candidate_boundary_indices, key=lambda item: item[0])[1]
            del boundaries[remove_idx]
            changed = True
            break
        if not changed:
            break

    segments = evaluate_segments(boundaries)
    if not segments:
        peak = int(np.argmax(pressure_smooth))
        return [TurnSegment(index=1, start=0, apex=peak, end=frame_count - 1, direction='unknown')]

    deadband = max(0.06, float(np.std(lean_smooth) * 0.12))
    turns: list[TurnSegment] = []
    previous_direction: str | None = None
    for segment in segments:
        weighted_lean = float(segment['weighted_lean'])
        if weighted_lean <= -deadband:
            direction = 'left'
        elif weighted_lean >= deadband:
            direction = 'right'
        elif previous_direction is None:
            local = lean_smooth[int(segment['start']):int(segment['end']) + 1]
            apex = int(segment['apex'])
            direction = 'left' if float(np.mean(local[: max(apex - int(segment['start']), 1)])) < 0.0 else 'right'
        else:
            direction = 'left' if previous_direction == 'right' else 'right'
        turns.append(
            TurnSegment(
                index=len(turns) + 1,
                start=int(segment['start']),
                apex=int(segment['apex']),
                end=int(segment['end']),
                direction=direction,
            )
        )
        previous_direction = direction

    if not turns:
        peak = int(np.argmax(pressure_smooth))
        turns = [TurnSegment(index=1, start=0, apex=peak, end=frame_count - 1, direction='unknown')]
    return turns


def build_ideal_curve(
    frame_count: int,
    turns: Iterable[TurnSegment],
    actual_pressure: np.ndarray | None = None,
    style: str = 'default',
) -> np.ndarray:
    ideal = np.zeros((frame_count,), dtype=np.float32)
    actual = None if actual_pressure is None else np.asarray(actual_pressure, dtype=np.float32)
    turns = list(turns)
    style = str(style or 'default').lower()

    if style == 'infinity':
        transition_floor = 0.34
        edge_floor = 0.20
        for turn_idx, turn in enumerate(turns):
            start = int(turn.start)
            apex = int(turn.apex)
            end = int(turn.end)
            start_level = edge_floor if turn_idx == 0 else transition_floor
            end_level = edge_floor if turn_idx == len(turns) - 1 else transition_floor
            if actual is not None and 0 <= start < frame_count and start == 0:
                start_level = float(np.clip(max(start_level, actual[start]), start_level, 0.55))
            if actual is not None and 0 <= end < frame_count and end == frame_count - 1:
                end_level = float(np.clip(max(end_level, actual[end]), end_level, 0.55))
            if apex > start:
                left_x = np.linspace(0.0, 1.0, apex - start + 1, dtype=np.float32)
                left_curve = start_level + (1.0 - start_level) * np.sin(0.5 * np.pi * left_x) ** 0.82
                ideal[start:apex + 1] = np.maximum(ideal[start:apex + 1], left_curve.astype(np.float32))
            else:
                ideal[apex] = max(float(ideal[apex]), 1.0)
            if end > apex:
                right_x = np.linspace(0.0, 1.0, end - apex + 1, dtype=np.float32)
                right_curve = end_level + (1.0 - end_level) * np.sin(0.5 * np.pi * (1.0 - right_x)) ** 0.88
                ideal[apex:end + 1] = np.maximum(ideal[apex:end + 1], right_curve.astype(np.float32))
        return np.clip(ideal, 0.0, 1.0).astype(np.float32)

    for turn in turns:
        start = int(turn.start)
        apex = int(turn.apex)
        end = int(turn.end)
        start_level = 0.0
        end_level = 0.0
        if actual is not None and 0 <= start < frame_count and start == 0:
            start_level = float(np.clip(actual[start], 0.0, 1.0))
        if actual is not None and 0 <= end < frame_count and end == frame_count - 1:
            end_level = float(np.clip(actual[end], 0.0, 1.0))
        if apex > start:
            left_x = np.linspace(0.0, 1.0, apex - start + 1, dtype=np.float32)
            left_curve = start_level + (1.0 - start_level) * np.sin(0.5 * np.pi * left_x)
            ideal[start:apex + 1] = np.maximum(ideal[start:apex + 1], left_curve.astype(np.float32))
        else:
            ideal[apex] = max(float(ideal[apex]), 1.0)
        if end > apex:
            right_x = np.linspace(0.0, 1.0, end - apex + 1, dtype=np.float32)
            right_curve = end_level + (1.0 - end_level) * np.sin(0.5 * np.pi * (1.0 - right_x))
            ideal[apex:end + 1] = np.maximum(ideal[apex:end + 1], right_curve)
    return np.clip(ideal, 0.0, 1.0).astype(np.float32)


def compute_metrics(actual: np.ndarray, ideal: np.ndarray, turns: list[TurnSegment], fps: float) -> dict[str, float | int]:
    diff = actual - ideal
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    mae = float(np.mean(np.abs(diff)))
    corr = float(np.corrcoef(actual, ideal)[0, 1]) if len(actual) >= 3 else 0.0
    apex_lags = []
    apex_values = []
    for turn in turns:
        window = actual[turn.start:turn.end + 1]
        if len(window) == 0:
            continue
        local_peak = int(np.argmax(window)) + turn.start
        apex_lags.append(local_peak - turn.apex)
        apex_values.append(float(actual[local_peak]))
    mean_apex_lag_frames = float(np.mean(apex_lags)) if apex_lags else 0.0
    mean_apex_lag_seconds = float(mean_apex_lag_frames / max(fps, 1e-6))
    mean_apex_pressure = float(np.mean(apex_values)) if apex_values else float(np.max(actual))
    sync_score = float(np.clip(100.0 * (0.58 * ((corr + 1.0) * 0.5) + 0.42 * (1.0 - rmse)), 0.0, 100.0))
    return {
        'turn_count': int(len(turns)),
        'rmse': rmse,
        'mae': mae,
        'correlation': corr,
        'mean_apex_lag_frames': mean_apex_lag_frames,
        'mean_apex_lag_seconds': mean_apex_lag_seconds,
        'mean_apex_pressure': mean_apex_pressure,
        'sync_score': sync_score,
    }


def save_timeseries_csv(
    out_path: Path,
    frame_indices: np.ndarray,
    time_axis: np.ndarray,
    actual: np.ndarray,
    ideal: np.ndarray,
    lean_signal: np.ndarray,
    turns: list[TurnSegment],
) -> None:
    turn_id = np.zeros((len(frame_indices),), dtype=np.int32)
    apex_mask = np.zeros((len(frame_indices),), dtype=np.int32)
    frame_to_local = {int(frame): idx for idx, frame in enumerate(frame_indices.tolist())}
    for turn in turns:
        for frame in range(turn.start, turn.end + 1):
            local_idx = frame_to_local.get(frame)
            if local_idx is not None:
                turn_id[local_idx] = int(turn.index)
        local_apex_idx = frame_to_local.get(turn.apex)
        if local_apex_idx is not None:
            apex_mask[local_apex_idx] = 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(['frame', 'time_sec', 'actual_pressure', 'ideal_pressure', 'deviation', 'lean_signal', 'turn_id', 'is_apex'])
        for idx, frame in enumerate(frame_indices.tolist()):
            writer.writerow([
                int(frame),
                float(time_axis[idx]),
                float(actual[idx]),
                float(ideal[idx]),
                float(actual[idx] - ideal[idx]),
                float(lean_signal[idx]),
                int(turn_id[idx]),
                int(apex_mask[idx]),
            ])


def save_metrics_json(out_path: Path, metrics: dict[str, float | int], window: AnalysisWindow) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metrics)
    payload['analysis_window_start'] = int(window.start)
    payload['analysis_window_end'] = int(window.end)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def summarize_turn_deviation(
    actual: np.ndarray,
    ideal: np.ndarray,
    turns: list[TurnSegment],
    window: AnalysisWindow,
) -> list[str]:
    notes: list[tuple[float, str]] = []
    for turn in turns:
        start = max(0, int(turn.start - window.start))
        end = min(len(actual) - 1, int(turn.end - window.start))
        if end <= start:
            continue
        mean_dev = float(np.mean(actual[start:end + 1] - ideal[start:end + 1]))
        if mean_dev >= 0.045:
            notes.append((abs(mean_dev), f'第{turn.index}弯整体偏早 / 偏重'))
        elif mean_dev <= -0.045:
            notes.append((abs(mean_dev), f'第{turn.index}弯整体偏晚 / 偏轻'))
    notes.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in notes[:3]]


def build_pressure_chart_figure(
    frame_indices: np.ndarray,
    time_axis: np.ndarray,
    actual: np.ndarray,
    ideal: np.ndarray,
    turns: list[TurnSegment],
    metrics: dict[str, float | int],
    window: AnalysisWindow,
    fps: float,
    title: str,
    display_direction: str = 'down',
    compact_layout: bool = False,
    figsize: tuple[float, float] = (18.0, 10.0),
    dpi: int = 220,
    cursor_time: float | None = None,
    current_actual: float | None = None,
    current_ideal: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    plt.style.use('seaborn-v0_8-whitegrid')
    configure_plot_fonts()

    figure, axis = plt.subplots(figsize=figsize, dpi=dpi)
    figure.patch.set_facecolor('white')
    axis.set_facecolor('#f8fafc')
    if compact_layout:
        figure.subplots_adjust(left=0.16, right=0.97, bottom=0.10, top=0.89)
    else:
        figure.subplots_adjust(left=0.07, right=0.94, bottom=0.16, top=0.74)

    x_values = np.asarray(time_axis, dtype=np.float32)
    actual = np.clip(np.asarray(actual, dtype=np.float32), 0.0, 1.0)
    ideal = np.clip(np.asarray(ideal, dtype=np.float32), 0.0, 1.0)

    axis.axhspan(0.00, 0.25, color='#e0f2fe', alpha=0.34, zorder=0)
    axis.axhspan(0.25, 0.55, color='#eef2ff', alpha=0.28, zorder=0)
    axis.axhspan(0.55, 1.02, color='#fef3c7', alpha=0.24, zorder=0)
    axis.axhline(0.0, color='#0ea5e9', linewidth=1.45, alpha=0.95, zorder=2)

    for turn in turns:
        left = float((turn.start - window.start) / max(fps, 1e-6))
        right = float((turn.end - window.start) / max(fps, 1e-6))
        apex_time = float((turn.apex - window.start) / max(fps, 1e-6))
        shade_color = '#dbeafe' if turn.index % 2 == 1 else '#e2e8f0'
        axis.axvspan(left, right, color=shade_color, alpha=0.22, lw=0, zorder=0)
        axis.axvline(apex_time, color='#64748b', linestyle=(0, (4, 4)), linewidth=1.05, alpha=0.78, zorder=1)
        center = 0.5 * (left + right)
        axis.text(center, 1.045, f'第{turn.index}弯', ha='center', va='bottom', fontsize=10.0, color='#334155', transform=axis.get_xaxis_transform())
        axis.text(apex_time, 1.002, f'极点{turn.index}', ha='center', va='bottom', fontsize=8.0, color='#64748b', rotation=90, transform=axis.get_xaxis_transform())
        apex_local = int(np.clip(turn.apex - window.start, 0, len(ideal) - 1))

    axis.plot(x_values, actual, color='#2563eb', linewidth=2.9, label='实际压力曲线', zorder=5)

    under_mask = actual < ideal
    over_mask = actual >= ideal
    axis.fill_between(x_values, actual, ideal, where=under_mask, color='#ef4444', alpha=0.20, interpolate=True, label='压力不足 / 发力偏晚', zorder=2)
    axis.fill_between(x_values, actual, ideal, where=over_mask, color='#f59e0b', alpha=0.16, interpolate=True, label='压力过大 / 发力偏早', zorder=1)

    if cursor_time is not None:
        axis.axvline(float(cursor_time), color='#dc2626', linewidth=2.15, alpha=0.95, zorder=8)
        if current_actual is not None:
            axis.scatter([cursor_time], [float(current_actual)], s=52, color='#2563eb', edgecolors='white', linewidths=1.2, zorder=9)

    if compact_layout:
        axis.set_title(title, fontsize=16, fontweight='bold', pad=6, color='#0f172a')
        figure.text(0.56, 0.935, '红线=当前时刻', ha='center', fontsize=9.0, color='#64748b')
    else:
        figure.suptitle(title, y=0.965, fontsize=21, fontweight='bold', color='#0f172a')
        subtitle = '说明：越往下代表施压越强；最上方蓝线表示雪面压力 = 0（完全卸压）。' if display_direction == 'down' else '说明：越往上代表施压越强；最下方蓝线表示雪面压力 = 0（完全卸压）。'
        figure.text(0.5, 0.925, subtitle, ha='center', fontsize=11.3, color='#475569')
        figure.text(0.5, 0.902, '蓝线是你的实际压力；视频联动版中，红色竖线表示当前视频时刻。', ha='center', fontsize=10.4, color='#64748b')
        turn_notes = summarize_turn_deviation(actual, ideal, turns, window)
        if turn_notes:
            figure.text(0.935, 0.12, '关键提示\n' + '\n'.join(f'• {item}' for item in turn_notes), ha='right', va='bottom', fontsize=10.1, color='#334155', bbox={'boxstyle': 'round,pad=0.45', 'facecolor': '#ffffff', 'edgecolor': '#cbd5e1', 'alpha': 0.96})

    axis.set_xlabel('时间（秒）', fontsize=13 if not compact_layout else 11, color='#0f172a')
    axis.set_ylabel('雪面相对压力（顶端 = 0压）', fontsize=13 if not compact_layout else 11, color='#0f172a')
    axis.set_xlim(float(x_values[0]), float(x_values[-1]) if len(x_values) > 1 else float(x_values[0] + 1.0 / max(fps, 1e-6)))
    axis.set_ylim(-0.02, 1.02)
    if display_direction == 'down':
        axis.invert_yaxis()

    axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.set_yticklabels(['0压', '轻压', '中压', '强压', '峰值'])
    axis.tick_params(axis='both', labelsize=11 if not compact_layout else 9.5, colors='#334155')
    axis.text(0.995, 0.0, '雪面压力 = 0（完全卸压）', ha='right', va='bottom', fontsize=9.6, color='#0369a1', transform=axis.get_yaxis_transform())
    axis.text(-0.055, 0.08, '伸展\n减压', ha='right', va='center', fontsize=9.6, color='#0369a1', transform=axis.get_yaxis_transform())
    axis.text(-0.055, 0.92, '下压\n压缩', ha='right', va='center', fontsize=9.6, color='#b45309', transform=axis.get_yaxis_transform())

    axis.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.36, color='#94a3b8')
    axis.grid(True, axis='x', linestyle=':', linewidth=0.6, alpha=0.22, color='#94a3b8')
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_color('#cbd5e1')
    axis.spines['bottom'].set_color('#cbd5e1')

    handles, labels = axis.get_legend_handles_labels()
    if compact_layout:
        figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.915), ncol=2, frameon=True, framealpha=0.96, facecolor='white', edgecolor='#cbd5e1', fontsize=8.2)
        compact_summary = f"同步 {metrics['sync_score']:.1f}/100  |  相关 {metrics['correlation']:.3f}"
        figure.text(0.5, 0.028, compact_summary, ha='center', va='center', fontsize=8.6, color='#334155')
    else:
        figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.875), ncol=4, frameon=True, framealpha=0.96, facecolor='white', edgecolor='#cbd5e1', fontsize=10.0)

        summary_text = (
            f"转弯数 {metrics['turn_count']}   |   同步得分 {metrics['sync_score']:.1f}/100   |   "
            f"RMSE {metrics['rmse']:.3f}   |   相关系数 {metrics['correlation']:.3f}   |   "
            f"极点滞后 {metrics['mean_apex_lag_frames']:+.1f} 帧 / {metrics['mean_apex_lag_seconds']:+.3f} 秒   |   "
            f"极点平均压力 {metrics['mean_apex_pressure']:.3f}   |   分析帧段 {window.start}-{window.end}"
        )
        figure.text(
            0.07,
            0.06,
            summary_text,
            ha='left',
            va='center',
            fontsize=10.8,
            color='#334155',
            bbox={'boxstyle': 'round,pad=0.45', 'facecolor': '#f8fafc', 'edgecolor': '#cbd5e1', 'alpha': 0.98},
        )
    return figure, axis


def figure_to_bgr(figure: plt.Figure) -> np.ndarray:
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def find_current_turn(turns: list[TurnSegment], frame_idx: int) -> TurnSegment | None:
    for turn in turns:
        if turn.start <= frame_idx <= turn.end:
            return turn
    if not turns:
        return None
    if frame_idx < turns[0].start:
        return turns[0]
    return turns[-1]


def describe_turn_phase(turn: TurnSegment | None, frame_idx: int) -> str:
    if turn is None:
        return '未识别到转弯阶段'
    span = max(turn.end - turn.start, 1)
    rel = (frame_idx - turn.start) / span
    if rel <= 0.12 or rel >= 0.88:
        return '换刃 / 释压阶段'
    if frame_idx < turn.apex:
        return '入弯建立压力'
    if frame_idx == turn.apex:
        return '弯道极点 / 峰值压力'
    return '出弯释放压力'


def build_coach_tip(current_actual: float, current_ideal: float) -> tuple[str, tuple[int, int, int]]:
    deviation = current_actual - current_ideal
    if deviation >= 0.08:
        return '建议：稍晚一点下压，避免发力偏早', (245, 158, 11)
    if deviation >= 0.03:
        return '建议：轻微减压，接近参考节奏', (250, 204, 21)
    if deviation <= -0.08:
        return '建议：更早建立压力，别等到弯后段', (239, 68, 68)
    if deviation <= -0.03:
        return '建议：再积极一点下压，别偏晚', (248, 113, 113)
    return '建议：当前节奏接近理想，继续保持', (74, 222, 128)


def paste_inside_bottom_panel(
    frame: np.ndarray,
    panel_frame: np.ndarray,
    track_boxes: np.ndarray | None,
    frame_idx: int,
    alpha: float = 0.50,
) -> np.ndarray:
    canvas = frame.copy()
    panel_h, panel_w = panel_frame.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, (frame_w - panel_w) // 2)
    y1 = max(0, frame_h - panel_h - 18)
    x2 = min(frame_w, x1 + panel_w)
    y2 = min(frame_h, y1 + panel_h)

    roi = canvas[y1:y2, x1:x2].copy()
    blurred = cv2.GaussianBlur(roi, (31, 31), 0)
    blended = cv2.addWeighted(blurred, 1.0 - alpha, panel_frame[: y2 - y1, : x2 - x1], alpha, 0.0)
    canvas[y1:y2, x1:x2] = blended

    if track_boxes is not None and frame_idx < len(track_boxes):
        box = np.asarray(track_boxes[frame_idx], dtype=np.float32)
        if np.isfinite(box).all():
            bx1 = max(x1, int(round(box[0] - 14)))
            by1 = max(y1, int(round(box[1] - 14)))
            bx2 = min(x2, int(round(box[2] + 14)))
            by2 = min(y2, int(round(box[3] + 14)))
            if bx2 > bx1 and by2 > by1:
                canvas[by1:by2, bx1:bx2] = frame[by1:by2, bx1:bx2]

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.9, canvas, 0.1, 0.0, dst=canvas)
    return canvas


def draw_hud(frame: np.ndarray, current_time: float, current_actual: float, current_ideal: float, current_turn: TurnSegment | None, phase_text: str) -> None:
    h, w = frame.shape[:2]
    narrow = w <= 420
    if narrow:
        box_w = min(max(int(round(w * 0.86)), 210), w - 16)
        box_h = 152
        x2 = w - 8
        y1 = 8
        x1 = x2 - box_w
        y2 = y1 + box_h
    else:
        box_w = min(350, max(270, int(round(w * 0.44))))
        box_h = 196
        x2 = w - 18
        y1 = 18
        x1 = x2 - box_w
        y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 18, 32), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.64, frame, 0.36, 0.0, dst=frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (170, 190, 220), 1, cv2.LINE_AA)

    deviation = current_actual - current_ideal
    deviation_text = '偏早 / 偏大' if deviation > 0.03 else ('偏晚 / 偏小' if deviation < -0.03 else '接近参考')
    turn_text = f'第{current_turn.index}弯' if current_turn is not None else '转弯未知'
    coach_tip, tip_color = build_coach_tip(current_actual, current_ideal)
    if narrow:
        rows = [
            ('压力同步 HUD', (245, 248, 255), 16),
            (f'{current_time:4.2f}s  {turn_text}', (226, 232, 240), 12),
            (f'当前 {current_actual:.3f}  参考 {current_ideal:.3f}', (59, 130, 246), 12),
            (f'{deviation_text}  偏差 {deviation:+.3f}', (245, 158, 11) if deviation > 0.03 else ((239, 68, 68) if deviation < -0.03 else (226, 232, 240)), 11),
            (phase_text, (226, 232, 240), 11),
            (coach_tip.replace('建议：', ''), tip_color, 11),
        ]
        x_pad = 10
        line_gap = 20
        title_gap = 22
    else:
        rows = [
            ('压力同步 HUD', (245, 248, 255), 26),
            (f'时间: {current_time:5.2f} 秒', (226, 232, 240), 20),
            (f'当前压力: {current_actual:.3f}    参考: {current_ideal:.3f}', (59, 130, 246), 19),
            (f'当前判断: {deviation_text}   偏差 {deviation:+.3f}', (245, 158, 11) if deviation > 0.03 else ((239, 68, 68) if deviation < -0.03 else (226, 232, 240)), 18),
            (f'阶段: {turn_text} · {phase_text}', (226, 232, 240), 18),
            (coach_tip, tip_color, 18),
        ]
        x_pad = 15
        line_gap = 26
        title_gap = 28

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    y = y1 + (8 if narrow else 12)
    for idx, (text_value, color, size) in enumerate(rows):
        font = load_hud_font(size)
        draw.text((x1 + x_pad, y), text_value, font=font, fill=(8, 10, 16))
        draw.text((x1 + x_pad - 1, y - 1), text_value, font=font, fill=color)
        y += title_gap if idx == 0 else line_gap

    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render_plot(
    out_path: Path,
    frame_indices: np.ndarray,
    time_axis: np.ndarray,
    actual: np.ndarray,
    ideal: np.ndarray,
    turns: list[TurnSegment],
    metrics: dict[str, float | int],
    window: AnalysisWindow,
    fps: float,
    title: str,
    display_direction: str = 'down',
) -> None:
    figure, _ = build_pressure_chart_figure(
        frame_indices=frame_indices,
        time_axis=time_axis,
        actual=actual,
        ideal=ideal,
        turns=turns,
        metrics=metrics,
        window=window,
        fps=fps,
        title=title,
        display_direction=display_direction,
        figsize=(18.0, 10.0),
        dpi=220,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, bbox_inches='tight', facecolor=figure.get_facecolor())
    plt.close(figure)


def render_pressure_sync_video(
    source_video_path: Path,
    out_path: Path,
    frame_indices: np.ndarray,
    time_axis: np.ndarray,
    actual: np.ndarray,
    ideal: np.ndarray,
    turns: list[TurnSegment],
    metrics: dict[str, float | int],
    window: AnalysisWindow,
    fps: float,
    title: str,
    display_direction: str = 'down',
    chart_height: int = 0,
    layout: str = 'auto',
    track_boxes: np.ndarray | None = None,
) -> None:
    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频用于压力联动渲染: {source_video_path}')

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if source_fps <= 1e-3:
        source_fps = fps if fps > 1e-3 else 30.0
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if video_width <= 0 or video_height <= 0:
        cap.release()
        raise RuntimeError(f'无法读取视频尺寸: {source_video_path}')

    narrow_video = video_width <= 420
    portrait_video = video_height > video_width
    if layout == 'auto':
        layout = 'inside_bottom' if portrait_video else 'bottom'
    if layout not in {'right', 'bottom', 'inside_bottom'}:
        layout = 'inside_bottom' if portrait_video else 'bottom'

    if layout == 'right':
        panel_width = max(320, min(520, int(round(video_width * 0.56))))
        panel_height = video_height
    elif layout == 'bottom':
        if int(chart_height) > 0:
            panel_height = int(chart_height)
        elif narrow_video:
            panel_height = max(250, min(360, int(round(video_width * 0.95))))
        else:
            panel_height = max(240, min(360, int(round(video_height * 0.34))))
        panel_height = max(220, panel_height)
        panel_width = video_width
    else:
        panel_width = max(180, min(int(round(video_width * 0.92)), video_width - 20))
        if int(chart_height) > 0:
            panel_height = int(chart_height)
        else:
            panel_height = int(round(panel_width / 1.95))
            panel_height = min(panel_height, int(round(video_height * 0.32)))
        panel_height = max(110, panel_height)

    compact_panel = layout in {'bottom', 'inside_bottom'}
    fig_dpi = 160
    fig_width = max(panel_width / fig_dpi, 4.5)
    fig_height = max(panel_height / fig_dpi, 4.5)
    compact_title = '压力同步' if compact_panel or layout != 'bottom' else f'{title} · 同步版'
    figure, axis = build_pressure_chart_figure(
        frame_indices=frame_indices,
        time_axis=time_axis,
        actual=actual,
        ideal=ideal,
        turns=turns,
        metrics=metrics,
        window=window,
        fps=fps,
        title=compact_title,
        display_direction=display_direction,
        compact_layout=compact_panel or (layout != 'bottom'),
        figsize=(fig_width, fig_height),
        dpi=fig_dpi,
    )
    panel = figure_to_bgr(figure)
    canvas_width, canvas_height = figure.canvas.get_width_height(physical=True)
    plot_bbox = axis.get_window_extent()

    actual_disp = axis.transData.transform(np.column_stack([time_axis, actual]))
    ideal_disp = axis.transData.transform(np.column_stack([time_axis, ideal]))
    x_pixels = np.round(actual_disp[:, 0]).astype(np.int32)
    actual_y_pixels = np.round(canvas_height - actual_disp[:, 1]).astype(np.int32)
    ideal_y_pixels = np.round(canvas_height - ideal_disp[:, 1]).astype(np.int32)

    plot_left = int(round(plot_bbox.x0))
    plot_right = int(round(plot_bbox.x1))
    plot_top = int(round(canvas_height - plot_bbox.y1))
    plot_bottom = int(round(canvas_height - plot_bbox.y0))
    plt.close(figure)

    if panel.shape[1] != panel_width or panel.shape[0] != panel_height:
        scale_x = float(panel_width) / max(float(panel.shape[1]), 1.0)
        scale_y = float(panel_height) / max(float(panel.shape[0]), 1.0)
        panel = cv2.resize(panel, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
        x_pixels = np.round(x_pixels * scale_x).astype(np.int32)
        actual_y_pixels = np.round(actual_y_pixels * scale_y).astype(np.int32)
        ideal_y_pixels = np.round(ideal_y_pixels * scale_y).astype(np.int32)
        plot_left = int(round(plot_left * scale_x))
        plot_right = int(round(plot_right * scale_x))
        plot_top = int(round(plot_top * scale_y))
        plot_bottom = int(round(plot_bottom * scale_y))

    if layout == 'right':
        out_size = (video_width + 10 + panel.shape[1], video_height)
    elif layout == 'bottom':
        out_size = (video_width, video_height + 10 + panel.shape[0])
    else:
        out_size = (video_width, video_height)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        source_fps,
        out_size,
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'无法创建压力联动视频: {out_path}')

    analysis_start = int(frame_indices[0]) if len(frame_indices) else 0
    analysis_end = int(frame_indices[-1]) if len(frame_indices) else -1
    local_count = len(time_axis)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        panel_frame = panel.copy()
        if local_count > 0:
            clamped_local_idx = int(np.clip(frame_idx - analysis_start, 0, local_count - 1))
            cursor_x = int(np.clip(x_pixels[clamped_local_idx], plot_left, plot_right))
            cursor_actual_y = int(np.clip(actual_y_pixels[clamped_local_idx], plot_top, plot_bottom))
            cursor_ideal_y = int(np.clip(ideal_y_pixels[clamped_local_idx], plot_top, plot_bottom))
            cv2.line(panel_frame, (cursor_x, plot_top), (cursor_x, plot_bottom), (38, 38, 220), 2, cv2.LINE_AA)
            current_actual = float(actual[clamped_local_idx])
            current_ideal = float(ideal[clamped_local_idx])
            current_time = float(time_axis[clamped_local_idx])
            current_turn = find_current_turn(turns, frame_idx)
            phase_text = describe_turn_phase(current_turn, frame_idx)
            if analysis_start <= frame_idx <= analysis_end:
                cv2.circle(panel_frame, (cursor_x, cursor_actual_y), 7, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(panel_frame, (cursor_x, cursor_actual_y), 5, (235, 99, 37), -1, cv2.LINE_AA)
                draw_hud(frame, current_time, current_actual, current_ideal, current_turn, phase_text)
            else:
                draw_hud(frame, current_time, current_actual, current_ideal, current_turn, '分析窗外')

        separator_color = (245, 248, 252)
        if layout == 'right':
            separator = np.full((video_height, 10, 3), separator_color, dtype=np.uint8)
            combined = cv2.hconcat([frame, separator, panel_frame])
        elif layout == 'bottom':
            separator = np.full((10, video_width, 3), separator_color, dtype=np.uint8)
            combined = cv2.vconcat([frame, separator, panel_frame])
        else:
            combined = paste_inside_bottom_panel(frame, panel_frame, track_boxes, frame_idx, alpha=0.50)
        writer.write(combined)
        frame_idx += 1

    writer.release()
    cap.release()



def infer_fps(video_path: Path | None, fps_override: float | None) -> float:
    if fps_override is not None and fps_override > 0.0:
        return float(fps_override)
    if video_path is None:
        return 30.0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 30.0
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps if fps > 1e-3 else 30.0


def default_output_paths(cache_path: Path, out_path: Path | None, csv_path: Path | None, metrics_path: Path | None) -> tuple[Path, Path, Path]:
    review_dir = cache_path.parent / 'review'
    png = out_path if out_path is not None else review_dir / 'pressure_comparison_analysis.png'
    csv_out = csv_path if csv_path is not None else review_dir / 'pressure_curve_timeseries.csv'
    json_out = metrics_path if metrics_path is not None else review_dir / 'pressure_curve_metrics.json'
    return png, csv_out, json_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='基于 2D 滑雪关键点生成 JSBA 风格压力曲线对比分析图。')
    parser.add_argument('--keypoint-cache', required=True, type=str)
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--fps', type=float, default=0.0)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--csv-output', type=str, default='')
    parser.add_argument('--metrics-output', type=str, default='')
    parser.add_argument('--title', type=str, default='JSBA 压力曲线对比分析')
    parser.add_argument('--display-direction', type=str, choices=('down', 'up'), default='down')
    parser.add_argument('--video-source', type=str, default='')
    parser.add_argument('--video-output', type=str, default='')
    parser.add_argument('--chart-height', type=int, default=0)
    parser.add_argument('--video-layout', type=str, choices=('auto', 'inside_bottom', 'bottom', 'right'), default='auto')
    parser.add_argument('--style', type=str, choices=('default', 'infinity'), default='default')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_path = Path(args.keypoint_cache).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve() if args.video else None
    video_source_path = Path(args.video_source).expanduser().resolve() if args.video_source else None
    out_path = Path(args.output).expanduser().resolve() if args.output else None
    csv_path = Path(args.csv_output).expanduser().resolve() if args.csv_output else None
    metrics_path = Path(args.metrics_output).expanduser().resolve() if args.metrics_output else None
    video_out_path = Path(args.video_output).expanduser().resolve() if args.video_output else None
    png_path, csv_out_path, metrics_out_path = default_output_paths(cache_path, out_path, csv_path, metrics_path)

    payload = load_payload(cache_path)
    keypoints = payload['keypoints']
    track_boxes = payload['track_boxes']
    fps = infer_fps(video_path, args.fps if args.fps > 0.0 else None)
    min_frames = max(48, int(round(fps * 1.3)))

    signals = build_kinematic_signals(keypoints, track_boxes)
    pressure_pack = compute_pressure_curve(signals)
    quality_mask = valid_count(keypoints) >= 10
    window = select_analysis_window(payload['found_mask'], payload['crowd_mask'], quality_mask, min_frames=min_frames)

    frame_slice = slice(window.start, window.end + 1)
    actual_pressure = pressure_pack['actual_pressure'][frame_slice].astype(np.float32)
    lean_signal = pressure_pack['lean_signal'][frame_slice].astype(np.float32)
    frame_indices = np.arange(window.start, window.end + 1, dtype=np.int32)
    local_turns = detect_turn_segments(lean_signal, actual_pressure, fps)

    global_turns = [
        TurnSegment(
            index=turn.index,
            start=turn.start + window.start,
            apex=turn.apex + window.start,
            end=turn.end + window.start,
            direction=turn.direction,
        )
        for turn in local_turns
    ]
    ideal_pressure = build_ideal_curve(len(actual_pressure), local_turns, actual_pressure=actual_pressure, style=args.style)
    time_axis = (frame_indices - frame_indices[0]) / max(fps, 1e-6)
    metrics = compute_metrics(actual_pressure, ideal_pressure, local_turns, fps)

    save_timeseries_csv(csv_out_path, frame_indices, time_axis, actual_pressure, ideal_pressure, lean_signal, global_turns)
    metrics['style'] = args.style
    save_metrics_json(metrics_out_path, metrics, window)
    render_plot(png_path, frame_indices, time_axis, actual_pressure, ideal_pressure, global_turns, metrics, window, fps, args.title, display_direction=args.display_direction)

    effective_video_source = video_source_path if video_source_path is not None else video_path
    if effective_video_source is not None and effective_video_source.exists():
        if video_out_path is None:
            video_out_path = cache_path.parent / 'review' / 'pressure_sync_video.mp4'
        render_pressure_sync_video(
            source_video_path=effective_video_source,
            out_path=video_out_path,
            frame_indices=frame_indices,
            time_axis=time_axis,
            actual=actual_pressure,
            ideal=ideal_pressure,
            turns=global_turns,
            metrics=metrics,
            window=window,
            fps=fps,
            title=args.title,
            display_direction=args.display_direction,
            chart_height=args.chart_height,
            layout=args.video_layout,
            track_boxes=track_boxes,
        )
        print(f'[Pressure Video] {video_out_path}')

    print(f'[Pressure] png={png_path}')
    print(f'[Pressure] csv={csv_out_path}')
    print(f'[Pressure] metrics={metrics_out_path}')
    print(f'[Pressure] style={args.style}')
    print(f'[Pressure] turns={metrics["turn_count"]} sync_score={metrics["sync_score"]:.1f} rmse={metrics["rmse"]:.3f} corr={metrics["correlation"]:.3f}')
    print(f'[Pressure] analysis_window={window.start}-{window.end}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
