from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .base import BaseEvaluator
from .common import interp_nan_1d, interp_nan_2d, joint_xy, knee_angle_deg, point_distance, weighted_mean_1d

PLOT_FILENAME = 'casi_beginner_snapshot.png'
METRICS_FILENAME = 'casi_metrics.json'
CSV_FILENAME = 'casi_timeseries.csv'
ALIGNMENT_CONF_MIN = 0.20
ALIGNMENT_TOL_RATIO = 0.12
KNEE_LOCK_THRESHOLD_DEG = 170.0
TRUNK_ACCEL_THRESHOLD = 3.5


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


def compute_center_alignment(
    keypoints_dict: dict[str, dict[str, float]],
    conf_min: float = ALIGNMENT_CONF_MIN,
    tol_ratio: float = ALIGNMENT_TOL_RATIO,
) -> dict[str, float | bool | str]:
    required = ('left_hip', 'right_hip', 'left_ankle', 'right_ankle')
    if any(float(keypoints_dict.get(name, {}).get('conf', 0.0) or 0.0) < conf_min for name in required):
        return {'ok': False, 'reason': 'missing_keypoints'}

    pelvis_x = weighted_mean_1d(
        [keypoints_dict['left_hip']['x'], keypoints_dict['right_hip']['x']],
        [keypoints_dict['left_hip']['conf'], keypoints_dict['right_hip']['conf']],
        min_conf=conf_min,
    )
    left_ankle_x = float(keypoints_dict['left_ankle']['x'])
    right_ankle_x = float(keypoints_dict['right_ankle']['x'])
    ankle_mid_x = 0.5 * (left_ankle_x + right_ankle_x)
    stance_width = abs(right_ankle_x - left_ankle_x)
    tolerance_px = max(8.0, stance_width * tol_ratio)
    offset_px = float(pelvis_x - ankle_mid_x)
    aligned = abs(offset_px) <= tolerance_px
    score = max(0.0, 1.0 - abs(offset_px) / max(tolerance_px, 1e-6))
    return {
        'ok': True,
        'aligned': aligned,
        'score': float(score),
        'offset_px': offset_px,
        'tolerance_px': float(tolerance_px),
        'pelvis_x': float(pelvis_x),
        'ankle_mid_x': float(ankle_mid_x),
    }


class CASIEvaluator(BaseEvaluator):
    def __init__(self, context):
        super().__init__(context)
        self.frame_indices: list[int] = []
        self.alignment_offsets: list[float] = []
        self.alignment_tolerances: list[float] = []
        self.alignment_scores: list[float] = []
        self.alignment_flags: list[float] = []
        self.knee_angles: list[float] = []
        self.knee_lock_flags: list[float] = []
        self.wrist_rel: list[list[float]] = []
        self.trunk_scales: list[float] = []

    def process_frame(self, frame_image, keypoints_dict: dict[str, dict[str, float]]) -> None:
        frame_idx = self.processed_frames
        self.processed_frames += 1
        self.frame_indices.append(frame_idx)

        alignment = compute_center_alignment(keypoints_dict)
        if bool(alignment.get('ok')):
            self.alignment_offsets.append(float(alignment['offset_px']))
            self.alignment_tolerances.append(float(alignment['tolerance_px']))
            self.alignment_scores.append(float(alignment['score']))
            self.alignment_flags.append(1.0 if bool(alignment['aligned']) else 0.0)
        else:
            self.alignment_offsets.append(float('nan'))
            self.alignment_tolerances.append(float('nan'))
            self.alignment_scores.append(float('nan'))
            self.alignment_flags.append(float('nan'))

        knee_values: list[float] = []
        for side in ('left', 'right'):
            hip_xy = joint_xy(keypoints_dict, f'{side}_hip', conf_min=ALIGNMENT_CONF_MIN)
            knee_xy = joint_xy(keypoints_dict, f'{side}_knee', conf_min=ALIGNMENT_CONF_MIN)
            ankle_xy = joint_xy(keypoints_dict, f'{side}_ankle', conf_min=ALIGNMENT_CONF_MIN)
            angle = knee_angle_deg(hip_xy, knee_xy, ankle_xy)
            if np.isfinite(angle):
                knee_values.append(float(angle))
        if knee_values:
            mean_knee_angle = float(np.mean(knee_values))
            self.knee_angles.append(mean_knee_angle)
            self.knee_lock_flags.append(1.0 if mean_knee_angle >= KNEE_LOCK_THRESHOLD_DEG else 0.0)
        else:
            self.knee_angles.append(float('nan'))
            self.knee_lock_flags.append(float('nan'))

        pelvis_left = joint_xy(keypoints_dict, 'left_hip', conf_min=ALIGNMENT_CONF_MIN)
        pelvis_right = joint_xy(keypoints_dict, 'right_hip', conf_min=ALIGNMENT_CONF_MIN)
        left_shoulder = joint_xy(keypoints_dict, 'left_shoulder', conf_min=ALIGNMENT_CONF_MIN)
        right_shoulder = joint_xy(keypoints_dict, 'right_shoulder', conf_min=ALIGNMENT_CONF_MIN)
        left_wrist = joint_xy(keypoints_dict, 'left_wrist', conf_min=ALIGNMENT_CONF_MIN)
        right_wrist = joint_xy(keypoints_dict, 'right_wrist', conf_min=ALIGNMENT_CONF_MIN)
        left_ankle = joint_xy(keypoints_dict, 'left_ankle', conf_min=ALIGNMENT_CONF_MIN)
        right_ankle = joint_xy(keypoints_dict, 'right_ankle', conf_min=ALIGNMENT_CONF_MIN)

        pelvis_points = np.stack([pelvis_left, pelvis_right], axis=0)
        valid_pelvis = np.isfinite(pelvis_points).all(axis=1)
        if int(valid_pelvis.sum()) > 0:
            pelvis_xy = np.mean(pelvis_points[valid_pelvis], axis=0)
        else:
            pelvis_xy = np.asarray([np.nan, np.nan], dtype=np.float32)

        wrist_points = np.stack([left_wrist, right_wrist], axis=0)
        valid_wrist = np.isfinite(wrist_points).all(axis=1)
        if np.isfinite(pelvis_xy).all() and int(valid_wrist.sum()) > 0:
            wrist_xy = np.mean(wrist_points[valid_wrist], axis=0)
            wrist_rel = wrist_xy - pelvis_xy
            self.wrist_rel.append([float(wrist_rel[0]), float(wrist_rel[1])])
        else:
            self.wrist_rel.append([float('nan'), float('nan')])

        scale_candidates = [
            point_distance(left_shoulder, right_shoulder),
            point_distance(pelvis_left, pelvis_right),
            point_distance(left_ankle, right_ankle),
        ]
        valid_scale = [value for value in scale_candidates if np.isfinite(value) and value > 1.0]
        self.trunk_scales.append(float(np.mean(valid_scale)) if valid_scale else float('nan'))

    def finalize(self) -> dict[str, Any]:
        self.context.review_dir.mkdir(parents=True, exist_ok=True)
        configure_plot_fonts()

        frame_idx = np.asarray(self.frame_indices, dtype=np.int32)
        alignment_offsets = np.asarray(self.alignment_offsets, dtype=np.float32)
        alignment_tolerances = np.asarray(self.alignment_tolerances, dtype=np.float32)
        alignment_scores = np.asarray(self.alignment_scores, dtype=np.float32)
        alignment_flags = np.asarray(self.alignment_flags, dtype=np.float32)
        knee_angles = np.asarray(self.knee_angles, dtype=np.float32)
        knee_lock_flags = np.asarray(self.knee_lock_flags, dtype=np.float32)
        wrist_rel = np.asarray(self.wrist_rel, dtype=np.float32)
        trunk_scales = np.asarray(self.trunk_scales, dtype=np.float32)

        fps = max(float(self.context.fps or 30.0), 1.0)
        filled_wrist_rel = interp_nan_2d(wrist_rel, fill_value=0.0)
        filled_trunk_scales = interp_nan_1d(trunk_scales, fill_value=float(np.nanmedian(trunk_scales[np.isfinite(trunk_scales)])) if np.isfinite(trunk_scales).any() else 1.0)
        filled_trunk_scales = np.maximum(filled_trunk_scales, 1.0)
        velocity = np.gradient(filled_wrist_rel, 1.0 / fps, axis=0)
        acceleration = np.gradient(velocity, 1.0 / fps, axis=0)
        trunk_acc_norm = np.linalg.norm(acceleration, axis=1) / filled_trunk_scales
        invalid_trunk = ~np.isfinite(wrist_rel).all(axis=1)
        trunk_acc_norm = trunk_acc_norm.astype(np.float32)
        trunk_acc_norm[invalid_trunk] = np.nan

        alignment_score = float(np.nanmean(alignment_scores) * 100.0) if np.isfinite(alignment_scores).any() else 0.0
        alignment_rate = float(np.nanmean(alignment_flags) * 100.0) if np.isfinite(alignment_flags).any() else 0.0
        mean_alignment_offset = float(np.nanmean(np.abs(alignment_offsets))) if np.isfinite(alignment_offsets).any() else float('nan')
        mean_knee_angle = float(np.nanmean(knee_angles)) if np.isfinite(knee_angles).any() else float('nan')
        knee_lock_ratio = float(np.nanmean(knee_lock_flags) * 100.0) if np.isfinite(knee_lock_flags).any() else 0.0
        trunk_compensation_ratio = float(np.nanmean(trunk_acc_norm >= TRUNK_ACCEL_THRESHOLD) * 100.0) if np.isfinite(trunk_acc_norm).any() else 0.0
        trunk_acc_p90 = float(np.nanpercentile(trunk_acc_norm, 90)) if np.isfinite(trunk_acc_norm).any() else 0.0
        stability_component = max(0.0, 1.0 - min(trunk_acc_p90 / (TRUNK_ACCEL_THRESHOLD * 2.0), 1.0))
        ratio_component = max(0.0, 1.0 - trunk_compensation_ratio / 100.0)
        trunk_stability_score = float(np.clip(100.0 * (0.65 * stability_component + 0.35 * ratio_component), 0.0, 100.0))

        notes: list[str] = []
        if alignment_rate < 72.0:
            notes.append('重心经常偏离脚踝支撑中线，建议先稳定骨盆投影。')
        if knee_lock_ratio > 28.0:
            notes.append('膝关节出现明显死锁，建议增加踝膝髋联动缓冲。')
        if trunk_compensation_ratio > 24.0:
            notes.append('上肢相对骨盆加速度偏高，存在手臂代偿风险。')
        if not notes:
            notes.append('初学者关键几何状态整体平稳，可进入下一轮节奏训练。')

        metrics_payload = {
            'engine': 'casi_beginner',
            'pipeline_id': self.context.pipeline_id,
            'frame_count': int(len(frame_idx)),
            'fps': round(fps, 3),
            'alignment_score': round(alignment_score, 1),
            'alignment_rate': round(alignment_rate, 1),
            'mean_alignment_offset_px': None if not np.isfinite(mean_alignment_offset) else round(mean_alignment_offset, 3),
            'knee_lock_ratio': round(knee_lock_ratio, 1),
            'mean_knee_angle_deg': None if not np.isfinite(mean_knee_angle) else round(mean_knee_angle, 3),
            'knee_lock_threshold_deg': KNEE_LOCK_THRESHOLD_DEG,
            'trunk_stability_score': round(trunk_stability_score, 1),
            'trunk_compensation_ratio': round(trunk_compensation_ratio, 1),
            'trunk_acceleration_p90': round(trunk_acc_p90, 3),
            'trunk_acceleration_threshold': TRUNK_ACCEL_THRESHOLD,
            'coaching_notes': notes,
        }

        metrics_path = self.context.review_dir / METRICS_FILENAME
        metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2))

        compatibility_metrics_path = self.context.pressure_metrics_path or (self.context.review_dir / 'pressure_curve_metrics.json')
        compatibility_metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2))

        csv_path = self.context.review_dir / CSV_FILENAME
        compatibility_csv_path = self.context.pressure_csv_path or (self.context.review_dir / 'pressure_curve_timeseries.csv')
        with csv_path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow([
                'frame_idx',
                'alignment_offset_px',
                'alignment_tolerance_px',
                'alignment_score',
                'mean_knee_angle_deg',
                'knee_lock_flag',
                'wrist_rel_x',
                'wrist_rel_y',
                'trunk_scale',
                'trunk_acc_norm',
            ])
            for idx in range(len(frame_idx)):
                writer.writerow([
                    int(frame_idx[idx]),
                    _safe_csv(alignment_offsets[idx]),
                    _safe_csv(alignment_tolerances[idx]),
                    _safe_csv(alignment_scores[idx]),
                    _safe_csv(knee_angles[idx]),
                    _safe_csv(knee_lock_flags[idx]),
                    _safe_csv(wrist_rel[idx, 0]),
                    _safe_csv(wrist_rel[idx, 1]),
                    _safe_csv(trunk_scales[idx]),
                    _safe_csv(trunk_acc_norm[idx]),
                ])
        compatibility_csv_path.write_text(csv_path.read_text())

        plot_path = self.context.review_dir / PLOT_FILENAME
        compatibility_plot_path = self.context.pressure_png_path or (self.context.review_dir / 'pressure_comparison_analysis.png')
        self._render_plot(
            frame_idx=frame_idx,
            alignment_offsets=alignment_offsets,
            alignment_tolerances=alignment_tolerances,
            knee_angles=knee_angles,
            trunk_acc_norm=trunk_acc_norm,
            plot_path=plot_path,
        )
        if compatibility_plot_path != plot_path:
            compatibility_plot_path.write_bytes(plot_path.read_bytes())

        return {
            'evaluator': 'casi',
            'metrics_path': str(metrics_path),
            'plot_path': str(plot_path),
            'csv_path': str(csv_path),
        }

    def _render_plot(
        self,
        *,
        frame_idx: np.ndarray,
        alignment_offsets: np.ndarray,
        alignment_tolerances: np.ndarray,
        knee_angles: np.ndarray,
        trunk_acc_norm: np.ndarray,
        plot_path: Path,
    ) -> None:
        plt.style.use('seaborn-v0_8-whitegrid')
        configure_plot_fonts()
        figure, axes = plt.subplots(3, 1, figsize=(13.5, 9.0), sharex=True)

        axes[0].plot(frame_idx, alignment_offsets, color='#2563eb', linewidth=2.0, label='骨盆偏移')
        axes[0].plot(frame_idx, alignment_tolerances, color='#22c55e', linewidth=1.2, linestyle='--', label='+容差')
        axes[0].plot(frame_idx, -alignment_tolerances, color='#22c55e', linewidth=1.2, linestyle='--', label='-容差')
        axes[0].set_ylabel('Offset (px)')
        axes[0].set_title('CASI 重心对齐度')
        axes[0].legend(loc='upper right')

        axes[1].plot(frame_idx, knee_angles, color='#f97316', linewidth=2.0, label='平均膝角')
        axes[1].axhline(KNEE_LOCK_THRESHOLD_DEG, color='#dc2626', linewidth=1.2, linestyle='--', label='死锁阈值')
        axes[1].set_ylabel('Angle (deg)')
        axes[1].set_title('膝关节直立警告')
        axes[1].legend(loc='lower right')

        axes[2].plot(frame_idx, trunk_acc_norm, color='#8b5cf6', linewidth=2.0, label='手腕-骨盆相对加速度')
        axes[2].axhline(TRUNK_ACCEL_THRESHOLD, color='#dc2626', linewidth=1.2, linestyle='--', label='代偿阈值')
        axes[2].set_ylabel('Norm Acc')
        axes[2].set_xlabel('Frame')
        axes[2].set_title('躯干静定状态')
        axes[2].legend(loc='upper right')

        figure.suptitle(f'CASI 新手模式 · {self.context.run_name}', fontsize=16, fontweight='bold')
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        figure.savefig(plot_path, dpi=160)
        plt.close(figure)


def _safe_csv(value: float) -> str:
    return '' if not np.isfinite(value) else f'{float(value):.6f}'
