#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch


VALID_CONF = 0.20


def load_tensor_dict(path: Path):
    return torch.load(path, map_location='cpu')


def to_numpy(value, default=None, dtype=np.float32):
    if value is None:
        return default
    return np.asarray(value, dtype=dtype)


def box_size(box_xyxy):
    if box_xyxy is None:
        return 1.0
    return max(float(box_xyxy[2] - box_xyxy[0]), float(box_xyxy[3] - box_xyxy[1]), 1.0)


def clamp_int(value, low, high):
    return max(low, min(high, int(value)))


def outside_ratio(points_xy, conf, box_xyxy, pad_ratio=0.08):
    if box_xyxy is None:
        return 0.0
    valid = conf >= VALID_CONF
    if int(valid.sum()) == 0:
        return 1.0
    size = box_size(box_xyxy)
    pad = size * pad_ratio
    x1 = float(box_xyxy[0] - pad)
    y1 = float(box_xyxy[1] - pad)
    x2 = float(box_xyxy[2] + pad)
    y2 = float(box_xyxy[3] + pad)
    pts = points_xy[valid]
    out = (pts[:, 0] < x1) | (pts[:, 0] > x2) | (pts[:, 1] < y1) | (pts[:, 1] > y2)
    return float(np.mean(out.astype(np.float32)))


def compute_frame_metrics(keypoints, raw_keypoints, track_boxes, found_mask, crowd_mask):
    frame_count = len(keypoints)
    rows = []
    prev_xy = None
    prev_valid = None
    for frame_idx in range(frame_count):
        kp = keypoints[frame_idx]
        raw_kp = raw_keypoints[frame_idx] if raw_keypoints is not None and frame_idx < len(raw_keypoints) else kp
        box = track_boxes[frame_idx] if track_boxes is not None and frame_idx < len(track_boxes) else None
        found = bool(found_mask[frame_idx]) if found_mask is not None and frame_idx < len(found_mask) else True
        crowded = bool(crowd_mask[frame_idx]) if crowd_mask is not None and frame_idx < len(crowd_mask) else False

        conf = kp[:, 2].astype(np.float32)
        valid = conf >= VALID_CONF
        valid_count = int(valid.sum())
        mean_conf = float(conf[valid].mean()) if valid_count > 0 else 0.0
        min_conf = float(conf[valid].min()) if valid_count > 0 else 0.0

        size = box_size(box)
        motion_norm = 0.0
        if prev_xy is not None:
            both_valid = valid & prev_valid
            if int(both_valid.sum()) > 0:
                motion = np.linalg.norm(kp[both_valid, :2] - prev_xy[both_valid], axis=1)
                motion_norm = float(np.median(motion) / size)

        raw_gap = float(np.mean(np.linalg.norm(kp[:, :2] - raw_kp[:, :2], axis=1)) / size)
        out_ratio = outside_ratio(kp[:, :2], conf, box)

        risk_score = 0.0
        reasons = []
        if not found:
            risk_score += 0.45 if crowded else 2.2
            reasons.append('track_lost_crowd' if crowded else 'track_lost')
        if valid_count < 12:
            risk_score += (12 - valid_count) * 0.20
            reasons.append('few_joints')
        if mean_conf < 0.45:
            risk_score += (0.45 - mean_conf) * 2.2
            reasons.append('low_conf')
        if motion_norm > 0.070:
            risk_score += min(1.8, (motion_norm - 0.070) * 10.0)
            reasons.append('jump')
        if raw_gap > 0.060:
            risk_score += min(1.4, (raw_gap - 0.060) * 8.0)
            reasons.append('raw_ref_gap')
        if out_ratio > 0.18:
            risk_score += min(1.6, out_ratio * 2.0)
            reasons.append('out_of_box')

        rows.append({
            'frame_idx': frame_idx,
            'risk_score': float(risk_score),
            'found': int(found),
            'valid_count': valid_count,
            'mean_conf': mean_conf,
            'min_conf': min_conf,
            'motion_norm': motion_norm,
            'raw_gap': raw_gap,
            'out_ratio': out_ratio,
            'reasons': '|'.join(dict.fromkeys(reasons)),
        })
        prev_xy = kp[:, :2].copy()
        prev_valid = valid.copy()
    return rows


def select_review_frames(rows, top_k=24, min_gap=6):
    ordered = sorted(rows, key=lambda item: (-item['risk_score'], item['frame_idx']))
    selected = []
    for row in ordered:
        if row['risk_score'] <= 0.0:
            break
        if all(abs(row['frame_idx'] - other['frame_idx']) >= min_gap for other in selected):
            selected.append(row)
        if len(selected) >= top_k:
            break
    return selected


def extract_contact_sheet(video_path: Path, rows, out_path: Path, cell_width=420):
    if not rows:
        return False
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    images = []
    for row in rows:
        frame_idx = int(row['frame_idx'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        scale = cell_width / frame.shape[1]
        cell = cv2.resize(frame, (cell_width, max(1, int(frame.shape[0] * scale))))
        label = f"f={frame_idx} score={row['risk_score']:.2f} {row['reasons']}"
        cv2.putText(cell, label[:60], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        images.append(cell)
    cap.release()
    if not images:
        return False

    cols = 2
    row_images = []
    for idx in range(0, len(images), cols):
        chunk = images[idx: idx + cols]
        max_h = max(image.shape[0] for image in chunk)
        padded = []
        for image in chunk:
            if image.shape[0] < max_h:
                image = cv2.copyMakeBorder(image, 0, max_h - image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            padded.append(image)
        if len(padded) < cols:
            blank = np.zeros_like(padded[0])
            padded.append(blank)
        row_images.append(cv2.hconcat(padded))

    max_w = max(image.shape[1] for image in row_images)
    aligned = []
    for image in row_images:
        if image.shape[1] < max_w:
            image = cv2.copyMakeBorder(image, 0, 0, 0, max_w - image.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        aligned.append(image)
    sheet = cv2.vconcat(aligned)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return True


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['frame_idx', 'risk_score', 'found', 'crowded', 'valid_count', 'mean_conf', 'min_conf', 'motion_norm', 'raw_gap', 'out_ratio', 'reasons']
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(rows, selected_rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    risk_values = np.asarray([row['risk_score'] for row in rows], dtype=np.float32) if rows else np.zeros((0,), dtype=np.float32)
    found_values = np.asarray([row['found'] for row in rows], dtype=np.float32) if rows else np.zeros((0,), dtype=np.float32)
    mean_conf_values = np.asarray([row['mean_conf'] for row in rows], dtype=np.float32) if rows else np.zeros((0,), dtype=np.float32)

    suspicious = [row for row in rows if row['risk_score'] >= 1.0]
    severe = [row for row in rows if row['risk_score'] >= 2.0]

    lines = [
        f'total_frames: {total}',
        f'track_found_rate: {float(found_values.mean()) if total else 0.0:.4f}',
        f'avg_mean_conf: {float(mean_conf_values.mean()) if total else 0.0:.4f}',
        f'max_risk_score: {float(risk_values.max()) if total else 0.0:.4f}',
        f'suspicious_frames(score>=1.0): {len(suspicious)}',
        f'severe_frames(score>=2.0): {len(severe)}',
        'top_flagged_frames:',
    ]
    for row in selected_rows[:12]:
        lines.append(
            f"  frame={row['frame_idx']} score={row['risk_score']:.3f} mean_conf={row['mean_conf']:.3f} found={row['found']} reasons={row['reasons']}"
        )
    out_path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Frame-level QA review for Ski_Avatar_Pro outputs')
    parser.add_argument('--keypoint-cache', required=True, type=str)
    parser.add_argument('--compare-video', required=True, type=str)
    parser.add_argument('--review-dir', required=True, type=str)
    args = parser.parse_args()

    cache_path = Path(args.keypoint_cache).expanduser()
    compare_video_path = Path(args.compare_video).expanduser()
    review_dir = Path(args.review_dir).expanduser()

    payload = load_tensor_dict(cache_path)
    if not isinstance(payload, dict):
        raise RuntimeError(f'缓存格式不正确: {cache_path}')

    keypoints = to_numpy(payload.get('keypoints'))
    raw_keypoints = to_numpy(payload.get('raw_keypoints'))
    track_boxes = to_numpy(payload.get('track_boxes'))
    found_mask = to_numpy(payload.get('found_mask'), dtype=bool)
    crowd_mask = to_numpy(payload.get('crowd_mask'), dtype=bool)
    if keypoints is None:
        raise RuntimeError(f'缓存缺少 keypoints: {cache_path}')
    if raw_keypoints is None:
        raw_keypoints = keypoints.copy()

    rows = compute_frame_metrics(keypoints, raw_keypoints, track_boxes, found_mask, crowd_mask)
    selected = select_review_frames(rows)

    write_csv(rows, review_dir / 'qa_flags.csv')
    write_summary(rows, selected, review_dir / 'qa_summary.txt')
    extract_contact_sheet(compare_video_path, selected, review_dir / 'qa_contact_sheet.jpg')

    print(f'[QA] review_dir={review_dir}')
    print(f'[QA] summary={review_dir / "qa_summary.txt"}')
    print(f'[QA] flags={review_dir / "qa_flags.csv"}')
    print(f'[QA] contact_sheet={review_dir / "qa_contact_sheet.jpg"}')


if __name__ == '__main__':
    main()
