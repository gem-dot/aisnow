#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import video_keypoint_tracker as v


def parse_ranges(spec: str):
    ranges = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        start_str, end_str = part.split('-', 1)
        start = int(start_str)
        end = int(end_str)
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    ranges.sort()
    merged = []
    for start, end in ranges:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def extract_clip(input_path: Path, clip_path: Path, start_frame: int, end_frame: int):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频: {input_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, min(start_frame, total - 1))
    end_frame = max(start_frame, min(end_frame, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_idx += 1
    writer.release()
    cap.release()
    return start_frame, frame_idx - 1


def load_track_payload(path: Path):
    payload = v.load_tensor_dict(path)
    if not isinstance(payload, dict) or 'boxes_xyxy' not in payload:
        raise RuntimeError(f'跟踪缓存格式错误: {path}')
    boxes = np.asarray(payload['boxes_xyxy'], dtype=np.float32)
    found = np.asarray(payload.get('found_mask', np.ones((len(boxes),), dtype=np.bool_)), dtype=bool)
    crowd = np.asarray(payload.get('crowd_mask', np.zeros((len(boxes),), dtype=np.bool_)), dtype=bool)
    return payload, boxes, found, crowd


def load_keypoint_payload(path: Path):
    payload = v.load_tensor_dict(path)
    if not isinstance(payload, dict) or 'raw_keypoints' not in payload:
        raise RuntimeError(f'关键点缓存格式错误: {path}')
    raw = np.asarray(payload['raw_keypoints'], dtype=np.float32)
    keypoints = np.asarray(payload.get('keypoints', raw), dtype=np.float32)
    return payload, raw, keypoints


def interpolate_box(left_box: np.ndarray | None, right_box: np.ndarray | None, ratio: float):
    if left_box is None and right_box is None:
        return None
    if left_box is None:
        return np.asarray(right_box, dtype=np.float32)
    if right_box is None:
        return np.asarray(left_box, dtype=np.float32)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return (np.asarray(left_box, dtype=np.float32) * (1.0 - ratio) + np.asarray(right_box, dtype=np.float32) * ratio).astype(np.float32)


def build_repair_local_bbx_xys(full_track_boxes: np.ndarray, actual_start: int, actual_end: int, start_frame: int, end_frame: int):
    clip_boxes = full_track_boxes[actual_start:actual_end + 1].astype(np.float32).copy()
    clip_xys = np.stack([v.bbox_xyxy_to_xys(box) for box in clip_boxes], axis=0)
    patch_start = max(0, start_frame - actual_start)
    patch_end = min(len(clip_boxes), end_frame - actual_start + 1)
    if patch_end <= patch_start:
        return clip_xys

    left_box = clip_boxes[patch_start - 1].copy() if patch_start - 1 >= 0 else None
    right_box = clip_boxes[patch_end].copy() if patch_end < len(clip_boxes) else None
    patch_len = max(patch_end - patch_start, 1)

    use_extrap = False
    left_center = None
    left_size = 0.0
    center_vel = np.zeros((2,), dtype=np.float32)
    size_vel = 0.0
    if left_box is not None and patch_start >= 3:
        history = clip_boxes[max(0, patch_start - 6):patch_start]
        if len(history) >= 2:
            hist_centers = np.stack([v.box_center(box) for box in history], axis=0)
            hist_sizes = np.asarray([v.box_size(box) for box in history], dtype=np.float32)
            center_deltas = np.diff(hist_centers, axis=0)
            size_deltas = np.diff(hist_sizes)
            center_vel = np.median(center_deltas, axis=0).astype(np.float32)
            size_vel = float(np.median(size_deltas)) if len(size_deltas) > 0 else 0.0
            if float(np.linalg.norm(center_vel)) >= 1.0:
                use_extrap = True
                left_center = v.box_center(left_box)
                left_size = v.box_size(left_box)

    for local_idx in range(patch_start, patch_end):
        if use_extrap:
            step = float(local_idx - patch_start + 1)
            center = left_center + center_vel * step
            size = float(np.clip(left_size + size_vel * step, left_size * 0.72, left_size * 1.35))
            base_box = np.asarray([center[0] - size * 0.5, center[1] - size * 0.5, center[0] + size * 0.5, center[1] + size * 0.5], dtype=np.float32)
        else:
            ratio = float((local_idx - patch_start + 1) / (patch_len + 1))
            base_box = interpolate_box(left_box, right_box, ratio)
            if base_box is None:
                base_box = clip_boxes[local_idx].copy()
        base_box = np.asarray(base_box, dtype=np.float32)
        center = v.box_center(base_box)
        size = max(v.box_size(base_box), 1.0)
        clip_xys[local_idx] = np.asarray([center[0], center[1], size], dtype=np.float32)

    return clip_xys.astype(np.float32)


def repair_video(args):
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    track_cache_path = Path(args.track_cache).expanduser().resolve()
    keypoint_cache_path = Path(args.keypoint_cache).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = output_dir / 'review'
    review_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / '_repair_work'
    work_dir.mkdir(parents=True, exist_ok=True)

    track_payload, full_track_boxes, full_found_mask, full_crowd_mask = load_track_payload(track_cache_path)
    keypoint_payload, full_raw_keypoints, _ = load_keypoint_payload(keypoint_cache_path)
    frame_count = min(len(full_track_boxes), len(full_raw_keypoints))
    full_track_boxes = full_track_boxes[:frame_count].copy()
    full_found_mask = full_found_mask[:frame_count].copy()
    full_crowd_mask = full_crowd_mask[:frame_count].copy()
    full_raw_keypoints = full_raw_keypoints[:frame_count].copy()

    sr_model = None if args.disable_sr else v.load_sr_model(Path(v.REAL_ESRGAN_PATH))
    ranges = parse_ranges(args.ranges)
    pre_roll = max(0, int(args.pre_roll))
    post_roll = max(0, int(args.post_roll))

    for range_idx, (start_frame, end_frame) in enumerate(ranges):
        clip_start = max(0, start_frame - pre_roll)
        clip_end = min(frame_count - 1, end_frame + post_roll)
        clip_video_path = work_dir / f'clip_{range_idx}_{clip_start}_{clip_end}.mp4'
        actual_start, actual_end = extract_clip(input_path, clip_video_path, clip_start, clip_end)
        clip_len = actual_end - actual_start + 1
        print(f'[Repair] clip={clip_video_path.name} frames={actual_start}-{actual_end} len={clip_len}')

        clip_track_cache = work_dir / f'clip_{range_idx}_{clip_start}_{clip_end}_track.pt'
        clip_keypoint_cache = work_dir / f'clip_{range_idx}_{clip_start}_{clip_end}_kp.pt'
        seed_arr = full_raw_keypoints[actual_start:actual_end + 1].copy()
        clip_local_bbx_xys = build_repair_local_bbx_xys(full_track_boxes, actual_start, actual_end, start_frame, end_frame)

        clip_track_boxes, clip_found_mask, clip_crowd_mask = v.track_target_bboxes(
            input_path=clip_video_path,
            detector_model_path=Path(v.YOLO_DETECTOR_PATH),
            tracker_config_path=Path(v.TRACKER_CONFIG_PATH),
            cache_path=clip_track_cache,
            local_bbx_xys=clip_local_bbx_xys,
            imgsz=v.YOLO_IMGSZ,
            conf=v.YOLO_CONF,
            max_frames=0,
            force=True,
        )
        clip_raw_keypoints = v.generate_keypoints(
            input_path=clip_video_path,
            track_boxes=clip_track_boxes,
            crowd_mask=clip_crowd_mask,
            keypoint_cache_path=clip_keypoint_cache,
            pose_model_ref=v.RTMPOSE_MODEL_URL,
            seed_arr=seed_arr,
            sr_model=sr_model,
            small_box_threshold=v.SMALL_BOX_THRESHOLD,
            max_frames=0,
            force=True,
        )

        patch_start = start_frame - actual_start
        patch_end = end_frame - actual_start + 1
        full_slice = slice(start_frame, end_frame + 1)
        clip_slice = slice(patch_start, patch_end)
        full_track_boxes[full_slice] = clip_track_boxes[clip_slice]
        full_found_mask[full_slice] = clip_found_mask[clip_slice]
        full_crowd_mask[full_slice] = clip_crowd_mask[clip_slice]
        full_raw_keypoints[full_slice] = clip_raw_keypoints[clip_slice]
        print(f'[Repair] patched frames={start_frame}-{end_frame}')

    refined_keypoints = v.offline_temporal_refine(full_raw_keypoints.astype(np.float32), full_track_boxes.astype(np.float32))
    refined_keypoints = v.reduce_temporal_lag(refined_keypoints.astype(np.float32), full_raw_keypoints.astype(np.float32), full_track_boxes.astype(np.float32))
    refined_keypoints = v.repair_brief_outliers(refined_keypoints.astype(np.float32), full_raw_keypoints.astype(np.float32), full_track_boxes.astype(np.float32))
    refined_keypoints = v.enforce_track_box_consistency(refined_keypoints.astype(np.float32), full_track_boxes.astype(np.float32))

    output_track_cache = output_dir / f'{args.name}_bytetrack_boxes.pt'
    output_keypoint_cache = output_dir / f'{args.name}_rtmpose_keypoints.pt'
    v.save_tensor_dict(output_track_cache, {
        'boxes_xyxy': torch.from_numpy(full_track_boxes.astype(np.float32)),
        'found_mask': torch.from_numpy(full_found_mask.astype(np.bool_)),
        'crowd_mask': torch.from_numpy(full_crowd_mask.astype(np.bool_)),
    })
    v.save_tensor_dict(output_keypoint_cache, {
        'raw_keypoints': torch.from_numpy(full_raw_keypoints.astype(np.float32)),
        'keypoints': torch.from_numpy(refined_keypoints.astype(np.float32)),
        'track_boxes': torch.from_numpy(full_track_boxes.astype(np.float32)),
        'found_mask': torch.from_numpy(full_found_mask.astype(np.bool_)),
        'crowd_mask': torch.from_numpy(full_crowd_mask.astype(np.bool_)),
    })

    overlay_path = output_dir / f'{args.name}_overlay.mp4'
    compare_path = output_dir / f'{args.name}_side_by_side.mp4'
    slow_overlay_path = output_dir / f'{args.name}_overlay_slow.mp4'
    slow_compare_path = output_dir / f'{args.name}_side_by_side_slow.mp4'
    v.render_from_keypoints(
        input_path,
        overlay_path,
        compare_path,
        refined_keypoints.astype(np.float32),
        model_label='patched ranges + rtmpose-x + track-box constraint + temporal refine',
        track_boxes=full_track_boxes.astype(np.float32),
        slow_overlay_path=slow_overlay_path,
        slow_compare_path=slow_compare_path,
        slow_factor=args.slow_factor,
    )

    qa_cmd = [
        sys.executable,
        'qa_review.py',
        '--keypoint-cache', str(output_keypoint_cache),
        '--compare-video', str(compare_path),
        '--review-dir', str(review_dir),
    ]
    subprocess.run(qa_cmd, check=True, cwd=str(Path(__file__).resolve().parent))
    print(f'[Done] overlay={overlay_path}')
    print(f'[Done] compare={compare_path}')
    print(f'[Done] keypoint_cache={output_keypoint_cache}')


def build_parser():
    parser = argparse.ArgumentParser(description='Repair video outputs by recomputing selected frame ranges.')
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--track-cache', required=True, type=str)
    parser.add_argument('--keypoint-cache', required=True, type=str)
    parser.add_argument('--ranges', required=True, type=str)
    parser.add_argument('--pre-roll', type=int, default=36)
    parser.add_argument('--post-roll', type=int, default=6)
    parser.add_argument('--slow-factor', type=float, default=0.30)
    parser.add_argument('--disable-sr', action='store_true')
    return parser


def main():
    args = build_parser().parse_args()
    repair_video(args)


if __name__ == '__main__':
    main()
