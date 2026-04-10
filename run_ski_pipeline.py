#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from evaluators.base import EvaluatorContext
from evaluators.common import probe_video_metadata
from evaluators.router import run_evaluator

PROJ_ROOT = Path(__file__).resolve().parent
TRACKER_SCRIPT = PROJ_ROOT / 'video_keypoint_tracker.py'
DEFAULT_OUTPUT_ROOT = PROJ_ROOT / 'output' / 'runs'
PIPELINE_IDS = ('jsba', 'saj', 'advanced', 'casi')


def sanitize_name(name: str) -> str:
    safe = re.sub(r'[^0-9A-Za-z._-]+', '_', name.strip())
    safe = safe.strip('._-')
    return safe or 'ski_run'


def default_missing_marker(run_dir: Path, tag: str) -> Path:
    return run_dir / f'__missing__{tag}.pt'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='High-precision skiing keypoint pipeline wrapper.'
    )
    parser.add_argument('video', type=str, help='Input video path.')
    parser.add_argument('--name', type=str, default='', help='Optional run name; defaults to input stem.')
    parser.add_argument('--output-root', type=str, default=str(DEFAULT_OUTPUT_ROOT), help='Root folder for this run.')
    parser.add_argument('--python', type=str, default=sys.executable, help='Python executable used to launch tracker.')
    parser.add_argument('--slow-factor', type=float, default=0.30, help='Slow-motion output factor.')
    parser.add_argument('--with-slow-output', action='store_true', help='Also generate slow-motion overlay/compare videos.')
    parser.add_argument('--force', action='store_true', help='Force redo track and pose for this run.')
    parser.add_argument('--seed-kp', type=str, default='', help='Optional seed keypoints file for the same video.')
    parser.add_argument('--local-bbx', type=str, default='', help='Optional local bbox file for the same video.')
    parser.add_argument('--track-cache', type=str, default='', help='Optional explicit track cache path.')
    parser.add_argument('--keypoint-cache', type=str, default='', help='Optional explicit keypoint cache path.')
    parser.add_argument('--pipeline-id', type=str, choices=PIPELINE_IDS, default='jsba', help='Evaluator route preset.')
    parser.add_argument('--analysis-style', type=str, choices=('default', 'infinity'), default='default', help='Pressure analysis style preset.')
    parser.add_argument('--max-frames', type=int, default=0, help='Optional frame cap for quick smoke tests.')
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_path = Path(args.video).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f'输入视频不存在: {input_path}')
    if not TRACKER_SCRIPT.exists():
        raise FileNotFoundError(f'核心脚本不存在: {TRACKER_SCRIPT}')

    run_name = args.name.strip() if args.name.strip() else sanitize_name(input_path.stem)
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (PROJ_ROOT / output_root).resolve()
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = run_dir / f'{run_name}_overlay.mp4'
    compare_path = run_dir / f'{run_name}_side_by_side.mp4'
    slow_overlay_path = run_dir / f'{run_name}_overlay_slow.mp4'
    slow_compare_path = run_dir / f'{run_name}_side_by_side_slow.mp4'
    review_dir = run_dir / 'review'
    pressure_png_path = review_dir / 'pressure_comparison_analysis.png'
    pressure_csv_path = review_dir / 'pressure_curve_timeseries.csv'
    pressure_metrics_path = review_dir / 'pressure_curve_metrics.json'
    pressure_sync_video_path = run_dir / f'{run_name}_pressure_sync.mp4'
    track_cache_path = Path(args.track_cache).expanduser() if args.track_cache else run_dir / f'{run_name}_bytetrack_boxes.pt'
    keypoint_cache_path = Path(args.keypoint_cache).expanduser() if args.keypoint_cache else run_dir / f'{run_name}_rtmpose_keypoints.pt'

    seed_path = Path(args.seed_kp).expanduser() if args.seed_kp else default_missing_marker(run_dir, 'seed')
    local_bbx_path = Path(args.local_bbx).expanduser() if args.local_bbx else default_missing_marker(run_dir, 'bbx')

    tracker_cmd = [
        args.python,
        str(TRACKER_SCRIPT),
        '--input', str(input_path),
        '--output', str(overlay_path),
        '--compare-output', str(compare_path),
        '--track-cache', str(track_cache_path),
        '--keypoint-cache', str(keypoint_cache_path),
        '--seed-kp', str(seed_path),
        '--local-bbx', str(local_bbx_path),
    ]

    if args.with_slow_output:
        tracker_cmd.extend([
            '--slow-output', str(slow_overlay_path),
            '--slow-compare-output', str(slow_compare_path),
            '--slow-factor', str(args.slow_factor),
        ])

    if args.max_frames and args.max_frames > 0:
        tracker_cmd.extend(['--max-frames', str(int(args.max_frames))])

    if args.force:
        tracker_cmd.extend(['--force-redo-track', '--force-redo-pose'])

    print('[Run Name]', run_name)
    print('[Input]', input_path)
    print('[Run Dir]', run_dir)
    print('[Pipeline]', args.pipeline_id)
    print('[Mode] ByteTrack + RTMPose-X + evaluator-router')
    print('[Tracker Command]', ' '.join(tracker_cmd))

    subprocess.run(tracker_cmd, check=True, cwd=str(PROJ_ROOT))

    review_dir.mkdir(parents=True, exist_ok=True)
    fps, frame_count = probe_video_metadata(input_path)
    evaluation_context = EvaluatorContext(
        run_name=run_name,
        pipeline_id=args.pipeline_id,
        analysis_style=args.analysis_style,
        input_path=input_path,
        keypoint_cache_path=keypoint_cache_path,
        overlay_path=overlay_path,
        compare_path=compare_path,
        review_dir=review_dir,
        output_dir=run_dir,
        python_bin=args.python,
        fps=fps,
        frame_count=frame_count,
        pressure_png_path=pressure_png_path,
        pressure_csv_path=pressure_csv_path,
        pressure_metrics_path=pressure_metrics_path,
        pressure_sync_video_path=pressure_sync_video_path,
    )
    evaluator_result = run_evaluator(evaluation_context)
    print('[Evaluator]', evaluator_result.get('evaluator', 'unknown'))

    print('[Done] 输出文件如下:')
    print(f'  Overlay: {overlay_path}')
    print(f'  Compare: {compare_path}')
    if pressure_sync_video_path.exists():
        print(f'  Pressure Sync: {pressure_sync_video_path}')
    if pressure_png_path.exists():
        print(f'  Review Plot: {pressure_png_path}')
    if pressure_metrics_path.exists():
        print(f'  Review Metrics: {pressure_metrics_path}')
    if args.with_slow_output:
        print(f'  Slow Overlay: {slow_overlay_path}')
        print(f'  Slow Compare: {slow_compare_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
