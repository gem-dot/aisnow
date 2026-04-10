#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd: Path):
    print('[Run]', ' '.join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main():
    parser = argparse.ArgumentParser(description='Batch run Ski_Avatar_Pro pipeline + QA')
    parser.add_argument('videos', nargs='+', help='Input video paths relative to project root, e.g. input/发发1.mov')
    parser.add_argument('--output-root', default='output', help='Output root directory')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    output_root = (project_root / args.output_root).resolve()
    cache_root = output_root / '.cache'
    python_bin = sys.executable

    for video in args.videos:
        input_path = (project_root / video).resolve()
        run_name = input_path.stem
        run_dir = output_root / run_name
        cache_dir = cache_root / run_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        track_cache = cache_dir / f'{run_name}_bytetrack_boxes.pt'
        keypoint_cache = cache_dir / f'{run_name}_rtmpose_keypoints.pt'

        run([
            python_bin,
            'run_ski_pipeline.py',
            str(input_path),
            '--name', run_name,
            '--output-root', str(output_root),
            '--track-cache', str(track_cache),
            '--keypoint-cache', str(keypoint_cache),
            '--force',
        ], project_root)

        run([
            python_bin,
            'qa_review.py',
            '--keypoint-cache', str(keypoint_cache),
            '--compare-video', str(run_dir / f'{run_name}_side_by_side.mp4'),
            '--review-dir', str(run_dir / 'review'),
        ], project_root)

    print('[Done] batch run complete')


if __name__ == '__main__':
    main()
