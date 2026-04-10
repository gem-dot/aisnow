from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .base import BaseEvaluator


class JSBAEvaluator(BaseEvaluator):
    streaming_required = False

    def process_frame(self, frame_image, keypoints_dict: dict[str, dict[str, float]]) -> None:
        self.processed_frames += 1

    def finalize(self) -> dict[str, Any]:
        proj_root = Path(__file__).resolve().parent.parent
        pressure_script = proj_root / 'pressure_curve_analysis.py'
        if not pressure_script.exists():
            raise FileNotFoundError(f'评估脚本不存在: {pressure_script}')

        self.context.review_dir.mkdir(parents=True, exist_ok=True)
        title_prefix_map = {
            'infinity': 'Infinity 连续圆弧压力分析',
            'advanced': '进阶技术诊断',
        }
        title_prefix = title_prefix_map.get(self.context.analysis_style, 'JSBA 压力曲线对比分析')
        if self.context.pipeline_id == 'advanced':
            title_prefix = '进阶技术诊断'

        command = [
            self.context.python_bin,
            str(pressure_script),
            '--keypoint-cache', str(self.context.keypoint_cache_path),
            '--video', str(self.context.input_path),
            '--output', str(self.context.pressure_png_path),
            '--csv-output', str(self.context.pressure_csv_path),
            '--metrics-output', str(self.context.pressure_metrics_path),
            '--video-source', str(self.context.overlay_path),
            '--video-output', str(self.context.pressure_sync_video_path),
            '--title', f'{title_prefix} · {self.context.run_name}',
            '--style', str(self.context.analysis_style),
        ]
        print('[Evaluator Command]', ' '.join(command), flush=True)
        subprocess.run(command, check=True, cwd=str(proj_root))
        return {
            'evaluator': 'jsba',
            'metrics_path': str(self.context.pressure_metrics_path) if self.context.pressure_metrics_path else '',
            'plot_path': str(self.context.pressure_png_path) if self.context.pressure_png_path else '',
            'csv_path': str(self.context.pressure_csv_path) if self.context.pressure_csv_path else '',
            'preview_video_path': str(self.context.pressure_sync_video_path) if self.context.pressure_sync_video_path else '',
        }
