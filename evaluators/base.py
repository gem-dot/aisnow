from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluatorContext:
    run_name: str
    pipeline_id: str
    analysis_style: str
    input_path: Path
    keypoint_cache_path: Path
    overlay_path: Path
    compare_path: Path
    review_dir: Path
    output_dir: Path
    python_bin: str
    fps: float = 30.0
    frame_count: int = 0
    pressure_png_path: Path | None = None
    pressure_csv_path: Path | None = None
    pressure_metrics_path: Path | None = None
    pressure_sync_video_path: Path | None = None


class BaseEvaluator(ABC):
    streaming_required = True

    def __init__(self, context: EvaluatorContext):
        self.context = context
        self.processed_frames = 0

    @abstractmethod
    def process_frame(self, frame_image, keypoints_dict: dict[str, dict[str, float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> dict[str, Any]:
        raise NotImplementedError
