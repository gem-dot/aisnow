from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import OUTPUT_DIR, PYTHON_BIN, RUN_PIPELINE_SCRIPT


@dataclass(frozen=True)
class PipelineContext:
    task_name: str
    input_path: Path
    output_dir: Path
    cache_dir: Path


@dataclass(frozen=True)
class PipelineExecution:
    command: list[str]
    track_cache: Path
    keypoint_cache: Path
    compare_path: Path
    output_dir: Path


@dataclass(frozen=True)
class PipelineSpec:
    pipeline_id: str
    label: str
    description: str
    analysis_style: str = 'default'
    evaluator_id: str = 'jsba'
    qa_enabled: bool = True

    def to_meta(self) -> dict[str, str | bool]:
        return {
            'id': self.pipeline_id,
            'label': self.label,
            'description': self.description,
            'analysis_style': self.analysis_style,
            'evaluator_id': self.evaluator_id,
            'qa_enabled': self.qa_enabled,
        }


DEFAULT_PIPELINE_ID = 'jsba'
LEGACY_PIPELINE_ALIASES = {
    'model1': 'jsba',
    'model2': 'saj',
    'model3': 'advanced',
}


PIPELINES: dict[str, PipelineSpec] = {
    'jsba': PipelineSpec(
        pipeline_id='jsba',
        label='JSBA · 技术滑行',
        description='竞赛/技术滑行评估，保持现有 JSBA 判读与报告风格。',
        analysis_style='default',
        evaluator_id='jsba',
        qa_enabled=True,
    ),
    'saj': PipelineSpec(
        pipeline_id='saj',
        label='SAJ · Infinity 连续圆弧',
        description='教材教学风格，强调连续圆弧、换刃不断刃、过渡压力不断线。',
        analysis_style='infinity',
        evaluator_id='jsba',
        qa_enabled=True,
    ),
    'advanced': PipelineSpec(
        pipeline_id='advanced',
        label='进阶 · 高阶诊断',
        description='面向高阶用户；当前先复用识别底座，后续扩展更细粒度动作与节奏诊断。',
        analysis_style='default',
        evaluator_id='jsba',
        qa_enabled=True,
    ),
    'casi': PipelineSpec(
        pipeline_id='casi',
        label='CASI · 新手模式',
        description='面向初学者的几何状态快检，聚焦重心对齐、膝关节死锁和躯干代偿。',
        analysis_style='default',
        evaluator_id='casi',
        qa_enabled=True,
    ),
}


def list_pipelines() -> list[dict[str, str | bool]]:
    return [spec.to_meta() for spec in PIPELINES.values()]


def normalize_pipeline_id(pipeline_id: str | None) -> str:
    raw = str(pipeline_id or '').strip()
    if raw in PIPELINES:
        return raw
    if raw in LEGACY_PIPELINE_ALIASES:
        return LEGACY_PIPELINE_ALIASES[raw]
    return DEFAULT_PIPELINE_ID


def get_pipeline(pipeline_id: str | None) -> PipelineSpec:
    return PIPELINES[normalize_pipeline_id(pipeline_id)]


def prepare_pipeline_execution(
    pipeline_id: str | None,
    *,
    task_name: str,
    input_path: Path,
    output_dir: Path,
    cache_dir: Path,
) -> PipelineExecution:
    spec = get_pipeline(pipeline_id)
    context = PipelineContext(
        task_name=task_name,
        input_path=input_path,
        output_dir=output_dir,
        cache_dir=cache_dir,
    )
    return _prepare_pipeline_execution(context, spec)


def _prepare_pipeline_execution(context: PipelineContext, spec: PipelineSpec) -> PipelineExecution:
    track_cache = context.cache_dir / f'{context.task_name}_bytetrack_boxes.pt'
    keypoint_cache = context.cache_dir / f'{context.task_name}_rtmpose_keypoints.pt'
    compare_path = context.output_dir / f'{context.task_name}_side_by_side.mp4'
    command = [
        str(PYTHON_BIN),
        str(RUN_PIPELINE_SCRIPT),
        str(context.input_path),
        '--name', context.task_name,
        '--output-root', str(OUTPUT_DIR),
        '--track-cache', str(track_cache),
        '--keypoint-cache', str(keypoint_cache),
        '--pipeline-id', spec.pipeline_id,
        '--analysis-style', spec.analysis_style,
        '--force',
    ]
    return PipelineExecution(
        command=command,
        track_cache=track_cache,
        keypoint_cache=keypoint_cache,
        compare_path=compare_path,
        output_dir=context.output_dir,
    )
