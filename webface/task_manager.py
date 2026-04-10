from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import (
    LOG_TAIL_LIMIT,
    MAX_CONCURRENT_JOBS,
    OUTPUT_DIR,
    PROJ_ROOT,
    PYTHON_BIN,
    QA_SCRIPT,
    STAGE_WEIGHTS,
    TASK_DIR,
    UPLOAD_DIR,
)
from .pipelines import DEFAULT_PIPELINE_ID, get_pipeline, prepare_pipeline_execution

PROGRESS_RE = re.compile(r'(ByteTrack|RTMPose-X|Render Overlay):\s+(\d+)%')
SAFE_NAME_RE = re.compile(r'[^0-9A-Za-z._\-\u4e00-\u9fff]+')


OUTPUT_SUFFIXES = {
    'overlay': '_overlay.mp4',
    'side_by_side': '_side_by_side.mp4',
    'pressure_sync': '_pressure_sync.mp4',
}

ISSUE_REPORT_FILENAME = 'user_issue_notes.jsonl'

REVIEW_FILENAMES = {
    'contact_sheet': 'qa_contact_sheet.jpg',
    'summary': 'qa_summary.txt',
    'flags': 'qa_flags.csv',
    'critical_sheet': 'critical_side_by_side_sheet.jpg',
    'pressure_plot': 'pressure_comparison_analysis.png',
    'pressure_csv': 'pressure_curve_timeseries.csv',
    'pressure_metrics': 'pressure_curve_metrics.json',
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def sanitize_name(name: str) -> str:
    cleaned = SAFE_NAME_RE.sub('_', name.strip())
    return cleaned.strip('._-') or 'ski_task'


def file_url(path: Path) -> str:
    rel = path.resolve().relative_to(PROJ_ROOT.resolve())
    return '/files/' + '/'.join(rel.parts)


def chunked_copy(src, dst, chunk_size: int = 1024 * 1024):
    while True:
        chunk = src.read(chunk_size)
        if not chunk:
            break
        dst.write(chunk)


def resolve_output_dir(run_name: str) -> Path | None:
    try:
        resolved = (OUTPUT_DIR / run_name).resolve()
        resolved.relative_to(OUTPUT_DIR.resolve())
    except Exception:
        return None
    return resolved


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def load_metrics_summary(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    try:
        payload = json.loads(metrics_path.read_text())
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    summary: dict[str, Any] = {}
    turn_count = payload.get('turn_count')
    if isinstance(turn_count, (int, float)):
        summary['turn_count'] = int(turn_count)

    sync_score = _safe_float(payload.get('sync_score'))
    if sync_score is not None:
        summary['sync_score'] = round(sync_score, 1)

    correlation = _safe_float(payload.get('correlation'))
    if correlation is not None:
        summary['correlation'] = round(correlation, 3)

    rmse = _safe_float(payload.get('rmse'))
    if rmse is not None:
        summary['rmse'] = round(rmse, 3)

    mae = _safe_float(payload.get('mae'))
    if mae is not None:
        summary['mae'] = round(mae, 3)

    start = payload.get('analysis_window_start')
    end = payload.get('analysis_window_end')
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        summary['analysis_window'] = f'{int(start)}-{int(end)}'

    alignment_score = _safe_float(payload.get('alignment_score'))
    if alignment_score is not None:
        summary['alignment_score'] = round(alignment_score, 1)

    alignment_rate = _safe_float(payload.get('alignment_rate'))
    if alignment_rate is not None:
        summary['alignment_rate'] = round(alignment_rate, 1)

    knee_lock_ratio = _safe_float(payload.get('knee_lock_ratio'))
    if knee_lock_ratio is not None:
        summary['knee_lock_ratio'] = round(knee_lock_ratio, 1)

    trunk_stability_score = _safe_float(payload.get('trunk_stability_score'))
    if trunk_stability_score is not None:
        summary['trunk_stability_score'] = round(trunk_stability_score, 1)

    return summary


def issue_report_path(output_dir: Path) -> Path:
    return output_dir / 'review' / ISSUE_REPORT_FILENAME


def load_issue_reports(output_dir: Path, limit: int = 6) -> list[dict[str, str]]:
    path = issue_report_path(output_dir)
    if not path.exists():
        return []

    reports: list[dict[str, str]] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            reports.append({
                'created_at': str(payload.get('created_at', '')),
                'category': str(payload.get('category', '其他')),
                'time_range': str(payload.get('time_range', '')),
                'note': str(payload.get('note', '')),
                'source': str(payload.get('source', 'web_ui')),
            })
    except Exception:
        return []
    reports = reports[-limit:]
    reports.reverse()
    return reports


def append_issue_report(
    output_dir: Path,
    *,
    note: str,
    time_range: str = '',
    category: str = '其他',
    source: str = 'web_ui',
) -> dict[str, str]:
    if not note.strip():
        raise ValueError('missing note')

    review_dir = output_dir / 'review'
    review_dir.mkdir(parents=True, exist_ok=True)
    record = {
        'created_at': now_iso(),
        'category': (category.strip() or '其他')[:32],
        'time_range': time_range.strip()[:64],
        'note': note.strip()[:1000],
        'source': source.strip() or 'web_ui',
    }
    path = issue_report_path(output_dir)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return record


def build_run_record(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_dir() or path.name.startswith('.') or path.name == '.cache':
        return None

    files: dict[str, Any] = {}
    review: dict[str, Any] = {}
    for key, suffix in OUTPUT_SUFFIXES.items():
        matches = list(path.glob(f'*{suffix}'))
        if matches:
            files[key] = {'path': str(matches[0]), 'url': file_url(matches[0])}

    review_dir = path / 'review'
    if review_dir.exists():
        for key, filename in REVIEW_FILENAMES.items():
            review_path = review_dir / filename
            if review_path.exists():
                review[key] = {'path': str(review_path), 'url': file_url(review_path)}

    if not files and not review:
        return None

    metrics_summary = load_metrics_summary(review_dir / REVIEW_FILENAMES['pressure_metrics'])
    issue_reports = load_issue_reports(path)
    return {
        'name': path.name,
        'output_dir': str(path),
        'updated_at': datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec='seconds'),
        'files': files,
        'review': review,
        'metrics_summary': metrics_summary,
        'issue_reports': issue_reports,
    }


def get_run_record(run_name: str) -> dict[str, Any] | None:
    output_dir = resolve_output_dir(run_name)
    if output_dir is None:
        return None
    return build_run_record(output_dir)


class TaskStore:
    def __init__(self, task_dir: Path):
        self.task_dir = task_dir
        self.lock = threading.Lock()

    def path_for(self, task_id: str) -> Path:
        return self.task_dir / f'{task_id}.json'

    def save(self, task: dict[str, Any]) -> None:
        path = self.path_for(task['id'])
        temp = path.with_suffix('.json.tmp')
        task['updated_at'] = now_iso()
        with self.lock:
            temp.write_text(json.dumps(task, ensure_ascii=False, indent=2))
            temp.replace(path)

    def load(self, task_id: str) -> dict[str, Any] | None:
        path = self.path_for(task_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def list_all(self) -> list[dict[str, Any]]:
        tasks = []
        for path in sorted(self.task_dir.glob('*.json')):
            try:
                tasks.append(json.loads(path.read_text()))
            except Exception:
                continue
        tasks.sort(key=lambda item: item.get('created_at', ''), reverse=True)
        return tasks


class TaskManager:
    def __init__(self):
        self.store = TaskStore(TASK_DIR)
        self.queue: queue.Queue[str] = queue.Queue()
        self.started = False
        self.workers: list[threading.Thread] = []
        self.state_lock = threading.Lock()
        self.active_task_ids: set[str] = set()
        self._recover_tasks()

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        for idx in range(MAX_CONCURRENT_JOBS):
            worker = threading.Thread(target=self._worker_loop, name=f'web-task-worker-{idx}', daemon=True)
            worker.start()
            self.workers.append(worker)

    def _recover_tasks(self) -> None:
        for task in self.store.list_all():
            if task.get('status') in {'queued', 'running'}:
                task['status'] = 'interrupted'
                task['message'] = 'Web 服务曾重启，旧任务状态已标记为 interrupted。'
                self.store.save(task)

    def list_tasks(self) -> list[dict[str, Any]]:
        tasks = self.store.list_all()
        queued_ids = list(self.queue.queue)
        active_ids = set(self.active_task_ids)
        for task in tasks:
            task.setdefault('pipeline_id', DEFAULT_PIPELINE_ID)
            task.setdefault('pipeline_label', get_pipeline(task['pipeline_id']).label)
            task.setdefault('metrics_summary', {})
            task.setdefault('issue_reports', [])
            self._hydrate_task_artifacts(task, Path(task.get('output_dir', '')))
            if task['id'] in active_ids:
                task['queue_state'] = 'running'
            elif task.get('status') == 'queued' and task['id'] in queued_ids:
                task['queue_state'] = f'queued #{queued_ids.index(task["id"]) + 1}'
            else:
                task['queue_state'] = task.get('status', 'unknown')
        return tasks

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        task = self.store.load(task_id)
        if task is not None:
            task.setdefault('pipeline_id', DEFAULT_PIPELINE_ID)
            task.setdefault('pipeline_label', get_pipeline(task['pipeline_id']).label)
            task.setdefault('metrics_summary', {})
            task.setdefault('issue_reports', [])
            self._hydrate_task_artifacts(task, Path(task.get('output_dir', '')))
        return task

    def _enqueue_task_record(
        self,
        *,
        upload_path: Path,
        original_name: str,
        requested_name: str = '',
        pipeline_id: str = DEFAULT_PIPELINE_ID,
    ) -> dict[str, Any]:
        pipeline = get_pipeline(pipeline_id)
        task_id = uuid.uuid4().hex[:10]
        stem = sanitize_name(Path(original_name).stem)
        task_name = sanitize_name(requested_name) if requested_name.strip() else f'web_{stem}_{task_id}'
        suffix = Path(original_name).suffix or '.mp4'
        final_upload_path = UPLOAD_DIR / f'{task_id}_{task_name}{suffix.lower()}'
        final_upload_path.parent.mkdir(parents=True, exist_ok=True)
        if upload_path.resolve() != final_upload_path.resolve():
            upload_path.replace(final_upload_path)
        else:
            final_upload_path = upload_path

        task = {
            'id': task_id,
            'created_at': now_iso(),
            'updated_at': now_iso(),
            'original_name': original_name,
            'requested_name': requested_name.strip(),
            'task_name': task_name,
            'pipeline_id': pipeline.pipeline_id,
            'pipeline_label': pipeline.label,
            'upload_path': str(final_upload_path),
            'status': 'queued',
            'queue_state': 'queued',
            'stage': 'queued',
            'progress': 0.0,
            'stage_progress': 0.0,
            'message': '已进入队列，等待开始。',
            'logs': [],
            'output_dir': str(OUTPUT_DIR / task_name),
            'output_files': {},
            'review_files': {},
            'metrics_summary': {},
            'issue_reports': [],
            'error': '',
        }
        self.store.save(task)
        self.queue.put(task_id)
        return task

    def create_task(
        self,
        source_stream,
        original_name: str,
        requested_name: str = '',
        pipeline_id: str = DEFAULT_PIPELINE_ID,
    ) -> dict[str, Any]:
        temp_name = f"incoming_{uuid.uuid4().hex[:10]}{Path(original_name).suffix or '.mp4'}"
        upload_path = UPLOAD_DIR / temp_name
        with upload_path.open('wb') as file_obj:
            chunked_copy(source_stream, file_obj)
        return self._enqueue_task_record(
            upload_path=upload_path,
            original_name=original_name,
            requested_name=requested_name,
            pipeline_id=pipeline_id,
        )

    def create_task_from_path(
        self,
        upload_path: Path,
        *,
        original_name: str,
        requested_name: str = '',
        pipeline_id: str = DEFAULT_PIPELINE_ID,
    ) -> dict[str, Any]:
        if not upload_path.exists() or not upload_path.is_file():
            raise FileNotFoundError('upload file not found')
        return self._enqueue_task_record(
            upload_path=upload_path,
            original_name=original_name,
            requested_name=requested_name,
            pipeline_id=pipeline_id,
        )

    def append_issue(
        self,
        task_id: str,
        *,
        note: str,
        time_range: str = '',
        category: str = '其他',
        source: str = 'web_ui',
    ) -> dict[str, Any]:
        task = self.store.load(task_id)
        if task is None:
            raise KeyError('task not found')
        output_dir = Path(task.get('output_dir', ''))
        if not output_dir.exists() or not output_dir.is_dir():
            raise FileNotFoundError('task output not found')
        report = append_issue_report(
            output_dir,
            note=note,
            time_range=time_range,
            category=category,
            source=source,
        )
        self._hydrate_task_artifacts(task, output_dir)
        self.store.save(task)
        return {'report': report, 'task': task}

    def _worker_loop(self) -> None:
        while True:
            task_id = self.queue.get()
            with self.state_lock:
                self.active_task_ids.add(task_id)
            try:
                self._run_task(task_id)
            except Exception as exc:
                task = self.store.load(task_id)
                if task is not None:
                    self._hydrate_task_artifacts(task, Path(task.get('output_dir', '')))
                    task['status'] = 'failed'
                    task['stage'] = 'failed'
                    task['message'] = '任务执行失败。'
                    task['error'] = repr(exc)
                    self.store.save(task)
            finally:
                with self.state_lock:
                    self.active_task_ids.discard(task_id)
                self.queue.task_done()

    def _append_log(self, task: dict[str, Any], message: str) -> None:
        message = message.strip()
        if not message:
            return
        logs = list(task.get('logs', []))
        if logs and logs[-1] == message:
            return
        logs.append(message)
        if len(logs) > LOG_TAIL_LIMIT:
            logs = logs[-LOG_TAIL_LIMIT:]
        task['logs'] = logs

    def _update_progress(self, task: dict[str, Any], stage: str, stage_progress: float, message: str = '') -> None:
        start, end = STAGE_WEIGHTS[stage]
        stage_progress = max(0.0, min(100.0, stage_progress))
        overall = start + (end - start) * (stage_progress / 100.0)
        task['stage'] = stage
        task['progress'] = round(overall, 2)
        task['stage_progress'] = round(stage_progress, 2)
        if message:
            task['message'] = message

    def _attach_file(self, record: dict[str, Any], key: str, path: Path) -> None:
        if path.exists():
            record[key] = {'path': str(path), 'url': file_url(path)}

    def _hydrate_task_artifacts(self, task: dict[str, Any], output_dir: Path) -> None:
        if not output_dir.exists() or not output_dir.is_dir():
            return
        output_files = dict(task.get('output_files', {}))
        review_files = dict(task.get('review_files', {}))
        for key, suffix in OUTPUT_SUFFIXES.items():
            matches = list(output_dir.glob(f'*{suffix}'))
            if matches:
                self._attach_file(output_files, key, matches[0])
        review_dir = output_dir / 'review'
        if review_dir.exists():
            for key, filename in REVIEW_FILENAMES.items():
                self._attach_file(review_files, key, review_dir / filename)
        task['output_files'] = output_files
        task['review_files'] = review_files
        task['metrics_summary'] = load_metrics_summary(review_dir / REVIEW_FILENAMES['pressure_metrics'])
        task['issue_reports'] = load_issue_reports(output_dir)

    def _parse_progress_line(self, task: dict[str, Any], line: str) -> None:
        match = PROGRESS_RE.search(line)
        if match:
            name, percent_str = match.groups()
            percent = float(percent_str)
            if name == 'ByteTrack':
                self._update_progress(task, 'tracking', percent, f'目标追踪中 {int(percent)}%')
            elif name == 'RTMPose-X':
                self._update_progress(task, 'pose', percent, f'关键点估计中 {int(percent)}%')
            else:
                self._update_progress(task, 'render', percent, f'视频渲染中 {int(percent)}%')
            return

        if line.startswith('[Success]'):
            self._update_progress(task, 'qa', 0.0, '主处理完成，开始 QA 复核。')
            return

        labels = {
            '[Overlay] ': 'overlay',
            '[Compare] ': 'side_by_side',
        }
        for prefix, key in labels.items():
            if line.startswith(prefix):
                path = Path(line[len(prefix):].strip())
                if path.exists():
                    task['output_files'][key] = {'path': str(path), 'url': file_url(path)}
                return

        qa_labels = {
            '[QA] contact_sheet=': 'contact_sheet',
            '[QA] summary=': 'summary',
            '[QA] flags=': 'flags',
            '[QA] critical_sheet=': 'critical_sheet',
            '[Pressure] png=': 'pressure_plot',
            '[Pressure] csv=': 'pressure_csv',
            '[Pressure] metrics=': 'pressure_metrics',
        }
        for prefix, key in qa_labels.items():
            if line.startswith(prefix):
                path = Path(line.split('=', 1)[1].strip())
                if path.exists():
                    task['review_files'][key] = {'path': str(path), 'url': file_url(path)}
                return

    def _read_process_output(self, process: subprocess.Popen, task: dict[str, Any]) -> None:
        buffer = ''
        fd = process.stdout.fileno()
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            buffer += chunk.decode('utf-8', errors='replace')
            parts = re.split(r'[\r\n]+', buffer)
            buffer = parts.pop()
            for part in parts:
                message = part.strip()
                if not message:
                    continue
                self._append_log(task, message)
                self._parse_progress_line(task, message)
                self.store.save(task)
        if buffer.strip():
            self._append_log(task, buffer.strip())
            self._parse_progress_line(task, buffer.strip())
            self.store.save(task)

    def _run_subprocess(self, cmd: list[str], task: dict[str, Any], env: dict[str, str] | None = None) -> None:
        task['message'] = '子进程启动中。'
        self.store.save(task)
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJ_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        task['pid'] = process.pid
        self.store.save(task)
        assert process.stdout is not None
        self._read_process_output(process, task)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f'子进程失败，退出码 {return_code}')

    def _run_task(self, task_id: str) -> None:
        task = self.store.load(task_id)
        if task is None:
            return

        pipeline = get_pipeline(task.get('pipeline_id'))
        task['pipeline_id'] = pipeline.pipeline_id
        task['pipeline_label'] = pipeline.label
        task['status'] = 'running'
        task['error'] = ''
        task['message'] = f'{pipeline.label} 任务已启动。'
        self._update_progress(task, 'tracking', 0.0, '目标追踪准备中。')
        self.store.save(task)

        task_name = task['task_name']
        input_path = Path(task['upload_path'])
        output_dir = OUTPUT_DIR / task_name
        cache_dir = OUTPUT_DIR / '.cache' / task_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        task['output_dir'] = str(output_dir)
        metadata = task.setdefault('metadata', {})
        metadata['python'] = str(PYTHON_BIN)

        execution = prepare_pipeline_execution(
            pipeline.pipeline_id,
            task_name=task_name,
            input_path=input_path,
            output_dir=output_dir,
            cache_dir=cache_dir,
        )

        base_env = os.environ.copy()
        base_env.setdefault('OMP_NUM_THREADS', '4')
        base_env.setdefault('MKL_NUM_THREADS', '4')
        base_env.setdefault('OPENBLAS_NUM_THREADS', '4')
        base_env.setdefault('NUMEXPR_NUM_THREADS', '4')
        base_env.setdefault('VECLIB_MAXIMUM_THREADS', '4')

        self._run_subprocess(execution.command, task, env=base_env)
        self._hydrate_task_artifacts(task, output_dir)

        if pipeline.qa_enabled:
            self._update_progress(task, 'qa', 15.0, '主处理完成，开始 QA。')
            self.store.save(task)
            review_dir = output_dir / 'review'
            qa_cmd = [
                execution.command[0],
                str(QA_SCRIPT),
                '--keypoint-cache', str(execution.keypoint_cache),
                '--compare-video', str(execution.compare_path),
                '--review-dir', str(review_dir),
            ]
            self._run_subprocess(qa_cmd, task, env=base_env)

        self._hydrate_task_artifacts(task, output_dir)
        task['status'] = 'completed'
        task['message'] = '任务完成，可在线预览或下载视频和复核图。'
        self._update_progress(task, 'qa', 100.0, task['message'])
        task['output_url'] = file_url(output_dir) if output_dir.exists() else ''
        self.store.save(task)



def scan_completed_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in OUTPUT_DIR.iterdir():
        record = build_run_record(path)
        if record is not None:
            runs.append(record)
    runs.sort(key=lambda item: item['updated_at'], reverse=True)
    return runs
