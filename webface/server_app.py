from __future__ import annotations

import html
import io
import json
import mimetypes
import socket
import uuid
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import (
    CHUNK_UPLOAD_SIZE,
    HOST,
    INCOMING_DIR,
    OUTPUT_DIR,
    PORT,
    PROJ_ROOT,
    STATIC_DIR,
    TEMPLATE_DIR,
    WEB_DATA_DIR,
    get_public_base_url,
)
from .pipelines import DEFAULT_PIPELINE_ID, list_pipelines
from .task_manager import TaskManager, append_issue_report, get_run_record, resolve_output_dir, scan_completed_runs


def json_response(start_response, payload, status='200 OK', head_only: bool = False):
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    headers = [('Content-Type', 'application/json; charset=utf-8'), ('Content-Length', str(len(body)))]
    start_response(status, headers)
    return [b''] if head_only else [body]


def text_response(start_response, text, status='200 OK', content_type='text/plain; charset=utf-8', head_only: bool = False):
    body = text.encode('utf-8')
    headers = [('Content-Type', content_type), ('Content-Length', str(len(body)))]
    start_response(status, headers)
    return [b''] if head_only else [body]


def html_response(start_response, html, status='200 OK', head_only: bool = False):
    return text_response(start_response, html, status=status, content_type='text/html; charset=utf-8', head_only=head_only)


def read_request_body(environ) -> bytes:
    try:
        content_length = int(environ.get('CONTENT_LENGTH', '0') or '0')
    except ValueError:
        content_length = 0
    return environ['wsgi.input'].read(content_length) if content_length > 0 else b''


def read_json_request(environ) -> dict:
    raw = read_request_body(environ)
    if not raw:
        return {}
    payload = json.loads(raw.decode('utf-8'))
    return payload if isinstance(payload, dict) else {}


def parse_multipart_form(environ) -> dict:
    content_type = environ.get('CONTENT_TYPE', '')
    if 'multipart/form-data' not in content_type:
        raise ValueError('expected multipart form data')

    boundary_token = ''
    for part in content_type.split(';'):
        part = part.strip()
        if part.startswith('boundary='):
            boundary_token = part.split('=', 1)[1].strip().strip('"')
            break
    if not boundary_token:
        raise ValueError('missing multipart boundary')

    boundary = f'--{boundary_token}'.encode('utf-8')
    stream = environ['wsgi.input']
    first_line = stream.readline()
    if not first_line:
        return {}
    if not first_line.strip().startswith(boundary):
        raise ValueError('invalid multipart payload')

    form: dict[str, object] = {}
    while True:
        headers: dict[str, str] = {}
        while True:
            line = stream.readline()
            if not line:
                return form
            stripped = line.strip()
            if not stripped:
                break
            decoded = line.decode('utf-8', errors='replace')
            key, _, value = decoded.partition(':')
            headers[key.strip().lower()] = value.strip()

        disposition = headers.get('content-disposition', '')
        name = ''
        filename = ''
        for item in disposition.split(';'):
            item = item.strip()
            if item.startswith('name='):
                name = item.split('=', 1)[1].strip().strip('"')
            elif item.startswith('filename='):
                filename = item.split('=', 1)[1].strip().strip('"')
        if not name:
            raise ValueError('multipart part missing name')

        sink = io.BytesIO()
        previous_line = stream.readline()
        if not previous_line:
            break
        reached_end = False
        while True:
            line = stream.readline()
            if not line:
                if previous_line.endswith(b'\r\n'):
                    sink.write(previous_line[:-2])
                else:
                    sink.write(previous_line)
                reached_end = True
                break
            if line.startswith(boundary):
                if previous_line.endswith(b'\r\n'):
                    sink.write(previous_line[:-2])
                else:
                    sink.write(previous_line)
                reached_end = line.rstrip().endswith(b'--')
                break
            sink.write(previous_line)
            previous_line = line

        sink.seek(0)
        if filename:
            form[name] = {'filename': filename, 'file': sink}
        else:
            form[name] = sink.read().decode('utf-8', errors='replace')

        if reached_end:
            break

    return form


def safe_resolve(path: Path, roots: list[Path]) -> Path | None:
    try:
        resolved = path.resolve()
    except Exception:
        return None
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except Exception:
            continue
    return None


def guess_lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(('8.8.8.8', 80))
        return sock.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        sock.close()


def build_access_urls() -> dict[str, str]:
    lan_ip = guess_lan_ip()
    urls = {
        'local': f'http://127.0.0.1:{PORT}',
        'lan': f'http://{lan_ip}:{PORT}',
        'public': get_public_base_url(),
    }
    return {key: value for key, value in urls.items() if value}


def parse_byte_range(range_header: str, file_size: int) -> tuple[int, int] | None:
    if not range_header or not range_header.startswith('bytes='):
        return None
    value = range_header[6:].split(',', 1)[0].strip()
    if '-' not in value:
        raise ValueError('invalid byte range')
    start_text, end_text = value.split('-', 1)
    if not start_text:
        suffix_len = int(end_text)
        if suffix_len <= 0:
            raise ValueError('invalid suffix range')
        start = max(file_size - suffix_len, 0)
        end = file_size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
    if start < 0 or end < start or start >= file_size:
        raise ValueError('range out of bounds')
    end = min(end, file_size - 1)
    return start, end


def serve_file(environ, start_response, target: Path, *, download_name: str | None = None):
    content_type = mimetypes.guess_type(str(target))[0] or 'application/octet-stream'
    file_size = target.stat().st_size
    range_header = environ.get('HTTP_RANGE', '').strip()
    head_only = environ.get('REQUEST_METHOD', 'GET').upper() == 'HEAD'
    disposition = None
    if download_name:
        disposition = f"attachment; filename*=UTF-8''{quote(download_name)}"

    try:
        byte_range = parse_byte_range(range_header, file_size)
    except ValueError:
        start_response(
            '416 Requested Range Not Satisfiable',
            [('Content-Range', f'bytes */{file_size}'), ('Content-Length', '0')],
        )
        return [b'']

    if byte_range is None:
        headers = [('Content-Type', content_type), ('Content-Length', str(file_size))]
        if disposition:
            headers.append(('Content-Disposition', disposition))
        if target.suffix.lower() in {'.mp4', '.mov'}:
            headers.append(('Accept-Ranges', 'bytes'))
        start_response('200 OK', headers)
        if head_only:
            return [b'']

        def stream_full(chunk_size: int = 1024 * 1024):
            with target.open('rb') as file_obj:
                while True:
                    chunk = file_obj.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        return stream_full()

    start, end = byte_range
    length = end - start + 1
    headers = [
        ('Content-Type', content_type),
        ('Content-Length', str(length)),
        ('Content-Range', f'bytes {start}-{end}/{file_size}'),
        ('Accept-Ranges', 'bytes'),
    ]
    if disposition:
        headers.append(('Content-Disposition', disposition))
    start_response('206 Partial Content', headers)
    if head_only:
        return [b'']

    def stream_range(chunk_size: int = 1024 * 1024):
        with target.open('rb') as file_obj:
            file_obj.seek(start)
            remaining = length
            while remaining > 0:
                chunk = file_obj.read(min(chunk_size, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return stream_range()


def upload_meta_path(upload_id: str) -> Path:
    return INCOMING_DIR / f'{upload_id}.json'


def upload_temp_path(upload_id: str, suffix: str) -> Path:
    return INCOMING_DIR / f'{upload_id}{suffix}'


def save_upload_session(session: dict) -> None:
    upload_meta_path(session['upload_id']).write_text(json.dumps(session, ensure_ascii=False, indent=2))


def load_upload_session(upload_id: str) -> dict | None:
    path = upload_meta_path(upload_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def cleanup_upload_session(session: dict) -> None:
    for path in [upload_meta_path(session['upload_id']), Path(session['temp_path'])]:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def init_chunk_upload(payload: dict) -> dict:
    original_name = str(payload.get('original_name', '')).strip()
    requested_name = str(payload.get('task_name', '')).strip()
    pipeline_id = str(payload.get('pipeline_id', DEFAULT_PIPELINE_ID)).strip() or DEFAULT_PIPELINE_ID
    total_size = int(payload.get('total_size', 0) or 0)
    total_chunks = int(payload.get('total_chunks', 0) or 0)
    if not original_name:
        raise ValueError('missing original_name')
    if total_size <= 0:
        raise ValueError('invalid total_size')
    if total_chunks <= 0:
        raise ValueError('invalid total_chunks')

    suffix = Path(original_name).suffix or '.mp4'
    upload_id = uuid.uuid4().hex[:12]
    temp_path = upload_temp_path(upload_id, suffix.lower())
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(b'')
    session = {
        'upload_id': upload_id,
        'original_name': original_name,
        'requested_name': requested_name,
        'pipeline_id': pipeline_id,
        'total_size': total_size,
        'total_chunks': total_chunks,
        'received_bytes': 0,
        'next_index': 0,
        'temp_path': str(temp_path),
    }
    save_upload_session(session)
    return session


def append_chunk(upload_id: str, chunk_index: int, chunk_body: bytes) -> dict:
    session = load_upload_session(upload_id)
    if session is None:
        raise FileNotFoundError('upload session not found')
    if chunk_index != int(session.get('next_index', 0)):
        raise ValueError('unexpected chunk index')
    if not chunk_body:
        raise ValueError('empty chunk body')
    temp_path = Path(session['temp_path'])
    with temp_path.open('ab') as handle:
        handle.write(chunk_body)
    session['received_bytes'] = int(session.get('received_bytes', 0)) + len(chunk_body)
    session['next_index'] = chunk_index + 1
    save_upload_session(session)
    return session


def finish_chunk_upload(upload_id: str, task_manager: TaskManager) -> dict:
    session = load_upload_session(upload_id)
    if session is None:
        raise FileNotFoundError('upload session not found')
    if int(session.get('next_index', 0)) != int(session.get('total_chunks', 0)):
        raise ValueError('upload not complete')
    temp_path = Path(session['temp_path'])
    if not temp_path.exists():
        raise FileNotFoundError('temp upload missing')
    if temp_path.stat().st_size != int(session.get('total_size', 0)):
        raise ValueError('uploaded size mismatch')
    task = task_manager.create_task_from_path(
        temp_path,
        original_name=str(session.get('original_name', 'uploaded_video.mp4')),
        requested_name=str(session.get('requested_name', '')),
        pipeline_id=str(session.get('pipeline_id', DEFAULT_PIPELINE_ID)),
    )
    meta_path = upload_meta_path(upload_id)
    if meta_path.exists():
        meta_path.unlink()
    return task


def create_app():
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=select_autoescape(['html']))
    task_manager = TaskManager()
    task_manager.start()

    def app(environ, start_response):
        method = environ.get('REQUEST_METHOD', 'GET').upper()
        head_only = method == 'HEAD'
        path = environ.get('PATH_INFO', '/') or '/'
        query = parse_qs(environ.get('QUERY_STRING', ''), keep_blank_values=True)

        if method in {'GET', 'HEAD'} and path == '/':
            template = env.get_template('index.html')
            html = template.render(
                host=HOST,
                port=PORT,
                access_urls=build_access_urls(),
                pipelines=list_pipelines(),
                default_pipeline_id=DEFAULT_PIPELINE_ID,
            )
            return html_response(start_response, html, head_only=head_only)

        if method in {'GET', 'HEAD'} and path == '/api/health':
            return json_response(start_response, {
                'ok': True,
                'host': HOST,
                'port': PORT,
                'access_urls': build_access_urls(),
                'pipelines': list_pipelines(),
                'default_pipeline_id': DEFAULT_PIPELINE_ID,
                'chunk_upload_size': CHUNK_UPLOAD_SIZE,
            }, head_only=head_only)

        if method in {'GET', 'HEAD'} and path == '/api/pipelines':
            return json_response(start_response, {
                'pipelines': list_pipelines(),
                'default_pipeline_id': DEFAULT_PIPELINE_ID,
                'chunk_upload_size': CHUNK_UPLOAD_SIZE,
            }, head_only=head_only)

        if method in {'GET', 'HEAD'} and path == '/api/tasks':
            return json_response(start_response, {'tasks': task_manager.list_tasks()}, head_only=head_only)

        if method in {'GET', 'HEAD'} and path.startswith('/api/tasks/'):
            task_id = path.rsplit('/', 1)[-1]
            task = task_manager.get_task(task_id)
            if task is None:
                return json_response(start_response, {'error': 'task not found'}, status='404 Not Found', head_only=head_only)
            return json_response(start_response, task, head_only=head_only)

        if method in {'GET', 'HEAD'} and path == '/api/runs':
            return json_response(start_response, {'runs': scan_completed_runs()}, head_only=head_only)

        if method in {'GET', 'HEAD'} and path == '/preview':
            src = (query.get('src') or [''])[0]
            if not src.startswith('/files/'):
                return text_response(start_response, 'Invalid preview target', status='400 Bad Request', head_only=head_only)
            content_type = mimetypes.guess_type(unquote(src))[0] or 'application/octet-stream'
            escaped_src = html.escape(src, quote=True)
            escaped_name = html.escape(Path(unquote(src)).name)
            download_href = f"/download?src={quote(src, safe='')}"
            if content_type.startswith('video/'):
                preview_body = f'<video controls autoplay playsinline preload="metadata" src="{escaped_src}"></video>'
            elif content_type.startswith('image/'):
                preview_body = f'<img src="{escaped_src}" alt="{escaped_name}">'
            else:
                return text_response(start_response, 'Unsupported preview type', status='415 Unsupported Media Type', head_only=head_only)
            preview_html = f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>在线预览 · {escaped_name}</title>
  <style>
    body {{ margin: 0; background: #050b15; color: #e5eef9; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    .page {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .actions {{ margin-bottom: 16px; display: flex; gap: 12px; flex-wrap: wrap; }}
    a {{ color: #7db2ff; text-decoration: none; }}
    .chip {{ display: inline-flex; padding: 10px 14px; border-radius: 999px; background: rgba(125,178,255,0.14); border: 1px solid rgba(125,178,255,0.22); }}
    video, img {{ width: 100%; max-height: calc(100vh - 140px); background: #000; border-radius: 14px; }}
  </style>
</head>
<body>
  <div class="page">
    <div class="actions">
      <a class="chip" href="{escaped_src}" target="_blank" rel="noreferrer">直接打开源文件</a>
      <a class="chip" href="{download_href}">下载文件</a>
    </div>
    {preview_body}
  </div>
</body>
</html>'''
            return html_response(start_response, preview_html, head_only=head_only)

        if method in {'GET', 'HEAD'} and path == '/download':
            src = (query.get('src') or [''])[0]
            if not src.startswith('/files/'):
                return text_response(start_response, 'Invalid download target', status='400 Bad Request', head_only=head_only)
            relative = unquote(src[len('/files/'):].strip('/'))
            target = safe_resolve(PROJ_ROOT / relative, [PROJ_ROOT / 'output', WEB_DATA_DIR, STATIC_DIR])
            if target is None or not target.exists() or not target.is_file():
                return text_response(start_response, 'Not Found', status='404 Not Found', head_only=head_only)
            return serve_file(environ, start_response, target, download_name=target.name)

        if method == 'POST' and path == '/api/issues':
            try:
                payload = read_json_request(environ)
            except Exception:
                return json_response(start_response, {'error': 'invalid json body'}, status='400 Bad Request')

            note = str(payload.get('note', '')).strip()
            time_range = str(payload.get('time_range', '')).strip()
            category = str(payload.get('category', '其他')).strip() or '其他'
            if not note:
                return json_response(start_response, {'error': 'missing note'}, status='400 Bad Request')

            task_id = str(payload.get('task_id', '')).strip()
            run_name = str(payload.get('run_name', '')).strip()
            if task_id:
                try:
                    result = task_manager.append_issue(task_id, note=note, time_range=time_range, category=category)
                except KeyError:
                    return json_response(start_response, {'error': 'task not found'}, status='404 Not Found')
                except FileNotFoundError:
                    return json_response(start_response, {'error': 'task output not ready'}, status='409 Conflict')
                except ValueError as exc:
                    return json_response(start_response, {'error': str(exc)}, status='400 Bad Request')
                return json_response(start_response, {'ok': True, 'task': result['task'], 'report': result['report']}, status='201 Created')

            if run_name:
                output_dir = resolve_output_dir(run_name)
                if output_dir is None or output_dir == OUTPUT_DIR or not output_dir.exists() or not output_dir.is_dir():
                    return json_response(start_response, {'error': 'run not found'}, status='404 Not Found')
                try:
                    report = append_issue_report(output_dir, note=note, time_range=time_range, category=category)
                except ValueError as exc:
                    return json_response(start_response, {'error': str(exc)}, status='400 Bad Request')
                run = get_run_record(run_name)
                return json_response(start_response, {'ok': True, 'run': run, 'report': report}, status='201 Created')

            return json_response(start_response, {'error': 'missing task_id or run_name'}, status='400 Bad Request')

        if method == 'POST' and path == '/api/uploads/init':
            try:
                payload = read_json_request(environ)
                session = init_chunk_upload(payload)
            except ValueError as exc:
                return json_response(start_response, {'error': str(exc)}, status='400 Bad Request')
            return json_response(start_response, {'ok': True, 'upload': session}, status='201 Created')

        if method == 'POST' and path.startswith('/api/uploads/') and path.endswith('/chunk'):
            parts = [item for item in path.strip('/').split('/') if item]
            if len(parts) != 4:
                return json_response(start_response, {'error': 'invalid upload path'}, status='404 Not Found')
            upload_id = parts[2]
            try:
                chunk_index = int((query.get('index') or ['0'])[0])
                chunk_body = read_request_body(environ)
                session = append_chunk(upload_id, chunk_index, chunk_body)
            except FileNotFoundError:
                return json_response(start_response, {'error': 'upload session not found'}, status='404 Not Found')
            except ValueError as exc:
                return json_response(start_response, {'error': str(exc)}, status='409 Conflict')
            return json_response(start_response, {'ok': True, 'upload': session})

        if method == 'POST' and path.startswith('/api/uploads/') and path.endswith('/finish'):
            parts = [item for item in path.strip('/').split('/') if item]
            if len(parts) != 4:
                return json_response(start_response, {'error': 'invalid upload path'}, status='404 Not Found')
            upload_id = parts[2]
            try:
                task = finish_chunk_upload(upload_id, task_manager)
            except FileNotFoundError:
                return json_response(start_response, {'error': 'upload session not found'}, status='404 Not Found')
            except ValueError as exc:
                return json_response(start_response, {'error': str(exc)}, status='409 Conflict')
            return json_response(start_response, {'ok': True, 'task': task}, status='201 Created')

        if method == 'POST' and path == '/api/upload':
            try:
                form = parse_multipart_form(environ)
            except ValueError as exc:
                return json_response(start_response, {'error': str(exc)}, status='400 Bad Request')

            if 'video' not in form:
                return json_response(start_response, {'error': 'missing video field'}, status='400 Bad Request')
            video_item = form['video']
            requested_name = str(form['task_name']) if 'task_name' in form else ''
            pipeline_id = str(form['pipeline_id']) if 'pipeline_id' in form else DEFAULT_PIPELINE_ID
            filename = video_item.get('filename', '') or 'uploaded_video.mp4'
            if not video_item.get('file'):
                return json_response(start_response, {'error': 'invalid upload'}, status='400 Bad Request')
            task = task_manager.create_task(video_item['file'], filename, requested_name=requested_name, pipeline_id=pipeline_id)
            return json_response(start_response, {'ok': True, 'task': task}, status='201 Created')

        if method in {'GET', 'HEAD'} and path.startswith('/files/'):
            relative = unquote(path[len('/files/'):].strip('/'))
            target = safe_resolve(PROJ_ROOT / relative, [PROJ_ROOT / 'output', WEB_DATA_DIR, STATIC_DIR])
            if target is None or not target.exists() or not target.is_file():
                return text_response(start_response, 'Not Found', status='404 Not Found', head_only=head_only)
            return serve_file(environ, start_response, target)

        if method in {'GET', 'HEAD'} and path.startswith('/static/'):
            relative = unquote(path[len('/static/'):].strip('/'))
            target = safe_resolve(STATIC_DIR / relative, [STATIC_DIR])
            if target is None or not target.exists() or not target.is_file():
                return text_response(start_response, 'Not Found', status='404 Not Found', head_only=head_only)
            return serve_file(environ, start_response, target)

        return text_response(start_response, 'Not Found', status='404 Not Found', head_only=head_only)

    return app
