const state = { tasks: [], runs: [], health: null, uploading: false };

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}

function statusClass(status) {
  if (['queued', 'running', 'completed', 'failed', 'interrupted'].includes(status)) return status;
  return 'queued';
}

function metricLabel(key) {
  const mapping = {
    turn_count: '转弯',
    sync_score: '同步分',
    correlation: '相关',
    rmse: 'RMSE',
    mae: 'MAE',
    analysis_window: '分析窗',
    alignment_score: '对齐分',
    alignment_rate: '对齐率',
    knee_lock_ratio: '死锁率',
    trunk_stability_score: '上身稳态',
  };
  return mapping[key] || key;
}

function metricValue(key, value) {
  if (value == null || value === '') return '';
  if (key === 'sync_score' && typeof value === 'number') return value.toFixed(1);
  if (['correlation', 'rmse', 'mae'].includes(key) && typeof value === 'number') return value.toFixed(3);
  return String(value);
}

function linksHtml(recordSet) {
  const entries = Object.entries(recordSet || {});
  if (!entries.length) return '';
  return `<div class="link-row">${entries.map(([key, item]) => `<a href="/download?src=${encodeURIComponent(item.url)}" rel="noreferrer">${escapeHtml(key)} 下载</a>`).join('')}</div>`;
}

function previewHtml(recordSet) {
  const preview = recordSet?.pressure_sync || recordSet?.side_by_side || recordSet?.overlay;
  if (!preview?.url) return '';
  return `
    <details class="preview-card">
      <summary>在线预览</summary>
      <video controls playsinline preload="metadata" src="${preview.url}"></video>
    </details>`;
}

function metricsHtml(summary) {
  const entries = Object.entries(summary || {}).filter(([, value]) => value !== '' && value != null);
  if (!entries.length) return '';
  return `<div class="metric-row">${entries.map(([key, value]) => `
    <div class="metric-chip">
      <span>${escapeHtml(metricLabel(key))}</span>
      <strong>${escapeHtml(metricValue(key, value))}</strong>
    </div>`).join('')}</div>`;
}

function pathHtml(path) {
  if (!path) return '';
  return `<div class="path-box" title="${escapeHtml(path)}">输出目录：<code>${escapeHtml(path)}</code></div>`;
}

function issueHistoryHtml(reports) {
  if (!reports?.length) return '';
  return `
    <details class="issue-history">
      <summary>最近问题备注（${reports.length}）</summary>
      <div class="issue-history-list">${reports.map((item) => `
        <article class="issue-item">
          <div class="issue-item-head">
            <span class="issue-tag">${escapeHtml(item.category || '其他')}</span>
            <span class="issue-date">${escapeHtml(item.created_at || '')}</span>
          </div>
          ${item.time_range ? `<div class="issue-time">时间段：${escapeHtml(item.time_range)}</div>` : ''}
          <div class="issue-note">${escapeHtml(item.note || '')}</div>
        </article>`).join('')}</div>
    </details>`;
}

function issueFormHtml(kind, targetValue) {
  return `
    <details class="issue-box">
      <summary>这条视频有问题？留个备注给我后面排查</summary>
      <form class="issue-form" data-kind="${escapeHtml(kind)}" data-target="${escapeHtml(targetValue)}">
        <div class="issue-grid">
          <select name="category">
            <option value="跟踪切人">跟踪切人</option>
            <option value="关键点漂移">关键点漂移</option>
            <option value="压力判弯">压力判弯</option>
            <option value="画面样式">画面样式</option>
            <option value="其他">其他</option>
          </select>
          <input name="time_range" type="text" placeholder="可选时间段，例如 00:12-00:18">
        </div>
        <textarea name="note" rows="3" placeholder="例如：第 3 个弯切到了旁边的人；右手有拖尾；pressure_sync 弯心偏早" required></textarea>
        <div class="issue-actions">
          <button type="submit" class="ghost">保存问题备注</button>
          <span class="issue-status"></span>
        </div>
      </form>
    </details>`;
}

function renderServiceUrls() {
  const container = document.getElementById('service-urls');
  if (!container || !state.health) return;
  const accessUrls = state.health.access_urls || {};
  const items = Object.entries(accessUrls);
  if (!items.length) {
    container.innerHTML = '<span class="tip">服务已启动，但还没有可展示的访问地址。</span>';
    return;
  }
  container.innerHTML = items.map(([key, url]) => `<a class="service-chip" href="${url}" target="_blank" rel="noreferrer">${escapeHtml(key)} · ${escapeHtml(url)}</a>`).join('');
}

function renderStats() {
  const container = document.getElementById('dashboard-stats');
  if (!container) return;
  const queued = state.tasks.filter((task) => task.status === 'queued').length;
  const running = state.tasks.filter((task) => task.status === 'running').length;
  const completed = state.runs.length;
  const pipelines = state.health?.pipelines || [];
  const defaultPipelineId = state.health?.default_pipeline_id;
  const defaultPipeline = pipelines.find((item) => item.id === defaultPipelineId);

  const cards = [
    { label: '任务队列', value: queued, sub: queued ? '等待处理' : '当前队列为空' },
    { label: '运行中', value: running, sub: running ? '正在后台计算' : '当前没有运行任务' },
    { label: '已完成成片', value: completed, sub: completed ? '可直接回看历史结果' : '还没有扫描到成片' },
    { label: '默认流程', value: defaultPipeline?.id || 'model1', sub: defaultPipeline?.description || '等待健康检查' },
  ];

  container.innerHTML = cards.map((item) => `
    <article class="stat-card">
      <div class="stat-label">${escapeHtml(item.label)}</div>
      <div class="stat-value">${escapeHtml(item.value)}</div>
      <div class="stat-sub">${escapeHtml(item.sub)}</div>
    </article>`).join('');
}

function renderTasks() {
  const container = document.getElementById('task-list');
  if (!state.tasks.length) {
    container.className = 'list empty';
    container.textContent = '暂时还没有网页上传任务。';
    return;
  }
  container.className = 'list';
  container.innerHTML = state.tasks.map((task) => {
    const logs = (task.logs || []).slice(-8).join('\n');
    return `
      <article class="task-card">
        <div class="task-head">
          <div>
            <div class="task-name">${escapeHtml(task.task_name || task.original_name)}</div>
            <div class="task-meta">${escapeHtml(task.pipeline_label || 'model1')} · ${escapeHtml(task.original_name)} · ${escapeHtml(task.queue_state || task.status)} · ${escapeHtml(task.created_at || '')}</div>
          </div>
          <span class="badge ${statusClass(task.status)}">${escapeHtml(task.status)}</span>
        </div>
        <div class="progress-wrap">
          <div class="progress-row">
            <span>${escapeHtml(task.stage || 'queued')}</span>
            <span>${Number(task.progress || 0).toFixed(1)}%</span>
          </div>
          <div class="progress-bar"><div class="progress-fill" style="width:${Number(task.progress || 0)}%"></div></div>
        </div>
        <div class="task-message">${escapeHtml(task.message || '')}</div>
        ${metricsHtml(task.metrics_summary)}
        ${pathHtml(task.output_dir)}
        ${linksHtml(task.output_files)}
        ${linksHtml(task.review_files)}
        ${previewHtml(task.output_files)}
        ${issueHistoryHtml(task.issue_reports)}
        ${task.status === 'completed' ? issueFormHtml('task', task.id) : ''}
        ${logs ? `<div class="log-box">${escapeHtml(logs)}</div>` : ''}
      </article>`;
  }).join('');
}

function renderRuns() {
  const container = document.getElementById('run-list');
  if (!state.runs.length) {
    container.className = 'list empty';
    container.textContent = '还没有扫描到成片。';
    return;
  }
  container.className = 'list';
  container.innerHTML = state.runs.map((run) => `
    <article class="task-card">
      <div class="task-head">
        <div>
          <div class="task-name">${escapeHtml(run.name)}</div>
          <div class="task-meta">最近更新：${escapeHtml(run.updated_at || '')}</div>
        </div>
        <span class="badge completed">ready</span>
      </div>
      ${metricsHtml(run.metrics_summary)}
      ${pathHtml(run.output_dir)}
      ${linksHtml(run.files)}
      ${linksHtml(run.review)}
      ${previewHtml(run.files)}
      ${issueHistoryHtml(run.issue_reports)}
      ${issueFormHtml('run', run.name)}
    </article>`).join('');
}

function renderSelectedFiles() {
  const container = document.getElementById('selected-files');
  const fileInput = document.getElementById('video-input');
  const files = Array.from(fileInput.files || []);
  if (!files.length) {
    container.textContent = '还未选择文件。';
    return;
  }
  container.innerHTML = files.map((file, index) => `<span class="file-chip">${index + 1}. ${escapeHtml(file.name)}</span>`).join('');
}

async function fetchHealth() {
  try {
    const response = await fetch('/api/health');
    const payload = await response.json();
    state.health = payload;
    renderServiceUrls();
    renderStats();
  } catch (error) {
    const container = document.getElementById('service-urls');
    if (container) container.innerHTML = `<span class="tip">健康检查失败：${escapeHtml(error.message)}</span>`;
  }
}

async function fetchTasks() {
  const response = await fetch('/api/tasks');
  const payload = await response.json();
  state.tasks = payload.tasks || [];
  renderTasks();
  renderStats();
}

async function fetchRuns() {
  const response = await fetch('/api/runs');
  const payload = await response.json();
  state.runs = payload.runs || [];
  renderRuns();
  renderStats();
}

function updatePipelineNote() {
  const note = document.getElementById('pipeline-note');
  const select = document.getElementById('pipeline-select');
  const pipelines = state.health?.pipelines || [];
  const chunkSizeMb = ((state.health?.chunk_upload_size || 0) / (1024 * 1024)).toFixed(0);
  const current = pipelines.find((item) => item.id === select.value);
  if (note && current) note.textContent = `当前流程：${current.description}；公网分片上传默认每片 ${chunkSizeMb}MB。`;
}

function buildRequestedName(baseName, index, total) {
  if (!baseName) return '';
  if (total <= 1) return baseName;
  return `${baseName}_${String(index + 1).padStart(2, '0')}`;
}

async function initChunkUpload(file, requestedName, pipelineId) {
  const chunkSize = Number(state.health?.chunk_upload_size || 32 * 1024 * 1024);
  const totalChunks = Math.max(1, Math.ceil(file.size / chunkSize));
  const response = await fetch('/api/uploads/init', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      original_name: file.name,
      task_name: requestedName,
      pipeline_id: pipelineId,
      total_size: file.size,
      total_chunks: totalChunks,
    }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || '初始化上传失败');
  return { uploadId: payload.upload.upload_id, chunkSize, totalChunks };
}

async function uploadFileByChunks(file, requestedName, pipelineId, onProgress) {
  const { uploadId, chunkSize, totalChunks } = await initChunkUpload(file, requestedName, pipelineId);
  for (let index = 0; index < totalChunks; index += 1) {
    const start = index * chunkSize;
    const end = Math.min(file.size, start + chunkSize);
    const chunk = file.slice(start, end);
    const response = await fetch(`/api/uploads/${uploadId}/chunk?index=${index}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: chunk,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `第 ${index + 1} 片上传失败`);
    if (onProgress) onProgress(index + 1, totalChunks);
  }

  const finishResponse = await fetch(`/api/uploads/${uploadId}/finish`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
  const finishPayload = await finishResponse.json();
  if (!finishResponse.ok) throw new Error(finishPayload.error || '完成上传失败');
  return finishPayload.task;
}

async function uploadVideo(event) {
  event.preventDefault();
  const fileInput = document.getElementById('video-input');
  const nameInput = document.getElementById('task-name');
  const pipelineSelect = document.getElementById('pipeline-select');
  const submitButton = document.getElementById('submit-btn');
  const message = document.getElementById('upload-message');
  const files = Array.from(fileInput.files || []);
  if (!files.length) {
    message.textContent = '请先选择视频文件。';
    return;
  }

  state.uploading = true;
  submitButton.disabled = true;
  const requestedPrefix = nameInput.value.trim();
  const successNames = [];
  const failures = [];

  try {
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      const requestedName = buildRequestedName(requestedPrefix, index, files.length);
      try {
        const task = await uploadFileByChunks(file, requestedName, pipelineSelect.value, (chunkIndex, totalChunks) => {
          message.textContent = `上传中 ${index + 1}/${files.length}：${file.name}（分片 ${chunkIndex}/${totalChunks}）`;
        });
        successNames.push(task.task_name);
      } catch (error) {
        failures.push(`${file.name}：${error.message}`);
      }
    }

    if (successNames.length && !failures.length) {
      message.textContent = `已入队 ${successNames.length} 个视频：${successNames.join('，')}`;
    } else if (successNames.length && failures.length) {
      message.textContent = `部分成功。已入队 ${successNames.length} 个；失败 ${failures.length} 个：${failures.join('；')}`;
    } else {
      message.textContent = `上传失败：${failures.join('；')}`;
    }
  } finally {
    state.uploading = false;
    submitButton.disabled = false;
    fileInput.value = '';
    nameInput.value = '';
    renderSelectedFiles();
    await fetchTasks();
    await fetchRuns();
  }
}

async function submitIssueForm(event) {
  const form = event.target;
  if (!(form instanceof HTMLFormElement) || !form.matches('.issue-form')) return;
  event.preventDefault();

  const formData = new FormData(form);
  const note = String(formData.get('note') || '').trim();
  const statusNode = form.querySelector('.issue-status');
  if (!note) {
    if (statusNode) statusNode.textContent = '请先写问题描述。';
    return;
  }

  const payload = {
    note,
    time_range: String(formData.get('time_range') || '').trim(),
    category: String(formData.get('category') || '其他').trim(),
  };
  if (form.dataset.kind === 'task') {
    payload.task_id = form.dataset.target;
  } else {
    payload.run_name = form.dataset.target;
  }

  if (statusNode) statusNode.textContent = '保存中…';
  try {
    const response = await fetch('/api/issues', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '保存失败');
    form.reset();
    if (statusNode) statusNode.textContent = '已保存，后面我就按这条备注排查。';
    await fetchTasks();
    await fetchRuns();
  } catch (error) {
    if (statusNode) statusNode.textContent = `保存失败：${error.message}`;
  }
}

function bindDropzone() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('video-input');
  ['dragenter', 'dragover'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove('dragover');
  }));
  dropzone.addEventListener('drop', (event) => {
    if (event.dataTransfer?.files?.length) {
      fileInput.files = event.dataTransfer.files;
      renderSelectedFiles();
    }
  });
  fileInput.addEventListener('change', renderSelectedFiles);
}

async function bootstrap() {
  document.getElementById('upload-form').addEventListener('submit', uploadVideo);
  document.getElementById('refresh-tasks').addEventListener('click', fetchTasks);
  document.getElementById('refresh-runs').addEventListener('click', fetchRuns);
  document.getElementById('pipeline-select').addEventListener('change', updatePipelineNote);
  document.addEventListener('submit', submitIssueForm);
  bindDropzone();
  renderSelectedFiles();
  await fetchHealth();
  updatePipelineNote();
  await fetchTasks();
  await fetchRuns();
  window.setInterval(() => { if (!state.uploading) fetchTasks(); }, 3000);
  window.setInterval(() => { if (!state.uploading) fetchRuns(); }, 12000);
  window.setInterval(() => { if (!state.uploading) fetchHealth(); }, 15000);
}

bootstrap();
