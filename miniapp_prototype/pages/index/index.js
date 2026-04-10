const api = require('../../utils/api')

Page({
  data: {
    baseUrl: api.getBaseUrl(),
    pipelines: [],
    pipelineLabels: [],
    pipelineIndex: 0,
    currentPipeline: {},
    tasks: [],
    runs: [],
    taskName: '',
    selectedVideoPath: '',
    selectedVideoName: '',
    uploadProgress: 0,
    uploading: false,
    statusMessage: '',
  },

  onLoad() {
    this.setData({ baseUrl: api.getBaseUrl() })
    this.refreshAll()
  },

  async refreshAll() {
    try {
      api.setBaseUrl(this.data.baseUrl)
      await this.fetchHealth()
      await Promise.all([this.fetchTasks(), this.fetchRuns()])
      this.setData({ statusMessage: '已刷新最新状态' })
    } catch (error) {
      this.setData({ statusMessage: `刷新失败：${error.message || error.errMsg || error}` })
    }
  },

  async fetchHealth() {
    const payload = await api.request('/api/health')
    const pipelines = payload.pipelines || []
    const pipelineLabels = pipelines.map((item) => item.label)
    const nextIndex = Math.min(this.data.pipelineIndex, Math.max(pipelines.length - 1, 0))
    const currentPipeline = pipelines[nextIndex] || {}
    this.setData({
      pipelines,
      pipelineLabels,
      pipelineIndex: nextIndex,
      currentPipeline,
    })
  },

  async fetchTasks() {
    const payload = await api.request('/api/tasks')
    this.setData({ tasks: payload.tasks || [] })
  },

  async fetchRuns() {
    const payload = await api.request('/api/runs')
    const runs = (payload.runs || []).slice(0, 3).map((item) => {
      const previewVideo = item.files?.side_by_side?.url ? api.joinUrl(item.files.side_by_side.url) : (item.files?.overlay?.url ? api.joinUrl(item.files.overlay.url) : '')
      const pressurePlot = item.review?.pressure_plot?.url ? api.joinUrl(item.review.pressure_plot.url) : ''
      return {
        ...item,
        previewVideo,
        pressurePlot,
      }
    })
    this.setData({ runs })
  },

  onBaseUrlInput(event) {
    this.setData({ baseUrl: event.detail.value })
  },

  onTaskNameInput(event) {
    this.setData({ taskName: event.detail.value })
  },

  onPipelineChange(event) {
    const pipelineIndex = Number(event.detail.value || 0)
    this.setData({
      pipelineIndex,
      currentPipeline: this.data.pipelines[pipelineIndex] || {},
    })
  },

  chooseVideo() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['video'],
      sourceType: ['album', 'camera'],
      success: (res) => {
        const item = (res.tempFiles || [])[0]
        this.setData({
          selectedVideoPath: item?.tempFilePath || '',
          selectedVideoName: item?.size ? `${item.tempFilePath.split('/').pop()} · ${(item.size / 1024 / 1024).toFixed(1)}MB` : (item?.tempFilePath || ''),
          statusMessage: '视频已选择，可以开始上传。',
        })
      },
      fail: (err) => {
        this.setData({ statusMessage: `选视频失败：${err.errMsg || err}` })
      }
    })
  },

  async uploadSelectedVideo() {
    if (!this.data.selectedVideoPath) {
      this.setData({ statusMessage: '请先选择视频。' })
      return
    }
    api.setBaseUrl(this.data.baseUrl)
    const pipeline = this.data.pipelines[this.data.pipelineIndex] || {}
    this.setData({ uploading: true, uploadProgress: 0, statusMessage: '开始上传...' })
    try {
      const payload = await api.uploadVideo(this.data.selectedVideoPath, {
        task_name: this.data.taskName,
        pipeline_id: pipeline.id || 'jsba',
      }, (progress) => {
        this.setData({ uploadProgress: progress.progress || 0, statusMessage: `上传中 ${progress.progress || 0}%` })
      })
      this.setData({
        uploading: false,
        uploadProgress: 100,
        statusMessage: `已入队：${payload.task?.task_name || '任务创建成功'}`,
      })
      await this.refreshAll()
    } catch (error) {
      this.setData({
        uploading: false,
        statusMessage: `上传失败：${error.message || error.errMsg || error}`,
      })
    }
  },

  previewImage(event) {
    const url = event.currentTarget.dataset.url
    if (!url) return
    wx.previewImage({
      urls: [url],
      current: url,
    })
  },
})
