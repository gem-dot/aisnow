const { BASE_URL } = require('../config')

let activeBaseUrl = ''

function normalizeBaseUrl(url) {
  return String(url || '').trim().replace(/\/$/, '')
}

function getBaseUrl() {
  if (activeBaseUrl) return activeBaseUrl
  try {
    const saved = wx.getStorageSync('skiMiniBaseUrl')
    activeBaseUrl = normalizeBaseUrl(saved || BASE_URL)
  } catch (error) {
    activeBaseUrl = normalizeBaseUrl(BASE_URL)
  }
  return activeBaseUrl
}

function setBaseUrl(url) {
  activeBaseUrl = normalizeBaseUrl(url || BASE_URL)
  try {
    wx.setStorageSync('skiMiniBaseUrl', activeBaseUrl)
  } catch (error) {}
  return activeBaseUrl
}

function joinUrl(path) {
  const baseUrl = getBaseUrl()
  if (!path) return baseUrl
  if (path.startsWith('http://') || path.startsWith('https://')) return path
  if (path.startsWith('/')) return `${baseUrl}${path}`
  return `${baseUrl}/${path}`
}

function request(path, options = {}) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: joinUrl(path),
      method: options.method || 'GET',
      header: options.header || { 'Content-Type': 'application/json' },
      data: options.data || {},
      success(res) {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve(res.data)
          return
        }
        reject(new Error((res.data && res.data.error) || `HTTP ${res.statusCode}`))
      },
      fail(err) {
        reject(err)
      }
    })
  })
}

function uploadVideo(filePath, formData = {}, onProgress) {
  return new Promise((resolve, reject) => {
    const uploadTask = wx.uploadFile({
      url: joinUrl('/api/upload'),
      filePath,
      name: 'video',
      formData,
      success(res) {
        let payload = {}
        try {
          payload = JSON.parse(res.data || '{}')
        } catch (error) {
          reject(new Error('服务端返回解析失败'))
          return
        }
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve(payload)
          return
        }
        reject(new Error(payload.error || `HTTP ${res.statusCode}`))
      },
      fail(err) {
        reject(err)
      }
    })
    if (uploadTask && typeof uploadTask.onProgressUpdate === 'function' && onProgress) {
      uploadTask.onProgressUpdate(onProgress)
    }
  })
}

module.exports = {
  BASE_URL,
  getBaseUrl,
  setBaseUrl,
  joinUrl,
  request,
  uploadVideo,
}
