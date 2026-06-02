/**
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
})

const _pendingRequests = new Map()

function dedupRequest(key, fn) {
  if (_pendingRequests.has(key)) return _pendingRequests.get(key)
  const promise = fn().finally(() => _pendingRequests.delete(key))
  _pendingRequests.set(key, promise)
  return promise
}

const _cache = new Map()
const CACHE_TTL = 2000

function cachedGet(key, fn) {
  const cached = _cache.get(key)
  if (cached && Date.now() - cached.ts < CACHE_TTL) return Promise.resolve(cached.data)
  _cache.delete(key)
  return fn().then((data) => {
    _cache.set(key, { data, ts: Date.now() })
    return data
  })
}

export const camerasAPI = {
  list: () => cachedGet('cameras:list', () => api.get('/cameras/').then((r) => r.data)),
}

export const detectionAPI = {
  start: (source = '0', config = 'configs/default.yaml') =>
    dedupRequest(`detection:start:${source}`, () =>
      api.post('/detection/start', { source, config }, { timeout: 90000 }).then((r) => r.data)
    ),
  stop: () =>
    dedupRequest('detection:stop', () =>
      api.post('/detection/stop', null, { timeout: 30000 }).then((r) => r.data)
    ),
  status: () => api.get('/detection/status').then((r) => r.data),
  progress: () => api.get('/detection/progress').then((r) => r.data),
  saveConfig: (configData) => api.post('/detection/save-config', configData).then((r) => r.data),
  uploadVideo: (file, onProgress) => {
    return dedupRequest(`detection:upload:${file.name}:${file.size}`, () => {
      const form = new FormData()
      form.append('file', file)
      return api.post('/detection/upload', form, {
        timeout: 120000,
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total)
            onProgress(percent)
          }
        },
      }).then((r) => r.data)
    })
  },
}

export const eventsAPI = {
  list: (params = {}) => api.get('/events/', { params }).then((r) => r.data),
  stats: () => cachedGet('events:stats', () => api.get('/events/stats').then((r) => r.data)),
  types: () => cachedGet('events:types', () => api.get('/events/types').then((r) => r.data)),
  deleteAll: () => api.delete('/events/all').then((r) => r.data),
  snapshotUrl: (eventId) => `${API_BASE}/events/${eventId}/snapshot`,
}

export const alarmsAPI = {
  list: (params = {}) => api.get('/alarms/', { params }).then((r) => r.data),
  stats: () => cachedGet('alarms:stats', () => api.get('/alarms/stats').then((r) => r.data)),
  acknowledge: (id) => api.post(`/alarms/${id}/acknowledge`).then((r) => r.data),
  resolve: (id) => api.post(`/alarms/${id}/resolve`).then((r) => r.data),
  deleteAll: () => api.delete('/alarms/all').then((r) => r.data),
}

export const mllmAPI = {
  status: () => cachedGet('mllm:status', () => api.get('/mllm/status').then((r) => r.data)),
  getConfig: () => api.get('/mllm/config').then((r) => r.data),
  updateConfig: (config) => api.post('/mllm/config', config).then((r) => r.data),
  enable: (enabled = true, shadowMode = true) =>
    api.post('/mllm/enable', { enabled, shadow_mode: shadowMode }).then((r) => r.data),
}

export const configAPI = {
  get: () => api.get('/config').then((r) => r.data),
  updateRules: (rules) => api.post('/config/rules', rules).then((r) => r.data),
  updateSettings: (settings) => api.post('/config/settings', settings).then((r) => r.data),
  updateZones: (zones) => api.post('/config/zones', zones).then((r) => r.data),
}

export const archivesAPI = {
  list: (params = {}) => api.get('/archives', { params }).then((r) => r.data),
  clipUrl: (clipId) => `${API_BASE}/archives/${clipId}`,
}

export default api
