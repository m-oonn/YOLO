/**
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: MIT
 */

import axios from 'axios'

const API_BASE = '/api'

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
      api.post('/detection/start', { source, config }).then((r) => r.data)
    ),
  stop: () =>
    dedupRequest('detection:stop', () =>
      api.post('/detection/stop').then((r) => r.data)
    ),
  status: () => api.get('/detection/status').then((r) => r.data),
  saveConfig: (configData) => api.post('/detection/save-config', configData).then((r) => r.data),
  uploadVideo: (file) => {
    const form = new FormData()
    form.append('file', file)
    return api.post('/detection/upload', form).then((r) => r.data)
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
}

export const mllmAPI = {
  status: () => cachedGet('mllm:status', () => api.get('/mllm/status').then((r) => r.data)),
  enable: (enabled = true, shadowMode = true) =>
    api.post('/mllm/enable', { enabled, shadow_mode: shadowMode }).then((r) => r.data),
}

export const archivesAPI = {
  list: (params = {}) => api.get('/archives', { params }).then((r) => r.data),
  clipUrl: (clipId) => `${API_BASE}/archives/${clipId}`,
}

export default api
