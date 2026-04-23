import axios from 'axios'

const API_BASE = '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
})

export const camerasAPI = {
  list: () => api.get('/cameras/').then(r => r.data),
  getInfo: (id) => api.get(`/cameras/${id}`).then(r => r.data),
}

export const detectionAPI = {
  start: (source = '0', config = 'configs/default.yaml') =>
    api.post('/detection/start', { source, config }).then(r => r.data),
  stop: () => api.post('/detection/stop').then(r => r.data),
  status: () => api.get('/detection/status').then(r => r.data),
  updateConfig: (config = 'configs/default.yaml') =>
    api.post('/detection/config', { source: '0', config }).then(r => r.data),
  saveConfig: (configData) =>
    api.post('/detection/save-config', configData).then(r => r.data),
  uploadVideo: (file) => {
    const form = new FormData()
    form.append('file', file)
    return api.post('/detection/upload', form).then(r => r.data)
  },
}

export const eventsAPI = {
  list: (params = {}) => api.get('/events/', { params }).then(r => r.data),
  stats: () => api.get('/events/stats').then(r => r.data),
  types: () => api.get('/events/types').then(r => r.data),
  deleteEvents: (params = {}) => api.delete('/events/', { params }).then(r => r.data),
  deleteAll: () => api.delete('/events/all').then(r => r.data),
  snapshotUrl: (eventId) => `${API_BASE}/events/${eventId}/snapshot`,
}

export default api
