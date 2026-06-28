/**
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import axios from 'axios'

vi.mock('axios')
const mockedAxios = vi.mocked(axios)

function createMockAxiosInstance() {
  return {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  }
}

describe('API Client', () => {
  beforeEach(() => {
    vi.resetModules()
    vi.clearAllMocks()
    const instance = createMockAxiosInstance()
    mockedAxios.create.mockReturnValue(instance)
    return instance
  })

  describe('camerasAPI', () => {
    it('list returns devices array', async () => {
      const instance = mockedAxios.create()
      instance.get.mockResolvedValue({
        data: { devices: [{ id: 0, name: 'Camera 0', available: true }] },
      })

      const { camerasAPI } = await import('../src/api/client')
      await camerasAPI.list()

      expect(instance.get).toHaveBeenCalledWith('/cameras/')
    })
  })

  describe('detectionAPI', () => {
    it('start sends source and config', async () => {
      const instance = mockedAxios.create()
      instance.post.mockResolvedValue({ data: { status: 'started' } })

      const { detectionAPI } = await import('../src/api/client')
      await detectionAPI.start('0', 'configs/default.yaml')

      expect(instance.post).toHaveBeenCalledWith(
        '/detection/start',
        {
          source: '0',
          config: 'configs/default.yaml',
        },
        { timeout: 90000 }
      )
    })

    it('stop returns status', async () => {
      const instance = mockedAxios.create()
      instance.post.mockResolvedValue({ data: { status: 'stopped' } })

      const { detectionAPI } = await import('../src/api/client')
      await detectionAPI.stop()

      expect(instance.post).toHaveBeenCalledWith('/detection/stop', null, { timeout: 30000 })
    })

    it('status returns detection status', async () => {
      const instance = mockedAxios.create()
      instance.get.mockResolvedValue({ data: { running: true, fps: 30, frame_count: 100 } })

      const { detectionAPI } = await import('../src/api/client')
      await detectionAPI.status()

      expect(instance.get).toHaveBeenCalledWith('/detection/status')
    })

    it('uploadVideo sends FormData with file', async () => {
      const instance = mockedAxios.create()
      instance.post.mockResolvedValue({ data: { status: 'uploaded', path: '/uploads/video.mp4' } })

      const { detectionAPI } = await import('../src/api/client')
      const mockFile = new File(['video content'], 'video.mp4', { type: 'video/mp4' })
      await detectionAPI.uploadVideo(mockFile)

      expect(instance.post).toHaveBeenCalled()
    })
  })

  describe('eventsAPI', () => {
    it('list sends params correctly', async () => {
      const instance = mockedAxios.create()
      instance.get.mockResolvedValue({ data: { events: [], total: 0, page: 1, page_size: 50 } })

      const { eventsAPI } = await import('../src/api/client')
      await eventsAPI.list({ event_type: 'running', limit: 10, offset: 0 })

      expect(instance.get).toHaveBeenCalledWith('/events/', {
        params: { event_type: 'running', limit: 10, offset: 0 },
      })
    })

    it('stats returns event statistics', async () => {
      const instance = mockedAxios.create()
      instance.get.mockResolvedValue({
        data: { total_events: 100, by_type: { running: 50, fall: 50 } },
      })

      const { eventsAPI } = await import('../src/api/client')
      await eventsAPI.stats()

      expect(instance.get).toHaveBeenCalledWith('/events/stats')
    })

    it('snapshotUrl generates correct URL', async () => {
      const { eventsAPI } = await import('../src/api/client')
      const url = eventsAPI.snapshotUrl(123)

      expect(url).toBe('/api/events/123/snapshot')
    })
  })
})
