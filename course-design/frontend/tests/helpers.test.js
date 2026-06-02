/**
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest'
import { eventTypeColor, formatTime, formatDateTime, eventTypeLabel } from '../src/utils/helpers'

describe('helpers.js', () => {
  describe('eventTypeColor', () => {
    it('returns warning for running', () => {
      expect(eventTypeColor('running')).toBe('warning')
    })

    it('returns danger for fall', () => {
      expect(eventTypeColor('fall')).toBe('danger')
    })

    it('returns info for crowd', () => {
      expect(eventTypeColor('crowd')).toBe('info')
    })

    it('returns danger for intrusion', () => {
      expect(eventTypeColor('intrusion')).toBe('danger')
    })

    it('returns danger for fight', () => {
      expect(eventTypeColor('fight')).toBe('danger')
    })

    it('returns info for unknown types', () => {
      expect(eventTypeColor('unknown')).toBe('info')
      expect(eventTypeColor('')).toBe('info')
    })
  })

  describe('formatTime', () => {
    it('returns empty string for null/undefined', () => {
      expect(formatTime(null)).toBe('')
      expect(formatTime(undefined)).toBe('')
    })

    it('returns a formatted time string for valid timestamp', () => {
      const result = formatTime(1713840000)
      expect(result).toBeTruthy()
      expect(typeof result).toBe('string')
    })
  })

  describe('formatDateTime', () => {
    it('returns empty string for null/undefined', () => {
      expect(formatDateTime(null)).toBe('')
      expect(formatDateTime(undefined)).toBe('')
    })

    it('returns a formatted datetime string for valid timestamp', () => {
      const result = formatDateTime(1713840000)
      expect(result).toBeTruthy()
      expect(typeof result).toBe('string')
    })
  })

  describe('eventTypeLabel', () => {
    it('returns Chinese labels for known types', () => {
      expect(eventTypeLabel('running')).toBe('奔跑')
      expect(eventTypeLabel('fall')).toBe('跌倒')
      expect(eventTypeLabel('crowd')).toBe('人群聚集')
      expect(eventTypeLabel('intrusion')).toBe('区域入侵')
      expect(eventTypeLabel('fight')).toBe('打架斗殴')
    })

    it('returns the type itself for unknown types', () => {
      expect(eventTypeLabel('unknown')).toBe('unknown')
      expect(eventTypeLabel('')).toBe('')
    })
  })
})
