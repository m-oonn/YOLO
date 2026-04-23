/**
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: MIT
 */

/**
 * Map event type strings to Element Plus tag colors.
 * @param {string} type - Event type (running, fall, crowd, intrusion, fight)
 * @returns {string} Element Plus color name
 */
export function eventTypeColor(type) {
  const colors = {
    running: 'warning',
    fall: 'danger',
    crowd: 'info',
    intrusion: 'danger',
    fight: 'danger',
  }
  return colors[type] || 'info'
}

/**
 * Format a Unix timestamp (seconds) to locale date-time string.
 * @param {number} ts - Unix timestamp in seconds
 * @returns {string} Formatted date-time string
 */
export function formatDateTime(ts) {
  if (!ts) return ''
  return new Date(ts * 1000).toLocaleString()
}

/**
 * Format a Unix timestamp (seconds) to locale time string (no date).
 * @param {number} ts - Unix timestamp in seconds
 * @returns {string} Formatted time string
 */
export function formatTime(ts) {
  if (!ts) return ''
  return new Date(ts * 1000).toLocaleTimeString()
}

/**
 * Get a human-readable label for event type.
 * @param {string} type - Event type
 * @returns {string} Chinese label
 */
export function eventTypeLabel(type) {
  const labels = {
    running: '奔跑',
    fall: '跌倒',
    crowd: '人群聚集',
    intrusion: '区域入侵',
    fight: '打架斗殴',
  }
  return labels[type] || type
}
