/**
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: MIT
 */

import { Timer, Warning, User, Location, Lightning } from '@element-plus/icons-vue'

/**
 * Mapping of event types to Element Plus tag colors.
 * @param {string} type
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
 * Mapping of event types to hex colors for stat cards.
 * @param {string} type
 * @returns {string} CSS hex color
 */
export function statColor(type) {
  const colors = {
    running: '#e6a23c',
    fall: '#f56c6c',
    crowd: '#909399',
    intrusion: '#f56c6c',
    fight: '#f56c6c',
  }
  return colors[type] || '#409eff'
}

/**
 * Mapping of event types to Element Plus icons.
 * @param {string} type
 * @returns {object} Icon component
 */
export function statIcon(type) {
  const icons = {
    running: Timer,
    fall: Warning,
    crowd: User,
    intrusion: Location,
    fight: Lightning,
  }
  return icons[type] || Warning
}

/**
 * Confidence color: green for high, yellow for medium, red for low.
 * @param {number} conf - confidence 0-1
 * @returns {string} CSS hex color
 */
export function confidenceColor(conf) {
  if (conf >= 0.8) return '#67c23a'
  if (conf >= 0.5) return '#e6a23c'
  return '#f56c6c'
}

/**
 * Alarm level to Element Plus tag color.
 * @param {number} level - 3=critical, 2=warning, 1=info
 * @returns {string} Element Plus color name
 */
export function levelColor(level) {
  if (level >= 3) return 'danger'
  if (level === 2) return 'warning'
  return 'info'
}

/**
 * Alarm status to Element Plus tag color.
 * @param {string} status
 * @returns {string} Element Plus color name
 */
export function statusColor(status) {
  const colors = {
    active: 'danger',
    escalated: 'danger',
    acknowledged: 'warning',
    resolved: 'success',
  }
  return colors[status] || 'info'
}

/**
 * Human-readable alarm status label.
 * @param {string} status
 * @returns {string} Chinese label
 */
export function statusLabel(status) {
  const labels = { active: '活跃', escalated: '已升级', acknowledged: '已确认', resolved: '已解决' }
  return labels[status] || status
}

/**
 * Format a Unix timestamp (seconds) to locale date-time string.
 * @param {number} ts - Unix timestamp in seconds
 * @returns {string}
 */
export function formatDateTime(ts) {
  if (!ts) return ''
  return new Date(ts * 1000).toLocaleString()
}

/**
 * Format a Unix timestamp (seconds) to locale time string (no date).
 * @param {number} ts - Unix timestamp in seconds
 * @returns {string}
 */
export function formatTime(ts) {
  if (!ts) return ''
  return new Date(ts * 1000).toLocaleTimeString()
}

/**
 * Human-readable event type label in Chinese.
 * @param {string} type
 * @returns {string}
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
