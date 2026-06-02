/**
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * WebSocket composable with exponential-backoff reconnection.
 * Extracted from MonitorView.vue for reuse and testability.
 */
import { ref } from 'vue'
import { logger } from '../utils/logger'

const WS_MAX_RECONNECT_DELAY = 30000
const WS_INITIAL_RECONNECT_DELAY = 1000
const WS_MAX_RECONNECT_ATTEMPTS = 10

export function useWebSocket() {
  const wsConnected = ref(false)
  let ws = null
  let wsReconnectTimer = null
  let wsReconnectDelay = WS_INITIAL_RECONNECT_DELAY
  let wsReconnectAttempts = 0
  let _onMessage = null
  let _lastUrl = null

  function connect(wsUrl, onMessage) {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return
    _onMessage = onMessage
    _lastUrl = wsUrl

    ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      logger.log('[WS] connected')
      wsConnected.value = true
      wsReconnectAttempts = 0
      wsReconnectDelay = WS_INITIAL_RECONNECT_DELAY
      if (wsReconnectTimer) {
        clearTimeout(wsReconnectTimer)
        wsReconnectTimer = null
      }
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (_onMessage) _onMessage(msg)
      } catch (e) {
        logger.error('[WS] message parse error:', e)
      }
    }

    ws.onclose = () => {
      wsConnected.value = false
      logger.log('[WS] disconnected')
      if (wsReconnectAttempts < WS_MAX_RECONNECT_ATTEMPTS) {
        wsReconnectAttempts++
        const delay = Math.min(
          wsReconnectDelay * Math.pow(1.5, wsReconnectAttempts - 1),
          WS_MAX_RECONNECT_DELAY
        )
        logger.log(`[WS] reconnecting in ${delay}ms (attempt ${wsReconnectAttempts})`)
        wsReconnectTimer = setTimeout(() => connect(wsUrl, onMessage), delay)
      }
    }

    ws.onerror = () => {
      // onclose will fire and handle reconnection
    }
  }

  function disconnect() {
    wsReconnectAttempts = WS_MAX_RECONNECT_ATTEMPTS
    if (wsReconnectTimer) {
      clearTimeout(wsReconnectTimer)
      wsReconnectTimer = null
    }
    wsReconnectDelay = WS_INITIAL_RECONNECT_DELAY
    wsConnected.value = false
    if (ws) {
      ws.close()
      ws = null
    }
  }

  function reconnect() {
    wsReconnectAttempts = 0
    wsReconnectDelay = WS_INITIAL_RECONNECT_DELAY
    if (wsReconnectTimer) {
      clearTimeout(wsReconnectTimer)
      wsReconnectTimer = null
    }
    if (ws) {
      ws.close()
      ws = null
    }
    if (_lastUrl && _onMessage) {
      connect(_lastUrl, _onMessage)
    }
  }

  return { wsConnected, connect, disconnect, reconnect }
}
