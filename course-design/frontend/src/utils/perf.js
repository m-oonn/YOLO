/**
 * Performance monitoring utilities.
 *
 * Provides lightweight performance measurement for:
 * - Page load timing (Navigation Timing API)
 * - Component render timing
 * - API request latency tracking
 * - FPS monitoring for video streams
 */

/**
 * Get page load performance metrics using modern Navigation Timing API.
 * @returns {Object|null} Performance metrics or null if API unavailable
 */
export function getPageLoadMetrics() {
  const entries = performance.getEntriesByType('navigation')
  if (!entries || entries.length === 0) return null
  const nav = entries[0]
  return {
    dns: nav.domainLookupEnd - nav.domainLookupStart,
    tcp: nav.connectEnd - nav.connectStart,
    ttfb: nav.responseStart - nav.requestStart,
    download: nav.responseEnd - nav.responseStart,
    domParse: nav.domComplete - nav.domLoading,
    total: nav.loadEventEnd - nav.startTime,
  }
}

/**
 * Measure function execution time (supports async).
 * @param {Function} fn - Function to measure
 * @param {string} label - Measurement label
 * @returns {*} Function result
 */
export async function measure(fn, label) {
  const start = performance.now()
  const result = await fn()
  const duration = performance.now() - start
  if (duration > 16) {
    console.warn(`[Perf] ${label}: ${duration.toFixed(1)}ms`)
  }
  return result
}

/**
 * Create an FPS counter for video/stream monitoring.
 * @param {Function} callback - Called with current FPS
 * @returns {Object} { start, stop }
 */
export function createFpsCounter(callback) {
  let frameCount = 0
  let lastTime = performance.now()
  let rafId = null

  function tick() {
    frameCount++
    const now = performance.now()
    if (now - lastTime >= 1000) {
      callback(frameCount)
      frameCount = 0
      lastTime = now
    }
    rafId = requestAnimationFrame(tick)
  }

  return {
    start() {
      frameCount = 0
      lastTime = performance.now()
      rafId = requestAnimationFrame(tick)
    },
    stop() {
      if (rafId) cancelAnimationFrame(rafId)
    },
  }
}

/**
 * Debounce function calls.
 * @param {Function} fn - Function to debounce
 * @param {number} delay - Delay in milliseconds
 * @returns {Function} Debounced function
 */
export function debounce(fn, delay) {
  let timer = null
  return function (...args) {
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => fn.apply(this, args), delay)
  }
}

/**
 * Throttle function calls.
 * @param {Function} fn - Function to throttle
 * @param {number} limit - Time limit in milliseconds
 * @returns {Function} Throttled function
 */
export function throttle(fn, limit) {
  let inThrottle = false
  return function (...args) {
    if (!inThrottle) {
      fn.apply(this, args)
      inThrottle = true
      setTimeout(() => (inThrottle = false), limit)
    }
  }
}
