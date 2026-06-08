/**
 * ┌──────────────────────────────────────────┐
 * │ 【逻辑】useCountAnimation.js — 数字动画     │
 * │ 职责：数字从N滚动到M的平滑过渡动画          │
 * │ 用于统计卡片中的数字变化效果               │
 * └──────────────────────────────────────────┘
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

import { ref, watch, onUnmounted } from 'vue'

export function useCountAnimation(target, { duration = 1200, easing = true } = {}) {
  const animatedValue = ref(0)
  let rafId = null

  const animate = (from, to) => {
    if (rafId) cancelAnimationFrame(rafId)
    if (from === to) {
      animatedValue.value = to
      return
    }
    const startTime = performance.now()

    const tick = (now) => {
      const elapsed = now - startTime
      const progress = Math.min(elapsed / duration, 1)
      const eased = easing ? 1 - Math.pow(1 - progress, 3) : progress
      animatedValue.value = Math.round(from + (to - from) * eased)
      if (progress < 1) {
        rafId = requestAnimationFrame(tick)
      }
    }

    rafId = requestAnimationFrame(tick)
  }

  watch(target, (newVal) => {
    animate(animatedValue.value, newVal)
  }, { immediate: true })

  onUnmounted(() => {
    if (rafId) cancelAnimationFrame(rafId)
  })

  return animatedValue
}
