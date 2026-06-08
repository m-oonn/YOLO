/**
 * ┌──────────────────────────────────────────┐
 * │ 【逻辑】useAction.js — 操作按钮逻辑        │
 * │ 职责：封装异步操作的loading/error/success  │
 * │ 用于报警确认/解决等带状态的操作            │
 * └──────────────────────────────────────────┘
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

import { ref } from 'vue'

/**
 * useAction - Unified async action handler with loading state management.
 *
 * Provides a consistent pattern for handling async button actions:
 * - Loading state tracking (prevents concurrent execution)
 * - Success/error callback support
 * - Automatic loading state cleanup (even on errors)
 *
 * @param {Object} options
 * @param {Function} options.action - The async function to execute
 * @param {Function} [options.onSuccess] - Callback on successful execution (receives result)
 * @param {Function} [options.onError] - Callback on error (receives error)
 * @returns {Object} { isLoading, execute }
 *
 * @example
 * const { isLoading, execute } = useAction({
 *   action: async (id) => await api.delete(id),
 *   onSuccess: (result) => ElMessage.success('Deleted'),
 *   onError: (err) => ElMessage.error(err.message),
 * })
 *
 * // In template:
 * <el-button :loading="isLoading" @click="execute(itemId)">Delete</el-button>
 */
export function useAction(options) {
  const { action, onSuccess, onError } = options
  const isLoading = ref(false)
  let currentPromise = null

  async function execute(...args) {
    // Prevent concurrent execution
    if (isLoading.value || currentPromise) {
      return currentPromise
    }

    isLoading.value = true

    currentPromise = (async () => {
      try {
        const result = await action(...args)
        if (onSuccess) {
          onSuccess(result)
        }
        return result
      } catch (error) {
        if (onError) {
          onError(error)
        }
        throw error
      } finally {
        isLoading.value = false
        currentPromise = null
      }
    })()

    return currentPromise
  }

  return {
    isLoading,
    execute,
  }
}
