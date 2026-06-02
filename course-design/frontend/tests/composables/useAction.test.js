/**
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ref, nextTick } from 'vue'
import { useAction } from '../../src/composables/useAction'

describe('useAction', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('returns isLoading as false initially', () => {
    const { isLoading } = useAction({ action: vi.fn() })
    expect(isLoading.value).toBe(false)
  })

  it('sets isLoading to true during action execution', async () => {
    let loadingDuringExecution = false
    const action = vi.fn(async () => {
      loadingDuringExecution = true
      await new Promise((resolve) => setTimeout(resolve, 10))
    })

    const { isLoading, execute } = useAction({ action })
    const promise = execute()

    expect(isLoading.value).toBe(true)
    await promise
    expect(isLoading.value).toBe(false)
    expect(loadingDuringExecution).toBe(true)
  })

  it('resets isLoading to false even when action throws', async () => {
    const action = vi.fn(async () => {
      throw new Error('test error')
    })

    const { isLoading, execute } = useAction({ action })

    await expect(execute()).rejects.toThrow('test error')
    expect(isLoading.value).toBe(false)
  })

  it('passes arguments to the action function', async () => {
    const action = vi.fn(async (a, b) => a + b)

    const { execute } = useAction({ action })
    const result = await execute(2, 3)

    expect(action).toHaveBeenCalledWith(2, 3)
    expect(result).toBe(5)
  })

  it('returns action result', async () => {
    const action = vi.fn(async () => ({ status: 'ok' }))

    const { execute } = useAction({ action })
    const result = await execute()

    expect(result).toEqual({ status: 'ok' })
  })

  it('prevents concurrent execution when already loading', async () => {
    const action = vi.fn(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50))
      return 'result'
    })

    const { isLoading, execute } = useAction({ action })

    const promise1 = execute()
    const promise2 = execute()

    expect(action).toHaveBeenCalledTimes(1)
    await promise1
    await promise2
    expect(action).toHaveBeenCalledTimes(1)
  })

  it('allows new execution after previous completes', async () => {
    const action = vi.fn(async () => 'result')

    const { execute } = useAction({ action })

    await execute()
    await execute()

    expect(action).toHaveBeenCalledTimes(2)
  })
})

describe('useAction with success callback', () => {
  it('calls onSuccess callback with result', async () => {
    const action = vi.fn(async () => ({ id: 1 }))
    const onSuccess = vi.fn()

    const { execute } = useAction({ action, onSuccess })
    await execute()

    expect(onSuccess).toHaveBeenCalledWith({ id: 1 })
  })

  it('does not call onSuccess when action throws', async () => {
    const action = vi.fn(async () => {
      throw new Error('fail')
    })
    const onSuccess = vi.fn()

    const { execute } = useAction({ action, onSuccess })

    await expect(execute()).rejects.toThrow('fail')
    expect(onSuccess).not.toHaveBeenCalled()
  })
})

describe('useAction with error callback', () => {
  it('calls onError callback with error', async () => {
    const error = new Error('test error')
    const action = vi.fn(async () => {
      throw error
    })
    const onError = vi.fn()

    const { execute } = useAction({ action, onError })

    await expect(execute()).rejects.toThrow('test error')
    expect(onError).toHaveBeenCalledWith(error)
  })

  it('does not call onError when action succeeds', async () => {
    const action = vi.fn(async () => 'ok')
    const onError = vi.fn()

    const { execute } = useAction({ action, onError })
    await execute()

    expect(onError).not.toHaveBeenCalled()
  })
})
