/**
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import ActionButton from '../../src/components/ActionButton.vue'

// Mock Element Plus components
vi.mock('element-plus', () => ({
  ElMessage: {
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
  },
  ElMessageBox: {
    confirm: vi.fn(),
  },
}))

import { ElMessage, ElMessageBox } from 'element-plus'

describe('ActionButton', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders button with default slot content', () => {
    const wrapper = mount(ActionButton, {
      slots: { default: 'Click Me' },
      props: { action: vi.fn() },
    })
    expect(wrapper.text()).toBe('Click Me')
  })

  it('shows loading state during action execution', async () => {
    const action = vi.fn(async () => {
      await new Promise((resolve) => setTimeout(resolve, 10))
    })

    const wrapper = mount(ActionButton, {
      props: { action },
      slots: { default: 'Submit' },
    })

    wrapper.find('button').trigger('click')
    await nextTick()

    expect(wrapper.props('loading')).toBe(true)

    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(wrapper.props('loading')).toBe(false)
  })

  it('disables button when disabled prop is true', () => {
    const wrapper = mount(ActionButton, {
      props: { action: vi.fn(), disabled: true },
      slots: { default: 'Submit' },
    })

    expect(wrapper.props('disabled')).toBe(true)
  })

  it('does not execute action when disabled', async () => {
    const action = vi.fn(async () => 'result')

    const wrapper = mount(ActionButton, {
      props: { action, disabled: true },
      slots: { default: 'Submit' },
    })

    await wrapper.find('button').trigger('click')

    expect(action).not.toHaveBeenCalled()
  })

  it('shows success message when action succeeds and successMessage is provided', async () => {
    const action = vi.fn(async () => ({ status: 'ok' }))

    const wrapper = mount(ActionButton, {
      props: {
        action,
        successMessage: 'Operation successful',
      },
      slots: { default: 'Submit' },
    })

    await wrapper.find('button').trigger('click')
    await nextTick()

    expect(ElMessage.success).toHaveBeenCalledWith('Operation successful')
  })

  it('shows error message when action fails and errorMessage is provided', async () => {
    const action = vi.fn(async () => {
      throw new Error('Network error')
    })

    const wrapper = mount(ActionButton, {
      props: {
        action,
        errorMessage: 'Operation failed',
      },
      slots: { default: 'Submit' },
    })

    await wrapper.find('button').trigger('click')
    await nextTick()

    expect(ElMessage.error).toHaveBeenCalledWith('Operation failed')
  })

  it('emits success event with result when action succeeds', async () => {
    const action = vi.fn(async () => ({ id: 1 }))

    const wrapper = mount(ActionButton, {
      props: { action },
      slots: { default: 'Submit' },
    })

    await wrapper.find('button').trigger('click')
    await nextTick()

    expect(wrapper.emitted('success')).toBeTruthy()
    expect(wrapper.emitted('success')[0]).toEqual([{ id: 1 }])
  })

  it('emits error event with error when action fails', async () => {
    const error = new Error('test error')
    const action = vi.fn(async () => {
      throw error
    })

    const wrapper = mount(ActionButton, {
      props: { action },
      slots: { default: 'Submit' },
    })

    await wrapper.find('button').trigger('click')
    await nextTick()

    expect(wrapper.emitted('error')).toBeTruthy()
    expect(wrapper.emitted('error')[0][0]).toBe(error)
  })

  it('passes type prop to el-button', () => {
    const wrapper = mount(ActionButton, {
      props: { action: vi.fn(), type: 'danger' },
      slots: { default: 'Delete' },
    })

    expect(wrapper.props('type')).toBe('danger')
  })

  it('passes size prop to el-button', () => {
    const wrapper = mount(ActionButton, {
      props: { action: vi.fn(), size: 'small' },
      slots: { default: 'Submit' },
    })

    expect(wrapper.props('size')).toBe('small')
  })
})

describe('ActionButton with confirm', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows confirm dialog when confirmConfig is provided', async () => {
    ElMessageBox.confirm.mockResolvedValue()

    const action = vi.fn(async () => 'result')
    const confirmConfig = {
      title: 'Confirm Delete',
      message: 'Are you sure?',
      type: 'warning',
    }

    const wrapper = mount(ActionButton, {
      props: { action, confirmConfig },
      slots: { default: 'Delete' },
    })

    await wrapper.find('button').trigger('click')
    await nextTick()

    expect(ElMessageBox.confirm).toHaveBeenCalledWith(
      'Are you sure?',
      'Confirm Delete',
      expect.objectContaining({ type: 'warning' })
    )
  })

  it('does not execute action when user cancels confirm', async () => {
    ElMessageBox.confirm.mockRejectedValue(new Error('cancel'))

    const action = vi.fn(async () => 'result')
    const confirmConfig = {
      title: 'Confirm',
      message: 'Are you sure?',
    }

    const wrapper = mount(ActionButton, {
      props: { action, confirmConfig },
      slots: { default: 'Delete' },
    })

    await wrapper.find('button').trigger('click')
    await nextTick()

    expect(action).not.toHaveBeenCalled()
  })

  it('executes action when user confirms', async () => {
    ElMessageBox.confirm.mockResolvedValue()

    const action = vi.fn(async () => 'result')
    const confirmConfig = {
      title: 'Confirm',
      message: 'Are you sure?',
    }

    const wrapper = mount(ActionButton, {
      props: { action, confirmConfig },
      slots: { default: 'Delete' },
    })

    await wrapper.find('button').trigger('click')
    await nextTick()

    expect(action).toHaveBeenCalled()
  })
})
