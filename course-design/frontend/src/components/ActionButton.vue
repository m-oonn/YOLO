<!--
  Copyright (c) 2025 YOLO Course Design Contributors
  SPDX-License-Identifier: Apache-2.0
-->

<template>
  <el-button
    :type="type"
    :size="size"
    :loading="isLoading"
    :disabled="disabled || isLoading"
    :plain="plain"
    :text="text"
    :link="link"
    :circle="circle"
    :icon="icon"
    @click="handleClick"
  >
    <slot />
  </el-button>
</template>

<script setup>
/**
 * ActionButton - Unified async action button with loading, confirm, and message handling.
 *
 * Wraps Element Plus el-button with consistent async action patterns:
 * - Automatic loading state during execution
 * - Optional confirmation dialog before action
 * - Automatic success/error message display
 * - Success/error event emission
 *
 * @example
 * <!-- Basic usage -->
 * <ActionButton :action="saveData" successMessage="Saved!">Save</ActionButton>
 *
 * <!-- With confirmation -->
 * <ActionButton
 *   type="danger"
 *   :action="deleteItem"
 *   :confirmConfig="{ title: 'Confirm', message: 'Delete this?', type: 'warning' }"
 *   successMessage="Deleted"
 *   @success="reloadList"
 * >
 *   Delete
 * </ActionButton>
 */
import { ElMessage, ElMessageBox } from 'element-plus'
import { useAction } from '../composables/useAction.js'

const props = defineProps({
  /** The async function to execute when clicked */
  action: { type: Function, required: true },
  /** Element Plus button type */
  type: { type: String, default: 'default' },
  /** Element Plus button size */
  size: { type: String, default: 'default' },
  /** Disable the button */
  disabled: { type: Boolean, default: false },
  /** Plain style */
  plain: { type: Boolean, default: false },
  /** Text style */
  text: { type: Boolean, default: false },
  /** Link style */
  link: { type: Boolean, default: false },
  /** Circle shape */
  circle: { type: Boolean, default: false },
  /** Icon component */
  icon: { type: [Object, Function], default: null },
  /** Confirmation dialog config: { title, message, type, confirmButtonText, cancelButtonText } */
  confirmConfig: { type: Object, default: null },
  /** Success message to display (string or function receiving result) */
  successMessage: { type: [String, Function], default: '' },
  /** Error message to display (string or function receiving error) */
  errorMessage: { type: [String, Function], default: '' },
})

const emit = defineEmits(['success', 'error'])

function getSuccessMessage(result) {
  if (!props.successMessage) return ''
  return typeof props.successMessage === 'function'
    ? props.successMessage(result)
    : props.successMessage
}

function getErrorMessage(error) {
  if (!props.errorMessage) return ''
  return typeof props.errorMessage === 'function'
    ? props.errorMessage(error)
    : props.errorMessage
}

function handleSuccess(result) {
  const message = getSuccessMessage(result)
  if (message) {
    ElMessage.success(message)
  }
  emit('success', result)
}

function handleError(error) {
  const message = getErrorMessage(error)
  if (message) {
    ElMessage.error(message)
  }
  emit('error', error)
}

const { isLoading, execute } = useAction({
  action: props.action,
  onSuccess: handleSuccess,
  onError: handleError,
})

async function handleClick() {
  if (props.disabled || isLoading.value) return

  // Show confirmation dialog if configured
  if (props.confirmConfig) {
    try {
      await ElMessageBox.confirm(
        props.confirmConfig.message,
        props.confirmConfig.title,
        {
          type: props.confirmConfig.type || 'warning',
          confirmButtonText: props.confirmConfig.confirmButtonText,
          cancelButtonText: props.confirmConfig.cancelButtonText,
        }
      )
    } catch {
      // User cancelled
      return
    }
  }

  await execute()
}
</script>
