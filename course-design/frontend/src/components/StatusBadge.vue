<template>
  <span class="status-badge" :class="[`badge-${status}`, { pulse: animated }]">
    <span class="status-dot" />
    <span class="status-label"><slot>{{ label }}</slot></span>
  </span>
</template>

<script setup>
defineProps({
  status: {
    type: String,
    default: 'online',
    validator: (v) => ['online', 'offline', 'loading', 'warning'].includes(v),
  },
  label: { type: String, default: '' },
  animated: { type: Boolean, default: false },
})
</script>

<style scoped>
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px 3px 8px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
  letter-spacing: 0.3px;
  font-family: 'JetBrains Mono', monospace;
  border: 1px solid transparent;
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
  position: relative;
}

/* Online */
.badge-online {
  background: rgba(16, 185, 129, 0.1);
  color: var(--color-success);
  border-color: rgba(16, 185, 129, 0.2);
}
.badge-online .status-dot {
  background: var(--color-success);
}

/* Offline */
.badge-offline {
  background: rgba(239, 68, 68, 0.1);
  color: var(--color-danger);
  border-color: rgba(239, 68, 68, 0.2);
}
.badge-offline .status-dot {
  background: var(--color-danger);
}

/* Loading */
.badge-loading {
  background: rgba(6, 182, 212, 0.1);
  color: var(--color-info);
  border-color: rgba(6, 182, 212, 0.2);
}
.badge-loading .status-dot {
  background: var(--color-info);
}

/* Warning */
.badge-warning {
  background: rgba(245, 158, 11, 0.1);
  color: var(--color-warning);
  border-color: rgba(245, 158, 11, 0.2);
}
.badge-warning .status-dot {
  background: var(--color-warning);
}

/* Pulse animation */
.pulse .status-dot {
  animation: pulse-dot 2s ease-in-out infinite;
}

.pulse .status-dot::after {
  content: '';
  position: absolute;
  inset: -3px;
  border-radius: 50%;
  border: 1px solid currentColor;
  animation: pulse-ring 2s ease-in-out infinite;
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.8); }
}

@keyframes pulse-ring {
  0% { opacity: 1; transform: scale(1); }
  100% { opacity: 0; transform: scale(2.5); }
}
</style>
