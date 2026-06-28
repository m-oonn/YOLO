<template>
  <div
    class="stat-card"
    :class="{ 'stat-loading': loading, 'stat-animated': animated }"
    :style="cardStyle"
  >
    <div class="stat-glow" :style="glowStyle" />
    <div class="stat-body">
      <div class="stat-left">
        <span class="stat-label">{{ label }}</span>
        <div class="stat-value-row">
          <span class="stat-value" :style="{ color: color }">
            {{ displayValue }}
          </span>
          <span v-if="unit" class="stat-unit">{{ unit }}</span>
        </div>
        <span
          v-if="trend !== undefined"
          class="stat-trend"
          :class="trend >= 0 ? 'trend-up' : 'trend-down'"
        >
          {{ trend >= 0 ? '↑' : '↓' }} {{ Math.abs(trend) }}%
        </span>
      </div>
      <div v-if="icon" class="stat-icon-wrap" :style="{ background: iconBg, color: color }">
        <el-icon :size="22">
          <component :is="icon" />
        </el-icon>
      </div>
    </div>
    <div v-if="$slots.footer" class="stat-footer">
      <slot name="footer" />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  label: { type: String, required: true },
  value: { type: [Number, String], default: 0 },
  unit: { type: String, default: '' },
  icon: { type: [Object, Function], default: null },
  color: { type: String, default: 'var(--color-primary)' },
  trend: { type: Number, default: undefined },
  loading: { type: Boolean, default: false },
  animated: { type: Boolean, default: false },
})

const cardStyle = computed(() => ({
  '--stat-color': props.color,
}))

const glowStyle = computed(() => ({
  background: `radial-gradient(circle at 80% 20%, ${props.color}15, transparent 60%)`,
}))

const iconBg = computed(() => {
  return `${props.color}18`
})

const displayValue = computed(() => {
  if (props.loading) return '—'
  return props.value
})
</script>

<style scoped>
.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 20px 22px;
  transition: all var(--transition-fast);
  position: relative;
  overflow: hidden;
}

.stat-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--stat-color, var(--color-primary));
  opacity: 0.5;
}

.stat-glow {
  position: absolute;
  top: 0;
  right: 0;
  width: 120px;
  height: 120px;
  pointer-events: none;
  opacity: 0;
  transition: opacity var(--transition-slow);
}

.stat-card:hover {
  border-color: var(--border-default);
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
}

.stat-card:hover .stat-glow {
  opacity: 1;
}

.stat-body {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  position: relative;
  z-index: 1;
}

.stat-left {
  flex: 1;
  min-width: 0;
}

.stat-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 1px;
  display: block;
  margin-bottom: 8px;
}

.stat-value-row {
  display: flex;
  align-items: baseline;
  gap: 6px;
}

.stat-value {
  font-size: 30px;
  font-weight: 700;
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
  font-family: 'JetBrains Mono', monospace;
}

.stat-unit {
  font-size: 13px;
  color: var(--text-secondary);
  font-weight: 400;
}

.stat-trend {
  font-size: 11px;
  font-weight: 600;
  margin-top: 6px;
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  padding: 2px 8px;
  border-radius: 4px;
}

.trend-up {
  color: var(--color-success);
  background: rgba(16, 185, 129, 0.1);
}

.trend-down {
  color: var(--color-danger);
  background: rgba(239, 68, 68, 0.1);
}

.stat-icon-wrap {
  width: 44px;
  height: 44px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: all var(--transition-fast);
}

.stat-card:hover .stat-icon-wrap {
  transform: scale(1.05);
}

.stat-footer {
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid var(--border-subtle);
  position: relative;
  z-index: 1;
}

.stat-loading {
  opacity: 0.5;
  pointer-events: none;
}

.stat-animated .stat-value {
  animation: value-pulse 2s ease-in-out infinite;
}

@keyframes value-pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
}

@media (max-width: 768px) {
  .stat-card {
    padding: 16px 18px;
  }
  .stat-value {
    font-size: 24px;
  }
}
</style>
