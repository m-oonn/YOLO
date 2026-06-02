<template>
  <div class="confidence-gauge" :style="{ width: `${size}px`, height: `${size}px` }">
    <svg :width="size" :height="size" viewBox="0 0 36 36">
      <path
        d="M18 2.0845
          a 15.9155 15.9155 0 0 1 0 31.831
          a 15.9155 15.9155 0 0 1 0 -31.831"
        fill="none"
        stroke="var(--bg-elevated)"
        stroke-width="4"
      />
      <path
        :d="arcPath"
        fill="none"
        :stroke="gaugeColor"
        stroke-width="4"
        stroke-linecap="round"
        :stroke-dasharray="circumference"
        :stroke-dashoffset="dashOffset"
      />
    </svg>
    <span class="gauge-text" :style="{ color: gaugeColor }">{{ displayValue }}</span>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  value: { type: Number, required: true },
  size: { type: Number, default: 40 },
})

const circumference = 2 * Math.PI * 15.9155
const displayValue = computed(() => Math.round(props.value * 100))

const gaugeColor = computed(() => {
  if (props.value >= 0.8) return 'var(--color-success)'
  if (props.value >= 0.5) return 'var(--color-warning)'
  return 'var(--color-danger)'
})

const arcPath = computed(() => {
  return 'M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831'
})

const dashOffset = computed(() => {
  const pct = Math.min(displayValue.value, 100) / 100
  return circumference * (1 - pct)
})
</script>

<style scoped>
.confidence-gauge {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.confidence-gauge svg {
  width: 100%;
  height: 100%;
}

.gauge-text {
  position: absolute;
  font-size: 10px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  line-height: 1;
}
</style>
