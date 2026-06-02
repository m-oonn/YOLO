<template>
  <div class="stream-player" :class="{ 'is-active': isRunning }">
    <!-- Corner accents -->
    <div class="corner-accent corner-tl" />
    <div class="corner-accent corner-tr" />
    <div class="corner-accent corner-bl" />
    <div class="corner-accent corner-br" />

    <!-- Live indicator -->
    <div v-if="isRunning" class="stream-live">
      <StatusBadge status="online" label="LIVE" :animated="true" />
    </div>

    <!-- Stream error (only after we've seen at least one frame) -->
    <div v-if="hasError && hasLoadedOnce" class="stream-overlay">
      <div class="overlay-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--color-danger)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
          <line x1="12" y1="9" x2="12" y2="13"/>
          <line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
      </div>
      <p class="overlay-text">画面连接中断</p>
      <el-button size="small" type="primary" @click="$emit('retry')">重试</el-button>
    </div>

    <!-- Loading / connecting (centered in video area) -->
    <div v-if="isRunning && !hasLoadedOnce && !hasError" class="stream-loading">
      <div class="loading-ring">
        <div class="loading-ring-inner" />
      </div>
      <p class="loading-text">正在连接视频流...</p>
      <p class="loading-sub">首次启动需加载模型，约需 5-10 秒</p>
    </div>

    <!-- Loading with retry (error during initial connection) -->
    <div v-if="isRunning && !hasLoadedOnce && hasError" class="stream-loading">
      <div class="loading-ring">
        <div class="loading-ring-inner" />
      </div>
      <p class="loading-text">连接超时，正在重试...</p>
      <el-button size="small" type="primary" @click="$emit('retry')">手动重试</el-button>
    </div>

    <!-- Stopped placeholder -->
    <div v-if="!isRunning" class="stream-overlay">
      <div class="overlay-icon">
        <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="var(--text-disabled)" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="2" y="5" width="20" height="14" rx="3"/>
          <circle cx="12" cy="12" r="3"/>
          <path d="M12 2v3M12 19v3M2 12h3M19 12h3" opacity="0.4"/>
        </svg>
      </div>
      <p class="overlay-text">{{ placeholderText || '点击开始以查看视频流' }}</p>
    </div>

    <!-- MJPEG stream -->
    <img
      v-show="isRunning && !hasError"
      ref="streamImg"
      :src="streamUrl"
      class="stream-img"
      :class="{ 'stream-ready': !isLoading }"
      alt="实时监控画面"
      @load="onStreamLoad"
      @error="onStreamError"
    />
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import StatusBadge from './StatusBadge.vue'

const props = defineProps({
  streamUrl: { type: String, default: '' },
  isRunning: { type: Boolean, default: false },
  placeholderText: { type: String, default: '' },
})

const emit = defineEmits(['error', 'retry'])

const hasError = ref(false)
const isLoading = ref(false)
const hasLoadedOnce = ref(false)
const streamImg = ref(null)

// Reset state when stream URL changes (new stream attempt)
watch(() => props.streamUrl, (newUrl) => {
  if (newUrl) {
    hasError.value = false
    isLoading.value = true
    hasLoadedOnce.value = false
  }
})

// Reset when detection stops
watch(() => props.isRunning, (running) => {
  if (!running) {
    hasLoadedOnce.value = false
  }
})

const onStreamLoad = () => {
  isLoading.value = false
  hasLoadedOnce.value = true
  hasError.value = false
}

const onStreamError = () => {
  hasError.value = true
  isLoading.value = false
  emit('error')
}
</script>

<style scoped>
.stream-player {
  position: relative;
  background: linear-gradient(135deg, #0a0e14 0%, #111827 100%);
  border-radius: var(--radius-md);
  overflow: hidden;
  aspect-ratio: 16 / 9;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--border-subtle);
  transition: all var(--transition-fast);
}

.stream-player.is-active {
  border-color: var(--border-default);
  box-shadow: 0 0 30px rgba(245, 158, 11, 0.06);
}

/* Corner accents */
.corner-accent {
  position: absolute;
  width: 20px;
  height: 20px;
  border-color: var(--color-primary);
  border-style: solid;
  opacity: 0.3;
  z-index: 5;
  pointer-events: none;
  transition: opacity var(--transition-fast);
}

.stream-player:hover .corner-accent {
  opacity: 0.6;
}

.corner-tl {
  top: 8px;
  left: 8px;
  border-width: 2px 0 0 2px;
  border-radius: 4px 0 0 0;
}

.corner-tr {
  top: 8px;
  right: 8px;
  border-width: 2px 2px 0 0;
  border-radius: 0 4px 0 0;
}

.corner-bl {
  bottom: 8px;
  left: 8px;
  border-width: 0 0 2px 2px;
  border-radius: 0 0 0 4px;
}

.corner-br {
  bottom: 8px;
  right: 8px;
  border-width: 0 2px 2px 0;
  border-radius: 0 0 4px 0;
}

.stream-live {
  position: absolute;
  top: 16px;
  left: 16px;
  z-index: 10;
  pointer-events: none;
}

.stream-img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  opacity: 0;
  transition: opacity 0.4s ease;
}

.stream-img.stream-ready {
  opacity: 1;
}

.stream-overlay {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  color: var(--text-secondary);
  padding: 20px;
  text-align: center;
  z-index: 3;
}

.overlay-icon {
  opacity: 0.5;
}

.overlay-text {
  font-size: 14px;
  margin: 0;
  color: var(--text-secondary);
}

.stream-loading {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 20px;
  z-index: 10;
  background: rgba(7, 10, 15, 0.85);
  backdrop-filter: blur(4px);
}

.loading-ring {
  width: 48px;
  height: 48px;
  position: relative;
}

.loading-ring-inner {
  width: 100%;
  height: 100%;
  border: 3px solid var(--border-default);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.loading-text {
  font-size: 14px;
  margin: 0;
  color: var(--color-primary);
  font-weight: 500;
}

.loading-sub {
  font-size: 12px;
  margin: 0;
  color: var(--text-disabled);
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
</style>
