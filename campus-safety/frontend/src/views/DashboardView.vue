<!--
  ┌──────────────────────────────────────────┐
  │ 【前端UI】DashboardView.vue — 仪表盘首页    │
  │ 路由：/                                    │
  │ 职责：系统概览 —                            │
  │   ① 统计卡片（总事件/活跃报警/严重报警）    │
  │   ② 事件分布图表                           │
  │   ③ 最近报警列表                           │
  │   ④ 系统运行时长                           │
  │ 使用组件：StatCard                          │
  └──────────────────────────────────────────┘
  Copyright (c) 2025 YOLO Course Design Contributors
  SPDX-License-Identifier: Apache-2.0
-->

<template>
  <div class="dashboard">
    <!-- Hero Stats Row -->
    <div class="stats-grid">
      <StatCard
        label="检测帧率"
        :value="fps"
        unit="FPS"
        :icon="TrendCharts"
        :color="fpsColor"
        :loading="loading"
        :animated="detectionRunning"
      >
        <template #footer>
          <span class="stat-meta">
            <span class="meta-dot" :class="fpsClass" />
            已检测 {{ formatNumber(frameCount) }} 帧
          </span>
        </template>
      </StatCard>
      <StatCard
        label="今日事件"
        :value="totalEvents"
        :icon="List"
        color="var(--color-warning)"
        :loading="loading"
      >
        <template #footer>
          <span class="stat-meta" v-if="eventsByType.length > 0">
            {{ eventsByType.slice(0, 3).map(t => `${t.label} ${t.count}`).join(' · ') }}
          </span>
        </template>
      </StatCard>
      <StatCard
        label="活跃告警"
        :value="activeAlarms"
        :icon="BellFilled"
        :color="activeAlarms > 0 ? 'var(--color-danger)' : 'var(--color-success)'"
        :loading="loading"
        :animated="activeAlarms > 0"
      >
        <template #footer>
          <span class="stat-meta">
            <span v-if="criticalAlarms > 0" class="meta-badge meta-danger">{{ criticalAlarms }} 紧急</span>
            <span v-else class="meta-badge meta-success">正常</span>
          </span>
        </template>
      </StatCard>
      <StatCard
        label="GPU 状态"
        :value="gpuName"
        unit=""
        :icon="Cpu"
        color="var(--color-info)"
        :loading="loading"
      >
        <template #footer>
          <span class="stat-meta">{{ gpuMemory }}</span>
        </template>
      </StatCard>
      <StatCard
        label="运行时长"
        :value="systemUptime"
        :icon="Timer"
        color="var(--text-regular)"
        :loading="loading"
      >
        <template #footer>
          <StatusBadge :status="detectionRunning ? 'online' : 'offline'" :label="detectionRunning ? '检测运行中' : '已停止'" />
        </template>
      </StatCard>
    </div>

    <!-- Main Grid: Video + Alerts -->
    <div class="dashboard-grid">
      <div class="grid-left">
        <el-card class="section-card video-card">
          <template #header>
            <div class="card-header">
              <div class="header-title-group">
                <span class="card-title">实时画面</span>
                <span class="card-subtitle">{{ detectionRunning ? '正在实时检测' : '检测未启动' }}</span>
              </div>
              <div class="card-actions">
                <StatusBadge
                  v-if="detectionRunning"
                  status="online"
                  label="LIVE"
                  :animated="true"
                />
                <el-button size="small" type="primary" @click="goToMonitor">
                  {{ detectionRunning ? '详细监控' : '开始检测' }}
                </el-button>
              </div>
            </div>
          </template>
          <StreamPlayer
            :stream-url="streamUrl"
            :is-running="detectionRunning"
            placeholder-text="点击上方按钮开始检测"
            :style="{ height: 'auto' }"
          />
        </el-card>
      </div>
      <div class="grid-right">
        <el-card class="section-card alerts-card">
          <template #header>
            <div class="card-header">
              <div class="header-title-group">
                <span class="card-title">实时告警</span>
                <span class="card-subtitle">最近 5 条</span>
              </div>
              <el-button v-if="activeAlarms > 0" size="small" text @click="goToAlarms">
                查看全部
              </el-button>
            </div>
          </template>
          <div v-if="recentAlarms.length === 0" class="empty-state">
            <div class="empty-icon">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--text-disabled)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                <polyline points="22 4 12 14.01 9 11.01"/>
              </svg>
            </div>
            <p class="empty-title">暂无活跃告警</p>
            <p class="empty-desc">系统运行正常</p>
          </div>
          <div v-else class="alert-list">
            <div
              v-for="alarm in recentAlarms"
              :key="alarm.id"
              class="alert-item"
              :class="{ 'alert-critical': alarm.level === 3 }"
              @click="goToAlarms"
            >
              <div class="alert-left">
                <span class="alert-dot" :class="`dot-${alarm.level === 3 ? 'danger' : alarm.level === 2 ? 'warning' : 'info'}`" />
                <div class="alert-info">
                  <span class="alert-type">{{ eventTypeLabel(alarm.event_type) }}</span>
                  <span class="alert-desc">{{ alarm.description || alarm.event_type }}</span>
                </div>
              </div>
              <span class="alert-time">{{ formatTimeAgo(alarm.last_event_time) }}</span>
            </div>
          </div>
        </el-card>
      </div>
    </div>

    <!-- Recent Events -->
    <el-card class="section-card events-card">
      <template #header>
        <div class="card-header">
          <div class="header-title-group">
            <span class="card-title">最近事件</span>
            <span class="card-subtitle">实时检测记录</span>
          </div>
          <el-button size="small" text @click="goToEvents">查看全部</el-button>
        </div>
      </template>
      <SkeletonLoader v-if="loading" type="table" :rows="3" />
      <el-table v-else :data="recentEvents" stripe style="width: 100%" @row-click="goToEvents" class="dashboard-table">
        <el-table-column prop="event_type" label="类型" width="110">
          <template #default="{ row }">
            <el-tag :type="eventTypeColor(row.event_type)" size="small" effect="dark" class="event-tag">
              {{ eventTypeLabel(row.event_type) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="description" label="描述" min-width="160">
          <template #default="{ row }">
            <span class="event-desc">{{ row.description || eventTypeLabel(row.event_type) }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="confidence" label="置信度" width="90" align="center">
          <template #default="{ row }">
            <ConfidenceGauge :value="row.confidence || 0" :size="32" />
          </template>
        </el-table-column>
        <el-table-column label="时间" width="170" align="right">
          <template #default="{ row }">
            <span class="event-time">{{ formatDateTime(row.timestamp) }}</span>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { TrendCharts, List, BellFilled, Cpu, Timer } from '@element-plus/icons-vue'
import { useWebSocket } from '../composables/useWebSocket'
import { eventsAPI, alarmsAPI, detectionAPI } from '../api/client'
import { eventTypeColor, eventTypeLabel, formatDateTime } from '../utils/helpers'
import StatCard from '../components/StatCard.vue'
import StatusBadge from '../components/StatusBadge.vue'
import StreamPlayer from '../components/StreamPlayer.vue'
import ConfidenceGauge from '../components/ConfidenceGauge.vue'
import SkeletonLoader from '../components/SkeletonLoader.vue'

const router = useRouter()

const loading = ref(true)
const fps = ref(0)
const frameCount = ref(0)
const detectionRunning = ref(false)
const streamUrl = ref('')
const gpuName = ref('—')
const gpuMemory = ref('')
const totalEvents = ref(0)
const eventsByType = ref([])
const activeAlarms = ref(0)
const criticalAlarms = ref(0)
const recentAlarms = ref([])
const recentEvents = ref([])
const systemUptime = ref('')
const uptimeSeconds = ref(0)
let uptimeTimer = null

let statusPoll = null
let alarmPoll = null

const fpsColor = computed(() => {
  if (fps.value >= 25) return 'var(--color-success)'
  if (fps.value >= 15) return 'var(--color-warning)'
  return 'var(--color-danger)'
})

const fpsClass = computed(() => {
  if (fps.value >= 25) return 'dot-success'
  if (fps.value >= 15) return 'dot-warning'
  return 'dot-danger'
})

const formatNumber = (n) => {
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M'
  if (n >= 1000) return (n / 1000).toFixed(1) + 'K'
  return String(n)
}

const formatTimeAgo = (ts) => {
  if (!ts) return ''
  const now = Date.now() / 1000
  const diff = now - ts
  if (diff < 60) return '刚刚'
  if (diff < 3600) return `${Math.floor(diff / 60)}分钟前`
  if (diff < 86400) return `${Math.floor(diff / 3600)}小时前`
  return formatDateTime(ts)
}

const onWsMessage = (msg) => {
  if (msg.type === 'status' && msg.data) {
    fps.value = msg.data.fps || 0
    frameCount.value = msg.data.frame_count || 0
    detectionRunning.value = msg.data.running || false
  }
  if (msg.type === 'gpu' && msg.data) {
    gpuName.value = msg.data.gpu_name || msg.data.name || gpuName.value
    const used = msg.data.gpu_used_memory_mb !== undefined ? msg.data.gpu_used_memory_mb : msg.data.memory_used_mb
    const total = msg.data.gpu_total_memory_mb !== undefined ? msg.data.gpu_total_memory_mb : msg.data.memory_total_mb
    if (used !== undefined && total !== undefined) {
      gpuMemory.value = `${used}MB / ${total}MB`
    }
  }
}

const { connect, disconnect } = useWebSocket()

const loadDashboardData = async () => {
  try {
    const [status, eStats, aStats, eList] = await Promise.all([
      detectionAPI.status(),
      eventsAPI.stats(),
      alarmsAPI.stats(),
      eventsAPI.list({ limit: 5 }),
    ])

    if (status.running) {
      detectionRunning.value = true
      streamUrl.value = `/api/detection/stream.mjpg?_=${Date.now()}`
    }
    fps.value = status.fps || 0
    frameCount.value = status.frame_count || 0

    totalEvents.value = eStats.total_events || 0
    if (eStats.by_type) {
      eventsByType.value = Object.entries(eStats.by_type).map(([key, count]) => ({
        type: key,
        label: eventTypeLabel(key),
        count,
      }))
    }

    activeAlarms.value = aStats.active_count || 0
    criticalAlarms.value = aStats.critical_count || 0

    recentEvents.value = (eList.items || []).slice(0, 5)
    recentAlarms.value = (aStats.recent || []).slice(0, 5)
  } catch (err) {
    // silently fail for dashboard initial load
  } finally {
    loading.value = false
  }
}

const loadAlarms = async () => {
  try {
    const aStats = await alarmsAPI.stats()
    activeAlarms.value = aStats.active_count || 0
    criticalAlarms.value = aStats.critical_count || 0
    recentAlarms.value = (aStats.recent || []).slice(0, 5)
  } catch (err) {
    // ignore
  }
}

const goToMonitor = () => router.push('/monitor')
const goToEvents = () => router.push('/events')
const goToAlarms = () => router.push('/alarms')

onMounted(async () => {
  await loadDashboardData()

  // Connect WebSocket for live updates
  const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/detection/stream`
  connect(wsUrl, onWsMessage)

  // Uptime counter
  uptimeSeconds.value = 0
  uptimeTimer = setInterval(() => {
    uptimeSeconds.value++
    const h = Math.floor(uptimeSeconds.value / 3600)
    const m = Math.floor((uptimeSeconds.value % 3600) / 60)
    const s = uptimeSeconds.value % 60
    systemUptime.value = h > 0
      ? `${h}h ${m}m ${s}s`
      : m > 0 ? `${m}m ${s}s` : `${s}s`
  }, 1000)

  statusPoll = setInterval(loadDashboardData, 5000)
  alarmPoll = setInterval(loadAlarms, 10000)
})

onUnmounted(() => {
  disconnect()
  if (statusPoll) clearInterval(statusPoll)
  if (alarmPoll) clearInterval(alarmPoll)
  if (uptimeTimer) clearInterval(uptimeTimer)
})
</script>

<style scoped>
.dashboard {
  display: flex;
  flex-direction: column;
  gap: 20px;
  max-width: 1440px;
  margin: 0 auto;
  position: relative;
  z-index: 1;
}

/* Stats Grid */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 16px;
}

.stat-meta {
  font-size: 12px;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  gap: 6px;
}

.meta-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}

.dot-success { background: var(--color-success); }
.dot-warning { background: var(--color-warning); }
.dot-danger { background: var(--color-danger); }

.meta-badge {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
}

.meta-danger {
  background: rgba(239, 68, 68, 0.1);
  color: var(--color-danger);
}

.meta-success {
  background: rgba(16, 185, 129, 0.1);
  color: var(--color-success);
}

/* Dashboard Grid */
.dashboard-grid {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 16px;
}

.section-card :deep(.el-card__body) {
  padding: 16px;
}

.section-card :deep(.el-card__header) {
  padding: 14px 16px;
  border-bottom: 1px solid var(--border-subtle);
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.header-title-group {
  display: flex;
  align-items: center;
  gap: 10px;
}

.card-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
}

.card-subtitle {
  font-size: 12px;
  color: var(--text-secondary);
}

.card-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

/* Video card */
.video-card :deep(.el-card__body) {
  padding: 0;
}

.video-card :deep(.stream-player) {
  border-radius: 0 0 var(--radius-md) var(--radius-md);
  border: none;
}

/* Alert list */
.alerts-card :deep(.el-card__body) {
  padding: 0;
  min-height: 280px;
}

.alert-list {
  max-height: 320px;
  overflow-y: auto;
}

.alert-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-subtle);
  cursor: pointer;
  transition: all var(--transition-fast);
}

.alert-item:last-child {
  border-bottom: none;
}

.alert-item:hover {
  background: var(--color-primary-muted);
}

.alert-critical {
  background: rgba(239, 68, 68, 0.04);
  border-left: 3px solid var(--color-danger);
}

.alert-left {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 1;
  min-width: 0;
}

.alert-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.dot-danger { background: var(--color-danger); box-shadow: 0 0 6px rgba(239, 68, 68, 0.4); }
.dot-warning { background: var(--color-warning); }
.dot-info { background: var(--color-info); }

.alert-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.alert-type {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
}

.alert-desc {
  font-size: 12px;
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.alert-time {
  font-size: 11px;
  color: var(--text-secondary);
  flex-shrink: 0;
  font-family: 'JetBrains Mono', monospace;
}

/* Empty state */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 48px 20px;
  color: var(--text-secondary);
}

.empty-icon {
  opacity: 0.4;
}

.empty-title {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-regular);
  margin: 0;
}

.empty-desc {
  font-size: 12px;
  color: var(--text-secondary);
  margin: 0;
}

/* Events card */
.events-card :deep(.el-card__body) {
  padding: 0;
}

.events-card :deep(.el-table__header-wrapper) {
  border-radius: 0;
}

.event-tag {
  font-weight: 500;
  letter-spacing: 0.3px;
}

.event-desc {
  font-size: 13px;
  color: var(--text-primary);
}

.event-time {
  font-size: 12px;
  color: var(--text-secondary);
  font-family: 'JetBrains Mono', monospace;
}

/* Responsive */
@media (max-width: 1200px) {
  .stats-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (max-width: 768px) {
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  .dashboard-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 480px) {
  .stats-grid {
    grid-template-columns: 1fr;
  }
}
</style>
