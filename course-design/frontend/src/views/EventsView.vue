<!--
  Copyright (c) 2025 YOLO Course Design Contributors
  SPDX-License-Identifier: MIT
-->

<template>
  <div class="events-container">
    <el-card shadow="never" class="filter-card">
      <div class="filter-row">
        <div class="filter-left">
          <el-select
            v-model="filterType"
            placeholder="全部类型"
            clearable
            class="type-select"
            @change="loadEvents"
          >
            <el-option v-for="t in eventTypes" :key="t" :label="eventTypeLabel(t)" :value="t" />
          </el-select>
          <el-button type="primary" :icon="Search" @click="loadEvents">查询</el-button>
          <el-button :icon="RefreshLeft" @click="resetFilters">重置</el-button>
        </div>
        <div class="filter-right">
          <span class="total-text">
            共
            <strong>{{ total }}</strong>
            条事件
          </span>
          <el-button type="danger" plain size="small" :icon="Delete" @click="confirmClearAll">
            清空全部
          </el-button>
        </div>
      </div>
    </el-card>

    <el-row :gutter="12" class="stats-row">
      <el-col v-for="(count, type) in statsByType" :key="type" :xs="12" :sm="8" :md="6" :lg="4">
        <transition name="card-pop" mode="out-in">
          <el-card shadow="never" :body-style="{ padding: '16px' }" class="stat-card">
            <div class="stat-item">
              <div class="stat-info">
                <span class="stat-label">{{ eventTypeLabel(type) }}</span>
                <span class="stat-value">{{ count }}</span>
              </div>
              <el-icon :size="28" class="stat-icon" :style="{ color: statColor(type) }">
                <component :is="statIcon(type)" />
              </el-icon>
            </div>
          </el-card>
        </transition>
      </el-col>
    </el-row>

    <el-card shadow="never">
      <div v-if="skeletonLoading" class="table-skeleton">
        <div v-for="i in 5" :key="i" class="skeleton-row">
          <div class="skeleton skeleton-cell" style="width: 80px" />
          <div class="skeleton skeleton-cell" style="width: 50px" />
          <div class="skeleton skeleton-cell" style="flex: 1" />
          <div class="skeleton skeleton-cell" style="width: 70px" />
          <div class="skeleton skeleton-cell" style="width: 120px" />
        </div>
      </div>
      <el-table
        v-else
        v-loading="loading"
        :data="events"
        stripe
        max-height="500"
        :empty-text="tableEmptyText"
        row-class-name="event-row"
      >
        <el-table-column prop="event_type" label="类型" width="120">
          <template #default="{ row }">
            <el-tag :type="eventTypeColor(row.event_type)" size="small" effect="light">
              {{ eventTypeLabel(row.event_type) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="快照" width="80" align="center" class-name="snap-col">
          <template #default="{ row }">
            <el-image
              v-if="row.snapshot_path"
              :src="snapshotUrl(row.id)"
              fit="cover"
              class="snapshot-thumb"
              :preview-src-list="[snapshotUrl(row.id)]"
              preview-teleported
              hide-on-click-modal
            >
              <template #error>
                <span class="no-snap">-</span>
              </template>
            </el-image>
            <span v-else class="no-snap">-</span>
          </template>
        </el-table-column>
        <el-table-column prop="description" label="描述" min-width="200" show-overflow-tooltip />
        <el-table-column prop="track_id" label="目标ID" width="90" class-name="hide-mobile" />
        <el-table-column prop="zone_name" label="区域" width="100" class-name="hide-tablet" />
        <el-table-column prop="confidence" label="置信度" width="90" class-name="hide-mobile">
          <template #default="{ row }">
            <el-progress
              v-if="row.confidence"
              :percentage="Number((row.confidence * 100).toFixed(1))"
              :stroke-width="6"
              :color="confidenceColor(row.confidence)"
            />
            <span v-else class="no-snap">-</span>
          </template>
        </el-table-column>
        <el-table-column prop="timestamp_s" label="时间" width="180">
          <template #default="{ row }">
            {{ formatDateTime(row.timestamp_s) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="100" fixed="right" class-name="hide-mobile">
          <template #default="{ row }">
            <el-button v-if="row.source" type="primary" size="small" text @click="redetect(row)">
              重新检测
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <div v-if="total > pageSize" class="pagination-wrapper">
        <el-pagination
          v-model:current-page="page"
          :page-size="pageSize"
          :total="total"
          layout="prev, pager, next"
          background
          @current-change="loadEvents"
        />
      </div>
    </el-card>

    <el-card v-if="clips.length > 0" shadow="never" class="clips-card">
      <template #header>
        <div class="clips-header">
          <span class="clips-title">事件视频回放</span>
          <span class="clips-count">{{ clips.length }} 个片段</span>
        </div>
      </template>
      <div class="clips-grid">
        <div
          v-for="clip in clips"
          :key="clip.clip_id"
          class="clip-item"
          role="button"
          tabindex="0"
          :aria-label="`播放 ${eventTypeLabel(clip.event_type)} 视频`"
          @click="playClip(clip)"
          @keydown.enter="playClip(clip)"
        >
          <div class="clip-thumb">
            <el-icon :size="32" class="clip-play-icon"><VideoPlay /></el-icon>
          </div>
          <div class="clip-info">
            <el-tag :type="eventTypeColor(clip.event_type)" size="small">
              {{ eventTypeLabel(clip.event_type) }}
            </el-tag>
            <span class="clip-dur">{{ clip.duration_s?.toFixed(1) }}s</span>
            <span class="clip-size">{{ formatFileSize(clip.file_size_bytes) }}</span>
          </div>
        </div>
      </div>
    </el-card>

    <el-dialog
      v-model="showPlayer"
      title="视频回放"
      width="720px"
      :close-on-click-modal="false"
      destroy-on-close
      class="player-dialog"
    >
      <video
        v-if="playingClip"
        :key="playingClip.clip_id"
        :src="archivesAPI.clipUrl(playingClip.clip_id)"
        controls
        autoplay
        class="clip-player"
      />
    </el-dialog>
  </div>
</template>

<script setup>
/**
 * EventsView.vue - Detection event history and statistics page.
 *
 * Displays a paginated, filterable list of detection events (intrusion,
 * fight, fall, crowd, running) with:
 * - Real-time event statistics (counts by type, severity distribution)
 * - Event type filtering and time-range selection
 * - Snapshot preview for each event
 * - Bulk delete and individual event management
 * - Video clip playback for archived events
 */
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessageBox } from 'element-plus'
import { VideoPlay, Search, RefreshLeft, Delete } from '@element-plus/icons-vue'
import { eventsAPI, archivesAPI } from '../api/client'
import { logger } from '../utils/logger'
import {
  eventTypeColor,
  eventTypeLabel,
  statColor,
  statIcon,
  confidenceColor,
  formatDateTime,
} from '../utils/helpers'

const events = ref([])
const total = ref(0)
const page = ref(1)
const pageSize = 50
const filterType = ref('')
const eventTypes = ref([])
const stats = ref({ by_type: {} })
const loading = ref(false)
const loadError = ref(false)
const skeletonLoading = ref(true)
const router = useRouter()

const clips = ref([])
const showPlayer = ref(false)
const playingClip = ref(null)

const statsByType = computed(() => stats.value.by_type || {})

const tableEmptyText = computed(() => (loadError.value ? '加载失败，请检查后端服务' : '暂无事件'))

const snapshotUrl = (id) => eventsAPI.snapshotUrl(id)

const loadEvents = async () => {
  loading.value = true
  loadError.value = false
  try {
    const params = { limit: pageSize, offset: (page.value - 1) * pageSize }
    if (filterType.value) params.event_type = filterType.value
    const res = await eventsAPI.list(params)
    events.value = res.items || res.events || []
    total.value = res.total || 0
  } catch (e) {
    logger.error('Failed to load events:', e)
    loadError.value = true
  } finally {
    loading.value = false
    skeletonLoading.value = false
  }
}

const loadStats = async () => {
  try {
    stats.value = await eventsAPI.stats()
  } catch (e) {
    logger.error('Failed to load stats:', e)
  }
}

const loadEventTypes = async () => {
  try {
    const res = await eventsAPI.types()
    eventTypes.value = res.event_types
  } catch (e) {
    logger.error('Failed to load event types:', e)
  }
}

const resetFilters = () => {
  filterType.value = ''
  page.value = 1
  loadEvents()
}

const confirmClearAll = () => {
  ElMessageBox.confirm('确定要清空所有事件记录吗？此操作不可恢复。', '确认清空', {
    confirmButtonText: '清空',
    cancelButtonText: '取消',
    type: 'warning',
  })
    .then(async () => {
      await eventsAPI.deleteAll()
      loadEvents()
      loadStats()
    })
    .catch(() => {})
}

const redetect = (row) => {
  router.push({
    path: '/monitor',
    query: { source: row.source, autoStart: '1' },
  })
}

const loadClips = async () => {
  try {
    const res = await archivesAPI.list({ limit: 12 })
    clips.value = res.clips || []
  } catch { /* archives may not be available when pipeline is off */ }
}

const playClip = (clip) => {
  playingClip.value = clip
  showPlayer.value = true
}

const formatFileSize = (bytes) => {
  if (!bytes) return '0 B'
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

onMounted(() => {
  loadEvents()
  loadStats()
  loadEventTypes()
  loadClips()
})
</script>

<style scoped>
.events-container {
  display: flex;
  flex-direction: column;
  gap: 16px;
  max-width: 1400px;
  margin: 0 auto;
}

.filter-card {
  flex-shrink: 0;
}

.filter-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.filter-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.type-select {
  width: 160px;
}

.filter-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.total-text {
  color: var(--text-secondary);
  font-size: 14px;
}

.total-text strong {
  color: var(--primary-color);
  font-size: 16px;
  font-variant-numeric: tabular-nums;
}

.stats-row {
  margin-bottom: 0;
}

.stat-card {
  margin-bottom: 12px;
  transition: transform var(--transition-fast), box-shadow var(--transition-fast);
}

.stat-card:hover {
  transform: translateY(-2px);
}

.stat-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.stat-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.stat-label {
  font-size: 13px;
  color: var(--text-secondary);
}

.stat-value {
  font-size: 26px;
  font-weight: 700;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
}

.stat-icon {
  opacity: 0.8;
}

.table-skeleton {
  padding: 12px 0;
}

.skeleton-row {
  display: flex;
  gap: 16px;
  padding: 12px 16px;
  align-items: center;
}

.skeleton-cell {
  height: 20px;
  border-radius: 4px;
}

.pagination-wrapper {
  display: flex;
  justify-content: center;
  margin-top: 16px;
}

.snapshot-thumb {
  width: 48px;
  height: 36px;
  border-radius: 4px;
  cursor: pointer;
  object-fit: cover;
  transition: transform var(--transition-fast);
}

.snapshot-thumb:hover {
  transform: scale(1.1);
}

.no-snap {
  color: #ccc;
  font-size: 12px;
}

.card-pop-enter-active {
  transition: all 0.3s ease;
}

.card-pop-leave-active {
  transition: all 0.2s ease;
}

.card-pop-enter-from {
  opacity: 0;
  transform: translateY(8px) scale(0.96);
}

.card-pop-leave-to {
  opacity: 0;
  transform: scale(0.96);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

:deep(.event-row) {
  transition: background 0.2s;
}

.clips-card {
  flex-shrink: 0;
}

.clips-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.clips-title {
  font-weight: 600;
  font-size: 14px;
}

.clips-count {
  color: var(--text-secondary);
  font-size: 13px;
}

.clips-grid {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.clip-item {
  width: 160px;
  cursor: pointer;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--border-color);
  transition: transform var(--transition-fast), box-shadow var(--transition-fast);
}

.clip-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.clip-item:focus-visible {
  outline: 2px solid var(--primary-color);
  outline-offset: 2px;
}

.clip-thumb {
  width: 100%;
  height: 90px;
  background: #1a1a2e;
  display: flex;
  align-items: center;
  justify-content: center;
}

.clip-play-icon {
  color: rgba(255, 255, 255, 0.7);
}

.clip-info {
  padding: 8px;
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  align-items: center;
}

.clip-dur,
.clip-size {
  font-size: 11px;
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
}

.clip-player {
  width: 100%;
  max-height: 480px;
  border-radius: 6px;
  background: #000;
}

@media (max-width: 1024px) {
  :deep(.hide-tablet) {
    display: none;
  }
}

@media (max-width: 768px) {
  .filter-row {
    flex-direction: column;
    align-items: flex-start;
  }

  .type-select {
    width: 100%;
  }

  .filter-left {
    width: 100%;
  }

  .filter-left .el-button {
    flex: 1;
  }

  .filter-right {
    width: 100%;
    justify-content: space-between;
  }

  :deep(.hide-mobile) {
    display: none;
  }

  .stat-value {
    font-size: 22px;
  }

  .clip-item {
    width: 140px;
  }

  .clip-thumb {
    height: 75px;
  }
}

@media (max-width: 480px) {
  .events-container {
    gap: 10px;
  }

  .stat-value {
    font-size: 20px;
  }

  .clip-item {
    width: 100%;
  }

  .clip-thumb {
    height: 120px;
  }
}
</style>
