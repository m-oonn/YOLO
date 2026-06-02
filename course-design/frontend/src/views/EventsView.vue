<!--
  Copyright (c) 2025 YOLO Course Design Contributors
  SPDX-License-Identifier: Apache-2.0
-->

<template>
  <div class="events-container">
    <PageHeader title="事件记录" subtitle="查看和管理所有检测到的事件">
      <template #actions>
        <ActionButton
          type="danger"
          plain
          size="small"
          :icon="Delete"
          :action="clearAllEvents"
          :confirmConfig="{ title: '确认清空', message: '确定要清空所有事件记录吗？此操作不可恢复！', type: 'warning' }"
          successMessage="事件已清空"
          errorMessage="清空失败"
          @success="loadEvents"
        >
          清空全部
        </ActionButton>
      </template>
    </PageHeader>

    <!-- Filter bar -->
    <SearchFilterBar
      :filters="eventTypeFilters"
      @search="onSearch"
      @reset="resetFilters"
    >
      <template #right>
        <span class="total-text">
          共 <strong>{{ total }}</strong> 条事件
        </span>
      </template>
    </SearchFilterBar>

    <!-- Stats cards row -->
    <div class="stats-row">
      <div v-for="(count, type) in statsByType" :key="type" class="stat-card-wrap">
        <StatCard
          :label="eventTypeLabel(type)"
          :value="count"
          :icon="statIcon(type)"
          :color="statColor(type)"
          size="small"
        />
      </div>
    </div>

    <!-- Events table -->
    <el-card shadow="never" class="table-card">
      <SkeletonLoader v-if="skeletonLoading" type="table" :rows="5" />
      <el-table
        v-else
        v-loading="loading"
        :data="events"
        stripe
        max-height="500"
        :empty-text="tableEmptyText"
        row-class-name="event-row"
        row-key="id"
        :expand-row-keys="expandedRow ? [expandedRow] : []"
        @row-click="onRowClick"
        highlight-current-row
        class="events-table"
      >
        <el-table-column type="expand">
          <template #default="{ row }">
            <div class="expand-detail">
              <div v-if="row.extra?.mllm_narrative" class="mllm-panel">
                <div class="mllm-header">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--color-primary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 6px;">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                  </svg>
                  <span class="mllm-title">AI 场景分析</span>
                  <el-tag v-if="row.extra?.mllm_risk_level" size="small" :type="row.extra.mllm_risk_level === '高' ? 'danger' : row.extra.mllm_risk_level === '中' ? 'warning' : 'info'">
                    风险: {{ row.extra.mllm_risk_level }}
                  </el-tag>
                </div>
                <p class="mllm-narrative">{{ row.extra.mllm_narrative }}</p>
                <p v-if="row.extra?.mllm_action" class="mllm-action">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px; vertical-align: -2px;">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                    <line x1="12" y1="9" x2="12" y2="13"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                  </svg>
                  建议: {{ row.extra.mllm_action }}
                </p>
              </div>
              <div v-else class="mllm-panel mllm-empty">
                <span class="mllm-title">AI 场景分析</span>
                <span class="mllm-pending">MLLM 推理中或未启用</span>
              </div>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="event_type" label="类型" width="120">
          <template #default="{ row }">
            <el-tag :type="eventTypeColor(row.event_type)" size="small" effect="dark" class="event-tag">
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
        <el-table-column prop="confidence" label="置信度" width="80" class-name="hide-mobile">
          <template #default="{ row }">
            <ConfidenceGauge v-if="row.confidence" :value="row.confidence" :size="36" />
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

    <!-- Video clips -->
    <el-card v-if="clips.length > 0" shadow="never" class="clips-card">
      <template #header>
        <div class="clips-header">
          <span class="clips-title">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 6px; vertical-align: -2px;">
              <path d="M23 7l-7 5 7 5V7z"/>
              <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
            </svg>
            事件视频回放
          </span>
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
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.5)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <polygon points="10 8 16 12 10 16 10 8"/>
            </svg>
          </div>
          <div class="clip-info">
            <el-tag :type="eventTypeColor(clip.event_type)" size="small" effect="dark">
              {{ eventTypeLabel(clip.event_type) }}
            </el-tag>
            <span class="clip-dur">{{ clip.duration_s?.toFixed(1) }}s</span>
            <span class="clip-size">{{ formatFileSize(clip.file_size_bytes) }}</span>
          </div>
        </div>
      </div>
    </el-card>

    <!-- Video player dialog -->
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
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { Delete, Cpu, WarningFilled, VideoPlay } from '@element-plus/icons-vue'
import { eventsAPI, archivesAPI } from '../api/client'
import { logger } from '../utils/logger'
import {
  eventTypeColor,
  eventTypeLabel,
  statColor,
  statIcon,
  formatDateTime,
} from '../utils/helpers'
import PageHeader from '../components/PageHeader.vue'
import StatCard from '../components/StatCard.vue'
import SkeletonLoader from '../components/SkeletonLoader.vue'
import ConfidenceGauge from '../components/ConfidenceGauge.vue'
import SearchFilterBar from '../components/SearchFilterBar.vue'
import ActionButton from '../components/ActionButton.vue'

const events = ref([])
const total = ref(0)
const page = ref(1)
const pageSize = 50
const filterType = ref('')
const filterSearch = ref('')
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

const eventTypeFilters = computed(() => [
  {
    key: 'search',
    type: 'input',
    label: '搜索特征',
    placeholder: '如：红色上衣、背包、戴帽子',
  },
  {
    key: 'event_type',
    type: 'select',
    label: '全部类型',
    options: eventTypes.value.map((t) => ({ value: t, label: eventTypeLabel(t) })),
  },
])

const snapshotUrl = (id) => eventsAPI.snapshotUrl(id)
const expandedRow = ref(null)

const onRowClick = (row) => {
  expandedRow.value = expandedRow.value === row.id ? null : row.id
}

const onSearch = (params) => {
  filterType.value = params.event_type || ''
  filterSearch.value = params.search || ''
  page.value = 1
  loadEvents()
  loadStats()
}

const loadEvents = async () => {
  loading.value = true
  loadError.value = false
  try {
    const params = { limit: pageSize, offset: (page.value - 1) * pageSize }
    if (filterType.value) params.event_type = filterType.value
    if (filterSearch.value) params.search = filterSearch.value
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
  filterSearch.value = ''
  page.value = 1
  loadEvents()
}

const clearAllEvents = async () => {
  await eventsAPI.deleteAll()
  loadStats()
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
  position: relative;
  z-index: 1;
}

.total-text {
  color: var(--text-secondary);
  font-size: 14px;
}

.total-text strong {
  color: var(--color-primary);
  font-size: 16px;
  font-variant-numeric: tabular-nums;
  font-family: 'JetBrains Mono', monospace;
}

.stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
}

.stat-card-wrap {
  min-width: 0;
}

/* Table card */
.table-card :deep(.el-card__body) {
  padding: 0;
}

.table-card :deep(.el-card__header) {
  padding: 14px 16px;
  border-bottom: 1px solid var(--border-subtle);
}

.events-table :deep(.el-tag) {
  font-weight: 500;
}

.pagination-wrapper {
  display: flex;
  justify-content: center;
  margin-top: 16px;
  padding: 0 16px 16px;
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
  color: var(--text-disabled);
  font-size: 12px;
}

/* Clips */
.clips-card :deep(.el-card__header) {
  padding: 14px 16px;
}

.clips-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.clips-title {
  font-weight: 600;
  font-size: 14px;
  color: var(--text-primary);
  display: flex;
  align-items: center;
}

.clips-count {
  color: var(--text-secondary);
  font-size: 13px;
  font-family: 'JetBrains Mono', monospace;
}

.clips-grid {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.clip-item {
  width: 160px;
  cursor: pointer;
  border-radius: var(--radius-md);
  overflow: hidden;
  border: 1px solid var(--border-subtle);
  transition: all var(--transition-fast);
  background: var(--bg-elevated);
}

.clip-item:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
  border-color: var(--border-default);
}

.clip-thumb {
  width: 100%;
  height: 90px;
  background: var(--bg-root);
  display: flex;
  align-items: center;
  justify-content: center;
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
  font-family: 'JetBrains Mono', monospace;
}

.clip-player {
  width: 100%;
  max-height: 480px;
  border-radius: var(--radius-sm);
  background: #000;
}

/* MLLM panel */
.expand-detail {
  padding: 8px 0;
}

.mllm-panel {
  background: var(--bg-root);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  padding: 14px 18px;
  max-width: 700px;
}

.mllm-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}

.mllm-title {
  font-weight: 600;
  font-size: 13px;
  color: var(--color-primary);
}

.mllm-narrative {
  font-size: 14px;
  line-height: 1.7;
  color: var(--text-primary);
  margin: 0 0 8px 0;
}

.mllm-action {
  font-size: 13px;
  color: var(--text-secondary);
  margin: 0;
  display: flex;
  align-items: center;
  gap: 4px;
}

.mllm-empty {
  color: var(--text-disabled);
  font-size: 13px;
}

.mllm-pending {
  margin-left: 8px;
  font-style: italic;
}

@media (max-width: 1024px) {
  :deep(.hide-tablet) {
    display: none;
  }
}

@media (max-width: 768px) {
  :deep(.hide-mobile) {
    display: none;
  }

  .stats-row {
    grid-template-columns: repeat(2, 1fr);
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

  .clip-item {
    width: 100%;
  }

  .clip-thumb {
    height: 120px;
  }
}
</style>
