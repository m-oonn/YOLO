<!--
  ┌──────────────────────────────────────────┐
  │ 【前端UI】AlarmsView.vue — 报警管理页面    │
  │ 路由：/alarms                             │
  │ 职责：① 报警列表（按状态/级别/类型筛选）  │
  │       ② 确认报警（标记已知悉）            │
  │       ③ 处理报警（标记已解决）            │
  │       ④ 报警统计卡片（总数/活跃/严重）    │
  │       ⑤ 批量处理（按类型全部解决）        │
  │ 使用组件：SearchFilterBar, StatCard       │
  └──────────────────────────────────────────┘
  Copyright (c) 2025 YOLO Course Design Contributors
  SPDX-License-Identifier: Apache-2.0
-->

<template>
  <div class="alarms-container">
    <PageHeader title="报警管理" subtitle="查看和处理系统报警">
      <template #actions>
        <ActionButton
          size="small"
          type="danger"
          plain
          :action="clearAllAlarms"
          :confirmConfig="{ title: '确认清空', message: '确定要清空所有报警记录吗？此操作不可恢复！', type: 'warning' }"
          successMessage="报警已清空"
          errorMessage="清空失败"
          @success="() => { loadAlarms(); loadStats(); }"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px; vertical-align: -2px;">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
          </svg>
          清空全部
        </ActionButton>
      </template>
    </PageHeader>

    <!-- Alert summary bar -->
    <div class="alert-summary">
      <div class="alert-item" :class="{ active: filterLevel === null }" @click="filterLevel = null">
        <span class="alert-count">{{ total }}</span>
        <span class="alert-label">全部</span>
      </div>
      <div class="alert-item alert-critical" :class="{ active: filterLevel === 3 }" @click="filterLevel = 3">
        <span class="alert-count">{{ criticalCount }}</span>
        <span class="alert-label">紧急</span>
      </div>
      <div class="alert-item alert-warning" :class="{ active: filterLevel === 2 }" @click="filterLevel = 2">
        <span class="alert-count">{{ warningCount }}</span>
        <span class="alert-label">警告</span>
      </div>
      <div class="alert-item alert-info" :class="{ active: filterLevel === 1 }" @click="filterLevel = 1">
        <span class="alert-count">{{ infoCount }}</span>
        <span class="alert-label">提示</span>
      </div>
      <div class="alert-item alert-active" :class="{ active: filterActive }" @click="filterActive = !filterActive">
        <span class="alert-count">{{ activeCount }}</span>
        <span class="alert-label">未处理</span>
      </div>
    </div>

    <!-- Alarms table -->
    <el-card shadow="never" class="table-card">
      <SkeletonLoader v-if="skeletonLoading" type="table" :rows="5" />
      <el-table
        v-else
        v-loading="loading"
        :data="filteredAlarms"
        stripe
        max-height="600"
        :empty-text="loadError ? '加载失败，请检查后端服务' : '暂无报警'"
        class="alarms-table"
      >
        <el-table-column prop="event_type" label="类型" width="110">
          <template #default="{ row }">
            <el-tag :type="eventTypeColor(row.event_type)" size="small" effect="dark" class="event-tag">
              {{ eventTypeLabel(row.event_type) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="level" label="级别" width="80">
          <template #default="{ row }">
            <span class="level-badge" :class="`level-${row.level}`">
              {{ row.level === 3 ? '紧急' : row.level === 2 ? '警告' : '提示' }}
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="description" label="描述" min-width="200" show-overflow-tooltip />
        <el-table-column prop="count" label="触发次数" width="90" align="center">
          <template #default="{ row }">
            <span class="count-value">{{ row.count }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="last_event_time" label="最近触发" width="170">
          <template #default="{ row }">
            {{ formatDateTime(row.last_event_time) }}
          </template>
        </el-table-column>
        <el-table-column prop="resolved" label="状态" width="80" align="center">
          <template #default="{ row }">
            <StatusBadge
              :status="row.resolved ? 'online' : 'warning'"
              :label="row.resolved ? '已处理' : '未处理'"
            />
          </template>
        </el-table-column>
        <el-table-column label="操作" width="110" fixed="right">
          <template #default="{ row }">
            <ActionButton
              v-if="!row.resolved"
              size="small"
              type="success"
              text
              :action="() => resolveAlarm(row.id)"
              successMessage="报警已处理"
              errorMessage="处理失败"
              @success="() => { loadAlarms(); loadStats(); }"
            >
              处理
            </ActionButton>
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
          @current-change="loadAlarms"
        />
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { alarmsAPI } from '../api/client'
import { logger } from '../utils/logger'
import { eventTypeColor, eventTypeLabel, formatDateTime } from '../utils/helpers'
import PageHeader from '../components/PageHeader.vue'
import StatusBadge from '../components/StatusBadge.vue'
import SkeletonLoader from '../components/SkeletonLoader.vue'
import ActionButton from '../components/ActionButton.vue'

const alarms = ref([])
const total = ref(0)
const page = ref(1)
const pageSize = 50
const loading = ref(false)
const loadError = ref(false)
const skeletonLoading = ref(true)
const filterLevel = ref(null)
const filterActive = ref(false)

const stats = ref({})

const filteredAlarms = computed(() => {
  let list = alarms.value
  if (filterLevel.value !== null) {
    list = list.filter(a => a.level === filterLevel.value)
  }
  if (filterActive.value) {
    list = list.filter(a => !a.resolved)
  }
  return list
})

const criticalCount = computed(() => alarms.value.filter(a => a.level === 3).length)
const warningCount = computed(() => alarms.value.filter(a => a.level === 2).length)
const infoCount = computed(() => alarms.value.filter(a => a.level === 1).length)
const activeCount = computed(() => alarms.value.filter(a => !a.resolved).length)

const loadAlarms = async () => {
  loading.value = true
  loadError.value = false
  try {
    const params = { limit: pageSize, offset: (page.value - 1) * pageSize }
    const res = await alarmsAPI.list(params)
    alarms.value = (res.items || res.alarms || []).map(a => ({
      ...a,
      resolved: a.status === 'resolved',
    }))
    total.value = res.total || 0
  } catch (e) {
    logger.error('Failed to load alarms:', e)
    loadError.value = true
  } finally {
    loading.value = false
    skeletonLoading.value = false
  }
}

const loadStats = async () => {
  try {
    stats.value = await alarmsAPI.stats()
  } catch (e) {
    logger.error('Failed to load alarm stats:', e)
  }
}

const resolveAlarm = async (id) => {
  await alarmsAPI.resolve(id)
  await loadAlarms()
  await loadStats()
}

const clearAllAlarms = async () => {
  await alarmsAPI.deleteAll()
}

onMounted(() => {
  loadAlarms()
  loadStats()
})
</script>

<style scoped>
.alarms-container {
  display: flex;
  flex-direction: column;
  gap: 16px;
  max-width: 1400px;
  margin: 0 auto;
  position: relative;
  z-index: 1;
}

/* Alert summary bar */
.alert-summary {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.alert-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 14px 24px;
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--transition-fast);
  min-width: 90px;
  position: relative;
  overflow: hidden;
}

.alert-item::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--text-secondary);
  opacity: 0.3;
}

.alert-item:hover {
  border-color: var(--border-default);
  box-shadow: var(--shadow-sm);
  transform: translateY(-1px);
}

.alert-item.active {
  border-color: var(--color-primary);
  box-shadow: 0 0 15px rgba(245, 158, 11, 0.1);
}

.alert-item.active::before {
  background: var(--color-primary);
  opacity: 0.8;
}

.alert-critical::before {
  background: var(--color-danger);
}

.alert-warning::before {
  background: var(--color-warning);
}

.alert-info::before {
  background: var(--color-info);
}

.alert-active::before {
  background: var(--color-success);
}

.alert-count {
  font-size: 22px;
  font-weight: 700;
  color: var(--text-primary);
  font-family: 'JetBrains Mono', monospace;
  line-height: 1;
}

.alert-label {
  font-size: 11px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* Table */
.table-card :deep(.el-card__body) {
  padding: 0;
}

.alarms-table :deep(.el-tag) {
  font-weight: 500;
}

.level-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
}

.level-3 {
  background: rgba(239, 68, 68, 0.1);
  color: var(--color-danger);
}

.level-2 {
  background: rgba(245, 158, 11, 0.1);
  color: var(--color-warning);
}

.level-1 {
  background: rgba(6, 182, 212, 0.1);
  color: var(--color-info);
}

.count-value {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
  color: var(--text-primary);
}

.pagination-wrapper {
  display: flex;
  justify-content: center;
  margin-top: 16px;
  padding: 0 16px 16px;
}

@media (max-width: 768px) {
  .alert-summary {
    gap: 8px;
  }

  .alert-item {
    padding: 10px 16px;
    min-width: 70px;
  }

  .alert-count {
    font-size: 18px;
  }
}
</style>
