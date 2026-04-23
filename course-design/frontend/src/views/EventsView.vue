<template>
  <div class="events-container">
    <!-- Filters -->
    <el-card shadow="never" class="filter-card">
      <el-row :gutter="16" align="middle">
        <el-col :span="6">
          <el-form-item label="事件类型" label-width="80px">
            <el-select v-model="filterType" placeholder="全部类型" clearable style="width: 100%">
              <el-option v-for="t in eventTypes" :key="t" :label="t" :value="t" />
            </el-select>
          </el-form-item>
        </el-col>
        <el-col :span="6">
          <el-button type="primary" @click="loadEvents">查询</el-button>
          <el-button @click="resetFilters">重置</el-button>
          <el-button type="danger" plain @click="confirmClearAll">清空全部</el-button>
        </el-col>
        <el-col :span="12" style="text-align: right">
          <span class="total-text">共 {{ total }} 条事件</span>
        </el-col>
      </el-row>
    </el-card>

    <!-- Stats Overview -->
    <el-row :gutter="16" style="margin-bottom: 16px">
      <el-col v-for="(count, type) in statsByType" :key="type" :span="typeSpan">
        <el-card shadow="never" :body-style="{ padding: '16px' }">
          <div class="stat-item">
            <span class="stat-label">{{ type }}</span>
            <span class="stat-value">{{ count }}</span>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Event List -->
    <el-card shadow="never">
      <el-table :data="events" stripe max-height="500">
        <el-table-column prop="event_type" label="类型" width="100">
          <template #default="{ row }">
            <el-tag :type="eventTypeColor(row.event_type)" size="small">
              {{ row.event_type }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="快照" width="80" align="center">
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
        <el-table-column prop="description" label="描述" min-width="200" />
        <el-table-column prop="track_id" label="目标ID" width="100" />
        <el-table-column prop="zone_name" label="区域" width="100" />
        <el-table-column prop="confidence" label="置信度" width="100">
          <template #default="{ row }">
            {{ row.confidence ? (row.confidence * 100).toFixed(1) + '%' : '-' }}
          </template>
        </el-table-column>
        <el-table-column prop="timestamp_s" label="时间" width="180">
          <template #default="{ row }">
            {{ formatTime(row.timestamp_s) }}
          </template>
        </el-table-column>
      </el-table>

      <!-- Pagination -->
      <div v-if="total > pageSize" class="pagination-wrapper">
        <el-pagination
          v-model:current-page="page"
          :page-size="pageSize"
          :total="total"
          layout="prev, pager, next"
          @current-change="loadEvents"
        />
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { ElMessageBox } from 'element-plus'
import { eventsAPI } from '../api/client'

const events = ref([])
const total = ref(0)
const page = ref(1)
const pageSize = 50
const filterType = ref('')
const eventTypes = ref([])
const stats = ref({ by_type: {} })

const statsByType = computed(() => stats.value.by_type || {})

const typeSpan = computed(() => {
  const count = Object.keys(statsByType.value).length
  if (count <= 0) return 24
  return Math.floor(24 / count)
})

const snapshotUrl = (id) => eventsAPI.snapshotUrl(id)

const loadEvents = async () => {
  try {
    const params = { limit: pageSize, offset: (page.value - 1) * pageSize }
    if (filterType.value) params.event_type = filterType.value
    const res = await eventsAPI.list(params)
    events.value = res.events
    total.value = res.total
  } catch (e) {
    console.error('Failed to load events:', e)
  }
}

const loadStats = async () => {
  try {
    stats.value = await eventsAPI.stats()
  } catch (e) {
    console.error('Failed to load stats:', e)
  }
}

const loadEventTypes = async () => {
  try {
    const res = await eventsAPI.types()
    eventTypes.value = res.event_types
  } catch (e) {
    console.error('Failed to load event types:', e)
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

const eventTypeColor = (type) => {
  const colors = {
    running: 'warning',
    fall: 'danger',
    crowd: 'info',
    intrusion: 'danger',
    fight: 'danger',
  }
  return colors[type] || 'info'
}

const formatTime = (ts) => {
  if (!ts) return ''
  return new Date(ts * 1000).toLocaleString()
}

onMounted(() => {
  loadEvents()
  loadStats()
  loadEventTypes()
})
</script>

<style scoped>
.events-container {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.filter-card {
  flex-shrink: 0;
}
.total-text {
  color: #666;
  font-size: 14px;
}
.stat-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.stat-label {
  font-size: 14px;
  color: #666;
}
.stat-value {
  font-size: 24px;
  font-weight: 700;
  color: #409eff;
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
}
.no-snap {
  color: #ccc;
  font-size: 12px;
}
</style>
