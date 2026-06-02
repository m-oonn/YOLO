<!--
  Copyright (c) 2025 YOLO Course Design Contributors
  SPDX-License-Identifier: Apache-2.0
-->

<template>
  <div class="config-container">
    <PageHeader title="系统配置" subtitle="调整检测规则、区域和系统参数" />

    <!-- Config sections -->
    <div class="config-grid">
      <!-- Detection Rules -->
      <el-card class="config-card">
        <template #header>
          <div class="card-header">
            <div class="header-title-group">
              <span class="card-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 6px; vertical-align: -2px;">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
                检测规则
              </span>
            </div>
            <ActionButton
              size="small"
              type="primary"
              :action="saveRules"
              successMessage="规则已保存"
              errorMessage="保存失败"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px; vertical-align: -2px;">
                <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
                <polyline points="17 21 17 13 7 13 7 21"/>
                <polyline points="7 3 7 8 15 8"/>
              </svg>
              保存
            </ActionButton>
          </div>
        </template>
        <div class="rules-list">
          <div v-for="(rule, key) in rules" :key="key" class="rule-item">
            <div class="rule-info">
              <span class="rule-name">{{ ruleName(key) }}</span>
              <span class="rule-desc">{{ ruleDesc(key) }}</span>
            </div>
            <div class="rule-controls">
              <el-switch v-model="rule.enabled" />
              <el-slider
                v-if="rule.threshold !== undefined"
                v-model="rule.threshold"
                :min="0"
                :max="1"
                :step="0.05"
                show-stops
                class="rule-slider"
              />
              <span v-if="rule.threshold !== undefined" class="rule-value">{{ (rule.threshold * 100).toFixed(0) }}%</span>
            </div>
          </div>
        </div>
      </el-card>

      <!-- Zones -->
      <el-card class="config-card">
        <template #header>
          <div class="card-header">
            <div class="header-title-group">
              <span class="card-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 6px; vertical-align: -2px;">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                  <line x1="3" y1="9" x2="21" y2="9"/>
                  <line x1="9" y1="21" x2="9" y2="9"/>
                </svg>
                检测区域
              </span>
            </div>
            <el-button size="small" type="primary" @click="showZoneDialog = true">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px; vertical-align: -2px;">
                <line x1="12" y1="5" x2="12" y2="19"/>
                <line x1="5" y1="12" x2="19" y2="12"/>
              </svg>
              添加区域
            </el-button>
          </div>
        </template>
        <div v-if="zones.length === 0" class="empty-state">
          <div class="empty-icon">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--text-disabled)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
              <line x1="3" y1="9" x2="21" y2="9"/>
              <line x1="9" y1="21" x2="9" y2="9"/>
            </svg>
          </div>
          <p class="empty-title">暂无检测区域</p>
          <p class="empty-desc">点击上方按钮添加</p>
        </div>
        <div v-else class="zones-list">
          <div v-for="(zone, idx) in zones" :key="idx" class="zone-item">
            <div class="zone-info">
              <span class="zone-name">{{ zone.name }}</span>
              <span class="zone-type">{{ zoneTypeLabel(zone.type) }}</span>
            </div>
            <div class="zone-actions">
              <el-button size="small" text @click="editZone(idx)">编辑</el-button>
              <ActionButton
                size="small"
                type="danger"
                text
                :action="() => deleteZone(idx)"
                :confirmConfig="{ title: '确认删除', message: `确定要删除区域「${zone.name}」吗？`, type: 'warning' }"
                successMessage="区域已删除"
                errorMessage="删除失败"
              >
                删除
              </ActionButton>
            </div>
          </div>
        </div>
      </el-card>

      <!-- System Settings -->
      <el-card class="config-card">
        <template #header>
          <div class="card-header">
            <div class="header-title-group">
              <span class="card-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 6px; vertical-align: -2px;">
                  <circle cx="12" cy="12" r="3"/>
                  <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                </svg>
                系统设置
              </span>
            </div>
            <ActionButton
              size="small"
              type="primary"
              :action="saveSettings"
              successMessage="设置已保存"
              errorMessage="保存失败"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px; vertical-align: -2px;">
                <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
                <polyline points="17 21 17 13 7 13 7 21"/>
                <polyline points="7 3 7 8 15 8"/>
              </svg>
              保存
            </ActionButton>
          </div>
        </template>
        <div class="settings-list">
          <div class="setting-item">
            <div class="setting-info">
              <span class="setting-name">检测置信度阈值</span>
              <span class="setting-desc">低于此值的检测结果将被忽略</span>
            </div>
            <div class="setting-control">
              <el-slider v-model="settings.confidence" :min="0" :max="1" :step="0.05" show-stops class="setting-slider" />
              <span class="setting-value">{{ (settings.confidence * 100).toFixed(0) }}%</span>
            </div>
          </div>
          <div class="setting-item">
            <div class="setting-info">
              <span class="setting-name">NMS 重叠阈值</span>
              <span class="setting-desc">非极大值抑制的 IoU 阈值</span>
            </div>
            <div class="setting-control">
              <el-slider v-model="settings.iou" :min="0" :max="1" :step="0.05" show-stops class="setting-slider" />
              <span class="setting-value">{{ (settings.iou * 100).toFixed(0) }}%</span>
            </div>
          </div>
          <div class="setting-item">
            <div class="setting-info">
              <span class="setting-name">帧跳过间隔</span>
              <span class="setting-desc">每 N 帧执行一次检测</span>
            </div>
            <div class="setting-control">
              <el-input-number v-model="settings.frame_skip" :min="1" :max="30" size="small" />
            </div>
          </div>
          <div class="setting-item">
            <div class="setting-info">
              <span class="setting-name">保存快照</span>
              <span class="setting-desc">检测到时自动保存图片</span>
            </div>
            <div class="setting-control">
              <el-switch v-model="settings.save_snapshots" />
            </div>
          </div>
          <div class="setting-item">
            <div class="setting-info">
              <span class="setting-name">录制视频</span>
              <span class="setting-desc">检测期间自动录制视频</span>
            </div>
            <div class="setting-control">
              <el-switch v-model="settings.record_video" />
            </div>
          </div>
        </div>
      </el-card>

      <!-- MLLM Settings -->
      <el-card class="config-card">
        <template #header>
          <div class="card-header">
            <div class="header-title-group">
              <span class="card-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 6px; vertical-align: -2px;">
                  <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                </svg>
                MLLM 场景理解
              </span>
            </div>
            <ActionButton
              size="small"
              type="primary"
              :action="saveMllmSettings"
              successMessage="MLLM 设置已保存"
              errorMessage="保存失败"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px; vertical-align: -2px;">
                <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
                <polyline points="17 21 17 13 7 13 7 21"/>
                <polyline points="7 3 7 8 15 8"/>
              </svg>
              保存
            </ActionButton>
          </div>
        </template>
        <div class="settings-list">
          <div class="setting-item">
            <div class="setting-info">
              <span class="setting-name">启用 MLLM</span>
              <span class="setting-desc">开启多模态大语言模型场景分析</span>
            </div>
            <div class="setting-control">
              <el-switch v-model="mllmSettings.enabled" />
            </div>
          </div>
          <div class="setting-item">
            <div class="setting-info">
              <span class="setting-name">分析间隔</span>
              <span class="setting-desc">每 N 秒执行一次场景分析</span>
            </div>
            <div class="setting-control">
              <el-input-number v-model="mllmSettings.interval" :min="5" :max="300" size="small" />
              <span class="setting-unit">秒</span>
            </div>
          </div>
          <div class="setting-item">
            <div class="setting-info">
              <span class="setting-name">告警增强</span>
              <span class="setting-desc">使用 MLLM 验证和增强告警信息</span>
            </div>
            <div class="setting-control">
              <el-switch v-model="mllmSettings.enhance_alarms" />
            </div>
          </div>
        </div>
      </el-card>
    </div>

    <!-- Zone dialog -->
    <el-dialog
      v-model="showZoneDialog"
      :title="editingZoneIndex !== null ? '编辑区域' : '添加区域'"
      width="480px"
      :close-on-click-modal="false"
      class="zone-dialog"
    >
      <el-form :model="zoneForm" label-width="80px" class="zone-form">
        <el-form-item label="名称">
          <el-input v-model="zoneForm.name" placeholder="如：教学楼入口" />
        </el-form-item>
        <el-form-item label="类型">
          <el-select v-model="zoneForm.type" placeholder="选择区域类型">
            <el-option label="禁区" value="forbidden" />
            <el-option label="关注区" value="attention" />
            <el-option label="计数区" value="counting" />
          </el-select>
        </el-form-item>
        <el-form-item label="坐标">
          <div class="coord-editor">
            <div v-for="(pt, i) in zonePoints" :key="i" class="coord-row">
              <el-input-number v-model="pt.x" :min="0" :precision="0" size="small" placeholder="X" class="coord-input" />
              <el-input-number v-model="pt.y" :min="0" :precision="0" size="small" placeholder="Y" class="coord-input" />
              <el-button size="small" text type="danger" @click="removePoint(i)">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"/>
                  <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </el-button>
            </div>
            <el-button size="small" text type="primary" @click="addPoint" class="add-point-btn">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px; vertical-align: -2px;">
                <line x1="12" y1="5" x2="12" y2="19"/>
                <line x1="5" y1="12" x2="19" y2="12"/>
              </svg>
              添加坐标点
            </el-button>
            <div class="coord-preview">
              <span class="preview-label">预览:</span>
              <code class="preview-code">{{ coordPreview }}</code>
            </div>
          </div>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showZoneDialog = false">取消</el-button>
        <el-button type="primary" @click="saveZone">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { configAPI, mllmAPI } from '../api/client'
import { logger } from '../utils/logger'
import PageHeader from '../components/PageHeader.vue'
import ActionButton from '../components/ActionButton.vue'

const rules = ref({})
const zones = ref([])
const settings = ref({ confidence: 0.5, iou: 0.45, frame_skip: 1, save_snapshots: true, record_video: false })
const mllmSettings = ref({ enabled: false, interval: 30, enhance_alarms: true })
const showZoneDialog = ref(false)
const editingZoneIndex = ref(null)
const zoneForm = ref({ name: '', type: 'forbidden', coordinates: '' })
const zonePoints = ref([{ x: 100, y: 100 }, { x: 200, y: 100 }, { x: 200, y: 200 }, { x: 100, y: 200 }])

const coordPreview = computed(() => {
  return JSON.stringify(zonePoints.value.map(p => [p.x, p.y]))
})

const addPoint = () => {
  zonePoints.value.push({ x: 0, y: 0 })
}

const removePoint = (i) => {
  if (zonePoints.value.length <= 3) {
    ElMessage.warning('区域至少需要 3 个坐标点')
    return
  }
  zonePoints.value.splice(i, 1)
}

watch(showZoneDialog, (visible) => {
  if (!visible) {
    editingZoneIndex.value = null
    zoneForm.value = { name: '', type: 'forbidden', coordinates: '' }
    zonePoints.value = [{ x: 100, y: 100 }, { x: 200, y: 100 }, { x: 200, y: 200 }, { x: 100, y: 200 }]
  }
})

const ruleName = (key) => {
  const names = {
    running: '奔跑检测',
    falling: '摔倒检测',
    fighting: '打架检测',
    crowd: '人群聚集',
    forbidden_zone: '禁区入侵',
    vehicle_zone: '车辆入侵',
  }
  return names[key] || key
}

const ruleDesc = (key) => {
  const descs = {
    running: '检测人员快速移动行为',
    falling: '检测人员倒地行为',
    fighting: '检测多人肢体冲突',
    crowd: '检测人员聚集超过阈值',
    forbidden_zone: '检测人员进入禁止区域',
    vehicle_zone: '检测车辆进入限制区域',
  }
  return descs[key] || ''
}

const zoneTypeLabel = (type) => {
  const labels = { forbidden: '禁区', attention: '关注区', counting: '计数区' }
  return labels[type] || type
}

const loadConfig = async () => {
  try {
    const res = await configAPI.get()
    const cfg = res.config || res
    rules.value = cfg.rules || {}
    zones.value = cfg.zones || []
    settings.value = { ...settings.value, ...cfg.settings }
  } catch (e) {
    logger.error('Failed to load config:', e)
  }
}

const loadMllmConfig = async () => {
  try {
    const res = await mllmAPI.getConfig()
    const cfg = res.config || res
    mllmSettings.value = { ...mllmSettings.value, ...cfg }
  } catch (e) {
    logger.error('Failed to load MLLM config:', e)
  }
}

const saveRules = async () => {
  await configAPI.updateRules(rules.value)
}

const saveSettings = async () => {
  await configAPI.updateSettings(settings.value)
}

const saveMllmSettings = async () => {
  await mllmAPI.updateConfig(mllmSettings.value)
}

const editZone = (idx) => {
  editingZoneIndex.value = idx
  const zone = zones.value[idx]
  zoneForm.value = {
    name: zone.name,
    type: zone.type,
    coordinates: JSON.stringify(zone.coordinates),
  }
  zonePoints.value = zone.coordinates.map(([x, y]) => ({ x, y }))
  showZoneDialog.value = true
}

const deleteZone = async (idx) => {
  const backup = [...zones.value]
  zones.value.splice(idx, 1)
  try {
    await saveZones()
  } catch (e) {
    zones.value = backup
    throw e
  }
}

const saveZone = async () => {
  try {
    if (zonePoints.value.length < 3) {
      ElMessage.warning('区域至少需要 3 个坐标点')
      return
    }
    const coords = zonePoints.value.map(p => [p.x, p.y])
    const zone = {
      name: zoneForm.value.name,
      type: zoneForm.value.type,
      coordinates: coords,
    }
    if (editingZoneIndex.value !== null) {
      zones.value[editingZoneIndex.value] = zone
    } else {
      zones.value.push(zone)
    }
    await saveZones()
    showZoneDialog.value = false
    ElMessage.success('区域已保存')
  } catch (e) {
    ElMessage.error('保存失败：' + (e.message || '未知错误'))
    logger.error('Failed to save zone:', e)
  }
}

const saveZones = async () => {
  await configAPI.updateZones(zones.value)
}

onMounted(() => {
  loadConfig()
  loadMllmConfig()
})
</script>

<style scoped>
.config-container {
  display: flex;
  flex-direction: column;
  gap: 16px;
  max-width: 1400px;
  margin: 0 auto;
  position: relative;
  z-index: 1;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

.config-card :deep(.el-card__header) {
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
  display: flex;
  align-items: center;
}

/* Rules */
.rules-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.rule-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 14px;
  background: var(--bg-root);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-subtle);
  transition: all var(--transition-fast);
}

.rule-item:hover {
  border-color: var(--border-default);
}

.rule-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex: 1;
  min-width: 0;
}

.rule-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
}

.rule-desc {
  font-size: 11px;
  color: var(--text-secondary);
}

.rule-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}

.rule-slider {
  width: 120px;
}

.rule-value {
  font-size: 12px;
  color: var(--color-primary);
  font-family: 'JetBrains Mono', monospace;
  min-width: 36px;
  text-align: right;
}

/* Zones */
.zones-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.zone-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 14px;
  background: var(--bg-root);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-subtle);
  transition: all var(--transition-fast);
}

.zone-item:hover {
  border-color: var(--border-default);
}

.zone-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex: 1;
  min-width: 0;
}

.zone-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
}

.zone-type {
  font-size: 11px;
  color: var(--text-secondary);
}

.zone-actions {
  display: flex;
  gap: 4px;
  flex-shrink: 0;
}

/* Settings */
.settings-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.setting-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 14px;
  background: var(--bg-root);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-subtle);
  transition: all var(--transition-fast);
}

.setting-item:hover {
  border-color: var(--border-default);
}

.setting-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex: 1;
  min-width: 0;
}

.setting-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
}

.setting-desc {
  font-size: 11px;
  color: var(--text-secondary);
}

.setting-control {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}

.setting-slider {
  width: 140px;
}

.setting-value {
  font-size: 12px;
  color: var(--color-primary);
  font-family: 'JetBrains Mono', monospace;
  min-width: 40px;
  text-align: right;
}

.setting-unit {
  font-size: 12px;
  color: var(--text-secondary);
}

/* Empty state */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 32px 20px;
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

/* Zone dialog */
.zone-dialog :deep(.el-dialog__body) {
  padding: 20px;
}

.zone-form :deep(.el-form-item__label) {
  color: var(--text-regular);
  font-weight: 500;
}

.coord-editor {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.coord-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.coord-input {
  width: 100px;
}

.coord-input :deep(.el-input__wrapper) {
  padding: 0 8px;
}

.add-point-btn {
  align-self: flex-start;
}

.coord-preview {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  background: var(--bg-root);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-subtle);
}

.preview-label {
  font-size: 11px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.preview-code {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--color-primary);
  background: transparent;
}

@media (max-width: 1024px) {
  .config-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .rule-item {
    flex-direction: column;
    align-items: flex-start;
  }

  .rule-controls {
    width: 100%;
  }

  .rule-slider {
    flex: 1;
  }

  .setting-item {
    flex-direction: column;
    align-items: flex-start;
  }

  .setting-control {
    width: 100%;
  }

  .setting-slider {
    flex: 1;
  }
}
</style>
