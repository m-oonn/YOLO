<!--
  Copyright (c) 2025 YOLO Course Design Contributors
  SPDX-License-Identifier: MIT
-->

<template>
  <div class="config-container">
    <el-card shadow="never">
      <template #header>
        <div class="config-header">
          <span class="config-title">系统配置</span>
          <div class="config-actions">
            <el-button type="primary" :loading="saving" @click="saveConfig">保存配置</el-button>
            <el-button @click="resetConfig">恢复默认</el-button>
          </div>
        </div>
      </template>

      <el-form ref="configFormRef" :model="config" :rules="formRules" label-width="130px" label-position="left" class="config-form">
        <el-divider content-position="left">
          <el-icon><Cpu /></el-icon>
          模型配置
        </el-divider>

        <el-row :gutter="16">
          <el-col :xs="24" :sm="12" :md="8">
            <el-form-item label="当前模型">
              <div class="current-model-display">
                <el-tag type="success" size="large" effect="plain">
                  {{ currentModel }}
                </el-tag>
                <el-button
                  type="primary"
                  plain
                  size="small"
                  style="margin-left: 12px"
                  @click="openModelSelector"
                >
                  切换模型
                </el-button>
              </div>
            </el-form-item>
          </el-col>
          <el-col :xs="24" :sm="12" :md="8">
            <el-form-item label="输入尺寸" prop="model.imgsz">
              <el-input-number
                v-model="config.model.imgsz"
                :min="320"
                :max="1280"
                :step="32"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="16">
          <el-col :xs="24" :sm="12" :md="8">
            <el-form-item label="置信度阈值" prop="model.conf">
              <el-slider
                v-model="config.model.conf"
                :min="0.1"
                :max="0.9"
                :step="0.05"
                show-input
              />
            </el-form-item>
          </el-col>
          <el-col :xs="24" :sm="12" :md="8">
            <el-form-item label="IoU 阈值">
              <el-slider v-model="config.model.iou" :min="0.1" :max="0.9" :step="0.05" show-input />
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">
          <el-icon><Setting /></el-icon>
          GPU加速
        </el-divider>

        <el-row :gutter="16">
          <el-col :span="24">
            <div v-if="gpuInfo" class="gpu-status-card">
              <div class="gpu-status-header">
                <el-tag
                  :type="gpuInfo.gpu_available ? 'success' : 'danger'"
                  size="large"
                  effect="dark"
                >
                  {{ gpuInfo.gpu_available ? 'GPU 已启用' : 'GPU 未检测到' }}
                </el-tag>
                <span v-if="gpuInfo.gpu_name" class="gpu-name">{{ gpuInfo.gpu_name }}</span>
              </div>
              <div v-if="gpuInfo.gpu_available" class="gpu-details">
                <div class="gpu-detail-row">
                  <div class="gpu-detail-item">
                    <span class="gpu-detail-label">显存</span>
                    <el-progress
                      :percentage="
                        gpuInfo.gpu_total_memory_mb
                          ? Math.round(
                              (gpuInfo.gpu_used_memory_mb / gpuInfo.gpu_total_memory_mb) * 100
                            )
                          : 0
                      "
                      :format="
                        () =>
                          `${gpuInfo.gpu_used_memory_mb || 0}/${gpuInfo.gpu_total_memory_mb || 0}MB`
                      "
                      :color="
                        gpuInfo.gpu_total_memory_mb &&
                        gpuInfo.gpu_used_memory_mb / gpuInfo.gpu_total_memory_mb > 0.9
                          ? '#f56c6c'
                          : '#409eff'
                      "
                      style="width: 180px"
                    />
                  </div>
                  <div class="gpu-detail-item">
                    <span class="gpu-detail-label">计算能力</span>
                    <span class="gpu-detail-value">{{ gpuInfo.gpu_compute_capability }}</span>
                  </div>
                  <div class="gpu-detail-item">
                    <span class="gpu-detail-label">CUDA版本</span>
                    <span class="gpu-detail-value">{{ gpuInfo.cuda_version || 'N/A' }}</span>
                  </div>
                </div>
                <div class="gpu-detail-row">
                  <div class="gpu-detail-item">
                    <span class="gpu-detail-label">半精度</span>
                    <el-tag :type="gpuInfo.supports_half ? 'success' : 'info'" size="small">
                      {{ gpuInfo.supports_half ? '支持' : '不支持' }}
                    </el-tag>
                  </div>
                  <div class="gpu-detail-item">
                    <span class="gpu-detail-label">Tensor Cores</span>
                    <el-tag :type="gpuInfo.supports_tensor_cores ? 'success' : 'info'" size="small">
                      {{ gpuInfo.supports_tensor_cores ? '支持' : '不支持' }}
                    </el-tag>
                  </div>
                </div>
              </div>
              <div v-if="gpuRecommendations.length > 0" class="gpu-recommendations">
                <div v-for="(rec, i) in gpuRecommendations" :key="i" class="gpu-rec-item">
                  <el-icon color="#e6a23c"><WarningFilled /></el-icon>
                  <span>{{ rec }}</span>
                </div>
              </div>
            </div>
            <div v-else class="gpu-status-card">
              <el-button type="primary" size="small" @click="fetchGPUInfo">检测GPU</el-button>
            </div>
          </el-col>
        </el-row>

        <el-divider content-position="left">
          <el-icon><Setting /></el-icon>
          性能优化
        </el-divider>

        <el-row :gutter="24">
          <el-col :xs="24" :sm="8">
            <el-form-item label="目标帧率" prop="model.camera_fps">
              <el-input-number
                v-model="config.model.camera_fps"
                :min="5"
                :max="60"
                :step="5"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :xs="24" :sm="8">
            <el-form-item label="骨架帧跳过">
              <el-input-number
                v-model="config.model.process_interval"
                :min="1"
                :max="8"
                :step="1"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :xs="24" :sm="8">
            <el-form-item label="JPEG质量" prop="model.jpeg_quality">
              <el-input-number
                v-model="config.model.jpeg_quality"
                :min="30"
                :max="95"
                :step="5"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="24">
          <el-col :xs="24" :md="16">
            <el-form-item label="推理缩放">
              <el-slider
                v-model="inferenceScalePercent"
                :min="25"
                :max="100"
                :step="25"
                show-stops
                :format-tooltip="(v) => (v / 100).toFixed(2) + 'x'"
                style="width: 100%; padding: 0 8px"
              />
              <div class="form-tip">缩小推理分辨率可显著提升帧率，但会降低小目标检测精度</div>
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">
          <el-icon><Setting /></el-icon>
          检测规则
        </el-divider>

        <el-row :gutter="12">
          <el-col v-for="(rule, key) in rulesList" :key="key" :xs="24" :sm="12" :md="8">
            <el-card shadow="never" class="rule-card" :body-style="{ padding: '16px' }">
              <div class="rule-header">
                <span class="rule-name">{{ rule.label }}</span>
                <el-switch v-model="config.rules[key].enabled" />
              </div>
              <transition name="fade">
                <div v-if="config.rules[key].enabled" class="rule-detail">
                  <span class="rule-detail-text">{{ rule.detail }}</span>
                </div>
              </transition>
            </el-card>
          </el-col>
        </el-row>

        <el-divider content-position="left">
          <el-icon><Location /></el-icon>
          入侵区域编辑
        </el-divider>

        <el-form-item label="监控区域">
          <div class="zone-editor">
            <div class="zone-list">
              <div
                v-for="(zone, zi) in config.rules.intrusion.zones"
                :key="zi"
                class="zone-item"
                :class="{ active: selectedZoneIdx === zi }"
                @click="selectedZoneIdx = zi"
              >
                <el-input
                  v-model="zone.name"
                  size="small"
                  style="width: 80px"
                  placeholder="区域名称"
                />
                <el-tag size="small" type="info">{{ zone.polygon.length }} 点</el-tag>
                <el-button
                  type="danger"
                  size="small"
                  circle
                  :icon="Delete"
                  @click.stop="removeZone(zi)"
                />
              </div>
              <div class="zone-actions">
                <el-button size="small" type="primary" plain @click="addZone">+ 添加区域</el-button>
                <el-button
                  v-if="config.rules.intrusion.zones.length > 0"
                  size="small"
                  type="danger"
                  plain
                  @click="clearAllZones"
                >
                  清除所有
                </el-button>
              </div>
              <div v-if="config.rules.intrusion.zones.length === 0" class="zone-empty">
                <el-icon :size="24" color="#999"><Location /></el-icon>
                <span>暂无入侵区域</span>
              </div>
            </div>
            <div class="canvas-wrapper">
              <canvas
                ref="polyCanvas"
                width="640"
                height="360"
                class="poly-canvas"
                @click="onCanvasClick"
                @contextmenu.prevent="onCanvasRightClick"
              ></canvas>
              <div class="canvas-tip">左键添加顶点 | 右键撤销 | 点击区域列表选择编辑</div>
            </div>
          </div>
        </el-form-item>

        <el-divider content-position="left">
          <el-icon><FolderOpened /></el-icon>
          输出配置
        </el-divider>

        <el-row :gutter="16">
          <el-col :xs="24" :sm="12">
            <el-form-item label="保存截图">
              <el-switch v-model="config.output.save_snapshots" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">
          <el-icon><MagicStick /></el-icon>
          MLLM 场景理解
        </el-divider>

        <el-row :gutter="16">
          <el-col :xs="24" :sm="8">
            <el-form-item label="启用MLLM">
              <el-switch v-model="config.mllm.enabled" />
            </el-form-item>
          </el-col>
          <el-col :xs="24" :sm="8">
            <el-form-item label="影子模式">
              <el-switch v-model="config.mllm.shadow_mode" :disabled="!config.mllm.enabled" />
            </el-form-item>
          </el-col>
          <el-col :xs="24" :sm="8">
            <el-form-item label="推理后端">
              <el-select
                v-model="config.mllm.inference_backend"
                :disabled="!config.mllm.enabled"
                style="width: 100%"
              >
                <el-option label="自动" value="auto" />
                <el-option label="PyTorch" value="pytorch" />
                <el-option label="TensorRT" value="tensorrt" />
                <el-option label="Mock(测试)" value="mock" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="16">
          <el-col :xs="24" :sm="8">
            <el-form-item label="模型类型">
              <el-select
                v-model="config.mllm.model_type"
                :disabled="!config.mllm.enabled"
                style="width: 100%"
              >
                <el-option label="Qwen2-VL-2B" value="qwen2-vl-2b" />
                <el-option label="SmolVLM-500M" value="smolvlm-500m" />
                <el-option label="Florence-2" value="florence-2" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :xs="24" :sm="8">
            <el-form-item label="关键帧间隔">
              <el-input-number
                v-model="config.mllm.key_frame_interval"
                :min="5"
                :max="60"
                :step="5"
                :disabled="!config.mllm.enabled"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :xs="24" :sm="8">
            <el-form-item label="最大Token数">
              <el-input-number
                v-model="config.mllm.max_new_tokens"
                :min="64"
                :max="1024"
                :step="64"
                :disabled="!config.mllm.enabled"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="16">
          <el-col :xs="24" :sm="8">
            <el-form-item label="场景描述">
              <el-switch
                v-model="config.mllm.scene_description_enabled"
                :disabled="!config.mllm.enabled"
              />
            </el-form-item>
          </el-col>
          <el-col :xs="24" :sm="8">
            <el-form-item label="告警增强">
              <el-switch
                v-model="config.mllm.alarm_enhance_enabled"
                :disabled="!config.mllm.enabled"
              />
            </el-form-item>
          </el-col>
          <el-col :xs="24" :sm="8">
            <el-form-item label="增强冷却(秒)">
              <el-input-number
                v-model="config.mllm.enhancement_cooldown_s"
                :min="5"
                :max="60"
                :step="5"
                :disabled="!config.mllm.enabled"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
    </el-card>

    <!-- 模型选择对话框 -->
    <el-dialog
      v-model="modelSelectDialogVisible"
      title="选择检测模型"
      width="500px"
      :close-on-click-modal="false"
      class="model-dialog"
    >
      <div class="model-select-content">
        <el-alert
          title="模型切换说明"
          type="info"
          :closable="false"
          show-icon
          style="margin-bottom: 20px"
        >
          <template #default>
            切换模型后，系统将重新加载模型文件。
            <br />
            这可能会短暂中断当前的检测服务。
          </template>
        </el-alert>

        <el-form label-width="100px" label-position="left">
          <el-form-item label="选择模型">
            <el-select
              v-model="selectedNewModel"
              placeholder="请选择模型"
              style="width: 100%"
              filterable
            >
              <el-option
                v-for="model in availableModels"
                :key="model.path"
                :label="model.name"
                :value="model.path"
              >
                <div class="model-option">
                  <span class="model-name">{{ model.name }}</span>
                  <span class="model-size">{{ model.size_mb }} MB</span>
                </div>
              </el-option>
            </el-select>
          </el-form-item>

          <el-form-item label="当前模型">
            <el-tag type="info">{{ currentModel }}</el-tag>
          </el-form-item>

          <el-form-item label="新模型">
            <el-tag v-if="selectedNewModel" type="success">{{ selectedNewModel }}</el-tag>
            <span v-else style="color: #999">请在上方选择新模型</span>
          </el-form-item>
        </el-form>
      </div>

      <template #footer>
        <div class="dialog-footer">
          <el-button @click="modelSelectDialogVisible = false">取消</el-button>
          <el-button
            type="primary"
            :loading="modelSwitching"
            :disabled="!selectedNewModel || selectedNewModel === currentModel"
            @click="confirmModelSwitch"
          >
            确认切换
          </el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
/**
 * ConfigView.vue - System configuration and model management page.
 *
 * Provides a comprehensive settings interface for:
 * - YOLO model selection and switching (nano/small/medium/large/xlarge)
 * - Detection parameters (confidence, IoU, image size, process interval)
 * - Camera settings (FPS, buffer size)
 * - Rule toggles (intrusion, fight, fall, crowd, running)
 * - Alarm configuration (cooldown, sound, zone polygons)
 * - MLLM scene description settings
 * - GPU status monitoring
 * - Pose estimation settings
 */
import { reactive, ref, watch, nextTick, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  Delete,
  Location,
  Setting,
  Cpu,
  WarningFilled,
  MagicStick,
  FolderOpened,
} from '@element-plus/icons-vue'
import { detectionAPI, mllmAPI } from '../api/client'
import { logger } from '../utils/logger'

const defaultConfig = {
  model: {
    path: 'models/yolo12s.pt',
    imgsz: 640,
    conf: 0.35,
    iou: 0.5,
    camera_fps: 30,
    inference_scale: 1.0,
    jpeg_quality: 80,
    process_interval: 2,
  },
  rules: {
    running: { enabled: true, speed_px_s: 50 },
    fall: { enabled: true, upright_aspect_min: 1.2 },
    crowd: { enabled: true, min_people: 3 },
    intrusion: {
      enabled: true,
      zones: [],
    },
    fight: { enabled: true },
  },
  output: { save_snapshots: true },
  mllm: {
    enabled: false,
    model_type: 'qwen2-vl-2b',
    inference_backend: 'mock',
    shadow_mode: true,
    key_frame_interval: 15,
    max_new_tokens: 256,
    scene_description_enabled: true,
    alarm_enhance_enabled: true,
    enhancement_cooldown_s: 10,
  },
}

const rulesList = {
  running: { label: '奔跑检测', detail: '基于速度分析的奔跑行为检测' },
  fall: { label: '摔倒检测', detail: '基于人体高宽比的摔倒行为检测' },
  crowd: { label: '人群检测', detail: '基于密集度分析的人群聚集检测' },
  intrusion: { label: '入侵检测', detail: '基于自定义区域的入侵检测' },
  fight: { label: '打架检测', detail: '基于多因素分析的打架行为检测' },
}

const config = reactive(JSON.parse(JSON.stringify(defaultConfig)))
const configFormRef = ref(null)
const polyCanvas = ref(null)
const selectedZoneIdx = ref(0)
const saving = ref(false)

const formRules = {
  'model.imgsz': [
    { required: true, message: '请输入输入尺寸', trigger: 'blur' },
    { type: 'number', min: 320, max: 1280, message: '范围 320-1280', trigger: 'blur' },
  ],
  'model.conf': [
    { required: true, message: '请设置置信度阈值', trigger: 'change' },
  ],
  'model.camera_fps': [
    { type: 'number', min: 5, max: 60, message: '范围 5-60', trigger: 'blur' },
  ],
  'model.jpeg_quality': [
    { type: 'number', min: 30, max: 95, message: '范围 30-95', trigger: 'blur' },
  ],
}

const inferenceScalePercent = computed({
  get: () => Math.round(config.model.inference_scale * 100),
  set: (v) => {
    config.model.inference_scale = v / 100
  },
})

const gpuInfo = ref(null)
const gpuRecommendations = ref([])

const fetchGPUInfo = async () => {
  try {
    const res = await fetch('/api/detection/gpu')
    const data = await res.json()
    gpuInfo.value = data
    gpuRecommendations.value = data.recommendations || []
  } catch (e) {
    logger.error('Failed to fetch GPU info:', e)
  }
}

// Model switching state
const availableModels = ref([])
const currentModel = ref('models/yolo12s.pt')
const modelSwitching = ref(false)
const modelSelectDialogVisible = ref(false)
const selectedNewModel = ref('')

onMounted(async () => {
  await fetchAvailableModels()
  fetchGPUInfo()
  await loadConfigFromBackend()
  loadSavedConfig()
})

const loadConfigFromBackend = async () => {
  try {
    const res = await fetch('/api/detection/config')
    const data = await res.json()
    if (data.status === 'ok' && data.config) {
      mergeBackendConfig(data.config)
    }
  } catch {
    // backend unavailable, fall back to localStorage
  }
}

const mergeBackendConfig = (backendCfg) => {
  const m = backendCfg.model || {}
  if (m.path) config.model.path = m.path
  if (m.imgsz) config.model.imgsz = m.imgsz
  if (m.conf) config.model.conf = m.conf
  if (m.iou) config.model.iou = m.iou

  const c = backendCfg.camera || {}
  if (c.fps) config.model.camera_fps = c.fps
  if (c.inference_scale) config.model.inference_scale = c.inference_scale
  if (c.jpeg_quality) config.model.jpeg_quality = c.jpeg_quality

  if (backendCfg.pose && backendCfg.pose.process_interval) {
    config.model.process_interval = backendCfg.pose.process_interval
  }

  if (backendCfg.rules) {
    Object.keys(backendCfg.rules).forEach((key) => {
      if (config.rules[key]) {
        Object.assign(config.rules[key], backendCfg.rules[key])
      }
    })
  }

  if (backendCfg.output) {
    if (backendCfg.output.save_snapshots !== undefined) {
      config.output.save_snapshots = backendCfg.output.save_snapshots
    }
  }

  if (backendCfg.mllm) {
    const ml = backendCfg.mllm
    if (ml.enabled !== undefined) config.mllm.enabled = ml.enabled
    if (ml.model_type) config.mllm.model_type = ml.model_type
    if (ml.inference_backend) config.mllm.inference_backend = ml.inference_backend
    if (ml.shadow_mode !== undefined) config.mllm.shadow_mode = ml.shadow_mode
    if (ml.key_frame_interval) config.mllm.key_frame_interval = ml.key_frame_interval
    if (ml.max_new_tokens) config.mllm.max_new_tokens = ml.max_new_tokens
    if (ml.scene_description_enabled !== undefined) {
      config.mllm.scene_description_enabled = ml.scene_description_enabled
    }
    if (ml.alarm_enhance_enabled !== undefined) {
      config.mllm.alarm_enhance_enabled = ml.alarm_enhance_enabled
    }
    if (ml.enhancement_cooldown_s) {
      config.mllm.enhancement_cooldown_s = ml.enhancement_cooldown_s
    }
  }
}

const loadSavedConfig = () => {
  try {
    const saved = localStorage.getItem('yolo-course-config')
    if (saved) {
      const parsed = JSON.parse(saved)
      Object.assign(config, parsed)
    }
  } catch {
    // ignore corrupted localStorage
  }
}

const fetchAvailableModels = async () => {
  try {
    const res = await fetch('/api/detection/models')
    const data = await res.json()
    if (data.available && data.available.length > 0) {
      availableModels.value = data.available
      if (data.current) {
        currentModel.value = data.current
      } else {
        currentModel.value = config.model.path
      }
    } else {
      availableModels.value = [
        { name: 'yolo12s.pt', path: 'models/yolo12s.pt', size_mb: 18.5 },
        { name: 'yolov8n.pt', path: 'models/yolov8n.pt', size_mb: 6.2 },
      ]
    }
  } catch {
    availableModels.value = [
      { name: 'yolo12s.pt', path: 'models/yolo12s.pt', size_mb: 18.5 },
      { name: 'yolov8n.pt', path: 'models/yolov8n.pt', size_mb: 6.2 },
    ]
  }
}

const openModelSelector = () => {
  selectedNewModel.value = currentModel.value
  modelSelectDialogVisible.value = true
}

const confirmModelSwitch = async () => {
  if (!selectedNewModel.value || selectedNewModel.value === currentModel.value) {
    modelSelectDialogVisible.value = false
    return
  }

  try {
    modelSwitching.value = true
    await ElMessageBox.confirm(
      `确定要切换模型吗？\n将从「${currentModel.value}」切换到「${selectedNewModel.value}」\n\n切换后系统将重新加载模型，这可能会短暂中断检测服务。`,
      '模型切换确认',
      {
        confirmButtonText: '确定切换',
        cancelButtonText: '取消',
        type: 'warning',
      }
    )
    await switchToModel(selectedNewModel.value)
  } catch {
    ElMessage.info('已取消模型切换')
  } finally {
    modelSwitching.value = false
  }
}

const switchToModel = async (modelPath) => {
  try {
    modelSwitching.value = true
    const res = await fetch('/api/detection/models/switch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_path: modelPath, reload_pipeline: true }),
    })
    const data = await res.json()

    if (data.status === 'success') {
      currentModel.value = modelPath
      config.model.path = modelPath
      localStorage.setItem('yolo-course-config', JSON.stringify(config))
      if (data.runtime_switch) {
        ElMessage.success(`模型切换成功：${modelPath}`)
      } else {
        ElMessage.success(`模型已更新：${modelPath}，请启动检测以加载新模型`)
      }
      modelSelectDialogVisible.value = false
    } else {
      ElMessage.error(`模型切换失败：${data.message || '未知错误'}`)
    }
  } catch (e) {
    ElMessage.error(`模型切换失败：${e.message || '网络错误'}`)
  } finally {
    modelSwitching.value = false
  }
}

const addZone = () => {
  config.rules.intrusion.zones.push({
    name: `zone${config.rules.intrusion.zones.length + 1}`,
    polygon: [],
  })
  selectedZoneIdx.value = config.rules.intrusion.zones.length - 1
}

const removeZone = (idx) => {
  config.rules.intrusion.zones.splice(idx, 1)
  if (selectedZoneIdx.value >= config.rules.intrusion.zones.length) {
    selectedZoneIdx.value = Math.max(0, config.rules.intrusion.zones.length - 1)
  }
  drawPolygons()
}

const clearAllZones = () => {
  ElMessageBox.confirm('确定要清除所有入侵区域吗？此操作不可恢复。', '清除确认', {
    confirmButtonText: '清除',
    cancelButtonText: '取消',
    type: 'warning',
  })
    .then(() => {
      config.rules.intrusion.zones.splice(0, config.rules.intrusion.zones.length)
      selectedZoneIdx.value = 0
      drawPolygons()
      ElMessage.success('已清除所有入侵区域')
    })
    .catch(() => {})
}

const onCanvasClick = (e) => {
  const zones = config.rules.intrusion.zones
  if (!zones.length) return
  const rect = polyCanvas.value.getBoundingClientRect()
  const x = Math.round(((e.clientX - rect.left) / rect.width) * 640)
  const y = Math.round(((e.clientY - rect.top) / rect.height) * 360)
  const zi = Math.min(selectedZoneIdx.value, zones.length - 1)
  zones[zi].polygon.push([x, y])
  drawPolygons()
}

const onCanvasRightClick = () => {
  const zones = config.rules.intrusion.zones
  if (!zones.length) return
  const zi = Math.min(selectedZoneIdx.value, zones.length - 1)
  if (zones[zi].polygon.length > 0) {
    zones[zi].polygon.pop()
    drawPolygons()
  }
}

const drawPolygons = () => {
  const canvas = polyCanvas.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, 640, 360)

  config.rules.intrusion.zones.forEach((zone, zi) => {
    const pts = zone.polygon
    if (pts.length < 2) return

    ctx.beginPath()
    ctx.strokeStyle = zi === selectedZoneIdx.value ? '#ff4444' : '#4488ff'
    ctx.lineWidth = 2
    ctx.moveTo(pts[0][0], pts[0][1])
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i][0], pts[i][1])
    }
    if (pts.length >= 3) {
      ctx.closePath()
      ctx.fillStyle =
        zi === selectedZoneIdx.value ? 'rgba(255,68,68,0.15)' : 'rgba(68,136,255,0.15)'
      ctx.fill()
    }
    ctx.stroke()

    pts.forEach((p, pi) => {
      ctx.beginPath()
      ctx.arc(p[0], p[1], 4, 0, Math.PI * 2)
      ctx.fillStyle = zi === selectedZoneIdx.value ? '#ff4444' : '#4488ff'
      ctx.fill()
      ctx.fillStyle = '#fff'
      ctx.font = '11px sans-serif'
      ctx.fillText(String(pi + 1), p[0] + 6, p[1] + 4)
    })

    if (pts.length >= 2) {
      ctx.fillStyle = '#fff'
      ctx.font = '13px sans-serif'
      ctx.fillText(zone.name, pts[0][0], Math.max(10, pts[0][1] - 8))
    }
  })
}

watch(
  () => config.rules.intrusion.zones,
  () => {
    nextTick(drawPolygons)
  },
  { deep: true }
)

const saveConfig = async () => {
  if (configFormRef.value) {
    const valid = await configFormRef.value.validate().catch(() => false)
    if (!valid) {
      ElMessage.warning('请检查表单中的错误项')
      return
    }
  }
  saving.value = true
  const snap = JSON.parse(JSON.stringify(config))
  const payload = {
    model: snap.model,
    rules: snap.rules,
    output: snap.output,
    mllm: snap.mllm,
    camera: {
      fps: snap.model.camera_fps || 30,
      url: '',
      buffer_size: 30,
    },
    alarm: {
      enabled: true,
      cooldown_s: 30,
      sound_enabled: true,
    },
  }
  try {
    await detectionAPI.saveConfig(payload)
    ElMessage.success('配置已保存并生效')
  } catch {
    ElMessage.warning('配置已保存至本地缓存（后端未运行，启动后生效）')
  } finally {
    localStorage.setItem('yolo-course-config', JSON.stringify(config))
    mllmAPI.enable(snap.mllm.enabled, snap.mllm.shadow_mode).catch(() => {})
    saving.value = false
  }
}

const resetConfig = () => {
  Object.assign(config, JSON.parse(JSON.stringify(defaultConfig)))
  localStorage.setItem('yolo-course-config', JSON.stringify(config))
  ElMessage.info('已恢复默认配置')
}
</script>

<style scoped>
.config-container {
  max-width: 960px;
  margin: 0 auto;
}

.config-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.config-title {
  font-weight: 600;
  font-size: 16px;
}

.config-actions {
  display: flex;
  gap: 8px;
}

.config-form {
  max-width: 100%;
}

.form-tip {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 12px;
  line-height: 1.5;
}

.rule-card {
  margin-bottom: 12px;
  transition: all var(--transition-fast);
}

.rule-card:hover {
  transform: translateY(-1px);
}

.rule-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.rule-name {
  font-weight: 600;
  font-size: 14px;
}

.rule-detail {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--border-color);
}

.rule-detail-text {
  font-size: 13px;
  color: var(--text-secondary);
}

.zone-editor {
  display: flex;
  gap: 16px;
  width: 100%;
  flex-wrap: wrap;
}

.zone-list {
  width: 200px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.zone-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background var(--transition-fast);
}

.zone-item:hover {
  background: #f5f7fa;
}

.zone-item.active {
  background: #ecf5ff;
}

.zone-actions {
  display: flex;
  gap: 8px;
  margin-top: 4px;
}

.zone-empty {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  color: #999;
  font-size: 13px;
  background: #f9f9f9;
  border-radius: 4px;
  margin-top: 8px;
}

.gpu-status-card {
  padding: 16px;
  background: #f5f7fa;
  border-radius: var(--radius-md);
  margin-bottom: 8px;
  overflow: hidden;
}

.gpu-status-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.gpu-name {
  font-size: 14px;
  font-weight: 600;
  color: #303133;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 300px;
}

.gpu-details {
  margin-top: 12px;
}

.gpu-detail-row {
  display: flex;
  flex-wrap: wrap;
  gap: 24px;
  margin-bottom: 10px;
}

.gpu-detail-row:last-child {
  margin-bottom: 0;
}

.gpu-detail-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  min-width: 140px;
}

.gpu-detail-label {
  color: #909399;
  min-width: 60px;
  text-align: right;
}

.gpu-detail-value {
  color: #303133;
  font-weight: 500;
}

.gpu-recommendations {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #ebeef5;
}

.gpu-rec-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: #606266;
  margin-bottom: 6px;
}

.canvas-wrapper {
  flex: 1;
  min-width: 300px;
}

.poly-canvas {
  width: 100%;
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  cursor: crosshair;
  background: #1a1a1a;
}

.canvas-tip {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 6px;
}

.fade-enter-active,
.fade-leave-active {
  transition:
    opacity 0.2s ease,
    max-height 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  max-height: 0;
}

.current-model-display {
  display: flex;
  align-items: center;
  gap: 8px;
}

.model-select-content {
  padding: 0 4px;
}

.model-option {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.model-name {
  font-weight: 500;
}

.model-size {
  font-size: 12px;
  color: var(--text-secondary);
  margin-left: 12px;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

@media (max-width: 1024px) {
  .config-form :deep(.el-form-item__label) {
    width: 110px !important;
  }
}

@media (max-width: 768px) {
  .config-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }

  .config-form :deep(.el-form-item__label) {
    width: 100% !important;
    text-align: left;
  }

  .config-form :deep(.el-form-item) {
    flex-direction: column;
    align-items: stretch;
  }

  .zone-editor {
    flex-direction: column;
  }

  .zone-list {
    width: 100%;
  }

  .canvas-wrapper {
    min-width: 100%;
  }

  .gpu-detail-item {
    min-width: 120px;
  }
}

@media (max-width: 480px) {
  .config-container {
    max-width: 100%;
  }

  .gpu-detail-row {
    gap: 12px;
  }

  .model-dialog :deep(.el-dialog) {
    width: 95% !important;
    margin: 16px auto;
  }
}
</style>
