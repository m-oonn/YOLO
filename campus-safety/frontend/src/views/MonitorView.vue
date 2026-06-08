<!--
  ┌──────────────────────────────────────────┐
  │ 【前端UI】MonitorView.vue — ★ 实时监控页面  │
  │ 路由：/monitor                             │
  │ 职责：最核心的前端页面 ——                   │
  │   ① 选择检测源（摄像头/视频文件）           │
  │   ② 控制检测启停（开始/停止按钮）           │
  │   ③ 播放MJPEG实时视频流                    │
  │   ④ 显示运行指标（FPS/帧数/事件数）         │
  │   ⑤ 实时事件列表（WebSocket推送）           │
  │ 使用组件：StreamPlayer, StatCard            │
  └──────────────────────────────────────────┘
  Copyright (c) 2025 YOLO Course Design Contributors
  SPDX-License-Identifier: Apache-2.0
-->

<template>
  <div class="monitor-container">
    <div class="sr-only" aria-live="polite" role="status">
      {{ detectionRunning ? '检测运行中' : detectionLoading ? '检测启动中' : '检测已停止' }}
    </div>

    <!-- Control bar -->
    <PageHeader title="实时监控" subtitle="摄像头检测、视频文件分析、实时画面">
      <template #actions>
        <el-select
          v-model="source"
          placeholder="选择检测源"
          class="source-select"
          @change="onSourceChange"
        >
          <el-option-group label="摄像头">
            <el-option
              v-for="cam in cameras"
              :key="cam.id"
              :label="cam.name"
              :value="String(cam.id)"
            />
          </el-option-group>
          <el-option-group label="已上传视频">
            <el-option
              v-for="f in uploadedFiles"
              :key="f.path"
              :label="f.name"
              :value="f.path"
            />
            <el-option label="+ 上传新视频" value="file" />
          </el-option-group>
        </el-select>
        <el-button
          type="success"
          :disabled="detectionRunning || detectionLoading || switchingMode"
          :icon="VideoPlay"
          :loading="starting"
          @click="startDetection"
        >
          启动
        </el-button>
        <el-button
          type="danger"
          :disabled="!detectionRunning || switchingMode"
          :icon="VideoPause"
          @click="stopDetection"
        >
          停止
        </el-button>
      </template>
    </PageHeader>

    <!-- Startup progress -->
    <div v-if="detectionLoading && startupProgress.step !== 'idle'" class="startup-progress">
      <div class="progress-header">
        <el-icon class="progress-icon"><Loading /></el-icon>
        <span class="progress-message">{{ startupProgress.message }}</span>
        <span class="progress-percent">{{ startupProgress.percent }}%</span>
      </div>
      <el-progress
        :percentage="startupProgress.percent"
        :status="startupProgress.step === 'error' ? 'exception' : ''"
        :stroke-width="8"
        :show-text="false"
        class="progress-bar"
      />
    </div>

    <!-- Status bar -->
    <div class="status-bar">
      <div class="status-section">
        <div class="status-group">
          <StatusBadge
            :status="detectionRunning ? 'online' : detectionLoading ? 'loading' : 'offline'"
            :label="detectionRunning ? '运行中' : detectionLoading ? '启动中...' : '已停止'"
            :animated="detectionRunning"
          />
          <div v-if="detectionRunning" class="metric">
            <span class="metric-label">FPS</span>
            <span class="metric-value" :class="fpsClass">{{ fps }}</span>
          </div>
          <div class="metric">
            <span class="metric-label">帧数</span>
            <span class="metric-value">{{ formatNumber(frameCount) }}</span>
          </div>
          <div class="metric">
            <span class="metric-label">事件</span>
            <span class="metric-value">{{ formatNumber(eventCount) }}</span>
          </div>
        </div>
        <div class="status-group">
          <div v-if="detectionRunning && perfData" class="metric" :title="perfTooltip">
            <span class="metric-label">推理</span>
            <span class="metric-value">{{ perfData.inference_ms }}ms</span>
          </div>
          <div v-if="gpuStatus && gpuStatus.gpu_available" class="gpu-metric" :title="perfTooltip">
            <span class="metric-label">显存</span>
            <div class="gpu-bar-wrap">
              <el-progress
                :percentage="gpuUsagePct"
                :stroke-width="6"
                :show-text="false"
                :color="gpuBarColor"
                class="gpu-bar"
              />
              <span class="metric-value gpu-pct">{{ gpuUsagePct }}%</span>
            </div>
          </div>
          <StatusBadge
            v-if="perfData"
            :status="perfData.device === 'cuda:0' ? 'online' : perfData.device === 'mps' ? 'warning' : 'offline'"
            :label="perfData.device === 'cuda:0' ? 'GPU' : perfData.device === 'mps' ? 'MPS' : 'CPU'"
          />
          <el-tag
            v-if="uploadedFile"
            size="small"
            closable
            @close="clearUploadedFile"
            class="file-tag"
          >
            {{ uploadedFile.name }}
          </el-tag>
        </div>
      </div>
    </div>

    <!-- Upload dialog -->
    <el-dialog
      v-model="showUpload"
      title="上传视频文件"
      width="420px"
      :close-on-click-modal="false"
      class="upload-dialog"
    >
      <el-upload
        drag
        accept="video/*,.mp4,.avi,.mov,.mkv"
        :auto-upload="false"
        :show-file-list="true"
        :on-change="onFileChange"
        :limit="10"
      >
        <div class="upload-area">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--color-primary)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
          <div class="upload-text">拖拽或点击选择视频文件</div>
        </div>
        <template #tip>
          <div class="upload-tip">支持 MP4, AVI, MOV, MKV 格式，最大 100MB</div>
        </template>
      </el-upload>
      <!-- Upload progress -->
      <div v-if="uploadProgress > 0 && uploadProgress < 100" class="upload-progress">
        <div class="upload-progress-label">上传中 {{ uploadProgress }}%</div>
        <el-progress :percentage="uploadProgress" :stroke-width="6" :show-text="false" />
      </div>
      <template #footer>
        <el-button @click="showUpload = false; selectedFile = null">取消</el-button>
        <el-button
          type="primary"
          :disabled="!selectedFile || uploadProgress > 0 || uploading"
          :loading="uploading"
          @click="confirmUpload"
        >
          上传并启动检测
        </el-button>
      </template>
    </el-dialog>

    <!-- Stream player -->
    <StreamPlayer
      :stream-url="streamUrl"
      :is-running="!!streamUrl && (detectionRunning || detectionLoading)"
      placeholder-text="请选择检测源并启动检测"
      @error="onStreamError"
      @retry="retryStream"
    />

    <!-- MLLM Panel -->
    <el-card v-if="mllmData && mllmData.stats" shadow="never" class="mllm-card">
      <template #header>
        <div class="mllm-header">
          <div class="header-title-group">
            <span class="mllm-title">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 6px; vertical-align: -2px;">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
              </svg>
              MLLM 场景理解
            </span>
          </div>
          <StatusBadge
            :status="mllmData.stats.enabled && mllmData.stats.running ? 'online' : mllmData.stats.enabled ? 'warning' : 'offline'"
            :label="mllmData.stats.enabled && mllmData.stats.running ? '运行中' : mllmData.stats.enabled ? '已配置' : '未启用'"
          />
          <el-switch
            v-if="mllmData"
            :model-value="mllmData.stats.enabled"
            size="small"
            active-text="开"
            inactive-text="关"
            @change="toggleMLLM"
            class="mllm-toggle"
          />
        </div>
      </template>
      <div v-if="mllmData.stats.enabled" class="mllm-content">
        <div v-if="!mllmData.stats.running" class="mllm-disabled">
          <span class="stat-text">MLLM已配置，将在检测启动后运行</span>
        </div>
        <div v-if="mllmData.stats.running" class="mllm-stats-row">
          <div class="mllm-stat">
            <span class="mllm-stat-label">场景描述</span>
            <span class="mllm-stat-value">{{ mllmData.stats.scenes_described || 0 }}次</span>
          </div>
          <div class="mllm-stat">
            <span class="mllm-stat-label">告警验证</span>
            <span class="mllm-stat-value">{{ mllmData.stats.alarms_enhanced || 0 }}次</span>
          </div>
          <div class="mllm-stat">
            <span class="mllm-stat-label">模型</span>
            <span class="mllm-stat-value">{{ mllmData.stats.model_loaded ? 'Qwen2-VL-2B' : '加载中' }}</span>
          </div>
          <div class="mllm-stat">
            <span class="mllm-stat-label">推理后端</span>
            <span class="mllm-stat-value">{{ mllmData.stats.engine?.backend || 'none' }}</span>
          </div>
        </div>
        <div
          v-if="mllmData.stats.running && mllmData.stats.last_scene && mllmData.stats.last_scene.scene_summary"
          class="mllm-scene"
        >
          <el-tag
            :type="
              mllmData.stats.last_scene.risk_level === '高'
                ? 'danger'
                : mllmData.stats.last_scene.risk_level === '中'
                  ? 'warning'
                  : 'success'
            "
            size="small"
            class="scene-tag"
          >
            {{ mllmData.stats.last_scene.activity_type }}
          </el-tag>
          <span class="mllm-summary">{{ mllmData.stats.last_scene.scene_summary }}</span>
          <p v-if="mllmData.stats.last_scene.narrative" class="mllm-narrative">{{ mllmData.stats.last_scene.narrative }}</p>
        </div>
      </div>
      <div v-else class="mllm-disabled">
        <span class="stat-text">MLLM场景理解未启用，可在系统配置中开启</span>
      </div>
    </el-card>

    <!-- Events Panel -->
    <el-card shadow="never" class="events-card">
      <template #header>
        <div class="events-header">
          <span class="events-title">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 6px; vertical-align: -2px;">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
              <line x1="16" y1="13" x2="8" y2="13"/>
              <line x1="16" y1="17" x2="8" y2="17"/>
              <polyline points="10 9 9 9 8 9"/>
            </svg>
            实时事件
          </span>
          <el-button size="small" text @click="loadEvents">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px; vertical-align: -2px;">
              <polyline points="23 4 23 10 17 10"/>
              <polyline points="1 20 1 14 7 14"/>
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
            </svg>
            刷新
          </el-button>
        </div>
      </template>
      <el-table :data="recentEvents" size="small" max-height="200" stripe :show-header="true" class="events-table">
        <el-table-column prop="event_type" label="类型" width="110">
          <template #default="{ row }">
            <el-tag :type="eventTypeColor(row.event_type)" size="small" effect="dark" class="event-tag">
              {{ eventTypeLabel(row.event_type) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="description" label="描述" min-width="200" show-overflow-tooltip />
        <el-table-column prop="timestamp_s" label="时间" width="120">
          <template #default="{ row }">
            {{ formatTime(row.timestamp_s) }}
          </template>
        </el-table-column>
      </el-table>
      <div v-if="!recentEvents.length" class="empty-events">
        <el-empty description="暂无事件" :image-size="60" />
      </div>
    </el-card>
  </div>
</template>

<script setup>
/**
 * MonitorView.vue - Real-time detection monitoring page.
 */
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Upload, Refresh, VideoPlay, VideoPause, Loading } from '@element-plus/icons-vue'
import { camerasAPI, detectionAPI, eventsAPI, mllmAPI } from '../api/client'
import { logger } from '../utils/logger'
import { useWebSocket } from '../composables/useWebSocket'
import { eventTypeColor, formatTime, eventTypeLabel } from '../utils/helpers'
import PageHeader from '../components/PageHeader.vue'
import StreamPlayer from '../components/StreamPlayer.vue'
import StatusBadge from '../components/StatusBadge.vue'
import ActionButton from '../components/ActionButton.vue'

const source = ref('0')
const cameras = ref([])
const detectionRunning = ref(false)
const detectionLoading = ref(false)
const switchingMode = ref(false)
const streamUrl = ref('')
const fps = ref(0)
const frameCount = ref(0)
const eventCount = ref(0)
const recentEvents = ref([])
const showUpload = ref(false)
const uploadedFile = ref(null)
const uploadedFiles = ref([])
const uploading = ref(false)
const selectedFile = ref(null)
const starting = ref(false)
const startupProgress = ref({ step: 'idle', message: '', percent: 0 })
const uploadProgress = ref(0)
let progressInterval = null
const _streamCounter = ref(0)
const perfData = ref(null)
const gpuStatus = ref(null)
const mllmData = ref(null)
const POLL_INTERVAL = 3000

const { wsConnected, connect: wsConnect, disconnect: wsDisconnect } = useWebSocket()

let statusInterval = null
const route = useRoute()

const formatNumber = (n) => {
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M'
  if (n >= 1000) return (n / 1000).toFixed(1) + 'K'
  return String(n)
}

const perfTooltip = computed(() => {
  if (!perfData.value) return ''
  const p = perfData.value
  const gpuInfo = gpuStatus.value
  let tip = `推理: ${p.inference_ms}ms | 骨架: ${p.skeleton_ms}ms | 规则: ${p.rules_ms}ms | 标注: ${p.annotate_ms}ms | 编码: ${p.encode_ms}ms | 总计: ${p.total_ms}ms\n设备: ${p.device || 'cpu'} | 半精度: ${p.half_precision ? '是' : '否'} | GPU预处理: ${p.gpu_preprocess ? '是' : '否'} | 缩放: ${p.inference_scale} | 跳帧: ${p.frame_skip_interval}`
  if (gpuInfo && gpuInfo.gpu_available) {
    tip += `\nGPU: ${gpuInfo.gpu_name} | 显存: ${gpuInfo.gpu_used_memory_mb}/${gpuInfo.gpu_total_memory_mb}MB | 利用率: ${gpuInfo.gpu_utilization_pct}% | 温度: ${gpuInfo.gpu_temperature_c}°C`
  }
  return tip
})

const fpsClass = computed(() => {
  if (fps.value >= 25) return 'fps-good'
  if (fps.value >= 15) return 'fps-warn'
  return 'fps-low'
})

const gpuUsagePct = computed(() => {
  const g = gpuStatus.value
  if (!g || !g.gpu_available) return 0
  if (g.gpu_memory_usage_pct > 0) return Math.round(g.gpu_memory_usage_pct)
  if (g.gpu_total_memory_mb > 0) {
    return Math.round((g.gpu_used_memory_mb / g.gpu_total_memory_mb) * 100)
  }
  return 0
})

const gpuBarColor = computed(() => {
  const pct = gpuUsagePct.value
  if (pct >= 85) return '#ef4444'
  if (pct >= 65) return '#f59e0b'
  return '#22c55e'
})

let mjpegRetryTimer = null
let mjpegRetryCount = 0
const MAX_RETRY_COUNT = 5
const BASE_RETRY_DELAY = 1000

const retryStream = () => {
  if (detectionRunning.value || detectionLoading.value) {
    _streamCounter.value++
    streamUrl.value = `/api/detection/stream.mjpg?_=${_streamCounter.value}&t=${Date.now()}`
  }
}

const scheduleRetry = () => {
  if (mjpegRetryTimer) {
    clearTimeout(mjpegRetryTimer)
  }
  if (mjpegRetryCount >= MAX_RETRY_COUNT) {
    ElMessage.error('视频流重试次数已达上限，请检查后端服务或手动刷新页面')
    mjpegRetryCount = 0
    return
  }
  const delay = BASE_RETRY_DELAY * Math.pow(2, mjpegRetryCount)
  mjpegRetryCount++
  mjpegRetryTimer = setTimeout(() => {
    if (detectionRunning.value || detectionLoading.value) {
      _streamCounter.value++
      streamUrl.value = `/api/detection/stream.mjpg?_=${_streamCounter.value}&t=${Date.now()}`
    }
    mjpegRetryTimer = null
  }, delay)
}

const onSourceChange = async (val) => {
  if (val === 'file') {
    if (detectionRunning.value) {
      await stopDetection()
    }
    showUpload.value = true
    return
  }
  uploadedFile.value = null
  selectedFile.value = null
  if (detectionRunning.value) {
    await stopDetection()
  }
}

const onFileChange = (file) => {
  selectedFile.value = file.raw
}

const clearUploadedFile = async () => {
  if (detectionRunning.value) {
    await stopDetection()
  }
  uploadedFile.value = null
  selectedFile.value = null
  if (source.value !== '0') source.value = '0'
}

const confirmUpload = async () => {
  if (!selectedFile.value) {
    ElMessage.warning('请选择视频文件')
    return
  }
  uploading.value = true
  uploadProgress.value = 0
  try {
    const res = await detectionAPI.uploadVideo(selectedFile.value, (percent) => {
      uploadProgress.value = percent
    })
    if (res.status === 'error') {
      ElMessage.error('上传失败: ' + (res.message || '服务器拒绝该文件'))
      uploadProgress.value = 0
      uploading.value = false
      return
    }
    const fileInfo = { name: selectedFile.value.name, path: res.path }
    source.value = res.path
    uploadedFile.value = fileInfo
    if (!uploadedFiles.value.find(f => f.path === res.path)) {
      uploadedFiles.value.push(fileInfo)
    }
    showUpload.value = false
    selectedFile.value = null
    uploadProgress.value = 0
    uploading.value = false
    ElMessage.success('上传成功，正在启动检测...')
    await startDetection()
  } catch (e) {
    uploadProgress.value = 0
    uploading.value = false
    const msg = e?.response?.data?.message || e?.message || '上传失败，请检查网络和文件大小（最大 100MB）'
    ElMessage.error(msg)
  }
}

const startDetection = async () => {
  if (switchingMode.value) return
  
  let loadingMsg = null
  
  try {
    if (detectionRunning.value) {
      switchingMode.value = true
      ElMessage.info('正在切换模式...')
      logger.log('Stopping current detection...')
      const stopRes = await detectionAPI.stop()
      if (stopRes.status !== 'stopped') {
        const errorMsg = '停止当前检测失败：' + (stopRes.message || '未知错误')
        logger.error('Stop detection failed:', stopRes)
        ElMessage.error(errorMsg)
        switchingMode.value = false
        return
      }
      streamUrl.value = ''
      logger.log('Current detection stopped successfully')
      await new Promise(resolve => setTimeout(resolve, 300))
      detectionRunning.value = false
    }
    
    starting.value = true
    const src = uploadedFile.value ? uploadedFile.value.path : source.value
    logger.log('Starting detection on source:', src)
    
    loadingMsg = ElMessage({
      message: '正在启动检测，请稍候...',
      type: 'info',
      duration: 0,
      showClose: false,
    })
    
    const res = await detectionAPI.start(src)
    
    if (res.status === 'started') {
      detectionLoading.value = true
      startupProgress.value = { step: 'init', message: '正在初始化检测...', percent: 5 }
      _streamCounter.value++
      streamUrl.value = `/api/detection/stream.mjpg?_=${_streamCounter.value}&t=${Date.now()}`
      connectWebSocket()
      
      if (loadingMsg) loadingMsg.close()
      logger.log('Detection started, waiting for pipeline to be ready...', src)
      _startProgressPolling()
      _verifyStartup(src)
    } else {
      if (loadingMsg) loadingMsg.close()
      const errorMsg = '启动检测失败：' + (res.message || '后端返回错误')
      logger.error('Start detection failed:', res)
      ElMessage.error(errorMsg)
    }
  } catch (e) {
    if (loadingMsg) loadingMsg.close()
    logger.error('Failed to start detection:', e)
    const errorDetail = e.response?.data?.detail || '请检查后端服务'
    const cameraKeywords = ['摄像头', '相机', 'camera', 'video source', '视频源', '无法打开']
    const isCameraError = cameraKeywords.some(kw =>
      errorDetail.toLowerCase().includes(kw.toLowerCase())
    )
    ElMessage.error({
      message: isCameraError
        ? '摄像头初始化失败：' + errorDetail + '\n请检查：1.摄像头连接 2.Windows隐私设置 3.设备管理器'
        : '启动检测失败：' + errorDetail,
      duration: isCameraError ? 10000 : 5000,
    })
  } finally {
    starting.value = false
    switchingMode.value = false
    logger.log('Start detection process completed')
  }
}

const _startProgressPolling = () => {
  if (progressInterval) clearInterval(progressInterval)
  progressInterval = setInterval(async () => {
    if (!detectionLoading.value) {
      clearInterval(progressInterval)
      progressInterval = null
      return
    }
    try {
      const progress = await detectionAPI.progress()
      if (progress && progress.step) {
        startupProgress.value = progress
      }
    } catch (e) {
      // ignore progress fetch errors
    }
  }, 1200)
}

const _stopProgressPolling = () => {
  if (progressInterval) {
    clearInterval(progressInterval)
    progressInterval = null
  }
  startupProgress.value = { step: 'idle', message: '', percent: 0 }
}

const onStreamError = async (e) => {
  logger.error('Stream error:', e)
  if (detectionRunning.value || detectionLoading.value) {
    try {
      const status = await detectionAPI.status()
      if (!status.running) {
        logger.log('Stream error but detection already stopped, not retrying')
        detectionRunning.value = false
        detectionLoading.value = false
        streamUrl.value = ''
        mjpegRetryCount = 0
        return
      }
    } catch (_) {}

    scheduleRetry()
    ElMessage.warning('视频流中断，正在尝试恢复...')
  }
}

const stopDetection = async () => {
  try {
    logger.log('Stopping detection...')
    const res = await detectionAPI.stop()
    if (res.status === 'stopped') {
      detectionRunning.value = false
      detectionLoading.value = false
      streamUrl.value = ''
      uploadedFile.value = null
      selectedFile.value = null
      _stopProgressPolling()
      disconnectWebSocket()
      ElMessage.info('检测已停止')
      logger.log('Detection stopped successfully')
    } else {
      logger.error('Stop detection failed:', res)
      ElMessage.error('停止检测失败：' + (res.message || '未知错误'))
    }
  } catch (e) {
    logger.error('Failed to stop detection:', e)
    ElMessage.error('停止检测失败：' + (e.message || '网络错误'))
  }
}

const handleWebSocketMessage = (msg) => {
  switch (msg.type) {
    case 'status':
      if (msg.data) {
        const status = msg.data
        if ((detectionLoading.value || detectionRunning.value) && !status.running) {
          if (status.state === 'loading' || status.state === 'loading_timeout') {
          } else {
            if (status.last_error) {
              ElMessage.error('检测异常停止: ' + status.last_error)
            } else if (uploadedFile.value) {
              ElMessage.success('视频检测已完成，共处理 ' + (status.frame_count || 0) + ' 帧')
            }
            detectionLoading.value = false
            detectionRunning.value = false
            uploadedFile.value = null
            selectedFile.value = null
            streamUrl.value = ''
            _stopProgressPolling()
          }
        }
        if (status.running && !status.state) {
          detectionLoading.value = false
          detectionRunning.value = true
          _stopProgressPolling()
        } else if (status.running && status.state === 'loading') {
          detectionLoading.value = true
          detectionRunning.value = false
        } else {
          detectionRunning.value = status.running
        }
        fps.value = status.fps || 0
        frameCount.value = status.frame_count || 0
      }
      break

    case 'performance':
      if (msg.data) {
        perfData.value = msg.data
      }
      break

    case 'gpu':
      if (msg.data) {
        gpuStatus.value = msg.data
      }
      break

    case 'events':
      if (msg.data && Array.isArray(msg.data.items)) {
        recentEvents.value = msg.data.items
        eventCount.value = msg.data.total || 0
      }
      break

    case 'mllm':
      if (msg.data) {
        mllmData.value = msg.data
      }
      break

    default:
      logger.log('Unknown WebSocket message type:', msg.type)
  }
}

const connectWebSocket = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${protocol}//${window.location.host}/api/detection/stream`
  wsConnect(wsUrl, handleWebSocketMessage)
}

const disconnectWebSocket = () => {
  if (mjpegRetryTimer) {
    clearTimeout(mjpegRetryTimer)
    mjpegRetryTimer = null
  }
  wsDisconnect()
}

const loadCameras = async () => {
  try {
    const res = await camerasAPI.list()
    cameras.value = res.devices
  } catch (e) {
    logger.error('Failed to load cameras:', e)
    cameras.value = [{ id: 0, name: 'Camera 0', available: true }]
  }
}

const loadEvents = async () => {
  try {
    const res = await eventsAPI.list({ limit: 20 })
    recentEvents.value = res.items || res.events || []
    eventCount.value = res.total || 0
  } catch (e) {
    logger.error('Failed to load events:', e)
  }
}

const fetchGPUStatus = async () => {
  try {
    const res = await fetch('/api/detection/gpu')
    gpuStatus.value = await res.json()
  } catch (e) {
    logger.error('Failed to fetch GPU status:', e)
  }
}

const fetchMLLMStatus = async () => {
  try {
    mllmData.value = await mllmAPI.status()
  } catch {
    // MLLM API may not be available
  }
}

const toggleMLLM = async (enabled) => {
  try {
    const res = await mllmAPI.enable(enabled, false)
    if (res.status === 'success' || res.status === 'ok') {
      ElMessage.success(enabled ? 'AI 场景理解已开启' : 'AI 场景理解已关闭')
      fetchMLLMStatus()
      fetchGPUStatus()
    } else {
      ElMessage.error(res.message || '切换 MLLM 状态失败')
    }
  } catch (e) {
    ElMessage.error('切换 MLLM 状态失败')
  }
}

const pollStatus = async () => {
  try {
    const status = await detectionAPI.status()
    if ((detectionLoading.value || detectionRunning.value) && !status.running) {
      if (status.state === 'loading' || status.state === 'loading_timeout') {
      } else {
        if (status.last_error) {
          ElMessage.error('检测异常停止: ' + status.last_error)
        } else if (uploadedFile.value) {
          ElMessage.success('视频检测已完成，共处理 ' + (status.frame_count || 0) + ' 帧')
        }
        detectionLoading.value = false
        detectionRunning.value = false
        uploadedFile.value = null
        selectedFile.value = null
        streamUrl.value = ''
        _stopProgressPolling()
      }
    } else if (status.running && status.warning) {
      ElMessage.error({
        message: '摄像头初始化超时: ' + (status.warning || '请检查摄像头连接和权限设置'),
        duration: 10000,
      })
      detectionLoading.value = false
      detectionRunning.value = false
      uploadedFile.value = null
      selectedFile.value = null
      streamUrl.value = ''
      _stopProgressPolling()
    } else if (status.running && !status.state) {
      detectionLoading.value = false
      detectionRunning.value = true
    } else if (status.running && status.state === 'loading') {
      detectionLoading.value = true
      detectionRunning.value = false
    } else {
      detectionRunning.value = status.running
    }
    fps.value = status.fps ?? 0
    frameCount.value = status.frame_count ?? 0
    if (status.performance) {
      perfData.value = status.performance
    }
  } catch (e) {
    logger.error('Status poll error:', e)
  }
}

let _startupTimer = null
function _verifyStartup(src, delayMs = 3000) {
  if (_startupTimer) clearTimeout(_startupTimer)
  _startupTimer = setTimeout(async () => {
    if (!detectionLoading.value && !detectionRunning.value) return
    try {
      const st = await detectionAPI.status()
      if (!st.running && st.last_error) {
        ElMessage.error('检测启动失败: ' + st.last_error)
        detectionLoading.value = false
        detectionRunning.value = false
        streamUrl.value = ''
        uploadedFile.value = null
        selectedFile.value = null
        _stopProgressPolling()
        return
      }
      if (st.running && (st.state === 'loading_timeout' || st.warning)) {
        ElMessage.error({
          message: '摄像头初始化超时: ' + (st.warning || '请检查摄像头连接和权限设置'),
          duration: 10000,
        })
        detectionLoading.value = false
        detectionRunning.value = false
        streamUrl.value = ''
        uploadedFile.value = null
        selectedFile.value = null
        _stopProgressPolling()
        return
      }
      if (st.running && st.last_error) {
        ElMessage.error('检测异常: ' + st.last_error)
        detectionLoading.value = false
        detectionRunning.value = false
        streamUrl.value = ''
        uploadedFile.value = null
        selectedFile.value = null
        _stopProgressPolling()
        return
      }
      if (!st.running && !st.last_error) {
        _startupTimer = setTimeout(async () => {
          if (!detectionRunning.value) return
          try {
            const st2 = await detectionAPI.status()
            if (st2.running && (st2.state === 'loading_timeout' || st2.warning)) {
              ElMessage.error({
                message: '摄像头初始化超时: ' + (st2.warning || '请检查摄像头连接和权限设置'),
                duration: 10000,
              })
              detectionRunning.value = false
              streamUrl.value = ''
              uploadedFile.value = null
              selectedFile.value = null
            } else if (!st2.running && st2.last_error) {
              ElMessage.error('检测启动失败: ' + st2.last_error)
              detectionRunning.value = false
              streamUrl.value = ''
              uploadedFile.value = null
              selectedFile.value = null
            }
          } catch { /* ignore */ }
        }, 5000)
      }
    } catch { /* 网络错误忽略，由轮询兜底 */ }
  }, delayMs)
}

onMounted(async () => {
  await pollStatus()
  await loadCameras()

  if ((detectionRunning.value || detectionLoading.value) && !streamUrl.value) {
    _streamCounter.value++
    streamUrl.value = `/api/detection/stream.mjpg?_=${_streamCounter.value}&t=${Date.now()}`
  }

  if (route.query.source && route.query.autoStart === '1') {
    const src = route.query.source
    if (
      src.startsWith('uploads/') ||
      src.includes('/') ||
      src.endsWith('.mp4') ||
      src.endsWith('.avi') ||
      src.endsWith('.mov')
    ) {
      uploadedFile.value = { name: src.split('/').pop(), path: src }
      source.value = src
    } else {
      source.value = src
    }
    await startDetection()
  }

  connectWebSocket()

  let _pollTimer = null
  const _schedulePoll = () => {
    _pollTimer = setTimeout(() => {
      if (!document.hidden) {
        if (!wsConnected.value) {
          pollStatus()
          loadEvents()
          fetchMLLMStatus()
        } else {
          loadEvents()
        }
      }
      _schedulePoll()
    }, POLL_INTERVAL)
  }
  _schedulePoll()

  statusInterval = { cleanup: () => { if (_pollTimer) clearTimeout(_pollTimer) } }

  // Reset retry count on successful load
  const img = document.querySelector('.stream-img')
  if (img) {
    img.addEventListener('load', () => { mjpegRetryCount = 0 })
  }
})

onUnmounted(() => {
  if (statusInterval) {
    if (typeof statusInterval.cleanup === 'function') {
      statusInterval.cleanup()
    } else {
      clearInterval(statusInterval)
    }
  }
  if (_startupTimer) { clearTimeout(_startupTimer); _startupTimer = null }
  if (mjpegRetryTimer) {
    clearTimeout(mjpegRetryTimer)
    mjpegRetryTimer = null
  }
  disconnectWebSocket()
  if (detectionRunning.value || detectionLoading.value) {
    detectionAPI.stop().catch(() => {})
    detectionRunning.value = false
    detectionLoading.value = false
  }
})
</script>

<style scoped>
.monitor-container {
  display: flex;
  flex-direction: column;
  gap: 16px;
  max-width: 1400px;
  margin: 0 auto;
  position: relative;
  z-index: 1;
}

.source-select {
  width: 200px;
}

/* Status bar */
.status-bar {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 12px 16px;
  transition: all var(--transition-fast);
}

.status-bar:hover {
  border-color: var(--border-default);
}

.status-section {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.status-group {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.metric {
  display: flex;
  align-items: center;
  gap: 6px;
}

.metric-label {
  font-size: 11px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-weight: 500;
}

.metric-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.metric-value.fps-good {
  color: var(--color-success);
}

.metric-value.fps-warn {
  color: var(--color-warning);
}

.metric-value.fps-low {
  color: var(--color-danger);
}

.gpu-metric {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 120px;
}

.gpu-bar-wrap {
  display: flex;
  align-items: center;
  gap: 6px;
  flex: 1;
}

.gpu-bar {
  width: 72px;
}

.gpu-pct {
  min-width: 32px;
  font-size: 12px;
}

.file-tag {
  background: rgba(245, 158, 11, 0.1) !important;
  color: var(--color-warning) !important;
  border: 1px solid rgba(245, 158, 11, 0.2) !important;
}

/* Upload dialog */
.upload-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 20px;
}

.upload-text {
  font-size: 14px;
  color: var(--text-regular);
}

.upload-tip {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 4px;
}

/* MLLM card */
.mllm-card :deep(.el-card__header) {
  padding: 12px 16px;
}

.mllm-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.mllm-title {
  font-weight: 600;
  font-size: 14px;
  color: var(--text-primary);
  display: flex;
  align-items: center;
}

.mllm-toggle {
  margin-left: 8px;
}

.mllm-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.mllm-stats-row {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
  padding: 8px 0;
  border-bottom: 1px solid var(--border-subtle);
}

.mllm-stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.mllm-stat-label {
  font-size: 11px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.mllm-stat-value {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  font-family: 'JetBrains Mono', monospace;
}

.mllm-scene {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px;
  background: var(--bg-root);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-subtle);
}

.scene-tag {
  align-self: flex-start;
}

.mllm-summary {
  font-size: 13px;
  color: var(--text-regular);
  font-weight: 500;
}

.mllm-narrative {
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.7;
  margin: 0;
  padding: 8px 10px;
  background: var(--bg-elevated);
  border-radius: var(--radius-sm);
  border-left: 2px solid var(--color-primary);
}

.mllm-disabled {
  text-align: center;
  padding: 8px 0;
}

.stat-text {
  color: var(--text-secondary);
  font-size: 13px;
}

/* Events card */
.events-card :deep(.el-card__header) {
  padding: 12px 16px;
}

.events-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.events-title {
  font-weight: 600;
  font-size: 14px;
  color: var(--text-primary);
  display: flex;
  align-items: center;
}

.events-table :deep(.el-tag) {
  font-weight: 500;
}

.empty-events {
  padding: 8px 0;
}

/* Startup progress */
.startup-progress {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 12px 16px;
  margin-bottom: 4px;
  animation: fadeIn 0.3s ease;
}

.progress-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.progress-icon {
  color: var(--color-primary);
  animation: spin 1s linear infinite;
}

.progress-message {
  flex: 1;
  font-size: 13px;
  color: var(--text-primary);
  font-weight: 500;
}

.progress-percent {
  font-size: 12px;
  color: var(--text-secondary);
  font-family: 'JetBrains Mono', monospace;
}

.progress-bar :deep(.el-progress-bar__outer) {
  background-color: var(--bg-elevated);
  border-radius: var(--radius-sm);
}

.progress-bar :deep(.el-progress-bar__inner) {
  border-radius: var(--radius-sm);
  transition: width 0.3s ease;
}

/* Upload progress */
.upload-progress {
  margin-top: 12px;
  padding: 8px 12px;
  background: var(--bg-elevated);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-subtle);
}

.upload-progress-label {
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 4px;
  font-family: 'JetBrains Mono', monospace;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-4px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Responsive */
@media (max-width: 1024px) {
  .source-select {
    width: 180px;
  }
}

@media (max-width: 768px) {
  .source-select {
    width: 100%;
  }

  .status-section {
    flex-direction: column;
    align-items: flex-start;
  }

  .status-group {
    width: 100%;
  }
}

@media (max-width: 480px) {
  .monitor-container {
    gap: 10px;
  }
}
</style>
