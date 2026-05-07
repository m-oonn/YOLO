<!--
  Copyright (c) 2025 YOLO Course Design Contributors
  SPDX-License-Identifier: MIT
-->

<template>
  <div class="monitor-container">
    <div class="sr-only" aria-live="polite" role="status">
      {{ detectionRunning ? '检测运行中' : detectionLoading ? '检测启动中' : '检测已停止' }}
    </div>
    <el-card shadow="never" class="control-card">
      <div class="control-row">
        <div class="control-left">
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
            <el-option-group label="其他">
              <el-option label="视频文件" value="file" />
            </el-option-group>
          </el-select>
          <el-button
            type="success"
            :disabled="detectionRunning || switchingMode"
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
        </div>
        <div class="status-bar">
          <transition name="fade" mode="out-in">
            <el-tag
              :key="detectionRunning + '-' + detectionLoading"
              :type="detectionRunning ? 'success' : detectionLoading ? 'warning' : 'info'"
              size="small"
              effect="dark"
            >
              {{ detectionRunning ? '运行中' : detectionLoading ? '启动中...' : '已停止' }}
            </el-tag>
          </transition>
          <span
            v-if="detectionRunning"
            class="fps-badge"
            :class="{ 'fps-low': fps < 15, 'fps-warn': fps >= 15 && fps < 30 }"
          >
            FPS: {{ fps }}
          </span>
          <span class="stat-text">帧: {{ frameCount }}</span>
          <span class="stat-text">事件: {{ eventCount }}</span>
          <span
            v-if="detectionRunning && perfData"
            class="stat-text perf-text"
            :title="perfTooltip"
          >
            推理: {{ perfData.inference_ms }}ms
          </span>
          <span
            v-if="detectionRunning && perfData"
            class="stat-text perf-text"
            :title="`设备: ${perfData.device || 'cpu'} | 半精度: ${perfData.half_precision ? '是' : '否'}`"
          >
            <el-tag
              :type="
                perfData.device === 'cuda:0'
                  ? 'success'
                  : perfData.device === 'mps'
                    ? 'warning'
                    : 'danger'
              "
              size="small"
              effect="plain"
            >
              {{ perfData.device === 'cuda:0' ? 'GPU' : perfData.device === 'mps' ? 'MPS' : 'CPU' }}
            </el-tag>
          </span>
          <span v-if="detectionRunning && perfData && perfData.gpu_preprocess" class="stat-text">
            <el-tag type="success" size="small" effect="plain">GPU预处理</el-tag>
          </span>
          <span v-if="gpuStatus && gpuStatus.gpu_available" class="stat-text gpu-mem-stat">
            <el-tag
              :type="gpuMemTagType"
              size="small"
              effect="plain"
            >
              显存 {{ gpuStatus.gpu_total_used_mb || gpuStatus.gpu_used_memory_mb || 0 }}/{{ gpuStatus.gpu_total_memory_mb }}MB
              <span v-if="gpuStatus.gpu_memory_usage_pct"> ({{ gpuStatus.gpu_memory_usage_pct }}%)</span>
            </el-tag>
          </span>
          <el-tag
            v-if="uploadedFile"
            size="small"
            type="warning"
            closable
            @close="clearUploadedFile"
          >
            {{ uploadedFile.name }}
          </el-tag>
        </div>
      </div>
    </el-card>

    <el-dialog
      v-model="showUpload"
      title="上传视频文件"
      width="420px"
      :close-on-click-modal="false"
    >
      <el-upload
        drag
        accept="video/*,.mp4,.avi,.mov,.mkv"
        :auto-upload="false"
        :show-file-list="true"
        :on-change="onFileChange"
        :limit="1"
      >
        <el-icon :size="48" class="upload-icon"><Upload /></el-icon>
        <div class="upload-text">拖拽或点击选择视频文件</div>
        <template #tip>
          <div class="upload-tip">支持 MP4, AVI, MOV, MKV 格式，最大 100MB</div>
        </template>
      </el-upload>
      <template #footer>
        <el-button @click="showUpload = false">取消</el-button>
        <el-button
          type="primary"
          :loading="uploading"
          :disabled="!selectedFile"
          @click="confirmUpload"
        >
          上传并启动检测
        </el-button>
      </template>
    </el-dialog>

    <el-card shadow="never" class="video-card">
      <div class="video-container">
        <transition name="fade" mode="out-in">
          <div v-if="streamUrl" key="player" class="stream-wrapper">
            <img
              :src="streamUrl"
              alt="检测画面实时视频流"
              class="stream-image"
              loading="eager"
              decoding="async"
              @error="onStreamError"
            />
          </div>
          <div v-else key="placeholder" class="no-stream">
            <el-icon :size="56" class="no-stream-icon"><VideoCamera /></el-icon>
            <p>请选择检测源并启动检测</p>
            <p class="no-stream-sub">支持摄像头实时检测和视频文件分析</p>
          </div>
        </transition>
        <transition name="fade">
          <div v-if="detectionRunning" class="live-badge">
            <span class="live-dot"></span>
            LIVE
          </div>
        </transition>
      </div>
    </el-card>

    <el-card v-if="mllmData && mllmData.stats" shadow="never" class="mllm-card">
      <template #header>
        <div class="mllm-header">
          <span class="mllm-title">MLLM 场景理解</span>
          <el-tag
            :type="mllmData.stats.enabled && mllmData.stats.running ? 'success' : mllmData.stats.enabled ? 'warning' : 'info'"
            size="small"
          >
            {{ mllmData.stats.enabled && mllmData.stats.running ? '运行中' : mllmData.stats.enabled ? '已配置' : '未启用' }}
          </el-tag>
        </div>
      </template>
      <div v-if="mllmData.stats.enabled" class="mllm-content">
        <div v-if="!mllmData.stats.running" class="mllm-disabled">
          <span class="stat-text">MLLM已配置，将在检测启动后运行（shadow mode）</span>
        </div>
        <div v-if="mllmData.stats.running" class="mllm-stats-row">
          <span class="stat-text">场景描述: {{ mllmData.stats.scenes_described || 0 }}次</span>
          <span class="stat-text">告警验证: {{ mllmData.stats.alarms_enhanced || 0 }}次</span>
          <span class="stat-text">推理后端: {{ mllmData.stats.engine?.backend || 'none' }}</span>
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
          >
            {{ mllmData.stats.last_scene.activity_type }}
          </el-tag>
          <span class="mllm-summary">{{ mllmData.stats.last_scene.scene_summary }}</span>
        </div>
      </div>
      <div v-else class="mllm-disabled">
        <span class="stat-text">MLLM场景理解未启用，可在系统配置中开启</span>
      </div>
    </el-card>

    <el-card shadow="never" class="events-card">
      <template #header>
        <div class="events-header">
          <span class="events-title">实时事件</span>
          <el-button size="small" text @click="loadEvents">
            <el-icon><Refresh /></el-icon>
            刷新
          </el-button>
        </div>
      </template>
      <el-table :data="recentEvents" size="small" max-height="200" stripe :show-header="true">
        <el-table-column prop="event_type" label="类型" width="110">
          <template #default="{ row }">
            <el-tag :type="eventTypeColor(row.event_type)" size="small" effect="light">
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
 *
 * Provides the primary user interface for the YOLO detection system:
 * - Camera/video source selection and switching
 * - Start/stop detection with loading state feedback
 * - Live MJPEG video stream display with auto-reconnect
 * - Real-time FPS, frame count, and event statistics
 * - WebSocket-based status updates (detection state, GPU info)
 * - Video file upload for offline detection
 * - MLLM (Multi-modal LLM) scene description panel
 *
 * State machine:
 *   idle → loading (启动中) → running (运行中) → idle
 *                ↓ error
 *               idle
 */
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Upload, Refresh, VideoPlay, VideoPause, VideoCamera } from '@element-plus/icons-vue'
import { camerasAPI, detectionAPI, eventsAPI, mllmAPI } from '../api/client'
import { logger } from '../utils/logger'
import { useWebSocket } from '../composables/useWebSocket'
import { eventTypeColor, formatTime, eventTypeLabel } from '../utils/helpers'

const source = ref('0')
const cameras = ref([])
const detectionRunning = ref(false)
const detectionLoading = ref(false)  // 摄像头/模型加载中状态
const switchingMode = ref(false)
const streamUrl = ref('')
const fps = ref(0)
const frameCount = ref(0)
const eventCount = ref(0)
const recentEvents = ref([])
const showUpload = ref(false)
const uploadedFile = ref(null)
const uploading = ref(false)
const selectedFile = ref(null)
const starting = ref(false)
const _streamCounter = ref(0)
const perfData = ref(null)
const gpuStatus = ref(null)
const mllmData = ref(null)
const POLL_INTERVAL = 3000  // 优化：从5秒减少到3秒，更快响应状态变化

const { wsConnected, connect: wsConnect, disconnect: wsDisconnect } = useWebSocket()

let statusInterval = null
const route = useRoute()

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

const gpuMemTagType = computed(() => {
  if (!gpuStatus.value) return 'info'
  const pressure = gpuStatus.value.gpu_memory_pressure
  if (pressure === 'critical') return 'danger'
  if (pressure === 'high') return 'warning'
  if (pressure === 'medium') return 'info'
  return 'success'
})

let mjpegRetryTimer = null

const onSourceChange = async (val) => {
  if (val === 'file') {
    if (detectionRunning.value) {
      await stopDetection()
    }
    showUpload.value = true
    return
  }
  // Switching to a camera source: clear stale uploaded file state
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
  try {
    const res = await detectionAPI.uploadVideo(selectedFile.value)
    // 先更新 source 再更新 uploadedFile，防止 onSourceChange 清空 uploadedFile
    source.value = res.path
    uploadedFile.value = { name: selectedFile.value.name, path: res.path }
    showUpload.value = false
    ElMessage.success('文件上传成功')
    await startDetection()
  } catch (e) {
    ElMessage.error('文件上传失败：' + (e.response?.data?.detail || '请检查文件格式'))
    logger.error('Upload failed:', e)
  } finally {
    uploading.value = false
  }
}

const startDetection = async () => {
  if (switchingMode.value) return
  
  let loadingMsg = null
  
  try {
    if (detectionRunning.value) {
      // 切换模式：先停止当前检测
      switchingMode.value = true
      ElMessage.info('正在切换模式...')
      
      // 先清空流 URL，立即消除视觉残留
      streamUrl.value = ''
      
      // 停止当前检测
      logger.log('Stopping current detection...')
      const stopRes = await detectionAPI.stop()
      if (stopRes.status !== 'stopped') {
        const errorMsg = '停止当前检测失败：' + (stopRes.message || '未知错误')
        logger.error('Stop detection failed:', stopRes)
        ElMessage.error(errorMsg)
        switchingMode.value = false
        return
      }
      logger.log('Current detection stopped successfully')
      
      // 等待 300ms 确保资源释放（减少等待时间）
      await new Promise(resolve => setTimeout(resolve, 300))
      
      detectionRunning.value = false
    }
    
    starting.value = true
    const src = uploadedFile.value ? uploadedFile.value.path : source.value
    logger.log('Starting detection on source:', src)
    
    // 显示加载提示，让用户知道系统正在工作
    loadingMsg = ElMessage({
      message: '正在启动检测，请稍候...',
      type: 'info',
      duration: 0, // 不自动关闭
      showClose: false,
    })
    
    const res = await detectionAPI.start(src)
    
    if (res.status === 'started') {
      // 不立即设置为运行中，而是设置为加载中，等待后端确认
      detectionLoading.value = true
      _streamCounter.value++
      // 强制刷新流，添加随机参数避免缓存
      streamUrl.value = `/api/detection/stream.mjpg?_=${_streamCounter.value}&t=${Date.now()}`
      connectWebSocket()
      
      // 关闭加载提示
      if (loadingMsg) loadingMsg.close()
      logger.log('Detection started, waiting for pipeline to be ready...', src)

      // 延迟验证：确认管线真正启动成功（模型加载 + 摄像头打开需要时间）
      _verifyStartup(src)
    } else {
      // 关闭加载提示，显示错误消息
      if (loadingMsg) loadingMsg.close()
      const errorMsg = '启动检测失败：' + (res.message || '后端返回错误')
      logger.error('Start detection failed:', res)
      ElMessage.error(errorMsg)
    }
  } catch (e) {
    // 关闭加载提示（如果还在显示）
    if (loadingMsg) loadingMsg.close()
    logger.error('Failed to start detection:', e)
    const errorDetail = e.response?.data?.detail || '请检查后端服务'

    // Check if this might be a camera-related failure
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

const onStreamError = async (e) => {
  logger.error('Stream error:', e)
  if (detectionRunning.value) {
    // 重试前先验证检测是否仍在运行
    try {
      const status = await detectionAPI.status()
      if (!status.running) {
        logger.log('Stream error but detection already stopped, not retrying')
        detectionRunning.value = false
        streamUrl.value = ''
        return
      }
    } catch (_) {
      // 网络错误时暂不处理，继续重试
    }

    // 清除旧的重试定时器
    if (mjpegRetryTimer) {
      clearTimeout(mjpegRetryTimer)
    }
    // 延迟重试加载 MJPEG 流
    mjpegRetryTimer = setTimeout(() => {
      if (detectionRunning.value) {
        _streamCounter.value++
        streamUrl.value = `/api/detection/stream.mjpg?_=${_streamCounter.value}&t=${Date.now()}`
      }
      mjpegRetryTimer = null
    }, 1000)
    ElMessage.warning('视频流中断，正在尝试恢复...')
  }
}

const stopDetection = async () => {
  try {
    logger.log('Stopping detection...')
    const res = await detectionAPI.stop()
    if (res.status === 'stopped') {
      detectionRunning.value = false
      streamUrl.value = ''
      uploadedFile.value = null
      selectedFile.value = null
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
            // 仍在加载中或加载中超时，保持状态
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
          }
        }
        // 如果检测到真正运行（非loading状态）
        if (status.running && !status.state) {
          detectionLoading.value = false
          detectionRunning.value = true
        } else if (status.running && status.state === 'loading') {
          detectionLoading.value = true
          detectionRunning.value = false
        } else {
          // 后端返回的 running 状态直接同步
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

const pollStatus = async () => {
  try {
    const status = await detectionAPI.status()
    if ((detectionLoading.value || detectionRunning.value) && !status.running) {
      if (status.state === 'loading' || status.state === 'loading_timeout') {
        // 仍在加载中或加载中超时，保持状态
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
      }
    } else if (status.running && status.warning) {
      // 摄像头启动超时
      ElMessage.error({
        message: '摄像头初始化超时: ' + (status.warning || '请检查摄像头连接和权限设置'),
        duration: 10000,
      })
      detectionLoading.value = false
      detectionRunning.value = false
      uploadedFile.value = null
      selectedFile.value = null
      streamUrl.value = ''
    } else if (status.running && !status.state) {
      // 管线已就绪，真正运行
      detectionLoading.value = false
      detectionRunning.value = true
    } else if (status.running && status.state === 'loading') {
      // 仍在加载中
      detectionLoading.value = true
      detectionRunning.value = false
    } else {
      // 后端返回的 running 状态直接同步
      detectionRunning.value = status.running
    }
    fps.value = status.fps
    frameCount.value = status.frame_count
    if (status.performance) {
      perfData.value = status.performance
    }
  } catch (e) {
    logger.error('Status poll error:', e)
  }
}

// 启动后延迟验证：避免"返回 started 但线程实际启动失败"的静默假死
let _startupTimer = null
function _verifyStartup(src, delayMs = 3000) {
  if (_startupTimer) clearTimeout(_startupTimer)
  _startupTimer = setTimeout(async () => {
    if (!detectionLoading.value && !detectionRunning.value) return // 用户已手动停止
    try {
      const st = await detectionAPI.status()
      if (!st.running && st.last_error) {
        // 启动失败：有错误信息
        ElMessage.error('检测启动失败: ' + st.last_error)
        detectionLoading.value = false
        detectionRunning.value = false
        streamUrl.value = ''
        uploadedFile.value = null
        selectedFile.value = null
        return
      }
      // Detect stuck "loading" state (camera hung but thread alive)
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
        return
      }
      if (st.running && st.last_error) {
        ElMessage.error('检测异常: ' + st.last_error)
        detectionLoading.value = false
        detectionRunning.value = false
        streamUrl.value = ''
        uploadedFile.value = null
        selectedFile.value = null
        return
      }
      // 如果还没 running 但也没 error（模型还在加载），再等一会
      if (!st.running && !st.last_error) {
        _startupTimer = setTimeout(async () => {
          if (!detectionRunning.value) return
          try {
            const st2 = await detectionAPI.status()
            // Check for loading_timeout on second attempt
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

  // 如果检测仍在运行或加载中（页面切换回来时），恢复 MJPEG 流
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

  // 优化：使用 requestAnimationFrame 调度，避免后台标签页浪费资源
  let _pollTimer = null
  const _schedulePoll = () => {
    _pollTimer = setTimeout(() => {
      // 页面可见时才执行轮询
      if (!document.hidden) {
        pollStatus()
        if (!wsConnected.value) {
          loadEvents()
          fetchMLLMStatus()
        }
      }
      _schedulePoll()
    }, POLL_INTERVAL)
  }
  _schedulePoll()
  
  // 保存清理函数
  statusInterval = { cleanup: () => { if (_pollTimer) clearTimeout(_pollTimer) } }
})

onUnmounted(() => {
  // 优化：兼容新的定时器清理方式
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
}

.control-card {
  flex-shrink: 0;
}

.control-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}

.control-left {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.source-select {
  width: 200px;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.fps-badge {
  color: var(--primary-color);
  font-weight: 700;
  font-size: 14px;
  font-variant-numeric: tabular-nums;
}

.fps-badge.fps-warn {
  color: #e6a23c;
}

.fps-badge.fps-low {
  color: #f56c6c;
}

.stat-text {
  color: var(--text-secondary);
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}

.perf-text {
  cursor: help;
  border-bottom: 1px dashed var(--text-secondary);
}

.video-card {
  flex: 1;
  min-height: 0;
}

.video-container {
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #0d0d0d;
  border-radius: var(--radius-md);
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}

.stream-wrapper {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.stream-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.no-stream {
  text-align: center;
  color: #666;
}

.no-stream-icon {
  color: #444;
  margin-bottom: 12px;
}

.no-stream p {
  font-size: 15px;
  margin-top: 4px;
}

.no-stream-sub {
  font-size: 13px;
  color: #555;
  margin-top: 4px !important;
}

.live-badge {
  position: absolute;
  top: 12px;
  left: 12px;
  background: rgba(255, 0, 0, 0.85);
  color: white;
  padding: 3px 10px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 1px;
  display: flex;
  align-items: center;
  gap: 6px;
  backdrop-filter: blur(4px);
}

.live-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: white;
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.3;
  }
}

.gpu-mem-stat {
  cursor: help;
}

.events-card {
  flex-shrink: 0;
}

.mllm-card {
  flex-shrink: 0;
}

.mllm-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.mllm-title {
  font-weight: 600;
  font-size: 14px;
}

.mllm-content {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.mllm-stats-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.mllm-scene {
  display: flex;
  align-items: center;
  gap: 8px;
}

.mllm-summary {
  font-size: 13px;
  color: var(--text-secondary);
}

.mllm-disabled {
  text-align: center;
  padding: 8px 0;
}

.events-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.events-title {
  font-weight: 600;
  font-size: 15px;
}

.empty-events {
  padding: 8px 0;
}

.upload-icon {
  color: var(--primary-color);
}

.upload-text {
  margin-top: 8px;
  font-size: 14px;
  color: var(--text-regular);
}

.upload-tip {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 4px;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* Responsive: Tablet */
@media (max-width: 1024px) {
  .source-select {
    width: 180px;
  }

  .video-container {
    aspect-ratio: 16 / 10;
  }
}

/* Responsive: Mobile */
@media (max-width: 768px) {
  .control-row {
    flex-direction: column;
    align-items: flex-start;
  }

  .source-select {
    width: 100%;
  }

  .control-left {
    width: 100%;
  }

  .control-left .el-button {
    flex: 1;
  }

  .status-bar {
    width: 100%;
    justify-content: flex-start;
  }

  .video-container {
    aspect-ratio: 4 / 3;
  }

  .stat-text {
    font-size: 12px;
  }
}

@media (max-width: 480px) {
  .monitor-container {
    gap: 10px;
  }

  .fps-badge {
    font-size: 12px;
  }

  .live-badge {
    top: 8px;
    left: 8px;
    font-size: 11px;
    padding: 2px 8px;
  }
}
</style>
