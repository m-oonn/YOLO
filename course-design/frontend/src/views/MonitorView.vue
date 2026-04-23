<template>
  <div class="monitor-container">
    <!-- Control Panel -->
    <el-card shadow="never" class="control-card">
      <el-row :gutter="20" align="middle">
        <el-col :span="6">
          <el-form-item label="检测源" label-width="80px">
            <el-select v-model="source" placeholder="选择摄像头或文件" @change="onSourceChange">
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
          </el-form-item>
        </el-col>
        <el-col :span="4">
          <el-button
            type="success"
            @click="startDetection"
            :disabled="detectionRunning"
            :icon="VideoPlay"
          >
            启动检测
          </el-button>
        </el-col>
        <el-col :span="4">
          <el-button
            type="danger"
            @click="stopDetection"
            :disabled="!detectionRunning"
            :icon="VideoPause"
          >
            停止
          </el-button>
        </el-col>
        <el-col :span="10">
          <div class="status-bar">
            <el-tag :type="detectionRunning ? 'success' : 'info'" size="small">
              {{ detectionRunning ? '运行中' : '已停止' }}
            </el-tag>
            <span class="fps-text">FPS: {{ fps }}</span>
            <span class="stat-text">帧数: {{ frameCount }}</span>
            <span class="stat-text">事件: {{ eventCount }}</span>
            <span class="stat-text" v-if="uploadedFile">{{ uploadedFile.name }}</span>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <!-- File Upload Dialog -->
    <el-dialog v-model="showUpload" title="上传视频文件" width="400px">
      <el-upload
        drag
        accept="video/*,.mp4,.avi,.mov,.mkv"
        :auto-upload="false"
        :show-file-list="true"
        :on-change="onFileChange"
        :limit="1"
      >
        <el-icon :size="48"><Upload /></el-icon>
        <div class="upload-text">拖拽或点击选择视频文件</div>
        <template #tip>
          <div class="upload-tip">支持 MP4, AVI, MOV, MKV 格式</div>
        </template>
      </el-upload>
      <template #footer>
        <el-button @click="showUpload = false">取消</el-button>
        <el-button type="primary" @click="confirmUpload" :loading="uploading">
          上传并启动检测
        </el-button>
      </template>
    </el-dialog>

    <!-- Video Stream -->
    <el-card shadow="never" class="video-card">
      <div class="video-container">
        <img
          v-if="streamUrl"
          :src="streamUrl"
          alt="Detection Stream"
          class="stream-image"
        />
        <div v-else class="no-stream">
          <el-icon :size="48"><VideoCamera /></el-icon>
          <p>请选择检测源并启动检测</p>
        </div>
      </div>
    </el-card>

    <!-- Recent Events -->
    <el-card shadow="never" class="events-card">
      <template #header>
        <div class="events-header">
          <span>实时事件</span>
          <el-button size="small" @click="loadEvents">刷新</el-button>
        </div>
      </template>
      <el-table :data="recentEvents" size="small" max-height="200" stripe>
        <el-table-column prop="event_type" label="类型" width="100">
          <template #default="{ row }">
            <el-tag :type="eventTypeColor(row.event_type)" size="small">
              {{ row.event_type }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="description" label="描述" min-width="200" />
        <el-table-column prop="timestamp_s" label="时间" width="180">
          <template #default="{ row }">
            {{ formatTime(row.timestamp_s) }}
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Upload } from '@element-plus/icons-vue'
import { camerasAPI, detectionAPI, eventsAPI } from '../api/client'
import { VideoPlay, VideoPause, VideoCamera } from '@element-plus/icons-vue'

const source = ref('0')
const cameras = ref([])
const detectionRunning = ref(false)
const streamUrl = ref('')
const fps = ref(0)
const frameCount = ref(0)
const eventCount = ref(0)
const recentEvents = ref([])
const showUpload = ref(false)
const uploadedFile = ref(null)
const uploading = ref(false)
const selectedFile = ref(null)

let statusInterval = null
let ws = null

const onSourceChange = (val) => {
  if (val === 'file') {
    showUpload.value = true
  }
}

const onFileChange = (file) => {
  selectedFile.value = file.raw
}

const confirmUpload = async () => {
  if (!selectedFile.value) {
    ElMessage.warning('请选择视频文件')
    return
  }
  uploading.value = true
  try {
    const res = await detectionAPI.uploadVideo(selectedFile.value)
    uploadedFile.value = { name: selectedFile.value.name, path: res.path }
    source.value = res.path
    showUpload.value = false
    ElMessage.success('文件上传成功')
    await startDetection()
  } catch (e) {
    ElMessage.error('文件上传失败')
    console.error('Upload failed:', e)
  } finally {
    uploading.value = false
  }
}

const startDetection = async () => {
  try {
    const src = uploadedFile.value ? uploadedFile.value.path : source.value
    const res = await detectionAPI.start(src)
    if (res.status === 'started') {
      detectionRunning.value = true
      streamUrl.value = `/api/detection/stream.mjpg?_=${Date.now()}`
      connectWebSocket()
    }
  } catch (e) {
    console.error('Failed to start detection:', e)
    ElMessage.error('启动检测失败')
  }
}

const stopDetection = async () => {
  try {
    await detectionAPI.stop()
    detectionRunning.value = false
    streamUrl.value = ''
    disconnectWebSocket()
  } catch (e) {
    console.error('Failed to stop detection:', e)
  }
}

const connectWebSocket = () => {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
  ws = new WebSocket(`${protocol}//${location.host}/api/detection/stream`)
  ws.onopen = () => ws.send('subscribe')
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      if (data.type === 'status') {
        fps.value = data.data.fps || 0
        frameCount.value = data.data.frame_count || 0
        eventCount.value = data.data.events?.total_events || eventCount.value
      }
    } catch (e) {
      // ignore non-JSON messages
    }
  }
  ws.onerror = () => { /* will fall back to polling */ }
}

const disconnectWebSocket = () => {
  if (ws) {
    ws.close()
    ws = null
  }
}

const loadCameras = async () => {
  try {
    const res = await camerasAPI.list()
    cameras.value = res.devices
  } catch (e) {
    console.error('Failed to load cameras:', e)
    cameras.value = [{ id: 0, name: 'Camera 0', available: true }]
  }
}

const loadEvents = async () => {
  try {
    const res = await eventsAPI.list({ limit: 20 })
    recentEvents.value = res.events
    eventCount.value = res.total
  } catch (e) {
    console.error('Failed to load events:', e)
  }
}

const pollStatus = async () => {
  try {
    const status = await detectionAPI.status()
    detectionRunning.value = status.running
    fps.value = status.fps
    frameCount.value = status.frame_count
  } catch (e) {
    console.error('Status poll error:', e)
  }
}

const eventTypeColor = (type) => {
  const colors = { running: 'warning', fall: 'danger', crowd: 'info', intrusion: 'danger', fight: 'danger' }
  return colors[type] || 'info'
}

const formatTime = (ts) => {
  if (!ts) return ''
  return new Date(ts * 1000).toLocaleTimeString()
}

onMounted(() => {
  loadCameras()
  loadEvents()
  statusInterval = setInterval(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      pollStatus()
    }
    loadEvents()
  }, 3000)
})

onUnmounted(() => {
  if (statusInterval) clearInterval(statusInterval)
  disconnectWebSocket()
})
</script>

<style scoped>
.monitor-container { display: flex; flex-direction: column; gap: 16px; }
.control-card { flex-shrink: 0; }
.status-bar { display: flex; align-items: center; gap: 16px; }
.fps-text { color: #409eff; font-weight: 600; font-size: 14px; }
.stat-text { color: #666; font-size: 13px; }
.video-card { flex: 1; }
.video-container {
  width: 100%;
  aspect-ratio: 16/9;
  background: #1a1a1a;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}
.stream-image { width: 100%; height: 100%; object-fit: contain; }
.no-stream { text-align: center; color: #999; }
.no-stream p { margin-top: 8px; font-size: 14px; }
.events-card { flex-shrink: 0; }
.events-header { display: flex; justify-content: space-between; align-items: center; }
.upload-text { margin-top: 8px; font-size: 14px; color: #666; }
.upload-tip { font-size: 12px; color: #999; margin-top: 4px; }
</style>
