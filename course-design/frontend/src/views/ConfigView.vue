<template>
  <div class="config-container">
    <el-card shadow="never">
      <template #header>
        <span>系统配置</span>
      </template>

      <el-form label-width="140px" label-position="left">
        <el-divider content-position="left">模型配置</el-divider>

        <el-form-item label="模型路径">
          <el-input v-model="config.model.path" placeholder="models/yolov11x.pt" />
          <div class="form-tip">支持 .pt, .onnx, .engine 格式</div>
        </el-form-item>

        <el-form-item label="输入尺寸">
          <el-input-number v-model="config.model.imgsz" :min="320" :max="1280" :step="32" />
        </el-form-item>

        <el-form-item label="置信度阈值">
          <el-slider
            v-model="config.model.conf"
            :min="0.1"
            :max="0.9"
            :step="0.05"
            style="width: 300px"
          />
        </el-form-item>

        <el-form-item label="IoU 阈值">
          <el-slider
            v-model="config.model.iou"
            :min="0.1"
            :max="0.9"
            :step="0.05"
            style="width: 300px"
          />
        </el-form-item>

        <el-divider content-position="left">检测规则</el-divider>

        <el-form-item label="奔跑检测">
          <el-switch v-model="config.rules.running.enabled" />
          <span v-if="config.rules.running.enabled" class="rule-detail">
            速度阈值: {{ config.rules.running.speed_px_s }} px/s
          </span>
        </el-form-item>

        <el-form-item label="摔倒检测">
          <el-switch v-model="config.rules.fall.enabled" />
          <span v-if="config.rules.fall.enabled" class="rule-detail">
            高宽比阈值: {{ config.rules.fall.upright_aspect_min }}
          </span>
        </el-form-item>

        <el-form-item label="人群检测">
          <el-switch v-model="config.rules.crowd.enabled" />
          <span v-if="config.rules.crowd.enabled" class="rule-detail">
            最少人数: {{ config.rules.crowd.min_people }}
          </span>
        </el-form-item>

        <el-form-item label="入侵检测">
          <el-switch v-model="config.rules.intrusion.enabled" />
        </el-form-item>

        <el-form-item label="打架检测">
          <el-switch v-model="config.rules.fight.enabled" />
        </el-form-item>

        <el-divider content-position="left">入侵区域编辑</el-divider>

        <el-form-item label="监控区域">
          <div class="zone-editor">
            <div class="zone-list">
              <div v-for="(zone, zi) in config.rules.intrusion.zones" :key="zi" class="zone-item">
                <el-input
                  v-model="zone.name"
                  size="small"
                  style="width: 120px"
                  placeholder="区域名称"
                />
                <el-tag size="small">{{ zone.polygon.length }} 个顶点</el-tag>
                <el-button type="danger" size="small" @click="removeZone(zi)">删除</el-button>
              </div>
              <el-button size="small" type="primary" class="add-zone-btn" @click="addZone">
                + 添加区域
              </el-button>
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
              <div class="canvas-tip">左键添加顶点 | 右键撤销 | 选区域后编辑</div>
            </div>
          </div>
        </el-form-item>

        <el-divider content-position="left">输出配置</el-divider>

        <el-form-item label="保存截图">
          <el-switch v-model="config.output.save_snapshots" />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="saveConfig">保存配置</el-button>
          <el-button @click="resetConfig">恢复默认</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup>
import { reactive, ref, watch, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { detectionAPI } from '../api/client'

const defaultConfig = {
  model: { path: 'models/yolov11x.pt', imgsz: 640, conf: 0.35, iou: 0.5 },
  rules: {
    running: { enabled: true, speed_px_s: 50 },
    fall: { enabled: true, upright_aspect_min: 1.2 },
    crowd: { enabled: true, min_people: 3 },
    intrusion: {
      enabled: true,
      zones: [
        {
          name: 'zone1',
          polygon: [
            [60, 60],
            [580, 60],
            [580, 340],
            [60, 340],
          ],
        },
      ],
    },
    fight: { enabled: true },
  },
  output: { save_snapshots: true },
}

const config = reactive(JSON.parse(JSON.stringify(defaultConfig)))
const polyCanvas = ref(null)
const selectedZoneIdx = ref(0)

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

    // Draw lines
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

    // Draw vertices
    pts.forEach((p, pi) => {
      ctx.beginPath()
      ctx.arc(p[0], p[1], 4, 0, Math.PI * 2)
      ctx.fillStyle = zi === selectedZoneIdx.value ? '#ff4444' : '#4488ff'
      ctx.fill()
      ctx.fillStyle = '#fff'
      ctx.font = '11px sans-serif'
      ctx.fillText(String(pi + 1), p[0] + 6, p[1] + 4)
    })

    // Draw zone name
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
  try {
    await detectionAPI.saveConfig(JSON.parse(JSON.stringify(config)))
    ElMessage.success('配置已保存并生效')
  } catch {
    ElMessage.warning('配置已保存至本地缓存（后端未运行，启动后生效）')
    localStorage.setItem('yolo-course-config', JSON.stringify(config))
  }
}

const resetConfig = () => {
  Object.assign(config, JSON.parse(JSON.stringify(defaultConfig)))
  ElMessage.info('已恢复默认配置')
}
</script>

<style scoped>
.config-container {
  max-width: 900px;
}
.form-tip {
  font-size: 12px;
  color: #999;
  margin-top: 4px;
}
.rule-detail {
  margin-left: 12px;
  font-size: 13px;
  color: #666;
}
.zone-editor {
  display: flex;
  gap: 16px;
  width: 100%;
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
  gap: 8px;
}
.add-zone-btn {
  margin-top: 4px;
}
.canvas-wrapper {
  flex: 1;
}
.poly-canvas {
  width: 100%;
  border: 1px solid #ddd;
  border-radius: 6px;
  cursor: crosshair;
  background: #1a1a1a;
}
.canvas-tip {
  font-size: 12px;
  color: #999;
  margin-top: 4px;
}
</style>
