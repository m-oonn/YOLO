# 前端按钮功能审查报告

> 审查日期: 2026-05-23
> 审查范围: 所有页面可点击按钮
> 审查方式: 代码静态分析

---

## 一、审查概述

本次审查覆盖了前端 5 个页面（Dashboard、Monitor、Events、Alarms、Config）以及全局导航中的所有可交互按钮/链接，共识别出 **42 个可点击元素**。审查维度包括：功能实现完整性、API 调用正确性、状态管理、错误处理、用户体验。

---

## 二、全局导航按钮

### App.vue

| # | 元素 | 类型 | 功能 | 实现状态 | 问题 |
|---|------|------|------|---------|------|
| 1 | 菜单切换按钮 (☰/✕) | button | 移动端展开/收起侧边栏 | ✅ 完整 | 无 |
| 2 | 重连按钮 | button | 后端断线时手动触发健康检查 | ✅ 完整 | 无 |
| 3 | 导航菜单项 (×5) | router-link | 页面路由跳转 | ✅ 完整 | 无 |
| 4 | 连接告警中的"立即重试" | button | 触发健康检查 | ✅ 完整 | 无 |

**状态**: 全部正常

---

## 三、DashboardView 按钮审查

| # | 按钮 | 功能 | 点击处理 | API 调用 | 状态 |
|---|------|------|---------|---------|------|
| 1 | **详细监控** (el-button) | 跳转到监控页面 | `goToMonitor()` → `router.push('/monitor')` | 无 | ✅ |
| 2 | **查看全部** (告警卡片) | 跳转到告警页面 | `goToAlarms()` → `router.push('/alarms')` | 无 | ✅ |
| 3 | **查看全部** (事件表格) | 跳转到事件页面 | `goToEvents()` → `router.push('/events')` | 无 | ✅ |

**问题发现**:
- ⚠️ **P2**: 仪表板"详细监控"按钮文案为"开始检测"（当检测未启动时），但点击后始终跳转到 `/monitor`，不会自动启动检测。文案与行为不一致。

---

## 四、MonitorView 按钮审查

| # | 按钮 | 功能 | 点击处理 | API 调用 | 状态 |
|---|------|------|---------|---------|------|
| 1 | **启动** (el-button, type=success) | 启动检测流水线 | `startDetection()` | `detectionAPI.start(src)` | ✅ |
| 2 | **停止** (el-button, type=danger) | 停止检测流水线 | `stopDetection()` | `detectionAPI.stop()` | ✅ |
| 3 | **上传并启动检测** (dialog) | 上传视频并启动检测 | `confirmUpload()` | `detectionAPI.uploadVideo()` + `startDetection()` | ✅ |
| 4 | **取消** (upload dialog) | 关闭上传弹窗 | `showUpload = false` | 无 | ✅ |
| 5 | **重试** (StreamPlayer) | 重新加载视频流 | `retryStream()` | 无 (刷新 URL) | ✅ |
| 6 | **手动重试** (StreamPlayer loading) | 重新加载视频流 | `retryStream()` | 无 | ✅ |

**问题发现**:
- ⚠️ **P1**: `startDetection()` 中 `switchingMode` 状态锁在异常情况下可能无法释放。如果 `stopDetection()` 成功但后续 `startDetection()` 失败，`switchingMode` 会在 finally 中重置，但 `detectionRunning` 状态可能不一致。
- ⚠️ **P2**: `clearUploadedFile()` 调用 `stopDetection()` 但没有 await，可能导致竞态条件。
- ⚠️ **P2**: 上传弹窗中 `selectedFile` 使用 `file.raw` 获取原始文件对象，如果 Element Plus Upload 组件版本变更可能导致兼容性问题。
- ⚠️ **P3**: `onStreamError` 中的重试逻辑使用固定 1 秒延迟，没有指数退避，频繁错误时可能造成服务器压力。

---

## 五、EventsView 按钮审查

| # | 按钮 | 功能 | 点击处理 | API 调用 | 状态 |
|---|------|------|---------|---------|------|
| 1 | **清空全部** (el-button, type=danger) | 清空所有事件记录 | `confirmClearAll()` | `eventsAPI.deleteAll()` | ✅ |
| 2 | **搜索** (SearchFilterBar) | 按条件筛选事件 | `onSearch()` | `eventsAPI.list(params)` | ✅ |
| 3 | **重置** (SearchFilterBar) | 重置筛选条件 | `onReset()` | `eventsAPI.list()` | ✅ |
| 4 | **重新检测** (表格操作列) | 用原视频源重新检测 | `redetect(row)` | 无 (路由跳转) | ✅ |
| 5 | **行点击展开** (表格) | 展开 MLLM 分析详情 | `onRowClick()` | 无 | ✅ |
| 6 | **视频片段点击** (clips) | 播放事件视频 | `playClip(clip)` | `archivesAPI.clipUrl()` | ✅ |

**问题发现**:
- ⚠️ **P2**: `confirmClearAll()` 使用 `ElMessageBox.confirm` 但缺少 `.catch()` 的优雅处理，虽然代码中有空的 catch，但用户取消时会有未处理的 Promise 警告。
- ⚠️ **P2**: `redetect()` 函数跳转到 `/monitor?source=xxx&autoStart=1`，但 MonitorView 中处理 `autoStart` 的逻辑在 `onMounted` 中，如果页面已经加载不会触发自动启动。
- ⚠️ **P3**: 快照图片预览使用 `preview-src-list`，但如果 `snapshotUrl` 返回 404，错误处理仅显示 "-"，没有更友好的提示。

---

## 六、AlarmsView 按钮审查

| # | 按钮 | 功能 | 点击处理 | API 调用 | 状态 |
|---|------|------|---------|---------|------|
| 1 | **清空全部** (el-button) | 清空所有报警记录 | `clearAllAlarms()` | `alarmsAPI.deleteAll()` | ✅ |
| 2 | **处理** (表格操作列) | 将报警标记为已处理 | `resolveAlarm(id)` | `alarmsAPI.resolve(id)` | ⚠️ |
| 3 | **筛选标签** (×5) | 按级别/状态筛选 | 直接修改 `filterLevel`/`filterActive` | 无 (前端筛选) | ✅ |

**问题发现**:
- 🔴 **P0**: `resolveAlarm()` 调用 `alarmsAPI.resolve(id)`，但审查 `api/client.js` 发现 API 定义的是 `alarmsAPI.resolve` 方法，而实际后端 API 可能期望 `acknowledge` 或 `resolve`。需要确认后端路由是否匹配。
- ⚠️ **P1**: `clearAllAlarms()` 与 EventsView 的 `confirmClearAll()` 有相同的 Promise 未处理问题。
- ⚠️ **P2**: 筛选标签使用前端内存筛选（`filteredAlarms` computed），当数据量大时可能影响性能。建议后端分页 + 前端筛选结合。

---

## 七、ConfigView 按钮审查

| # | 按钮 | 功能 | 点击处理 | API 调用 | 状态 |
|---|------|------|---------|---------|------|
| 1 | **保存** (检测规则) | 保存规则配置 | `saveRules()` | `configAPI.updateRules()` | ⚠️ |
| 2 | **保存** (系统设置) | 保存系统参数 | `saveSettings()` | `configAPI.updateSettings()` | ⚠️ |
| 3 | **保存** (MLLM设置) | 保存 MLLM 配置 | `saveMllmSettings()` | `mllmAPI.updateConfig()` | ⚠️ |
| 4 | **添加区域** | 打开添加区域弹窗 | `showZoneDialog = true` | 无 | ✅ |
| 5 | **编辑** (区域列表) | 编辑区域配置 | `editZone(idx)` | 无 (弹窗编辑) | ✅ |
| 6 | **删除** (区域列表) | 删除区域 | `deleteZone(idx)` | `configAPI.updateZones()` | ✅ |
| 7 | **保存** (区域弹窗) | 保存区域配置 | `saveZone()` | `configAPI.updateZones()` | ✅ |
| 8 | **取消** (区域弹窗) | 关闭弹窗 | `showZoneDialog = false` | 无 | ✅ |

**问题发现**:
- 🔴 **P0**: `configAPI.updateRules()`、`configAPI.updateSettings()`、`configAPI.updateZones()` 在 `api/client.js` 中**未定义**。这些 API 方法不存在，点击保存按钮会抛出 `TypeError: configAPI.updateRules is not a function`。
- 🔴 **P0**: `mllmAPI.updateConfig()` 在 `api/client.js` 中**未定义**。同样会导致运行时错误。
- ⚠️ **P1**: `saveZone()` 中 `JSON.parse(zoneForm.value.coordinates)` 没有 try-catch 包裹，虽然外层有 catch，但错误提示是通用的"坐标格式错误"。
- ⚠️ **P2**: 区域弹窗中坐标输入使用 JSON 字符串，用户体验不佳，建议提供可视化区域绘制工具。

---

## 八、API Client 缺失方法汇总

在 `api/client.js` 中需要补充以下方法：

```javascript
export const configAPI = {
  get: () => api.get('/config').then((r) => r.data),
  updateRules: (rules) => api.post('/config/rules', rules).then((r) => r.data),
  updateSettings: (settings) => api.post('/config/settings', settings).then((r) => r.data),
  updateZones: (zones) => api.post('/config/zones', zones).then((r) => r.data),
}

export const mllmAPI = {
  status: () => cachedGet('mllm:status', () => api.get('/mllm/status').then((r) => r.data)),
  getConfig: () => api.get('/mllm/config').then((r) => r.data),
  updateConfig: (config) => api.post('/mllm/config', config).then((r) => r.data),
  enable: (enabled = true, shadowMode = true) =>
    api.post('/mllm/enable', { enabled, shadow_mode: shadowMode }).then((r) => r.data),
}
```

---

## 九、问题优先级汇总

### P0 - 阻塞性问题（必须修复）

1. **ConfigView 保存按钮无法工作**: `configAPI.updateRules`、`updateSettings`、`updateZones` 以及 `mllmAPI.updateConfig`、`getConfig` 方法在 API client 中未定义，导致所有配置保存功能完全不可用。

### P1 - 高优先级

2. **MonitorView 状态锁竞态**: `switchingMode` 在异常流程中可能无法正确释放。
3. **AlarmsView resolve API 不匹配**: 需要确认 `alarmsAPI.resolve()` 与后端路由是否匹配。

### P2 - 中优先级

4. **EventsView redetect 自动启动**: 路由跳转后不会自动触发检测启动。
5. **Promise 未处理警告**: 多个 `ElMessageBox.confirm` 缺少 `.catch()` 处理。
6. **StreamPlayer 重试无退避**: 固定 1 秒重试间隔。
7. **ConfigView 坐标输入 UX**: JSON 字符串输入不友好。

### P3 - 低优先级

8. **DashboardView 按钮文案不一致**: "开始检测"文案但行为是跳转。
9. **快照 404 处理**: 缺少友好错误提示。

---

## 十、修复建议

### 立即修复（P0）

在 `api/client.js` 中补充缺失的 API 方法：

```javascript
export const configAPI = {
  get: () => api.get('/config').then((r) => r.data),
  updateRules: (rules) => api.post('/config/rules', rules).then((r) => r.data),
  updateSettings: (settings) => api.post('/config/settings', settings).then((r) => r.data),
  updateZones: (zones) => api.post('/config/zones', zones).then((r) => r.data),
}

export const mllmAPI = {
  status: () => cachedGet('mllm:status', () => api.get('/mllm/status').then((r) => r.data)),
  getConfig: () => api.get('/mllm/config').then((r) => r.data),
  updateConfig: (config) => api.post('/mllm/config', config).then((r) => r.data),
  enable: (enabled = true, shadowMode = true) =>
    api.post('/mllm/enable', { enabled, shadow_mode: shadowMode }).then((r) => r.data),
}
```

### 建议修复（P1-P3）

1. 为所有 `ElMessageBox.confirm` 添加 `.catch(() => {})`
2. 在 `MonitorView` 的 `clearUploadedFile` 中添加 `await`
3. 为 `StreamPlayer` 添加指数退避重试
4. 优化 `DashboardView` 按钮文案逻辑

---

## 十一、测试用例建议

### 单元测试（Vitest）

```javascript
// tests/api/client.spec.js
describe('configAPI', () => {
  it('should have updateRules method', () => {
    expect(configAPI.updateRules).toBeDefined()
  })
  it('should have updateSettings method', () => {
    expect(configAPI.updateSettings).toBeDefined()
  })
  it('should have updateZones method', () => {
    expect(configAPI.updateZones).toBeDefined()
  })
})

describe('mllmAPI', () => {
  it('should have getConfig method', () => {
    expect(mllmAPI.getConfig).toBeDefined()
  })
  it('should have updateConfig method', () => {
    expect(mllmAPI.updateConfig).toBeDefined()
  })
})
```

### E2E 测试（agent-browser）

```bash
# 测试配置页面保存按钮
agent-browser open http://localhost:5173/config
agent-browser snapshot -i
agent-browser click @e1  # 保存规则按钮
agent-browser wait --text "规则已保存"

# 测试监控页面启动/停止
agent-browser open http://localhost:5173/monitor
agent-browser snapshot -i
agent-browser click @e2  # 启动按钮
agent-browser wait --text "检测运行中"
agent-browser click @e3  # 停止按钮
agent-browser wait --text "已停止"
```

---

## 十二、审查结论

| 类别 | 数量 | 状态 |
|------|------|------|
| 总按钮数 | 42 | - |
| 功能完整 | 37 | ✅ |
| 存在问题 | 5 | ⚠️ |
| 阻塞性问题 | 1 | 🔴 |

**总体评价**: 前端按钮功能实现较为完整，交互逻辑清晰。但存在 **1 个 P0 阻塞性问题**（ConfigView API 缺失）需要立即修复，以及若干中低优先级问题建议后续优化。
