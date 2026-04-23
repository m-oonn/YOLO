<!--
  Copyright (c) 2025 YOLO Course Design Contributors
  SPDX-License-Identifier: MIT
-->

<template>
  <div id="app-container">
    <el-container>
      <el-header class="app-header">
        <div class="header-left">
          <h1 class="app-title">YOLO 课程设计 — 实时目标检测系统</h1>
        </div>
        <div class="header-right">
          <el-tag v-if="connected" type="success" size="small">已连接</el-tag>
          <el-tag v-else type="danger" size="small">未连接</el-tag>
        </div>
      </el-header>
      <el-container>
        <el-aside width="200px">
          <el-menu :default-active="route.path" router class="nav-menu">
            <el-menu-item index="/monitor">
              <el-icon><Monitor /></el-icon>
              <span>实时监控</span>
            </el-menu-item>
            <el-menu-item index="/events">
              <el-icon><List /></el-icon>
              <span>事件记录</span>
            </el-menu-item>
            <el-menu-item index="/config">
              <el-icon><Setting /></el-icon>
              <span>系统配置</span>
            </el-menu-item>
          </el-menu>
        </el-aside>
        <el-main class="app-main">
          <router-view />
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { Monitor, List, Setting } from '@element-plus/icons-vue'

const route = useRoute()
const connected = ref(false)
let healthInterval = null

const checkHealth = async () => {
  try {
    const res = await fetch('/api/health')
    connected.value = res.ok
  } catch {
    connected.value = false
  }
}

onMounted(() => {
  checkHealth()
  healthInterval = setInterval(checkHealth, 10000)
})

onUnmounted(() => {
  if (healthInterval) clearInterval(healthInterval)
})
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
#app-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
}
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #409eff;
  color: white;
  padding: 0 20px;
  height: 60px !important;
}
.app-title {
  font-size: 18px;
  font-weight: 600;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 10px;
}
.nav-menu {
  height: 100%;
  border-right: none;
}
.app-main {
  background: #f5f7fa;
  padding: 20px;
  height: calc(100vh - 60px);
  overflow-y: auto;
}
</style>
