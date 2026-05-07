<!--
  Copyright (c) 2025 YOLO Course Design Contributors
  SPDX-License-Identifier: MIT
-->

<template>
  <div id="app-container">
    <el-container>
      <el-header class="app-header" :class="{ 'header-collapsed': isMobile }" role="banner">
        <div class="header-left">
          <el-button
            v-if="isMobile"
            class="menu-toggle"
            :icon="isMobileMenuOpen ? 'Close' : 'Expand'"
            :aria-label="isMobileMenuOpen ? '关闭菜单' : '打开菜单'"
            text
            @click="isMobileMenuOpen = !isMobileMenuOpen"
          />
          <h1 class="app-title">
            <el-icon class="title-icon"><VideoCamera /></el-icon>
            <span>YOLO 实时检测系统</span>
          </h1>
        </div>
        <div class="header-right">
          <transition name="fade" mode="out-in">
            <el-tag v-if="connected" :key="'on'" type="success" size="small" effect="dark" aria-label="后端已连接">
              <el-icon class="tag-icon"><CircleCheck /></el-icon>
              已连接
            </el-tag>
            <div v-else :key="'off'" class="disconnect-wrap">
              <el-tag type="danger" size="small" effect="dark" aria-label="后端未连接">
                <el-icon class="tag-icon"><CircleClose /></el-icon>
                未连接
              </el-tag>
              <el-button
                text
                size="small"
                class="retry-btn"
                :loading="checking"
                @click="checkHealth"
              >
                重试
              </el-button>
            </div>
          </transition>
        </div>
      </el-header>
      <el-container>
        <transition name="slide">
          <el-aside
            v-show="!isMobile || isMobileMenuOpen"
            width="200px"
            class="app-aside"
            role="navigation"
            aria-label="主导航"
          >
            <el-menu :default-active="route.path" router class="nav-menu" @select="onMenuSelect">
              <el-menu-item index="/monitor">
                <el-icon><Monitor /></el-icon>
                <span>实时监控</span>
              </el-menu-item>
              <el-menu-item index="/events">
                <el-icon><List /></el-icon>
                <span>事件记录</span>
              </el-menu-item>
              <el-menu-item index="/alarms">
                <el-icon><Bell /></el-icon>
                <span>报警管理</span>
              </el-menu-item>
              <el-menu-item index="/config">
                <el-icon><Setting /></el-icon>
                <span>系统配置</span>
              </el-menu-item>
            </el-menu>
          </el-aside>
        </transition>
        <div
          v-if="isMobile && isMobileMenuOpen"
          class="mobile-overlay"
          @click="isMobileMenuOpen = false"
        />
        <el-main class="app-main" role="main">
          <el-alert
            v-if="!connected"
            title="后端服务未连接"
            type="warning"
            :closable="false"
            show-icon
            class="connection-alert"
          >
            <template #default>
              <span>请确保后端服务已启动。</span>
              <el-button link type="primary" size="small" @click="checkHealth">立即重试</el-button>
            </template>
          </el-alert>
          <router-view v-slot="{ Component }">
            <transition name="page" mode="out-in">
              <component :is="Component" />
            </transition>
          </router-view>
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script setup>
/**
 * App.vue - Application root component.
 *
 * Manages the top-level layout (sidebar navigation + main content area)
 * and responsive behavior for mobile devices.
 *
 * Key responsibilities:
 * - Render sidebar navigation with route-aware active states
 * - Detect mobile viewport and toggle hamburger menu
 * - Display backend connection status indicator
 */
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { logger } from './utils/logger'
import { debounce } from './utils/perf'

const route = useRoute()
const connected = ref(false)
const checking = ref(false)
const isMobileMenuOpen = ref(false)
const windowWidth = ref(window.innerWidth)

const isMobile = computed(() => windowWidth.value < 768)

const onMenuSelect = () => {
  if (isMobile.value) {
    isMobileMenuOpen.value = false
  }
}

let healthInterval = null
let resizeHandler = null
let visibilityHandler = null
let pollingPaused = false

const checkHealth = async () => {
  if (pollingPaused || checking.value) return
  checking.value = true
  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 5000)
    const res = await fetch('/api/detection/health', {
      signal: controller.signal,
      cache: 'no-store',
    })
    clearTimeout(timeoutId)
    connected.value = res.ok
  } catch (err) {
    if (err.name === 'AbortError') {
      logger.warn('Health check timed out')
    }
    connected.value = false
  } finally {
    checking.value = false
  }
}

const onResize = () => {
  windowWidth.value = window.innerWidth
  if (!isMobile.value) {
    isMobileMenuOpen.value = false
  }
}

const onVisibilityChange = () => {
  if (document.hidden) {
    pollingPaused = true
  } else {
    pollingPaused = false
    checkHealth()
  }
}

onMounted(() => {
  checkHealth()
  healthInterval = setInterval(checkHealth, 30000)
  resizeHandler = debounce(onResize, 200)
  window.addEventListener('resize', resizeHandler)
  visibilityHandler = onVisibilityChange
  document.addEventListener('visibilitychange', visibilityHandler)
})

onUnmounted(() => {
  if (healthInterval) clearInterval(healthInterval)
  if (resizeHandler) window.removeEventListener('resize', resizeHandler)
  if (visibilityHandler) document.removeEventListener('visibilitychange', visibilityHandler)
})
</script>

<style>
:root {
  --primary-color: #409eff;
  --primary-dark: #337ecc;
  --primary-light: #ecf5ff;
  --bg-color: #f0f2f5;
  --card-bg: #ffffff;
  --text-primary: #303133;
  --text-regular: #606266;
  --text-secondary: #909399;
  --text-placeholder: #c0c4cc;
  --border-color: #e4e7ed;
  --border-light: #ebeef5;
  --shadow-sm: 0 1px 4px rgba(0, 0, 0, 0.06);
  --shadow-md: 0 2px 12px rgba(0, 0, 0, 0.08);
  --shadow-lg: 0 4px 20px rgba(0, 0, 0, 0.12);
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 14px;
  --transition-fast: 0.15s ease;
  --transition-normal: 0.25s ease;
  --header-height: 56px;
  --sidebar-width: 200px;
}

*,
*::before,
*::after {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html {
  font-size: 14px;
  -webkit-text-size-adjust: 100%;
}

body {
  font-family:
    -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue',
    Arial, 'Noto Sans SC', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: var(--text-primary);
  background: var(--bg-color);
  line-height: 1.6;
  overflow: hidden;
}

#app-container {
  height: 100vh;
  height: 100dvh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

a {
  color: var(--primary-color);
  text-decoration: none;
}

:focus-visible {
  outline: 2px solid var(--primary-color);
  outline-offset: 2px;
  border-radius: 2px;
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
  color: white;
  padding: 0 20px;
  height: var(--header-height) !important;
  box-shadow: var(--shadow-md);
  z-index: 100;
  transition: padding var(--transition-fast);
  flex-shrink: 0;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.menu-toggle {
  color: white !important;
  font-size: 20px;
}

.app-title {
  font-size: 17px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
  letter-spacing: 0.5px;
  white-space: nowrap;
}

.title-icon {
  font-size: 22px;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.tag-icon {
  margin-right: 4px;
  vertical-align: middle;
}

.disconnect-wrap {
  display: flex;
  align-items: center;
  gap: 4px;
}

.retry-btn {
  color: rgba(255, 255, 255, 0.9) !important;
  font-size: 12px;
  padding: 0 4px;
}

.retry-btn:hover {
  color: #fff !important;
}

.connection-alert {
  margin-bottom: 12px;
  flex-shrink: 0;
}

.app-aside {
  background: var(--card-bg);
  border-right: 1px solid var(--border-color);
  transition: all var(--transition-normal);
  z-index: 90;
  width: var(--sidebar-width) !important;
}

.nav-menu {
  height: 100%;
  border-right: none;
  padding-top: 8px;
}

.nav-menu .el-menu-item {
  height: 48px;
  line-height: 48px;
  margin: 4px 8px;
  border-radius: var(--radius-sm);
  transition: all var(--transition-fast);
}

.nav-menu .el-menu-item:hover {
  background: var(--primary-light);
}

.nav-menu .el-menu-item.is-active {
  background: var(--primary-light);
  color: var(--primary-color);
  font-weight: 600;
}

.app-main {
  background: var(--bg-color);
  padding: 20px;
  height: calc(100vh - var(--header-height));
  height: calc(100dvh - var(--header-height));
  overflow-y: auto;
  scroll-behavior: smooth;
}

/* Page transition */
.page-enter-active,
.page-leave-active {
  transition:
    opacity 0.2s ease,
    transform 0.2s ease;
}

.page-enter-from {
  opacity: 0;
  transform: translateY(6px);
}

.page-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity var(--transition-fast);
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.slide-enter-active,
.slide-leave-active {
  transition: transform var(--transition-normal);
}

.slide-enter-from,
.slide-leave-to {
  transform: translateX(-100%);
}

/* Responsive: Tablet */
@media (max-width: 1024px) {
  .app-main {
    padding: 16px;
  }
}

/* Responsive: Mobile */
@media (max-width: 768px) {
  .app-header {
    padding: 0 12px;
  }

  .app-title {
    font-size: 15px;
  }

  .app-aside {
    position: fixed;
    top: var(--header-height);
    left: 0;
    bottom: 0;
    z-index: 99;
    box-shadow: var(--shadow-lg);
  }

  .mobile-overlay {
    position: fixed;
    top: var(--header-height);
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.4);
    z-index: 98;
    transition: opacity var(--transition-normal);
  }

  .app-main {
    padding: 12px;
  }
}

@media (max-width: 480px) {
  .app-title span {
    display: none;
  }

  .app-main {
    padding: 8px;
  }
}

/* Global Element Plus style overrides */
.el-card {
  border-radius: var(--radius-md) !important;
  border: none !important;
  box-shadow: var(--shadow-sm) !important;
  transition: box-shadow var(--transition-fast);
}

.el-card:hover {
  box-shadow: var(--shadow-md) !important;
}

.el-button {
  border-radius: var(--radius-sm) !important;
  transition: all var(--transition-fast);
}

.el-tag {
  border-radius: 4px !important;
}

.el-table {
  border-radius: var(--radius-sm) !important;
}

.el-dialog {
  border-radius: var(--radius-md) !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: #c0c4cc;
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: #909399;
}

/* Loading skeleton animation */
@keyframes shimmer {
  0% { background-position: -200px 0; }
  100% { background-position: calc(200px + 100%) 0; }
}

.skeleton {
  background: linear-gradient(90deg, #f0f2f5 25%, #e4e7ed 50%, #f0f2f5 75%);
  background-size: 200px 100%;
  animation: shimmer 1.5s infinite;
  border-radius: var(--radius-sm);
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
</style>
