<!--
  Copyright (c) 2025 YOLO Course Design Contributors
  SPDX-License-Identifier: Apache-2.0
-->

<template>
  <div id="app-container">
    <el-container>
      <el-header class="app-header" :class="{ 'header-collapsed': isMobile }" role="banner">
        <div class="header-left">
          <el-button
            v-if="isMobile"
            class="menu-toggle"
            :aria-label="isMobileMenuOpen ? '关闭菜单' : '打开菜单'"
            text
            @click="isMobileMenuOpen = !isMobileMenuOpen"
          >
            {{ isMobileMenuOpen ? '✕' : '☰' }}
          </el-button>
          <h1 class="app-title">
            <span class="title-icon">
              <svg
                width="28"
                height="28"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <rect
                  x="2"
                  y="5"
                  width="20"
                  height="14"
                  rx="3"
                  stroke="currentColor"
                  stroke-width="1.8"
                  fill="none"
                />
                <circle
                  cx="12"
                  cy="12"
                  r="3"
                  stroke="currentColor"
                  stroke-width="1.5"
                  fill="none"
                />
                <circle cx="12" cy="12" r="1.2" fill="currentColor" />
                <path
                  d="M12 2v3M12 19v3M2 12h3M19 12h3"
                  stroke="currentColor"
                  stroke-width="1.2"
                  opacity="0.5"
                />
              </svg>
            </span>
            <div class="title-text">
              <span class="title-main">
                CAMPUS
                <span class="title-highlight">SAFE</span>
              </span>
              <span class="title-sub">智能安防监控系统</span>
            </div>
          </h1>
        </div>
        <div class="header-right">
          <div class="header-time">
            <span class="time-clock">{{ currentTime }}</span>
            <span class="time-date">{{ currentDate }}</span>
          </div>
          <div class="header-separator" />
          <div class="connection-status" :class="{ 'is-connected': connected }">
            <span class="conn-dot" />
            <span class="conn-label">{{ connected ? '在线' : '离线' }}</span>
          </div>
          <el-button
            v-if="!connected"
            text
            size="small"
            class="retry-btn"
            :loading="checking"
            @click="checkHealth"
          >
            重连
          </el-button>
        </div>
      </el-header>
      <el-container>
        <transition name="slide">
          <el-aside
            v-show="!isMobile || isMobileMenuOpen"
            width="240px"
            class="app-aside"
            role="navigation"
            aria-label="主导航"
          >
            <div class="sidebar-brand">
              <div class="brand-icon">
                <svg
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    fill="none"
                  />
                </svg>
              </div>
              <div class="brand-text">
                <span class="brand-name">CampusSafe</span>
                <span class="brand-version">v2.0</span>
              </div>
            </div>
            <div class="sidebar-divider" />
            <el-menu :default-active="route.path" router class="nav-menu" @select="onMenuSelect">
              <el-menu-item index="/">
                <span class="nav-icon">
                  <svg
                    width="18"
                    height="18"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  >
                    <rect x="3" y="3" width="7" height="7" />
                    <rect x="14" y="3" width="7" height="7" />
                    <rect x="14" y="14" width="7" height="7" />
                    <rect x="3" y="14" width="7" height="7" />
                  </svg>
                </span>
                <span class="nav-label">数据仪表盘</span>
              </el-menu-item>
              <el-menu-item index="/monitor">
                <span class="nav-icon">
                  <svg
                    width="18"
                    height="18"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  >
                    <path d="M23 7l-7 5 7 5V7z" />
                    <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                  </svg>
                </span>
                <span class="nav-label">实时监控</span>
              </el-menu-item>
              <el-menu-item index="/events">
                <span class="nav-icon">
                  <svg
                    width="18"
                    height="18"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  >
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                    <line x1="16" y1="13" x2="8" y2="13" />
                    <line x1="16" y1="17" x2="8" y2="17" />
                    <polyline points="10 9 9 9 8 9" />
                  </svg>
                </span>
                <span class="nav-label">事件记录</span>
              </el-menu-item>
              <el-menu-item index="/alarms">
                <span class="nav-icon">
                  <svg
                    width="18"
                    height="18"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  >
                    <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
                    <path d="M13.73 21a2 2 0 0 1-3.46 0" />
                  </svg>
                </span>
                <span class="nav-label">报警管理</span>
                <span v-if="activeAlarmCount > 0" class="nav-badge">{{ activeAlarmCount }}</span>
              </el-menu-item>
              <el-menu-item index="/config">
                <span class="nav-icon">
                  <svg
                    width="18"
                    height="18"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  >
                    <circle cx="12" cy="12" r="3" />
                    <path
                      d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"
                    />
                  </svg>
                </span>
                <span class="nav-label">系统配置</span>
              </el-menu-item>
            </el-menu>
            <div class="sidebar-footer">
              <div class="footer-line" />
              <div class="footer-info">
                <span class="footer-label">系统状态</span>
                <span class="footer-value" :class="{ 'is-online': connected }">
                  {{ connected ? '正常运行' : '连接中断' }}
                </span>
              </div>
            </div>
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
const currentTime = ref('')
const currentDate = ref('')
const activeAlarmCount = ref(0)

const isMobile = computed(() => windowWidth.value < 768)

const onMenuSelect = () => {
  if (isMobile.value) {
    isMobileMenuOpen.value = false
  }
}

let healthInterval = null
let alarmInterval = null
let resizeHandler = null
let visibilityHandler = null
let clockInterval = null
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

const updateClock = () => {
  const now = new Date()
  const h = String(now.getHours()).padStart(2, '0')
  const m = String(now.getMinutes()).padStart(2, '0')
  const s = String(now.getSeconds()).padStart(2, '0')
  currentTime.value = `${h}:${m}:${s}`
  const y = now.getFullYear()
  const mo = String(now.getMonth() + 1).padStart(2, '0')
  const d = String(now.getDate()).padStart(2, '0')
  const weekdays = ['周日', '周一', '周二', '周三', '周四', '周五', '周六']
  currentDate.value = `${y}-${mo}-${d} ${weekdays[now.getDay()]}`
}

const fetchActiveAlarms = async () => {
  try {
    const res = await fetch('/api/alarms/stats')
    const stats = await res.json()
    activeAlarmCount.value = stats.active_count || 0
  } catch {
    // silently fail
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
  updateClock()
  clockInterval = setInterval(updateClock, 1000)
  checkHealth()
  healthInterval = setInterval(checkHealth, 30000)
  fetchActiveAlarms()
  alarmInterval = setInterval(fetchActiveAlarms, 15000)
  resizeHandler = debounce(onResize, 200)
  window.addEventListener('resize', resizeHandler)
  visibilityHandler = onVisibilityChange
  document.addEventListener('visibilitychange', visibilityHandler)
})

onUnmounted(() => {
  if (clockInterval) clearInterval(clockInterval)
  if (healthInterval) clearInterval(healthInterval)
  if (alarmInterval) clearInterval(alarmInterval)
  if (resizeHandler) window.removeEventListener('resize', resizeHandler)
  if (visibilityHandler) document.removeEventListener('visibilitychange', visibilityHandler)
})
</script>

<style>
/* ============================================================
   DESIGN SYSTEM: Industrial Surveillance Aesthetic
   Theme: Dark industrial with amber alert accents
   ============================================================ */

@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700;800;900&display=swap');

:root {
  /* Primary palette: Cyber-industrial amber + neon cyan */
  --color-primary: #f59e0b;
  --color-primary-hover: #fbbf24;
  --color-primary-muted: rgba(245, 158, 11, 0.12);
  --color-primary-glow: 0 0 20px rgba(245, 158, 11, 0.25);
  --color-neon-orange: #ff6b35;
  --color-cyan: #00d4ff;
  --color-cyan-muted: rgba(0, 212, 255, 0.12);
  --color-cyan-glow: 0 0 20px rgba(0, 212, 255, 0.25);

  /* Alert colors */
  --color-success: #10b981;
  --color-warning: #f59e0b;
  --color-danger: #ef4444;
  --color-info: #00d4ff;

  /* Background layers: Deeper cyber dark */
  --bg-root: #050810;
  --bg-surface: #0a0f18;
  --bg-elevated: #111827;
  --bg-overlay: #1a2332;
  --bg-card: rgba(16, 24, 39, 0.7);
  --bg-grid: rgba(0, 212, 255, 0.03);

  /* Text hierarchy */
  --text-primary: #f8fafc;
  --text-regular: #94a3b8;
  --text-secondary: #64748b;
  --text-disabled: #475569;
  --text-muted: #334155;

  /* Borders */
  --border-subtle: rgba(148, 163, 184, 0.06);
  --border-default: rgba(148, 163, 184, 0.1);
  --border-strong: rgba(148, 163, 184, 0.18);
  --border-accent: rgba(245, 158, 11, 0.3);
  --border-cyan: rgba(0, 212, 255, 0.2);

  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.5);
  --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.5);
  --shadow-lg: 0 8px 30px rgba(0, 0, 0, 0.6);
  --shadow-glow: 0 0 30px rgba(245, 158, 11, 0.08);
  --shadow-cyan: 0 0 30px rgba(0, 212, 255, 0.08);

  /* Radii */
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --radius-xl: 20px;

  /* Transitions */
  --transition-fast: 0.15s cubic-bezier(0.4, 0, 0.2, 1);
  --transition-normal: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  --transition-slow: 0.4s cubic-bezier(0.4, 0, 0.2, 1);

  /* Layout */
  --header-height: 64px;
  --sidebar-width: 240px;

  /* Element Plus overrides */
  --el-fill-color-blank: var(--bg-elevated);
  --el-bg-color: var(--bg-surface);
  --el-border-color: var(--border-default);
  --el-text-color-primary: var(--text-primary);
  --el-text-color-regular: var(--text-regular);
  --el-color-primary: var(--color-primary);
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
    'Noto Sans SC',
    -apple-system,
    BlinkMacSystemFont,
    'Segoe UI',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: var(--text-primary);
  background: var(--bg-root);
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
  color: var(--color-primary);
  text-decoration: none;
}

:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
  border-radius: 2px;
}

/* ============================================================
   HEADER
   ============================================================ */
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--bg-surface);
  color: var(--text-primary);
  padding: 0 24px;
  height: var(--header-height) !important;
  border-bottom: 1px solid var(--border-default);
  box-shadow: var(--shadow-sm);
  z-index: 100;
  transition: padding var(--transition-fast);
  flex-shrink: 0;
  position: relative;
}

.app-header::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--color-primary) 50%, transparent);
  opacity: 0.3;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.menu-toggle {
  color: var(--text-primary) !important;
  font-size: 20px;
}

.app-title {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 0;
}

.title-icon {
  color: var(--color-primary);
  display: flex;
  align-items: center;
  justify-content: center;
}

.title-text {
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.title-main {
  font-family: 'Orbitron', 'JetBrains Mono', monospace;
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 2px;
  color: var(--text-primary);
  line-height: 1.2;
}

.title-highlight {
  color: var(--color-primary);
}

.title-sub {
  font-size: 11px;
  font-weight: 400;
  color: var(--text-secondary);
  letter-spacing: 2px;
  text-transform: uppercase;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.header-time {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 1px;
}

.time-clock {
  font-family: 'JetBrains Mono', monospace;
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: 0.5px;
  line-height: 1.2;
}

.time-date {
  font-size: 11px;
  color: var(--text-secondary);
  letter-spacing: 0.5px;
}

.header-separator {
  width: 1px;
  height: 28px;
  background: var(--border-default);
}

.connection-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 12px;
  border-radius: 20px;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.2);
  transition: all var(--transition-fast);
}

.connection-status.is-connected {
  background: rgba(16, 185, 129, 0.1);
  border-color: rgba(16, 185, 129, 0.2);
}

.conn-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--color-danger);
  position: relative;
}

.connection-status.is-connected .conn-dot {
  background: var(--color-success);
}

.connection-status.is-connected .conn-dot::after {
  content: '';
  position: absolute;
  inset: -3px;
  border-radius: 50%;
  border: 1px solid var(--color-success);
  animation: conn-pulse 2s ease-in-out infinite;
}

@keyframes conn-pulse {
  0%,
  100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0;
    transform: scale(1.5);
  }
}

.conn-label {
  font-size: 12px;
  font-weight: 500;
  color: var(--color-danger);
}

.connection-status.is-connected .conn-label {
  color: var(--color-success);
}

.retry-btn {
  color: var(--text-regular) !important;
  font-size: 12px;
}

.retry-btn:hover {
  color: var(--text-primary) !important;
}

/* ============================================================
   SIDEBAR
   ============================================================ */
.app-aside {
  background: var(--bg-surface);
  border-right: 1px solid var(--border-default);
  transition: all var(--transition-normal);
  z-index: 90;
  width: var(--sidebar-width) !important;
  display: flex;
  flex-direction: column;
  position: relative;
}

.app-aside::before {
  content: '';
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: 1px;
  background: linear-gradient(
    180deg,
    transparent,
    var(--color-primary) 30%,
    var(--color-primary) 70%,
    transparent
  );
  opacity: 0.15;
}

.sidebar-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px 20px 16px;
}

.brand-icon {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  background: linear-gradient(135deg, var(--color-primary), #d97706);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--bg-root);
  box-shadow: 0 4px 12px rgba(245, 158, 11, 0.25);
}

.brand-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.brand-name {
  font-family: 'Orbitron', 'JetBrains Mono', monospace;
  font-size: 15px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: 1px;
}

.brand-version {
  font-size: 11px;
  color: var(--text-secondary);
  font-family: 'JetBrains Mono', monospace;
}

.sidebar-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-default), transparent);
  margin: 0 16px 8px;
}

.nav-menu {
  height: 100%;
  border-right: none;
  padding: 8px 12px;
  background: transparent;
}

.nav-menu .el-menu-item {
  height: 48px;
  line-height: 48px;
  margin: 4px 0;
  border-radius: var(--radius-sm);
  transition: all var(--transition-fast);
  color: var(--text-regular);
  padding: 0 14px !important;
  display: flex;
  align-items: center;
  gap: 12px;
  position: relative;
  overflow: hidden;
}

.nav-menu .el-menu-item::before {
  content: '';
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 3px;
  height: 0;
  background: var(--color-primary);
  border-radius: 0 2px 2px 0;
  transition: height var(--transition-fast);
}

.nav-menu .el-menu-item:hover {
  background: var(--color-primary-muted);
  color: var(--color-primary);
}

.nav-menu .el-menu-item.is-active {
  background: var(--color-primary-muted);
  color: var(--color-primary);
  font-weight: 600;
}

.nav-menu .el-menu-item.is-active::before {
  height: 24px;
}

.nav-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  flex-shrink: 0;
}

.nav-label {
  font-size: 13px;
  letter-spacing: 0.3px;
}

.nav-badge {
  margin-left: auto;
  min-width: 20px;
  height: 20px;
  padding: 0 6px;
  border-radius: 10px;
  background: var(--color-danger);
  color: white;
  font-size: 11px;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'JetBrains Mono', monospace;
  animation: badge-glow 2s ease-in-out infinite;
}

@keyframes badge-glow {
  0%,
  100% {
    box-shadow: 0 0 5px rgba(239, 68, 68, 0.4);
  }
  50% {
    box-shadow: 0 0 15px rgba(239, 68, 68, 0.7);
  }
}

.sidebar-footer {
  margin-top: auto;
  padding: 16px 20px 20px;
}

.footer-line {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-default), transparent);
  margin-bottom: 12px;
}

.footer-info {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.footer-label {
  font-size: 11px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 1px;
}

.footer-value {
  font-size: 11px;
  font-weight: 600;
  color: var(--color-danger);
  font-family: 'JetBrains Mono', monospace;
}

.footer-value.is-online {
  color: var(--color-success);
}

/* ============================================================
   MAIN CONTENT
   ============================================================ */
.app-main {
  background: var(--bg-root);
  padding: 24px;
  height: calc(100vh - var(--header-height));
  height: calc(100dvh - var(--header-height));
  overflow-y: auto;
  scroll-behavior: smooth;
  position: relative;
}

.app-main::before {
  content: '';
  position: fixed;
  top: var(--header-height);
  left: var(--sidebar-width);
  right: 0;
  bottom: 0;
  background-image:
    linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px),
    radial-gradient(circle at 20% 50%, rgba(245, 158, 11, 0.02) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(0, 212, 255, 0.015) 0%, transparent 40%);
  background-size:
    60px 60px,
    60px 60px,
    100% 100%,
    100% 100%;
  pointer-events: none;
  z-index: 0;
  animation: grid-pulse 4s ease-in-out infinite;
}

@keyframes grid-pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
}

.connection-alert {
  margin-bottom: 16px;
  flex-shrink: 0;
  position: relative;
  z-index: 1;
}

/* ============================================================
   PAGE TRANSITIONS
   ============================================================ */
.page-enter-active,
.page-leave-active {
  transition:
    opacity 0.3s cubic-bezier(0.4, 0, 0.2, 1),
    transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.page-enter-from {
  opacity: 0;
  transform: translateY(12px);
}

.page-leave-to {
  opacity: 0;
  transform: translateY(-8px);
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

/* ============================================================
   ELEMENT PLUS OVERRIDES
   ============================================================ */
.el-card {
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--border-subtle) !important;
  background: var(--bg-card) !important;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  box-shadow:
    0 8px 32px rgba(0, 0, 0, 0.4),
    inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
  transition: all var(--transition-fast);
  position: relative;
  overflow: hidden;
}

.el-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(245, 158, 11, 0.3),
    rgba(0, 212, 255, 0.2),
    transparent
  );
  opacity: 0.5;
  transition: opacity var(--transition-fast);
}

.el-card::after {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle at 50% 50%, rgba(0, 212, 255, 0.03) 0%, transparent 70%);
  pointer-events: none;
  opacity: 0;
  transition: opacity var(--transition-slow);
}

.el-card:hover {
  box-shadow:
    0 12px 40px rgba(0, 0, 0, 0.5),
    0 0 30px rgba(0, 212, 255, 0.05),
    inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
  border-color: var(--border-cyan) !important;
  transform: translateY(-2px);
}

.el-card:hover::before {
  opacity: 1;
}

.el-card:hover::after {
  opacity: 1;
}

.el-button {
  border-radius: var(--radius-sm) !important;
  transition: all var(--transition-fast);
  font-weight: 500;
}

.el-button--primary {
  --el-button-bg-color: var(--color-primary);
  --el-button-border-color: var(--color-primary);
  --el-button-hover-bg-color: var(--color-primary-hover);
  --el-button-hover-border-color: var(--color-primary-hover);
  --el-button-text-color: var(--bg-root);
}

.el-button--primary:hover {
  box-shadow:
    0 4px 12px rgba(245, 158, 11, 0.3),
    0 0 20px rgba(245, 158, 11, 0.2);
}

.el-button--success {
  --el-button-bg-color: var(--color-success);
  --el-button-border-color: var(--color-success);
  --el-button-hover-bg-color: #34d399;
  --el-button-hover-border-color: #34d399;
}

.el-button--success:hover {
  box-shadow:
    0 4px 12px rgba(16, 185, 129, 0.3),
    0 0 20px rgba(16, 185, 129, 0.15);
}

.el-button--danger {
  --el-button-bg-color: var(--color-danger);
  --el-button-border-color: var(--color-danger);
  --el-button-hover-bg-color: #f87171;
  --el-button-hover-border-color: #f87171;
}

.el-button--danger:hover {
  box-shadow:
    0 4px 12px rgba(239, 68, 68, 0.3),
    0 0 20px rgba(239, 68, 68, 0.15);
}

.el-tag {
  border-radius: 6px !important;
  font-weight: 500;
}

/* Dark table */
.el-table {
  border-radius: var(--radius-sm) !important;
  --el-table-bg-color: var(--bg-card);
  --el-table-tr-bg-color: var(--bg-card);
  --el-table-header-bg-color: var(--bg-elevated);
  --el-table-row-hover-bg-color: rgba(245, 158, 11, 0.04);
  --el-table-border-color: var(--border-subtle);
  --el-table-text-color: var(--text-primary);
  --el-table-header-text-color: var(--text-regular);
}

.el-table th.el-table__cell {
  background-color: var(--bg-elevated) !important;
  color: var(--text-regular) !important;
  font-weight: 600;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.el-table td.el-table__cell {
  background-color: var(--bg-card) !important;
  color: var(--text-primary) !important;
}

.el-table--striped .el-table__body tr.el-table__row--striped td.el-table__cell {
  background-color: rgba(245, 158, 11, 0.02) !important;
}

.el-table__body tr:hover > td.el-table__cell {
  background-color: rgba(245, 158, 11, 0.04) !important;
}

/* Dark dialog */
.el-dialog {
  border-radius: var(--radius-lg) !important;
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-default) !important;
  box-shadow: var(--shadow-lg) !important;
}

.el-dialog__title {
  color: var(--text-primary) !important;
  font-weight: 600;
}

.el-dialog__body {
  color: var(--text-regular) !important;
}

/* Dark input / select */
.el-input__wrapper {
  background-color: var(--bg-elevated) !important;
  box-shadow: 0 0 0 1px var(--border-default) inset !important;
  border-radius: var(--radius-sm) !important;
}

.el-input__inner {
  color: var(--text-primary) !important;
  font-family: 'Noto Sans SC', sans-serif;
}

.el-input.is-focus .el-input__wrapper {
  box-shadow: 0 0 0 1px var(--color-primary) inset !important;
}

.el-select-dropdown {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-default) !important;
  border-radius: var(--radius-md) !important;
}

.el-select-dropdown__item {
  color: var(--text-regular) !important;
}

.el-select-dropdown__item.hover,
.el-select-dropdown__item:hover {
  background-color: var(--color-primary-muted) !important;
  color: var(--color-primary) !important;
}

.el-select-dropdown__item.is-selected {
  color: var(--color-primary) !important;
}

/* Dark alert */
.el-alert {
  border-radius: var(--radius-sm) !important;
}

.el-alert--warning {
  background: rgba(245, 158, 11, 0.08) !important;
  border: 1px solid rgba(245, 158, 11, 0.2) !important;
}

/* Dark pagination */
.el-pagination {
  color: var(--text-regular) !important;
}

.el-pagination button:disabled {
  color: var(--text-disabled) !important;
}

/* Dark progress */
.el-progress-bar__outer {
  background-color: var(--bg-elevated) !important;
}

/* Dark divider */
.el-divider__text {
  background: var(--bg-surface) !important;
  color: var(--text-secondary);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.el-divider {
  border-top-color: var(--border-subtle) !important;
}

/* Dark slider */
.el-slider__runway {
  background-color: var(--bg-elevated) !important;
}

.el-slider__button {
  border: 2px solid var(--color-primary) !important;
  background: var(--bg-surface) !important;
}

/* Dark switch */
.el-switch__core {
  background: var(--bg-elevated) !important;
}

.el-switch.is-checked .el-switch__core {
  background-color: var(--color-primary) !important;
}

/* Dark upload */
.el-upload-dragger {
  background: var(--bg-elevated) !important;
  border-color: var(--border-default) !important;
  border-radius: var(--radius-md) !important;
}

.el-upload-dragger:hover {
  border-color: var(--color-primary) !important;
}

/* Dark input-number */
.el-input-number__decrease,
.el-input-number__increase {
  background: var(--bg-elevated) !important;
  color: var(--text-regular) !important;
}

.el-input-number__decrease:hover,
.el-input-number__increase:hover {
  color: var(--color-primary) !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 5px;
  height: 5px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: var(--border-strong);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--text-secondary);
}

/* Loading skeleton animation */
@keyframes shimmer {
  0% {
    background-position: -200px 0;
  }
  100% {
    background-position: calc(200px + 100%) 0;
  }
}

.skeleton {
  background: linear-gradient(
    90deg,
    var(--bg-elevated) 25%,
    var(--bg-overlay) 50%,
    var(--bg-elevated) 75%
  );
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

/* ============================================================
   RESPONSIVE
   ============================================================ */
@media (max-width: 1024px) {
  .app-main {
    padding: 16px;
  }
}

@media (max-width: 768px) {
  .app-header {
    padding: 0 16px;
  }

  .title-sub {
    display: none;
  }

  .time-date {
    display: none;
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
    background: rgba(0, 0, 0, 0.5);
    z-index: 98;
    transition: opacity var(--transition-normal);
    backdrop-filter: blur(4px);
  }

  .app-main {
    padding: 12px;
  }

  .app-main::before {
    left: 0;
  }
}

@media (max-width: 480px) {
  .app-title {
    gap: 8px;
  }

  .title-main {
    font-size: 15px;
  }

  .header-right {
    gap: 8px;
  }

  .connection-status {
    padding: 3px 8px;
  }

  .conn-label {
    display: none;
  }

  .app-main {
    padding: 8px;
  }
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
