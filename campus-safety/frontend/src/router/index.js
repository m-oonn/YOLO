/**
 * ┌──────────────────────────────────────────┐
 * │ 【前端UI】router/index.js — 路由配置       │
 * │ 职责：定义5个页面的URL路由映射             │
 * │ / → Dashboard（仪表盘）                   │
 * │ /monitor → MonitorView（实时监控）         │
 * │ /events → EventsView（事件记录）           │
 * │ /alarms → AlarmsView（报警管理）           │
 * │ /config → ConfigView（系统配置）           │
 * │ 使用 hash 模式（#/xxx）支持 GitHub Pages   │
 * └──────────────────────────────────────────┘
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Dashboard',
    component: () => import('../views/DashboardView.vue'),
    meta: { title: '仪表盘' },
  },
  {
    path: '/monitor',
    name: 'Monitor',
    component: () => import('../views/MonitorView.vue'),
    meta: { title: '实时监控' },
  },
  {
    path: '/events',
    name: 'Events',
    component: () => import('../views/EventsView.vue'),
    meta: { title: '事件记录' },
  },
  {
    path: '/alarms',
    name: 'Alarms',
    component: () => import('../views/AlarmsView.vue'),
    meta: { title: '报警管理' },
  },
  {
    path: '/config',
    name: 'Config',
    component: () => import('../views/ConfigView.vue'),
    meta: { title: '系统配置' },
  },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
  scrollBehavior: () => ({ top: 0 }),
})

router.beforeEach((to) => {
  document.title = `${to.meta.title || 'YOLO'} - 实时检测系统`
})

export default router
