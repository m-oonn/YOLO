/**
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: MIT
 */

import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', redirect: '/monitor' },
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
  history: createWebHistory(),
  routes,
  scrollBehavior: () => ({ top: 0 }),
})

router.beforeEach((to) => {
  document.title = `${to.meta.title || 'YOLO'} - 实时检测系统`
})

export default router
