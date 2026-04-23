import { createRouter, createWebHistory } from 'vue-router'
import MonitorView from '../views/MonitorView.vue'
import EventsView from '../views/EventsView.vue'
import ConfigView from '../views/ConfigView.vue'

const routes = [
  { path: '/', redirect: '/monitor' },
  { path: '/monitor', name: 'Monitor', component: MonitorView, meta: { title: '实时监控' } },
  { path: '/events', name: 'Events', component: EventsView, meta: { title: '事件记录' } },
  { path: '/config', name: 'Config', component: ConfigView, meta: { title: '系统配置' } },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
