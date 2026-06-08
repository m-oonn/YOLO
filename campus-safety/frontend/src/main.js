/**
 * ┌──────────────────────────────────────────┐
 * │ 【前端UI】main.js — Vue 应用入口          │
 * │ 职责：创建Vue实例、挂载路由、挂载到#app   │
 * │ 这是整个前端的启动文件                     │
 * └──────────────────────────────────────────┘
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

const app = createApp(App)

app.use(router)
app.mount('#app')
