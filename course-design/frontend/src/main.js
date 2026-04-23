/**
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: MIT
 */

import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import { Monitor, List, Setting, VideoPlay, VideoPause, VideoCamera } from '@element-plus/icons-vue'
import App from './App.vue'
import router from './router'

const app = createApp(App)

// Register only the icons used across all views
app.component('Monitor', Monitor)
app.component('List', List)
app.component('Setting', Setting)
app.component('VideoPlay', VideoPlay)
app.component('VideoPause', VideoPause)
app.component('VideoCamera', VideoCamera)

app.use(ElementPlus)
app.use(router)
app.mount('#app')
