/**
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

const app = createApp(App)

app.use(router)
app.mount('#app')
