/**
 * Copyright (c) 2025 YOLO Course Design Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Dev-only logger that strips all output in production builds.
 * Use instead of console.log / console.error throughout the app.
 */
const isDev = import.meta.env.DEV

export const logger = {
  log(...args) {
    if (isDev) console.log(...args)
  },
  warn(...args) {
    if (isDev) console.warn(...args)
  },
  error(...args) {
    console.error(...args)
  },
}
