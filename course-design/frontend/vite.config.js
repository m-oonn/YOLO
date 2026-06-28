import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'
import ElementPlus from 'unplugin-element-plus/vite'
import viteCompression from 'vite-plugin-compression'

const isProduction = process.env.NODE_ENV === 'production'
const rawBase = process.env.VITE_BASE
const BASE = rawBase ? `${rawBase.replace(/\/+$/, '')}/` : '/'

export default defineConfig({
  base: BASE,
  plugins: [
    vue(),
    Components({
      resolvers: [ElementPlusResolver()],
      directoryAsNamespace: false,
    }),
    ElementPlus({}),
    isProduction &&
      viteCompression({
        verbose: false,
        threshold: 10240,
      }),
  ].filter(Boolean),
  server: {
    port: 8080,
    warmup: {
      clientFiles: ['./src/main.js', './src/views/MonitorView.vue'],
    },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    target: 'es2020',
    minify: 'esbuild',
    cssCodeSplit: true,
    sourcemap: false,
    reportCompressedSize: false, // 优化：禁用报告加速构建
    chunkSizeWarningLimit: 1000,
    // 优化：启用预加载和预取关键资源
    modulePreload: {
      polyfill: false,
    },
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (id.includes('element-plus')) {
              if (id.includes('@element-plus/icons-vue')) return 'ep-icons'
              return 'element-plus'
            }
            if (id.includes('vue') || id.includes('@vue')) return 'vue-vendor'
            return 'vendor'
          }
        },
        // 优化：添加资源文件名hash，支持长期缓存
        entryFileNames: 'js/[name]-[hash].js',
        chunkFileNames: 'js/[name]-[hash].js',
        assetFileNames: (info) => {
          if (/\.(png|jpe?g|gif|svg|webp|ico)$/.test(info.name)) {
            return 'img/[name]-[hash][extname]'
          }
          if (/\.(css)$/.test(info.name)) {
            return 'css/[name]-[hash][extname]'
          }
          return 'assets/[name]-[hash][extname]'
        },
      },
    },
  },
})
