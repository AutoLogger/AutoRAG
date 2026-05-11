import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

export default defineConfig({
  plugins: [react()],
  base: '/viz-assets/',
  build: {
    outDir: path.resolve(__dirname, '../src/autorag/static/viz'),
    emptyOutDir: true,
    assetsDir: 'assets',
    sourcemap: false,
    rollupOptions: { input: path.resolve(__dirname, 'index.html') },
  },
  server: {
    proxy: {
      '/viz/data': 'http://127.0.0.1:8000',
      '/viz/search': 'http://127.0.0.1:8000',
    },
  },
})
