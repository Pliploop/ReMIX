import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Served from https://pliploop.github.io/ReMIX/ once the repo is renamed to ReMIX.
export default defineConfig({
  plugins: [react()],
  base: '/ReMIX/',
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    // three.js is heavy and only /explore needs it; keep it out of the landing chunk.
    rollupOptions: {
      output: {
        manualChunks: {
          three: ['three', '@react-three/fiber', '@react-three/drei'],
        },
      },
    },
    chunkSizeWarningLimit: 1200,
  },
})
