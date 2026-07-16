import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Served from https://pliploop.github.io/ReMIX/ once the repo is renamed to ReMIX.
export default defineConfig({
  plugins: [react()],
  base: '/ReMIX/',
  define: {
    // A real compile-time literal (`true`/`false`), so rollup can drop the
    // rating app's chunk entirely from the public build. Testing
    // `Boolean(import.meta.env.VITE_API_BASE)` is not enough: it is not folded
    // to a constant, so the dead branch survives and the code still ships.
    __HAS_BACKEND__: JSON.stringify(Boolean(process.env.VITE_API_BASE)),
  },
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
