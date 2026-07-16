// Config for the SSR smoke test only (see src/ssr-check.jsx). The main config's
// manualChunks conflicts with SSR externals, so this one omits it.
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: { ssr: true, outDir: 'dist-ssr', rollupOptions: { input: 'src/ssr-check.jsx' } },
})
