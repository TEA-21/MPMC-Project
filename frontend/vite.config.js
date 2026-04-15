import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BACKEND = 'http://127.0.0.1:8080'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // ── REST + MJPEG ──────────────────────────────────────────────────────
      '/video_feed':  { target: BACKEND, changeOrigin: true },
      '/force-open':  { target: BACKEND, changeOrigin: true },
      '/system-reset':{ target: BACKEND, changeOrigin: true },
      '/state':       { target: BACKEND, changeOrigin: true },

      // ── Socket.IO  (HTTP long-poll + WebSocket upgrade) ───────────────────
      // The path must match what python-socketio mounts at (default: /socket.io)
      '/socket.io': {
        target: BACKEND,
        ws: true,             // enable WebSocket proxying
        changeOrigin: true,
        rewriteWsOrigin: true,// fixes WS origin header on Node 18+
        configure: (proxy) => {
          proxy.on('error', (err) => {
            // Suppress the noisy ECONNREFUSED logs while the backend starts up
            if (!err.message.includes('ECONNREFUSED')) {
              console.error('[proxy error]', err.message);
            }
          });
        },
      },
    },
  },
})
