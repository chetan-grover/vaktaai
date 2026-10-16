import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// VAKTA_API lets a second dev server point at another backend (e.g. a test copy on :8001).
const api = process.env.VAKTA_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: api, changeOrigin: true } },
  },
});
