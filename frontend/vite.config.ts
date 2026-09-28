import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Phase 1: bare dev-server config. Build/proxy tuning (if any) revisited
// when the dashboard grows past a single status page.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
  },
});
