import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The React app talks to the FastAPI server through this proxy, so there is no CORS setup.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": { target: "http://127.0.0.1:8000", rewrite: (p) => p.replace(/^\/api/, "") } } },
});
