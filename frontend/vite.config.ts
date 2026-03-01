import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": "http://localhost:7860",
      "/gradio": "http://localhost:7860",
      "/health": "http://localhost:7860",
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
