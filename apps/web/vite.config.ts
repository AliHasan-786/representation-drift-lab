import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  plugins: [react()],
  publicDir: fileURLToPath(new URL("../../public", import.meta.url)),
  build: {
    target: "es2022",
    sourcemap: true,
    cssCodeSplit: true,
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    globals: true,
  },
});
