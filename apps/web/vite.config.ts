import { defineConfig } from "vitest/config";
import { loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    base: env.VITE_BASE_PATH ?? "/",
    plugins: [react()],
    // The repository preserves every reviewed run artifact, but the deployed
    // visitor experience needs only its evidence-backed web bundle. Staging is
    // performed by `npm run stage:static` before dev/build so Vite does not
    // copy unrelated multi-megabyte run dumps into the deployment.
    publicDir: fileURLToPath(new URL("./.public", import.meta.url)),
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
  };
});
