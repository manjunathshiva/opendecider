import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    env: { MASTRA_TELEMETRY_DISABLED: "1" }, // agent frameworks report usage by default; the tests send nothing
  },
});
