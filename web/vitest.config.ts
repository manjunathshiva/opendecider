import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    testTimeout: 60_000, // the model tests run a small ONNX model in WebAssembly
  },
});
