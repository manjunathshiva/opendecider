import { defineConfig } from "vite";

// Served from the docs site's /demo/ folder (GitHub Pages: no cross-origin isolation, so WebAssembly runs one thread)
export default defineConfig({ base: "./", build: { outDir: "dist", emptyOutDir: true, target: "es2022" } });
