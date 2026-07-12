import { cpSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

const ROOT = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  publicDir: false,
  resolve: {
    alias: {
      "@plva-transformers-wasm":
        path.join(
          ROOT,
          "node_modules/@huggingface/transformers/dist/ort-wasm-simd-threaded.jsep.wasm",
        ) + "?url",
    },
  },
  plugins: [copyRuntimeModels()],
  build: {
    assetsInlineLimit: 0,
    manifest: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        assetFileNames(asset) {
          const name = asset.names?.[0] ?? asset.name ?? "asset";
          if (name === "detector.onnx") return "visual/detector.onnx";
          if (name.endsWith(".wasm")) return "wasm/[name][extname]";
          return "assets/[name]-[hash][extname]";
        },
      },
    },
  },
});

function copyRuntimeModels() {
  let outputDirectory;
  return {
    name: "plva-v2-baseline-runtime-models",
    configResolved(config) {
      outputDirectory = path.resolve(config.root, config.build.outDir);
    },
    closeBundle() {
      for (const relative of ["ocr", "semantic/rampart"]) {
        const source = path.join(ROOT, "runtime/models", relative);
        const destination = path.join(outputDirectory, relative);
        mkdirSync(path.dirname(destination), { recursive: true });
        cpSync(source, destination, { recursive: true });
      }
    },
  };
}
