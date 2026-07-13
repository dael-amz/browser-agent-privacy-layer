import { cpSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const TRANSFORMERS_DIST = path.dirname(require.resolve("@huggingface/transformers"));
const TRANSFORMERS_WASM = path.join(
  TRANSFORMERS_DIST,
  "ort-wasm-simd-threaded.jsep.wasm",
);
const TRANSFORMERS_WASM_URL =
  "/wasm/transformers-ort-wasm-simd-threaded.jsep.wasm";

function runtimeAssetFileName(asset) {
  const name = asset.names?.[0] ?? asset.name ?? "asset";
  if (name === "detector.onnx") return "visual/detector.onnx";
  if (name === "safety-best-thresholds.json") {
    return "visual/safety-best-thresholds.json";
  }
  if (name.endsWith(".wasm")) return "wasm/[name][extname]";
  return "assets/[name]-[hash][extname]";
}

export default defineConfig({
  publicDir: false,
  plugins: [localTransformersWasm(), copyRuntimeModels()],
  worker: {
    rollupOptions: {
      output: {
        assetFileNames: runtimeAssetFileName,
      },
    },
  },
  build: {
    assetsInlineLimit: 0,
    manifest: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        assetFileNames: runtimeAssetFileName,
      },
    },
  },
});

function copyRuntimeModels() {
  let outputDirectory;
  return {
    name: "plva-v3-runtime-models",
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
      const wasmDestination = path.join(
        outputDirectory,
        TRANSFORMERS_WASM_URL.slice(1),
      );
      mkdirSync(path.dirname(wasmDestination), { recursive: true });
      cpSync(TRANSFORMERS_WASM, wasmDestination);
    },
  };
}

function localTransformersWasm() {
  const virtualId = "\0plva-transformers-wasm";
  return {
    name: "plva-v3-transformers-wasm",
    enforce: "pre",
    resolveId(id) {
      if (id === "@plva-transformers-wasm") return virtualId;
      return null;
    },
    load(id) {
      if (id === virtualId) {
        return `export default ${JSON.stringify(TRANSFORMERS_WASM_URL)};`;
      }
      return null;
    },
  };
}
