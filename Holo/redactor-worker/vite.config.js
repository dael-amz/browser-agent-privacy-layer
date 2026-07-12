import { cpSync, existsSync, mkdirSync } from "node:fs";
import path from "node:path";

import { defineConfig } from "vite";

const BASELINE = path.resolve(
  process.env.PLVA_BASELINE_ROOT ?? "../plva-v2-baseline",
);
const V3_VISUAL_MODEL = path.resolve(
  "../plvas-v3/harness/plva-v2-baseline/runtime/training/artifacts/plva-visual-agpl-test-v2/visual/detector.onnx",
);
const VISUAL_MODEL = path.resolve(
  process.env.PLVA_VISUAL_MODEL ??
    (existsSync(V3_VISUAL_MODEL)
      ? V3_VISUAL_MODEL
      : path.join(
          BASELINE,
          "runtime/training/artifacts/plva-visual-agpl-test-v2/visual/detector.onnx",
        )),
);

export default defineConfig({
  publicDir: false,
  resolve: {
    alias: {
      "@plva-baseline": path.join(BASELINE, "runtime/src"),
      "@huggingface/transformers": path.resolve(
        "node_modules/@huggingface/transformers/dist/transformers.web.js",
      ),
      "@nationaldesignstudio/rampart": path.resolve(
        "node_modules/@nationaldesignstudio/rampart/dist/index.js",
      ),
      "@plva-transformers-wasm":
        path.resolve(
          "node_modules/@huggingface/transformers/dist/ort-wasm-simd-threaded.jsep.wasm",
        ) + "?url",
      "@plva-ort-accelerated": path.resolve(
        "node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs",
      ),
      "@plva-ort-ocr": path.resolve(
        "node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs",
      ),
    },
  },
  plugins: [selectableExecutionProvider(), copyRuntimeModels()],
  build: {
    assetsInlineLimit: 0,
    target: "esnext",
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

function selectableExecutionProvider() {
  const runtimeRoot = path.join(BASELINE, "runtime/src");
  return {
    name: "plva-selectable-execution-provider",
    enforce: "pre",
    transform(source, id) {
      if (!id.startsWith(runtimeRoot)) return null;
      if (!id.endsWith("visual/detector.js") && !id.endsWith("ocr/rapidocr.js")) {
        return null;
      }
      const visual = id.endsWith("visual/detector.js");
      const ortModule = visual ? "@plva-ort-accelerated" : "@plva-ort-ocr";
      const providerGlobal = visual
        ? "globalThis.__PLVA_VISUAL_PROVIDERS__"
        : "globalThis.__PLVA_OCR_PROVIDERS__";
      let replaced = source.replace(
        'from "onnxruntime-web";',
        `from "${ortModule}";`,
      ).replaceAll(
        'executionProviders: ["wasm"],',
        `executionProviders: ${providerGlobal} ?? ["wasm"],`,
      ).replaceAll(
        "ort.env.wasm.numThreads = globalThis.crossOriginIsolated\n      ? Math.max(1, Math.min(4, globalThis.navigator?.hardwareConcurrency ?? 1))\n      : 1;",
        "ort.env.wasm.numThreads = globalThis.__PLVA_WASM_THREADS__ ?? 1;",
      );
      if (id.endsWith("ocr/rapidocr.js")) {
        replaced = replaced.replace(
          `const [detector, recognizer, dictionary] = await Promise.all([\n    getDetector(),\n    getRecognizer(),\n    getDictionary(),\n  ]);`,
          `const detector = await getDetector();\n  const recognizer = await getRecognizer();\n  const dictionary = await getDictionary();`,
        );
      }
      if (!replaced.includes(providerGlobal)) {
        throw new Error(`could not patch execution provider in ${id}`);
      }
      return { code: replaced, map: null };
    },
  };
}

function copyRuntimeModels() {
  let outputDirectory;
  return {
    name: "plva-accelerated-runtime-models",
    configResolved(config) {
      outputDirectory = path.resolve(config.root, config.build.outDir);
    },
    closeBundle() {
      if (!existsSync(VISUAL_MODEL)) {
        throw new Error(`visual detector not found: ${VISUAL_MODEL}`);
      }
      const visualDestination = path.join(outputDirectory, "visual/detector.onnx");
      mkdirSync(path.dirname(visualDestination), { recursive: true });
      cpSync(VISUAL_MODEL, visualDestination);
      for (const relative of ["ocr", "semantic/rampart"]) {
        const source = path.join(BASELINE, "runtime/models", relative);
        const destination = path.join(outputDirectory, relative);
        mkdirSync(path.dirname(destination), { recursive: true });
        cpSync(source, destination, { recursive: true });
      }
    },
  };
}
