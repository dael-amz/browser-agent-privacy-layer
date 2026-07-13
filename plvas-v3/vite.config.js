import { createReadStream, cpSync, mkdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const RUNTIME_MODEL_MOUNTS = Object.freeze([
  { route: "/ocr/", source: path.join(ROOT, "models/ocr"), output: "ocr" },
  {
    route: "/semantic/rampart/",
    source: path.join(ROOT, "models/semantic/rampart"),
    output: "semantic/rampart",
  },
]);

export default defineConfig(() => {
  const includeEvaluation = process.env.PLVA_INCLUDE_EVALUATION === "1";
  return {
    publicDir: false,
    resolve: {
      alias: {
        "@plva-transformers-wasm": path.join(
          ROOT,
          "node_modules/@huggingface/transformers/dist/ort-wasm-simd-threaded.jsep.wasm",
        ) + "?url",
      },
    },
    plugins: [runtimeModelsPlugin()],
    build: {
      rollupOptions: includeEvaluation
        ? {
            input: {
              app: path.join(ROOT, "index.html"),
              evaluation: path.join(ROOT, "evaluation.html"),
            },
          }
        : undefined,
    },
  };
});

function runtimeModelsPlugin() {
  let outputDirectory;
  return {
    name: "plva-runtime-models",
    configResolved(config) {
      outputDirectory = path.resolve(config.root, config.build.outDir);
    },
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const pathname = decodeURIComponent(
          new URL(request.url ?? "/", "http://127.0.0.1").pathname,
        );
        const file = resolveRuntimeModel(pathname);
        if (!file) return next();
        let details;
        try {
          details = statSync(file);
        } catch {
          return next();
        }
        if (!details.isFile()) return next();
        response.statusCode = 200;
        response.setHeader("Content-Length", String(details.size));
        response.setHeader("Content-Type", contentType(file));
        createReadStream(file).pipe(response);
      });
    },
    closeBundle() {
      for (const mount of RUNTIME_MODEL_MOUNTS) {
        const destination = path.join(outputDirectory, mount.output);
        mkdirSync(path.dirname(destination), { recursive: true });
        cpSync(mount.source, destination, { recursive: true });
      }
    },
  };
}

function resolveRuntimeModel(pathname) {
  for (const mount of RUNTIME_MODEL_MOUNTS) {
    if (!pathname.startsWith(mount.route)) continue;
    const relative = pathname.slice(mount.route.length);
    const candidate = path.resolve(mount.source, relative);
    const inside = path.relative(mount.source, candidate);
    if (!inside || inside.startsWith("..") || path.isAbsolute(inside)) return null;
    return candidate;
  }
  return null;
}

function contentType(file) {
  if (file.endsWith(".json")) return "application/json; charset=utf-8";
  if (file.endsWith(".txt")) return "text/plain; charset=utf-8";
  return "application/octet-stream";
}
