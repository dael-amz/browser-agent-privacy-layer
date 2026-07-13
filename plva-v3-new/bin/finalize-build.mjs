#!/usr/bin/env node

import { createHash } from "node:crypto";
import {
  copyFile,
  mkdir,
  readFile,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const DIST = path.join(ROOT, "dist");
const WASM_DIRECTORY = path.join(DIST, "wasm");
const CANONICAL_NAME = "plva-ort-wasm-simd-threaded.jsep.wasm";
const EXPECTED_SHA256 =
  "78feeeb3d08f6bcee94d938ed322f69073bb8076b5f9d34697a574ffba8deb48";
const require = createRequire(import.meta.url);
const ortDirectory = path.dirname(require.resolve("onnxruntime-web"));
const source = path.join(ortDirectory, "ort-wasm-simd-threaded.jsep.wasm");

await mkdir(WASM_DIRECTORY, { recursive: true });
await copyFile(source, path.join(WASM_DIRECTORY, CANONICAL_NAME));

for (const name of await readdir(WASM_DIRECTORY)) {
  if (
    name !== CANONICAL_NAME &&
    name !== "transformers-ort-wasm-simd-threaded.jsep.wasm" &&
    /^ort-wasm-simd-threaded\d*\.jsep\.wasm$/u.test(name)
  ) {
    await rm(path.join(WASM_DIRECTORY, name));
  }
}

const canonical = await readFile(path.join(WASM_DIRECTORY, CANONICAL_NAME));
const actual = createHash("sha256").update(canonical).digest("hex");
if (actual !== EXPECTED_SHA256) {
  throw new Error(`canonical ONNX Runtime WASM SHA-256 mismatch: ${actual}`);
}

const manifestPath = path.join(DIST, ".vite", "manifest.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
for (const [key, value] of Object.entries(manifest)) {
  if (/onnxruntime-web\/dist\/ort-wasm-simd-threaded\.jsep\.wasm$/u.test(key)) {
    delete manifest[key];
    continue;
  }
  if (Array.isArray(value.assets)) {
    value.assets = value.assets.filter(
      (asset) => !/^wasm\/ort-wasm-simd-threaded\d*\.jsep\.wasm$/u.test(asset),
    );
  }
}
await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

const scripts = (await readdir(path.join(DIST, "assets")))
  .filter((name) => name.endsWith(".js"))
  .map((name) => path.join(DIST, "assets", name));
const builtJavaScript = (
  await Promise.all(scripts.map((file) => readFile(file, "utf8")))
).join("\n");
if (!builtJavaScript.includes(`/wasm/${CANONICAL_NAME}`)) {
  throw new Error("built runtime does not bind the canonical ONNX Runtime WASM URL");
}
