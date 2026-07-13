#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const targets = [["dist", "dist-integrity.json"]];

for (const [directory, output] of targets) {
  const files = await inventory(path.join(root, directory));
  await writeFile(
    path.join(root, output),
    `${JSON.stringify({ schemaVersion: 2, sealedRoots: [directory], files }, null, 2)}\n`,
  );
}

const snapshotPath = path.join(root, "snapshot.json");
const snapshot = JSON.parse(await readFile(snapshotPath, "utf8"));
snapshot.files = await inventory(path.join(root, "runtime"));
snapshot.sealedRoots = ["runtime"];
await writeFile(snapshotPath, `${JSON.stringify(snapshot, null, 2)}\n`);

const sourcePaths = [
  ...(await walk(path.join(root, "bin"))),
  ...(await walk(path.join(root, "src"))),
  ...(await walk(path.join(root, "fixtures"))),
  ...[
    ".gitignore",
    "BENCHMARK.md",
    "index.html",
    "LICENSES.md",
    "package-lock.json",
    "package.json",
    "PROVENANCE.md",
    "README.md",
    "SMOKE_TEST.md",
    "snapshot.json",
    "vite.config.js",
  ].map((file) => path.join(root, file)),
];
const sourceFiles = await inventoryFiles(sourcePaths);
await writeFile(
  path.join(root, "source-integrity.json"),
  `${JSON.stringify({ schemaVersion: 2, sealedRoots: ["bin", "src", "fixtures"], files: sourceFiles }, null, 2)}\n`,
);

const sumRoots = ["runtime", "dist"];
const sums = [];
for (const directory of sumRoots) {
  for (const entry of await inventory(path.join(root, directory))) {
    sums.push(`${entry.sha256}  ${entry.path}`);
  }
}
for (const entry of sourceFiles) sums.push(`${entry.sha256}  ${entry.path}`);
for (const file of [
  "dist-integrity.json",
  "source-integrity.json",
]) {
  const [entry] = await inventoryFiles([path.join(root, file)]);
  sums.push(`${entry.sha256}  ${entry.path}`);
}
await writeFile(path.join(root, "SHA256SUMS"), `${sums.sort().join("\n")}\n`);

async function inventory(directory) {
  return inventoryFiles(await walk(directory));
}

async function inventoryFiles(paths) {
  const entries = [];
  for (const file of paths.sort()) {
    const bytes = await readFile(file);
    entries.push({
      path: path.relative(root, file),
      bytes: bytes.length,
      sha256: createHash("sha256").update(bytes).digest("hex"),
    });
  }
  return entries;
}

async function walk(directory) {
  const output = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const candidate = path.join(directory, entry.name);
    if (entry.isDirectory()) output.push(...(await walk(candidate)));
    else if (entry.isFile() && (await stat(candidate)).isFile()) output.push(candidate);
  }
  return output;
}
