import { createHash } from "node:crypto";
import { lstat, readFile, readdir, realpath } from "node:fs/promises";
import path from "node:path";

export async function verifyIntegrityManifest(root, manifestName) {
  const manifestPath = path.join(root, manifestName);
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  const failures = [];

  for (const entry of manifest.files ?? []) {
    const file = path.resolve(root, entry.path);
    const inside = path.relative(root, file);
    if (!inside || inside.startsWith("..") || path.isAbsolute(inside)) {
      failures.push(`${entry.path}: unsafe manifest path`);
      continue;
    }
    try {
      const details = await lstat(file);
      if (details.isSymbolicLink()) throw new Error("symbolic links are forbidden");
      if (!details.isFile()) throw new Error("not a regular file");
      const resolved = await realpath(file);
      const resolvedInside = path.relative(await realpath(root), resolved);
      if (!resolvedInside || resolvedInside.startsWith("..") || path.isAbsolute(resolvedInside)) {
        throw new Error("resolved path escapes the harness root");
      }
      const bytes = await readFile(file);
      const digest = createHash("sha256").update(bytes).digest("hex");
      if (details.size !== entry.bytes) {
        failures.push(`${entry.path}: expected ${entry.bytes} bytes, got ${details.size}`);
      }
      if (digest !== entry.sha256) {
        failures.push(`${entry.path}: SHA-256 mismatch`);
      }
    } catch (error) {
      failures.push(`${entry.path}: ${error.message}`);
    }
  }

  const listed = new Set((manifest.files ?? []).map((entry) => entry.path));
  for (const sealedRoot of manifest.sealedRoots ?? []) {
    const directory = path.resolve(root, sealedRoot);
    const inside = path.relative(root, directory);
    if (!inside || inside.startsWith("..") || path.isAbsolute(inside)) {
      failures.push(`${sealedRoot}: unsafe sealed root`);
      continue;
    }
    for (const file of await walk(directory)) {
      const relative = path.relative(root, file);
      if (!listed.has(relative)) failures.push(`${relative}: unlisted file in sealed root`);
    }
  }

  if (failures.length > 0) {
    throw new Error(`integrity verification failed:\n${failures.join("\n")}`);
  }
  return manifest.files.length;
}

async function walk(directory) {
  const output = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const candidate = path.join(directory, entry.name);
    if (entry.isDirectory()) output.push(...(await walk(candidate)));
    else output.push(candidate);
  }
  return output;
}
