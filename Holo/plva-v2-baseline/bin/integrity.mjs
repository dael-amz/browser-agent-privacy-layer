import { createHash } from "node:crypto";
import { lstat, readFile, realpath } from "node:fs/promises";
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

  if (failures.length > 0) {
    throw new Error(`integrity verification failed:\n${failures.join("\n")}`);
  }
  return manifest.files.length;
}
