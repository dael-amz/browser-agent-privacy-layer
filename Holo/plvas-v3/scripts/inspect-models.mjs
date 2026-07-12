#!/usr/bin/env node

import { existsSync } from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const REPO_ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const candidates = [
  process.env.PLVA_PYTHON,
  path.join(REPO_ROOT, "training", ".venv", "bin", "python"),
  "python3",
].filter(Boolean);
const python = candidates.find(
  (candidate) => !candidate.includes(path.sep) || existsSync(candidate),
);
if (!python) {
  process.stderr.write("No Python interpreter found. Set PLVA_PYTHON.\n");
  process.exit(1);
}
const result = spawnSync(
  python,
  ["-m", "training.inspect_models", ...process.argv.slice(2)],
  { cwd: REPO_ROOT, stdio: "inherit", env: process.env },
);
if (result.error) throw result.error;
process.exit(result.status ?? 1);
