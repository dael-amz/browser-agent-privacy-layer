#!/usr/bin/env node

import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  loadLock,
  parseCli,
  resolveAssetRoot,
  selectAssets,
  verifyAssetFile,
} from "./model-lock.mjs";

const REPO_ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

function usage() {
  return `Usage: node scripts/verify-models.mjs [options]

Options match fetch-models.mjs. Verification never downloads or changes files.
`;
}

async function main() {
  const options = parseCli(process.argv.slice(2));
  if (options.help) {
    process.stdout.write(usage());
    return;
  }
  const lock = await loadLock(
    options.lock ?? path.join(REPO_ROOT, "models.lock.json"),
  );
  const root = resolveAssetRoot(lock, options.output);
  const selected = selectAssets(lock.data, options);
  const results = [];
  for (const [name, asset] of selected) {
    results.push(await verifyAssetFile(name, asset, root));
  }
  const ok = results.every((result) => result.ok);
  process.stdout.write(
    `${JSON.stringify({ ok, asset_root: root, assets: results }, null, 2)}\n`,
  );
  if (!ok) process.exitCode = 1;
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error.message}\n`);
  process.exitCode = 1;
});
