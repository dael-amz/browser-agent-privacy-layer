#!/usr/bin/env node

import path from "node:path";
import { fileURLToPath } from "node:url";

import { verifyIntegrityManifest } from "./integrity.mjs";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const runtime = await verifyIntegrityManifest(root, "snapshot.json");
const source = await verifyIntegrityManifest(root, "source-integrity.json");
const executable = await verifyIntegrityManifest(root, "dist-integrity.json");
const baselineApp = await verifyIntegrityManifest(root, "baseline-app-dist-integrity.json");
process.stdout.write(
  `Verified ${runtime} frozen runtime files, ${source} source files, ${executable} harness files, and ${baselineApp} baseline app files.\n`,
);
