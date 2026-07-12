#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import { buildProfileReport } from "../src/evaluation/metrics.js";

const [inputArg, outputArg] = process.argv.slice(2);
if (!inputArg || !outputArg) {
  throw new Error("usage: node scripts/recompute-evaluation.mjs INPUT OUTPUT");
}

const input = path.resolve(inputArg);
const output = path.resolve(outputArg);
const raw = await readFile(input);
const report = JSON.parse(raw);
const profiles = report.profiles.map((profile) => {
  const fixtures = profile.fixtures.map((run) => ({
    ...run,
    degradations:
      run.degradations ??
      (run.warnings ?? []).filter(
        (warning) =>
          warning.includes("unavailable") ||
          warning.includes("heuristics-only"),
      ),
    outputIntegrity: {
      ...run.outputIntegrity,
      passed:
        run.outputIntegrity?.insideMismatch === 0 &&
        run.outputIntegrity?.outsideMismatch === 0 &&
        run.outputIntegrity?.encodedDimensionsMatch === true,
      truthCoveragePassed:
        (run.outputIntegrity?.minimumTruthBlackCoverage ?? 0) >= 0.98,
    },
  }));
  return buildProfileReport(profile.profile, fixtures);
});

const recomputed = {
  ...report,
  metricsVersion: 2,
  sourceReportSha256: createHash("sha256").update(raw).digest("hex"),
  recomputedAt: new Date().toISOString(),
  profiles,
};
await writeFile(output, `${JSON.stringify(recomputed, null, 2)}\n`);
