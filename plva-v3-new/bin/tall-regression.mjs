#!/usr/bin/env node

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const temporary = await mkdtemp(path.join(os.tmpdir(), "plva-v3-tall-"));
const input = path.join(ROOT, "fixtures", "webpii-tall-cart.png");
const truthPath = path.join(ROOT, "fixtures", "webpii-tall-cart.truth.json");
const output = path.join(temporary, "redacted.png");
const reportPath = path.join(temporary, "report.json");

try {
  const execution = await run([
    path.join(ROOT, "bin", "plva-v3.mjs"),
    input,
    "--output",
    output,
    "--report",
    reportPath,
    "--profile",
    "high-recall",
    "--timeout-ms",
    "300000",
  ]);
  assert.equal(
    execution.code,
    0,
    `CLI exited ${execution.code}.\nstdout:\n${execution.stdout}\nstderr:\n${execution.stderr}`,
  );

  const [report, truth] = await Promise.all([
    readFile(reportPath, "utf8").then(JSON.parse),
    readFile(truthPath, "utf8").then(JSON.parse),
  ]);
  assert.equal(report.input.width, truth.width);
  assert.equal(report.input.height, truth.height);
  const coverages = truth.annotations.map((annotation) =>
    Math.max(
      0,
      ...report.regions.map((region) =>
        intersectionOverTruth(region, annotation.bbox_xyxy),
      ),
    ),
  );
  assert.ok(
    coverages.every((coverage) => coverage >= 0.98),
    `tall-page truth coverage regressed: ${coverages.join(", ")}`,
  );
  assert.equal(report.runtime.visualEngine, "worker");
  assert.equal(report.runtime.visualTileCount, 1);
  assert.equal(report.runtime.visualWorkerFallback, false);
  assert.equal(report.integrity.passed, true);
  assert.ok(
    report.diagnostics.ocrSemantic.some((region) =>
      region.sources.includes("OCR+CONTEXT_RULE"),
    ),
    "tall-page regression did not exercise contextual OCR rules",
  );

  process.stdout.write(
    `PASS: ${coverages.length}/${truth.annotations.length} tall-page truths covered at >=98%, ` +
      `${report.counts.fused} final masks, ${report.timings.totalMs} ms.\n`,
  );
} finally {
  await rm(temporary, { recursive: true, force: true });
}

function run(arguments_) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, arguments_, {
      cwd: ROOT,
      env: { ...process.env, PLVA_DEBUG: "0", PLVA_CHROME_DEBUG: "0" },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.once("error", reject);
    child.once("exit", (code) => resolve({ code, stdout, stderr }));
  });
}

function intersectionOverTruth(region, truth) {
  const intersection =
    Math.max(0, Math.min(region.x2, truth[2]) - Math.max(region.x1, truth[0])) *
    Math.max(0, Math.min(region.y2, truth[3]) - Math.max(region.y1, truth[1]));
  const area = (truth[2] - truth[0]) * (truth[3] - truth[1]);
  return area > 0 ? intersection / area : 0;
}
