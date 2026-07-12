#!/usr/bin/env node

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const temporary = await mkdtemp(path.join(os.tmpdir(), "plva-v2-smoke-"));
const inputPath = path.join(root, "fixtures/ats-smoke.png");
const outputPath = path.join(temporary, "redacted.png");
const reportPath = path.join(temporary, "report.json");

try {
  const execution = await run([
    path.join(root, "bin/plva-v2.mjs"),
    inputPath,
    "--output",
    outputPath,
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

  const [input, output, reportBytes] = await Promise.all([
    readFile(inputPath),
    readFile(outputPath),
    readFile(reportPath),
  ]);
  const report = JSON.parse(reportBytes.toString("utf8"));
  const inputDimensions = pngDimensions(input);
  const outputDimensions = pngDimensions(output);

  assert.deepEqual(outputDimensions, inputDimensions, "redacted PNG dimensions changed");
  assert.equal(report.input.width, inputDimensions.width);
  assert.equal(report.input.height, inputDimensions.height);
  assert.equal(report.output.width, outputDimensions.width);
  assert.equal(report.output.height, outputDimensions.height);
  assert.ok(report.counts.fused > 0, "smoke fixture produced zero fused masks");
  assert.ok(report.regions.length > 0, "smoke report contains no final regions");
  assert.ok(report.integrity.maskedPixels > 0, "smoke run masked no pixels");
  assert.equal(report.integrity.outsideChangedPixels, 0);
  assert.equal(report.integrity.insideWrongColorPixels, 0);
  assert.equal(report.integrity.dimensionsMatch, true);
  assert.equal(report.integrity.passed, true);
  assert.equal(report.network.localOnly, true);
  assert.equal(report.network.externalNetworkBlocked, true);
  assert.equal(report.semanticMode, "rampart", "semantic model did not load in full mode");
  assert.equal(report.releaseEligible, false);
  assert.equal(
    report.models.visualDetector,
    "450ee07452d618eb3159d565f1787b17a8cabbfee3e72686e932460be576cc2e",
  );
  assert.notEqual(sha256(output), sha256(input), "redacted PNG is byte-identical to input");
  assertReportSanitized(report);

  process.stdout.write(
    `PASS: ${report.counts.fused} mask(s), ${report.integrity.maskedPixels} masked pixels, ` +
      `dimensions ${outputDimensions.width}x${outputDimensions.height}, ` +
      `outside changes ${report.integrity.outsideChangedPixels}, local-only runtime.\n`,
  );
} finally {
  await rm(temporary, { recursive: true, force: true });
}

function run(arguments_) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, arguments_, {
      cwd: root,
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

function pngDimensions(bytes) {
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  assert.ok(bytes.subarray(0, signature.length).equals(signature), "file is not a PNG");
  assert.equal(bytes.subarray(12, 16).toString("ascii"), "IHDR", "PNG has no IHDR");
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
}

function assertReportSanitized(value, trail = []) {
  if (Array.isArray(value)) {
    value.forEach((entry, index) => assertReportSanitized(entry, [...trail, index]));
    return;
  }
  if (!value || typeof value !== "object") {
    if (typeof value === "string") {
      assert.doesNotMatch(value, /\b[^\s@]+@[^\s@]+\.[^\s@]+\b/, "report leaked an email-like value");
      assert.doesNotMatch(value, /(?:\/Users\/|[A-Za-z]:\\\\)/, "report leaked a filesystem path");
    }
    return;
  }
  const forbidden = /^(text|recognizedText|ocrText|value|inputPath|outputPath|fileName|filename)$/i;
  for (const [key, entry] of Object.entries(value)) {
    assert.doesNotMatch(key, forbidden, `forbidden report field: ${[...trail, key].join(".")}`);
    assertReportSanitized(entry, [...trail, key]);
  }
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
