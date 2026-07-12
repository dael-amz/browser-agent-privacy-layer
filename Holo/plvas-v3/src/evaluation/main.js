import "./styles.css";

import {
  COVERAGE_THRESHOLD,
  VISUAL_PROPOSAL_THRESHOLD,
  buildProfileReport,
  intersectionOverTruth,
} from "./metrics.js";
import {
  syntheticValidationRaw,
  validationFixtures,
  webPiiQuick100Fixtures,
  webPiiQuick100Raw,
} from "./fixtures.js";
import { detectHybridSensitiveRegions } from "../visual/hybrid.js";
import { VISUAL_MODEL_URL } from "../visual/detector.js";
import { OCR_ASSET_URLS } from "../ocr/rapidocr.js";
import {
  REDACTION_RGB,
  burnRegionsIntoCanvas,
  canvasToPngBlob,
  integerMask,
} from "../visual/render.js";

const PROFILES = ["high-recall", "balanced"];
const runButton = document.querySelector("#run-evaluation");
const webPiiButton = document.querySelector("#run-webpii");
const downloadButton = document.querySelector("#download-report");
const status = document.querySelector("#evaluation-status");
const progress = document.querySelector("#evaluation-progress");
const summary = document.querySelector("#evaluation-summary");
const resultsBody = document.querySelector("#results-body");

let latestReport;

runButton.addEventListener("click", () =>
  void runEvaluation({
    fixtures: validationFixtures,
    profiles: PROFILES,
    name: "plva-screen-native-synthetic-validation",
    sourceRaw: syntheticValidationRaw,
    syntheticOnly: true,
  }),
);
webPiiButton.addEventListener("click", () =>
  void runEvaluation({
    fixtures: webPiiQuick100Fixtures,
    profiles: ["high-recall"],
    name: "webpii-published-test-quick100-diagnostic",
    sourceRaw: webPiiQuick100Raw,
    syntheticOnly: false,
  }),
);
downloadButton.addEventListener("click", downloadReport);

async function runEvaluation({ fixtures, profiles, name, sourceRaw, syntheticOnly }) {
  runButton.disabled = true;
  webPiiButton.disabled = true;
  downloadButton.disabled = true;
  summary.replaceChildren();
  resultsBody.replaceChildren();
  const startedAt = performance.now();
  const reports = [];

  try {
    let completed = 0;
    const total = fixtures.length * profiles.length;
    for (const profile of profiles) {
      const runs = [];
      for (const fixture of fixtures) {
        const fixtureLabel = fixture.template_id ?? fixture.page_type ?? fixture.id;
        setStatus(`Running ${profile}: ${fixtureLabel}`);
        progress.value = completed;
        progress.max = total;
        const canvas = await loadFixture(fixture.imageUrl);
        const runStarted = performance.now();
        const result = await detectHybridSensitiveRegions(canvas, {
          profile,
          includeDiagnostics: true,
          onStage: (stage) => setStatus(`${profile} · ${fixtureLabel} · ${stage}`),
        });
        const elapsedMs = Math.max(1, Math.round(performance.now() - runStarted));
        const outputIntegrity = await verifyBurnedOutput(
          canvas,
          result.regions,
          fixture.annotations,
        );
        runs.push({
          id: fixture.id,
          templateId: fixtureLabel,
          pageType: fixture.page_type ?? fixture.template_id,
          width: fixture.width,
          height: fixture.height,
          annotations: fixture.annotations,
          hardNegativeBoxes: fixture.hard_negative_boxes,
          elapsedMs,
          warnings: result.warnings,
          degradations: result.degradations,
          semanticMode: result.semanticMode,
          counts: result.counts,
          timings: result.timings,
          visual: result.diagnostics.visual,
          ocrSemantic: result.diagnostics.ocrSemantic,
          fused: result.regions,
          outputIntegrity,
        });
        canvas.width = 0;
        canvas.height = 0;
        completed += 1;
        progress.value = completed;
      }
      const profileReport = buildProfileReport(profile, runs);
      reports.push(profileReport);
      renderProfileSummary(profileReport);
      renderRows(profileReport);
    }

    latestReport = {
      schemaVersion: 1,
      generatedAt: new Date().toISOString(),
      runtime: {
        userAgent: navigator.userAgent,
        crossOriginIsolated: globalThis.crossOriginIsolated,
      },
      elapsedMs: Math.round(performance.now() - startedAt),
      fixtureSet: {
        name,
        records: fixtures.length,
        syntheticOnly,
        sha256: await sha256Text(sourceRaw),
      },
      runtimeAssets: await hashRuntimeAssets(),
      networkOrigins: [...new Set(
        performance
          .getEntriesByType("resource")
          .map((entry) => new URL(entry.name, location.href).origin),
      )].sort(),
      profiles: reports,
    };
    const allPassed = reports.every((report) => report.gates.passed);
    setStatus(
      `${allPassed ? "PASS" : "FAIL"} · ${total} full-pipeline runs · ${latestReport.elapsedMs} ms`,
      allPassed ? "pass" : "fail",
    );
    downloadButton.disabled = false;
  } catch (error) {
    console.error(error);
    setStatus(`Evaluation failed: ${friendlyError(error)}`, "fail");
  } finally {
    runButton.disabled = false;
    webPiiButton.disabled = false;
  }
}

function renderProfileSummary(report) {
  const { fused, visual, visualProposal, ocrSemantic } = report.metrics;
  const card = document.createElement("article");
  card.className = `metric-card ${report.gates.passed ? "pass" : "fail"}`;
  card.innerHTML = `
    <p>${escapeHtml(report.profile)}</p>
    <strong>${formatPercent(fused.recall)} fused recall</strong>
    <span>Visual final ${formatPercent(visual.recall)} · visual proposals ${formatPercent(visualProposal.recall)} · OCR-semantic ${formatPercent(ocrSemantic.recall)}</span>
    <span>${fused.secretMisses} secret misses · ${fused.hardNegativeRecords ? `${fused.hardNegativeFalseMaskRecords}/${fused.hardNegativeRecords} clean screens falsely masked` : "no clean-screen records"}</span>
    <span>${formatPercent(fused.proposalPrecision)} compatible precision · ${fused.uncertaintyMasks} uncertainty masks</span>
    <span>Median ${fused.latencyMs.median} ms · p95 ${fused.latencyMs.p95} ms</span>
  `;
  summary.append(card);
}

function renderRows(report) {
  for (const run of report.fixtures) {
    const row = document.createElement("tr");
    const truth = run.annotations.length;
    row.innerHTML = `
      <td>${escapeHtml(report.profile)}</td>
      <td>${escapeHtml(run.templateId)}</td>
      <td>${truth ? run.annotations.map((item) => escapeHtml(item.source_class ?? item.class)).join(", ") : "HARD NEGATIVE"}</td>
      <td>${coverageCount(run.visual, run.annotations, VISUAL_PROPOSAL_THRESHOLD)}/${truth}</td>
      <td>${coverageCount(run.ocrSemantic, run.annotations, COVERAGE_THRESHOLD)}/${truth}</td>
      <td>${coverageCount(run.fused, run.annotations, COVERAGE_THRESHOLD)}/${truth}</td>
      <td>${run.fused.length}</td>
      <td>${escapeHtml([...new Set(run.fused.flatMap((item) => item.sources))].join(" + ") || "—")}</td>
      <td>${run.elapsedMs}</td>
      <td>${run.warnings.length ? escapeHtml(run.warnings.join("; ")) : "—"}</td>
    `;
    resultsBody.append(row);
  }
}

async function loadFixture(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`fixture image returned ${response.status}`);
  const bitmap = await createImageBitmap(await response.blob());
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("2D canvas is unavailable");
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  return canvas;
}

async function verifyBurnedOutput(source, regions, annotations) {
  const output = document.createElement("canvas");
  burnRegionsIntoCanvas(source, output, regions);
  const sourceContext = source.getContext("2d", { willReadFrequently: true });
  const context = output.getContext("2d", { willReadFrequently: true });
  if (!sourceContext || !context) {
    return { passed: false, reason: "2D canvas unavailable" };
  }
  const sourcePixels = sourceContext.getImageData(0, 0, source.width, source.height).data;
  const outputPixels = context.getImageData(0, 0, output.width, output.height).data;
  const mask = new Uint8Array(source.width * source.height);
  for (const region of regions) {
    const rectangle = integerMask(region, source.width, source.height);
    for (let y = rectangle.y; y < rectangle.y + rectangle.height; y += 1) {
      mask.fill(1, y * source.width + rectangle.x, y * source.width + rectangle.x + rectangle.width);
    }
  }
  let insideMismatch = 0;
  let outsideMismatch = 0;
  for (let pixel = 0; pixel < mask.length; pixel += 1) {
    const offset = pixel * 4;
    if (mask[pixel]) {
      if (
        outputPixels[offset] !== REDACTION_RGB[0] ||
        outputPixels[offset + 1] !== REDACTION_RGB[1] ||
        outputPixels[offset + 2] !== REDACTION_RGB[2] ||
        outputPixels[offset + 3] !== 255
      ) {
        insideMismatch += 1;
      }
    } else if (
      outputPixels[offset] !== sourcePixels[offset] ||
      outputPixels[offset + 1] !== sourcePixels[offset + 1] ||
      outputPixels[offset + 2] !== sourcePixels[offset + 2] ||
      outputPixels[offset + 3] !== sourcePixels[offset + 3]
    ) {
      outsideMismatch += 1;
    }
  }
  const coverages = annotations.map((annotation) => {
    const [x1, y1, x2, y2] = annotation.bbox_xyxy.map(Math.round);
    const width = Math.max(1, x2 - x1);
    const height = Math.max(1, y2 - y1);
    const pixels = context.getImageData(x1, y1, width, height).data;
    let black = 0;
    for (let offset = 0; offset < pixels.length; offset += 4) {
      if (pixels[offset] < 15 && pixels[offset + 1] < 15 && pixels[offset + 2] < 15) {
        black += 1;
      }
    }
    return black / (pixels.length / 4);
  });
  const blob = await canvasToPngBlob(output);
  const decoded = await createImageBitmap(blob);
  const encodedDimensionsMatch =
    decoded.width === source.width && decoded.height === source.height;
  decoded.close();
  return {
    passed:
      insideMismatch === 0 &&
      outsideMismatch === 0 &&
      encodedDimensionsMatch,
    truthCoveragePassed: coverages.every((coverage) => coverage >= 0.98),
    minimumTruthBlackCoverage: coverages.length ? Math.min(...coverages) : 1,
    insideMismatch,
    outsideMismatch,
    encodedDimensionsMatch,
    pngBytes: blob.size,
    width: output.width,
    height: output.height,
  };
}

function coverageCount(regions, annotations, threshold) {
  return annotations.filter((annotation) =>
    regions.some(
      (region) => intersectionOverTruth(region, annotation.bbox_xyxy) >= threshold,
    ),
  ).length;
}

function downloadReport() {
  if (!latestReport) return;
  const blob = new Blob([JSON.stringify(latestReport, null, 2) + "\n"], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "plva-full-pipeline-evaluation.json";
  link.click();
  URL.revokeObjectURL(url);
}

function setStatus(message, state = "running") {
  status.textContent = message;
  status.dataset.state = state;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function friendlyError(error) {
  const message = error instanceof Error ? error.message : String(error);
  return message.length > 180 ? `${message.slice(0, 177)}…` : message;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function hashRuntimeAssets() {
  const assets = {
    visualDetector: VISUAL_MODEL_URL,
    ocrDetector: OCR_ASSET_URLS.detector,
    ocrRecognizer: OCR_ASSET_URLS.recognizer,
    ocrDictionary: OCR_ASSET_URLS.dictionary,
    rampartQ4: "/semantic/rampart/onnx/model_q4.onnx",
  };
  const output = {};
  for (const [name, url] of Object.entries(assets)) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`runtime asset ${name} returned ${response.status}`);
    const bytes = await response.arrayBuffer();
    output[name] = {
      url,
      bytes: bytes.byteLength,
      sha256: toHex(await crypto.subtle.digest("SHA-256", bytes)),
    };
  }
  return output;
}

async function sha256Text(value) {
  const bytes = new TextEncoder().encode(value);
  return toHex(await crypto.subtle.digest("SHA-256", bytes));
}

function toHex(buffer) {
  return [...new Uint8Array(buffer)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}
