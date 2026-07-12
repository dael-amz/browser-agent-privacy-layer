import { classifyOcrRegions } from "@plva-baseline/ocr/semantic.js";
import { recognizeScreenshotText } from "@plva-baseline/ocr/rapidocr.js";
import { getRampartClassifier } from "@plva-baseline/rampart.js";
import { detectSensitiveRegions } from "@plva-baseline/visual/detector.js";
import { fuseSensitiveRegions } from "@plva-baseline/visual/fusion.js";
import {
  burnRegionsIntoCanvas,
  canvasToPngBlob,
} from "@plva-baseline/visual/render.js";
import { env as transformersEnv } from "@huggingface/transformers";

const parameters = new URLSearchParams(location.search);
const requestedBackend = parameters.get("backend") ?? "auto";
const token = parameters.get("token") ?? "";
const status = document.querySelector("#status");
const activeBackend = chooseBackend(requestedBackend);

globalThis.__PLVA_VISUAL_PROVIDERS__ = visualProvidersFor(activeBackend);
globalThis.__PLVA_OCR_PROVIDERS__ = ["wasm"];
globalThis.__PLVA_WASM_THREADS__ = 1;
transformersEnv.backends.onnx.wasm.numThreads = 1;

run().catch(async (error) => {
  if (parameters.get("debug") === "1") console.error(error);
  setStatus("Worker failed");
  await postJson(secured("/__fatal"), { error: safeError(error) }).catch(() => {});
});

async function run() {
  setStatus(`Warming ${activeBackend} models…`);
  await warmModels(activeBackend);

  await postJson(secured("/__ready"), {
    backend: activeBackend,
    crossOriginIsolated: globalThis.crossOriginIsolated === true,
  });

  while (true) {
    const response = await fetch(secured("/__job"), { cache: "no-store" });
    if (response.status === 204) continue;
    if (!response.ok) throw new Error(`job poll failed (${response.status})`);
    const job = await response.json();
    await processAndPost(job);
  }
}

async function warmModels(backend) {
  const canvas = document.createElement("canvas");
  canvas.width = 960;
  canvas.height = 960;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("could not create warm-up canvas");
  context.fillStyle = "white";
  context.fillRect(0, 0, canvas.width, canvas.height);
  // Session creation is serialized because ORT's WASM initializer is not re-entrant.
  // Once initialized, per-frame visual and OCR inference runs concurrently below.
  debugLog("warm visual start");
  await detectSensitiveRegions(canvas, { profile: "high-recall" });
  debugLog("warm visual done; OCR start");
  await recognizeScreenshotText(canvas);
  debugLog("warm OCR done; semantic start");
  await getRampartClassifier("wasm");
  debugLog("warm semantic done");
  setStatus(`${backend} models ready`);
}

async function processAndPost(job) {
  try {
    const started = performance.now();
    const response = await fetch(secured(`/__input/${encodeURIComponent(job.id)}`), {
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`input fetch failed (${response.status})`);
    const image = await createImageBitmap(await response.blob());
    const source = document.createElement("canvas");
    source.width = image.width;
    source.height = image.height;
    const context = source.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("could not create source canvas");
    context.drawImage(image, 0, 0);
    image.close();

    const result = await detectParallel(source, job.profile);
    const output = document.createElement("canvas");
    burnRegionsIntoCanvas(source, output, result.regions);
    const png = await canvasToPngBlob(output);
    const totalMs = Math.round(performance.now() - started);

    await postBinary(secured(`/__output/${encodeURIComponent(job.id)}`), png, "image/png");
    await postJson(secured(`/__report/${encodeURIComponent(job.id)}`), {
      backend: activeBackend,
      dimensions: { width: source.width, height: source.height },
      counts: result.counts,
      semanticMode: result.semanticMode,
      timings: { ...result.timings, workerTotalMs: totalMs },
      regions: result.regions.map(sanitizeRegion),
    });
    setStatus(`Completed frame with ${activeBackend}`);
  } catch (error) {
    if (parameters.get("debug") === "1") console.error(error);
    await postJson(secured(`/__error/${encodeURIComponent(job.id)}`), {
      error: safeError(error),
    });
  }
}

async function detectParallel(source, profile) {
  const pipelineStarted = performance.now();
  const visualStarted = performance.now();
  const visualPromise = detectSensitiveRegions(source, { profile }).then((regions) => ({
    regions,
    elapsed: Math.round(performance.now() - visualStarted),
  }));

  const ocrStarted = performance.now();
  const ocrPromise = recognizeScreenshotText(source).then((ocr) => ({
    ocr,
    ocrMs: Math.round(performance.now() - ocrStarted),
  }));

  // Both branches are mandatory: rejection of either branch fails the whole frame closed.
  const [visual, text] = await Promise.all([visualPromise, ocrPromise]);
  const semanticStarted = performance.now();
  const semantic = await classifyOcrRegions(text.ocr.regions, { device: "wasm" });
  const semanticMs = Math.round(performance.now() - semanticStarted);
  if (semantic.warning) throw new Error("semantic classifier degraded");
  const acceptedOcr = [
    ...semantic.regions,
    ...(profile === "high-recall" ? text.ocr.uncertainRegions : []),
  ];
  const regions = fuseSensitiveRegions(visual.regions, acceptedOcr);
  return {
    regions,
    semanticMode: semantic.mode,
    counts: {
      visual: visual.regions.length,
      ocrText: text.ocr.regions.length,
      ocrDetected: text.ocr.detectedCount,
      ocrUncertain: text.ocr.uncertainRegions.length,
      ocrSensitive: semantic.regions.length,
      fused: regions.length,
    },
    timings: {
      visualMs: visual.elapsed,
      ocrMs: text.ocrMs,
      semanticMs,
      totalMs: Math.round(performance.now() - pipelineStarted),
    },
  };
}

function chooseBackend(requested) {
  if (requested === "wasm") return "wasm";
  if (requested === "webgpu") {
    if (!navigator.gpu) throw new Error("WebGPU was requested but is unavailable");
    return "webgpu";
  }
  return navigator.gpu ? "webgpu" : "wasm";
}

function visualProvidersFor(backend) {
  return backend === "webgpu" ? ["webgpu"] : ["wasm"];
}

function sanitizeRegion(region) {
  return {
    x1: finite(region.x1),
    y1: finite(region.y1),
    x2: finite(region.x2),
    y2: finite(region.y2),
    label: String(region.label ?? "UNKNOWN").slice(0, 120),
    labels: (region.labels ?? [region.label ?? "UNKNOWN"])
      .map((label) => String(label).slice(0, 120)),
    sources: (region.sources ?? ["UNKNOWN"])
      .map((source) => String(source).slice(0, 120)),
    score: finite(region.score),
  };
}

function finite(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function safeError(error) {
  return String(error?.name ?? "Error").slice(0, 80);
}

function setStatus(message) {
  if (status) status.textContent = message;
}

function debugLog(message) {
  if (parameters.get("debug") === "1") console.log(`[plva-worker] ${message}`);
}

function secured(path) {
  return `${path}?token=${encodeURIComponent(token)}`;
}

async function postBinary(path, body, contentType) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": contentType },
    body,
  });
  if (!response.ok) throw new Error(`binary post failed (${response.status})`);
}

async function postJson(path, value) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  });
  if (!response.ok) throw new Error(`JSON post failed (${response.status})`);
}
