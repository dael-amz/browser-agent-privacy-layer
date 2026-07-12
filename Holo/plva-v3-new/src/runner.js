import { detectHybridSensitiveRegions } from "../runtime/src/visual/hybrid.js";
import {
  REDACTION_RGB,
  burnRegionsIntoCanvas,
  canvasToPngBlob,
  integerMask,
} from "../runtime/src/visual/render.js";
import { VISUAL_ARTIFACT_CONTRACT } from "../runtime/src/visual/model-contract.js";

globalThis.__PLVA_ORT_WASM_URL__ =
  "/wasm/plva-ort-wasm-simd-threaded.jsep.wasm";

const MAX_IMAGE_PIXELS = 16_000_000;
const MAX_IMAGE_DIMENSION = 16_384;
const INTEGRITY_STRIPE_HEIGHT = 256;

const status = document.querySelector("#status");
const parameters = new URLSearchParams(location.search);
const token = parameters.get("token") ?? "";
const profile = parameters.get("profile") ?? "";
const threads = Number(parameters.get("threads") ?? "4");

if (!/^[a-f0-9]{32}$/.test(token)) {
  fail(new Error("missing or invalid harness token"));
} else if (!new Set(["balanced", "high-recall"]).has(profile)) {
  fail(new Error("missing or invalid sensitivity profile"));
} else if (!Number.isInteger(threads) || threads < 1 || threads > 8) {
  fail(new Error("missing or invalid WASM thread count"));
} else {
  globalThis.__PLVA_WASM_THREADS__ = threads;
  run().catch(fail);
}

async function run() {
  setStatus("Loading screenshot…");
  const response = await fetch(`/__input/${token}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`input load failed (${response.status})`);
  const image = await createImageBitmap(await response.blob());
  if (
    image.width > MAX_IMAGE_DIMENSION ||
    image.height > MAX_IMAGE_DIMENSION ||
    image.width * image.height > MAX_IMAGE_PIXELS
  ) {
    image.close();
    throw new Error(
      `screenshot exceeds the supported ${MAX_IMAGE_DIMENSION}px / ${MAX_IMAGE_PIXELS}-pixel limit`,
    );
  }
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = image.width;
  sourceCanvas.height = image.height;
  const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
  if (!sourceContext) throw new Error("could not create the source canvas");
  sourceContext.drawImage(image, 0, 0);
  image.close();

  const result = await detectHybridSensitiveRegions(sourceCanvas, {
    profile,
    includeDiagnostics: true,
    onStage: setStatus,
  });

  setStatus("Burning permanent redactions…");
  const outputCanvas = document.createElement("canvas");
  burnRegionsIntoCanvas(sourceCanvas, outputCanvas, result.regions);
  const png = await canvasToPngBlob(outputCanvas);

  setStatus("Verifying redacted pixels…");
  const integrity = await verifyEncodedOutput(sourceCanvas, png, result.regions);
  if (!integrity.passed) {
    throw new Error("redacted PNG failed the pixel-integrity check");
  }

  const resources = performance
    .getEntriesByType("resource")
    .map((entry) => new URL(entry.name, location.href).origin);
  const network = {
    localOnly: resources.every((origin) => origin === location.origin),
    resourceCount: resources.length,
  };
  if (!network.localOnly) throw new Error("a non-loopback runtime resource was requested");

  await postBinary(`/__output/${token}`, png, "image/png");
  await postJson(`/__report/${token}`, {
    schemaVersion: 2,
    pipeline: "plva-v3",
    snapshot: "plva-visual-agpl-ats-v3-safe-head-v10",
    profile,
    dimensions: { width: sourceCanvas.width, height: sourceCanvas.height },
    counts: result.counts,
    semanticMode: result.semanticMode,
    warnings: result.warnings,
    degradations: result.degradations,
    timings: result.timings,
    regions: result.regions.map(sanitizeRegion),
    diagnostics: {
      visual: result.diagnostics.visual.map(sanitizeRegion),
      ocrSemantic: result.diagnostics.ocrSemantic.map(sanitizeRegion),
      ocrUncertain: result.diagnostics.ocrUncertain.map(sanitizeRegion),
    },
    integrity,
    network,
    runtime: {
      crossOriginIsolated: globalThis.crossOriginIsolated === true,
      requestedWasmThreads: threads,
      effectiveWasmThreads: globalThis.crossOriginIsolated ? threads : 1,
      visualEngine: result.visualRuntime.engine,
      visualTileCount: result.visualRuntime.tileCount,
      visualWorkerFallback: result.visualRuntime.workerFallback,
      visualRuntimePolicySha256: VISUAL_ARTIFACT_CONTRACT.runtimePolicySha256,
    },
  });
  setStatus("Complete.");
}

async function verifyEncodedOutput(sourceCanvas, png, regions) {
  const decodedImage = await createImageBitmap(png);
  const decodedCanvas = document.createElement("canvas");
  decodedCanvas.width = decodedImage.width;
  decodedCanvas.height = decodedImage.height;
  const decodedContext = decodedCanvas.getContext("2d", { willReadFrequently: true });
  if (!decodedContext) throw new Error("could not create the verification canvas");
  decodedContext.drawImage(decodedImage, 0, 0);
  decodedImage.close();

  const dimensionsMatch =
    decodedCanvas.width === sourceCanvas.width && decodedCanvas.height === sourceCanvas.height;
  if (!dimensionsMatch) {
    return {
      passed: false,
      dimensionsMatch: false,
      maskedPixels: 0,
      outsideChangedPixels: 0,
      insideWrongColorPixels: 0,
    };
  }

  const width = sourceCanvas.width;
  const height = sourceCanvas.height;
  const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
  if (!sourceContext) throw new Error("could not read source pixels for verification");
  const masks = regions.map((region) => integerMask(region, width, height));

  let maskedPixels = 0;
  let outsideChangedPixels = 0;
  let insideWrongColorPixels = 0;
  for (let stripeY = 0; stripeY < height; stripeY += INTEGRITY_STRIPE_HEIGHT) {
    const stripeHeight = Math.min(INTEGRITY_STRIPE_HEIGHT, height - stripeY);
    const source = sourceContext.getImageData(0, stripeY, width, stripeHeight).data;
    const decoded = decodedContext.getImageData(0, stripeY, width, stripeHeight).data;
    const mask = new Uint8Array(width * stripeHeight);
    for (const rectangle of masks) {
      const startY = Math.max(stripeY, rectangle.y);
      const endY = Math.min(stripeY + stripeHeight, rectangle.y + rectangle.height);
      for (let y = startY; y < endY; y += 1) {
        const row = (y - stripeY) * width;
        mask.fill(1, row + rectangle.x, row + rectangle.x + rectangle.width);
      }
    }
    for (let pixel = 0; pixel < mask.length; pixel += 1) {
      const offset = pixel * 4;
      if (mask[pixel]) {
        maskedPixels += 1;
        if (
          decoded[offset] !== REDACTION_RGB[0] ||
          decoded[offset + 1] !== REDACTION_RGB[1] ||
          decoded[offset + 2] !== REDACTION_RGB[2] ||
          decoded[offset + 3] !== 255
        ) {
          insideWrongColorPixels += 1;
        }
      } else if (
        decoded[offset] !== source[offset] ||
        decoded[offset + 1] !== source[offset + 1] ||
        decoded[offset + 2] !== source[offset + 2] ||
        decoded[offset + 3] !== source[offset + 3]
      ) {
        outsideChangedPixels += 1;
      }
    }
  }

  return {
    passed: outsideChangedPixels === 0 && insideWrongColorPixels === 0,
    dimensionsMatch,
    maskedPixels,
    outsideChangedPixels,
    insideWrongColorPixels,
  };
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
    sources: (region.sources ?? ["VISUAL"])
      .map((source) => String(source).slice(0, 120)),
    score: finite(region.score),
  };
}

function finite(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

async function postBinary(path, body, contentType) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": contentType },
    body,
  });
  if (!response.ok) throw new Error(`output delivery failed (${response.status})`);
}

async function postJson(path, value) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  });
  if (!response.ok) throw new Error(`report delivery failed (${response.status})`);
}

function setStatus(message) {
  if (status) status.textContent = String(message);
}

async function fail(error) {
  setStatus(`Failed: ${error?.message ?? error}`);
  if (!/^[a-f0-9]{32}$/.test(token)) return;
  try {
    await postJson(`/__error/${token}`, {
      name: String(error?.name ?? "Error").slice(0, 80),
      message: String(error?.message ?? error).slice(0, 500),
    });
  } catch {
    // The CLI will report a timeout or browser exit if it cannot receive this.
  }
}
