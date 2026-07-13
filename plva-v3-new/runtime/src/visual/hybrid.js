import {
  classifyOcrRegions,
  warmOcrSemanticClassifier,
} from "../ocr/semantic.js";
import { recognizeScreenshotText, warmRapidOcr } from "../ocr/rapidocr.js";
import { detectSensitiveRegions, warmVisualDetector } from "./detector.js";
import { fuseSensitiveRegions } from "./fusion.js";

export async function detectHybridSensitiveRegions(
  sourceCanvas,
  {
    profile = "high-recall",
    onStage = () => {},
    includeDiagnostics = false,
  } = {},
) {
  const pipelineStarted = performance.now();
  const timings = {};
  const warnings = [];
  const degradations = [];
  let visual = [];
  let ocr = { regions: [], uncertainRegions: [], detectedCount: 0 };
  let semantic = { regions: [], mode: "not-run", warning: null };
  let visualRuntime = {
    engine: "unavailable",
    tileCount: 0,
    workerFallback: false,
  };
  let visualError;
  let ocrError;

  // Fetch and initialize independent models together, but do not run two
  // onnxruntime-web WASM sessions concurrently: ORT shares native run state in
  // one page. This overlaps cold loading while preserving deterministic runs.
  onStage("Loading local visual, OCR, and semantic models…");
  const loadStarted = performance.now();
  const semanticWarmup = warmOcrSemanticClassifier().catch(() => null);
  const [visualWarmup] = await Promise.allSettled([
    warmVisualDetector(profile),
    warmRapidOcr(),
  ]);
  timings.modelLoadMs = Math.round(performance.now() - loadStarted);

  const visualOperation = async () => {
    const visualStarted = performance.now();
    const detection = await detectSensitiveRegions(sourceCanvas, {
      profile,
      onStage: (message) => onStage(`Visual: ${lowerFirst(message)}`),
    });
    return {
      detection,
      milliseconds: Math.round(performance.now() - visualStarted),
    };
  };
  const ocrOperation = async () => {
    const ocrStarted = performance.now();
    const result = await recognizeScreenshotText(sourceCanvas, { onStage });
    return { result, milliseconds: Math.round(performance.now() - ocrStarted) };
  };

  let visualOutcome;
  let ocrOutcome;
  if (visualWarmup.status === "fulfilled" && visualWarmup.value === "worker") {
    [visualOutcome, ocrOutcome] = await Promise.allSettled([
      visualOperation(),
      ocrOperation(),
    ]);
  } else {
    visualOutcome = await settle(visualOperation());
    ocrOutcome = await settle(ocrOperation());
  }

  if (visualOutcome.status === "fulfilled") {
    visual = visualOutcome.value.detection.regions;
    visualRuntime = {
      engine: visualOutcome.value.detection.engine,
      tileCount: visualOutcome.value.detection.tileCount,
      workerFallback: visualOutcome.value.detection.workerFallback,
    };
    timings.visualMs = visualOutcome.value.milliseconds;
    if (visualRuntime.workerFallback) {
      warnings.push("visual worker failed; detector retried successfully on the main runtime");
    }
  } else {
    visualError = visualOutcome.reason;
    warnings.push("visual detector unavailable");
    degradations.push("visual detector unavailable");
  }

  if (ocrOutcome.status === "fulfilled") {
    ocr = ocrOutcome.value.result;
    timings.ocrMs = ocrOutcome.value.milliseconds;
    await semanticWarmup;
    const semanticStarted = performance.now();
    semantic = await classifyOcrRegions(ocr.regions, { onStage });
    timings.semanticMs = Math.round(performance.now() - semanticStarted);
    if (semantic.warning) {
      warnings.push(semantic.warning);
      degradations.push(semantic.warning);
    }
    if (ocr.uncertainRegions.length > 0) {
      warnings.push(
        `${ocr.uncertainRegions.length} low-confidence OCR region${ocr.uncertainRegions.length === 1 ? "" : "s"} ${profile === "high-recall" ? "masked as UNREADABLE" : "not masked in balanced mode"}`,
      );
    }
  } else {
    ocrError = ocrOutcome.reason;
    warnings.push("OCR path unavailable");
    degradations.push("OCR path unavailable");
  }

  if (visualError && ocrError) {
    throw new AggregateError(
      [visualError, ocrError],
      "both visual and OCR redaction paths failed",
    );
  }

  onStage("Fusing visual and OCR evidence…");
  const fusionStarted = performance.now();
  const acceptedOcr = [
    ...semantic.regions,
    ...(profile === "high-recall" ? ocr.uncertainRegions : []),
  ];
  const regions = fuseSensitiveRegions(visual, acceptedOcr);
  timings.fusionMs = Math.round(performance.now() - fusionStarted);
  timings.totalMs = Math.round(performance.now() - pipelineStarted);
  const result = {
    regions,
    counts: {
      visual: visual.length,
      ocrText: ocr.regions.length,
      ocrDetected: ocr.detectedCount,
      ocrUncertain: ocr.uncertainRegions.length,
      ocrSensitive: semantic.regions.length,
      fused: regions.length,
    },
    semanticMode: semantic.mode,
    visualRuntime,
    warnings,
    degradations,
    timings,
  };
  if (includeDiagnostics) {
    // Evaluation-only geometry: no recognized OCR text is retained here.
    result.diagnostics = {
      visual: visual.map(sanitizeRegion),
      ocrSemantic: semantic.regions.map(sanitizeRegion),
      ocrUncertain: ocr.uncertainRegions.map(sanitizeRegion),
    };
  }
  return result;
}

function lowerFirst(text) {
  return text ? text[0].toLowerCase() + text.slice(1) : text;
}

function settle(promise) {
  return promise.then(
    (value) => ({ status: "fulfilled", value }),
    (reason) => ({ status: "rejected", reason }),
  );
}

function sanitizeRegion(region) {
  return {
    x1: region.x1,
    y1: region.y1,
    x2: region.x2,
    y2: region.y2,
    label: region.label,
    labels: region.labels ?? [region.label],
    sources: region.sources ?? ["VISUAL"],
    score: region.score,
  };
}
