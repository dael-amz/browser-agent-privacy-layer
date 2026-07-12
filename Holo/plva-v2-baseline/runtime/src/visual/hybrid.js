import { classifyOcrRegions } from "../ocr/semantic.js";
import { recognizeScreenshotText } from "../ocr/rapidocr.js";
import { detectSensitiveRegions } from "./detector.js";
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
  let visualError;
  let ocrError;

  try {
    const visualStarted = performance.now();
    visual = await detectSensitiveRegions(sourceCanvas, {
      profile,
      onStage: (message) => onStage(`Visual: ${lowerFirst(message)}`),
    });
    timings.visualMs = Math.round(performance.now() - visualStarted);
  } catch (error) {
    visualError = error;
    warnings.push("visual detector unavailable");
    degradations.push("visual detector unavailable");
  }

  try {
    const ocrStarted = performance.now();
    ocr = await recognizeScreenshotText(sourceCanvas, { onStage });
    timings.ocrMs = Math.round(performance.now() - ocrStarted);
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
  } catch (error) {
    ocrError = error;
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
    warnings,
    degradations,
    timings,
  };
  if (includeDiagnostics) {
    // Evaluation-only geometry: no recognized OCR text is retained here.
    result.diagnostics = {
      visual: visual.map(sanitizeRegion),
      ocrSemantic: acceptedOcr.map(sanitizeRegion),
      ocrUncertain: ocr.uncertainRegions.map(sanitizeRegion),
    };
  }
  return result;
}

function lowerFirst(text) {
  return text ? text[0].toLowerCase() + text.slice(1) : text;
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
