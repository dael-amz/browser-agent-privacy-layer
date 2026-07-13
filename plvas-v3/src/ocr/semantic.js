import {
  applyPolicy,
  detectHeuristics,
  detectNer,
  premask,
  projectMaskedSpan,
} from "@nationaldesignstudio/rampart";

import { getRampartClassifier, RAMPART_MIN_SCORE } from "../rampart.js";
import { filterContextualHits } from "./policy.js";
import { detectSensitiveCues } from "./rules.js";

export async function classifyOcrRegions(
  regions,
  { device = "wasm", onStage = () => {} } = {},
) {
  const document = buildOcrDocument(regions);
  if (!document.text) {
    return { regions: [], mode: "not-needed", warning: null };
  }

  const heuristic = detectHeuristics(document.text);
  let allSpans = heuristic;
  let mode = "rampart";
  let warning = null;

  try {
    onStage("OCR: loading the local Rampart classifier…");
    const classifier = await getRampartClassifier(device);
    onStage("OCR: classifying recognized text locally…");
    const map = premask(document.text, heuristic);
    const maskedSpans = await detectNer(
      map.masked,
      classifier,
      RAMPART_MIN_SCORE,
    );
    const contextual = maskedSpans
      .map((span) => projectMaskedSpan(span, document.text, map))
      .filter(Boolean);
    allSpans = [...heuristic, ...contextual];
  } catch {
    mode = "heuristics-only";
    warning = "Rampart NER unavailable; OCR used structured and secret-cue checks only";
  }

  const redactable = applyPolicy(allSpans);
  const sensitive = [];
  for (const entry of document.entries) {
    const rampartHits = filterContextualHits(
      redactable.filter(
        (span) => span.start < entry.end && span.end > entry.start,
      ),
      entry.text,
    );
    const cueLabels = detectSensitiveCues(entry.text);
    if (rampartHits.length === 0 && cueLabels.length === 0) continue;

    const labels = new Set(cueLabels);
    for (const hit of rampartHits) labels.add(hit.label);
    const semanticScore = Math.max(
      cueLabels.length > 0 ? 0.98 : 0,
      ...rampartHits.map((hit) => hit.score),
    );
    sensitive.push({
      x1: entry.region.x1,
      y1: entry.region.y1,
      x2: entry.region.x2,
      y2: entry.region.y2,
      detectorScore: entry.region.detectorScore,
      ocrConfidence: entry.region.ocrConfidence,
      label: [...labels].join(" + "),
      labels: [...labels],
      sources: [
        ...(rampartHits.length > 0 ? ["OCR+RAMPART"] : []),
        ...(cueLabels.length > 0 ? ["OCR+RULE"] : []),
      ],
      score: Math.min(entry.region.ocrConfidence, semanticScore),
    });
  }

  return { regions: sensitive, mode, warning };
}

function buildOcrDocument(regions) {
  const entries = [];
  let text = "";
  for (const region of regions) {
    const recognized = region.text.trim();
    if (!recognized) continue;
    if (text) text += "\n";
    const start = text.length;
    text += recognized;
    entries.push({
      region,
      text: recognized,
      start,
      end: text.length,
    });
  }
  return { text, entries };
}
