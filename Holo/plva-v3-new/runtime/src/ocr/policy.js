import { applyPolicy } from "@nationaldesignstudio/rampart";

const ADDRESS_LABELS = new Set([
  "BUILDING_NUMBER",
  "STREET_NAME",
  "SECONDARY_ADDRESS",
]);

const ADDRESS_CONTEXT = /\b(?:address|street|st\.?|road|rd\.?|avenue|ave\.?|lane|ln\.?|boulevard|blvd\.?|drive|dr\.?|way|court|ct\.?|terrace|suite|apt\.?|apartment|ship\s+to|deliver\s+to)\b/iu;
const SCREENSHOT_KEEP_LABELS = new Set();

// Rampart's chat policy intentionally keeps coarse geography. Screenshot
// redaction has a stricter contract: city, state, and postal code are PII and
// must remain eligible for masking.
export function applyScreenshotRedactionPolicy(spans) {
  return applyPolicy(spans, SCREENSHOT_KEEP_LABELS);
}

// Rampart deliberately redacts address components, but a lone number predicted
// in developer/version text is not enough evidence to mask an entire OCR line.
// Keep address spans when the line has address language or the model emits at
// least two distinct address components in the same OCR region.
export function filterContextualHits(hits, text) {
  const addressKinds = new Set(
    hits.filter((hit) => ADDRESS_LABELS.has(hit.label)).map((hit) => hit.label),
  );
  const hasAddressContext = ADDRESS_CONTEXT.test(text) || addressKinds.size >= 2;
  return hits.filter(
    (hit) => !ADDRESS_LABELS.has(hit.label) || hasAddressContext,
  );
}
