import { detectSensitiveValueCue } from "./rules.js";

const DEFAULT_HORIZONTAL_GAP = 480;
const DEFAULT_VERTICAL_GAP = 96;
const ASSOCIATABLE_CUES = new Set([
  "EMAIL",
  "PHONE",
  "NAME",
  "ADDRESS",
  "CARD_NUMBER",
  "CVC",
  "DOB",
  "GOVERNMENT_ID",
  "BANK_ACCOUNT",
]);
const NON_NAME_WORDS = new Set([
  "address",
  "account",
  "admin",
  "buyer",
  "company",
  "customer",
  "door",
  "everyone",
  "friend",
  "guest",
  "home",
  "locker",
  "member",
  "office",
  "pickup",
  "shopper",
  "store",
  "support",
  "team",
  "there",
  "this",
  "user",
  "visitor",
  "world",
  "you",
  "your",
]);
const PROPER_NAME_WORD = "[\\p{Lu}][\\p{L}'’.-]{1,30}";
const NAME_PARTICLE = "(?:al|bin|d[ae]|del|della|den|der|di|du|la|le|van|von)";
const NAME_TOKEN =
  `(${PROPER_NAME_WORD}(?:\\s+(?:${PROPER_NAME_WORD}|${NAME_PARTICLE})){0,7})`;
const GREETING_NAME = new RegExp(
  `^(?:Hello|Hi|Welcome(?:\\s+back)?)[,!]?\\s*${NAME_TOKEN}\\s*$`,
  "u",
);
const DELIVERY_NAME = new RegExp(
  `^(?:Deliver|Ship|Send)\\s+to\\s+${NAME_TOKEN}\\s*$`,
  "u",
);
const ORDER_NAME = new RegExp(
  `^${NAME_TOKEN}\\s*,?\\s+pay\\s+for\\s+this\\s+order\\b`,
  "u",
);

// OCR often emits a form label and its value as separate boxes. Associate a
// cue-only label with the closest plausible value to its right or immediately
// below. The returned regions intentionally omit recognized text.
export function associateCueValues(
  regions,
  {
    maximumHorizontalGap = DEFAULT_HORIZONTAL_GAP,
    maximumVerticalGap = DEFAULT_VERTICAL_GAP,
  } = {},
) {
  const entries = regions
    .map((region, index) => ({ region, index, cue: detectSensitiveValueCue(region.text) }))
    .filter(({ region }) => validRegion(region));
  const usedValues = new Set();
  const associated = [];

  for (const cueEntry of entries.filter(({ cue }) => ASSOCIATABLE_CUES.has(cue))) {
    const candidates = entries
      .filter(
        (entry) =>
          entry.index !== cueEntry.index &&
          !entry.cue &&
          !usedValues.has(entry.index) &&
          plausibleValue(cueEntry.cue, entry.region.text),
      )
      .map((entry) => ({
        entry,
        rank: associationRank(cueEntry.region, entry.region, {
          maximumHorizontalGap,
          maximumVerticalGap,
        }),
      }))
      .filter(({ rank }) => rank !== null)
      .sort((left, right) => left.rank - right.rank || left.entry.index - right.entry.index);
    const match = candidates[0];
    if (!match) continue;

    const value = match.entry.region;
    usedValues.add(match.entry.index);
    associated.push({
      x1: value.x1,
      y1: value.y1,
      x2: value.x2,
      y2: value.y2,
      detectorScore: value.detectorScore,
      ocrConfidence: value.ocrConfidence,
      label: cueEntry.cue,
      labels: [cueEntry.cue],
      sources: ["OCR+CUE_ASSOCIATION"],
      score: Math.min(0.96, Math.max(0.5, Number(value.ocrConfidence) || 0.5)),
    });
  }

  return associated;
}

// Commerce headers often expose an account name outside a form field. OCR can
// read these short lines even when the visual detector cannot resolve the tiny
// text in a full-page screenshot. Project only the matched name substring back
// into the OCR box; the delivery variant also covers the immediately attached
// city/postcode line rendered below it.
export function associateContextualIdentityRegions(regions) {
  const associated = [];
  for (const region of regions) {
    if (!validRegion(region)) continue;
    const text = region.text.trim();
    const greeting = GREETING_NAME.exec(text);
    const delivery = DELIVERY_NAME.exec(text);
    const order = ORDER_NAME.exec(text);
    const match = greeting ?? delivery ?? order;
    if (!match || isNonName(match[1])) continue;

    const matchedNameStart = match.index + match[0].indexOf(match[1]);
    associated.push(
      semanticRegion(
        greeting
          ? {
              x1: region.x1 - 2,
              y1: region.y1 - 2,
              x2: region.x2 + 2,
              y2: region.y2 + 2,
            }
          : projectTextRange(
              region,
              text,
              matchedNameStart,
              match[1].length,
            ),
        "NAME",
        region,
      ),
    );

    if (delivery) {
      const width = region.x2 - region.x1;
      const boxHeight = region.y2 - region.y1;
      associated.push(
        semanticRegion(
          {
            x1: region.x1,
            y1: region.y1,
            x2: region.x2 + width * 0.25,
            y2: region.y2 + boxHeight,
          },
          "ADDRESS",
          region,
        ),
      );
    }
  }
  return associated;
}

function associationRank(
  cue,
  value,
  { maximumHorizontalGap, maximumVerticalGap },
) {
  const cueHeight = height(cue);
  const valueHeight = height(value);
  const verticalOverlap = overlapLength(cue.y1, cue.y2, value.y1, value.y2);
  const sameRow = verticalOverlap >= Math.min(cueHeight, valueHeight) * 0.45;
  const horizontalGap = value.x1 - cue.x2;
  if (sameRow && horizontalGap >= -cueHeight * 0.35) {
    const scaledMaximum = Math.min(
      maximumHorizontalGap,
      Math.max(64, cueHeight * 12),
    );
    if (horizontalGap > scaledMaximum) return null;
    return Math.max(0, horizontalGap) + Math.abs(centerY(cue) - centerY(value)) * 2;
  }

  const verticalGap = value.y1 - cue.y2;
  if (verticalGap < -cueHeight * 0.35) return null;
  const scaledMaximum = Math.min(
    maximumVerticalGap,
    Math.max(32, cueHeight * 3),
  );
  if (verticalGap > scaledMaximum) return null;
  const horizontalOverlap = overlapLength(cue.x1, cue.x2, value.x1, value.x2);
  const leftAlignment = Math.abs(cue.x1 - value.x1);
  if (horizontalOverlap <= 0 && leftAlignment > Math.max(cueHeight * 5, 120)) return null;
  return 10_000 + Math.max(0, verticalGap) * 4 + leftAlignment;
}

function validRegion(region) {
  return (
    region &&
    typeof region.text === "string" &&
    region.text.trim() &&
    [region.x1, region.y1, region.x2, region.y2].every(Number.isFinite) &&
    region.x2 > region.x1 &&
    region.y2 > region.y1
  );
}

function height(region) {
  return Math.max(1, region.y2 - region.y1);
}

function centerY(region) {
  return (region.y1 + region.y2) / 2;
}

function overlapLength(leftStart, leftEnd, rightStart, rightEnd) {
  return Math.max(0, Math.min(leftEnd, rightEnd) - Math.max(leftStart, rightStart));
}

function plausibleValue(label, text) {
  const normalized = String(text ?? "").trim();
  if (!normalized || /^(?:continue|next|back|submit|save|cancel|edit|add|remove|upload|choose|select)$/iu.test(normalized)) {
    return false;
  }
  const digits = normalized.replace(/\D/gu, "");
  switch (label) {
    case "EMAIL":
      return /^[^\s@]+@[^\s@]+(?:\.[^\s@]+)?$/u.test(normalized);
    case "PHONE":
      return digits.length >= 7;
    case "NAME":
      return /^[\p{L}][\p{L}'’.-]*(?:\s+[\p{L}][\p{L}'’.-]*){0,5}$/u.test(normalized);
    case "ADDRESS":
      return (/\p{L}/u.test(normalized) && /\d/u.test(normalized)) || normalized.split(/\s+/u).length >= 2;
    case "CARD_NUMBER":
      return digits.length >= 12 && digits.length <= 19;
    case "CVC":
      return digits.length >= 3 && digits.length <= 4 && digits.length === normalized.replace(/\s/gu, "").length;
    case "DOB":
      return /(?:\d{1,4}[./-]){2}\d{1,4}/u.test(normalized) || /\b\d{4}\b/u.test(normalized);
    case "GOVERNMENT_ID":
      return digits.length >= 4;
    case "BANK_ACCOUNT":
      return digits.length >= 6 || /^[A-Z]{2}\d{2}[A-Z0-9\s-]{8,}$/u.test(normalized);
    default:
      return normalized.length >= 4;
  }
}

function projectTextRange(region, text, start, length) {
  const width = region.x2 - region.x1;
  const boxHeight = region.y2 - region.y1;
  const padding = Math.max(4, boxHeight * 0.4);
  const denominator = Math.max(1, text.length);
  return {
    x1: region.x1 + (width * start) / denominator - padding,
    y1: region.y1 - 2,
    x2: region.x1 + (width * (start + length)) / denominator + padding,
    y2: region.y2 + 2,
  };
}

function semanticRegion(geometry, label, source) {
  return {
    ...geometry,
    detectorScore: source.detectorScore,
    ocrConfidence: source.ocrConfidence,
    label,
    labels: [label],
    sources: ["OCR+CONTEXT_RULE"],
    score: Math.min(0.98, Math.max(0.5, Number(source.ocrConfidence) || 0.5)),
  };
}

function isNonName(value) {
  return String(value)
    .toLocaleLowerCase("en-US")
    .split(/\s+/u)
    .some((word) => NON_NAME_WORDS.has(word));
}
