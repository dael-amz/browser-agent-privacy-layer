import assert from "node:assert/strict";
import test from "node:test";

import {
  computeOcrDetectorSize,
  decodeCtc,
  extractOcrBoxes,
  parseOcrDictionary,
} from "../src/ocr/postprocess.js";
import { detectSensitiveCues } from "../src/ocr/rules.js";
import { filterContextualHits } from "../src/ocr/policy.js";
import { fuseSensitiveRegions } from "../src/visual/fusion.js";

test("caps an ultra-wide OCR detector tensor and rounds to multiples of 32", () => {
  assert.deepEqual(computeOcrDetectorSize(1600, 280), {
    width: 2048,
    height: 352,
    scaleX: 1.28,
    scaleY: 352 / 280,
  });
});

test("keeps the dictionary's literal space and appends RapidOCR's second space", () => {
  assert.deepEqual(parseOcrDictionary("a\n \n"), ["<blank>", "a", " ", " "]);
});

test("CTC collapses duplicate classes but preserves repeats separated by blank", () => {
  const dictionary = ["<blank>", "a", "b", " "];
  const sequence = [1, 1, 0, 1, 2, 0, 3];
  const values = new Float32Array(sequence.length * dictionary.length);
  for (let step = 0; step < sequence.length; step += 1) {
    values[step * dictionary.length + sequence[step]] = 0.9;
  }
  const decoded = decodeCtc(
    values,
    [1, sequence.length, dictionary.length],
    dictionary,
  );
  assert.equal(decoded.text, "aab ");
  assert.ok(Math.abs(decoded.confidence - 0.9) < 1e-6);
});

test("extracts and conservatively expands a connected OCR text component", () => {
  const width = 12;
  const height = 8;
  const probability = new Float32Array(width * height);
  for (let y = 2; y <= 4; y += 1) {
    for (let x = 2; x <= 7; x += 1) probability[y * width + x] = 0.9;
  }
  const boxes = extractOcrBoxes(probability, width, height, width, height);
  assert.equal(boxes.length, 1);
  assert.ok(boxes[0].x1 < 2);
  assert.ok(boxes[0].y1 < 2);
  assert.ok(boxes[0].x2 > 8);
  assert.ok(boxes[0].y2 > 5);
});

test("secret cue rules catch OCR-tolerant CVC and credential labels", () => {
  assert.deepEqual(
    detectSensitiveCues("Card number 4000 0000 0000 0002 Security c0de 007"),
    ["CVC", "CARD_NUMBER"],
  );
  assert.deepEqual(detectSensitiveCues("API key sk_test_123"), ["API_KEY"]);
  assert.deepEqual(detectSensitiveCues("Order 4242-2026 total $412.30"), []);
  assert.deepEqual(
    detectSensitiveCues("Recovery phone: +1 (202) 555-0812 ext 700012"),
    ["PHONE"],
  );
  assert.deepEqual(
    detectSensitiveCues("eyJhbGciOiJub25lIn0.eyJzdWIiOiJmYWtlIn0.signature"),
    ["AUTH_TOKEN"],
  );
});

test("drops an isolated address-component guess from developer status text", () => {
  const hit = { label: "BUILDING_NUMBER", start: 52, end: 57 };
  assert.deepEqual(
    filterContextualHits(
      [hit],
      "localhost:5173 commit a1b2c3d4 version 2026.07 PID 55301",
    ),
    [],
  );
  assert.deepEqual(
    filterContextualHits(
      [hit, { label: "STREET_NAME", start: 4, end: 15 }],
      "275 Sara Summit",
    ).length,
    2,
  );
});

test("fuses a visual box contained by an OCR-semantic line and preserves evidence", () => {
  const fused = fuseSensitiveRegions(
    [{ x1: 100, y1: 20, x2: 220, y2: 50, label: "CARD_NUMBER", score: 0.8 }],
    [{
      x1: 40,
      y1: 15,
      x2: 300,
      y2: 55,
      label: "CVC + CREDIT_CARD",
      labels: ["CVC", "CREDIT_CARD"],
      sources: ["OCR+RAMPART", "OCR+RULE"],
      score: 0.9,
    }],
  );
  assert.equal(fused.length, 1);
  assert.deepEqual(
    new Set(fused[0].sources),
    new Set(["OCR+RAMPART", "OCR+RULE", "VISUAL"]),
  );
  assert.deepEqual(
    new Set(fused[0].labels),
    new Set(["CVC", "CREDIT_CARD", "CARD_NUMBER"]),
  );
  assert.deepEqual(
    [fused[0].x1, fused[0].y1, fused[0].x2, fused[0].y2],
    [40, 15, 300, 55],
  );
});
