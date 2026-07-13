import assert from "node:assert/strict";
import test from "node:test";

import {
  buildProfileReport,
  compatibleRegion,
  evaluateRegions,
  intersectionOverTruth,
} from "../src/evaluation/metrics.js";

test("intersection-over-truth rewards complete privacy coverage", () => {
  assert.equal(
    intersectionOverTruth({ x1: 5, y1: 5, x2: 25, y2: 25 }, [10, 10, 20, 20]),
    1,
  );
  assert.equal(
    intersectionOverTruth({ x1: 10, y1: 10, x2: 15, y2: 20 }, [10, 10, 20, 20]),
    0.5,
  );
});

test("evaluation reports secret misses and hard-negative false masks", () => {
  const metrics = evaluateRegions([
    {
      width: 100,
      height: 100,
      elapsedMs: 10,
      warnings: [],
      annotations: [
        { class: "CARD_NUMBER", source_class: "CARD_NUMBER", bbox_xyxy: [10, 10, 30, 20] },
        { class: "CVC", source_class: "CVC", bbox_xyxy: [40, 10, 50, 20] },
      ],
      regions: [{ x1: 8, y1: 8, x2: 32, y2: 22, sources: ["VISUAL"] }],
    },
    {
      width: 100,
      height: 100,
      elapsedMs: 20,
      warnings: [],
      annotations: [],
      regions: [{ x1: 1, y1: 1, x2: 5, y2: 5, sources: ["VISUAL"] }],
    },
  ]);
  assert.equal(metrics.recall, 0.5);
  assert.equal(metrics.secretMisses, 1);
  assert.equal(metrics.hardNegativeFalseMaskRecords, 1);
  assert.equal(metrics.pathAttribution.VISUAL, 2);
});

test("profile gates pass path integrity but reject a non-decision-grade corpus", () => {
  const report = buildProfileReport("balanced", [{
    id: "fixture",
    templateId: "panel.payment",
    width: 100,
    height: 100,
    elapsedMs: 10,
    warnings: [],
    annotations: [
      { class: "CARD_NUMBER", source_class: "CARD_NUMBER", bbox_xyxy: [10, 10, 30, 20] },
    ],
    visual: [{ x1: 8, y1: 8, x2: 32, y2: 22, sources: ["VISUAL"] }],
    ocrSemantic: [{ x1: 5, y1: 5, x2: 40, y2: 25, sources: ["OCR+RAMPART", "OCR+RULE"] }],
    fused: [{ x1: 5, y1: 5, x2: 40, y2: 25, sources: ["VISUAL", "OCR+RAMPART", "OCR+RULE"] }],
    outputIntegrity: { passed: true },
  }]);
  assert.equal(report.gates.passed, false);
  assert.equal(report.gates.checks.decisionGradeDataset.passed, false);
  assert.equal(report.gates.checks.secretMisses.passed, true);
  assert.equal(report.gates.checks.outputIntegrity.passed, true);
  assert.equal(report.gates.checks.allPathsExercised.passed, true);
});

test("class-compatible precision maps split semantic labels", () => {
  assert.equal(
    compatibleRegion(
      { labels: ["GIVEN_NAME", "SURNAME"] },
      { class: "NAME", source_class: "NAME" },
    ),
    true,
  );
  assert.equal(
    compatibleRegion(
      { labels: ["BUILDING_NUMBER"] },
      { class: "ADDRESS", source_class: "ADDRESS" },
    ),
    true,
  );
  assert.equal(
    compatibleRegion(
      { labels: ["CVC"] },
      { class: "EMAIL", source_class: "EMAIL" },
    ),
    false,
  );
});
