import assert from "node:assert/strict";
import test from "node:test";

import {
  DETECTOR_CLASSES,
  THRESHOLD_PROFILES,
  computeLetterboxTransform,
  decodeDetections,
  nonMaximumSuppression,
} from "../src/visual/postprocess.js";

test("letterboxes a wide screenshot without stretching it", () => {
  assert.deepEqual(computeLetterboxTransform(1600, 280), {
    sourceWidth: 1600,
    sourceHeight: 280,
    size: 640,
    scale: 0.4,
    scaledWidth: 640,
    scaledHeight: 112,
    padLeft: 0,
    padTop: 264,
  });
});

test("decodes channel-major YOLO boxes into source pixels", () => {
  const anchors = 1;
  const channels = 4 + DETECTOR_CLASSES.length;
  const output = new Float32Array(channels * anchors);
  output[0] = 320;
  output[1] = 320;
  output[2] = 100;
  output[3] = 50;
  output[4 + DETECTOR_CLASSES.indexOf("EMAIL")] = 0.8;

  const [detection] = decodeDetections(
    output,
    [1, channels, anchors],
    computeLetterboxTransform(640, 640),
    { thresholds: THRESHOLD_PROFILES.balanced, paddingFraction: 0 },
  );

  assert.equal(detection.label, "EMAIL");
  assert.ok(Math.abs(detection.score - 0.8) < 1e-6);
  assert.deepEqual(
    [detection.x1, detection.y1, detection.x2, detection.y2],
    [268, 293, 372, 347],
  );
});

test("uses the recall-first CVC threshold without applying sigmoid", () => {
  const anchors = 1;
  const channels = 4 + DETECTOR_CLASSES.length;
  const output = new Float32Array(channels * anchors);
  output[0] = 200;
  output[1] = 200;
  output[2] = 40;
  output[3] = 20;
  output[4 + DETECTOR_CLASSES.indexOf("CVC")] = 0.02;

  const highRecall = decodeDetections(
    output,
    [1, channels, anchors],
    computeLetterboxTransform(640, 640),
    { thresholds: THRESHOLD_PROFILES["high-recall"] },
  );
  const balanced = decodeDetections(
    output,
    [1, channels, anchors],
    computeLetterboxTransform(640, 640),
    { thresholds: THRESHOLD_PROFILES.balanced },
  );

  assert.equal(highRecall.length, 1);
  assert.equal(highRecall[0].label, "CVC");
  assert.equal(balanced.length, 0);
});

test("NMS suppresses same-class duplicates but preserves another class", () => {
  const box = { x1: 10, y1: 10, x2: 110, y2: 60 };
  const selected = nonMaximumSuppression(
    [
      { ...box, classId: 1, score: 0.9 },
      { ...box, classId: 1, score: 0.8 },
      { ...box, classId: 4, score: 0.7 },
    ],
    { iouThreshold: 0.7 },
  );

  assert.deepEqual(
    selected.map(({ classId, score }) => [classId, score]),
    [
      [1, 0.9],
      [4, 0.7],
    ],
  );
});
