import assert from "node:assert/strict";
import test from "node:test";

import { integerMask } from "../src/visual/render.js";

test("redaction masks floor starts, ceil ends, and clamp to the canvas", () => {
  assert.deepEqual(
    integerMask({ x1: -2.2, y1: 3.8, x2: 10.1, y2: 12.01 }, 10, 20),
    { x: 0, y: 3, width: 10, height: 10 },
  );
});
