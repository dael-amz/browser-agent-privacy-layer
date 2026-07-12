const DETECTOR_THRESHOLD = 0.3;
const BOX_THRESHOLD = 0.5;
const MIN_BOX_SIDE = 3;
const MIN_UNCLIPPED_SIDE = 5;

export function computeOcrDetectorSize(
  sourceWidth,
  sourceHeight,
  { limitSide = 736, maxSide = 2048 } = {},
) {
  if (!Number.isFinite(sourceWidth) || !Number.isFinite(sourceHeight)) {
    throw new TypeError("image dimensions must be finite numbers");
  }
  if (sourceWidth <= 0 || sourceHeight <= 0) {
    throw new RangeError("image dimensions must be positive");
  }

  let scale = Math.min(sourceWidth, sourceHeight) < limitSide
    ? limitSide / Math.min(sourceWidth, sourceHeight)
    : 1;
  if (Math.max(sourceWidth, sourceHeight) * scale > maxSide) {
    scale = maxSide / Math.max(sourceWidth, sourceHeight);
  }

  const width = Math.max(32, roundToMultipleOf32(sourceWidth * scale));
  const height = Math.max(32, roundToMultipleOf32(sourceHeight * scale));
  return {
    width,
    height,
    scaleX: width / sourceWidth,
    scaleY: height / sourceHeight,
  };
}

export function extractOcrBoxes(
  probability,
  width,
  height,
  sourceWidth,
  sourceHeight,
  {
    threshold = DETECTOR_THRESHOLD,
    boxThreshold = BOX_THRESHOLD,
    maxCandidates = 1000,
    unclipRatio = 1.6,
  } = {},
) {
  if (probability.length !== width * height) {
    throw new Error("OCR probability map length does not match its dimensions");
  }

  const binary = new Uint8Array(probability.length);
  for (let index = 0; index < probability.length; index += 1) {
    binary[index] = probability[index] > threshold ? 1 : 0;
  }
  const dilated = dilate2x2(binary, width, height);
  const visited = new Uint8Array(dilated.length);
  const queue = new Int32Array(dilated.length);
  const candidates = [];

  for (let start = 0; start < dilated.length; start += 1) {
    if (!dilated[start] || visited[start]) continue;

    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    visited[start] = 1;
    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;

    while (head < tail) {
      const index = queue[head++];
      const y = Math.floor(index / width);
      const x = index - y * width;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);

      const y0 = Math.max(0, y - 1);
      const y1 = Math.min(height - 1, y + 1);
      const x0 = Math.max(0, x - 1);
      const x1 = Math.min(width - 1, x + 1);
      for (let nearY = y0; nearY <= y1; nearY += 1) {
        for (let nearX = x0; nearX <= x1; nearX += 1) {
          const near = nearY * width + nearX;
          if (dilated[near] && !visited[near]) {
            visited[near] = 1;
            queue[tail++] = near;
          }
        }
      }
    }

    const boxWidth = maxX - minX + 1;
    const boxHeight = maxY - minY + 1;
    if (Math.min(boxWidth, boxHeight) < MIN_BOX_SIDE) continue;

    let scoreSum = 0;
    for (let y = minY; y <= maxY; y += 1) {
      const row = y * width;
      for (let x = minX; x <= maxX; x += 1) {
        scoreSum += probability[row + x];
      }
    }
    const score = scoreSum / (boxWidth * boxHeight);
    if (score < boxThreshold) continue;

    // Axis-aligned approximation of DBNet's polygon offset. It intentionally
    // expands rather than shrinks so black boxes cover antialiased glyph edges.
    const distance =
      (boxWidth * boxHeight * unclipRatio) / (2 * (boxWidth + boxHeight));
    const expandedWidth = boxWidth + 2 * distance;
    const expandedHeight = boxHeight + 2 * distance;
    if (Math.min(expandedWidth, expandedHeight) < MIN_UNCLIPPED_SIDE) continue;

    candidates.push({
      x1: clamp((minX - distance) * (sourceWidth / width), 0, sourceWidth),
      y1: clamp((minY - distance) * (sourceHeight / height), 0, sourceHeight),
      x2: clamp((maxX + 1 + distance) * (sourceWidth / width), 0, sourceWidth),
      y2: clamp((maxY + 1 + distance) * (sourceHeight / height), 0, sourceHeight),
      detectorScore: score,
    });
    if (candidates.length >= maxCandidates) break;
  }

  return sortReadingOrder(candidates);
}

export function parseOcrDictionary(raw) {
  const lines = raw.split(/\r?\n/);
  if (lines.at(-1) === "") lines.pop();
  return ["<blank>", ...lines, " "];
}

export function decodeCtc(probabilities, dimensions, dictionary, sample = 0) {
  const [batch, steps, classes] = dimensions;
  if (sample < 0 || sample >= batch || dictionary.length !== classes) {
    throw new Error("unexpected OCR recognition output shape");
  }
  const offset = sample * steps * classes;
  let previous = -1;
  let text = "";
  let confidenceSum = 0;
  let emitted = 0;

  for (let step = 0; step < steps; step += 1) {
    let bestClass = 0;
    let bestScore = Number(probabilities[offset + step * classes]);
    for (let classId = 1; classId < classes; classId += 1) {
      const score = Number(probabilities[offset + step * classes + classId]);
      if (score > bestScore) {
        bestClass = classId;
        bestScore = score;
      }
    }
    if (bestClass !== 0 && bestClass !== previous) {
      text += dictionary[bestClass];
      confidenceSum += bestScore;
      emitted += 1;
    }
    previous = bestClass;
  }

  return {
    text,
    confidence: emitted === 0 ? 0 : confidenceSum / emitted,
  };
}

function dilate2x2(binary, width, height) {
  const output = new Uint8Array(binary.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (!binary[index]) continue;
      output[index] = 1;
      if (x + 1 < width) output[index + 1] = 1;
      if (y + 1 < height) output[index + width] = 1;
      if (x + 1 < width && y + 1 < height) output[index + width + 1] = 1;
    }
  }
  return output;
}

function sortReadingOrder(boxes) {
  return [...boxes].sort((left, right) => {
    if (Math.abs(left.y1 - right.y1) <= 10) return left.x1 - right.x1;
    return left.y1 - right.y1;
  });
}

function roundToMultipleOf32(value) {
  return roundHalfToEven(value / 32) * 32;
}

function roundHalfToEven(value) {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (Math.abs(fraction - 0.5) < Number.EPSILON * Math.max(1, Math.abs(value))) {
    return floor % 2 === 0 ? floor : floor + 1;
  }
  return Math.round(value);
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}
