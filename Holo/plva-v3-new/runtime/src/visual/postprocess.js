export const DETECTOR_CLASSES = Object.freeze([
  "NAME",
  "EMAIL",
  "PHONE",
  "ADDRESS",
  "CARD_NUMBER",
  "CVC",
  "SECRET",
  "SENSITIVE_FIELD",
  "SENSITIVE_IMAGE",
]);

// These are tuning-only operating points for the development model. The CVC
// score is intentionally much lower because the current checkpoint ranks CVC
// correctly but is poorly calibrated for that class.
export const THRESHOLD_PROFILES = Object.freeze({
  "high-recall": Object.freeze({
    NAME: 0.099099,
    EMAIL: 0.099099,
    PHONE: 0.099099,
    ADDRESS: 0.099099,
    CARD_NUMBER: 0.079079,
    CVC: 0.009009,
    SECRET: 0.079079,
    SENSITIVE_FIELD: 0.079079,
    SENSITIVE_IMAGE: 0.349349,
  }),
  balanced: Object.freeze({
    NAME: 0.25,
    EMAIL: 0.25,
    PHONE: 0.25,
    ADDRESS: 0.25,
    CARD_NUMBER: 0.2,
    CVC: 0.03,
    SECRET: 0.2,
    SENSITIVE_FIELD: 0.2,
    SENSITIVE_IMAGE: 0.5,
  }),
});

export function computeLetterboxTransform(sourceWidth, sourceHeight, size = 640) {
  if (!Number.isFinite(sourceWidth) || !Number.isFinite(sourceHeight)) {
    throw new TypeError("image dimensions must be finite numbers");
  }
  if (sourceWidth <= 0 || sourceHeight <= 0 || size <= 0) {
    throw new RangeError("image and model dimensions must be positive");
  }

  const scale = Math.min(size / sourceWidth, size / sourceHeight);
  const scaledWidth = Math.max(1, roundHalfToEven(sourceWidth * scale));
  const scaledHeight = Math.max(1, roundHalfToEven(sourceHeight * scale));
  const padLeft = Math.floor((size - scaledWidth) / 2);
  const padTop = Math.floor((size - scaledHeight) / 2);

  return {
    sourceWidth,
    sourceHeight,
    size,
    scale,
    scaledWidth,
    scaledHeight,
    padLeft,
    padTop,
  };
}

export function mapBoxToSource(box, transform, paddingFraction = 0.04) {
  const { sourceWidth, sourceHeight, scale, padLeft, padTop } = transform;
  const rawLeft = (box.centerX - box.width / 2 - padLeft) / scale;
  const rawTop = (box.centerY - box.height / 2 - padTop) / scale;
  const rawRight = (box.centerX + box.width / 2 - padLeft) / scale;
  const rawBottom = (box.centerY + box.height / 2 - padTop) / scale;
  const width = Math.max(0, rawRight - rawLeft);
  const height = Math.max(0, rawBottom - rawTop);
  const padding = Math.max(2, Math.min(width, height) * paddingFraction);

  return {
    ...box,
    x1: clamp(rawLeft - padding, 0, sourceWidth),
    y1: clamp(rawTop - padding, 0, sourceHeight),
    x2: clamp(rawRight + padding, 0, sourceWidth),
    y2: clamp(rawBottom + padding, 0, sourceHeight),
  };
}

export function decodeDetections(
  outputData,
  outputDimensions,
  transform,
  {
    thresholds = THRESHOLD_PROFILES["high-recall"],
    iouThreshold = 0.7,
    maxDetections = 300,
    paddingFraction = 0.04,
  } = {},
) {
  if (!outputData || !Array.isArray(outputDimensions)) {
    throw new TypeError("detector output data and dimensions are required");
  }
  const [batch, channels, anchors] = outputDimensions;
  if (batch !== 1 || channels !== 4 + DETECTOR_CLASSES.length || anchors <= 0) {
    throw new Error(
      `unexpected detector output shape: ${outputDimensions.join("x")}`,
    );
  }
  if (outputData.length !== channels * anchors) {
    throw new Error("detector output length does not match its tensor shape");
  }

  const proposals = [];
  for (let anchor = 0; anchor < anchors; anchor += 1) {
    let bestClass = 0;
    let bestScore = Number(outputData[4 * anchors + anchor]);

    for (let classId = 1; classId < DETECTOR_CLASSES.length; classId += 1) {
      const score = Number(outputData[(4 + classId) * anchors + anchor]);
      if (score > bestScore) {
        bestClass = classId;
        bestScore = score;
      }
    }

    const bestLabel = DETECTOR_CLASSES[bestClass];
    if (bestScore < (thresholds[bestLabel] ?? 1)) continue;
    const width = Number(outputData[2 * anchors + anchor]);
    const height = Number(outputData[3 * anchors + anchor]);
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
      continue;
    }

    const proposal = mapBoxToSource(
      {
        centerX: Number(outputData[anchor]),
        centerY: Number(outputData[anchors + anchor]),
        width,
        height,
        classId: bestClass,
        label: bestLabel,
        score: bestScore,
      },
      transform,
      paddingFraction,
    );
    if (proposal.x2 - proposal.x1 >= 2 && proposal.y2 - proposal.y1 >= 2) {
      proposals.push(proposal);
    }
  }

  return nonMaximumSuppression(proposals, { iouThreshold, maxDetections });
}

export function nonMaximumSuppression(
  proposals,
  { iouThreshold = 0.7, maxDetections = 300 } = {},
) {
  const ordered = [...proposals].sort((left, right) => right.score - left.score);
  const selected = [];
  const selectedByClass = new Map();

  for (const proposal of ordered) {
    const sameClass = selectedByClass.get(proposal.classId) ?? [];
    if (
      sameClass.some(
        (current) => intersectionOverUnion(current, proposal) > iouThreshold,
      )
    ) {
      continue;
    }
    selected.push(proposal);
    sameClass.push(proposal);
    selectedByClass.set(proposal.classId, sameClass);
    if (selected.length >= maxDetections) break;
  }

  return selected;
}

export function intersectionOverUnion(left, right) {
  const x1 = Math.max(left.x1, right.x1);
  const y1 = Math.max(left.y1, right.y1);
  const x2 = Math.min(left.x2, right.x2);
  const y2 = Math.min(left.y2, right.y2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const leftArea = Math.max(0, left.x2 - left.x1) * Math.max(0, left.y2 - left.y1);
  const rightArea = Math.max(0, right.x2 - right.x1) * Math.max(0, right.y2 - right.y1);
  const union = leftArea + rightArea - intersection;
  return union > 0 ? intersection / union : 0;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function roundHalfToEven(value) {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (Math.abs(fraction - 0.5) < Number.EPSILON * Math.max(1, Math.abs(value))) {
    return floor % 2 === 0 ? floor : floor + 1;
  }
  return Math.round(value);
}
