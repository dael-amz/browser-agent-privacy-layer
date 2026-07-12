export function fuseSensitiveRegions(
  visualRegions,
  semanticRegions,
  { iouThreshold = 0.35, containedThreshold = 0.55 } = {},
) {
  const regions = [
    ...visualRegions.map((region) => ({
      ...region,
      labels: region.labels ?? [region.label],
      sources: region.sources ?? ["VISUAL"],
    })),
    ...semanticRegions.map((region) => ({
      ...region,
      labels: region.labels ?? [region.label],
      sources: region.sources ?? ["OCR+RAMPART"],
    })),
  ];

  const fused = [];
  for (const candidate of regions.sort((left, right) => right.score - left.score)) {
    const overlapping = [];
    for (let index = 0; index < fused.length; index += 1) {
      if (shouldMerge(candidate, fused[index], iouThreshold, containedThreshold)) {
        overlapping.push(index);
      }
    }
    if (overlapping.length === 0) {
      fused.push(candidate);
      continue;
    }

    let merged = candidate;
    for (const index of overlapping.reverse()) {
      merged = unionRegions(merged, fused[index]);
      fused.splice(index, 1);
    }
    // A union can now overlap a region that neither original crossed enough.
    // Re-run until the merged region reaches a fixed point.
    let changed = true;
    while (changed) {
      changed = false;
      for (let index = fused.length - 1; index >= 0; index -= 1) {
        if (shouldMerge(merged, fused[index], iouThreshold, containedThreshold)) {
          merged = unionRegions(merged, fused[index]);
          fused.splice(index, 1);
          changed = true;
        }
      }
    }
    fused.push(merged);
  }

  return fused.sort((left, right) => left.y1 - right.y1 || left.x1 - right.x1);
}

function shouldMerge(left, right, iouThreshold, containedThreshold) {
  const intersection = intersectionArea(left, right);
  if (intersection <= 0) return false;
  const leftArea = area(left);
  const rightArea = area(right);
  const union = leftArea + rightArea - intersection;
  const iou = union > 0 ? intersection / union : 0;
  const contained = intersection / Math.max(1, Math.min(leftArea, rightArea));
  return iou >= iouThreshold || contained >= containedThreshold;
}

function unionRegions(left, right) {
  const labels = [...new Set([...left.labels, ...right.labels])];
  const sources = [...new Set([...left.sources, ...right.sources])];
  return {
    x1: Math.min(left.x1, right.x1),
    y1: Math.min(left.y1, right.y1),
    x2: Math.max(left.x2, right.x2),
    y2: Math.max(left.y2, right.y2),
    label: labels.join(" + "),
    labels,
    sources,
    score: Math.max(left.score, right.score),
  };
}

function intersectionArea(left, right) {
  return (
    Math.max(0, Math.min(left.x2, right.x2) - Math.max(left.x1, right.x1)) *
    Math.max(0, Math.min(left.y2, right.y2) - Math.max(left.y1, right.y1))
  );
}

function area(region) {
  return Math.max(0, region.x2 - region.x1) * Math.max(0, region.y2 - region.y1);
}
