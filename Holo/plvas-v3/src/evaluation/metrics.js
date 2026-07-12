export const COVERAGE_THRESHOLD = 0.98;
export const VISUAL_PROPOSAL_THRESHOLD = 0.9;
export const SECRET_CLASSES = new Set(["CARD_NUMBER", "CVC", "SECRET"]);

export function evaluateRegions(
  records,
  { coverageThreshold = COVERAGE_THRESHOLD } = {},
) {
  const totals = createTotals();
  const perClass = new Map();
  const perSourceClass = new Map();
  const latencies = [];

  for (const record of records) {
    const truth = record.annotations ?? [];
    const regions = record.regions ?? [];
    const matchedRegions = new Set();
    latencies.push(record.elapsedMs);
    totals.records += 1;
    totals.truth += truth.length;
    totals.predictions += regions.length;
    totals.maskedAreaFraction += maskedAreaFraction(
      regions,
      record.width,
      record.height,
    );
    if (record.degradations?.length) totals.degradedRuns += 1;
    if (truth.length === 0) {
      totals.hardNegativeRecords += 1;
      if (regions.length > 0) {
        totals.hardNegativeFalseMaskRecords += 1;
        totals.hardNegativeFalseMasks += regions.length;
      }
    }

    for (const annotation of truth) {
      const truthClass = annotation.class;
      const sourceClass = annotation.source_class ?? truthClass;
      const classCounts = getCounts(perClass, truthClass);
      const sourceCounts = getCounts(perSourceClass, sourceClass);
      classCounts.support += 1;
      sourceCounts.support += 1;
      if (SECRET_CLASSES.has(truthClass)) totals.secretSupport += 1;

      let bestCoverage = 0;
      let bestIndex = -1;
      for (let index = 0; index < regions.length; index += 1) {
        const coverage = intersectionOverTruth(
          regions[index],
          annotation.bbox_xyxy,
        );
        if (coverage > bestCoverage) {
          bestCoverage = coverage;
          bestIndex = index;
        }
      }
      if (bestCoverage >= coverageThreshold) {
        totals.covered += 1;
        classCounts.covered += 1;
        sourceCounts.covered += 1;
        if (SECRET_CLASSES.has(truthClass)) totals.secretCovered += 1;
      }
      for (let index = 0; index < regions.length; index += 1) {
        if (
          intersectionOverTruth(regions[index], annotation.bbox_xyxy) >=
            coverageThreshold &&
          compatibleRegion(regions[index], annotation)
        ) {
          matchedRegions.add(index);
        }
      }
    }
    totals.matchedPredictions += matchedRegions.size;
    totals.falsePredictions += regions.length - matchedRegions.size;

    for (const region of regions) {
      if ((region.sources ?? []).includes("OCR+UNCERTAIN")) {
        totals.uncertaintyMasks += 1;
      }
      for (const source of region.sources ?? []) {
        totals.pathAttribution[source] =
          (totals.pathAttribution[source] ?? 0) + 1;
      }
    }
  }

  const sortedLatencies = latencies.filter(Number.isFinite).sort((a, b) => a - b);
  return {
    ...totals,
    recall: ratio(totals.covered, totals.truth),
    proposalPrecision: ratio(totals.matchedPredictions, totals.predictions),
    secretMisses: totals.secretSupport - totals.secretCovered,
    hardNegativePassRate: ratio(
      totals.hardNegativeRecords - totals.hardNegativeFalseMaskRecords,
      totals.hardNegativeRecords,
    ),
    meanMaskedAreaFraction: ratio(totals.maskedAreaFraction, totals.records),
    latencyMs: {
      median: percentile(sortedLatencies, 0.5),
      p95: percentile(sortedLatencies, 0.95),
      maximum: sortedLatencies.at(-1) ?? 0,
    },
    perClass: finalizeCounts(perClass),
    perSourceClass: finalizeCounts(perSourceClass),
  };
}

export function buildProfileReport(profile, fixtureRuns) {
  const fusedRecords = fixtureRuns.map((run) => ({ ...run, regions: run.fused }));
  const visualRecords = fixtureRuns.map((run) => ({ ...run, regions: run.visual }));
  const ocrRecords = fixtureRuns.map((run) => ({ ...run, regions: run.ocrSemantic }));
  const fused = evaluateRegions(fusedRecords, {
    coverageThreshold: COVERAGE_THRESHOLD,
  });
  const visual = evaluateRegions(visualRecords, {
    coverageThreshold: COVERAGE_THRESHOLD,
  });
  const visualProposal = evaluateRegions(visualRecords, {
    coverageThreshold: VISUAL_PROPOSAL_THRESHOLD,
  });
  const ocrSemantic = evaluateRegions(ocrRecords, {
    coverageThreshold: COVERAGE_THRESHOLD,
  });
  const saves = countPathSaves(fixtureRuns);
  const outputIntegrityPassed = fixtureRuns.every(
    (run) => run.outputIntegrity?.passed === true,
  );
  const checks = {
    decisionGradeDataset: datasetAdequacyCheck(fixtureRuns),
    fusedRecall: { actual: fused.recall, required: 0.97, passed: fused.recall >= 0.97 },
    visualProposalRecall: { actual: visualProposal.recall, required: 0.95, passed: visualProposal.recall >= 0.95 },
    compatiblePrecision: { actual: fused.proposalPrecision, required: 0.95, passed: fused.proposalPrecision >= 0.95 },
    secretMisses: { actual: fused.secretMisses, required: 0, passed: fused.secretSupport > 0 && fused.secretMisses === 0 },
    hardNegativeFalseMaskRecords: { actual: fused.hardNegativeFalseMaskRecords, required: 0, passed: fused.hardNegativeFalseMaskRecords === 0 },
    degradedRuns: { actual: fused.degradedRuns, required: 0, passed: fused.degradedRuns === 0 },
    outputIntegrity: { actual: outputIntegrityPassed, required: true, passed: outputIntegrityPassed },
    fusionRegressions: { actual: saves.fusionRegressions, required: 0, passed: saves.fusionRegressions === 0 },
    allPathsExercised: {
      actual: Object.keys(fused.pathAttribution).sort(),
      required: ["OCR+RAMPART", "OCR+RULE", "VISUAL"],
      passed: ["OCR+RAMPART", "OCR+RULE", "VISUAL"].every(
        (path) => (fused.pathAttribution[path] ?? 0) > 0,
      ),
    },
  };
  return {
    schemaVersion: 1,
    profile,
    coverageThreshold: COVERAGE_THRESHOLD,
    visualProposalCoverageThreshold: VISUAL_PROPOSAL_THRESHOLD,
    fixtures: fixtureRuns,
    metrics: { fused, visual, visualProposal, ocrSemantic, saves },
    gates: {
      passed: Object.values(checks).every((check) => check.passed),
      checks,
    },
  };
}

export function intersectionOverTruth(region, truthBox) {
  const x1 = Math.max(region.x1, truthBox[0]);
  const y1 = Math.max(region.y1, truthBox[1]);
  const x2 = Math.min(region.x2, truthBox[2]);
  const y2 = Math.min(region.y2, truthBox[3]);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const truthArea =
    Math.max(0, truthBox[2] - truthBox[0]) *
    Math.max(0, truthBox[3] - truthBox[1]);
  return truthArea > 0 ? intersection / truthArea : 0;
}

export function compatibleRegion(region, annotation) {
  const labels = new Set(region.labels ?? [region.label]);
  const truth = annotation.class;
  const source = annotation.source_class ?? truth;
  if (labels.has(truth) || labels.has(source)) return true;
  if (truth === "NAME") {
    return labels.has("GIVEN_NAME") || labels.has("SURNAME");
  }
  if (truth === "ADDRESS") {
    return ["BUILDING_NUMBER", "STREET_NAME", "SECONDARY_ADDRESS"].some(
      (label) => labels.has(label),
    );
  }
  if (truth === "CARD_NUMBER") return labels.has("CREDIT_CARD");
  if (truth === "CVC") return labels.has("CVC");
  if (truth === "SENSITIVE_FIELD") {
    const map = {
      DOB: ["DOB"],
      GOV_ID: ["GOVERNMENT_ID", "SSN", "TAX_ID", "PASSPORT", "DRIVERS_LICENSE"],
      BANK_ACCOUNT: ["BANK_ACCOUNT", "ROUTING_NUMBER"],
    };
    return (map[source] ?? []).some((label) => labels.has(label));
  }
  if (truth === "SECRET") {
    const map = {
      PASSWORD: ["PASSWORD"],
      API_KEY: ["API_KEY"],
      AUTH_TOKEN: ["AUTH_TOKEN"],
      PRIVATE_KEY: ["PRIVATE_KEY"],
    };
    return labels.has("SECRET") || (map[source] ?? []).some((label) => labels.has(label));
  }
  return false;
}

function countPathSaves(fixtureRuns) {
  const totals = {
    visualOnlySaves: 0,
    ocrOnlySaves: 0,
    fusionRegressions: 0,
    secretFusionRegressions: 0,
  };
  for (const run of fixtureRuns) {
    for (const annotation of run.annotations ?? []) {
      const visual = pathCovers(
        run.visual,
        annotation,
        COVERAGE_THRESHOLD,
      );
      const ocr = pathCovers(
        run.ocrSemantic,
        annotation,
        COVERAGE_THRESHOLD,
      );
      const fused = pathCovers(run.fused, annotation, COVERAGE_THRESHOLD);
      if (fused && visual && !ocr) totals.visualOnlySaves += 1;
      if (fused && ocr && !visual) totals.ocrOnlySaves += 1;
      if ((visual || ocr) && !fused) {
        totals.fusionRegressions += 1;
        if (SECRET_CLASSES.has(annotation.class)) {
          totals.secretFusionRegressions += 1;
        }
      }
    }
  }
  return totals;
}

function pathCovers(regions, annotation, threshold) {
  return (regions ?? []).some(
    (region) =>
      intersectionOverTruth(region, annotation.bbox_xyxy) >= threshold,
  );
}

function datasetAdequacyCheck(fixtureRuns) {
  const records = fixtureRuns.length;
  const truth = fixtureRuns.reduce(
    (sum, run) => sum + (run.annotations?.length ?? 0),
    0,
  );
  const secrets = fixtureRuns.reduce(
    (sum, run) =>
      sum +
      (run.annotations ?? []).filter((item) => SECRET_CLASSES.has(item.class)).length,
    0,
  );
  const hardNegatives = fixtureRuns.filter(
    (run) => (run.annotations?.length ?? 0) === 0,
  ).length;
  const families = new Set(
    fixtureRuns.map((run) => run.pageType ?? run.templateId).filter(Boolean),
  ).size;
  const actual = { records, truth, secrets, hardNegatives, families };
  const required = {
    records: 200,
    truth: 1000,
    secrets: 300,
    hardNegatives: 100,
    families: 12,
  };
  return {
    actual,
    required,
    passed: Object.keys(required).every(
      (key) => actual[key] >= required[key],
    ),
  };
}

function createTotals() {
  return {
    records: 0,
    truth: 0,
    covered: 0,
    predictions: 0,
    matchedPredictions: 0,
    falsePredictions: 0,
    secretSupport: 0,
    secretCovered: 0,
    hardNegativeRecords: 0,
    hardNegativeFalseMaskRecords: 0,
    hardNegativeFalseMasks: 0,
    degradedRuns: 0,
    maskedAreaFraction: 0,
    uncertaintyMasks: 0,
    pathAttribution: {},
  };
}

function getCounts(map, label) {
  if (!map.has(label)) map.set(label, { support: 0, covered: 0 });
  return map.get(label);
}

function finalizeCounts(map) {
  return Object.fromEntries(
    [...map.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([label, counts]) => [
        label,
        { ...counts, recall: ratio(counts.covered, counts.support) },
      ]),
  );
}

function maskedAreaFraction(regions, width, height) {
  const imageArea = width * height;
  if (!imageArea) return 0;
  const area = regions.reduce(
    (sum, region) =>
      sum +
      Math.max(0, region.x2 - region.x1) *
        Math.max(0, region.y2 - region.y1),
    0,
  );
  return Math.min(1, area / imageArea);
}

function percentile(sorted, value) {
  if (sorted.length === 0) return 0;
  const index = Math.min(sorted.length - 1, Math.ceil(sorted.length * value) - 1);
  return sorted[Math.max(0, index)];
}

function ratio(numerator, denominator) {
  return denominator > 0 ? numerator / denominator : 0;
}
