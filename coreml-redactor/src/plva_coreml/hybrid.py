"""Full accelerated visual + OCR + Rampart redaction pipeline."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from plva_coreml.ocr import OCRFinding, OCRPipeline, OCRResult
from plva_coreml.semantics import SemanticPipeline
from plva_coreml.visual_ane import (
    VisualANESession,
    prepare_fixed_visual_model,
    visual_model_cache_key,
)
from plva_coreml.visual_redactor import Region, detect_regions, render_masks


@dataclass(frozen=True, slots=True)
class HybridRegion:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    labels: tuple[str, ...]
    sources: tuple[str, ...]
    score: float


@dataclass(frozen=True, slots=True)
class HybridResult:
    png: bytes
    regions: tuple[HybridRegion, ...]
    findings: tuple[OCRFinding, ...]
    counts: dict[str, int]
    timings: dict[str, float]


class HybridANERedactor:
    """Own warm Core ML sessions and run independent detector branches concurrently."""

    def __init__(
        self,
        baseline: Path,
        cache: Path,
        *,
        profile: str = "high-recall",
        visual_model: Path | None = None,
        visual_enabled: bool = True,
        semantic_engine: str = "rampart",
    ) -> None:
        self._profile = profile
        self._visual: VisualANESession | None = None
        if visual_enabled:
            source_visual_model = visual_model or baseline / "dist/visual/detector.onnx"
            visual_key = visual_model_cache_key(source_visual_model)
            prepared_visual_model = prepare_fixed_visual_model(
                source_visual_model,
                cache / "models" / f"visual-fixed-{visual_key}.onnx",
            )
            self._visual = VisualANESession(
                prepared_visual_model,
                cache_directory=cache / "compiled/visual" / visual_key,
            )
        self._ocr = OCRPipeline(baseline, cache)
        self._semantic = SemanticPipeline(baseline, cache, engine=semantic_engine)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="plva-detect")

    def warm(self) -> None:
        visual = self._executor.submit(self._visual.warm) if self._visual is not None else None
        ocr = self._executor.submit(self._ocr.warm)
        if visual is not None:
            visual.result()
        ocr.result()
        self._semantic.warm()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def classify_texts(self, texts: tuple[str, ...]) -> tuple[OCRFinding, ...]:
        """Run the warm Core ML Rampart/rule path over outbound history strings."""

        findings = tuple(
            OCRFinding(0, index, 1, index + 1, text, 1.0, 1.0)
            for index, text in enumerate(texts)
        )
        return self._semantic.classify(findings).findings

    def process(self, source: Image.Image) -> HybridResult:
        total_started = time.perf_counter()
        visual_future = (
            self._executor.submit(
                detect_regions, self._visual, source.copy(), profile=self._profile
            )
            if self._visual is not None
            else None
        )
        ocr_future = self._executor.submit(self._ocr.recognize, source.copy())
        visual = visual_future.result() if visual_future is not None else None
        ocr: OCRResult = ocr_future.result()
        semantic = self._semantic.classify(ocr.findings)
        ocr_regions = _ocr_regions(semantic.findings, profile=self._profile)
        visual_regions = tuple(
            HybridRegion(
                region.x1,
                region.y1,
                region.x2,
                region.y2,
                region.label,
                (region.label,),
                ("VISUAL",),
                region.score,
            )
            for region in (visual.regions if visual is not None else ())
        )
        fused = _fuse_regions((*visual_regions, *ocr_regions))
        render_started = time.perf_counter()
        masks = tuple(
            Region(
                region.x1,
                region.y1,
                region.x2,
                region.y2,
                -1,
                region.label,
                region.score,
            )
            for region in fused
        )
        png = render_masks(source, masks)
        render_ms = (time.perf_counter() - render_started) * 1000
        return HybridResult(
            png=png,
            regions=fused,
            findings=semantic.findings,
            counts={
                "visual": len(visual.regions) if visual is not None else 0,
                "ocr_detected": ocr.detected_count,
                "ocr_recognized": sum(bool(finding.text) for finding in semantic.findings),
                "ocr_uncertain": sum(finding.uncertain for finding in semantic.findings),
                "ocr_sensitive": semantic.sensitive_count,
                "fused": len(fused),
            },
            timings={
                "visual_ms": visual.total_ms if visual is not None else 0.0,
                "visual_inference_ms": visual.inference_ms if visual is not None else 0.0,
                "ocr_ms": ocr.total_ms,
                "ocr_detector_ms": ocr.detector_ms,
                "ocr_recognizer_ms": ocr.recognizer_ms,
                "rampart_ms": semantic.rampart_ms,
                "render_ms": render_ms,
                "total_ms": (time.perf_counter() - total_started) * 1000,
            },
        )


def _ocr_regions(findings: tuple[OCRFinding, ...], *, profile: str) -> tuple[HybridRegion, ...]:
    regions: list[HybridRegion] = []
    for finding in findings:
        if not finding.sensitive or (finding.uncertain and profile != "high-recall"):
            continue
        labels = finding.labels or ("UNREADABLE",)
        value_scores = [value.score for value in finding.values]
        score = max(finding.detector_score, finding.ocr_confidence, *value_scores)
        regions.append(
            HybridRegion(
                finding.x1,
                finding.y1,
                finding.x2,
                finding.y2,
                " + ".join(labels),
                labels,
                finding.sources,
                score,
            )
        )
    return tuple(regions)


def _fuse_regions(regions: tuple[HybridRegion, ...]) -> tuple[HybridRegion, ...]:
    fused: list[HybridRegion] = []
    for candidate in sorted(regions, key=lambda region: region.score, reverse=True):
        overlapping = [
            index for index, current in enumerate(fused) if _should_merge(candidate, current)
        ]
        if not overlapping:
            fused.append(candidate)
            continue
        merged = candidate
        for index in reversed(overlapping):
            merged = _union(merged, fused.pop(index))
        changed = True
        while changed:
            changed = False
            for index in range(len(fused) - 1, -1, -1):
                if _should_merge(merged, fused[index]):
                    merged = _union(merged, fused.pop(index))
                    changed = True
        fused.append(merged)
    return tuple(sorted(fused, key=lambda region: (region.y1, region.x1)))


def _should_merge(left: HybridRegion, right: HybridRegion) -> bool:
    intersection = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1)) * max(
        0.0, min(left.y2, right.y2) - max(left.y1, right.y1)
    )
    if intersection <= 0:
        return False
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return (
        intersection / max(1.0, union) >= 0.35
        or intersection / max(1.0, min(left_area, right_area)) >= 0.55
    )


def _union(left: HybridRegion, right: HybridRegion) -> HybridRegion:
    labels = tuple(dict.fromkeys((*left.labels, *right.labels)))
    sources = tuple(dict.fromkeys((*left.sources, *right.sources)))
    return HybridRegion(
        min(left.x1, right.x1),
        min(left.y1, right.y1),
        max(left.x2, right.x2),
        max(left.y2, right.y2),
        " + ".join(labels),
        labels,
        sources,
        max(left.score, right.score),
    )
