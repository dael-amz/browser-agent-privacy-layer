"""Low-latency native Vision OCR + Core ML visual/Rampart redaction pipeline."""

from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

from PIL import Image

from plva_coreml.hybrid import HybridRegion, HybridResult, _fuse_regions, _ocr_regions
from plva_coreml.ocr import OCRFinding
from plva_coreml.semantics import SemanticPipeline
from plva_coreml.vision import VisionOCRClient, VisionResult, VisionROI
from plva_coreml.visual_ane import VisualANESession, prepare_fixed_visual_model
from plva_coreml.visual_redactor import Region, detect_regions, render_masks

VISION_PIPELINE_MODES: Final = ("fast", "cascade", "accurate")


class HybridVisionRedactor:
    """Run native Vision OCR and the Core ML visual detector concurrently."""

    def __init__(
        self,
        baseline: Path,
        cache: Path,
        *,
        profile: str = "high-recall",
        mode: str = "cascade",
    ) -> None:
        if mode not in VISION_PIPELINE_MODES:
            raise ValueError(f"mode must be one of: {', '.join(VISION_PIPELINE_MODES)}")
        self._profile = profile
        self._mode = mode
        visual_model = prepare_fixed_visual_model(
            baseline / "dist/visual/detector.onnx", cache / "models/visual-fixed.onnx"
        )
        self._visual = VisualANESession(visual_model, cache_directory=cache / "compiled/visual")
        self._vision = VisionOCRClient(cache)
        self._semantic = SemanticPipeline(baseline, cache)
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="plva-vision")

    def warm(self) -> None:
        visual = self._executor.submit(self._visual.warm)
        vision = self._executor.submit(self._vision.warm)
        semantic = self._executor.submit(self._semantic.warm)
        visual.result()
        vision.result()
        semantic.result()

    def close(self) -> None:
        self._vision.close()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def process(self, png: bytes) -> HybridResult:
        total_started = time.perf_counter()
        try:
            with Image.open(io.BytesIO(png)) as loaded:
                source = loaded.convert("RGB")
                source.load()
        except (OSError, ValueError) as exc:
            raise RuntimeError("frame is not a decodable image") from exc
        visual_future = self._executor.submit(
            detect_regions, self._visual, source.copy(), profile=self._profile
        )
        initial_mode = "accurate" if self._mode == "accurate" else "fast"
        vision_future = self._executor.submit(
            self._vision.recognize,
            png,
            source.width,
            source.height,
            mode=initial_mode,
        )
        initial_vision: VisionResult = vision_future.result()
        semantic = self._semantic.classify(initial_vision.findings)
        accurate_ms = 0.0
        if self._mode == "cascade":
            rois = _fallback_rois(semantic.findings, source.width, source.height)
            if rois:
                accurate = self._vision.recognize(
                    png,
                    source.width,
                    source.height,
                    mode="accurate",
                    rois=rois,
                )
                accurate_ms = accurate.duration_ms
                combined = _merge_accurate(
                    initial_vision.findings,
                    accurate.findings,
                    rois,
                    source.width,
                    source.height,
                )
                semantic = self._semantic.classify(combined)
        visual = visual_future.result()
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
            for region in visual.regions
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
        redacted = render_masks(source, masks)
        render_ms = (time.perf_counter() - render_started) * 1000
        return HybridResult(
            png=redacted,
            regions=fused,
            findings=semantic.findings,
            counts={
                "visual": len(visual.regions),
                "ocr_detected": len(semantic.findings),
                "ocr_recognized": sum(bool(finding.text) for finding in semantic.findings),
                "ocr_uncertain": sum(finding.uncertain for finding in semantic.findings),
                "ocr_sensitive": semantic.sensitive_count,
                "fused": len(fused),
            },
            timings={
                "visual_ms": visual.total_ms,
                "visual_inference_ms": visual.inference_ms,
                "ocr_ms": initial_vision.duration_ms + accurate_ms,
                "vision_fast_ms": initial_vision.duration_ms if initial_mode == "fast" else 0.0,
                "vision_accurate_ms": (
                    initial_vision.duration_ms if initial_mode == "accurate" else accurate_ms
                ),
                "rampart_ms": semantic.rampart_ms,
                "render_ms": render_ms,
                "total_ms": (time.perf_counter() - total_started) * 1000,
            },
        )


def _fallback_rois(
    findings: tuple[OCRFinding, ...], width: int, height: int
) -> tuple[VisionROI, ...]:
    rois: list[VisionROI] = []
    for finding in findings:
        if not finding.sensitive and not finding.uncertain:
            continue
        padding = max(4.0, min(16.0, (finding.y2 - finding.y1) * 0.3))
        x1 = max(0.0, finding.x1 - padding)
        y1 = max(0.0, finding.y1 - padding)
        x2 = min(float(width), finding.x2 + padding)
        y2 = min(float(height), finding.y2 + padding)
        candidate = VisionROI(x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height)
        merged = False
        for index, current in enumerate(rois):
            if _roi_intersects(candidate, current):
                left = min(candidate.x, current.x)
                top = min(candidate.y, current.y)
                right = max(candidate.x + candidate.width, current.x + current.width)
                bottom = max(candidate.y + candidate.height, current.y + current.height)
                rois[index] = VisionROI(left, top, right - left, bottom - top)
                merged = True
                break
        if not merged:
            rois.append(candidate)
    return tuple(rois)


def _merge_accurate(
    fast: tuple[OCRFinding, ...],
    accurate: tuple[OCRFinding, ...],
    rois: tuple[VisionROI, ...],
    width: int,
    height: int,
) -> tuple[OCRFinding, ...]:
    if not accurate:
        return fast
    retained = [finding for finding in fast if not _inside_any_roi(finding, rois, width, height)]
    retained.extend(accurate)
    return tuple(sorted(retained, key=lambda finding: (finding.y1, finding.x1)))


def _inside_any_roi(
    finding: OCRFinding, rois: tuple[VisionROI, ...], width: int, height: int
) -> bool:
    center_x = (finding.x1 + finding.x2) / (2 * width)
    center_y = (finding.y1 + finding.y2) / (2 * height)
    for roi in rois:
        if roi.x <= center_x <= roi.x + roi.width and roi.y <= center_y <= roi.y + roi.height:
            return True
    return False


def _roi_intersects(left: VisionROI, right: VisionROI) -> bool:
    return not (
        left.x + left.width < right.x
        or right.x + right.width < left.x
        or left.y + left.height < right.y
        or right.y + right.height < left.y
    )
