"""Visual-only PLVA preprocessing, decoding, and irreversible mask rendering."""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from typing import Final

import numpy as np
from PIL import Image, ImageDraw

from plva_coreml.visual_ane import VisualANESession

MODEL_SIZE: Final = 640
DETECTOR_CLASSES: Final = (
    "NAME",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "CARD_NUMBER",
    "CVC",
    "SECRET",
    "SENSITIVE_FIELD",
    "SENSITIVE_IMAGE",
)
THRESHOLD_PROFILES: Final = {
    "high-recall": (0.1, 0.1, 0.1, 0.1, 0.08, 0.01, 0.08, 0.08, 0.35),
    "balanced": (0.25, 0.25, 0.25, 0.25, 0.2, 0.03, 0.2, 0.2, 0.5),
}
REDACTION_RGB: Final = (5, 8, 7)


@dataclass(frozen=True, slots=True)
class Transform:
    source_width: int
    source_height: int
    scale: float
    scaled_width: int
    scaled_height: int
    pad_left: int
    pad_top: int


@dataclass(frozen=True, slots=True)
class Region:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    label: str
    score: float


@dataclass(frozen=True, slots=True)
class RedactionResult:
    png: bytes
    regions: tuple[Region, ...]
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class VisualDetectionResult:
    regions: tuple[Region, ...]
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float


def prepare_tensor(source: Image.Image) -> tuple[np.ndarray, Transform]:
    """Apply the frozen detector's 640-square letterbox and NCHW normalization."""

    source = source.convert("RGB")
    scale = min(MODEL_SIZE / source.width, MODEL_SIZE / source.height)
    scaled_width = max(1, round(source.width * scale))
    scaled_height = max(1, round(source.height * scale))
    pad_left = (MODEL_SIZE - scaled_width) // 2
    pad_top = (MODEL_SIZE - scaled_height) // 2
    resized = source.resize((scaled_width, scaled_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (MODEL_SIZE, MODEL_SIZE), (114, 114, 114))
    canvas.paste(resized, (pad_left, pad_top))
    tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(tensor), Transform(
        source_width=source.width,
        source_height=source.height,
        scale=scale,
        scaled_width=scaled_width,
        scaled_height=scaled_height,
        pad_left=pad_left,
        pad_top=pad_top,
    )


def decode_detections(
    output: np.ndarray, transform: Transform, *, profile: str = "high-recall"
) -> tuple[Region, ...]:
    """Decode the frozen model output with class-aware NMS."""

    thresholds = THRESHOLD_PROFILES.get(profile)
    if thresholds is None:
        raise ValueError(f"unknown detector profile: {profile}")
    if output.shape != (1, 13, 8400):
        raise ValueError(f"unexpected detector output shape: {output.shape}")
    values = output[0]
    proposals: list[Region] = []
    for anchor in range(values.shape[1]):
        scores = values[4:, anchor]
        class_id = int(np.argmax(scores))
        score = float(scores[class_id])
        if score < thresholds[class_id]:
            continue
        center_x, center_y, width, height = (float(values[index, anchor]) for index in range(4))
        if not all(math.isfinite(value) for value in (center_x, center_y, width, height)):
            continue
        if width <= 0 or height <= 0:
            continue
        raw_x1 = (center_x - width / 2 - transform.pad_left) / transform.scale
        raw_y1 = (center_y - height / 2 - transform.pad_top) / transform.scale
        raw_x2 = (center_x + width / 2 - transform.pad_left) / transform.scale
        raw_y2 = (center_y + height / 2 - transform.pad_top) / transform.scale
        padding = max(2.0, min(max(0.0, raw_x2 - raw_x1), max(0.0, raw_y2 - raw_y1)) * 0.04)
        region = Region(
            x1=min(transform.source_width, max(0.0, raw_x1 - padding)),
            y1=min(transform.source_height, max(0.0, raw_y1 - padding)),
            x2=min(transform.source_width, max(0.0, raw_x2 + padding)),
            y2=min(transform.source_height, max(0.0, raw_y2 + padding)),
            class_id=class_id,
            label=DETECTOR_CLASSES[class_id],
            score=score,
        )
        if region.x2 - region.x1 >= 2 and region.y2 - region.y1 >= 2:
            proposals.append(region)
    return _non_maximum_suppression(proposals)


def _non_maximum_suppression(regions: list[Region]) -> tuple[Region, ...]:
    selected: list[Region] = []
    for candidate in sorted(regions, key=lambda region: region.score, reverse=True):
        if any(
            current.class_id == candidate.class_id
            and _intersection_over_union(current, candidate) > 0.7
            for current in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= 300:
            break
    return tuple(selected)


def _intersection_over_union(left: Region, right: Region) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def render_masks(source: Image.Image, regions: tuple[Region, ...]) -> bytes:
    """Burn opaque masks into a copy and return PNG bytes."""

    output = source.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for region in regions:
        x1 = min(output.width, max(0, math.floor(region.x1)))
        y1 = min(output.height, max(0, math.floor(region.y1)))
        x2 = min(output.width, max(0, math.ceil(region.x2)))
        y2 = min(output.height, max(0, math.ceil(region.y2)))
        if x2 > x1 and y2 > y1:
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=REDACTION_RGB)
    buffer = io.BytesIO()
    output.save(buffer, format="PNG")
    return buffer.getvalue()


def redact_image(
    session: VisualANESession, source: Image.Image, *, profile: str = "high-recall"
) -> RedactionResult:
    """Run one in-memory image through the experimental visual-only path."""

    detected = detect_regions(session, source, profile=profile)
    started = time.perf_counter()
    png = render_masks(source, detected.regions)
    render_ms = (time.perf_counter() - started) * 1000
    return RedactionResult(
        png=png,
        regions=detected.regions,
        preprocess_ms=detected.preprocess_ms,
        inference_ms=detected.inference_ms,
        postprocess_ms=detected.postprocess_ms + render_ms,
        total_ms=detected.total_ms + render_ms,
    )


def detect_regions(
    session: VisualANESession, source: Image.Image, *, profile: str = "high-recall"
) -> VisualDetectionResult:
    """Run the visual branch without rendering so it can overlap OCR."""

    total_started = time.perf_counter()
    started = time.perf_counter()
    tensor, transform = prepare_tensor(source)
    preprocess_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    output = session.infer(tensor)
    inference_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    regions = decode_detections(output, transform, profile=profile)
    postprocess_ms = (time.perf_counter() - started) * 1000
    return VisualDetectionResult(
        regions=regions,
        preprocess_ms=preprocess_ms,
        inference_ms=inference_ms,
        postprocess_ms=postprocess_ms,
        total_ms=(time.perf_counter() - total_started) * 1000,
    )
