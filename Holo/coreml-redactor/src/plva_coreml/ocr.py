"""Core ML accelerated RapidOCR detection, recognition, and findings emission."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from functools import cmp_to_key
from pathlib import Path
from typing import Final

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy import ndimage

from plva_coreml.coreml_session import create_ane_session
from plva_coreml.model_cache import prepare_fixed_model

DETECTOR_SIZE: Final = 960
RECOGNIZER_BATCH: Final = 6
RECOGNIZER_HEIGHT: Final = 48
RECOGNIZER_WIDTH: Final = 320
WIDE_RECOGNIZER_WIDTH: Final = 1536
MIN_TEXT_CONFIDENCE: Final = 0.5


@dataclass(frozen=True, slots=True)
class OCRBox:
    x1: float
    y1: float
    x2: float
    y2: float
    detector_score: float


@dataclass(frozen=True, slots=True)
class PIIValue:
    label: str
    value: str
    start: int
    end: int
    score: float
    source: str


@dataclass(frozen=True, slots=True)
class OCRFinding:
    x1: float
    y1: float
    x2: float
    y2: float
    text: str
    detector_score: float
    ocr_confidence: float
    labels: tuple[str, ...] = ()
    sources: tuple[str, ...] = ("OCR",)
    values: tuple[PIIValue, ...] = ()
    sensitive: bool = False
    uncertain: bool = False


@dataclass(frozen=True, slots=True)
class OCRResult:
    findings: tuple[OCRFinding, ...]
    detected_count: int
    detector_ms: float
    recognizer_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class _OCRTransform:
    scale: float
    pad_left: int
    pad_top: int
    source_width: int
    source_height: int


class OCRPipeline:
    """Warm fixed-shape OCR models with text retained only in returned findings."""

    def __init__(self, baseline: Path, cache: Path) -> None:
        detector = prepare_fixed_model(
            baseline / "dist/ocr/ch_PP-OCRv4_det_mobile.onnx",
            cache / "models/ocr-detector-960.onnx",
            {"x": (1, 3, DETECTOR_SIZE, DETECTOR_SIZE)},
        )
        recognizer = prepare_fixed_model(
            baseline / "dist/ocr/en_PP-OCRv4_rec_mobile.onnx",
            cache / "models/ocr-recognizer-6x320.onnx",
            {"x": (RECOGNIZER_BATCH, 3, RECOGNIZER_HEIGHT, RECOGNIZER_WIDTH)},
        )
        wide_recognizer = prepare_fixed_model(
            baseline / "dist/ocr/en_PP-OCRv4_rec_mobile.onnx",
            cache / "models/ocr-recognizer-1x1536.onnx",
            {"x": (1, 3, RECOGNIZER_HEIGHT, WIDE_RECOGNIZER_WIDTH)},
        )
        self._detector = create_ane_session(detector, cache_directory=cache / "compiled/ocr-det")
        self._recognizer = create_ane_session(
            recognizer, cache_directory=cache / "compiled/ocr-rec"
        )
        self._wide_recognizer = create_ane_session(
            wide_recognizer, cache_directory=cache / "compiled/ocr-rec-wide"
        )
        raw_dictionary = (baseline / "dist/ocr/en_dict.txt").read_text("utf-8")
        lines = raw_dictionary.splitlines()
        self._dictionary = ("<blank>", *lines, " ")

    def warm(self) -> None:
        self._run_detector(np.zeros((1, 3, DETECTOR_SIZE, DETECTOR_SIZE), np.float32))
        self._run_recognizer(
            self._recognizer,
            np.zeros(
                (RECOGNIZER_BATCH, 3, RECOGNIZER_HEIGHT, RECOGNIZER_WIDTH),
                np.float32,
            ),
        )
        self._run_recognizer(
            self._wide_recognizer,
            np.zeros((1, 3, RECOGNIZER_HEIGHT, WIDE_RECOGNIZER_WIDTH), np.float32),
        )

    def recognize(self, source: Image.Image) -> OCRResult:
        total_started = time.perf_counter()
        tensor, transform = _detector_tensor(source)
        started = time.perf_counter()
        probability = self._run_detector(tensor)
        boxes = _extract_boxes(probability, transform)
        detector_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        findings = self._recognize_boxes(source, boxes)
        recognizer_ms = (time.perf_counter() - started) * 1000
        return OCRResult(
            findings=findings,
            detected_count=len(boxes),
            detector_ms=detector_ms,
            recognizer_ms=recognizer_ms,
            total_ms=(time.perf_counter() - total_started) * 1000,
        )

    def _run_detector(self, tensor: np.ndarray) -> np.ndarray:
        output = self._detector.run(None, {"x": tensor})[0]
        if not isinstance(output, np.ndarray) or output.shape != (
            1,
            1,
            DETECTOR_SIZE,
            DETECTOR_SIZE,
        ):
            raise RuntimeError("OCR detector returned an unexpected output")
        return np.asarray(output[0, 0])

    def _run_recognizer(self, session: ort.InferenceSession, tensor: np.ndarray) -> np.ndarray:
        output = session.run(None, {"x": tensor})[0]
        if not isinstance(output, np.ndarray) or output.ndim != 3:
            raise RuntimeError("OCR recognizer returned an unexpected output")
        if output.shape[0] != tensor.shape[0] or output.shape[2] != len(self._dictionary):
            raise RuntimeError("OCR recognizer output does not match its dictionary")
        return output

    def _recognize_boxes(
        self, source: Image.Image, boxes: tuple[OCRBox, ...]
    ) -> tuple[OCRFinding, ...]:
        ordered = sorted(
            enumerate(boxes),
            key=lambda entry: max(
                1.0,
                (entry[1].x2 - entry[1].x1) / max(1.0, entry[1].y2 - entry[1].y1),
            ),
        )
        results: list[OCRFinding | None] = [None] * len(boxes)
        for offset in range(0, len(ordered), RECOGNIZER_BATCH):
            batch = ordered[offset : offset + RECOGNIZER_BATCH]
            tensor = np.zeros(
                (RECOGNIZER_BATCH, 3, RECOGNIZER_HEIGHT, RECOGNIZER_WIDTH),
                dtype=np.float32,
            )
            for sample, (_, box) in enumerate(batch):
                _put_recognizer_crop(source, box, tensor[sample])
            probabilities = self._run_recognizer(self._recognizer, tensor)
            for sample, (original_index, box) in enumerate(batch):
                text, confidence = _decode_ctc(probabilities[sample], self._dictionary)
                valid = confidence >= MIN_TEXT_CONFIDENCE and bool(text.strip())
                results[original_index] = OCRFinding(
                    x1=box.x1,
                    y1=box.y1,
                    x2=box.x2,
                    y2=box.y2,
                    text=text if valid else "",
                    detector_score=box.detector_score,
                    ocr_confidence=confidence,
                    labels=() if valid else ("UNREADABLE",),
                    sources=("OCR",) if valid else ("OCR+UNCERTAIN",),
                    sensitive=not valid,
                    uncertain=not valid,
                )
        for index, result in enumerate(results):
            if result is None or not result.uncertain:
                continue
            box = boxes[index]
            ratio = (box.x2 - box.x1) / max(1.0, box.y2 - box.y1)
            if ratio <= RECOGNIZER_WIDTH / RECOGNIZER_HEIGHT:
                continue
            tensor = np.zeros((1, 3, RECOGNIZER_HEIGHT, WIDE_RECOGNIZER_WIDTH), dtype=np.float32)
            _put_recognizer_crop(source, box, tensor[0])
            probabilities = self._run_recognizer(self._wide_recognizer, tensor)
            text, confidence = _decode_ctc(probabilities[0], self._dictionary)
            if confidence >= MIN_TEXT_CONFIDENCE and text.strip():
                results[index] = replace(
                    result,
                    text=text,
                    ocr_confidence=confidence,
                    labels=(),
                    sources=("OCR",),
                    sensitive=False,
                    uncertain=False,
                )
        return tuple(result for result in results if result is not None)


def _detector_tensor(source: Image.Image) -> tuple[np.ndarray, _OCRTransform]:
    source = source.convert("RGB")
    scale = min(DETECTOR_SIZE / source.width, DETECTOR_SIZE / source.height)
    width = max(1, round(source.width * scale))
    height = max(1, round(source.height * scale))
    left = (DETECTOR_SIZE - width) // 2
    top = (DETECTOR_SIZE - height) // 2
    resized = source.resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (DETECTOR_SIZE, DETECTOR_SIZE), "white")
    canvas.paste(resized, (left, top))
    pixels = np.asarray(canvas, dtype=np.float32)
    bgr = pixels[:, :, ::-1] / 127.5 - 1.0
    tensor = np.ascontiguousarray(bgr.transpose(2, 0, 1)[None])
    return tensor, _OCRTransform(scale, left, top, source.width, source.height)


def _extract_boxes(probability: np.ndarray, transform: _OCRTransform) -> tuple[OCRBox, ...]:
    binary = probability > 0.3
    dilated = binary.copy()
    dilated[:, 1:] |= binary[:, :-1]
    dilated[1:, :] |= binary[:-1, :]
    dilated[1:, 1:] |= binary[:-1, :-1]
    components, count = ndimage.label(dilated, structure=np.ones((3, 3), dtype=np.uint8))
    slices = ndimage.find_objects(components, max_label=count)
    boxes: list[OCRBox] = []
    for component in slices:
        if component is None:
            continue
        y_slice, x_slice = component
        min_y, max_y = y_slice.start, y_slice.stop - 1
        min_x, max_x = x_slice.start, x_slice.stop - 1
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        if min(width, height) < 3:
            continue
        score = float(probability[y_slice, x_slice].mean())
        if score < 0.5:
            continue
        distance = (width * height * 1.6) / (2 * (width + height))
        if min(width + 2 * distance, height + 2 * distance) < 5:
            continue
        x1 = (min_x - distance - transform.pad_left) / transform.scale
        y1 = (min_y - distance - transform.pad_top) / transform.scale
        x2 = (max_x + 1 + distance - transform.pad_left) / transform.scale
        y2 = (max_y + 1 + distance - transform.pad_top) / transform.scale
        box = OCRBox(
            x1=min(transform.source_width, max(0.0, x1)),
            y1=min(transform.source_height, max(0.0, y1)),
            x2=min(transform.source_width, max(0.0, x2)),
            y2=min(transform.source_height, max(0.0, y2)),
            detector_score=score,
        )
        if box.x2 - box.x1 >= 2 and box.y2 - box.y1 >= 2:
            boxes.append(box)
        if len(boxes) >= 1000:
            break

    def reading_order(left: OCRBox, right: OCRBox) -> int:
        if abs(left.y1 - right.y1) <= 10:
            return -1 if left.x1 < right.x1 else (1 if left.x1 > right.x1 else 0)
        return -1 if left.y1 < right.y1 else 1

    return tuple(sorted(boxes, key=cmp_to_key(reading_order)))


def _put_recognizer_crop(source: Image.Image, box: OCRBox, target: np.ndarray) -> None:
    ratio = max(1.0, (box.x2 - box.x1) / max(1.0, box.y2 - box.y1))
    target_width = target.shape[2]
    resized_width = min(target_width, max(1, math.ceil(RECOGNIZER_HEIGHT * ratio)))
    crop = source.convert("RGB").crop(
        (math.floor(box.x1), math.floor(box.y1), math.ceil(box.x2), math.ceil(box.y2))
    )
    resized = crop.resize((resized_width, RECOGNIZER_HEIGHT), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)
    target[:, :, :resized_width] = (pixels[:, :, ::-1] / 127.5 - 1.0).transpose(2, 0, 1)


def _decode_ctc(probabilities: np.ndarray, dictionary: tuple[str, ...]) -> tuple[str, float]:
    best_classes = probabilities.argmax(axis=1)
    best_scores = probabilities.max(axis=1)
    previous = -1
    text: list[str] = []
    confidence = 0.0
    emitted = 0
    for class_id, score in zip(best_classes, best_scores, strict=True):
        index = int(class_id)
        if index != 0 and index != previous:
            text.append(dictionary[index])
            confidence += float(score)
            emitted += 1
        previous = index
    return "".join(text), confidence / emitted if emitted else 0.0
