from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

from training.schema import sha256_file
from training.visual.create_onnx_goldens import create as create_onnx_goldens
from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.train_detector import SECRET_VISUAL_CLASSES


IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
MODEL_SIZE = 640
MINIMUM_CALIBRATION_IMAGES = 256
MINIMUM_EVALUATION_IMAGES = 400
MINIMUM_CLASS_SUPPORT = 20
MINIMUM_SIZE_REDUCTION_FRACTION = 0.50
MINIMUM_CLASS_ARGMAX_AGREEMENT = 0.99
MAXIMUM_MEAN_CLASS_SCORE_ERROR = 0.005
MINIMUM_FP32_PROPOSAL_COVERAGE = 0.99
MAXIMUM_RECALL_DROP = 0.005
MAXIMUM_CLASS_RECALL_DROP = 0.01
MAXIMUM_PRECISION_DROP = 0.02
REQUIRED_PARITY_BACKENDS = ("node", "wasm")
RUNTIME_PACKAGE_VERSION = "1.27.0"

THRESHOLD_PROFILES = {
    "high-recall": {
        "NAME": 0.10,
        "EMAIL": 0.10,
        "PHONE": 0.10,
        "ADDRESS": 0.10,
        "CARD_NUMBER": 0.08,
        "CVC": 0.01,
        "SECRET": 0.08,
        "SENSITIVE_FIELD": 0.08,
        "SENSITIVE_IMAGE": 0.35,
    },
    "balanced": {
        "NAME": 0.25,
        "EMAIL": 0.25,
        "PHONE": 0.25,
        "ADDRESS": 0.25,
        "CARD_NUMBER": 0.20,
        "CVC": 0.03,
        "SECRET": 0.20,
        "SENSITIVE_FIELD": 0.20,
        "SENSITIVE_IMAGE": 0.50,
    },
}


@dataclass(frozen=True)
class Detection:
    class_id: int
    score: float
    box: tuple[float, float, float, float]


def _images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"no supported images found under {root}")
    return paths


def _evenly_spaced(paths: Sequence[Path], maximum: int) -> list[Path]:
    if maximum <= 0 or len(paths) <= maximum:
        return list(paths)
    return [paths[index * len(paths) // maximum] for index in range(maximum)]


def _content_hashes(paths: Iterable[Path]) -> set[str]:
    return {sha256_file(path) for path in paths}


def _label_path(image: Path, images_root: Path, labels_root: Path) -> Path:
    return (labels_root / image.relative_to(images_root)).with_suffix(".txt")


def _truth(
    image: Path, images_root: Path, labels_root: Path
) -> list[tuple[int, tuple[float, float, float, float]]]:
    from PIL import Image

    label_path = _label_path(image, images_root, labels_root)
    if not label_path.is_file():
        raise RuntimeError(f"missing evaluation label for {image}: {label_path}")
    with Image.open(image) as source:
        source_width, source_height = source.size
    scale = min(MODEL_SIZE / source_width, MODEL_SIZE / source_height)
    scaled_width = max(1, round(source_width * scale))
    scaled_height = max(1, round(source_height * scale))
    pad_left = (MODEL_SIZE - scaled_width) // 2
    pad_top = (MODEL_SIZE - scaled_height) // 2
    annotations: list[tuple[int, tuple[float, float, float, float]]] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise RuntimeError(f"invalid YOLO row in {label_path}:{line_number}")
        try:
            class_id = int(parts[0])
            center_x, center_y, width, height = map(float, parts[1:])
        except ValueError as exc:
            raise RuntimeError(
                f"invalid YOLO value in {label_path}:{line_number}"
            ) from exc
        if class_id not in range(len(DETECTOR_CLASSES)):
            raise RuntimeError(
                f"unknown class {class_id} in {label_path}:{line_number}"
            )
        values = (center_x, center_y, width, height)
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"non-finite YOLO row in {label_path}:{line_number}")
        if not (
            0 <= center_x <= 1
            and 0 <= center_y <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            raise RuntimeError(
                f"out-of-range YOLO row in {label_path}:{line_number}"
            )
        annotations.append(
            (
                class_id,
                (
                    (center_x - width / 2) * source_width * scale + pad_left,
                    (center_y - height / 2) * source_height * scale + pad_top,
                    (center_x + width / 2) * source_width * scale + pad_left,
                    (center_y + height / 2) * source_height * scale + pad_top,
                ),
            )
        )
    return annotations


def _class_support(
    images: Sequence[Path], images_root: Path, labels_root: Path
) -> Counter[int]:
    support: Counter[int] = Counter()
    for image in images:
        support.update(class_id for class_id, _ in _truth(image, images_root, labels_root))
    return support


def _letterbox(path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    scale = min(MODEL_SIZE / image.width, MODEL_SIZE / image.height)
    # Python round is ties-to-even, matching the browser preprocessing contract.
    scaled_width = max(1, round(image.width * scale))
    scaled_height = max(1, round(image.height * scale))
    resized = image.resize((scaled_width, scaled_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (MODEL_SIZE, MODEL_SIZE), (114, 114, 114))
    canvas.paste(
        resized,
        ((MODEL_SIZE - scaled_width) // 2, (MODEL_SIZE - scaled_height) // 2),
    )
    values = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)
    return values[None, ...] / np.float32(255.0)


class CalibrationReader:
    def __init__(self, paths: Sequence[Path], input_name: str) -> None:
        self.paths = list(paths)
        self.input_name = input_name
        self.rewind()

    def get_next(self) -> dict[str, Any] | None:
        try:
            path = next(self._iterator)
        except StopIteration:
            return None
        return {self.input_name: _letterbox(path)}

    def rewind(self) -> None:
        self._iterator = iter(self.paths)


def _model_input_name(path: Path) -> str:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].type != "tensor(float)":
        raise RuntimeError("detector must have exactly one float32 input")
    return str(inputs[0].name)


def quantize_qdq(
    source: Path,
    destination: Path,
    calibration_images: Sequence[Path],
) -> None:
    """Create the only candidate representation accepted by the browser gate.

    UINT8 activations avoid the severe S8S8 accuracy loss observed on the
    detector. QDQ keeps ordinary Conv nodes in the graph; ConvInteger and
    QLinearConv candidates are intentionally not produced.
    """
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    destination.parent.mkdir(parents=True, exist_ok=True)
    preprocessed = destination.parent / "detector.preprocessed.onnx"
    quant_pre_process(
        str(source),
        str(preprocessed),
        skip_optimization=False,
        skip_onnx_shape=False,
        # ORT 1.27 symbolic inference is incomplete for the dynamic YOLO11
        # graph. Standard ONNX shape inference still succeeds and is checked.
        skip_symbolic_shape=True,
        save_as_external_data=False,
        all_tensors_to_one_file=False,
    )
    reader = CalibrationReader(calibration_images, _model_input_name(preprocessed))
    quantize_static(
        str(preprocessed),
        str(destination),
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        reduce_range=False,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv"],
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    preprocessed.unlink(missing_ok=True)


def graph_contract(source: Path, candidate: Path) -> dict[str, Any]:
    import onnx

    source_model = onnx.load(str(source), load_external_data=False)
    candidate_model = onnx.load(str(candidate), load_external_data=False)
    onnx.checker.check_model(source_model)
    onnx.checker.check_model(candidate_model)

    def signature(model: Any) -> dict[str, list[dict[str, Any]]]:
        def values(items: Any) -> list[dict[str, Any]]:
            return [
                {
                    "name": item.name,
                    "element_type": int(item.type.tensor_type.elem_type),
                    "dimensions": [
                        (
                            int(dimension.dim_value)
                            if dimension.HasField("dim_value")
                            else str(dimension.dim_param)
                        )
                        for dimension in item.type.tensor_type.shape.dim
                    ],
                }
                for item in items
            ]

        return {"inputs": values(model.graph.input), "outputs": values(model.graph.output)}

    source_signature = signature(source_model)
    candidate_signature = signature(candidate_model)
    operators = Counter(node.op_type for node in candidate_model.graph.node)
    forbidden = sorted(
        name
        for name in ("ConvInteger", "DynamicQuantizeLinear", "QLinearConv")
        if operators[name]
    )
    checks = {
        "onnx_checker_passed": True,
        "signature_preserved": source_signature == candidate_signature,
        "has_qdq": operators["QuantizeLinear"] > 0
        and operators["DequantizeLinear"] > 0,
        "forbidden_operators": forbidden,
    }
    return {
        "passed": all(
            (
                checks["onnx_checker_passed"],
                checks["signature_preserved"],
                checks["has_qdq"],
                not checks["forbidden_operators"],
            )
        ),
        "checks": checks,
        "operators": dict(sorted(operators.items())),
        "source_signature": source_signature,
        "candidate_signature": candidate_signature,
    }


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _decode(output: Any, profile: str) -> list[Detection]:
    import numpy as np

    if output.shape[0] != 1 or output.shape[1] != 4 + len(DETECTOR_CLASSES):
        raise RuntimeError(f"unexpected detector output shape: {output.shape}")
    channels = output[0]
    class_scores = channels[4:]
    class_ids = np.argmax(class_scores, axis=0)
    anchors = np.arange(class_scores.shape[1])
    scores = class_scores[class_ids, anchors]
    thresholds = THRESHOLD_PROFILES[profile]
    detections: list[Detection] = []
    for anchor in np.flatnonzero(
        scores
        >= np.asarray(
            [thresholds[DETECTOR_CLASSES[int(class_id)]] for class_id in class_ids],
            dtype=np.float32,
        )
    ):
        center_x, center_y, width, height = map(float, channels[:4, anchor])
        if not all(map(math.isfinite, (center_x, center_y, width, height))):
            continue
        if width <= 0 or height <= 0:
            continue
        box = (
            max(0.0, center_x - width / 2),
            max(0.0, center_y - height / 2),
            min(float(MODEL_SIZE), center_x + width / 2),
            min(float(MODEL_SIZE), center_y + height / 2),
        )
        if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
            detections.append(
                Detection(int(class_ids[anchor]), float(scores[anchor]), box)
            )
    selected: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        if any(
            detection.class_id == current.class_id
            and _iou(detection.box, current.box) > 0.7
            for current in selected
        ):
            continue
        selected.append(detection)
        if len(selected) == 300:
            break
    return selected


def _truth_matches(
    truth: Sequence[tuple[int, tuple[float, float, float, float]]],
    predictions: Sequence[Detection],
) -> tuple[Counter[int], Counter[int]]:
    support: Counter[int] = Counter(class_id for class_id, _ in truth)
    matched: Counter[int] = Counter()
    used: set[int] = set()
    for class_id, box in truth:
        candidates = [
            (_iou(box, prediction.box), index)
            for index, prediction in enumerate(predictions)
            if prediction.class_id == class_id and index not in used
        ]
        overlap, index = max(candidates, default=(0.0, -1))
        if overlap >= 0.5:
            matched[class_id] += 1
            used.add(index)
    return support, matched


def _metrics(
    support: Counter[int], matched: Counter[int], detections: Counter[int]
) -> dict[str, Any]:
    total_support = sum(support.values())
    total_matched = sum(matched.values())
    total_detections = sum(detections.values())
    per_class = {}
    for class_id, name in enumerate(DETECTOR_CLASSES):
        class_support = support[class_id]
        class_matched = matched[class_id]
        class_detections = detections[class_id]
        per_class[name] = {
            "support": class_support,
            "matched": class_matched,
            "detections": class_detections,
            "recall": class_matched / class_support if class_support else 0.0,
            "precision": (
                class_matched / class_detections if class_detections else 0.0
            ),
        }
    return {
        "support": total_support,
        "matched": total_matched,
        "detections": total_detections,
        "recall": total_matched / total_support if total_support else 0.0,
        "precision": total_matched / total_detections if total_detections else 0.0,
        "per_class": per_class,
    }


def compare_models(
    source: Path,
    candidate: Path,
    images: Sequence[Path],
    images_root: Path,
    labels_root: Path,
) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    source_session = ort.InferenceSession(
        str(source), providers=["CPUExecutionProvider"]
    )
    candidate_session = ort.InferenceSession(
        str(candidate), providers=["CPUExecutionProvider"]
    )
    source_input = source_session.get_inputs()[0].name
    candidate_input = candidate_session.get_inputs()[0].name
    if source_input != candidate_input:
        raise RuntimeError("quantized detector input name differs from FP32")

    raw_argmax_matches = 0
    raw_anchors = 0
    raw_error_sum = 0.0
    raw_score_count = 0
    raw_maximum_error = 0.0
    non_finite_values = 0
    profile_counts = {
        profile: {
            "support": Counter(),
            "source_matched": Counter(),
            "candidate_matched": Counter(),
            "source_detections": Counter(),
            "candidate_detections": Counter(),
            "source_proposals": 0,
            "source_proposals_covered": 0,
        }
        for profile in THRESHOLD_PROFILES
    }

    for image_path in images:
        image = _letterbox(image_path)
        source_output = source_session.run(None, {source_input: image})[0]
        candidate_output = candidate_session.run(None, {candidate_input: image})[0]
        if source_output.shape != candidate_output.shape:
            raise RuntimeError("quantized detector output shape differs from FP32")
        non_finite_values += int(np.size(candidate_output) - np.isfinite(candidate_output).sum())
        source_scores = source_output[:, 4:, :]
        candidate_scores = candidate_output[:, 4:, :]
        errors = np.abs(source_scores - candidate_scores)
        raw_error_sum += float(errors.sum(dtype=np.float64))
        raw_score_count += int(errors.size)
        raw_maximum_error = max(raw_maximum_error, float(errors.max(initial=0.0)))
        raw_argmax_matches += int(
            np.count_nonzero(
                np.argmax(source_scores, axis=1)
                == np.argmax(candidate_scores, axis=1)
            )
        )
        raw_anchors += int(source_scores.shape[-1])

        truth = _truth(image_path, images_root, labels_root)
        for profile, counters in profile_counts.items():
            source_detections = _decode(source_output, profile)
            candidate_detections = _decode(candidate_output, profile)
            support, source_matched = _truth_matches(truth, source_detections)
            _, candidate_matched = _truth_matches(truth, candidate_detections)
            counters["support"].update(support)
            counters["source_matched"].update(source_matched)
            counters["candidate_matched"].update(candidate_matched)
            counters["source_detections"].update(
                detection.class_id for detection in source_detections
            )
            counters["candidate_detections"].update(
                detection.class_id for detection in candidate_detections
            )
            counters["source_proposals"] += len(source_detections)
            counters["source_proposals_covered"] += sum(
                any(
                    source_detection.class_id == candidate_detection.class_id
                    and _iou(source_detection.box, candidate_detection.box) >= 0.9
                    for candidate_detection in candidate_detections
                )
                for source_detection in source_detections
            )

    profiles: dict[str, Any] = {}
    for profile, counters in profile_counts.items():
        source_metrics = _metrics(
            counters["support"],
            counters["source_matched"],
            counters["source_detections"],
        )
        candidate_metrics = _metrics(
            counters["support"],
            counters["candidate_matched"],
            counters["candidate_detections"],
        )
        source_proposals = counters["source_proposals"]
        profiles[profile] = {
            "fp32": source_metrics,
            "candidate": candidate_metrics,
            "fp32_proposal_coverage": (
                counters["source_proposals_covered"] / source_proposals
                if source_proposals
                else 0.0
            ),
        }
    return {
        "images": len(images),
        "raw_output": {
            "non_finite_candidate_values": non_finite_values,
            "class_argmax_agreement": raw_argmax_matches / raw_anchors,
            "mean_class_score_absolute_error": raw_error_sum / raw_score_count,
            "maximum_class_score_absolute_error": raw_maximum_error,
        },
        "profiles": profiles,
    }


def run_cross_runtime_parity(
    candidate: Path, work_dir: Path, runtime_parity_dir: Path
) -> dict[str, Any]:
    runtime_parity_dir = runtime_parity_dir.resolve()
    verifier = runtime_parity_dir / "verify_visual.mjs"
    if not verifier.is_file():
        raise RuntimeError(f"missing visual parity verifier: {verifier}")
    artifact_dir = work_dir / "runtime-parity"
    artifact_dir.mkdir()
    shutil.copy2(candidate, artifact_dir / "detector.onnx")
    create_onnx_goldens(
        SimpleNamespace(
            model=artifact_dir / "detector.onnx",
            output=artifact_dir / "detector_goldens.json",
            image_size=MODEL_SIZE,
        )
    )
    report_path = artifact_dir / "visual_cross_runtime_report.json"
    for backend in REQUIRED_PARITY_BACKENDS:
        completed = subprocess.run(
            [
                "node",
                str(verifier),
                "--backend",
                backend,
                "--artifact-dir",
                str(artifact_dir),
                "--output",
                str(report_path),
            ],
            cwd=runtime_parity_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{backend} detector parity failed: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
    return json.loads(report_path.read_text(encoding="utf-8"))


def gate_report(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    source = report["source"]
    candidate = report["candidate"]
    reduction = 1.0 - candidate["bytes"] / source["bytes"]
    require(
        reduction >= MINIMUM_SIZE_REDUCTION_FRACTION,
        "candidate does not reduce model bytes by at least 50%",
    )
    require(report["graph_contract"]["passed"], "candidate graph contract failed")
    data = report["data"]
    require(
        data["calibration_images"] >= MINIMUM_CALIBRATION_IMAGES,
        f"fewer than {MINIMUM_CALIBRATION_IMAGES} calibration images",
    )
    require(
        data["evaluation_images"] >= MINIMUM_EVALUATION_IMAGES,
        f"fewer than {MINIMUM_EVALUATION_IMAGES} evaluation images",
    )
    require(
        data["calibration_evaluation_hash_overlap"] == 0,
        "calibration and evaluation images overlap by content hash",
    )
    for name in DETECTOR_CLASSES:
        require(
            data["class_support"].get(name, 0) >= MINIMUM_CLASS_SUPPORT,
            f"evaluation support for {name} is below {MINIMUM_CLASS_SUPPORT}",
        )

    comparison = report["comparison"]
    raw = comparison["raw_output"]
    require(
        raw["non_finite_candidate_values"] == 0,
        "candidate output contains non-finite values",
    )
    require(
        raw["class_argmax_agreement"] >= MINIMUM_CLASS_ARGMAX_AGREEMENT,
        "candidate class argmax agreement is below 99%",
    )
    require(
        raw["mean_class_score_absolute_error"]
        <= MAXIMUM_MEAN_CLASS_SCORE_ERROR,
        "candidate mean class-score error exceeds 0.005",
    )
    for profile, metrics in comparison["profiles"].items():
        fp32 = metrics["fp32"]
        quantized = metrics["candidate"]
        require(
            metrics["fp32_proposal_coverage"] >= MINIMUM_FP32_PROPOSAL_COVERAGE,
            f"{profile} candidate preserves fewer than 99% of FP32 proposals",
        )
        require(
            quantized["recall"] >= fp32["recall"] - MAXIMUM_RECALL_DROP,
            f"{profile} overall recall drops by more than 0.005",
        )
        require(
            quantized["precision"] >= fp32["precision"] - MAXIMUM_PRECISION_DROP,
            f"{profile} precision drops by more than 0.02",
        )
        for name in DETECTOR_CLASSES:
            fp32_recall = fp32["per_class"][name]["recall"]
            candidate_recall = quantized["per_class"][name]["recall"]
            allowed_drop = 0.0 if name in SECRET_VISUAL_CLASSES else MAXIMUM_CLASS_RECALL_DROP
            require(
                candidate_recall >= fp32_recall - allowed_drop,
                f"{profile} {name} recall regresses beyond its allowed bound",
            )

    parity = report["cross_runtime"]
    require(
        parity.get("model_sha256") == candidate["sha256"],
        "cross-runtime report belongs to a different candidate hash",
    )
    for backend in REQUIRED_PARITY_BACKENDS:
        backend_report = parity.get("backends", {}).get(backend, {})
        expected_package = "onnxruntime-node" if backend == "node" else "onnxruntime-web"
        require(backend_report.get("passed") is True, f"{backend} parity did not pass")
        require(
            backend_report.get("package") == expected_package,
            f"{backend} parity used the wrong package",
        )
        require(
            backend_report.get("package_version") == RUNTIME_PACKAGE_VERSION,
            f"{backend} parity used an unpinned runtime version",
        )
        require(
            int(backend_report.get("vectors_checked", 0)) > 0,
            f"{backend} parity checked no vectors",
        )

    return {
        "passed": not failures,
        "size_reduction_fraction": reduction,
        "failures": failures,
        "policy": {
            "minimum_size_reduction_fraction": MINIMUM_SIZE_REDUCTION_FRACTION,
            "minimum_class_argmax_agreement": MINIMUM_CLASS_ARGMAX_AGREEMENT,
            "maximum_mean_class_score_absolute_error": MAXIMUM_MEAN_CLASS_SCORE_ERROR,
            "minimum_fp32_proposal_coverage": MINIMUM_FP32_PROPOSAL_COVERAGE,
            "maximum_recall_drop": MAXIMUM_RECALL_DROP,
            "maximum_class_recall_drop": MAXIMUM_CLASS_RECALL_DROP,
            "secret_class_recall_drop": 0.0,
            "maximum_precision_drop": MAXIMUM_PRECISION_DROP,
            "required_parity_backends": list(REQUIRED_PARITY_BACKENDS),
            "runtime_package_version": RUNTIME_PACKAGE_VERSION,
        },
    }


def quantize_and_validate(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    if not source.is_file() or source.suffix.lower() != ".onnx":
        raise RuntimeError("source must be an existing ONNX model")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model = output_dir / "detector.int8.onnx"
    previous_output = (
        {
            "path": final_model.name,
            "bytes": final_model.stat().st_size,
            "sha256": sha256_file(final_model),
        }
        if final_model.exists()
        else None
    )
    if final_model.exists() and not args.force:
        raise FileExistsError(final_model)

    calibration_all = _images(args.calibration_images.resolve())
    evaluation_all = _images(args.evaluation_images.resolve())
    calibration = _evenly_spaced(calibration_all, args.max_calibration_images)
    evaluation = _evenly_spaced(evaluation_all, args.max_evaluation_images)
    support = _class_support(
        evaluation,
        args.evaluation_images.resolve(),
        args.evaluation_labels.resolve(),
    )
    calibration_hashes = _content_hashes(calibration)
    evaluation_hashes = _content_hashes(evaluation)

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "rejected",
        "release_eligible": False,
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "strategy": {
            "format": "QDQ",
            "activation_type": "QUInt8",
            "weight_type": "QInt8",
            "per_channel": True,
            "calibration": "MinMax",
            "operators": ["Conv"],
            "runtime_target": "onnxruntime-web@1.27.0/wasm",
        },
        "data": {
            "calibration_root": str(args.calibration_images.resolve()),
            "calibration_images_available": len(calibration_all),
            "calibration_images": len(calibration),
            "evaluation_root": str(args.evaluation_images.resolve()),
            "evaluation_images_available": len(evaluation_all),
            "evaluation_images": len(evaluation),
            "calibration_evaluation_hash_overlap": len(
                calibration_hashes & evaluation_hashes
            ),
            "class_support": {
                name: support[index] for index, name in enumerate(DETECTOR_CLASSES)
            },
        },
    }

    report_path = output_dir / "detector_quantization_report.json"
    with tempfile.TemporaryDirectory(prefix=".detector-quant-", dir=output_dir) as raw:
        work_dir = Path(raw)
        candidate = work_dir / "detector.qdq.uint8.candidate.onnx"
        try:
            quantize_qdq(source, candidate, calibration)
            report["candidate"] = {
                "bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
            report["graph_contract"] = graph_contract(source, candidate)
            report["comparison"] = compare_models(
                source,
                candidate,
                evaluation,
                args.evaluation_images.resolve(),
                args.evaluation_labels.resolve(),
            )
            report["cross_runtime"] = run_cross_runtime_parity(
                candidate, work_dir, args.runtime_parity_dir
            )
            report["gate"] = gate_report(report)
            if report["gate"]["passed"]:
                staged = output_dir / ".detector.int8.onnx.tmp"
                shutil.copy2(candidate, staged)
                os.replace(staged, final_model)
                report["status"] = "approved"
                report["release_eligible"] = True
                report["output"] = {
                    "path": final_model.name,
                    "bytes": final_model.stat().st_size,
                    "sha256": sha256_file(final_model),
                }
            else:
                report["output"] = None
                if previous_output is not None:
                    report["preserved_previous_output"] = previous_output
        except Exception as exc:
            report["output"] = None
            if previous_output is not None:
                report["preserved_previous_output"] = previous_output
            report["error"] = f"{type(exc).__name__}: {exc}"
            report.setdefault("gate", {"passed": False, "failures": [str(exc)]})
        finally:
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    if args.require_output and report["status"] != "approved":
        failures = report.get("gate", {}).get("failures", [report.get("error")])
        raise RuntimeError("detector quantization rejected: " + "; ".join(failures))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a static-QDQ detector candidate and promote it only after "
            "size, holdout-safety, and Node/WASM parity gates pass"
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calibration-images", type=Path, required=True)
    parser.add_argument("--evaluation-images", type=Path, required=True)
    parser.add_argument("--evaluation-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--runtime-parity-dir",
        type=Path,
        default=Path("training/runtime-parity"),
    )
    parser.add_argument("--max-calibration-images", type=int, default=512)
    parser.add_argument("--max-evaluation-images", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(
        json.dumps(
            quantize_and_validate(parse_args()),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
