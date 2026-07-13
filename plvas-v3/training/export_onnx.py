from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .modeling import (
    choose_subset,
    labels_from_merged_windows,
    load_raw_datasets,
    load_thresholds,
    sequence_metrics,
    tokenize_dataset,
)
from .prepare_data import BASE_MODEL_ID, BASE_MODEL_REVISION
from .schema import LABEL_CONFIG, LABEL_CONFIG_PATH, iter_jsonl, sha256_file


RUNTIME_TOKENIZER_FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)

REQUIRED_CROSS_RUNTIME_BACKENDS = ("node", "wasm", "webgpu")
RELEASE_QUANTIZATION_POLICY = {
    "minimum_argmax_agreement": 0.99,
    "maximum_probability_error": 0.03,
    "maximum_recall_drop": 0.01,
}


def add_onnx_metadata(path: Path, metadata: dict[str, Any]) -> None:
    import onnx

    model = onnx.load(str(path))
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = (
            json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
        )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def export_fp32(model_dir: Path, output_path: Path, max_length: int) -> list[str]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    sample = tokenizer(
        "Full legal name: Avery Chen",
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_names = [
        name
        for name in ("input_ids", "attention_mask", "token_type_ids")
        if name in sample
    ]

    class Wrapper(torch.nn.Module):
        def __init__(self, wrapped, names):
            super().__init__()
            self.wrapped = wrapped
            self.names = names

        def forward(self, *values):
            kwargs = dict(zip(self.names, values))
            return self.wrapped(**kwargs).logits

    wrapper = Wrapper(model, input_names)
    inputs = tuple(sample[name] for name in input_names)
    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names} | {
        "logits": {0: "batch", 1: "sequence"}
    }
    kwargs = {
        "input_names": input_names,
        "output_names": ["logits"],
        "dynamic_axes": dynamic_axes,
        "opset_version": 17,
        "do_constant_folding": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(wrapper, inputs, str(output_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(wrapper, inputs, str(output_path), **kwargs)
    return input_names


def quantize(fp32_path: Path, int8_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )


def session_contract(session) -> dict[str, Any]:
    def describe(value) -> dict[str, Any]:
        return {
            "name": value.name,
            "type": value.type,
            "shape": [
                item if isinstance(item, int) else str(item) for item in value.shape
            ],
        }

    return {
        "inputs": [describe(value) for value in session.get_inputs()],
        "outputs": [describe(value) for value in session.get_outputs()],
        "providers": session.get_providers(),
    }


def session_feeds(session, encoded: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        item.name: np.asarray(encoded[item.name], dtype=np.int64)
        for item in session.get_inputs()
    }


def compare_models(
    fp32_path: Path,
    int8_path: Path,
    model_dir: Path,
    tokenizer,
    records: list[dict[str, Any]],
    *,
    max_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForTokenClassification

    fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    pytorch = AutoModelForTokenClassification.from_pretrained(model_dir)
    pytorch.eval()
    fp32_int8_agreements: list[float] = []
    pytorch_fp32_agreements: list[float] = []
    pytorch_int8_agreements: list[float] = []
    fp32_int8_probability_errors: list[float] = []
    pytorch_fp32_probability_errors: list[float] = []
    pytorch_int8_probability_errors: list[float] = []
    golden_vectors: list[dict[str, Any]] = []
    from .modeling import softmax

    for record in records:
        encoded = tokenizer(
            record["text"],
            return_tensors="np",
            truncation=True,
            max_length=max_length,
        )
        torch_feeds = {
            name: torch.from_numpy(np.asarray(encoded[name], dtype=np.int64))
            for name in ("input_ids", "attention_mask", "token_type_ids")
            if name in encoded
        }
        with torch.inference_mode():
            pytorch_logits = pytorch(**torch_feeds).logits.detach().cpu().numpy()
        float_logits = fp32.run(None, session_feeds(fp32, encoded))[0]
        int8_logits = int8.run(None, session_feeds(int8, encoded))[0]
        pytorch_probabilities = softmax(pytorch_logits)
        float_probabilities = softmax(float_logits)
        int8_probabilities = softmax(int8_logits)
        valid_length = int(np.asarray(encoded["attention_mask"]).sum())
        pytorch_ids = np.argmax(pytorch_probabilities[0, :valid_length], axis=-1)
        float_ids = np.argmax(float_probabilities[0, :valid_length], axis=-1)
        int8_ids = np.argmax(int8_probabilities[0, :valid_length], axis=-1)
        fp32_int8_agreements.append(float(np.mean(float_ids == int8_ids)))
        pytorch_fp32_agreements.append(float(np.mean(pytorch_ids == float_ids)))
        pytorch_int8_agreements.append(float(np.mean(pytorch_ids == int8_ids)))
        fp32_int8_probability_errors.append(
            float(
                np.mean(
                    np.abs(
                        float_probabilities[0, :valid_length]
                        - int8_probabilities[0, :valid_length]
                    )
                )
            )
        )
        pytorch_fp32_probability_errors.append(
            float(
                np.mean(
                    np.abs(
                        pytorch_probabilities[0, :valid_length]
                        - float_probabilities[0, :valid_length]
                    )
                )
            )
        )
        pytorch_int8_probability_errors.append(
            float(
                np.mean(
                    np.abs(
                        pytorch_probabilities[0, :valid_length]
                        - int8_probabilities[0, :valid_length]
                    )
                )
            )
        )
        if len(golden_vectors) < 64:
            golden_vectors.append(
                {
                    "id": record["id"],
                    "input_ids": np.asarray(encoded["input_ids"])[0].tolist(),
                    "attention_mask": np.asarray(encoded["attention_mask"])[0].tolist(),
                    "token_type_ids": (
                        np.asarray(encoded["token_type_ids"])[0].tolist()
                        if "token_type_ids" in encoded
                        else None
                    ),
                    "predicted_label_ids": int8_ids.tolist(),
                    "max_probabilities": [
                        round(float(value), 6)
                        for value in np.max(
                            int8_probabilities[0, :valid_length], axis=-1
                        )
                    ],
                }
            )
    return (
        {
            "records": len(records),
            "mean_argmax_agreement": statistics.fmean(fp32_int8_agreements),
            "minimum_record_argmax_agreement": min(fp32_int8_agreements),
            "mean_probability_absolute_error": statistics.fmean(
                fp32_int8_probability_errors
            ),
            "pytorch_fp32": {
                "mean_argmax_agreement": statistics.fmean(pytorch_fp32_agreements),
                "minimum_record_argmax_agreement": min(pytorch_fp32_agreements),
                "mean_probability_absolute_error": statistics.fmean(
                    pytorch_fp32_probability_errors
                ),
            },
            "pytorch_int8": {
                "mean_argmax_agreement": statistics.fmean(pytorch_int8_agreements),
                "minimum_record_argmax_agreement": min(pytorch_int8_agreements),
                "mean_probability_absolute_error": statistics.fmean(
                    pytorch_int8_probability_errors
                ),
            },
            "fp32_int8": {
                "mean_argmax_agreement": statistics.fmean(fp32_int8_agreements),
                "minimum_record_argmax_agreement": min(fp32_int8_agreements),
                "mean_probability_absolute_error": statistics.fmean(
                    fp32_int8_probability_errors
                ),
            },
            "fp32_contract": session_contract(fp32),
            "int8_contract": session_contract(int8),
        },
        golden_vectors,
    )


def evaluate_onnx(
    model_path: Path,
    tokenizer,
    raw_dataset,
    thresholds: dict[str, float],
    *,
    max_records: int,
    max_length: int,
    stride: int,
    batch_size: int,
    num_proc: int,
) -> tuple[dict[str, float], dict[str, float]]:
    import onnxruntime as ort

    selected = choose_subset(raw_dataset, max_records)
    tokenized = tokenize_dataset(
        selected,
        tokenizer,
        max_length=max_length,
        stride=stride,
        num_proc=num_proc,
        desc=f"Tokenizing ONNX evaluation for {model_path.name}",
        include_metadata=True,
    )
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    window_logits: list[np.ndarray] = []
    window_labels: list[list[int]] = []
    record_indices: list[int] = []
    source_offsets: list[list[list[int]]] = []
    source_truth_spans: list[list[list[int]]] = []
    latencies: list[float] = []
    for start_index in range(0, len(tokenized), batch_size):
        features = [
            dict(tokenized[index])
            for index in range(
                start_index,
                min(len(tokenized), start_index + batch_size),
            )
        ]
        label_rows = [feature.pop("labels") for feature in features]
        batch_record_indices = [
            int(feature.pop("source_record_index")) for feature in features
        ]
        batch_offsets = [feature.pop("source_offsets") for feature in features]
        batch_truth_spans = [feature.pop("source_truth_spans") for feature in features]
        encoded = tokenizer.pad(features, padding=True, return_tensors="np")
        started = time.perf_counter()
        logits = session.run(None, session_feeds(session, encoded))[0]
        latencies.append((time.perf_counter() - started) * 1000)
        for row_index, label_row in enumerate(label_rows):
            row_length = len(label_row)
            window_logits.append(np.asarray(logits[row_index, :row_length]))
            window_labels.append([int(value) for value in label_row])
        record_indices.extend(batch_record_indices)
        source_offsets.extend(batch_offsets)
        source_truth_spans.extend(batch_truth_spans)

    maximum_tokens = max(len(row) for row in window_labels)
    label_count = int(window_logits[0].shape[-1])
    merged_logits = np.zeros(
        (len(window_logits), maximum_tokens, label_count), dtype=np.float32
    )
    merged_labels = np.full((len(window_labels), maximum_tokens), -100, dtype=np.int64)
    for row_index, (logit_row, label_row) in enumerate(
        zip(window_logits, window_labels)
    ):
        merged_logits[row_index, : len(logit_row)] = logit_row
        merged_labels[row_index, : len(label_row)] = label_row
    truth, predicted = labels_from_merged_windows(
        merged_logits,
        merged_labels,
        record_indices,
        source_offsets,
        source_truth_spans,
        thresholds=thresholds,
    )
    metrics = sequence_metrics(truth, predicted)
    performance = {
        "records": len(selected),
        "windows": len(tokenized),
        "batches": len(latencies),
        "batch_size": batch_size,
        "p50_batch_ms": float(np.percentile(latencies, 50)),
        "p95_batch_ms": float(np.percentile(latencies, 95)),
    }
    return metrics, performance


def artifact_entry(path: Path, root: Path, *, runtime: bool) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "runtime": runtime,
    }


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cross_runtime_status(
    path: Path | None,
    *,
    model_sha256: str,
    golden_vectors_sha256: str,
    vector_count: int,
) -> dict[str, Any]:
    report = load_optional_json(path)
    backends: dict[str, Any] = {}
    report_hash_matches = bool(report and report.get("model_sha256") == model_sha256)
    golden_hash_matches = bool(
        report and report.get("golden_vectors_sha256") == golden_vectors_sha256
    )
    supplied = report.get("backends", {}) if report else {}
    expected_packages = {
        "node": ("onnxruntime-node", "cpu"),
        "wasm": ("onnxruntime-web", "wasm"),
        "webgpu": ("onnxruntime-web", "webgpu"),
    }
    for backend in REQUIRED_CROSS_RUNTIME_BACKENDS:
        item = supplied.get(backend, {})
        expected_package, expected_provider = expected_packages[backend]
        errors = []
        if item:
            if item.get("passed") is not True:
                errors.append("backend did not pass")
            if item.get("package") != expected_package:
                errors.append("runtime package mismatch")
            if item.get("package_version") != "1.27.0":
                errors.append("runtime package version mismatch")
            if item.get("vectors_checked") != vector_count:
                errors.append("golden vector count mismatch")
            if item.get("failed_vector_ids"):
                errors.append("one or more golden vectors failed")
            if expected_provider not in item.get("execution_providers_requested", []):
                errors.append("requested execution provider is missing")
            if backend == "webgpu":
                provider_trace = item.get("provider_trace", {})
                if provider_trace.get("verified") is not True:
                    errors.append("WebGPU provider trace is not verified")
                fallback_nodes = provider_trace.get("fallback_nodes")
                if not isinstance(fallback_nodes, int) or fallback_nodes < 0:
                    errors.append("WebGPU fallback node count is missing")
                elif (
                    fallback_nodes > 0
                    and provider_trace.get("fallback_measured") is not True
                ):
                    errors.append("WebGPU fallback is not measured")
        passed = bool(
            report_hash_matches and golden_hash_matches and item and not errors
        )
        backends[backend] = {
            "status": "passed" if passed else "failed" if item else "pending",
            "passed": passed,
            "details": item if item else None,
            "validation_errors": errors,
        }
    return {
        "report": str(path) if path else None,
        "report_sha256": sha256_file(path) if path and path.exists() else None,
        "model_hash_matches": report_hash_matches,
        "golden_vectors_hash_matches": golden_hash_matches,
        "expected_vector_count": vector_count,
        "required_backends": list(REQUIRED_CROSS_RUNTIME_BACKENDS),
        "backends": backends,
        "passed": all(item["passed"] for item in backends.values()),
    }


def release_checks(
    *,
    training_run: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
    quantization: dict[str, Any],
    cross_runtime: dict[str, Any],
    artifact_license: str,
) -> dict[str, Any]:
    baseline_gate = (training_run or {}).get("baseline_gate") or {}
    checks = {
        "frozen_rampart_baseline": {
            "passed": baseline_gate.get("baseline_frozen") is True
            and baseline_gate.get("replacement_release_allowed") is True,
            "actual": baseline_gate.get("status", "missing"),
            "required": "frozen",
        },
        "evaluation_gates": {
            "passed": (evaluation or {}).get("gates", {}).get("passed") is True,
            "actual": (evaluation or {}).get("gates", {}).get("passed"),
            "required": True,
        },
        "quantization_parity": {
            "passed": (
                quantization["mean_argmax_agreement"]
                >= RELEASE_QUANTIZATION_POLICY["minimum_argmax_agreement"]
                and quantization["mean_probability_absolute_error"]
                <= RELEASE_QUANTIZATION_POLICY["maximum_probability_error"]
                and quantization["minimum_holdout_recall_delta"]
                >= -RELEASE_QUANTIZATION_POLICY["maximum_recall_drop"]
            ),
            "actual": {
                "mean_argmax_agreement": quantization["mean_argmax_agreement"],
                "mean_probability_absolute_error": quantization[
                    "mean_probability_absolute_error"
                ],
                "minimum_holdout_recall_delta": quantization[
                    "minimum_holdout_recall_delta"
                ],
            },
            "required": RELEASE_QUANTIZATION_POLICY,
        },
        "cross_runtime_parity": {
            "passed": cross_runtime["passed"],
            "actual": {
                name: item["status"] for name, item in cross_runtime["backends"].items()
            },
            "required": list(REQUIRED_CROSS_RUNTIME_BACKENDS),
        },
        "artifact_license": {
            "passed": bool(artifact_license)
            and not artifact_license.lower().startswith("pending"),
            "actual": artifact_license,
            "required": "maintainer-approved SPDX expression",
        },
    }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
    }


def build_model_card(
    manifest: dict[str, Any],
    evaluation: dict[str, Any] | None,
) -> str:
    holdout = (
        evaluation.get("slices", {}).get("holdout/all", {}).get("metrics", {})
        if evaluation
        else {}
    )
    lines = [
        "---",
        "language:",
        "  - en",
        "library_name: transformers",
        "pipeline_tag: token-classification",
        f"base_model: {manifest['model']['base_model']}",
        "license: other",
        "---",
        "",
        "# PLVA compact PII tagger candidate",
        "",
        "This optional replacement model classifies OCR text spans for PLVA's on-device screenshot redaction pipeline.",
        "",
        "## Artifact",
        "",
        f"- Runtime bytes: {manifest['runtime_bytes']:,}",
        f"- INT8 SHA-256: `{manifest['artifacts']['model.int8.onnx']['sha256']}`",
        f"- Argmax agreement with FP32: {manifest['quantization']['mean_argmax_agreement']:.4f}",
        f"- Release eligible: {str(manifest['release_eligible']).lower()}",
        "",
        "## Holdout evaluation",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Precision | {holdout.get('precision', 0):.4f} |",
        f"| Recall | {holdout.get('recall', 0):.4f} |",
        f"| F1 | {holdout.get('f1', 0):.4f} |",
        f"| Safety score | {holdout.get('safety_score', 0):.4f} |",
        "",
        "## Data and attribution",
        "",
        "- AI4Privacy OpenPII 1.5M, pinned revision, English subset, CC BY 4.0.",
        "- Deterministic synthetic screen text and span-preserving OCR corruption.",
        "- Google compact BERT base, Apache-2.0.",
        "- Rampart is the pinned semantic bootstrap baseline and replacement gate.",
        "",
        "All generated PII is fake. The model does not detect non-text PII and remains dependent on OCR coverage.",
        "",
        f"Distribution license: {manifest['licenses']['artifact']}.",
        "",
    ]
    return "\n".join(lines)


def export(args: argparse.Namespace) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort
    from transformers import AutoTokenizer

    training_run = load_optional_json(args.training_run)
    evaluation = load_optional_json(args.evaluation)
    base_model = (training_run or {}).get("base_model", BASE_MODEL_ID)
    base_revision = (training_run or {}).get("base_revision", BASE_MODEL_REVISION)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = args.output_dir / "reference"
    reference_dir.mkdir(exist_ok=True)
    fp32_path = reference_dir / "model.fp32.onnx"
    int8_path = args.output_dir / "model.int8.onnx"

    input_names = export_fp32(args.model_dir, fp32_path, args.max_length)
    common_metadata = {
        "plva.schema_version": "1",
        "plva.base_model": base_model,
        "plva.base_revision": base_revision,
        "plva.labels": list(LABEL_CONFIG.labels),
        "plva.input_names": input_names,
    }
    add_onnx_metadata(fp32_path, common_metadata | {"plva.quantization": "fp32"})
    quantize(fp32_path, int8_path)
    add_onnx_metadata(
        int8_path,
        common_metadata
        | {
            "plva.quantization": "dynamic-int8-per-channel",
            "plva.runtime": "onnxruntime",
        },
    )
    onnx.checker.check_model(onnx.load(str(int8_path)))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tokenizer.save_pretrained(args.output_dir)
    for filename in RUNTIME_TOKENIZER_FILES:
        source = args.model_dir / filename
        destination = args.output_dir / filename
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
    shutil.copy2(LABEL_CONFIG_PATH, args.output_dir / "labels.json")
    shutil.copy2(args.thresholds, args.output_dir / "thresholds.json")

    holdout_records = list(iter_jsonl(args.data_dir / "holdout.jsonl"))
    agreement_records = holdout_records[: args.agreement_records]
    quantization, golden_vectors = compare_models(
        fp32_path,
        int8_path,
        args.model_dir,
        tokenizer,
        agreement_records,
        max_length=args.max_length,
    )
    (args.output_dir / "golden_vectors.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_sha256": sha256_file(int8_path),
                "vectors": golden_vectors,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    thresholds = load_thresholds(args.thresholds)
    raw = load_raw_datasets(args.data_dir)
    fp32_metrics, fp32_performance = evaluate_onnx(
        fp32_path,
        tokenizer,
        raw["holdout"],
        thresholds,
        max_records=args.onnx_eval_records,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.onnx_batch_size,
        num_proc=args.num_proc,
    )
    int8_metrics, int8_performance = evaluate_onnx(
        int8_path,
        tokenizer,
        raw["holdout"],
        thresholds,
        max_records=args.onnx_eval_records,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.onnx_batch_size,
        num_proc=args.num_proc,
    )
    quantization["holdout_fp32"] = fp32_metrics
    quantization["holdout_int8"] = int8_metrics
    quantization["holdout_f1_delta"] = int8_metrics["f1"] - fp32_metrics["f1"]
    quantization["holdout_recall_delta"] = (
        int8_metrics["recall"] - fp32_metrics["recall"]
    )
    quantization["fp32_performance"] = fp32_performance
    quantization["int8_performance"] = int8_performance
    recall_deltas = [quantization["holdout_recall_delta"]]
    if "ocr_holdout" in raw:
        ocr_fp32_metrics, ocr_fp32_performance = evaluate_onnx(
            fp32_path,
            tokenizer,
            raw["ocr_holdout"],
            thresholds,
            max_records=args.onnx_eval_records,
            max_length=args.max_length,
            stride=args.stride,
            batch_size=args.onnx_batch_size,
            num_proc=args.num_proc,
        )
        ocr_int8_metrics, ocr_int8_performance = evaluate_onnx(
            int8_path,
            tokenizer,
            raw["ocr_holdout"],
            thresholds,
            max_records=args.onnx_eval_records,
            max_length=args.max_length,
            stride=args.stride,
            batch_size=args.onnx_batch_size,
            num_proc=args.num_proc,
        )
        quantization["ocr_holdout_fp32"] = ocr_fp32_metrics
        quantization["ocr_holdout_int8"] = ocr_int8_metrics
        quantization["ocr_holdout_recall_delta"] = (
            ocr_int8_metrics["recall"] - ocr_fp32_metrics["recall"]
        )
        quantization["ocr_holdout_fp32_performance"] = ocr_fp32_performance
        quantization["ocr_holdout_int8_performance"] = ocr_int8_performance
        recall_deltas.append(quantization["ocr_holdout_recall_delta"])
    quantization["minimum_holdout_recall_delta"] = min(recall_deltas)

    if quantization["mean_argmax_agreement"] < args.minimum_argmax_agreement:
        raise RuntimeError("INT8 argmax agreement is below the required threshold")
    if quantization["mean_probability_absolute_error"] > args.maximum_probability_error:
        raise RuntimeError("INT8 probability error is above the required threshold")
    if quantization["minimum_holdout_recall_delta"] < -args.maximum_metric_drop:
        raise RuntimeError("INT8 recall degradation exceeds the required threshold")

    provenance_dir = args.output_dir / "provenance"
    provenance_dir.mkdir(exist_ok=True)
    for source in (
        args.data_dir / "data_manifest.json",
        args.training_run,
        args.evaluation,
    ):
        if source and source.exists():
            shutil.copy2(source, provenance_dir / source.name)
    artifact_paths = [
        int8_path,
        fp32_path,
        args.output_dir / "labels.json",
        args.output_dir / "thresholds.json",
        args.output_dir / "golden_vectors.json",
    ]
    artifact_paths.extend(
        args.output_dir / filename
        for filename in RUNTIME_TOKENIZER_FILES
        if (args.output_dir / filename).exists()
    )
    artifacts = {
        str(path.relative_to(args.output_dir)): artifact_entry(
            path,
            args.output_dir,
            runtime=path == int8_path
            or path.parent == args.output_dir
            and path.name
            in RUNTIME_TOKENIZER_FILES + ("labels.json", "thresholds.json"),
        )
        for path in artifact_paths
    }
    runtime_bytes = sum(item["bytes"] for item in artifacts.values() if item["runtime"])
    if runtime_bytes > args.runtime_budget_bytes:
        raise RuntimeError(
            f"runtime artifacts are {runtime_bytes:,} bytes, over {args.runtime_budget_bytes:,}"
        )

    int8_session = ort.InferenceSession(
        str(int8_path), providers=["CPUExecutionProvider"]
    )
    cross_runtime = cross_runtime_status(
        args.cross_runtime_report,
        model_sha256=sha256_file(int8_path),
        golden_vectors_sha256=sha256_file(args.output_dir / "golden_vectors.json"),
        vector_count=len(golden_vectors),
    )
    release = release_checks(
        training_run=training_run,
        evaluation=evaluation,
        quantization=quantization,
        cross_runtime=cross_runtime,
        artifact_license=args.artifact_license,
    )
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": "plva-compact-pii-tagger",
            "base_model": base_model,
            "base_revision": base_revision,
            "labels": list(LABEL_CONFIG.labels),
            "label_count": len(LABEL_CONFIG.labels),
            "max_length": args.max_length,
        },
        "runtime_bytes": runtime_bytes,
        "runtime_budget_bytes": args.runtime_budget_bytes,
        "artifacts": artifacts,
        "tensor_contract": session_contract(int8_session),
        "quantization": quantization,
        "evaluation_gates": evaluation.get("gates") if evaluation else None,
        "baseline_gate": (training_run or {}).get("baseline_gate"),
        "cross_runtime": cross_runtime,
        "release_eligible": release["passed"],
        "release_checks": release,
        "licenses": {
            "base_model": "Apache-2.0",
            "openpii": "CC-BY-4.0",
            "artifact": args.artifact_license,
        },
    }
    # Add convenient basename keys for the model card and consumers.
    manifest["artifacts"]["model.int8.onnx"] = artifacts["model.int8.onnx"]
    (args.output_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "MODEL_CARD.md").write_text(
        build_model_card(manifest, evaluation),
        encoding="utf-8",
    )
    if not manifest["release_eligible"] and not args.allow_development_artifact:
        failed = [
            name for name, item in release["checks"].items() if not item["passed"]
        ]
        raise RuntimeError("candidate is not release eligible: " + ", ".join(failed))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and quantize the PLVA tagger")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, default=None)
    parser.add_argument("--evaluation", type=Path, default=None)
    parser.add_argument("--cross-runtime-report", type=Path, default=None)
    parser.add_argument("--artifact-license", default="pending-maintainer-review")
    parser.add_argument(
        "--allow-development-artifact",
        action="store_true",
        help="write a non-publishable smoke artifact when release gates are incomplete",
    )
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--agreement-records", type=int, default=512)
    parser.add_argument("--onnx-eval-records", type=int, default=10_000)
    parser.add_argument("--onnx-batch-size", type=int, default=256)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--runtime-budget-bytes", type=int, default=20_000_000)
    parser.add_argument("--minimum-argmax-agreement", type=float, default=0.99)
    parser.add_argument("--maximum-probability-error", type=float, default=0.03)
    parser.add_argument("--maximum-metric-drop", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(export(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
