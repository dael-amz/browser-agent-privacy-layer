from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .schema import LABEL_CONFIG


def load_raw_datasets(data_dir: Path):
    from datasets import DatasetDict, load_dataset

    files = {
        split: str(data_dir / f"{split}.jsonl")
        for split in ("train", "validation", "holdout", "ocr_holdout")
        if (data_dir / f"{split}.jsonl").exists()
    }
    if not files:
        raise FileNotFoundError(f"no JSONL data files found under {data_dir}")
    # Load each split independently. Provenance is intentionally source-specific,
    # so forcing one inferred Arrow struct across synthetic, OpenPII, and OCR
    # holdout records would discard fields or fail schema casting.
    return DatasetDict(
        {
            split: load_dataset(
                "json",
                data_files={split: filename},
                split=split,
            )
            for split, filename in files.items()
        }
    )


def align_offsets(
    offsets: list[list[int]] | list[tuple[int, int]],
    spans: list[dict[str, Any]],
    label_to_id: dict[str, int],
) -> list[int]:
    ordered = sorted(spans, key=lambda span: (int(span["start"]), int(span["end"])))
    labels: list[int] = []
    previous_span_index: int | None = None
    for raw_start, raw_end in offsets:
        start, end = int(raw_start), int(raw_end)
        if start == end:
            labels.append(-100)
            previous_span_index = None
            continue

        best_index: int | None = None
        best_overlap = 0
        for span_index, span in enumerate(ordered):
            span_start, span_end = int(span["start"]), int(span["end"])
            overlap = max(0, min(end, span_end) - max(start, span_start))
            if overlap > best_overlap:
                best_index = span_index
                best_overlap = overlap
        if best_index is None:
            labels.append(label_to_id["O"])
            previous_span_index = None
            continue

        span = ordered[best_index]
        prefix = "I" if previous_span_index == best_index else "B"
        labels.append(label_to_id[f"{prefix}-{span['label']}"])
        previous_span_index = best_index
    return labels


def truth_span_offsets(
    offsets: list[list[int]] | list[tuple[int, int]],
    spans: list[dict[str, Any]],
) -> list[list[int]]:
    ordered = sorted(spans, key=lambda span: (int(span["start"]), int(span["end"])))
    result: list[list[int]] = []
    for raw_start, raw_end in offsets:
        start, end = int(raw_start), int(raw_end)
        best: dict[str, Any] | None = None
        best_overlap = 0
        for span in ordered:
            overlap = max(
                0,
                min(end, int(span["end"])) - max(start, int(span["start"])),
            )
            if overlap > best_overlap:
                best = span
                best_overlap = overlap
        result.append(
            [int(best["start"]), int(best["end"])] if best is not None else [-1, -1]
        )
    return result


def tokenize_dataset(
    dataset,
    tokenizer,
    *,
    max_length: int,
    stride: int,
    num_proc: int = 1,
    desc: str = "Tokenizing",
    include_metadata: bool = False,
):
    label_to_id = LABEL_CONFIG.label_to_id
    source_columns = dataset.column_names

    def tokenize_batch(
        batch: dict[str, list[Any]], indices: list[int] | None = None
    ) -> dict[str, list[Any]]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            padding=False,
        )
        sample_mapping = encoded.pop("overflow_to_sample_mapping")
        all_offsets = encoded.pop("offset_mapping")
        encoded["labels"] = [
            align_offsets(offsets, batch["spans"][int(sample_index)], label_to_id)
            for offsets, sample_index in zip(all_offsets, sample_mapping)
        ]
        if include_metadata:
            if indices is None:
                raise RuntimeError("evaluation metadata requires source record indices")
            encoded["source_record_index"] = [
                int(indices[int(sample_index)]) for sample_index in sample_mapping
            ]
            encoded["source_offsets"] = [
                [[int(start), int(end)] for start, end in offsets]
                for offsets in all_offsets
            ]
            encoded["source_truth_spans"] = [
                truth_span_offsets(offsets, batch["spans"][int(sample_index)])
                for offsets, sample_index in zip(all_offsets, sample_mapping)
            ]
        return dict(encoded)

    return dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=512,
        remove_columns=source_columns,
        num_proc=num_proc,
        desc=desc,
        with_indices=include_metadata,
    )


def labels_from_predictions(
    predictions: np.ndarray,
    label_ids: np.ndarray,
    *,
    thresholds: dict[str, float] | None = None,
) -> tuple[list[list[str]], list[list[str]]]:
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    probabilities = softmax(predictions, axis=-1)
    predicted_ids = np.argmax(probabilities, axis=-1)
    id_to_label = dict(enumerate(LABEL_CONFIG.labels))
    true_sequences: list[list[str]] = []
    predicted_sequences: list[list[str]] = []

    for row_predictions, row_probabilities, row_labels in zip(
        predicted_ids, probabilities, label_ids
    ):
        truth: list[str] = []
        predicted: list[str] = []
        prior_class: str | None = None
        for predicted_id, token_probabilities, true_id in zip(
            row_predictions, row_probabilities, row_labels
        ):
            if int(true_id) == -100:
                continue
            truth_label = id_to_label[int(true_id)]
            predicted_label = id_to_label[int(predicted_id)]
            if predicted_label != "O" and thresholds:
                entity_class = predicted_label[2:]
                class_probability = max(
                    float(
                        token_probabilities[
                            LABEL_CONFIG.label_to_id[f"B-{entity_class}"]
                        ]
                    ),
                    float(
                        token_probabilities[
                            LABEL_CONFIG.label_to_id[f"I-{entity_class}"]
                        ]
                    ),
                )
                if class_probability < thresholds.get(entity_class, 0.5):
                    predicted_label = "O"
            if predicted_label.startswith("I-"):
                entity_class = predicted_label[2:]
                if prior_class != entity_class:
                    predicted_label = f"B-{entity_class}"
            prior_class = predicted_label[2:] if predicted_label != "O" else None
            truth.append(truth_label)
            predicted.append(predicted_label)
        true_sequences.append(truth)
        predicted_sequences.append(predicted)
    return true_sequences, predicted_sequences


def labels_from_merged_windows(
    predictions: np.ndarray,
    label_ids: np.ndarray,
    record_indices: list[int],
    offsets: list[list[list[int]]],
    truth_spans: list[list[list[int]]],
    *,
    thresholds: dict[str, float] | None = None,
) -> tuple[list[list[str]], list[list[str]]]:
    """Merge overlapping-window token predictions into one sequence per record.

    Each unique source token offset is scored once. For recall-oriented runtime
    parity, any entity class that clears its frozen threshold in any overlapping
    window remains eligible; the highest such class score wins that token.
    """

    if isinstance(predictions, tuple):
        predictions = predictions[0]
    probabilities = softmax(predictions, axis=-1)
    id_to_label = dict(enumerate(LABEL_CONFIG.labels))
    label_to_id = LABEL_CONFIG.label_to_id
    records: dict[int, dict[tuple[int, int], dict[str, Any]]] = {}

    for (
        row_probabilities,
        row_labels,
        record_index,
        row_offsets,
        row_truth_spans,
    ) in zip(probabilities, label_ids, record_indices, offsets, truth_spans):
        tokens = records.setdefault(int(record_index), {})
        for token_probabilities, true_id, raw_offset, raw_truth_span in zip(
            row_probabilities, row_labels, row_offsets, row_truth_spans
        ):
            start, end = int(raw_offset[0]), int(raw_offset[1])
            if int(true_id) == -100 or start == end:
                continue
            key = (start, end)
            truth_label = id_to_label[int(true_id)]
            truth_class = truth_label[2:] if truth_label != "O" else "O"
            truth_span = (
                (truth_class, int(raw_truth_span[0]), int(raw_truth_span[1]))
                if truth_class != "O"
                else None
            )
            entry = tokens.setdefault(
                key,
                {
                    "truth_class": truth_class,
                    "truth_span": truth_span,
                    "entity_scores": {
                        entity_class: 0.0
                        for entity_class in LABEL_CONFIG.entity_classes
                    },
                    "entity_prefixes": {
                        entity_class: "B"
                        for entity_class in LABEL_CONFIG.entity_classes
                    },
                },
            )
            if entry["truth_class"] == "O" and truth_class != "O":
                entry["truth_class"] = truth_class
                entry["truth_span"] = truth_span
            elif truth_class not in {"O", entry["truth_class"]}:
                raise ValueError(
                    f"conflicting truth classes at record {record_index} offset {key}"
                )
            for entity_class in LABEL_CONFIG.entity_classes:
                begin_score = float(
                    token_probabilities[label_to_id[f"B-{entity_class}"]]
                )
                inside_score = float(
                    token_probabilities[label_to_id[f"I-{entity_class}"]]
                )
                score, prefix = max((begin_score, "B"), (inside_score, "I"))
                if score > entry["entity_scores"][entity_class]:
                    entry["entity_scores"][entity_class] = score
                    entry["entity_prefixes"][entity_class] = prefix

    true_sequences: list[list[str]] = []
    predicted_sequences: list[list[str]] = []
    frozen_thresholds = thresholds or {
        entity_class: 0.5 for entity_class in LABEL_CONFIG.entity_classes
    }
    for record_index in sorted(records):
        truth: list[str] = []
        predicted: list[str] = []
        prior_truth_span: tuple[str, int, int] | None = None
        prior_predicted: str | None = None
        for _offset, item in sorted(records[record_index].items()):
            truth_class = item["truth_class"]
            if truth_class == "O":
                truth_label = "O"
                prior_truth_span = None
            else:
                current_truth_span = item["truth_span"]
                truth_label = f"{'I' if prior_truth_span == current_truth_span else 'B'}-{truth_class}"
                prior_truth_span = current_truth_span

            passing = [
                (score, entity_class, item["entity_prefixes"][entity_class])
                for entity_class, score in item["entity_scores"].items()
                if score >= frozen_thresholds.get(entity_class, 0.5)
            ]
            if passing:
                _score, predicted_class, predicted_prefix = max(
                    passing,
                    key=lambda value: (
                        value[0],
                        -LABEL_CONFIG.entity_classes.index(value[1]),
                    ),
                )
                predicted_label = (
                    f"{'I' if prior_predicted == predicted_class and predicted_prefix == 'I' else 'B'}-"
                    f"{predicted_class}"
                )
                prior_predicted = predicted_class
            else:
                predicted_label = "O"
                prior_predicted = None
            truth.append(truth_label)
            predicted.append(predicted_label)
        true_sequences.append(truth)
        predicted_sequences.append(predicted)
    return true_sequences, predicted_sequences


def merged_class_probabilities(
    predictions: np.ndarray,
    label_ids: np.ndarray,
    record_indices: list[int],
    offsets: list[list[list[int]]],
) -> tuple[np.ndarray, list[str]]:
    """Deduplicate overlapping windows for per-token threshold calibration."""

    if isinstance(predictions, tuple):
        predictions = predictions[0]
    probabilities = softmax(predictions, axis=-1)
    id_to_label = dict(enumerate(LABEL_CONFIG.labels))
    label_to_id = LABEL_CONFIG.label_to_id
    records: dict[int, dict[tuple[int, int], dict[str, Any]]] = {}
    for row_probabilities, row_labels, record_index, row_offsets in zip(
        probabilities, label_ids, record_indices, offsets
    ):
        tokens = records.setdefault(int(record_index), {})
        for token_probabilities, true_id, raw_offset in zip(
            row_probabilities, row_labels, row_offsets
        ):
            start, end = int(raw_offset[0]), int(raw_offset[1])
            if int(true_id) == -100 or start == end:
                continue
            truth_label = id_to_label[int(true_id)]
            truth_class = truth_label[2:] if truth_label != "O" else "O"
            item = tokens.setdefault(
                (start, end),
                {
                    "truth_class": truth_class,
                    "scores": np.zeros(
                        len(LABEL_CONFIG.entity_classes), dtype=np.float32
                    ),
                },
            )
            if item["truth_class"] == "O" and truth_class != "O":
                item["truth_class"] = truth_class
            elif truth_class not in {"O", item["truth_class"]}:
                raise ValueError(
                    f"conflicting truth classes at record {record_index} offset {(start, end)}"
                )
            for class_index, entity_class in enumerate(LABEL_CONFIG.entity_classes):
                score = max(
                    float(token_probabilities[label_to_id[f"B-{entity_class}"]]),
                    float(token_probabilities[label_to_id[f"I-{entity_class}"]]),
                )
                item["scores"][class_index] = max(item["scores"][class_index], score)
    ordered = [
        item
        for record_index in sorted(records)
        for _offset, item in sorted(records[record_index].items())
    ]
    if not ordered:
        return np.empty((0, len(LABEL_CONFIG.entity_classes)), dtype=np.float32), []
    return np.stack([item["scores"] for item in ordered]), [
        item["truth_class"] for item in ordered
    ]


def entity_confusion_matrix(
    true_sequences: list[list[str]], predicted_sequences: list[list[str]]
) -> dict[str, dict[str, int]]:
    from seqeval.metrics.sequence_labeling import get_entities

    columns = [*LABEL_CONFIG.entity_classes, "MISSED"]
    matrix: dict[str, dict[str, int]] = {
        entity_class: {column: 0 for column in columns}
        for entity_class in LABEL_CONFIG.entity_classes
    }
    matrix["SPURIOUS"] = {column: 0 for column in LABEL_CONFIG.entity_classes}

    for truth, predicted in zip(true_sequences, predicted_sequences):
        truth_entities = list(get_entities(truth))
        predicted_entities = list(get_entities(predicted))
        unused = set(range(len(predicted_entities)))
        for truth_class, truth_start, truth_end in truth_entities:
            candidates: list[tuple[float, int]] = []
            for index in unused:
                _predicted_class, predicted_start, predicted_end = predicted_entities[
                    index
                ]
                intersection = max(
                    0,
                    min(truth_end, predicted_end)
                    - max(truth_start, predicted_start)
                    + 1,
                )
                if intersection == 0:
                    continue
                union = (
                    truth_end
                    - truth_start
                    + 1
                    + predicted_end
                    - predicted_start
                    + 1
                    - intersection
                )
                candidates.append((intersection / union, index))
            if not candidates:
                matrix[truth_class]["MISSED"] += 1
                continue
            _iou, best_index = max(candidates, key=lambda value: (value[0], -value[1]))
            unused.remove(best_index)
            predicted_class = predicted_entities[best_index][0]
            matrix[truth_class][predicted_class] += 1
        for index in unused:
            matrix["SPURIOUS"][predicted_entities[index][0]] += 1
    return matrix


def sequence_metrics(
    true_sequences: list[list[str]],
    predicted_sequences: list[list[str]],
) -> dict[str, float]:
    from seqeval.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    report = classification_report(
        true_sequences,
        predicted_sequences,
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, float] = {
        "precision": float(precision_score(true_sequences, predicted_sequences)),
        "recall": float(recall_score(true_sequences, predicted_sequences)),
        "f1": float(f1_score(true_sequences, predicted_sequences)),
        "accuracy": float(accuracy_score(true_sequences, predicted_sequences)),
    }
    recalls: dict[str, float] = {}
    for entity_class in LABEL_CONFIG.entity_classes:
        entity_metrics = report.get(entity_class, {})
        recall = float(entity_metrics.get("recall", 0.0))
        recalls[entity_class] = recall
        metrics[f"recall_{entity_class.lower()}"] = recall
        metrics[f"precision_{entity_class.lower()}"] = float(
            entity_metrics.get("precision", 0.0)
        )
        metrics[f"f1_{entity_class.lower()}"] = float(
            entity_metrics.get("f1-score", 0.0)
        )
        metrics[f"support_{entity_class.lower()}"] = float(
            entity_metrics.get("support", 0.0)
        )

    weights = {
        entity_class: (
            3.0
            if entity_class in LABEL_CONFIG.secret_classes
            else 2.0
            if entity_class in LABEL_CONFIG.contextual_classes
            else 1.0
        )
        for entity_class in LABEL_CONFIG.entity_classes
    }
    supported = [
        entity_class
        for entity_class in LABEL_CONFIG.entity_classes
        if metrics[f"support_{entity_class.lower()}"] > 0
    ]
    if supported:
        weighted_recall = sum(
            weights[name] * recalls[name] for name in supported
        ) / sum(weights[name] for name in supported)
        critical = [
            recalls[name]
            for name in supported
            if name in LABEL_CONFIG.secret_classes | LABEL_CONFIG.contextual_classes
        ]
        minimum_critical = min(critical) if critical else weighted_recall
        metrics["safety_score"] = 0.7 * weighted_recall + 0.3 * minimum_critical
    else:
        metrics["safety_score"] = 0.0
    return metrics


def trainer_metrics() -> Callable[[Any], dict[str, float]]:
    def compute(eval_prediction: Any) -> dict[str, float]:
        truth, predicted = labels_from_predictions(
            eval_prediction.predictions,
            eval_prediction.label_ids,
        )
        return sequence_metrics(truth, predicted)

    return compute


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def load_thresholds(path: Path | None) -> dict[str, float]:
    if path is None:
        return {entity_class: 0.5 for entity_class in LABEL_CONFIG.entity_classes}
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = raw.get("thresholds", raw)
    return {
        entity_class: float(values.get(entity_class, 0.5))
        for entity_class in LABEL_CONFIG.entity_classes
    }


def choose_subset(dataset, max_records: int | None):
    if max_records is None or max_records <= 0 or len(dataset) <= max_records:
        return dataset
    return dataset.select(range(max_records))


def finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {value}")
    return result
