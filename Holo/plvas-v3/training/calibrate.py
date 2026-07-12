from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .modeling import (
    choose_subset,
    load_raw_datasets,
    merged_class_probabilities,
    tokenize_dataset,
)
from .schema import LABEL_CONFIG


def threshold_metrics(
    probabilities: np.ndarray, truth: np.ndarray, threshold: float
) -> dict:
    predicted = probabilities >= threshold
    true_positive = int(np.logical_and(predicted, truth).sum())
    false_positive = int(np.logical_and(predicted, ~truth).sum())
    false_negative = int(np.logical_and(~predicted, truth).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    beta_sq = 4.0
    f2 = (1 + beta_sq) * precision * recall / max(1e-12, beta_sq * precision + recall)
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def choose_threshold(
    probabilities: np.ndarray,
    truth: np.ndarray,
    *,
    target_recall: float,
) -> dict[str, Any]:
    candidates = [
        threshold_metrics(probabilities, truth, float(threshold))
        for threshold in np.linspace(0.05, 0.95, 91)
    ]
    meeting_target = [item for item in candidates if item["recall"] >= target_recall]
    if meeting_target:
        best = max(
            meeting_target,
            key=lambda item: (item["precision"], item["f2"], item["threshold"]),
        )
        best["met_target"] = True
    else:
        best = max(candidates, key=lambda item: (item["f2"], item["recall"]))
        best["met_target"] = False
    best["target_recall"] = target_recall
    best["positive_tokens"] = int(truth.sum())
    return best


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    raw = load_raw_datasets(args.data_dir)
    validation = choose_subset(raw["validation"], args.max_records)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    tokenized = tokenize_dataset(
        validation,
        tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        num_proc=args.num_proc,
        desc="Tokenizing calibration",
        include_metadata=True,
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output.parent / ".calibration-tmp"),
            per_device_eval_batch_size=args.batch_size,
            report_to="none",
            eval_accumulation_steps=16,
            dataloader_num_workers=args.dataloader_workers,
            remove_unused_columns=True,
        ),
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
    )
    prediction = trainer.predict(tokenized)
    probabilities, truth_classes = merged_class_probabilities(
        prediction.predictions,
        prediction.label_ids,
        tokenized["source_record_index"],
        tokenized["source_offsets"],
    )

    thresholds: dict[str, float] = {}
    calibration: dict[str, Any] = {}
    truth_array = np.asarray(truth_classes, dtype=object)
    for class_index, entity_class in enumerate(LABEL_CONFIG.entity_classes):
        class_probability = probabilities[:, class_index]
        truth = truth_array == entity_class
        if entity_class in LABEL_CONFIG.secret_classes:
            target = args.secret_target_recall
        elif entity_class in LABEL_CONFIG.contextual_classes:
            target = args.contextual_target_recall
        else:
            target = args.default_target_recall
        if int(truth.sum()) == 0:
            details = {
                "threshold": 0.5,
                "precision": 0.0,
                "recall": 0.0,
                "f2": 0.0,
                "met_target": False,
                "target_recall": target,
                "positive_tokens": 0,
            }
        else:
            details = choose_threshold(class_probability, truth, target_recall=target)
        thresholds[entity_class] = round(float(details["threshold"]), 4)
        calibration[entity_class] = details

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(args.model_dir),
        "data_dir": str(args.data_dir),
        "records": len(validation),
        "windows": len(tokenized),
        "unique_tokens": len(truth_classes),
        "thresholds": thresholds,
        "calibration": calibration,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate PLVA per-class thresholds")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=20_000)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--secret-target-recall", type=float, default=1.0)
    parser.add_argument("--contextual-target-recall", type=float, default=0.95)
    parser.add_argument("--default-target-recall", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(calibrate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
