from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .modeling import (
    choose_subset,
    entity_confusion_matrix,
    labels_from_merged_windows,
    load_raw_datasets,
    load_thresholds,
    sequence_metrics,
    tokenize_dataset,
)
from .schema import LABEL_CONFIG


def evaluate_dataset(
    dataset,
    *,
    name: str,
    model,
    tokenizer,
    thresholds: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from transformers import (
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    selected = choose_subset(dataset, args.max_records_per_slice)
    tokenized = tokenize_dataset(
        selected,
        tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        num_proc=args.num_proc,
        desc=f"Tokenizing {name}",
        include_metadata=True,
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / ".eval-tmp" / name.replace("/", "_")),
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
    truth, predicted = labels_from_merged_windows(
        prediction.predictions,
        prediction.label_ids,
        tokenized["source_record_index"],
        tokenized["source_offsets"],
        tokenized["source_truth_spans"],
        thresholds=thresholds,
    )
    return {
        "records": len(selected),
        "windows": len(tokenized),
        "metrics": sequence_metrics(truth, predicted),
        "confusion_matrix": entity_confusion_matrix(truth, predicted),
    }


def _filter(dataset, predicate):
    return dataset.filter(predicate, desc="Selecting evaluation slice")


def gate_results(
    holdout: dict[str, Any],
    ocr_holdout: dict[str, Any] | None,
) -> dict[str, Any]:
    holdout_metrics = holdout["metrics"]
    confusion = holdout["confusion_matrix"]
    checks: dict[str, Any] = {}
    for entity_class in LABEL_CONFIG.entity_classes:
        key = f"recall_{entity_class.lower()}"
        support = int(holdout_metrics.get(f"support_{entity_class.lower()}", 0))
        required = 1.0 if entity_class in LABEL_CONFIG.secret_classes else 0.95
        recall = holdout_metrics.get(key, 0.0)
        checks[f"{entity_class}_support"] = {
            "actual": support,
            "required": 1,
            "passed": support >= 1,
        }
        checks[f"{entity_class}_recall"] = {
            "actual": recall,
            "required": required,
            "passed": support >= 1 and recall >= required,
        }
    for entity_class in LABEL_CONFIG.secret_classes:
        support = int(holdout_metrics.get(f"support_{entity_class.lower()}", 0))
        correct = int(confusion.get(entity_class, {}).get(entity_class, 0))
        missed = max(0, support - correct)
        checks[f"{entity_class}_missed_entities"] = {
            "actual": missed,
            "required": 0,
            "passed": support >= 1 and missed == 0,
        }
    checks["aggregate_recall"] = {
        "actual": holdout_metrics.get("recall", 0.0),
        "required": 0.97,
        "passed": holdout_metrics.get("recall", 0.0) >= 0.97,
    }
    checks["aggregate_precision"] = {
        "actual": holdout_metrics.get("precision", 0.0),
        "required": 0.95,
        "passed": holdout_metrics.get("precision", 0.0) >= 0.95,
    }
    checks["ocr_stack_holdout_present"] = {
        "actual": ocr_holdout["records"] if ocr_holdout else 0,
        "required": 1,
        "passed": bool(ocr_holdout and ocr_holdout["records"] > 0),
    }
    if ocr_holdout:
        ocr_metrics = ocr_holdout["metrics"]
        ocr_confusion = ocr_holdout["confusion_matrix"]
        for entity_class in LABEL_CONFIG.entity_classes:
            key = entity_class.lower()
            support = int(ocr_metrics.get(f"support_{key}", 0))
            required_recall = (
                1.0 if entity_class in LABEL_CONFIG.secret_classes else 0.95
            )
            recall = ocr_metrics.get(f"recall_{key}", 0.0)
            checks[f"ocr_stack_{entity_class}_support"] = {
                "actual": support,
                "required": 1,
                "passed": support >= 1,
            }
            checks[f"ocr_stack_{entity_class}_recall"] = {
                "actual": recall,
                "required": required_recall,
                "passed": support >= 1 and recall >= required_recall,
            }
        secret_support = sum(
            int(ocr_metrics.get(f"support_{name.lower()}", 0))
            for name in LABEL_CONFIG.secret_classes
        )
        secret_correct = sum(
            int(ocr_confusion.get(name, {}).get(name, 0))
            for name in LABEL_CONFIG.secret_classes
        )
        checks["ocr_stack_secret_missed_entities"] = {
            "actual": max(0, secret_support - secret_correct),
            "required": 0,
            "passed": secret_support > 0 and secret_support == secret_correct,
        }
        checks["ocr_stack_aggregate_recall"] = {
            "actual": ocr_metrics.get("recall", 0.0),
            "required": 0.97,
            "passed": ocr_metrics.get("recall", 0.0) >= 0.97,
        }
        checks["ocr_stack_aggregate_precision"] = {
            "actual": ocr_metrics.get("precision", 0.0),
            "required": 0.95,
            "passed": ocr_metrics.get("precision", 0.0) >= 0.95,
        }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
    }


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# PLVA tagger evaluation",
        "",
        f"Model: `{result['model_dir']}`",
        "",
        f"Overall gate: **{'PASS' if result['gates']['passed'] else 'FAIL'}**",
        "",
        "| Slice | Records | Precision | Recall | F1 | Safety score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, details in result["slices"].items():
        metrics = details["metrics"]
        if not metrics:
            lines.append(f"| {name} | {details['records']} | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {name} | {details['records']} | {metrics.get('precision', 0):.4f} | "
            f"{metrics.get('recall', 0):.4f} | {metrics.get('f1', 0):.4f} | "
            f"{metrics.get('safety_score', 0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Holdout per-class results",
            "",
            "| Class | Support | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    holdout = result["slices"]["holdout/all"]["metrics"]
    for entity_class in LABEL_CONFIG.entity_classes:
        key = entity_class.lower()
        lines.append(
            f"| {entity_class} | {int(holdout.get(f'support_{key}', 0))} | "
            f"{holdout.get(f'precision_{key}', 0):.4f} | "
            f"{holdout.get(f'recall_{key}', 0):.4f} | "
            f"{holdout.get(f'f1_{key}', 0):.4f} |"
        )
    lines.extend(["", "Thresholds are calibrated on validation data only.", ""])
    lines.extend(
        [
            "## Holdout entity confusion matrix",
            "",
            "Rows are truth classes. `MISSED` means no overlapping prediction.",
            "",
            "```json",
            json.dumps(
                result["slices"]["holdout/all"].get("confusion_matrix", {}),
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw_datasets(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    thresholds = load_thresholds(args.thresholds)

    slices = {
        "validation/all": raw["validation"],
        "validation/synthetic-clean": _filter(
            raw["validation"],
            lambda row: row["source"] == "synthetic_screen" and not row["noisy"],
        ),
        "validation/synthetic-noisy": _filter(
            raw["validation"],
            lambda row: row["source"] == "synthetic_screen" and row["noisy"],
        ),
        "validation/openpii": _filter(
            raw["validation"], lambda row: row["source"] == "openpii"
        ),
        "holdout/all": raw["holdout"],
        "holdout/clean": _filter(raw["holdout"], lambda row: not row["noisy"]),
        "holdout/noisy": _filter(raw["holdout"], lambda row: row["noisy"]),
    }
    if "ocr_holdout" in raw:
        slices["holdout/ocr-stack"] = raw["ocr_holdout"]
    for region in sorted(set(raw["validation"]["region"])):
        slices[f"validation/region/{region}"] = _filter(
            raw["validation"], lambda row, target=region: row["region"] == target
        )
    results: dict[str, Any] = {}
    for name, dataset in slices.items():
        if len(dataset) == 0:
            results[name] = {
                "records": 0,
                "windows": 0,
                "metrics": {},
                "confusion_matrix": {},
            }
            continue
        results[name] = evaluate_dataset(
            dataset,
            name=name,
            model=model,
            tokenizer=tokenizer,
            thresholds=thresholds,
            args=args,
        )
    gates = gate_results(
        results["holdout/all"],
        results.get("holdout/ocr-stack"),
    )
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(args.model_dir),
        "data_dir": str(args.data_dir),
        "thresholds": thresholds,
        "slices": results,
        "gates": gates,
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "evaluation.md").write_text(
        markdown_report(result),
        encoding="utf-8",
    )
    if args.enforce_gates and not gates["passed"]:
        failed = [name for name, item in gates["checks"].items() if not item["passed"]]
        raise RuntimeError(f"tagger evaluation gates failed: {', '.join(failed)}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the PLVA tagger")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records-per-slice", type=int, default=30_000)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--enforce-gates", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
