from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SECRET_CLASSES = {"CARD_NUMBER", "CVC", "SECRET"}


def intersection_over_truth(prediction: list[float], truth: list[float]) -> float:
    x1 = max(prediction[0], truth[0])
    y1 = max(prediction[1], truth[1])
    x2 = min(prediction[2], truth[2])
    y2 = min(prediction[3], truth[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    truth_area = max(0.0, truth[2] - truth[0]) * max(0.0, truth[3] - truth[1])
    return intersection / truth_area if truth_area else 0.0


def compatible(predicted: str, truth: str) -> bool:
    if predicted == truth:
        return True
    if predicted == "text":
        return truth != "SENSITIVE_IMAGE"
    if predicted == "image":
        return truth == "SENSITIVE_IMAGE"
    return False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    truth_records = {item["id"]: item for item in read_jsonl(args.truth)}
    prediction_records = {item["id"]: item for item in read_jsonl(args.predictions)}
    if truth_records.keys() != prediction_records.keys():
        missing = sorted(truth_records.keys() - prediction_records.keys())
        extra = sorted(prediction_records.keys() - truth_records.keys())
        raise ValueError(
            f"prediction IDs differ: missing={missing[:5]}, extra={extra[:5]}"
        )

    truth_counts: Counter[str] = Counter()
    covered_counts: Counter[str] = Counter()
    family_truth: Counter[str] = Counter()
    family_covered: Counter[str] = Counter()
    prediction_count = 0
    matched_predictions = 0
    false_proposals: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []

    for record_id, truth_record in truth_records.items():
        predictions = prediction_records[record_id].get("proposals", [])
        prediction_count += len(predictions)
        matched = set()
        family = truth_record["page_type"]
        for truth_index, annotation in enumerate(truth_record["annotations"]):
            truth_class = annotation["class"]
            truth_counts[truth_class] += 1
            family_truth[family] += 1
            candidates = [
                (
                    intersection_over_truth(proposal["xyxy"], annotation["bbox_xyxy"]),
                    index,
                )
                for index, proposal in enumerate(predictions)
                if compatible(proposal["class"], truth_class)
            ]
            coverage, best = max(candidates, default=(0.0, -1))
            if coverage >= args.coverage_threshold:
                covered_counts[truth_class] += 1
                family_covered[family] += 1
                matched.add(best)
            else:
                misses.append(
                    {
                        "id": record_id,
                        "truth_index": truth_index,
                        "class": truth_class,
                        "page_type": family,
                        "best_coverage": coverage,
                    }
                )
        matched_predictions += len(matched)
        false_proposals.extend(
            {
                "id": record_id,
                "proposal_index": index,
                "class": proposal["class"],
                "confidence": proposal.get("confidence"),
            }
            for index, proposal in enumerate(predictions)
            if index not in matched
        )

    total_truth = sum(truth_counts.values())
    total_covered = sum(covered_counts.values())
    per_class = {
        entity_class: {
            "support": truth_counts[entity_class],
            "covered": covered_counts[entity_class],
            "recall": covered_counts[entity_class] / truth_counts[entity_class]
            if truth_counts[entity_class]
            else 0.0,
        }
        for entity_class in sorted(truth_counts)
    }
    by_family = {
        family: {
            "support": family_truth[family],
            "covered": family_covered[family],
            "recall": family_covered[family] / family_truth[family],
        }
        for family in sorted(family_truth)
    }
    secret_support = sum(truth_counts[name] for name in SECRET_CLASSES)
    secret_covered = sum(covered_counts[name] for name in SECRET_CLASSES)
    metrics = {
        "support": total_truth,
        "covered": total_covered,
        "proposal_recall": total_covered / total_truth if total_truth else 0.0,
        "proposal_precision": matched_predictions / prediction_count
        if prediction_count
        else 0.0,
        "prediction_count": prediction_count,
        "secret_support": secret_support,
        "secret_misses": secret_support - secret_covered,
        "per_class": per_class,
        "by_page_type": by_family,
    }
    gates = {
        "visual_proposal_recall": {
            "actual": metrics["proposal_recall"],
            "required": 0.95,
            "passed": metrics["proposal_recall"] >= 0.95,
        },
        "secret_misses": {
            "actual": metrics["secret_misses"],
            "required": 0,
            "passed": secret_support > 0 and metrics["secret_misses"] == 0,
        },
    }
    result = {
        "schema_version": 1,
        "coverage_threshold": args.coverage_threshold,
        "metrics": metrics,
        "gates": {
            "passed": all(item["passed"] for item in gates.values()),
            "checks": gates,
        },
        "misses": misses,
        "false_proposals": false_proposals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate visual proposal predictions")
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-threshold", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
