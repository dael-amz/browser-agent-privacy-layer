from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.schema import make_span, sha256_file, write_jsonl_atomic


def quad_bounds(quad: list[list[float]]) -> list[float]:
    xs = [float(point[0]) for point in quad]
    ys = [float(point[1]) for point in quad]
    return [min(xs), min(ys), max(xs), max(ys)]


def overlap_score(one: list[float], two: list[float]) -> float:
    intersection = max(0.0, min(one[2], two[2]) - max(one[0], two[0])) * max(
        0.0, min(one[3], two[3]) - max(one[1], two[1])
    )
    one_area = max(0.0, one[2] - one[0]) * max(0.0, one[3] - one[1])
    two_area = max(0.0, two[2] - two[0]) * max(0.0, two[3] - two[1])
    if not intersection or not one_area or not two_area:
        return 0.0
    return max(intersection / one_area, intersection / two_area)


def fixture_annotations(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        annotation | {"line_text": line["text"]}
        for line in fixture["truth_lines"]
        for annotation in line.get("annotations", [])
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
    goldens = json.loads(args.ocr_goldens.read_text(encoding="utf-8"))
    if fixtures.get("synthetic_only") is not True:
        raise ValueError(
            "OCR semantic holdout must be generated from synthetic fixtures"
        )
    fixtures_by_id = {item["id"]: item for item in fixtures["fixtures"]}
    golden_ids = {item["id"] for item in goldens["fixtures"]}
    if golden_ids != fixtures_by_id.keys():
        raise ValueError("OCR goldens and fixture IDs differ")

    records = []
    fallback_spans = 0
    exact_spans = 0
    for golden_fixture in goldens["fixtures"]:
        fixture = fixtures_by_id[golden_fixture["id"]]
        annotations = fixture_annotations(fixture)
        for region_index, region in enumerate(golden_fixture["regions"]):
            text = str(region["text"])
            if not text:
                continue
            region_box = quad_bounds(region["quad"])
            candidates = sorted(
                (
                    (overlap_score(region_box, annotation["box"]), annotation)
                    for annotation in annotations
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            spans = []
            mapping = "negative"
            if candidates and candidates[0][0] >= args.minimum_overlap:
                _score, annotation = candidates[0]
                expected = annotation["value"]
                start = text.find(expected)
                if start >= 0:
                    spans = [
                        make_span(
                            text,
                            start,
                            start + len(expected),
                            annotation["label"],
                        )
                    ]
                    mapping = "exact-text"
                    exact_spans += 1
                else:
                    spans = [make_span(text, 0, len(text), annotation["label"])]
                    mapping = "conservative-region"
                    fallback_spans += 1
            stable_id = f"{fixture['id']}:{region_index}:{text}"
            seed = int(hashlib.sha256(stable_id.encode()).hexdigest()[:12], 16)
            records.append(
                {
                    "id": f"ocr-holdout:{fixture['id']}:{region_index}",
                    "text": text,
                    "spans": spans,
                    "split": "holdout",
                    "source": "ocr_stack_holdout",
                    "template_id": f"ocr.{fixture['family']}",
                    "seed": seed,
                    "language": "en",
                    "region": "SYNTHETIC_OCR",
                    "noisy": True,
                    "provenance": {
                        "never_train": True,
                        "fixture_id": fixture["id"],
                        "fixture_sha256": fixture["sha256"],
                        "ocr_model_revision": goldens["reference"]["source_revision"],
                        "ocr_contract_sha256": goldens["contract_sha256"],
                        "ocr_region_index": region_index,
                        "ocr_confidence": region["confidence"],
                        "ocr_quad": region["quad"],
                        "span_mapping": mapping,
                    },
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "ocr_holdout.jsonl"
    summary = write_jsonl_atomic(records_path, records)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "never-trained-semantic-evaluation",
        "synthetic_only": True,
        "never_train": True,
        "fixtures": {
            "path": str(args.fixtures),
            "sha256": sha256_file(args.fixtures),
        },
        "ocr_goldens": {
            "path": str(args.ocr_goldens),
            "sha256": sha256_file(args.ocr_goldens),
            "reference": goldens["reference"],
            "contract_sha256": goldens["contract_sha256"],
        },
        "records": summary,
        "span_mapping": {
            "exact_text": exact_spans,
            "conservative_region": fallback_spans,
            "minimum_overlap": args.minimum_overlap,
        },
    }
    manifest_path = args.output_dir / "ocr_holdout_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a never-trained semantic holdout from pinned OCR outputs"
    )
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--ocr-goldens", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-overlap", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
