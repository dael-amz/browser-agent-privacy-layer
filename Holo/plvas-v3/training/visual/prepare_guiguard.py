from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_ID = "ShaofantuoshuzhengzhiSha/GUIGuard-Bench"
DATASET_REVISION = "c2b5ca415cd854452585260b9eaa74699042d920"
DATA_FILE = "data/eval.jsonl"
REQUIRED_FIELDS = {
    "id",
    "platform",
    "task_folder",
    "task_goal",
    "step_index",
    "image",
    "privacy_labels",
}


def text_fingerprint(value: str) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        "characters": len(value),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.output_dir} is not empty; pass --force")
    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.output_dir / "records.jsonl"
    counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    platform_counts: Counter[str] = Counter()
    trajectory_counts: Counter[str] = Counter()
    record_digest = hashlib.sha256()

    dataset = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        data_files=DATA_FILE,
        split="train",
        streaming=True,
    )
    with record_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            if args.max_records is not None and index >= args.max_records:
                break
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"GUIGuard row {index} missing {sorted(missing)}")
            annotations = []
            for label in row["privacy_labels"]:
                risk = label.get("label", "unknown")
                category = str(label.get("category", "unknown"))
                text = label.get("ocr_text") or ""
                annotation = {
                    "risk": risk,
                    "draw_type": label.get("draw_type"),
                    "category": category,
                    "is_task_essential": bool(label.get("is_task_essential", False)),
                    "bbox_percent": label.get("bbox_percent") or [],
                    "ocr_text_fingerprint": text_fingerprint(text),
                }
                if args.include_sensitive_text:
                    annotation["ocr_text"] = text
                annotations.append(annotation)
                risk_counts[risk] += 1
                category_counts[category] += 1
            image_relative = None
            if args.download_images:
                from huggingface_hub import hf_hub_download

                source = Path(
                    hf_hub_download(
                        DATASET_ID,
                        filename=row["image"],
                        repo_type="dataset",
                        revision=DATASET_REVISION,
                    )
                )
                destination = args.output_dir / "images" / row["image"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
                image_relative = str(destination.relative_to(args.output_dir))
            record = {
                "id": int(row["id"]),
                "platform": row["platform"],
                "task_folder": row["task_folder"],
                "task_goal_fingerprint": text_fingerprint(row["task_goal"] or ""),
                "step_index": int(row["step_index"]),
                "source_image": row["image"],
                "local_image": image_relative,
                "privacy_labels": annotations,
                "outcome": row.get("outcome"),
                "done": bool(row.get("done", False)),
            }
            encoded = (
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            handle.write(encoded.decode())
            record_digest.update(encoded)
            counts["records"] += 1
            counts["annotations"] += len(annotations)
            if not annotations:
                counts["unannotated_images"] += 1
            platform_counts[row["platform"]] += 1
            trajectory_counts[row["task_folder"]] += 1

    manifest = {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "license": "CC-BY-NC-4.0",
        "purpose": "evaluation",
        "public_ladder_trajectories_are_not_training_data": True,
        "contains_real_sensitive_screenshots": True,
        "raw_ocr_text_persisted": args.include_sensitive_text,
        "images_materialized": args.download_images,
        "counts": dict(counts),
        "risk_counts": dict(sorted(risk_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "platform_counts": dict(sorted(platform_counts.items())),
        "trajectory_count": len(trajectory_counts),
        "records_sha256": record_digest.hexdigest(),
        "records_path": record_path.name,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare GUIGuard as evaluation-only data"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument(
        "--include-sensitive-text",
        action="store_true",
        help="persist real OCR text locally; output must remain ignored and access-controlled",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(prepare(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
