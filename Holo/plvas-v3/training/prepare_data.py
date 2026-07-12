from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import (
    LABEL_CONFIG,
    iter_jsonl,
    make_span,
    record_values,
    sha256_file,
    validate_record,
    write_jsonl_atomic,
)
from .synth_screen import generate_records


DATASET_ID = "ai4privacy/pii-masking-openpii-1.5m"
DATASET_REVISION = "a785eb528e28be2693c3718a27e066970de5dadb"
BASE_MODEL_ID = "google/bert_uncased_L-4_H-256_A-4"
BASE_MODEL_REVISION = "387825ce42dbb39b87911cdf8e383ee3b25184f8"

DIRECT_LABEL_MAP = {
    "GIVENNAME": "NAME",
    "SURNAME": "NAME",
    "STREET": "ADDRESS",
    "BUILDINGNUM": "ADDRESS",
    "BUILDINGNUMBER": "ADDRESS",
    "CITY": "ADDRESS",
    "ZIPCODE": "ADDRESS",
    "SOCIALNUM": "GOV_ID",
    "SOCIALNUMBER": "GOV_ID",
    "SSN": "GOV_ID",
    "IDCARDNUM": "GOV_ID",
    "IDCARDNUMBER": "GOV_ID",
    "PASSPORTNUM": "GOV_ID",
    "PASSPORTNUMBER": "GOV_ID",
    "DRIVERLICENSENUM": "GOV_ID",
    "DRIVERLICENSENUMBER": "GOV_ID",
    "TAXNUM": "GOV_ID",
    "EMAIL": "EMAIL",
    "TELEPHONENUM": "PHONE",
    "TELEPHONENUMBER": "PHONE",
    "PHONE": "PHONE",
    "CREDITCARDNUMBER": "CARD_NUMBER",
}
BIRTH_CONTEXT_RE = re.compile(
    r"\b(?:date\s+of\s+birth|birth\s*date|dob|born|birthday)\b",
    re.IGNORECASE,
)
MERGEABLE_GAP_RE = re.compile(r"^[\s,./'()-]*$")


def normalize_source_label(label: Any) -> str:
    value = str(label).upper().replace("-", "").replace("_", "")
    value = re.sub(r"\d+$", "", value)
    return value


def map_source_label(text: str, source_span: dict[str, Any]) -> str | None:
    raw_label = normalize_source_label(source_span.get("label", ""))
    mapped = DIRECT_LABEL_MAP.get(raw_label)
    if mapped:
        return mapped
    if raw_label in {"DATE", "DATEOFBIRTH", "DOB"}:
        start = int(source_span["start"])
        context = text[max(0, start - 48) : start]
        if raw_label in {"DATEOFBIRTH", "DOB"} or BIRTH_CONTEXT_RE.search(context):
            return "DOB"
    return None


def merge_spans(text: str, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda span: (span["start"], span["end"]))
    merged: list[dict[str, Any]] = [ordered[0]]
    for span in ordered[1:]:
        previous = merged[-1]
        gap = text[previous["end"] : span["start"]]
        if (
            span["label"] == previous["label"]
            and span["start"] >= previous["end"]
            and len(gap) <= 8
            and MERGEABLE_GAP_RE.fullmatch(gap)
        ):
            merged[-1] = make_span(text, previous["start"], span["end"], span["label"])
        elif span["start"] >= previous["end"]:
            merged.append(span)
        else:
            # Dataset masks should not overlap. Keep the longest mapped span if they do.
            previous_len = previous["end"] - previous["start"]
            current_len = span["end"] - span["start"]
            if current_len > previous_len:
                merged[-1] = span
    return merged


def convert_openpii_row(
    row: dict[str, Any],
    split: str,
    *,
    negative_ratio: float,
) -> dict[str, Any] | None:
    text = row.get("source_text")
    if not isinstance(text, str) or not text.strip():
        return None
    language = str(row.get("language", ""))
    if not (
        language == "en" or language.startswith("en-") or language.startswith("en_")
    ):
        return None

    masks = row.get("privacy_mask") or []
    if isinstance(masks, str):
        try:
            masks = json.loads(masks)
        except json.JSONDecodeError:
            return None
    mapped_spans: list[dict[str, Any]] = []
    raw_labels: list[str] = []
    for source_span in masks:
        try:
            start, end = int(source_span["start"]), int(source_span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= start < end <= len(text)):
            continue
        mapped_label = map_source_label(text, source_span)
        raw_labels.append(str(source_span.get("label", "")))
        if mapped_label is None:
            continue
        mapped_spans.append(make_span(text, start, end, mapped_label))

    mapped_spans = merge_spans(text, mapped_spans)
    uid = str(row.get("uid") or hashlib.sha256(text.encode()).hexdigest()[:20])
    if not mapped_spans:
        sample = int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if sample >= negative_ratio:
            return None

    seed = int(hashlib.sha256(uid.encode()).hexdigest()[:12], 16)
    region = str(row.get("region") or "UNKNOWN")
    record = {
        "id": f"openpii:{split}:{uid}",
        "text": text,
        "spans": mapped_spans,
        "split": split,
        "source": "openpii",
        "template_id": f"openpii.{region.lower()}",
        "seed": seed,
        "language": language,
        "region": region,
        "noisy": False,
        "provenance": {
            "dataset": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "source_uid": uid,
            "source_labels": sorted(set(raw_labels)),
        },
    }
    validate_record(record)
    return record


def iter_openpii(
    split: str,
    limit: int,
    *,
    seed: int,
    negative_ratio: float,
    dataset_revision: str,
) -> Iterator[dict[str, Any]]:
    if limit <= 0:
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for OpenPII; use --no-openpii for a pure synthetic smoke test"
        ) from exc

    dataset = load_dataset(
        DATASET_ID,
        split=split,
        revision=dataset_revision,
        streaming=True,
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    accepted = 0
    for row in dataset:
        record = convert_openpii_row(row, split, negative_ratio=negative_ratio)
        if record is None:
            continue
        yield record
        accepted += 1
        if accepted >= limit:
            break
    if accepted < limit:
        raise RuntimeError(
            f"requested {limit} {split} rows but accepted only {accepted}"
        )


class SplitFilter:
    def __init__(self) -> None:
        self.train_values: set[tuple[str, str]] = set()
        self.seen_ids: set[str] = set()
        self.skipped_overlap = {"validation": 0, "holdout": 0}
        self.skipped_duplicate_id = 0

    def filter(
        self,
        records: Iterable[dict[str, Any]],
        split: str,
    ) -> Iterator[dict[str, Any]]:
        for record in records:
            if record["id"] in self.seen_ids:
                self.skipped_duplicate_id += 1
                continue
            values = record_values(record)
            if split != "train" and values & self.train_values:
                self.skipped_overlap[split] += 1
                continue
            self.seen_ids.add(record["id"])
            if split == "train":
                self.train_values.update(values)
            yield record


def _chain(*iterables: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for iterable in iterables:
        yield from iterable


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    split_filter = SplitFilter()

    openpii_train: Iterable[dict[str, Any]] = ()
    openpii_validation: Iterable[dict[str, Any]] = ()
    if args.include_openpii:
        openpii_train = iter_openpii(
            "train",
            args.max_openpii_train,
            seed=args.seed,
            negative_ratio=args.negative_ratio,
            dataset_revision=args.dataset_revision,
        )
        openpii_validation = iter_openpii(
            "validation",
            args.max_openpii_validation,
            seed=args.seed + 1,
            negative_ratio=args.negative_ratio,
            dataset_revision=args.dataset_revision,
        )

    train_records = split_filter.filter(
        _chain(
            generate_records(
                "train",
                args.synth_train,
                seed=args.seed,
                noise_fraction=args.noise_fraction,
            ),
            openpii_train,
        ),
        "train",
    )
    train_summary = write_jsonl_atomic(output_dir / "train.jsonl", train_records)

    validation_records = split_filter.filter(
        _chain(
            generate_records(
                "validation",
                args.synth_validation,
                seed=args.seed,
                noise_fraction=args.noise_fraction,
            ),
            openpii_validation,
        ),
        "validation",
    )
    validation_summary = write_jsonl_atomic(
        output_dir / "validation.jsonl", validation_records
    )

    holdout_records = split_filter.filter(
        generate_records(
            "holdout",
            args.synth_holdout,
            seed=args.seed,
            noise_fraction=args.noise_fraction,
        ),
        "holdout",
    )
    holdout_summary = write_jsonl_atomic(output_dir / "holdout.jsonl", holdout_records)

    ocr_holdout_path = getattr(args, "ocr_holdout", None)
    ocr_holdout_manifest = getattr(args, "ocr_holdout_manifest", None)
    ocr_holdout_summary = None
    if ocr_holdout_path:

        def verified_ocr_holdout() -> Iterator[dict[str, Any]]:
            for record in iter_jsonl(ocr_holdout_path):
                validate_record(record)
                if record["source"] != "ocr_stack_holdout":
                    raise ValueError("OCR holdout contains a non-OCR source record")
                if record.get("provenance", {}).get("never_train") is not True:
                    raise ValueError("OCR holdout record is not marked never_train")
                if record_values(record) & split_filter.train_values:
                    raise ValueError("OCR holdout value overlaps training data")
                yield record

        ocr_holdout_summary = write_jsonl_atomic(
            output_dir / "ocr_holdout.jsonl",
            verified_ocr_holdout(),
        )
        ocr_holdout_summary["never_train"] = True
        if ocr_holdout_manifest:
            ocr_holdout_summary["source_manifest"] = str(ocr_holdout_manifest)
            ocr_holdout_summary["source_manifest_sha256"] = sha256_file(
                ocr_holdout_manifest
            )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "id": DATASET_ID,
            "revision": args.dataset_revision,
            "included": args.include_openpii,
            "published_splits_preserved": True,
            "language_filter": "en and en-*",
            "license": "CC-BY-4.0",
        },
        "base_model": {
            "id": BASE_MODEL_ID,
            "revision": BASE_MODEL_REVISION,
            "license": "Apache-2.0",
        },
        "labels": list(LABEL_CONFIG.labels),
        "config": {
            "seed": args.seed,
            "max_openpii_train": args.max_openpii_train,
            "max_openpii_validation": args.max_openpii_validation,
            "synth_train": args.synth_train,
            "synth_validation": args.synth_validation,
            "synth_holdout": args.synth_holdout,
            "noise_fraction": args.noise_fraction,
            "negative_ratio": args.negative_ratio,
        },
        "files": {
            "train": train_summary,
            "validation": validation_summary,
            "holdout": holdout_summary,
            **(
                {"ocr_holdout": ocr_holdout_summary}
                if ocr_holdout_summary is not None
                else {}
            ),
        },
        "leakage_filter": {
            "train_canonical_values": len(split_filter.train_values),
            "skipped_validation_overlap": split_filter.skipped_overlap["validation"],
            "skipped_holdout_overlap": split_filter.skipped_overlap["holdout"],
            "skipped_duplicate_id": split_filter.skipped_duplicate_id,
        },
    }
    manifest_path = output_dir / "data_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    defaults = json.loads((Path(__file__).parent / "data_manifest.json").read_text())
    config = defaults["defaults"]
    parser = argparse.ArgumentParser(
        description="Prepare leakage-safe PLVA tagger data"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument(
        "--max-openpii-train", type=int, default=config["openpii_train"]
    )
    parser.add_argument(
        "--max-openpii-validation",
        type=int,
        default=config["openpii_validation"],
    )
    parser.add_argument("--synth-train", type=int, default=config["synthetic_train"])
    parser.add_argument(
        "--synth-validation",
        type=int,
        default=config["synthetic_validation"],
    )
    parser.add_argument(
        "--synth-holdout",
        type=int,
        default=config["synthetic_holdout"],
    )
    parser.add_argument(
        "--noise-fraction",
        type=float,
        default=config["synthetic_noise_fraction"],
    )
    parser.add_argument("--negative-ratio", type=float, default=0.15)
    parser.add_argument(
        "--ocr-holdout",
        type=Path,
        help="never-trained JSONL generated by training.ocr.build_semantic_holdout",
    )
    parser.add_argument(
        "--ocr-holdout-manifest",
        type=Path,
        help="manifest paired with --ocr-holdout",
    )
    parser.add_argument(
        "--no-openpii",
        action="store_false",
        dest="include_openpii",
        help="Generate synthetic data only",
    )
    parser.set_defaults(include_openpii=True)
    return parser.parse_args()


def main() -> None:
    manifest = prepare(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
