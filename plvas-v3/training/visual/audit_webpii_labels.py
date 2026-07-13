from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from training.visual.prepare_webpii import (
    WEBPII_ID,
    WEBPII_REVISION,
    SOURCE_MAPPING_VERSION,
    rows,
    source_disposition,
)


def audit_split(split: str, maximum: int | None) -> dict[str, Any]:
    records = 0
    visible_annotations = 0
    dispositions: Counter[str] = Counter()
    for row in rows(split, maximum, page_delay_seconds=0.25):
        records += 1
        for element in json.loads(row["pii_elements_json"]):
            if not element.get("visible", False):
                continue
            visible_annotations += 1
            source_key = str(element["key"])
            disposition, class_name = source_disposition(source_key)
            dispositions[f"{disposition}:{class_name or '-'}:{source_key}"] += 1
    unmapped = {
        key.removeprefix("unmapped:-:"): count
        for key, count in dispositions.items()
        if key.startswith("unmapped:-:")
    }
    return {
        "records": records,
        "visible_annotations": visible_annotations,
        "unmapped": dict(sorted(unmapped.items())),
        "dispositions": dict(sorted(dispositions.items())),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    # Sequential paging respects the datasets-server rate limit. Screenshot
    # download and GPU training must never begin on an incomplete label audit.
    splits = {
        "train": audit_split("train", args.max_train),
        "test": audit_split("test", args.max_test),
    }
    unmapped = {
        split: details["unmapped"]
        for split, details in splits.items()
        if details["unmapped"]
    }
    document = {
        "schema_version": 1,
        "dataset": WEBPII_ID,
        "revision": WEBPII_REVISION,
        "source_mapping_version": SOURCE_MAPPING_VERSION,
        "metadata_only": True,
        "values_persisted": False,
        "passed": not unmapped,
        "unmapped": unmapped,
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.fail_on_unmapped and unmapped:
        raise RuntimeError(f"unmapped WebPII labels remain: {unmapped}")
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit pinned WebPII source labels without downloading screenshots"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--fail-on-unmapped", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
