from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


WEBPII_ID = "WebPII/webpii"
WEBPII_REVISION = "6d3317721b72bde719a361c564ceaf1fbded3a8e"
SOURCE_MAPPING_VERSION = 3
EXPECTED_SPLIT_RECORDS = {"train": 40_384, "test": 4_481}
DETECTOR_CLASSES = (
    "NAME",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "CARD_NUMBER",
    "CVC",
    "SECRET",
    "SENSITIVE_FIELD",
    "SENSITIVE_IMAGE",
)


def source_disposition(source_key: str) -> tuple[str, str | None]:
    key = source_key.removeprefix("PII_").upper()
    # Explicit non-PII lookalikes must be resolved before broad token rules.
    if (
        key.startswith("PROMO_CODE")
        or key in {"CARD_TYPE", "COMPANY", "JOB_CODE"}
        or key.endswith("_COMPANY")
    ):
        return "hard-negative", None
    if any(
        token in key
        for token in ("FIRSTNAME", "LASTNAME", "FULLNAME", "NAME_FULL")
    ) or key in {"INITIALS", "CARD_NAME"}:
        return "map", "NAME"
    if re.fullmatch(r"LOCATION\d+_NAME", key):
        return "map", "NAME"
    if "EMAIL" in key:
        return "map", "EMAIL"
    if "PHONE" in key:
        return "map", "PHONE"
    address_tokens = ("STREET", "CITY", "STATE", "POSTCODE", "ADDRESS", "COUNTRY")
    if any(token in key for token in address_tokens) or (
        re.match(r"LOCATION\d+_", key)
        and any(token in key for token in address_tokens)
    ):
        return "map", "ADDRESS"
    if key.startswith(("CARD_NUMBER", "CARD_LAST4")):
        return "map", "CARD_NUMBER"
    if key.startswith("CARD_CVV"):
        return "map", "CVC"
    if (
        any(token in key for token in ("SECURITY_CODE", "PASSWORD", "PASSCODE"))
        or key.startswith(("GIFT_CODE", "GIFT_PIN"))
    ):
        return "map", "SECRET"
    if (
        key.startswith(("AVATAR", "CARD_IMAGE"))
        or key.endswith("_IMAGE")
        or key == "MAP"
    ):
        return "map", "SENSITIVE_IMAGE"
    if key.startswith(
        (
            "ACCOUNT_ID",
            "CARD_EXPIRY",
            "DELIVERY_INSTRUCTIONS",
            "GIFT_MESSAGE",
            "LOGIN_USERNAME",
            "REVIEW",
            "USER_RATING",
            "WALMART_CASH",
        )
    ) or key.startswith("DOB"):
        return "map", "SENSITIVE_FIELD"
    return "unmapped", None


def clamp_box(element: dict[str, Any], width: int, height: int) -> list[float] | None:
    # Compute the far edge from the raw origin before clamping.  Computing it
    # from clamped x1/y1 makes a box beginning off-screen artificially wider.
    raw_x1 = float(element["bbox_x"])
    raw_y1 = float(element["bbox_y"])
    raw_x2 = raw_x1 + float(element["bbox_width"])
    raw_y2 = raw_y1 + float(element["bbox_height"])
    x1 = max(0.0, min(raw_x1, float(width)))
    y1 = max(0.0, min(raw_y1, float(height)))
    x2 = max(0.0, min(raw_x2, float(width)))
    y2 = max(0.0, min(raw_y2, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def yolo_line(class_name: str, box: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    return (
        f"{DETECTOR_CLASSES.index(class_name)} "
        f"{((x1 + x2) / 2) / width:.8f} {((y1 + y2) / 2) / height:.8f} "
        f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}"
    )


def safe_stem(row: dict[str, Any]) -> str:
    raw = f"{row['company']}-{row['page_type']}-{row['source_id']}-{row['variant']}"
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in raw
    )
    suffix = hashlib.sha256(raw.encode()).hexdigest()[:10]
    return f"{cleaned[:100]}-{suffix}"


def request_json(url: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(12):
        try:
            with urlopen(
                Request(url, headers={"User-Agent": "plva-webpii-prep/1"}), timeout=120
            ) as response:
                return json.load(response)
        except Exception as exc:  # network failures receive bounded retries
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                raise
            if attempt < 11:
                retry_after = 0.0
                if isinstance(exc, HTTPError):
                    try:
                        retry_after = float(exc.headers.get("Retry-After", 0))
                    except (TypeError, ValueError):
                        retry_after = 0.0
                # The datasets server can impose a multi-minute 429 window.
                # Keep the audit/download alive rather than discarding hours
                # of verified work after the old one-minute retry budget.
                backoff = min(120.0, max(5.0, float(2**attempt)))
                time.sleep(max(retry_after, backoff))
    raise RuntimeError(f"failed to fetch WebPII metadata: {last_error}")


def rows(
    split: str,
    maximum: int | None,
    *,
    page_delay_seconds: float = 0.0,
) -> Iterator[dict[str, Any]]:
    offset = 0
    page_size = 100
    while maximum is None or offset < maximum:
        length = page_size if maximum is None else min(page_size, maximum - offset)
        query = urlencode(
            {
                "dataset": WEBPII_ID,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": length,
                "revision": WEBPII_REVISION,
            }
        )
        payload = request_json(f"https://datasets-server.huggingface.co/rows?{query}")
        page = payload.get("rows", [])
        if not page:
            break
        for item in page:
            yield item["row"]
        offset += len(page)
        if len(page) < length:
            break
        if page_delay_seconds > 0:
            time.sleep(page_delay_seconds)


def download_image(url: str, maximum_bytes: int = 30_000_000) -> bytes:
    parsed_host = Request(url).host
    if parsed_host != "datasets-server.huggingface.co":
        raise ValueError(f"unexpected WebPII image host: {parsed_host}")
    retryable_statuses = {429, 500, 502, 503, 504}
    last_error: Exception | None = None
    jitter = int(hashlib.sha256(url.encode()).hexdigest()[:8], 16) % 1000 / 1000
    for attempt in range(10):
        try:
            with urlopen(
                Request(url, headers={"User-Agent": "plva-webpii-prep/1"}),
                timeout=120,
            ) as response:
                content = response.read(maximum_bytes + 1)
            if len(content) > maximum_bytes:
                raise ValueError("WebPII image exceeded the 30 MB safety limit")
            return content
        except HTTPError as exc:
            last_error = exc
            try:
                exc.close()
            except Exception:
                pass
            if exc.code not in retryable_statuses:
                raise RuntimeError(
                    f"WebPII image fetch failed with non-retryable HTTP {exc.code}"
                ) from None
            retry_after = 0.0
            try:
                retry_after = float(exc.headers.get("Retry-After", 0))
            except (TypeError, ValueError):
                retry_after = 0.0
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
            retry_after = 0.0
        if attempt < 9:
            time.sleep(max(retry_after, min(60.0, float(2**attempt)) + jitter))
    raise RuntimeError(
        f"WebPII image fetch failed after 10 attempts: {type(last_error).__name__}"
    ) from None


def rows_with_images(
    source_rows: Iterator[dict[str, Any]],
    *,
    metadata_only: bool,
    download_workers: int,
    batch_size: int = 128,
) -> Iterator[tuple[dict[str, Any], bytes | None]]:
    """Prefetch a bounded batch of screenshot bytes without reordering rows."""
    if metadata_only:
        for row in source_rows:
            yield row, None
        return
    if download_workers < 1:
        raise ValueError("download_workers must be positive")
    if batch_size < download_workers:
        raise ValueError("download batch size must be at least download_workers")

    def fetch(row: dict[str, Any]) -> bytes:
        image = row.get("image")
        if not isinstance(image, dict) or not image.get("src"):
            raise ValueError(f"WebPII row {row.get('source_id')} has no image URL")
        return download_image(str(image["src"]))

    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        while True:
            batch = list(islice(source_rows, batch_size))
            if not batch:
                break
            futures = [executor.submit(fetch, row) for row in batch]
            for row, future in zip(batch, futures, strict=True):
                yield row, future.result()


def prepare_split(
    root: Path,
    split: str,
    maximum: int | None,
    *,
    metadata_only: bool,
    fail_on_unmapped: bool,
    download_workers: int,
) -> dict[str, Any]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    record_path = root / "records" / f"{split}.jsonl"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    companies: Counter[str] = Counter()
    page_types: Counter[str] = Counter()
    digest = hashlib.sha256()

    with record_path.open("w", encoding="utf-8") as record_file:
        prepared_rows = rows_with_images(
            rows(split, maximum),
            metadata_only=metadata_only,
            download_workers=download_workers,
        )
        for index, (row, image_bytes) in enumerate(prepared_rows):
            width, height = int(row["image_width"]), int(row["image_height"])
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid WebPII dimensions for {row['source_id']}")
            stem = safe_stem(row)
            image_relative = Path("images") / split / f"{stem}.png"
            label_relative = Path("labels") / split / f"{stem}.txt"
            annotations = []
            annotation_indices: dict[tuple[Any, ...], int] = {}
            negative_boxes = []
            elements = json.loads(row["pii_elements_json"])
            for element in elements:
                if not element.get("visible", False):
                    continue
                source_key = str(element["key"])
                disposition, class_name = source_disposition(source_key)
                source_counts[f"{disposition}:{source_key}"] += 1
                box = clamp_box(element, width, height)
                if box is None:
                    counts["invalid_boxes"] += 1
                    continue
                annotation = {
                    "source_key": source_key,
                    "element_type": element.get("element_type"),
                    "bbox_xyxy": box,
                    "clipped": bool(element.get("clipped", False)),
                }
                if disposition == "map" and class_name is not None:
                    annotation["class"] = class_name
                    identity = (
                        class_name,
                        *(round(coordinate, 4) for coordinate in box),
                    )
                    if identity in annotation_indices:
                        existing = annotations[annotation_indices[identity]]
                        existing.setdefault("source_keys", [existing["source_key"]])
                        existing["source_keys"].append(source_key)
                        existing["clipped"] = bool(
                            existing["clipped"] or annotation["clipped"]
                        )
                        counts["duplicate_annotations_merged"] += 1
                    else:
                        annotation_indices[identity] = len(annotations)
                        annotations.append(annotation)
                        class_counts[class_name] += 1
                elif disposition == "hard-negative":
                    negative_boxes.append(
                        annotation | {"reason": "explicit-non-pii-lookalike"}
                    )
                elif fail_on_unmapped:
                    raise ValueError(f"unmapped WebPII key: {source_key}")
            label_lines = [
                yolo_line(item["class"], item["bbox_xyxy"], width, height)
                for item in annotations
            ]
            for field in (
                "product_elements_json",
                "order_elements_json",
                "search_elements_json",
                "misc_elements_json",
            ):
                negative_boxes.extend(
                    {
                        "source_key": item.get("key"),
                        "bbox_xyxy": clamp_box(item, width, height),
                        "reason": field.removesuffix("_elements_json"),
                    }
                    for item in json.loads(row[field])
                    if item.get("visible", False)
                    and clamp_box(item, width, height) is not None
                )

            if not metadata_only:
                if image_bytes is None:
                    raise RuntimeError("WebPII image prefetch returned no bytes")
                (root / image_relative).write_bytes(image_bytes)
            (root / label_relative).write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )
            record = {
                "id": stem,
                "published_split": split,
                "source_id": str(row["source_id"]),
                "variant": row["variant"],
                "company": row["company"],
                "page_type": row["page_type"],
                "width": width,
                "height": height,
                "image": str(image_relative) if not metadata_only else None,
                "label_file": str(label_relative),
                "annotations": annotations,
                "hard_negative_boxes": negative_boxes,
            }
            encoded = (
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            record_file.write(encoded.decode())
            digest.update(encoded)
            counts["records"] += 1
            counts["annotations"] += len(annotations)
            counts["hard_negative_boxes"] += len(negative_boxes)
            if not annotations:
                counts["negative_images"] += 1
            companies[str(row["company"])] += 1
            page_types[str(row["page_type"])] += 1

    return {
        **counts,
        "class_counts": dict(sorted(class_counts.items())),
        "source_dispositions": dict(sorted(source_counts.items())),
        "companies": dict(sorted(companies.items())),
        "page_types": dict(sorted(page_types.items())),
        "records_sha256": digest.hexdigest(),
        "records_path": str(record_path.relative_to(root)),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.force:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --force")
        shutil.rmtree(args.output_dir)
    temporary = args.output_dir.with_name(
        f".{args.output_dir.name}.partial-{os.getpid()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        splits = {
            "train": prepare_split(
                temporary,
                "train",
                args.max_train,
                metadata_only=args.metadata_only,
                fail_on_unmapped=args.fail_on_unmapped,
                download_workers=getattr(args, "download_workers", 8),
            ),
            "test": prepare_split(
                temporary,
                "test",
                args.max_test,
                metadata_only=args.metadata_only,
                fail_on_unmapped=args.fail_on_unmapped,
                download_workers=getattr(args, "download_workers", 8),
            ),
        }
        dataset_yaml = [
            f"path: {temporary.resolve()}",
            "train: images/train",
            "test: images/test",
            f"nc: {len(DETECTOR_CLASSES)}",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(DETECTOR_CLASSES)],
            "",
        ]
        (temporary / "dataset.yaml").write_text(
            "\n".join(dataset_yaml), encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "dataset": WEBPII_ID,
            "revision": WEBPII_REVISION,
            "license": "Apache-2.0",
            "published_splits_preserved": True,
            "transport": "pinned-revision-datasets-server-row-stream",
            "test_used_for_checkpoint_selection": False,
            "metadata_only": args.metadata_only,
            "classes": list(DETECTOR_CLASSES),
            "source_mapping_version": SOURCE_MAPPING_VERSION,
            "splits": splits,
            "unmapped_source_labels_are_reported": True,
            "unmapped_source_labels_fail_closed": args.fail_on_unmapped,
            "values_persisted": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        if args.output_dir.exists():
            args.output_dir.rmdir()
        temporary.rename(args.output_dir)
        # The absolute training root changes after the atomic rename.
        yaml_path = args.output_dir / "dataset.yaml"
        yaml_path.write_text(
            yaml_path.read_text(encoding="utf-8").replace(
                str(temporary.resolve()), str(args.output_dir.resolve()), 1
            ),
            encoding="utf-8",
        )
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the pinned WebPII detector dataset"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--fail-on-unmapped", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(prepare(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
