from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from training.schema import sha256_file
from training.visual.prepare_webpii import DETECTOR_CLASSES, clamp_box, yolo_line


DATASET_NAME = "PLVA user-provided WebPII-format ATS synthetic"
SOURCE_MAPPING_VERSION = 1

# This mapping is intentionally exhaustive for the supplied generator.  A new
# key is a schema change and must be reviewed rather than silently relabeled.
PII_SOURCE_MAPPING: dict[str, str | None] = {
    "PII_FULLNAME": "NAME",
    "PII_FULLNAME2": "NAME",
    "PII_EMAIL": "EMAIL",
    "PII_EMAIL2": "EMAIL",
    "PII_EMAIL3": "EMAIL",
    "PII_EMAIL4": "EMAIL",
    "PII_PHONE": "PHONE",
    "PII_CITY_STATE": "ADDRESS",
    "PII_STREET": "ADDRESS",
    "PII_LINKEDIN": "SENSITIVE_FIELD",
    "PII_RESUME_FILENAME": "SENSITIVE_FIELD",
    "PII_SALARY": "SENSITIVE_FIELD",
    "PII_START_DATE": "SENSITIVE_FIELD",
    "PII_US_BASED": "SENSITIVE_FIELD",
    "PII_VISA_SPONSORSHIP": "SENSITIVE_FIELD",
    "PII_VOICE_FILENAME": "SENSITIVE_FIELD",
    "PII_WEBSITE": "SENSITIVE_FIELD",
    "PII_PRONOUNS": "SENSITIVE_FIELD",
    # Company names are contextual labels/employers in this source and are not
    # user PII under the PLVA detector contract.
    "PII_COMPANY": None,
}

EXPLICIT_HARD_NEGATIVE_KEYS = {
    "PII_COMPANY",
    "MISC_APPLIED_BEFORE",
    "MISC_COMPANY",
    "MISC_NOTICE_PERIOD",
    "MISC_SOURCE",
}

ELEMENT_FIELDS = {
    "pii_elements_json": "num_pii_elements",
    "misc_elements_json": "num_misc_elements",
    "product_elements_json": "num_product_elements",
    "order_elements_json": "num_order_elements",
    "search_elements_json": "num_search_elements",
}

IDENTITY_MARKER_KEYS = {"PII_EMAIL", "PII_EMAIL2", "PII_EMAIL3", "PII_EMAIL4", "PII_PHONE"}

CONTENT_HASH_ALGORITHM = {
    "name": "sha256-sorted-path-nul-file-sha256-lf-v1",
    "path_format": "relative-posix",
    "path_encoding": "utf-8",
    "ordering": "ascending-utf8-path-bytes",
    "entry_format": "<relative-path>\\0<lowercase-file-sha256>\\n",
}


def source_disposition(source_key: str) -> tuple[str, str | None]:
    if source_key not in PII_SOURCE_MAPPING:
        if source_key in EXPLICIT_HARD_NEGATIVE_KEYS:
            return "hard-negative", None
        return "unmapped", None
    class_name = PII_SOURCE_MAPPING[source_key]
    return ("map", class_name) if class_name is not None else ("hard-negative", None)


def aggregate_content_hash(fingerprints: list[tuple[str, str]]) -> str:
    """Hash canonical path/hash entries independently of metadata row order."""
    digest = hashlib.sha256()
    for relative, file_sha in sorted(fingerprints, key=lambda item: item[0].encode("utf-8")):
        if "\0" in relative or "\n" in relative:
            raise ValueError("content-hash paths cannot contain NUL or newline")
        if not re.fullmatch(r"[0-9a-f]{64}", file_sha):
            raise ValueError("content-hash file digest must be lowercase SHA-256")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _safe_stem(source_id: str, variant: str) -> str:
    raw = f"{source_id}:{variant}"
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._") or "record"
    return f"ats-{clean[:90]}-{hashlib.sha256(raw.encode()).hexdigest()[:10]}"


def _parse_elements(row: dict[str, Any], field: str, line_number: int) -> list[dict[str, Any]]:
    try:
        elements = json.loads(row[field])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"metadata line {line_number}: invalid {field}") from exc
    if not isinstance(elements, list) or not all(isinstance(item, dict) for item in elements):
        raise ValueError(f"metadata line {line_number}: {field} must be a JSON list of objects")
    count_field = ELEMENT_FIELDS[field]
    if row.get(count_field) != len(elements):
        raise ValueError(
            f"metadata line {line_number}: {count_field} does not match {field}"
        )
    return elements


def _validated_box(
    element: dict[str, Any],
    width: int,
    height: int,
    *,
    line_number: int,
) -> list[float] | None:
    source_key = str(element.get("key", "<missing>"))
    for key in ("bbox_x", "bbox_y", "bbox_width", "bbox_height"):
        value = element.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"metadata line {line_number}: {source_key} has invalid {key}")
        if not math.isfinite(float(value)):
            raise ValueError(f"metadata line {line_number}: {source_key} has non-finite {key}")
    if float(element["bbox_width"]) <= 0 or float(element["bbox_height"]) <= 0:
        raise ValueError(f"metadata line {line_number}: {source_key} has a non-positive box")
    if type(element.get("visible")) is not bool or type(element.get("clipped")) is not bool:
        raise ValueError(
            f"metadata line {line_number}: {source_key} visible/clipped must be booleans"
        )

    raw_x1 = float(element["bbox_x"])
    raw_y1 = float(element["bbox_y"])
    raw_x2 = raw_x1 + float(element["bbox_width"])
    raw_y2 = raw_y1 + float(element["bbox_height"])
    intersects = raw_x2 > 0 and raw_y2 > 0 and raw_x1 < width and raw_y1 < height
    extends_outside = raw_x1 < 0 or raw_y1 < 0 or raw_x2 > width or raw_y2 > height

    if element["visible"]:
        if not intersects:
            raise ValueError(
                f"metadata line {line_number}: visible {source_key} does not intersect the image"
            )
        if element["clipped"] != extends_outside:
            raise ValueError(
                f"metadata line {line_number}: {source_key} clipped flag disagrees with its box"
            )
        box = clamp_box(element, width, height)
        if box is None:
            raise ValueError(f"metadata line {line_number}: visible {source_key} clamps empty")
        return box

    if intersects:
        raise ValueError(
            f"metadata line {line_number}: invisible {source_key} still intersects the image"
        )
    if element["clipped"]:
        raise ValueError(
            f"metadata line {line_number}: off-screen {source_key} cannot be marked clipped"
        )
    return None


def _metadata_rows(metadata_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with metadata_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"metadata line {line_number}: blank lines are not allowed")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"metadata line {line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"metadata line {line_number}: row must be an object")
            yield line_number, row


def _provenance_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _safe_source_artifact(source_root: Path, relative: str, label: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"source manifest has an unsafe {label} path")
    resolved = (source_root / path).resolve()
    if source_root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"source manifest {label} artifact is missing")
    return resolved


def _load_source_safety_manifest(
    source_root: Path, metadata_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = source_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("supplemental source requires a v2 manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError("supplemental source manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("supplemental source manifest must be an object")

    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version < 2:
        raise ValueError("supplemental source manifest schema_version must be >= 2")
    if type(manifest.get("failures")) is not int or manifest["failures"] != 0:
        raise ValueError("supplemental source manifest failures must equal 0")

    audit = manifest.get("audit")
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise ValueError("supplemental source audit must be passed")
    if audit.get("identity_disjoint") is not True:
        raise ValueError("supplemental source audit must verify identity disjointness")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or splits.get("identity_disjoint") is not True:
        raise ValueError("supplemental source manifest must declare identity-disjoint splits")

    capture = manifest.get("capture")
    if not isinstance(capture, dict) or capture.get("mode") != "tiles":
        raise ValueError("supplemental source capture mode must be tiles")
    if capture.get("full_page_downsampling_avoided") is not True:
        raise ValueError("supplemental source must avoid full-page downsampling")

    label_contract = manifest.get("label_policy")
    if not isinstance(label_contract, dict):
        raise ValueError("supplemental source label_policy is missing")
    if label_contract.get("fail_on_unmapped") is not True:
        raise ValueError("supplemental source label_policy must fail_on_unmapped")
    if label_contract.get("company_is_hard_negative") is not True:
        raise ValueError("supplemental source label_policy must hard-negative company")
    label_relative = label_contract.get("path")
    if not isinstance(label_relative, str) or not label_relative:
        raise ValueError("supplemental source label_policy path is missing")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("supplemental source artifact hashes are missing")
    metadata_artifact = artifacts.get("metadata.jsonl")
    if not isinstance(metadata_artifact, dict):
        raise ValueError("supplemental source metadata artifact hash is missing")
    actual_metadata_sha = sha256_file(metadata_path)
    if metadata_artifact.get("sha256") != actual_metadata_sha:
        raise ValueError("supplemental source metadata hash does not match manifest")
    if audit.get("metadata_sha256") != actual_metadata_sha:
        raise ValueError("supplemental source metadata hash does not match audit")

    label_path = _safe_source_artifact(source_root, label_relative, "label policy")
    label_artifact = artifacts.get(label_relative)
    if not isinstance(label_artifact, dict) or label_artifact.get("sha256") != sha256_file(
        label_path
    ):
        raise ValueError("supplemental source label-policy hash does not match manifest")
    try:
        source_policy = json.loads(label_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError("supplemental source label policy is not valid JSON") from exc
    if not isinstance(source_policy, dict) or source_policy.get("fail_on_unmapped") is not True:
        raise ValueError("supplemental source label policy must fail_on_unmapped")
    company = source_policy.get("source_dispositions", {}).get("MISC_COMPANY")
    if not isinstance(company, dict) or company.get("disposition") != "hard-negative":
        raise ValueError("supplemental source policy must map MISC_COMPANY hard-negative")

    safety = {
        "schema_version": schema_version,
        "failures": 0,
        "audit_passed": True,
        "capture_mode": "tiles",
        "full_page_downsampling_avoided": True,
        "identity_disjoint": True,
        "fail_on_unmapped": True,
        "metadata_sha256": actual_metadata_sha,
        "source_manifest_sha256": sha256_file(manifest_path),
        "label_policy_sha256": sha256_file(label_path),
    }
    provenance = {
        "manifest": _provenance_file(manifest_path),
        "label_policy": _provenance_file(label_path),
    }
    return manifest, source_policy, safety | provenance


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    metadata_path = source_root / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"supplemental metadata is missing: {metadata_path}")
    source_manifest, source_label_policy, source_safety = _load_source_safety_manifest(
        source_root, metadata_path
    )
    if output_dir == source_root or source_root in output_dir.parents:
        raise ValueError("output directory must be outside the immutable source dataset")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            raise FileExistsError(f"{output_dir} is not empty; pass --force")
        shutil.rmtree(output_dir)

    temporary = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    split_identities: dict[str, set[str]] = defaultdict(set)
    identity_split: dict[str, str] = {}
    identity_markers: dict[str, set[str]] = defaultdict(set)
    seen_records: set[tuple[str, str]] = set()
    seen_images: set[str] = set()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_counts: Counter[str] = Counter()
    layout_counts: Counter[str] = Counter()
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    output_image_fingerprints: list[tuple[str, str]] = []
    output_label_fingerprints: list[tuple[str, str]] = []
    source_image_fingerprints: list[tuple[str, str]] = []

    try:
        from PIL import Image

        for line_number, row in _metadata_rows(metadata_path):
            split = row.get("split")
            if split not in {"train", "test"}:
                raise ValueError(f"metadata line {line_number}: split must be train or test")
            if row.get("capture_mode") != "tile":
                raise ValueError(
                    f"metadata line {line_number}: v2 training rows must use tile capture"
                )
            source_id = str(row.get("source_id", "")).strip()
            variant = str(row.get("variant", "")).strip()
            if not source_id or not variant:
                raise ValueError(f"metadata line {line_number}: source_id and variant are required")
            record_key = (source_id, variant)
            if record_key in seen_records:
                raise ValueError(f"metadata line {line_number}: duplicate source_id/variant {record_key}")
            seen_records.add(record_key)
            previous_split = identity_split.setdefault(source_id, split)
            if previous_split != split:
                raise ValueError(f"identity {source_id} occurs in both train and test")
            split_identities[split].add(source_id)

            width, height = row.get("image_width"), row.get("image_height")
            if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
                raise ValueError(f"metadata line {line_number}: invalid image dimensions")
            image_value = row.get("image")
            if not isinstance(image_value, str) or not image_value:
                raise ValueError(f"metadata line {line_number}: image must be a relative path")
            image_relative = Path(image_value)
            if image_relative.is_absolute() or ".." in image_relative.parts:
                raise ValueError(f"metadata line {line_number}: unsafe image path")
            source_image = (source_root / image_relative).resolve()
            if source_root not in source_image.parents or not source_image.is_file():
                raise ValueError(f"metadata line {line_number}: image is outside source root or missing")
            if str(source_image) in seen_images:
                raise ValueError(f"metadata line {line_number}: image is referenced more than once")
            seen_images.add(str(source_image))
            with Image.open(source_image) as image:
                if image.format != "PNG" or image.size != (width, height):
                    raise ValueError(
                        f"metadata line {line_number}: PNG dimensions disagree with metadata"
                    )
                image.verify()
            source_image_sha = sha256_file(source_image)
            if row.get("image_sha256") != source_image_sha:
                raise ValueError(
                    f"metadata line {line_number}: image hash disagrees with metadata"
                )
            source_image_fingerprints.append((image_relative.as_posix(), source_image_sha))

            annotations: list[dict[str, Any]] = []
            annotation_indices: dict[tuple[Any, ...], int] = {}
            hard_negatives: list[dict[str, Any]] = []
            for field in ELEMENT_FIELDS:
                for element in _parse_elements(row, field, line_number):
                    source_key = element.get("key")
                    if not isinstance(source_key, str) or not source_key:
                        raise ValueError(f"metadata line {line_number}: element key is required")
                    if not isinstance(element.get("value"), str):
                        raise ValueError(
                            f"metadata line {line_number}: {source_key} value must be text"
                        )
                    disposition, class_name = source_disposition(source_key)
                    if disposition == "unmapped":
                        raise ValueError(
                            f"metadata line {line_number}: unmapped supplemental key {source_key}"
                        )
                    policy_entry = source_label_policy.get("source_dispositions", {}).get(
                        source_key
                    )
                    if (
                        not isinstance(policy_entry, dict)
                        or policy_entry.get("disposition") != disposition
                        or policy_entry.get("class") != class_name
                    ):
                        raise ValueError(
                            f"metadata line {line_number}: source policy disagrees for {source_key}"
                        )
                    if field != "pii_elements_json" and disposition != "hard-negative":
                        raise ValueError(
                            f"metadata line {line_number}: {source_key} outside pii_elements_json must be a hard negative"
                        )
                    box = _validated_box(
                        element,
                        width,
                        height,
                        line_number=line_number,
                    )
                    source_counts[f"{disposition}:{source_key}"] += 1
                    if source_key in IDENTITY_MARKER_KEYS and element["value"].strip():
                        identity_markers[split].add(element["value"].strip().casefold())
                    if box is None:
                        split_counts[split]["offscreen_elements"] += 1
                        continue
                    sanitized = {
                        "source_key": source_key,
                        "element_type": element.get("element_type"),
                        "bbox_xyxy": box,
                        "clipped": bool(element["clipped"]),
                    }
                    if disposition == "map" and class_name is not None:
                        identity = (class_name, *(round(value, 4) for value in box))
                        if identity in annotation_indices:
                            existing = annotations[annotation_indices[identity]]
                            existing.setdefault("source_keys", [existing["source_key"]])
                            existing["source_keys"].append(source_key)
                            existing["clipped"] = bool(existing["clipped"] or element["clipped"])
                            split_counts[split]["duplicate_annotations_merged"] += 1
                        else:
                            sanitized["class"] = class_name
                            annotation_indices[identity] = len(annotations)
                            annotations.append(sanitized)
                            class_counts[split][class_name] += 1
                    else:
                        hard_negatives.append(
                            sanitized | {"reason": "explicit-non-pii-lookalike"}
                        )

            stem = _safe_stem(source_id, variant)
            output_image_relative = Path("images") / split / f"{stem}.png"
            output_label_relative = Path("labels") / split / f"{stem}.txt"
            output_image = temporary / output_image_relative
            output_label = temporary / output_label_relative
            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, output_image)
            label_lines = [
                yolo_line(item["class"], item["bbox_xyxy"], width, height)
                for item in annotations
            ]
            output_label.write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )
            image_hash = sha256_file(output_image)
            if image_hash != source_image_sha:
                raise ValueError(f"copied image hash changed for {image_relative}")
            label_hash = sha256_file(output_label)
            output_image_fingerprints.append(
                (output_image_relative.as_posix(), image_hash)
            )
            output_label_fingerprints.append(
                (output_label_relative.as_posix(), label_hash)
            )
            record = {
                "id": stem,
                "source_id": source_id,
                "variant": variant,
                "split": split,
                "width": width,
                "height": height,
                "image": output_image_relative.as_posix(),
                "image_sha256": image_hash,
                "label_file": output_label_relative.as_posix(),
                "label_sha256": label_hash,
                "annotations": annotations,
                "hard_negative_boxes": hard_negatives,
            }
            records[split].append(record)
            split_counts[split]["records"] += 1
            split_counts[split]["annotations"] += len(annotations)
            split_counts[split]["hard_negative_boxes"] += len(hard_negatives)
            if not annotations:
                split_counts[split]["negative_images"] += 1
            layout_counts[f"{row.get('company')}:{row.get('page_type')}"] += 1

        identity_overlap = split_identities["train"] & split_identities["test"]
        marker_overlap = identity_markers["train"] & identity_markers["test"]
        if identity_overlap:
            raise ValueError("supplemental source_id identity leakage detected")
        if marker_overlap:
            raise ValueError("supplemental email/phone identity-marker leakage detected")
        if not records["train"] or not records["test"]:
            raise ValueError("supplemental dataset must contain both train and test identities")
        audit = source_manifest["audit"]
        generation = source_manifest.get("generation", {})
        actual_split_counts = {
            split: split_counts[split]["records"] for split in ("train", "test")
        }
        if audit.get("records") != len(seen_records):
            raise ValueError("supplemental source audit record count disagrees with metadata")
        if generation.get("rendered_images") != len(seen_records):
            raise ValueError("supplemental source rendered-image count disagrees with metadata")
        if audit.get("split_counts") != actual_split_counts:
            raise ValueError("supplemental source audit split counts disagree with metadata")
        if audit.get("identities") != len(identity_split):
            raise ValueError("supplemental source audit identity count disagrees with metadata")
        source_image_set = hashlib.sha256()
        for relative, image_sha in sorted(source_image_fingerprints):
            source_image_set.update(f"{relative} {image_sha}\n".encode())
        if source_manifest.get("image_set_sha256") != source_image_set.hexdigest():
            raise ValueError("supplemental source image-set fingerprint does not match manifest")

        split_manifest: dict[str, Any] = {}
        for split in ("train", "test"):
            record_path = temporary / "records" / f"{split}.jsonl"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            with record_path.open("w", encoding="utf-8") as destination:
                for record in records[split]:
                    destination.write(
                        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    )
            split_manifest[split] = {
                **dict(split_counts[split]),
                "identities": len(split_identities[split]),
                "class_counts": dict(sorted(class_counts[split].items())),
                "records_path": str(record_path.relative_to(temporary)),
                "records_sha256": sha256_file(record_path),
            }

        generator = _provenance_file(source_root / "_generator_source.py")
        generator_readme = _provenance_file(source_root.parent / "README.md")
        source_template = _provenance_file(source_root / "_template_stripped.html")
        manifest = {
            "schema_version": 2,
            "dataset": DATASET_NAME,
            "license": "user-provided-synthetic-data",
            "supplemental_only": True,
            "training_use": "train-split-only",
            "test_used_for_checkpoint_selection": False,
            "classes": list(DETECTOR_CLASSES),
            "source_mapping_version": SOURCE_MAPPING_VERSION,
            "source_mapping": {
                key: (class_name or "HARD_NEGATIVE")
                for key, class_name in sorted(PII_SOURCE_MAPPING.items())
            }
            | {
                key: "HARD_NEGATIVE"
                for key in sorted(EXPLICIT_HARD_NEGATIVE_KEYS - PII_SOURCE_MAPPING.keys())
            },
            "unmapped_source_labels_fail_closed": True,
            "values_persisted": False,
            "source": {
                "kind": "user-provided-local-WebPII-format-synthetic",
                "root": str(source_root),
                "safety_verification": source_safety,
                "metadata": {
                    "path": str(metadata_path),
                    "bytes": metadata_path.stat().st_size,
                    "sha256": sha256_file(metadata_path),
                },
                "generator": generator,
                "generator_readme": generator_readme,
                "template": source_template,
            },
            "identity_split": {
                "key": "source_id",
                "verified": True,
                "source_id_overlap": 0,
                "identity_marker_keys_checked": sorted(IDENTITY_MARKER_KEYS),
                "identity_marker_overlap": 0,
            },
            "layout_counts": dict(sorted(layout_counts.items())),
            "source_dispositions": dict(sorted(source_counts.items())),
            "splits": split_manifest,
            "content_hashes": {
                "images_aggregate_sha256": aggregate_content_hash(
                    output_image_fingerprints
                ),
                "labels_aggregate_sha256": aggregate_content_hash(
                    output_label_fingerprints
                ),
            },
            "content_hash_algorithm": CONTENT_HASH_ALGORITHM,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            output_dir.rmdir()
        temporary.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and convert the user-provided WebPII-format ATS dataset"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(prepare(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
