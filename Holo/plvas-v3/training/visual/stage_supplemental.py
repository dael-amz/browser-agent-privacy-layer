from __future__ import annotations

import json
import math
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.schema import sha256_file
from training.visual.prepare_supplemental import (
    CONTENT_HASH_ALGORITHM,
    aggregate_content_hash,
)
from training.visual.prepare_webpii import DETECTOR_CLASSES


REQUIRED_MANIFEST_FLAGS: dict[str, Any] = {
    "supplemental_only": True,
    "training_use": "train-split-only",
    "test_used_for_checkpoint_selection": False,
    "values_persisted": False,
    "unmapped_source_labels_fail_closed": True,
}


def _load_object(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is missing or is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _safe_file(root: Path, relative_value: Any, description: str) -> tuple[Path, str]:
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError(f"{description} must be a non-empty relative path")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{description} contains an unsafe path: {relative_value}")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is missing or is not a regular file: {path}")
    root_resolved = root.resolve()
    resolved = path.resolve()
    if root_resolved not in resolved.parents:
        raise ValueError(
            f"{description} escapes the supplemental root: {relative_value}"
        )
    return path, relative.as_posix()


def _reject_persisted_values(value: Any, context: str) -> None:
    if isinstance(value, dict):
        if "value" in value:
            raise ValueError(f"{context} persists a raw value field")
        for key, child in value.items():
            _reject_persisted_values(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_persisted_values(child, f"{context}[{index}]")


def _validate_box(box: Any, width: int, height: int, context: str) -> None:
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"{context} bbox_xyxy must contain four numbers")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in box
    ):
        raise ValueError(f"{context} bbox_xyxy contains an invalid number")
    x1, y1, x2, y2 = (float(value) for value in box)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{context} bbox_xyxy is outside the image")


def _validate_png(path: Path, width: int, height: int, context: str) -> None:
    from PIL import Image

    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (width, height):
                raise ValueError(
                    f"{context} PNG format or dimensions disagree with its record"
                )
            image.verify()
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"{context} is not a valid PNG") from exc


def _validate_yolo_label(
    path: Path,
    annotations: list[dict[str, Any]],
    context: str,
) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{context} label is not valid UTF-8 text") from exc
    if len(lines) != len(annotations):
        raise ValueError(f"{context} label count does not match record annotations")
    for index, (line, annotation) in enumerate(zip(lines, annotations, strict=True)):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{context} label line {index + 1} is not YOLO format")
        try:
            class_index = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(
                f"{context} label line {index + 1} is not numeric"
            ) from exc
        expected_class = DETECTOR_CLASSES.index(annotation["class"])
        if class_index != expected_class:
            raise ValueError(f"{context} label class disagrees with its record")
        if any(not math.isfinite(value) for value in coordinates):
            raise ValueError(f"{context} label contains a non-finite coordinate")
        center_x, center_y, box_width, box_height = coordinates
        if not (
            0 <= center_x <= 1
            and 0 <= center_y <= 1
            and 0 < box_width <= 1
            and 0 < box_height <= 1
        ):
            raise ValueError(f"{context} label contains an out-of-range coordinate")


def _actual_files(root: Path, directory: str) -> set[str]:
    base = root / directory
    if base.is_symlink() or not base.is_dir():
        raise ValueError(f"prepared supplemental directory is missing: {base}")
    files: set[str] = set()
    for path in base.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                f"prepared supplemental data may not contain symlinks: {path}"
            )
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def validate_prepared_supplemental(root: Path) -> dict[str, Any]:
    """Validate an immutable output from ``prepare_supplemental``.

    This deliberately re-hashes every artifact.  A manifest alone is not enough
    evidence that a large Modal volume upload completed without truncation.
    """

    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"prepared supplemental root is missing: {root}")
    manifest_path = root / "manifest.json"
    manifest = _load_object(manifest_path, "prepared supplemental manifest")
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version < 2:
        raise ValueError(
            "prepared supplemental manifest must come from the verified v2 adapter"
        )
    for key, expected in REQUIRED_MANIFEST_FLAGS.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"prepared supplemental manifest requires {key}={expected!r}"
            )
    if manifest.get("classes") != list(DETECTOR_CLASSES):
        raise ValueError(
            "prepared supplemental class order differs from the detector contract"
        )
    identity = manifest.get("identity_split")
    if not isinstance(identity, dict) or identity.get("verified") is not True:
        raise ValueError("prepared supplemental identity split is not verified")
    if identity.get("source_id_overlap") != 0:
        raise ValueError("prepared supplemental source_id identity leakage detected")
    if identity.get("identity_marker_overlap") != 0:
        raise ValueError("prepared supplemental identity-marker leakage detected")
    safety = manifest.get("source", {}).get("safety_verification", {})
    required_safety = {
        "audit_passed": True,
        "failures": 0,
        "capture_mode": "tiles",
        "full_page_downsampling_avoided": True,
        "identity_disjoint": True,
        "fail_on_unmapped": True,
    }
    if not isinstance(safety, dict) or any(
        safety.get(key) != expected for key, expected in required_safety.items()
    ):
        raise ValueError(
            "prepared supplemental v2 source-safety verification is incomplete"
        )
    for key in ("metadata_sha256", "source_manifest_sha256", "label_policy_sha256"):
        digest = safety.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"prepared supplemental v2 safety verification lacks a valid {key}"
            )

    split_manifest = manifest.get("splits")
    if not isinstance(split_manifest, dict) or set(split_manifest) != {"train", "test"}:
        raise ValueError(
            "prepared supplemental manifest must contain only train and test splits"
        )

    referenced_images: set[str] = set()
    referenced_labels: set[str] = set()
    referenced_records: set[str] = set()
    seen_ids: set[str] = set()
    identities: dict[str, set[str]] = {"train": set(), "test": set()}
    image_fingerprints: list[tuple[str, str]] = []
    label_fingerprints: list[tuple[str, str]] = []
    measured: dict[str, dict[str, Any]] = {}

    for split in ("train", "test"):
        split_info = split_manifest[split]
        if not isinstance(split_info, dict):
            raise ValueError(
                f"prepared supplemental {split} split manifest must be an object"
            )
        expected_record_relative = f"records/{split}.jsonl"
        if split_info.get("records_path") != expected_record_relative:
            raise ValueError(
                f"prepared supplemental {split} records_path must be {expected_record_relative}"
            )
        records_path, records_relative = _safe_file(
            root,
            split_info.get("records_path"),
            f"prepared supplemental {split} records",
        )
        referenced_records.add(records_relative)
        records_hash = sha256_file(records_path)
        if split_info.get("records_sha256") != records_hash:
            raise ValueError(f"prepared supplemental {split} records SHA-256 mismatch")

        counters: Counter[str] = Counter()
        classes: Counter[str] = Counter()
        with records_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise ValueError(
                        f"prepared supplemental {split} records contain a blank line"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"prepared supplemental {split} record {line_number} is invalid JSON"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"prepared supplemental {split} record {line_number} is not an object"
                    )
                context = f"prepared supplemental {split} record {line_number}"
                _reject_persisted_values(record, context)
                if record.get("split") != split:
                    raise ValueError(f"{context} has the wrong split")
                record_id = record.get("id")
                source_id = record.get("source_id")
                if (
                    not isinstance(record_id, str)
                    or not record_id
                    or record_id in seen_ids
                ):
                    raise ValueError(f"{context} has a missing or duplicate id")
                if not isinstance(source_id, str) or not source_id:
                    raise ValueError(f"{context} has a missing source_id")
                seen_ids.add(record_id)
                identities[split].add(source_id)
                width, height = record.get("width"), record.get("height")
                if (
                    type(width) is not int
                    or type(height) is not int
                    or width <= 0
                    or height <= 0
                ):
                    raise ValueError(f"{context} has invalid image dimensions")

                image_path, image_relative = _safe_file(
                    root, record.get("image"), f"{context} image"
                )
                label_path, label_relative = _safe_file(
                    root, record.get("label_file"), f"{context} label"
                )
                if not image_relative.startswith(
                    f"images/{split}/"
                ) or not image_relative.endswith(".png"):
                    raise ValueError(f"{context} image is outside images/{split}")
                if not label_relative.startswith(
                    f"labels/{split}/"
                ) or not label_relative.endswith(".txt"):
                    raise ValueError(f"{context} label is outside labels/{split}")
                if (
                    image_relative in referenced_images
                    or label_relative in referenced_labels
                ):
                    raise ValueError(f"{context} reuses an image or label file")
                referenced_images.add(image_relative)
                referenced_labels.add(label_relative)

                image_hash = sha256_file(image_path)
                label_hash = sha256_file(label_path)
                if record.get("image_sha256") != image_hash:
                    raise ValueError(f"{context} image SHA-256 mismatch")
                if record.get("label_sha256") != label_hash:
                    raise ValueError(f"{context} label SHA-256 mismatch")
                _validate_png(image_path, width, height, context)

                annotations = record.get("annotations")
                hard_negatives = record.get("hard_negative_boxes")
                if not isinstance(annotations, list) or not all(
                    isinstance(item, dict) for item in annotations
                ):
                    raise ValueError(f"{context} annotations must be a list of objects")
                if not isinstance(hard_negatives, list) or not all(
                    isinstance(item, dict) for item in hard_negatives
                ):
                    raise ValueError(
                        f"{context} hard_negative_boxes must be a list of objects"
                    )
                for item_index, annotation in enumerate(annotations):
                    class_name = annotation.get("class")
                    if class_name not in DETECTOR_CLASSES:
                        raise ValueError(
                            f"{context} annotation has an unknown detector class"
                        )
                    _validate_box(
                        annotation.get("bbox_xyxy"),
                        width,
                        height,
                        f"{context} annotation {item_index}",
                    )
                    if type(annotation.get("clipped")) is not bool:
                        raise ValueError(
                            f"{context} annotation clipped must be boolean"
                        )
                    classes[class_name] += 1
                for item_index, negative in enumerate(hard_negatives):
                    _validate_box(
                        negative.get("bbox_xyxy"),
                        width,
                        height,
                        f"{context} hard negative {item_index}",
                    )
                    if negative.get("reason") != "explicit-non-pii-lookalike":
                        raise ValueError(
                            f"{context} hard negative lacks an explicit reason"
                        )
                _validate_yolo_label(label_path, annotations, context)

                image_fingerprints.append((image_relative, image_hash))
                label_fingerprints.append((label_relative, label_hash))
                counters["records"] += 1
                counters["annotations"] += len(annotations)
                counters["hard_negative_boxes"] += len(hard_negatives)
                if not annotations:
                    counters["negative_images"] += 1

        if counters["records"] <= 0:
            raise ValueError(f"prepared supplemental {split} split is empty")
        expected_counts = {
            "records": counters["records"],
            "annotations": counters["annotations"],
            "hard_negative_boxes": counters["hard_negative_boxes"],
            "negative_images": counters["negative_images"],
            "identities": len(identities[split]),
            "class_counts": dict(sorted(classes.items())),
        }
        for key, value in expected_counts.items():
            if split_info.get(key) != value:
                raise ValueError(
                    f"prepared supplemental {split} manifest {key} disagrees with records"
                )
        measured[split] = expected_counts | {
            "records_sha256": records_hash,
        }

    if identities["train"] & identities["test"]:
        raise ValueError("prepared supplemental records leak source_id across splits")
    actual_records = _actual_files(root, "records")
    actual_images = _actual_files(root, "images")
    actual_labels = _actual_files(root, "labels")
    if actual_records != referenced_records:
        raise ValueError(
            "prepared supplemental records directory contains missing or untracked files"
        )
    if actual_images != referenced_images:
        raise ValueError(
            "prepared supplemental images directory contains missing or untracked files"
        )
    if actual_labels != referenced_labels:
        raise ValueError(
            "prepared supplemental labels directory contains missing or untracked files"
        )

    content_hashes = manifest.get("content_hashes")
    if not isinstance(content_hashes, dict):
        raise ValueError("prepared supplemental manifest content_hashes is missing")
    if manifest.get("content_hash_algorithm") != CONTENT_HASH_ALGORITHM:
        raise ValueError("prepared supplemental content-hash algorithm is unsupported")
    declared_content_hashes: dict[str, str] = {}
    for key in ("images_aggregate_sha256", "labels_aggregate_sha256"):
        digest = content_hashes.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"prepared supplemental manifest has an invalid {key}")
        declared_content_hashes[key] = digest

    verified_image_hash = aggregate_content_hash(image_fingerprints)
    verified_label_hash = aggregate_content_hash(label_fingerprints)
    if declared_content_hashes["images_aggregate_sha256"] != verified_image_hash:
        raise ValueError("prepared supplemental aggregate image SHA-256 mismatch")
    if declared_content_hashes["labels_aggregate_sha256"] != verified_label_hash:
        raise ValueError("prepared supplemental aggregate label SHA-256 mismatch")

    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "classes": list(DETECTOR_CLASSES),
        "splits": measured,
        "content_hashes": declared_content_hashes,
        "verified_content_hashes": {
            "algorithm": CONTENT_HASH_ALGORITHM,
            "images_sha256": verified_image_hash,
            "labels_sha256": verified_label_hash,
        },
        "test_excluded_from_training_and_selection": True,
    }


def stage_prepared_supplemental(
    source_root: Path,
    destination_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Validate and atomically copy a prepared supplement into a run."""

    source = validate_prepared_supplemental(source_root)
    if destination_root.exists() and not force:
        staged = validate_prepared_supplemental(destination_root)
        if staged["manifest_sha256"] != source["manifest_sha256"]:
            raise FileExistsError(
                "staged supplemental data differs from the uploaded input; pass force to replace it"
            )
        provenance_path = destination_root / "staging_provenance.json"
        provenance = _load_object(provenance_path, "supplemental staging provenance")
        if provenance.get("source_manifest_sha256") != source["manifest_sha256"]:
            raise ValueError(
                "supplemental staging provenance does not match the immutable uploaded input"
            )
        staged["staging_provenance"] = str(provenance_path)
        staged["staging_provenance_sha256"] = sha256_file(provenance_path)
        return {
            "mode": "reused-validated-staging",
            "uploaded_input": str(source_root),
            "staged_root": str(destination_root),
            "source": source,
            "staged": staged,
        }

    destination_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_root.with_name(
        f".{destination_root.name}.partial-{os.getpid()}"
    )
    backup = destination_root.with_name(
        f".{destination_root.name}.previous-{os.getpid()}"
    )
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    try:
        shutil.copytree(source_root, temporary, copy_function=shutil.copy2)
        copied = validate_prepared_supplemental(temporary)
        if copied["manifest_sha256"] != source["manifest_sha256"]:
            raise RuntimeError(
                "supplemental manifest changed while it was being staged"
            )
        if copied["verified_content_hashes"] != source["verified_content_hashes"]:
            raise RuntimeError("supplemental content changed while it was being staged")
        provenance = {
            "schema_version": 1,
            "staged_at": datetime.now(timezone.utc).isoformat(),
            "uploaded_input": str(source_root),
            "staged_root": str(destination_root),
            "source_manifest_sha256": source["manifest_sha256"],
            "content_hashes": source["content_hashes"],
            "verified_content_hashes": source["verified_content_hashes"],
            "copy_verified": True,
        }
        provenance_path = temporary / "staging_provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination_root.exists():
            destination_root.rename(backup)
        temporary.rename(destination_root)
        shutil.rmtree(backup, ignore_errors=True)
        staged = copied | {
            "manifest": str(destination_root / "manifest.json"),
            "staging_provenance": str(destination_root / "staging_provenance.json"),
            "staging_provenance_sha256": sha256_file(
                destination_root / "staging_provenance.json"
            ),
        }
        return {
            "mode": "copied-and-verified",
            "uploaded_input": str(source_root),
            "staged_root": str(destination_root),
            "source": source,
            "staged": staged,
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if backup.exists() and not destination_root.exists():
            backup.rename(destination_root)
        raise
