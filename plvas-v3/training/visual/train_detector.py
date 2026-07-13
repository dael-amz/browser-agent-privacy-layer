from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.visual.prepare_webpii import DETECTOR_CLASSES


SECRET_VISUAL_CLASSES = ("CARD_NUMBER", "CVC", "SECRET")
OTHER_SENSITIVE_CLASSES = tuple(
    name for name in DETECTOR_CLASSES if name not in SECRET_VISUAL_CLASSES
)
SYNTHETIC_VALIDATION_SOURCE = "screen-native-synthetic-validation"
PUBLISHED_TEST_SOURCE = "WebPII/test"
EXPECTED_SUPPLEMENTAL_TEST_RECORDS = 200
ULTRALYTICS_RUN_NAME = "ultralytics-run"
SAFETY_PROGRESS_NAME = "safety-selection-progress.json"
SEED_INCUMBENT_NAME = "seed-incumbent.json"
SELECTION_EPSILON = 1e-12
SAFETY_SELECTION_POLICY = {
    "schema_version": 1,
    "meaningful_secret_gain": 0.01,
    "maximum_individual_secret_drop": 0.01,
    "maximum_other_sensitive_mean_drop_from_incumbent": 0.03,
    "maximum_other_sensitive_mean_drop_from_high_water": 0.05,
    "maximum_individual_other_sensitive_drop": 0.05,
    "maximum_precision_mean_drop": 0.05,
    "tie_band_other_sensitive_mean_gain": 0.01,
    "precision_tiebreak_requires_no_recall_regression": True,
    "epsilon": SELECTION_EPSILON,
}
# Ultralytics' natural-image defaults include 50% horizontal flips, 100%
# mosaic, +/-50% scale, +/-10% translation, and high saturation/value jitter.
# Those transforms can mirror text, tile unrelated pages, or destroy the layout
# semantics of UI screenshots. Keep only small viewport-like motion and modest
# lighting/display variation. Every detection augmentation knob used by the
# pinned 8.4.92 data pipeline is explicit here so a framework default cannot
# silently re-enter training.
SCREENSHOT_AUGMENTATION_POLICY: dict[str, float | int | str] = {
    "hsv_h": 0.0,
    "hsv_s": 0.1,
    "hsv_v": 0.1,
    "degrees": 0.0,
    "translate": 0.02,
    "scale": 0.05,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "bgr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",  # Inert because copy_paste is zero.
    "close_mosaic": 0,
    "multi_scale": 0.0,
}

# Ultralytics 8.4.92 selects MuSGD for schedules over 10,000 iterations and
# triples the learning rate for model.23.cv3 classification-head parameters.
# That auto policy destabilized the warm-started detector as its default
# three-epoch warmup approached a 0.03 head LR. These settings are explicit and
# immutable across resume so framework defaults cannot silently re-enter.
STABLE_OPTIMIZER_POLICY: dict[str, float | int | str | bool] = {
    "optimizer": "AdamW",
    "lr0": 5e-4,
    "lrf": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "warmup_epochs": 1.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0,
    "cos_lr": True,
    "nbs": 64,
}

MONITORING_POLICY = {
    "automatic_stop": False,
    "map50_absolute_drop": 0.10,
    "validation_classification_loss_multiple": 2.0,
    "consecutive_epochs": 2,
    "action": "manual-review-and-stop; retry with lr0=0.00025",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_class_metrics(
    box: Any, names: dict[int, str]
) -> tuple[dict[int, float], dict[int, float], list[str]]:
    """Align Ultralytics' present-class metrics to the full detector taxonomy.

    Ultralytics omits classes with no validation labels from ``box.p`` and
    ``box.r``. ``box.ap_class_index`` is the authoritative mapping back to the
    dataset class IDs. Missing classes receive zero rather than inheriting a
    neighbouring class metric or crashing checkpoint selection.
    """
    normalized_names = {int(index): str(name) for index, name in names.items()}
    recalls = [float(value) for value in getattr(box, "r", [])]
    precisions = [float(value) for value in getattr(box, "p", [])]
    class_indices = [int(value) for value in getattr(box, "ap_class_index", [])]
    if len(recalls) != len(precisions) or len(recalls) != len(class_indices):
        raise RuntimeError(
            "Ultralytics per-class metrics cannot be aligned: p, r, and "
            "ap_class_index have different lengths"
        )
    unknown = sorted(set(class_indices) - set(normalized_names))
    if unknown:
        raise RuntimeError(
            f"Ultralytics reported metrics for unknown detector classes: {unknown}"
        )
    if len(set(class_indices)) != len(class_indices):
        raise RuntimeError(
            "Ultralytics reported duplicate detector class metric indices"
        )
    recall_by_class = {index: 0.0 for index in normalized_names}
    precision_by_class = {index: 0.0 for index in normalized_names}
    for position, class_index in enumerate(class_indices):
        recall_by_class[class_index] = recalls[position]
        precision_by_class[class_index] = precisions[position]
    reported = set(class_indices)
    missing = [
        normalized_names[index]
        for index in sorted(normalized_names)
        if index not in reported
    ]
    return recall_by_class, precision_by_class, missing


def _validated_metric_map(
    values: dict[str, float], context: str
) -> dict[str, float]:
    if not isinstance(values, dict) or set(values) != set(DETECTOR_CLASSES):
        raise ValueError(f"{context} must contain every detector class exactly once")
    normalized: dict[str, float] = {}
    for name in DETECTOR_CLASSES:
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{context} contains an invalid value for {name}")
        normalized[name] = float(value)
    return normalized


def summarize_safety_metrics(
    *,
    epoch: int,
    recall_by_class: dict[str, float],
    precision_by_class: dict[str, float],
) -> dict[str, Any]:
    recalls = _validated_metric_map(recall_by_class, "per-class recall")
    precisions = _validated_metric_map(precision_by_class, "per-class precision")
    return {
        "epoch": int(epoch),
        "per_class_recall": recalls,
        "per_class_precision": precisions,
        "secret_recall_minimum": min(
            recalls[name] for name in SECRET_VISUAL_CLASSES
        ),
        "other_sensitive_recall_mean": sum(
            recalls[name] for name in OTHER_SENSITIVE_CLASSES
        )
        / len(OTHER_SENSITIVE_CLASSES),
        "precision_mean": sum(precisions.values()) / len(precisions),
    }


def consider_safety_candidate(
    state: dict[str, Any] | None,
    *,
    epoch: int,
    recall_by_class: dict[str, float],
    precision_by_class: dict[str, float],
    missing_classes: list[str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply the constrained PLVA checkpoint rule without recall ratcheting."""
    invalid_reason = None
    if missing_classes:
        invalid_reason = "candidate-validation-missing-classes"
    try:
        candidate = summarize_safety_metrics(
            epoch=epoch,
            recall_by_class=recall_by_class,
            precision_by_class=precision_by_class,
        )
    except (TypeError, ValueError) as exc:
        candidate = None
        invalid_reason = f"candidate-validation-invalid: {exc}"
    if invalid_reason is not None or candidate is None:
        return (
            {
                "selected": False,
                "valid": False,
                "reason": invalid_reason,
                "candidate": candidate,
                "deltas": None,
                "gates": {},
            },
            state,
        )

    if state is None:
        initialized = {
            "schema_version": 1,
            "incumbent": candidate,
            "high_water": {
                "other_sensitive_recall_mean": candidate[
                    "other_sensitive_recall_mean"
                ]
            },
        }
        return (
            {
                "selected": True,
                "valid": True,
                "reason": "initial-frozen-seed-incumbent",
                "candidate": candidate,
                "deltas": None,
                "gates": {"complete_finite_all_class_metrics": True},
            },
            initialized,
        )

    incumbent = state.get("incumbent")
    high_water = state.get("high_water")
    if not isinstance(incumbent, dict) or not isinstance(high_water, dict):
        raise RuntimeError("safety selection state is malformed")
    # Revalidate persisted state before it can influence a new decision.
    incumbent = summarize_safety_metrics(
        epoch=int(incumbent.get("epoch")),
        recall_by_class=incumbent.get("per_class_recall"),
        precision_by_class=incumbent.get("per_class_precision"),
    )
    historical_other = high_water.get("other_sensitive_recall_mean")
    if (
        isinstance(historical_other, bool)
        or not isinstance(historical_other, (int, float))
        or not math.isfinite(float(historical_other))
        or not 0.0 <= float(historical_other) <= 1.0
    ):
        raise RuntimeError("safety selection high-water mark is malformed")
    historical_other = float(historical_other)

    secret_delta = (
        candidate["secret_recall_minimum"] - incumbent["secret_recall_minimum"]
    )
    other_delta = (
        candidate["other_sensitive_recall_mean"]
        - incumbent["other_sensitive_recall_mean"]
    )
    precision_delta = candidate["precision_mean"] - incumbent["precision_mean"]
    eps = SELECTION_EPSILON
    gates = {
        "individual_secret_drop": all(
            candidate["per_class_recall"][name]
            >= incumbent["per_class_recall"][name]
            - SAFETY_SELECTION_POLICY["maximum_individual_secret_drop"]
            - eps
            for name in SECRET_VISUAL_CLASSES
        ),
        "other_mean_drop_from_incumbent": candidate[
            "other_sensitive_recall_mean"
        ]
        >= incumbent["other_sensitive_recall_mean"]
        - SAFETY_SELECTION_POLICY[
            "maximum_other_sensitive_mean_drop_from_incumbent"
        ]
        - eps,
        "other_mean_drop_from_high_water": candidate[
            "other_sensitive_recall_mean"
        ]
        >= historical_other
        - SAFETY_SELECTION_POLICY[
            "maximum_other_sensitive_mean_drop_from_high_water"
        ]
        - eps,
        "individual_other_sensitive_drop": all(
            candidate["per_class_recall"][name]
            >= incumbent["per_class_recall"][name]
            - SAFETY_SELECTION_POLICY["maximum_individual_other_sensitive_drop"]
            - eps
            for name in OTHER_SENSITIVE_CLASSES
        ),
        "precision_mean_drop": candidate["precision_mean"]
        >= incumbent["precision_mean"]
        - SAFETY_SELECTION_POLICY["maximum_precision_mean_drop"]
        - eps,
    }
    hard_gates_passed = all(gates.values())
    all_recalls_nondecreasing = all(
        candidate["per_class_recall"][name]
        >= incumbent["per_class_recall"][name] - eps
        for name in DETECTOR_CLASSES
    )
    reason = "rejected-no-meaningful-safe-improvement"
    selected = False
    if hard_gates_passed and secret_delta >= (
        SAFETY_SELECTION_POLICY["meaningful_secret_gain"] - eps
    ):
        selected = True
        reason = "selected-meaningful-secret-improvement"
    elif (
        hard_gates_passed
        and secret_delta >= -eps
        and secret_delta
        < SAFETY_SELECTION_POLICY["meaningful_secret_gain"] - eps
        and other_delta
        >= SAFETY_SELECTION_POLICY["tie_band_other_sensitive_mean_gain"] - eps
    ):
        selected = True
        reason = "selected-tie-band-other-sensitive-improvement"
    elif (
        hard_gates_passed
        and all_recalls_nondecreasing
        and precision_delta > eps
    ):
        selected = True
        reason = "selected-precision-tiebreak-without-recall-regression"
    elif not hard_gates_passed:
        reason = "rejected-regression-guard"

    new_state = state
    if selected:
        new_state = {
            "schema_version": 1,
            "incumbent": candidate,
            "high_water": {
                "other_sensitive_recall_mean": max(
                    historical_other, candidate["other_sensitive_recall_mean"]
                )
            },
        }
    return (
        {
            "selected": selected,
            "valid": True,
            "reason": reason,
            "candidate": candidate,
            "deltas": {
                "secret_recall_minimum": secret_delta,
                "other_sensitive_recall_mean": other_delta,
                "precision_mean": precision_delta,
                "other_sensitive_from_high_water": candidate[
                    "other_sensitive_recall_mean"
                ]
                - historical_other,
            },
            "gates": {
                **gates,
                "all_hard_gates": hard_gates_passed,
                "all_recalls_nondecreasing": all_recalls_nondecreasing,
            },
        },
        new_state,
    )


def require_valid_safety_decision(decision: dict[str, Any]) -> None:
    if decision.get("valid") is not True:
        raise RuntimeError(
            "safety selection validation is incomplete or non-finite: "
            + str(decision.get("reason"))
        )


def validation_uses_test_split(value: Any) -> bool:
    paths = value if isinstance(value, list) else [value]
    return any(
        part.lower() == "test" for item in paths for part in Path(str(item)).parts
    )


def normalize_class_names(value: Any, context: str) -> dict[int, str]:
    if isinstance(value, list):
        normalized = {index: name for index, name in enumerate(value)}
    elif isinstance(value, dict):
        normalized = {}
        for raw_index, name in value.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"{context} has a non-integer class index") from exc
            if index in normalized:
                raise RuntimeError(f"{context} has a duplicate class index {index}")
            normalized[index] = name
    else:
        raise RuntimeError(f"{context} must declare detector class names")
    if sorted(normalized) != list(range(len(normalized))):
        raise RuntimeError(f"{context} class indices must be contiguous from zero")
    if any(not isinstance(name, str) or not name for name in normalized.values()):
        raise RuntimeError(f"{context} has an invalid detector class name")
    if len(set(normalized.values())) != len(normalized):
        raise RuntimeError(f"{context} has duplicate detector class names")
    return {index: str(normalized[index]) for index in sorted(normalized)}


def _split_paths(value: Any, context: str) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    if not values or any(not isinstance(item, (str, Path)) for item in values):
        raise RuntimeError(f"{context} must contain one or more dataset paths")
    paths = [Path(str(item)) for item in values]
    if any(not str(path) for path in paths):
        raise RuntimeError(f"{context} contains an empty dataset path")
    return paths


def _resolve_split_path(dataset: dict[str, Any], path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    root = Path(str(dataset.get("path", ".")))
    return (root / path).resolve()


def _declared_manifest_candidates(value: Any, source_manifest_path: Path) -> list[Path]:
    if not isinstance(value, str) or not value:
        raise RuntimeError("composed source manifest path is missing")
    path = Path(value)
    if path.name != "manifest.json":
        raise RuntimeError("composed source manifest path must end in manifest.json")
    if path.is_absolute():
        return [path.resolve()]
    if ".." in path.parts:
        raise RuntimeError("composed source manifest path may not traverse parents")
    candidates = [path.resolve(), (source_manifest_path.parent / path).resolve()]
    return list(dict.fromkeys(candidates))


def _expected_split_paths(
    source: Any,
    source_manifest_path: Path,
    split: str,
    context: str,
) -> list[Path]:
    if not isinstance(source, dict):
        raise RuntimeError(f"composed manifest is missing its {context} source")
    return [
        manifest.parent / "images" / split
        for manifest in _declared_manifest_candidates(
            source.get("manifest"), source_manifest_path
        )
    ]


def validate_composed_dataset_contract(
    dataset: dict[str, Any],
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
) -> dict[int, str]:
    """Bind selection and held-out evaluation to the composed data sources."""

    if source_manifest.get("published_splits_preserved") is not True:
        raise RuntimeError(
            "detector source manifest does not preserve WebPII published splits"
        )
    if source_manifest.get("test_used_for_checkpoint_selection") is not False:
        raise RuntimeError("WebPII test may not be used for checkpoint selection")
    if source_manifest.get("selection_data") != SYNTHETIC_VALIDATION_SOURCE:
        raise RuntimeError(
            "checkpoint selection must use screen-native synthetic validation"
        )
    if source_manifest.get("test_source") != PUBLISHED_TEST_SOURCE:
        raise RuntimeError("published detector test source must be WebPII/test")
    if "train" not in dataset or "val" not in dataset:
        raise RuntimeError(
            "training YAML must include train and a separate tuning val split; "
            "prepare_webpii intentionally emits no val split"
        )
    if "test" not in dataset:
        raise RuntimeError(
            "training YAML must include the composed WebPII published test split"
        )
    if validation_uses_test_split(dataset["val"]):
        raise RuntimeError(
            "the tuning val path must not point at the published test split"
        )

    names = normalize_class_names(dataset.get("names"), "training YAML")
    manifest_names = normalize_class_names(
        source_manifest.get("classes"), "composed source manifest"
    )
    if names != manifest_names or tuple(names.values()) != DETECTOR_CLASSES:
        raise RuntimeError(
            "training YAML classes differ from the composed detector contract"
        )

    val_paths = _split_paths(dataset["val"], "training YAML val")
    test_paths = _split_paths(dataset["test"], "training YAML test")
    if len(val_paths) != 1 or len(test_paths) != 1:
        raise RuntimeError(
            "composed detector validation and test must each contain exactly one path"
        )
    actual_val = _resolve_split_path(dataset, val_paths[0])
    actual_test = _resolve_split_path(dataset, test_paths[0])
    sources = source_manifest.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("composed detector manifest is missing source contracts")
    expected_val = _expected_split_paths(
        sources.get("synthetic"),
        source_manifest_path,
        "validation",
        "synthetic validation",
    )
    expected_test = _expected_split_paths(
        sources.get("webpii"),
        source_manifest_path,
        "test",
        "WebPII",
    )
    if actual_val not in expected_val:
        raise RuntimeError(
            "training YAML val does not point at the composed synthetic validation split"
        )
    if actual_test not in expected_test:
        raise RuntimeError(
            "training YAML test does not point at the composed WebPII published test split"
        )
    if actual_val == actual_test:
        raise RuntimeError(
            "synthetic validation and WebPII test paths must be distinct"
        )
    return names


def _safe_supplemental_file(root: Path, value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{context} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{context} contains an unsafe path")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context} is missing or is not a regular file")
    if root.resolve() not in path.resolve().parents:
        raise RuntimeError(f"{context} escapes the staged supplemental root")
    return path


def prepare_supplemental_test_yaml(
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
    output_dir: Path,
    names: dict[int, str],
) -> Path | None:
    sources = source_manifest.get("sources")
    if not isinstance(sources, dict) or "supplemental_ats" not in sources:
        return None
    source = sources["supplemental_ats"]
    if not isinstance(source, dict):
        raise RuntimeError("composed supplemental source contract must be an object")
    if source.get("test_excluded_from_training_and_selection") is not True:
        raise RuntimeError(
            "supplemental test must be excluded from training and checkpoint selection"
        )
    candidates = _declared_manifest_candidates(
        source.get("manifest"), source_manifest_path
    )
    existing = [path for path in candidates if not path.is_symlink() and path.is_file()]
    if len(existing) != 1:
        raise RuntimeError(
            "staged supplemental manifest is missing or resolves ambiguously"
        )
    manifest_path = existing[0]
    expected_hash = source.get("manifest_sha256")
    if (
        not isinstance(expected_hash, str)
        or sha256_file(manifest_path) != expected_hash
    ):
        raise RuntimeError("staged supplemental manifest SHA-256 mismatch")
    try:
        supplemental = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "staged supplemental manifest is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(supplemental, dict):
        raise RuntimeError("staged supplemental manifest must be a JSON object")
    schema_version = supplemental.get("schema_version")
    if type(schema_version) is not int or schema_version < 2:
        raise RuntimeError(
            "staged supplemental manifest must use the verified v2 contract"
        )
    required_flags = {
        "supplemental_only": True,
        "training_use": "train-split-only",
        "test_used_for_checkpoint_selection": False,
        "values_persisted": False,
        "unmapped_source_labels_fail_closed": True,
    }
    if any(supplemental.get(key) != value for key, value in required_flags.items()):
        raise RuntimeError("staged supplemental manifest safety flags are incomplete")
    if supplemental.get("classes") != list(names.values()):
        raise RuntimeError(
            "staged supplemental classes differ from the detector contract"
        )
    content_hashes = supplemental.get("content_hashes")
    if not isinstance(content_hashes, dict) or any(
        source.get(key) != content_hashes.get(key)
        for key in ("images_aggregate_sha256", "labels_aggregate_sha256")
    ):
        raise RuntimeError(
            "composed supplemental content hashes differ from the staged manifest"
        )
    splits = supplemental.get("splits")
    test_split = splits.get("test") if isinstance(splits, dict) else None
    if not isinstance(test_split, dict):
        raise RuntimeError("staged supplemental manifest is missing its test split")
    if test_split.get("records") != EXPECTED_SUPPLEMENTAL_TEST_RECORDS:
        raise RuntimeError(
            "staged supplemental test split must contain exactly "
            f"{EXPECTED_SUPPLEMENTAL_TEST_RECORDS} records"
        )
    root = manifest_path.parent
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("staged supplemental root is missing or unsafe")
    records_path = _safe_supplemental_file(
        root,
        test_split.get("records_path"),
        "staged supplemental test records",
    )
    if records_path.relative_to(root).as_posix() != "records/test.jsonl":
        raise RuntimeError(
            "staged supplemental test records must be records/test.jsonl"
        )
    if test_split.get("records_sha256") != sha256_file(records_path):
        raise RuntimeError("staged supplemental test records SHA-256 mismatch")
    try:
        record_lines = records_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            "staged supplemental test records are not valid UTF-8"
        ) from exc
    if len(record_lines) != EXPECTED_SUPPLEMENTAL_TEST_RECORDS or any(
        not line for line in record_lines
    ):
        raise RuntimeError(
            "staged supplemental test records do not contain exactly "
            f"{EXPECTED_SUPPLEMENTAL_TEST_RECORDS} non-empty rows"
        )
    seen_images: set[str] = set()
    seen_labels: set[str] = set()
    for line_number, line in enumerate(record_lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"staged supplemental test record {line_number} is invalid JSON"
            ) from exc
        if not isinstance(record, dict) or record.get("split") != "test":
            raise RuntimeError(
                f"staged supplemental test record {line_number} has the wrong split"
            )
        image_path = _safe_supplemental_file(
            root,
            record.get("image"),
            f"staged supplemental test record {line_number} image",
        )
        label_path = _safe_supplemental_file(
            root,
            record.get("label_file"),
            f"staged supplemental test record {line_number} label",
        )
        image_relative = image_path.relative_to(root).as_posix()
        label_relative = label_path.relative_to(root).as_posix()
        if (
            not image_relative.startswith("images/test/")
            or not label_relative.startswith("labels/test/")
            or image_relative in seen_images
            or label_relative in seen_labels
        ):
            raise RuntimeError(
                f"staged supplemental test record {line_number} has unsafe or reused files"
            )
        seen_images.add(image_relative)
        seen_labels.add(label_relative)
        if record.get("image_sha256") != sha256_file(image_path):
            raise RuntimeError(
                f"staged supplemental test record {line_number} image SHA-256 mismatch"
            )
        if record.get("label_sha256") != sha256_file(label_path):
            raise RuntimeError(
                f"staged supplemental test record {line_number} label SHA-256 mismatch"
            )
    for directory in (root / "images/test", root / "labels/test"):
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError(
                "staged supplemental test image/label directories are missing or unsafe"
            )

    yaml_path = output_dir / "supplemental-test.yaml"
    yaml_lines = [
        f"path: {root.resolve()}",
        "train: images/train",
        "val: images/test",
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
        *[f"  {index}: {name}" for index, name in names.items()],
        "",
    ]
    yaml_path.write_text("\n".join(yaml_lines), encoding="utf-8")
    return yaml_path


def evaluation_report(
    evaluation: Any,
    expected_names: dict[int, str],
    missing_key: str,
) -> dict[str, Any]:
    names = normalize_class_names(
        getattr(evaluation, "names", None), "Ultralytics evaluation"
    )
    if names != expected_names:
        raise RuntimeError(
            "Ultralytics evaluation classes differ from the composed detector contract"
        )
    box = getattr(evaluation, "box", None)
    aggregate = getattr(evaluation, "results_dict", None)
    if box is None or not isinstance(aggregate, dict):
        raise RuntimeError("Ultralytics evaluation did not return complete metrics")
    recalls, precisions, missing = align_class_metrics(box, names)
    per_class = {
        names[index]: {
            "precision": precisions[index],
            "recall": recalls[index],
        }
        for index in sorted(names)
    }
    missing_secret_classes = [
        name for name in SECRET_VISUAL_CLASSES if name not in per_class
    ]
    if missing_secret_classes:
        raise RuntimeError(
            "detector evaluation is missing secret visual classes: "
            f"{missing_secret_classes}"
        )
    return {
        "aggregate": aggregate,
        "per_class": per_class,
        "minimum_secret_class_recall": min(
            per_class[name]["recall"] for name in SECRET_VISUAL_CLASSES
        ),
        "missing_classes": missing,
        missing_key: missing,
    }


def _checkpoint_record(value: Any) -> dict[str, Any]:
    if value is None:
        return {"path": None, "exists": False}
    path = Path(str(value))
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must contain a JSON object")
    return value


def build_training_resume_contract(
    args: argparse.Namespace, names: dict[int, str]
) -> dict[str, Any]:
    """Fingerprint the immutable inputs that a resumed checkpoint may consume."""

    staging_path_value = getattr(args, "local_staging_manifest", None)
    staging_path = Path(staging_path_value) if staging_path_value else None
    staging_sha256 = None
    source_contract_sha256 = None
    if staging_path is not None:
        staging = _read_json_object(
            staging_path, "local detector staging manifest"
        )
        if staging.get("schema_version") != 1:
            raise RuntimeError("local detector staging manifest schema is unsupported")
        dataset_hash = sha256_file(args.dataset_yaml)
        source_hash = sha256_file(args.source_manifest)
        if staging.get("local_dataset_yaml_sha256") != dataset_hash:
            raise RuntimeError(
                "local staging manifest dataset YAML hash does not match training"
            )
        if staging.get("local_source_manifest_sha256") != source_hash:
            raise RuntimeError(
                "local staging manifest source hash does not match training"
            )
        source_contract_sha256 = staging.get("source_contract_sha256")
        if not isinstance(source_contract_sha256, str) or len(
            source_contract_sha256
        ) != 64:
            raise RuntimeError("local staging source-contract hash is invalid")
        staging_sha256 = sha256_file(staging_path)
    return {
        "run_id": getattr(args, "run_id", None),
        "dataset_yaml_sha256": sha256_file(args.dataset_yaml),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "staging_manifest_sha256": staging_sha256,
        "source_contract_sha256": source_contract_sha256,
        "class_contract_sha256": _canonical_json_sha256(list(names.values())),
        "augmentation_policy_sha256": _canonical_json_sha256(
            SCREENSHOT_AUGMENTATION_POLICY
        ),
        "optimizer_policy_sha256": _canonical_json_sha256(
            STABLE_OPTIMIZER_POLICY
        ),
        "attempt_id": getattr(args, "attempt_id", "default"),
        "attempt_manifest_sha256": getattr(args, "attempt_manifest_sha256", None),
        "seed_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "output_dir": str(args.output_dir.resolve()),
        "local_stage_root": str(args.dataset_yaml.resolve().parent),
        "total_epochs": int(args.epochs),
    }


def validate_resume_data_contract(
    args: argparse.Namespace, contract: dict[str, Any]
) -> None:
    """Fail closed unless run identity and all staged data hashes still match."""

    run_id = contract.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("visual resume requires the original run ID")
    output_dir = args.output_dir.resolve()
    expected_name = (
        "training"
        if contract.get("attempt_id") == "default"
        else f"training-{contract.get('attempt_id')}"
    )
    if (
        contract.get("output_dir") != str(output_dir)
        or output_dir.parent.name != "visual"
        or output_dir.parent.parent.name != run_id
        or output_dir.name != expected_name
    ):
        raise RuntimeError("visual resume output directory does not match the run ID")
    for path, context in (
        (args.dataset_yaml, "dataset YAML"),
        (args.source_manifest, "source manifest"),
        (getattr(args, "local_staging_manifest", None), "staging manifest"),
    ):
        if (
            path is None
            or str(Path(path).resolve().parent) != contract.get("local_stage_root")
        ):
            raise RuntimeError(f"visual resume {context} does not match the run ID")

    staging = getattr(args, "data_staging", None)
    if not isinstance(staging, dict):
        raise RuntimeError("visual resume requires persisted staging metadata")
    expected = {
        "dataset_yaml": (
            args.output_dir / "staged-dataset.yaml",
            contract["dataset_yaml_sha256"],
        ),
        "source_manifest": (
            args.output_dir / "staged-source-manifest.json",
            contract["source_manifest_sha256"],
        ),
        "staging_manifest": (
            args.output_dir / "dataset-staging-manifest.json",
            contract["staging_manifest_sha256"],
        ),
    }
    for name, (path, current_hash) in expected.items():
        declared_path = staging.get(f"{name}_path")
        declared_hash = staging.get(f"{name}_sha256")
        if (
            not isinstance(declared_path, str)
            or Path(declared_path).resolve() != path.resolve()
            or path.is_symlink()
            or not path.is_file()
        ):
            raise RuntimeError(f"visual resume persisted {name} path is invalid")
        actual_hash = sha256_file(path)
        if (
            not isinstance(current_hash, str)
            or declared_hash != actual_hash
            or actual_hash != current_hash
        ):
            raise RuntimeError(f"visual resume persisted {name} hash mismatch")


def _validated_progress_checkpoint(
    progress: dict[str, Any], name: str, expected_path: Path
) -> Path:
    checkpoints = progress.get("checkpoints")
    checkpoint = checkpoints.get(name) if isinstance(checkpoints, dict) else None
    if not isinstance(checkpoint, dict) or checkpoint.get("exists") is not True:
        raise RuntimeError(f"resume progress is missing the {name} checkpoint")
    declared_path = checkpoint.get("path")
    if (
        not isinstance(declared_path, str)
        or Path(declared_path).resolve() != expected_path.resolve()
        or expected_path.is_symlink()
        or not expected_path.is_file()
    ):
        raise RuntimeError(f"resume progress {name} checkpoint path is invalid")
    if checkpoint.get("bytes") != expected_path.stat().st_size:
        raise RuntimeError(f"resume progress {name} checkpoint size mismatch")
    actual_hash = sha256_file(expected_path)
    if checkpoint.get("sha256") != actual_hash:
        raise RuntimeError(f"resume progress {name} checkpoint SHA-256 mismatch")
    return expected_path


def _restore_selection_history(
    progress: dict[str, Any]
) -> tuple[
    list[dict[str, Any]], tuple[float, float, float], dict[str, Any], dict[str, Any]
]:
    progress_epoch = progress.get("epoch")
    history = progress.get("history")
    if type(progress_epoch) is not int or progress_epoch < 0:
        raise RuntimeError("resume progress epoch is invalid")
    if not isinstance(history, list) or len(history) != progress_epoch + 1:
        raise RuntimeError("resume selection history is incomplete")
    seed_incumbent = progress.get("seed_incumbent")
    if not isinstance(seed_incumbent, dict):
        raise RuntimeError("resume progress seed incumbent is missing")
    seed_decision, state = consider_safety_candidate(
        None,
        epoch=-1,
        recall_by_class=seed_incumbent.get("per_class_recall"),
        precision_by_class=seed_incumbent.get("per_class_precision"),
        missing_classes=seed_incumbent.get("missing_validation_classes", []),
    )
    if not seed_decision["selected"] or state is None:
        raise RuntimeError("resume progress seed incumbent is invalid")
    restored: list[dict[str, Any]] = []
    for expected_epoch, record in enumerate(history):
        if not isinstance(record, dict) or record.get("epoch") != expected_epoch:
            raise RuntimeError("resume selection history epochs are not contiguous")
        missing = record.get("missing_validation_classes")
        if not isinstance(missing, list) or any(
            name not in DETECTOR_CLASSES for name in missing
        ):
            raise RuntimeError("resume selection history has invalid missing classes")
        decision, next_state = consider_safety_candidate(
            state,
            epoch=expected_epoch,
            recall_by_class=record.get("per_class_recall"),
            precision_by_class=record.get("per_class_precision"),
            missing_classes=missing,
        )
        if (
            record.get("selected") is not decision["selected"]
            or record.get("selection_reason") != decision["reason"]
            or record.get("selection_deltas") != decision["deltas"]
            or record.get("selection_gates") != decision["gates"]
        ):
            raise RuntimeError("resume selection history has inconsistent selection")
        candidate = decision["candidate"]
        if (
            candidate is None
            or record.get("secret_recall_minimum")
            != candidate["secret_recall_minimum"]
            or record.get("other_sensitive_recall_mean")
            != candidate["other_sensitive_recall_mean"]
            or record.get("precision_mean") != candidate["precision_mean"]
        ):
            raise RuntimeError("resume selection history metric summary was altered")
        if next_state is not None:
            state = next_state
        restored.append(dict(record))
    if state is None:
        raise RuntimeError("resume selection state is missing")
    incumbent = state["incumbent"]
    running_best = (
        float(incumbent["secret_recall_minimum"]),
        float(incumbent["other_sensitive_recall_mean"]),
        float(incumbent["precision_mean"]),
    )
    declared_best = progress.get("best_score")
    if declared_best != list(running_best):
        raise RuntimeError("resume selection history best score is inconsistent")
    if progress.get("selection_state") != state:
        raise RuntimeError("resume selection state does not match replayed history")
    return restored, running_best, state, seed_incumbent


def load_visual_resume_state(
    args: argparse.Namespace,
    contract: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    """Validate and load exactly the durable last.pt for an interrupted run."""

    validate_resume_data_contract(args, contract)
    progress_path = args.output_dir / SAFETY_PROGRESS_NAME
    progress = _read_json_object(progress_path, "safety selection progress")
    schema_version = progress.get("schema_version")
    if schema_version != 3:
        raise RuntimeError(
            "legacy visual progress cannot resume under the stable optimizer and "
            "constrained safety-selection policy; start a named frozen-seed attempt"
        )
    if progress.get("resume_contract") != contract:
        raise RuntimeError("resume progress data or run contract has changed")
    persisted_seed = _read_json_object(
        args.output_dir / SEED_INCUMBENT_NAME, "frozen seed incumbent"
    )
    if persisted_seed != progress.get("seed_incumbent"):
        raise RuntimeError("frozen seed incumbent differs from resume progress")
    if (
        persisted_seed.get("checkpoint_sha256")
        != contract.get("seed_checkpoint_sha256")
        or not isinstance(persisted_seed.get("checkpoint"), str)
        or Path(persisted_seed["checkpoint"]).resolve()
        != args.base_checkpoint.resolve()
    ):
        raise RuntimeError("frozen seed incumbent does not match the immutable seed")

    weights = args.output_dir / ULTRALYTICS_RUN_NAME / "weights"
    last_path = _validated_progress_checkpoint(
        progress, "last", weights / "last.pt"
    )
    _validated_progress_checkpoint(progress, "best", weights / "best.pt")
    _validated_progress_checkpoint(
        progress, "safety_best", args.output_dir / "safety-best.pt"
    )
    history, best_key, selection_state, seed_incumbent = (
        _restore_selection_history(progress)
    )
    try:
        checkpoint = torch_module.load(
            str(last_path), map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise RuntimeError("persisted last.pt could not be loaded for resume") from exc
    if not isinstance(checkpoint, dict):
        raise RuntimeError("persisted last.pt is not an Ultralytics checkpoint")
    if checkpoint.get("epoch") != progress.get("epoch"):
        raise RuntimeError("last.pt epoch differs from safety selection progress")
    if checkpoint.get("optimizer") is None:
        raise RuntimeError("last.pt lacks optimizer state required for resume")
    if checkpoint.get("scaler") is None or checkpoint.get("ema") is None:
        raise RuntimeError("last.pt lacks scaler or EMA state required for resume")
    train_args = checkpoint.get("train_args")
    if not isinstance(train_args, dict):
        raise RuntimeError("last.pt lacks original Ultralytics training arguments")
    if train_args.get("epochs") != args.epochs:
        raise RuntimeError("resume total epochs differ from the original schedule")
    if progress["epoch"] + 1 >= args.epochs:
        raise RuntimeError("the persisted visual training schedule is already complete")
    checkpoint_data = train_args.get("data")
    if (
        not isinstance(checkpoint_data, str)
        or Path(checkpoint_data).resolve() != args.dataset_yaml.resolve()
    ):
        raise RuntimeError("last.pt dataset path differs from the resumed dataset")
    expected_save_dir = args.output_dir / ULTRALYTICS_RUN_NAME
    checkpoint_save_dir = train_args.get("save_dir")
    if (
        not isinstance(checkpoint_save_dir, str)
        or Path(checkpoint_save_dir).resolve() != expected_save_dir.resolve()
    ):
        raise RuntimeError("last.pt output directory differs from the resumed run")
    for key, expected in (
        ("imgsz", args.image_size),
        ("batch", args.batch_size),
        ("seed", args.seed),
        *STABLE_OPTIMIZER_POLICY.items(),
        *SCREENSHOT_AUGMENTATION_POLICY.items(),
    ):
        if train_args.get(key) != expected:
            raise RuntimeError(f"last.pt training argument {key} changed on resume")
    return {
        "last_path": last_path,
        "history": history,
        "best_key": best_key,
        "selection_state": selection_state,
        "seed_incumbent": seed_incumbent,
        "progress": progress,
        "progress_sha256": sha256_file(progress_path),
    }


def write_safety_selection_progress(
    output_dir: Path,
    trainer: Any,
    history: list[dict[str, Any]],
    best_key: tuple[float, float, float] | None,
    safety_best_path: Path,
    resume_contract: dict[str, Any],
    seed_incumbent: dict[str, Any],
    selection_state: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Atomically record every checkpoint needed to survive preemption."""

    document = {
        "schema_version": 3,
        "epoch": int(trainer.epoch),
        "selection_policy": SAFETY_SELECTION_POLICY,
        "best_score": list(best_key) if best_key is not None else None,
        "seed_incumbent": seed_incumbent,
        "selection_state": selection_state,
        "history": history,
        "resume_contract": resume_contract,
        "checkpoints": {
            "last": _checkpoint_record(getattr(trainer, "last", None)),
            "best": _checkpoint_record(getattr(trainer, "best", None)),
            "safety_best": _checkpoint_record(safety_best_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / SAFETY_PROGRESS_NAME
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path, document


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.base_license in {"", "NOASSERTION"}:
        raise RuntimeError(
            "replacement detector base checkpoint requires an explicit license"
        )
    commercial_license_approved = bool(
        getattr(args, "commercial_license_approved", False)
    )
    checkpoint_commit_hook = getattr(args, "checkpoint_commit_hook", None)
    if checkpoint_commit_hook is not None and not callable(checkpoint_commit_hook):
        raise TypeError("checkpoint_commit_hook must be callable")
    resume_requested = bool(getattr(args, "resume", False))
    if resume_requested and getattr(args, "local_staging_manifest", None) is None:
        args.local_staging_manifest = (
            args.dataset_yaml.parent / "staging_manifest.json"
        )
    if resume_requested and not isinstance(getattr(args, "data_staging", None), dict):
        persisted = {
            "dataset_yaml": args.output_dir / "staged-dataset.yaml",
            "source_manifest": args.output_dir / "staged-source-manifest.json",
            "staging_manifest": args.output_dir / "dataset-staging-manifest.json",
        }
        args.data_staging = {
            f"{name}_{field}": (
                str(path) if field == "path" else sha256_file(path)
            )
            for name, path in persisted.items()
            for field in ("path", "sha256")
        }
    agpl_mode = args.base_license.startswith("AGPL-3.0")
    import torch
    import yaml
    from ultralytics import YOLO

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if not isinstance(source_manifest, dict):
        raise RuntimeError("detector source manifest must be a JSON object")
    dataset = yaml.safe_load(args.dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise RuntimeError("training YAML must contain a dataset mapping")
    expected_names = validate_composed_dataset_contract(
        dataset, source_manifest, args.source_manifest
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resume_contract = build_training_resume_contract(args, expected_names)
    resume_state = (
        load_visual_resume_state(args, resume_contract, torch)
        if resume_requested
        else None
    )
    supplemental_test_yaml = prepare_supplemental_test_yaml(
        source_manifest,
        args.source_manifest,
        args.output_dir,
        expected_names,
    )
    best_path = args.output_dir / "safety-best.pt"
    if resume_state is not None:
        history: list[dict[str, Any]] = list(resume_state["history"])
        best_key: tuple[float, float, float] = resume_state["best_key"]
        selection_state: dict[str, Any] = resume_state["selection_state"]
        seed_incumbent: dict[str, Any] = resume_state["seed_incumbent"]
    else:
        seed_validation = YOLO(str(args.base_checkpoint)).val(
            data=str(args.dataset_yaml),
            split="val",
            imgsz=args.image_size,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            plots=False,
            verbose=False,
            project=str(args.output_dir),
            name="frozen-seed-validation",
            exist_ok=False,
        )
        seed_names = normalize_class_names(
            getattr(seed_validation, "names", None), "frozen seed validation"
        )
        if seed_names != expected_names:
            raise RuntimeError("frozen seed validation class contract changed")
        seed_box = getattr(seed_validation, "box", None)
        if seed_box is None:
            raise RuntimeError("frozen seed validation returned no class metrics")
        seed_recalls, seed_precisions, seed_missing = align_class_metrics(
            seed_box, seed_names
        )
        seed_recall_by_class = {
            seed_names[index]: seed_recalls[index] for index in seed_names
        }
        seed_precision_by_class = {
            seed_names[index]: seed_precisions[index] for index in seed_names
        }
        seed_decision, initialized_state = consider_safety_candidate(
            None,
            epoch=-1,
            recall_by_class=seed_recall_by_class,
            precision_by_class=seed_precision_by_class,
            missing_classes=seed_missing,
        )
        if not seed_decision["selected"] or initialized_state is None:
            raise RuntimeError(
                "frozen seed cannot initialize safety selection: "
                + str(seed_decision["reason"])
            )
        selection_state = initialized_state
        seed_incumbent = {
            **seed_decision["candidate"],
            "missing_validation_classes": seed_missing,
            "checkpoint": str(args.base_checkpoint),
            "checkpoint_sha256": sha256_file(args.base_checkpoint),
        }
        incumbent = selection_state["incumbent"]
        best_key = (
            incumbent["secret_recall_minimum"],
            incumbent["other_sensitive_recall_mean"],
            incumbent["precision_mean"],
        )
        history = []
        shutil.copy2(args.base_checkpoint, best_path)
        if sha256_file(best_path) != seed_incumbent["checkpoint_sha256"]:
            raise RuntimeError("frozen safety incumbent copy failed verification")
        (args.output_dir / SEED_INCUMBENT_NAME).write_text(
            json.dumps(seed_incumbent, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if checkpoint_commit_hook is not None:
            checkpoint_commit_hook()
    baseline_published_test = YOLO(str(args.base_checkpoint)).val(
        data=str(args.dataset_yaml),
        split="test",
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=False,
        project=str(args.output_dir),
        name="frozen-base-webpii-test",
        exist_ok=resume_requested,
    )
    baseline_published_test_evaluation = evaluation_report(
        baseline_published_test,
        expected_names,
        "missing_test_classes",
    )
    baseline_supplemental_test_evaluation = None
    if supplemental_test_yaml is not None:
        baseline_supplemental_test = YOLO(str(args.base_checkpoint)).val(
            data=str(supplemental_test_yaml),
            split="test",
            imgsz=args.image_size,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            plots=False,
            verbose=False,
            project=str(args.output_dir),
            name="frozen-base-supplemental-test",
            exist_ok=resume_requested,
        )
        baseline_supplemental_test_evaluation = evaluation_report(
            baseline_supplemental_test,
            expected_names,
            "missing_test_classes",
        )
    model = YOLO(
        str(resume_state["last_path"] if resume_state else args.base_checkpoint)
    )
    checkpoint_commit_count = len(history)
    invocation_commit_count = 0

    def on_model_save(trainer) -> None:
        nonlocal best_key, selection_state
        nonlocal checkpoint_commit_count, invocation_commit_count
        metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
        box = getattr(metrics, "box", None)
        raw_names = getattr(getattr(trainer, "model", None), "names", None)
        if box is None or raw_names is None:
            if checkpoint_commit_hook is not None:
                raise RuntimeError(
                    "cannot durably commit an epoch without complete selection metrics"
                )
            return
        names = normalize_class_names(raw_names, "Ultralytics training model")
        if names != expected_names:
            raise RuntimeError(
                "Ultralytics training classes differ from the composed detector contract"
            )
        recalls, precisions, missing_validation_classes = align_class_metrics(
            box, names
        )
        recall_by_class = {names[index]: recalls[index] for index in names}
        precision_by_class = {names[index]: precisions[index] for index in names}
        decision, candidate_state = consider_safety_candidate(
            selection_state,
            epoch=int(trainer.epoch),
            recall_by_class=recall_by_class,
            precision_by_class=precision_by_class,
            missing_classes=missing_validation_classes,
        )
        require_valid_safety_decision(decision)
        candidate = decision["candidate"]
        record = {
            "epoch": int(trainer.epoch),
            "secret_recall_minimum": (
                candidate["secret_recall_minimum"] if candidate else None
            ),
            "other_sensitive_recall_mean": (
                candidate["other_sensitive_recall_mean"] if candidate else None
            ),
            # Backward-readable alias; selection no longer uses lexicographic order.
            "non_secret_recall_mean": (
                candidate["other_sensitive_recall_mean"] if candidate else None
            ),
            "precision_mean": candidate["precision_mean"] if candidate else None,
            "per_class_recall": recall_by_class,
            "per_class_precision": precision_by_class,
            "missing_validation_classes": missing_validation_classes,
            "selected": decision["selected"],
            "selection_reason": decision["reason"],
            "selection_deltas": decision["deltas"],
            "selection_gates": decision["gates"],
        }
        history.append(record)
        if record["selected"]:
            if candidate_state is None:
                raise RuntimeError("selected checkpoint produced no safety state")
            selection_state = candidate_state
            incumbent = selection_state["incumbent"]
            best_key = (
                incumbent["secret_recall_minimum"],
                incumbent["other_sensitive_recall_mean"],
                incumbent["precision_mean"],
            )
            checkpoint = Path(trainer.last)
            if checkpoint.exists():
                shutil.copy2(checkpoint, best_path)
        progress_path, progress = write_safety_selection_progress(
            args.output_dir,
            trainer,
            history,
            best_key,
            best_path,
            resume_contract,
            seed_incumbent,
            selection_state,
        )
        if checkpoint_commit_hook is not None:
            missing = [
                name
                for name, checkpoint in progress["checkpoints"].items()
                if checkpoint.get("exists") is not True
            ]
            if missing:
                raise RuntimeError(
                    "cannot commit incomplete epoch checkpoints: " + ", ".join(missing)
                )
            output_root = args.output_dir.resolve()
            outside_output = [
                name
                for name, checkpoint in progress["checkpoints"].items()
                if output_root not in Path(checkpoint["path"]).resolve().parents
            ]
            if outside_output:
                raise RuntimeError(
                    "cannot commit checkpoints outside the durable output root: "
                    + ", ".join(outside_output)
                )
            if not progress_path.is_file():
                raise RuntimeError("safety selection progress was not persisted")
            checkpoint_commit_hook()
            checkpoint_commit_count += 1
            invocation_commit_count += 1

    model.add_callback("on_model_save", on_model_save)
    result = model.train(
        data=str(args.dataset_yaml),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        project=str(args.output_dir),
        name=ULTRALYTICS_RUN_NAME,
        exist_ok=False,
        workers=args.workers,
        patience=args.patience,
        save=True,
        save_period=1,
        resume=resume_requested,
        plots=False,
        verbose=True,
        **STABLE_OPTIMIZER_POLICY,
        **SCREENSHOT_AUGMENTATION_POLICY,
    )
    if not best_path.exists():
        raise RuntimeError("training completed without a safety-selected checkpoint")
    selected_validation = YOLO(str(best_path)).val(
        data=str(args.dataset_yaml),
        split="val",
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=False,
        project=str(args.output_dir),
        name="safety-best-validation",
        exist_ok=False,
    )
    selected_evaluation = evaluation_report(
        selected_validation,
        expected_names,
        "missing_validation_classes",
    )
    published_test = YOLO(str(best_path)).val(
        data=str(args.dataset_yaml),
        split="test",
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=False,
        project=str(args.output_dir),
        name="safety-best-webpii-test",
        exist_ok=False,
    )
    published_test_evaluation = evaluation_report(
        published_test,
        expected_names,
        "missing_test_classes",
    )
    supplemental_test_evaluation = None
    if supplemental_test_yaml is not None:
        supplemental_test = YOLO(str(best_path)).val(
            data=str(supplemental_test_yaml),
            split="test",
            imgsz=args.image_size,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            plots=False,
            verbose=False,
            project=str(args.output_dir),
            name="safety-best-supplemental-test",
            exist_ok=False,
        )
        supplemental_test_evaluation = evaluation_report(
            supplemental_test,
            expected_names,
            "missing_test_classes",
        )
    manifest = {
        "schema_version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "trained-unpublished",
        "selection_policy": SAFETY_SELECTION_POLICY,
        "seed_incumbent": seed_incumbent,
        "selection_state": selection_state,
        "base_checkpoint": {
            "path": str(args.base_checkpoint),
            "sha256": sha256_file(args.base_checkpoint),
            "license": args.base_license,
            "source": getattr(args, "base_source", None),
        },
        "data": {
            "yaml": str(args.dataset_yaml),
            "yaml_sha256": sha256_file(args.dataset_yaml),
            "source_manifest": str(args.source_manifest),
            "source_manifest_sha256": sha256_file(args.source_manifest),
            "staging": getattr(args, "data_staging", None),
        },
        "training_attempt": {
            "attempt_id": getattr(args, "attempt_id", "default"),
            "output_root": str(args.output_dir.resolve()),
            "attempt_manifest_sha256": getattr(
                args, "attempt_manifest_sha256", None
            ),
            "new_non_resume_seed": not resume_requested,
        },
        "best_checkpoint": {
            "path": best_path.name,
            "sha256": sha256_file(best_path),
            "score": best_key,
        },
        "history": history,
        "environment": {
            "torch": torch.__version__,
            "ultralytics": importlib.metadata.version("ultralytics"),
            "cuda_available": torch.cuda.is_available(),
            "device": args.device,
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "gpu_memory_bytes": (
                int(torch.cuda.get_device_properties(0).total_memory)
                if torch.cuda.is_available()
                else None
            ),
        },
        "training_config": {
            "epochs": args.epochs,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "patience": args.patience,
            "seed": args.seed,
            "resume_requested": resume_requested,
            "save": True,
            "save_period": 1,
            "optimizer_policy": {
                "name": "stable-adamw-v1",
                "ultralytics_overrides": dict(STABLE_OPTIMIZER_POLICY),
                "sha256": _canonical_json_sha256(STABLE_OPTIMIZER_POLICY),
                "auto_optimizer_disabled": True,
            },
            "monitoring_policy": dict(MONITORING_POLICY),
            "augmentation_policy": {
                "name": "screen-native-conservative-v1",
                "ultralytics_overrides": dict(SCREENSHOT_AUGMENTATION_POLICY),
                "disabled_ui_invalidating_transforms": [
                    "rotation",
                    "shear",
                    "perspective",
                    "horizontal-flip",
                    "vertical-flip",
                    "rgb-bgr-channel-swap",
                    "mosaic",
                    "mixup",
                    "cutmix",
                    "copy-paste",
                    "multi-scale",
                ],
                "safe_variation": {
                    "translation_fraction": 0.02,
                    "scale_gain": 0.05,
                    "saturation_gain": 0.1,
                    "value_gain": 0.1,
                    "hue_gain": 0.0,
                },
            },
        },
        "checkpoint_persistence": {
            "commit_hook_enabled": checkpoint_commit_hook is not None,
            "committed_epochs": checkpoint_commit_count,
            "committed_epochs_this_invocation": invocation_commit_count,
            "progress_path": SAFETY_PROGRESS_NAME,
            "progress_sha256": sha256_file(
                args.output_dir / SAFETY_PROGRESS_NAME
            ),
            "resume_supported": resume_contract.get("run_id") is not None,
            "resume_contract": resume_contract,
            "resumed": resume_requested,
            "resumed_from_epoch": (
                resume_state["progress"]["epoch"] if resume_state else None
            ),
            "resume_progress_sha256": (
                resume_state["progress_sha256"] if resume_state else None
            ),
            "progress_schema_version": 3,
            "seed_incumbent_path": SEED_INCUMBENT_NAME,
            "seed_incumbent_sha256": sha256_file(
                args.output_dir / SEED_INCUMBENT_NAME
            ),
        },
        "train_results": getattr(selected_validation, "results_dict", {}),
        "baseline_published_test_evaluation": {
            "dataset": PUBLISHED_TEST_SOURCE,
            "source": PUBLISHED_TEST_SOURCE,
            "split": "test",
            "timing": "pre-training",
            "checkpoint": str(args.base_checkpoint),
            "checkpoint_status": "frozen",
            "used_for_checkpoint_selection": False,
            **baseline_published_test_evaluation,
        },
        **(
            {
                "baseline_supplemental_test_evaluation": {
                    "dataset": "supplemental_ats/test",
                    "source": "user-provided ATS synthetic/test",
                    "split": "test",
                    "timing": "pre-training",
                    "checkpoint": str(args.base_checkpoint),
                    "checkpoint_status": "frozen",
                    "used_for_checkpoint_selection": False,
                    "dataset_yaml": {
                        "path": supplemental_test_yaml.name,
                        "sha256": sha256_file(supplemental_test_yaml),
                    },
                    **baseline_supplemental_test_evaluation,
                }
            }
            if baseline_supplemental_test_evaluation is not None
            and supplemental_test_yaml is not None
            else {}
        ),
        "selected_checkpoint_evaluation": {
            "dataset": "PLVA synthetic/validation",
            "source": SYNTHETIC_VALIDATION_SOURCE,
            "split": "val",
            "timing": "checkpoint-selection",
            "used_for_checkpoint_selection": True,
            **selected_evaluation,
        },
        "published_test_evaluation": {
            "dataset": PUBLISHED_TEST_SOURCE,
            "source": PUBLISHED_TEST_SOURCE,
            "split": "test",
            "timing": "post-selection",
            "checkpoint": best_path.name,
            "used_for_checkpoint_selection": False,
            **published_test_evaluation,
        },
        **(
            {
                "supplemental_test_evaluation": {
                    "dataset": "supplemental_ats/test",
                    "source": "user-provided ATS synthetic/test",
                    "split": "test",
                    "timing": "post-selection",
                    "checkpoint": best_path.name,
                    "used_for_checkpoint_selection": False,
                    "dataset_yaml": {
                        "path": supplemental_test_yaml.name,
                        "sha256": sha256_file(supplemental_test_yaml),
                    },
                    **supplemental_test_evaluation,
                }
            }
            if supplemental_test_evaluation is not None
            and supplemental_test_yaml is not None
            else {}
        ),
        "ultralytics_default_best_results": getattr(result, "results_dict", {}),
        "distribution": {
            "mode": "agpl-development" if agpl_mode else "commercial-license",
            "commercial_license_approved": commercial_license_approved,
            "closed_source_release_allowed": commercial_license_approved,
        },
        "publication_blockers": [
            "cross-runtime ONNX parity not yet proven",
            *(
                [
                    "AGPL development output cannot ship in a closed-source product; obtain an Ultralytics Enterprise license"
                ]
                if agpl_mode and not commercial_license_approved
                else []
            ),
            "WebPII and base-checkpoint license obligations require review",
        ],
    }
    # Ultralytics may expose NumPy or Torch scalar subclasses in results_dict.
    # Persist and return only plain JSON values so Modal clients do not need the
    # remote ML runtime merely to deserialize a completed run's metadata.
    serialized = json.dumps(manifest, indent=2, sort_keys=True, default=float)
    manifest_document = json.loads(serialized)
    (args.output_dir / "training_manifest.json").write_text(
        serialized + "\n",
        encoding="utf-8",
    )
    return manifest_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the optional PLVA visual detector"
    )
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-license", required=True)
    parser.add_argument("--base-source")
    parser.add_argument("--commercial-license-approved", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--local-staging-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1311)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(train(parse_args()), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
