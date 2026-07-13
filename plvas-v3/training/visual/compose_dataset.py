from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.schema import sha256_file
from training.visual.prepare_supplemental import CONTENT_HASH_ALGORITHM
from training.visual.prepare_webpii import DETECTOR_CLASSES


def compose(args: argparse.Namespace) -> dict[str, Any]:
    webpii_manifest_path = args.webpii_root / "manifest.json"
    synthetic_manifest_path = args.synthetic_root / "manifest.json"
    supplemental_root: Path | None = getattr(args, "supplemental_root", None)
    supplemental_manifest_path = (
        supplemental_root / "manifest.json" if supplemental_root is not None else None
    )
    webpii = json.loads(webpii_manifest_path.read_text(encoding="utf-8"))
    synthetic = json.loads(synthetic_manifest_path.read_text(encoding="utf-8"))
    supplemental = (
        json.loads(supplemental_manifest_path.read_text(encoding="utf-8"))
        if supplemental_manifest_path is not None
        else None
    )
    if webpii.get("published_splits_preserved") is not True:
        raise ValueError("WebPII published splits were not preserved")
    if webpii.get("test_used_for_checkpoint_selection") is not False:
        raise ValueError("WebPII test cannot be used for checkpoint selection")
    if synthetic.get("synthetic_only") is not True:
        raise ValueError("tuning validation must be synthetic-only")
    if webpii.get("classes") != list(DETECTOR_CLASSES):
        raise ValueError("WebPII class order differs from the detector contract")
    if synthetic.get("classes") != list(DETECTOR_CLASSES):
        raise ValueError("synthetic class order differs from the detector contract")
    if supplemental is not None:
        if type(supplemental.get("schema_version")) is not int or supplemental[
            "schema_version"
        ] < 2:
            raise ValueError("supplemental source was not prepared from a verified v2 dataset")
        if supplemental.get("supplemental_only") is not True:
            raise ValueError("supplemental source is not marked supplemental-only")
        if supplemental.get("training_use") != "train-split-only":
            raise ValueError("supplemental source must be train-only")
        if supplemental.get("test_used_for_checkpoint_selection") is not False:
            raise ValueError("supplemental test cannot be used for checkpoint selection")
        if supplemental.get("values_persisted") is not False:
            raise ValueError("supplemental records must not persist raw PII values")
        if supplemental.get("unmapped_source_labels_fail_closed") is not True:
            raise ValueError("supplemental label mapping must fail closed")
        if supplemental.get("identity_split", {}).get("verified") is not True:
            raise ValueError("supplemental identity split is not verified")
        if supplemental.get("identity_split", {}).get("source_id_overlap") != 0:
            raise ValueError("supplemental source contains identity leakage")
        if supplemental.get("identity_split", {}).get("identity_marker_overlap") != 0:
            raise ValueError("supplemental source contains identity-marker leakage")
        if supplemental.get("classes") != list(DETECTOR_CLASSES):
            raise ValueError("supplemental class order differs from the detector contract")
        safety = supplemental.get("source", {}).get("safety_verification", {})
        required_safety = {
            "audit_passed": True,
            "failures": 0,
            "capture_mode": "tiles",
            "full_page_downsampling_avoided": True,
            "identity_disjoint": True,
            "fail_on_unmapped": True,
        }
        if any(safety.get(key) != value for key, value in required_safety.items()):
            raise ValueError("supplemental source v2 safety verification is incomplete")
        if not isinstance(safety.get("metadata_sha256"), str) or len(
            safety["metadata_sha256"]
        ) != 64:
            raise ValueError("supplemental source metadata hash verification is missing")
        if supplemental.get("content_hash_algorithm") != CONTENT_HASH_ALGORITHM:
            raise ValueError("supplemental source content-hash algorithm is unsupported")

    train_roots = [
        args.webpii_root.resolve() / "images/train",
        args.synthetic_root.resolve() / "images/train",
    ]
    if supplemental_root is not None:
        train_roots.append(supplemental_root.resolve() / "images/train")
    dataset_yaml = [
        "path: /",
        "train:",
        *[f"  - {root}" for root in train_roots],
        f"val: {args.synthetic_root.resolve() / 'images/validation'}",
        f"test: {args.webpii_root.resolve() / 'images/test'}",
        f"nc: {len(DETECTOR_CLASSES)}",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(DETECTOR_CLASSES)],
        "",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = args.output_dir / "dataset.yaml"
    yaml_path.write_text("\n".join(dataset_yaml), encoding="utf-8")
    train_sources = ["WebPII/train", "PLVA synthetic/train"]
    if supplemental is not None:
        train_sources.append("user-provided ATS synthetic/train")
    sources = {
        "webpii": {
            "manifest": str(webpii_manifest_path),
            "manifest_sha256": sha256_file(webpii_manifest_path),
            "revision": webpii["revision"],
            "license": webpii["license"],
        },
        "synthetic": {
            "manifest": str(synthetic_manifest_path),
            "manifest_sha256": sha256_file(synthetic_manifest_path),
            "license": synthetic["license"],
        },
    }
    if supplemental is not None and supplemental_manifest_path is not None:
        sources["supplemental_ats"] = {
            "manifest": str(supplemental_manifest_path),
            "manifest_sha256": sha256_file(supplemental_manifest_path),
            "license": supplemental["license"],
            "training_split_records": supplemental["splits"]["train"]["records"],
            "test_excluded_from_training_and_selection": True,
            "source_metadata_sha256": supplemental["source"]["metadata"]["sha256"],
            "images_aggregate_sha256": supplemental["content_hashes"][
                "images_aggregate_sha256"
            ],
            "labels_aggregate_sha256": supplemental["content_hashes"][
                "labels_aggregate_sha256"
            ],
            "content_hash_algorithm": supplemental["content_hash_algorithm"],
        }
    manifest = {
        "schema_version": 2 if supplemental is not None else 1,
        "published_splits_preserved": True,
        "test_used_for_checkpoint_selection": False,
        "classes": list(DETECTOR_CLASSES),
        "selection_data": "screen-native-synthetic-validation",
        "train_sources": train_sources,
        "test_source": "WebPII/test",
        "sources": sources,
        "dataset_yaml": str(yaml_path),
        "dataset_yaml_sha256": sha256_file(yaml_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose WebPII train/test with a separate synthetic tuning split"
    )
    parser.add_argument("--webpii-root", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--supplemental-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(compose(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
