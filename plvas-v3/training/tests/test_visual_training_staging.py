from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from training.schema import sha256_file
from training.visual.prepare_supplemental import (
    CONTENT_HASH_ALGORITHM,
    aggregate_content_hash,
)
from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.stage_training_dataset import (
    persist_stage_metadata,
    stage_composed_dataset,
    verify_persisted_stage_metadata,
)
from training.visual.train_detector import (
    normalize_class_names,
    prepare_supplemental_test_yaml,
    validate_composed_dataset_contract,
)


def write_simple_source(root: Path, source: str) -> None:
    splits = ("train", "test") if source == "webpii" else ("train", "validation")
    for split in splits:
        image = root / f"images/{split}/fixture.png"
        label = root / f"labels/{split}/fixture.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"fixture-image")
        label.write_text("", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "classes": list(DETECTOR_CLASSES),
        "license": "fixture",
        **(
            {
                "published_splits_preserved": True,
                "test_used_for_checkpoint_selection": False,
            }
            if source == "webpii"
            else {"synthetic_only": True}
        ),
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_supplemental(root: Path) -> dict[str, object]:
    image_fingerprints: list[tuple[str, str]] = []
    label_fingerprints: list[tuple[str, str]] = []
    splits: dict[str, dict[str, object]] = {}
    for split, count in (("train", 1), ("test", 200)):
        records: list[str] = []
        for index in range(count):
            image_relative = f"images/{split}/{index:03d}.png"
            label_relative = f"labels/{split}/{index:03d}.txt"
            image = root / image_relative
            label = root / label_relative
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (2, 2), "white").save(image)
            label.write_text("", encoding="utf-8")
            image_hash = sha256_file(image)
            label_hash = sha256_file(label)
            image_fingerprints.append((image_relative, image_hash))
            label_fingerprints.append((label_relative, label_hash))
            records.append(
                json.dumps(
                    {
                        "id": f"{split}-{index}",
                        "source_id": f"identity-{split}-{index}",
                        "variant": "empty",
                        "split": split,
                        "width": 2,
                        "height": 2,
                        "image": image_relative,
                        "image_sha256": image_hash,
                        "label_file": label_relative,
                        "label_sha256": label_hash,
                        "annotations": [],
                        "hard_negative_boxes": [],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        records_path = root / f"records/{split}.jsonl"
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.write_text("\n".join(records) + "\n", encoding="utf-8")
        splits[split] = {
            "records": count,
            "annotations": 0,
            "hard_negative_boxes": 0,
            "negative_images": count,
            "offscreen_elements": 0,
            "identities": count,
            "class_counts": {},
            "records_path": f"records/{split}.jsonl",
            "records_sha256": sha256_file(records_path),
        }
    content_hashes = {
        "images_aggregate_sha256": aggregate_content_hash(image_fingerprints),
        "labels_aggregate_sha256": aggregate_content_hash(label_fingerprints),
    }
    manifest = {
        "schema_version": 2,
        "license": "fixture",
        "supplemental_only": True,
        "training_use": "train-split-only",
        "test_used_for_checkpoint_selection": False,
        "values_persisted": False,
        "unmapped_source_labels_fail_closed": True,
        "classes": list(DETECTOR_CLASSES),
        "identity_split": {
            "verified": True,
            "source_id_overlap": 0,
            "identity_marker_overlap": 0,
        },
        "source": {
            "metadata": {"sha256": "1" * 64},
            "safety_verification": {
                "audit_passed": True,
                "failures": 0,
                "capture_mode": "tiles",
                "full_page_downsampling_avoided": True,
                "identity_disjoint": True,
                "fail_on_unmapped": True,
                "metadata_sha256": "1" * 64,
                "source_manifest_sha256": "2" * 64,
                "label_policy_sha256": "3" * 64,
            },
        },
        "splits": splits,
        "content_hashes": content_hashes,
        "content_hash_algorithm": CONTENT_HASH_ALGORITHM,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "license": "fixture",
        "training_split_records": 1,
        "test_excluded_from_training_and_selection": True,
        "source_metadata_sha256": "1" * 64,
        **content_hashes,
        "content_hash_algorithm": CONTENT_HASH_ALGORITHM,
    }


def write_composed(root: Path) -> tuple[Path, Path, dict[str, Path]]:
    webpii = root / "webpii"
    synthetic = root / "synthetic"
    supplemental = root / "supplemental"
    write_simple_source(webpii, "webpii")
    write_simple_source(synthetic, "synthetic")
    supplemental_source = write_supplemental(supplemental)
    composed = root / "composed"
    composed.mkdir()
    dataset_yaml = composed / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                "path: /",
                "train:",
                f"  - {webpii / 'images/train'}",
                f"  - {synthetic / 'images/train'}",
                f"  - {supplemental / 'images/train'}",
                f"val: {synthetic / 'images/validation'}",
                f"test: {webpii / 'images/test'}",
                f"nc: {len(DETECTOR_CLASSES)}",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(DETECTOR_CLASSES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    source_manifest = composed / "manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "published_splits_preserved": True,
                "test_used_for_checkpoint_selection": False,
                "selection_data": "screen-native-synthetic-validation",
                "test_source": "WebPII/test",
                "classes": list(DETECTOR_CLASSES),
                "dataset_yaml": str(dataset_yaml),
                "dataset_yaml_sha256": sha256_file(dataset_yaml),
                "sources": {
                    "webpii": {
                        "manifest": str(webpii / "manifest.json"),
                        "manifest_sha256": sha256_file(webpii / "manifest.json"),
                    },
                    "synthetic": {
                        "manifest": str(synthetic / "manifest.json"),
                        "manifest_sha256": sha256_file(synthetic / "manifest.json"),
                    },
                    "supplemental_ats": supplemental_source,
                },
            }
        ),
        encoding="utf-8",
    )
    return (
        dataset_yaml,
        source_manifest,
        {
            "webpii": webpii,
            "synthetic": synthetic,
            "supplemental_ats": supplemental,
        },
    )


class LocalVisualDatasetStagingTests(unittest.TestCase):
    def test_stage_rewrites_exact_contract_and_keeps_supplemental_test(self) -> None:
        import yaml

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_yaml, source_manifest, roots = write_composed(root / "volume")
            local = root / "ephemeral/run"
            result = stage_composed_dataset(
                dataset_yaml,
                source_manifest,
                roots,
                local,
                copy_workers=64,
            )
            self.assertEqual(result["mode"], "copied-and-verified-local-stage")
            self.assertEqual(result["metadata"]["copy_workers"], 64)
            local_dataset = yaml.safe_load(
                Path(result["dataset_yaml"]).read_text(encoding="utf-8")
            )
            local_manifest = json.loads(
                Path(result["source_manifest"]).read_text(encoding="utf-8")
            )
            names = validate_composed_dataset_contract(
                local_dataset,
                local_manifest,
                Path(result["source_manifest"]),
            )
            self.assertNotIn("/test", "\n".join(local_dataset["train"]))
            evaluation_dir = root / "evaluation"
            evaluation_dir.mkdir()
            supplemental_yaml = prepare_supplemental_test_yaml(
                local_manifest,
                Path(result["source_manifest"]),
                evaluation_dir,
                normalize_class_names(list(DETECTOR_CLASSES), "fixture"),
            )
            self.assertIsNotNone(supplemental_yaml)
            assert supplemental_yaml is not None
            self.assertIn(
                str(local / "supplemental_ats"),
                supplemental_yaml.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                names, normalize_class_names(list(DETECTOR_CLASSES), "fixture")
            )

            persisted = persist_stage_metadata(result, root / "durable")
            self.assertTrue(Path(persisted["dataset_yaml_path"]).is_file())
            reused = stage_composed_dataset(
                dataset_yaml,
                source_manifest,
                roots,
                local,
            )
            self.assertEqual(reused["mode"], "reused-verified-local-stage")
            self.assertEqual(
                verify_persisted_stage_metadata(reused, root / "durable"),
                persisted,
            )
            self.assertEqual(reused["metadata"]["copy_workers"], 64)
            Path(reused["dataset_yaml"]).write_text(
                "tampered: true\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "differs from the original"):
                verify_persisted_stage_metadata(reused, root / "durable")

    def test_webpii_test_cannot_enter_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_yaml, source_manifest, roots = write_composed(root / "volume")
            text = dataset_yaml.read_text(encoding="utf-8")
            text = text.replace(
                str(roots["webpii"] / "images/train"),
                str(roots["webpii"] / "images/test"),
            )
            dataset_yaml.write_text(text, encoding="utf-8")
            manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
            manifest["dataset_yaml_sha256"] = sha256_file(dataset_yaml)
            source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "train paths must be exactly"):
                stage_composed_dataset(
                    dataset_yaml,
                    source_manifest,
                    roots,
                    root / "ephemeral/run",
                )


if __name__ == "__main__":
    unittest.main()
