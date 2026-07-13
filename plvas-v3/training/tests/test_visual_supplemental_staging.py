from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

from training.schema import sha256_file
from training.visual.prepare_supplemental import (
    CONTENT_HASH_ALGORITHM,
    aggregate_content_hash,
)
from training.visual.prepare_webpii import DETECTOR_CLASSES, yolo_line
from training.visual.stage_supplemental import (
    stage_prepared_supplemental,
    validate_prepared_supplemental,
)


def write_prepared_fixture(root: Path) -> None:
    split_manifests: dict[str, dict[str, object]] = {}
    image_fingerprints: list[tuple[str, str]] = []
    label_fingerprints: list[tuple[str, str]] = []
    for split, class_name in (("train", "NAME"), ("test", "EMAIL")):
        image_relative = f"images/{split}/{split}.png"
        label_relative = f"labels/{split}/{split}.txt"
        image_path = root / image_relative
        label_path = root / label_relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "white").save(image_path)
        annotation = {
            "source_key": "PII_FULLNAME" if split == "train" else "PII_EMAIL",
            "element_type": "input",
            "bbox_xyxy": [1.0, 1.0, 4.0, 4.0],
            "clipped": False,
            "class": class_name,
        }
        label_path.write_text(
            yolo_line(class_name, annotation["bbox_xyxy"], 8, 8) + "\n",
            encoding="utf-8",
        )
        image_hash = sha256_file(image_path)
        label_hash = sha256_file(label_path)
        image_fingerprints.append((image_relative, image_hash))
        label_fingerprints.append((label_relative, label_hash))
        record = {
            "id": f"fixture-{split}",
            "source_id": f"identity-{split}",
            "variant": "full",
            "split": split,
            "width": 8,
            "height": 8,
            "image": image_relative,
            "image_sha256": image_hash,
            "label_file": label_relative,
            "label_sha256": label_hash,
            "annotations": [annotation],
            "hard_negative_boxes": [],
        }
        records_relative = f"records/{split}.jsonl"
        records_path = root / records_relative
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.write_text(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        split_manifests[split] = {
            "records": 1,
            "annotations": 1,
            "hard_negative_boxes": 0,
            "negative_images": 0,
            "offscreen_elements": 0,
            "identities": 1,
            "class_counts": dict(Counter([class_name])),
            "records_path": records_relative,
            "records_sha256": sha256_file(records_path),
        }
    manifest = {
        "schema_version": 2,
        "dataset": "fixture",
        "license": "user-provided-synthetic-data",
        "supplemental_only": True,
        "training_use": "train-split-only",
        "test_used_for_checkpoint_selection": False,
        "classes": list(DETECTOR_CLASSES),
        "unmapped_source_labels_fail_closed": True,
        "values_persisted": False,
        "source": {
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
            "metadata": {"sha256": "1" * 64},
        },
        "identity_split": {
            "verified": True,
            "source_id_overlap": 0,
            "identity_marker_overlap": 0,
        },
        "splits": split_manifests,
        "content_hashes": {
            "images_aggregate_sha256": aggregate_content_hash(image_fingerprints),
            "labels_aggregate_sha256": aggregate_content_hash(label_fingerprints),
        },
        "content_hash_algorithm": CONTENT_HASH_ALGORITHM,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class SupplementalStagingTests(unittest.TestCase):
    def test_validated_copy_is_atomic_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "uploaded"
            destination = root / "run/visual/supplemental"
            source.mkdir()
            write_prepared_fixture(source)

            first = stage_prepared_supplemental(source, destination)
            self.assertEqual(first["mode"], "copied-and-verified")
            self.assertTrue((destination / "staging_provenance.json").is_file())
            self.assertEqual(
                first["source"]["manifest_sha256"],
                first["staged"]["manifest_sha256"],
            )

            second = stage_prepared_supplemental(source, destination)
            self.assertEqual(second["mode"], "reused-validated-staging")
            self.assertIn("staging_provenance_sha256", second["staged"])

    def test_partial_upload_fails_before_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "uploaded"
            source.mkdir()
            write_prepared_fixture(source)
            (source / "labels/train/train.txt").unlink()

            with self.assertRaisesRegex(ValueError, "label.*missing"):
                stage_prepared_supplemental(
                    source,
                    root / "run/visual/supplemental",
                )
            self.assertFalse((root / "run/visual/supplemental").exists())

    def test_unverified_v1_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "prepared"
            root.mkdir(parents=True)
            write_prepared_fixture(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "verified v2 adapter"):
                validate_prepared_supplemental(root)


if __name__ == "__main__":
    unittest.main()
