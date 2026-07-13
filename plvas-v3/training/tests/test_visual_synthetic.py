from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from training.synth_screen import SECRET_TEMPLATE_LABELS, SENSITIVE_IMAGE_KINDS
from training.visual.prepare_synthetic import prepare
from training.visual.prepare_webpii import DETECTOR_CLASSES


class VisualSyntheticTests(unittest.TestCase):
    def test_balances_all_detector_classes_and_sensitive_subtypes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "synthetic"
            manifest = prepare(
                argparse.Namespace(
                    output_dir=output,
                    train_records=200,
                    validation_records=200,
                    seed=1311,
                    force=False,
                )
            )

            self.assertEqual(manifest["schema_version"], 2)
            for split in ("train", "validation"):
                summary = manifest["splits"][split]
                self.assertEqual(
                    summary["class_counts"],
                    {class_name: 20 for class_name in sorted(DETECTOR_CLASSES)},
                )
                self.assertEqual(
                    summary["secret_source_counts"],
                    {
                        source_class: 5
                        for source_class in sorted(SECRET_TEMPLATE_LABELS)
                    },
                )
                self.assertEqual(
                    summary["sensitive_image_kind_counts"],
                    {kind: 5 for kind in sorted(SENSITIVE_IMAGE_KINDS)},
                )
                self.assertEqual(summary["hard_negative_boxes"], 40)
                self.assertEqual(summary["pure_negative_records"], 20)
                self.assertTrue(summary["support_validation"]["passed"])

    def test_split_templates_seeds_images_and_negatives_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "synthetic"
            manifest = prepare(
                argparse.Namespace(
                    output_dir=output,
                    train_records=100,
                    validation_records=100,
                    seed=1311,
                    force=False,
                )
            )
            records: dict[str, list[dict]] = {}
            for split in ("train", "validation"):
                path = output / manifest["splits"][split]["records_path"]
                records[split] = [
                    json.loads(line) for line in path.read_text().splitlines()
                ]

            train_templates = {record["template_id"] for record in records["train"]}
            validation_templates = {
                record["template_id"] for record in records["validation"]
            }
            self.assertFalse(train_templates & validation_templates)
            self.assertFalse(
                {record["seed"] for record in records["train"]}
                & {record["seed"] for record in records["validation"]}
            )
            self.assertFalse(
                {record["image_sha256"] for record in records["train"]}
                & {record["image_sha256"] for record in records["validation"]}
            )

            for split_records in records.values():
                image_records = [
                    record
                    for record in split_records
                    if any(
                        annotation["class"] == "SENSITIVE_IMAGE"
                        for annotation in record["annotations"]
                    )
                ]
                self.assertTrue(image_records)
                for record in image_records:
                    image_annotations = [
                        annotation
                        for annotation in record["annotations"]
                        if annotation["class"] == "SENSITIVE_IMAGE"
                    ]
                    self.assertEqual(len(image_annotations), 1)
                    label = (output / record["label_file"]).read_text().strip()
                    self.assertEqual(
                        label.split()[0],
                        str(DETECTOR_CLASSES.index("SENSITIVE_IMAGE")),
                    )
                    self.assertEqual(
                        record["hard_negative_boxes"][0]["reason"],
                        "synthetic-sensitive-image-caption",
                    )

                pure_negatives = [
                    record for record in split_records if not record["annotations"]
                ]
                self.assertTrue(pure_negatives)
                for record in pure_negatives:
                    self.assertEqual(
                        (output / record["label_file"]).read_text(), ""
                    )
                    self.assertEqual(
                        record["hard_negative_boxes"][0]["reason"],
                        "synthetic-negative-ui-text",
                    )


if __name__ == "__main__":
    unittest.main()
