from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from training.prepare_data import (
    DATASET_REVISION,
    SplitFilter,
    convert_openpii_row,
    prepare,
)
from training.schema import (
    LABEL_CONFIG,
    iter_jsonl,
    record_values,
    validate_record,
)
from training.synth_screen import apply_ocr_noise, generate_records


def luhn(value: str) -> bool:
    digits = [int(char) for char in value if char.isdigit()]
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


class LabelTests(unittest.TestCase):
    def test_label_order_is_runtime_contract(self) -> None:
        expected = ["O"]
        for entity_class in LABEL_CONFIG.entity_classes:
            expected.extend((f"B-{entity_class}", f"I-{entity_class}"))
        self.assertEqual(list(LABEL_CONFIG.labels), expected)
        self.assertEqual(len(LABEL_CONFIG.labels), 27)


class SyntheticTests(unittest.TestCase):
    def test_records_validate_and_cover_every_class(self) -> None:
        seen_classes: set[str] = set()
        for split in ("train", "validation", "holdout"):
            records = list(generate_records(split, 64, seed=1311, noise_fraction=0.5))
            for record in records:
                validate_record(record)
                seen_classes.update(span["label"] for span in record["spans"])
        self.assertEqual(seen_classes, set(LABEL_CONFIG.entity_classes))

    def test_template_families_and_values_are_split_disjoint(self) -> None:
        by_split = {
            split: list(generate_records(split, 160, seed=1311, noise_fraction=0.0))
            for split in ("train", "validation", "holdout")
        }
        templates = {
            split: {record["template_id"] for record in records}
            for split, records in by_split.items()
        }
        self.assertFalse(templates["train"] & templates["validation"])
        self.assertFalse(templates["train"] & templates["holdout"])
        values = {
            split: set().union(*(record_values(record) for record in records))
            for split, records in by_split.items()
        }
        self.assertFalse(values["train"] & values["validation"])
        self.assertFalse(values["train"] & values["holdout"])

    def test_generated_cards_pass_luhn(self) -> None:
        cards = [
            span["value"]
            for record in generate_records("train", 128, noise_fraction=0.0)
            for span in record["spans"]
            if span["label"] == "CARD_NUMBER"
        ]
        self.assertTrue(cards)
        self.assertTrue(all(luhn(card) for card in cards))

    def test_hard_negative_pixels_can_be_split_disjoint(self) -> None:
        train = list(generate_records("train", 32, seed=1311, noise_fraction=0.0))
        validation = list(
            generate_records("validation", 32, seed=1311, noise_fraction=0.0)
        )
        train_negatives = {
            record["text"] for record in train if not record["spans"]
        }
        validation_negatives = {
            record["text"] for record in validation if not record["spans"]
        }
        self.assertTrue(train_negatives)
        self.assertTrue(validation_negatives)
        self.assertFalse(train_negatives & validation_negatives)

    def test_noise_preserves_span_round_trip(self) -> None:
        import random

        record = next(generate_records("train", 1, noise_fraction=0.0))
        text, spans, _ = apply_ocr_noise(
            record["text"],
            record["spans"],
            random.Random(7),
            replace_rate=1.0,
            delete_rate=0.02,
        )
        noisy = record | {"text": text, "spans": spans, "noisy": True}
        validate_record(noisy)


class OpenPiiMappingTests(unittest.TestCase):
    def test_maps_actual_schema_and_merges_names(self) -> None:
        text = (
            "Avery Chen was born on 03/22/1988 and can be reached at avery@example.com."
        )
        row = {
            "uid": "fixture-1",
            "source_text": text,
            "language": "en-US",
            "region": "US",
            "privacy_mask": [
                {"start": 0, "end": 5, "label": "GIVENNAME", "value": "Avery"},
                {"start": 6, "end": 10, "label": "SURNAME", "value": "Chen"},
                {"start": 23, "end": 33, "label": "DATE", "value": "03/22/1988"},
                {
                    "start": 56,
                    "end": 73,
                    "label": "EMAIL",
                    "value": "avery@example.com",
                },
            ],
        }
        record = convert_openpii_row(row, "train", negative_ratio=0.0)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(
            [(span["label"], span["value"]) for span in record["spans"]],
            [
                ("NAME", "Avery Chen"),
                ("DOB", "03/22/1988"),
                ("EMAIL", "avery@example.com"),
            ],
        )

    def test_generic_date_is_not_dob(self) -> None:
        text = "The release date is 07/11/2026."
        row = {
            "uid": "fixture-date",
            "source_text": text,
            "language": "en",
            "region": "US",
            "privacy_mask": [
                {"start": 20, "end": 30, "label": "DATE", "value": "07/11/2026"}
            ],
        }
        self.assertIsNone(convert_openpii_row(row, "train", negative_ratio=0.0))

    def test_split_filter_rejects_value_overlap(self) -> None:
        train = next(generate_records("train", 1, noise_fraction=0.0))
        leaked = train | {"id": "validation-leak", "split": "validation"}
        split_filter = SplitFilter()
        self.assertEqual(len(list(split_filter.filter([train], "train"))), 1)
        self.assertEqual(len(list(split_filter.filter([leaked], "validation"))), 0)
        self.assertEqual(split_filter.skipped_overlap["validation"], 1)


class PrepareTests(unittest.TestCase):
    def test_synthetic_only_prepare_writes_measured_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = prepare(
                argparse.Namespace(
                    output_dir=output,
                    seed=1311,
                    dataset_revision=DATASET_REVISION,
                    max_openpii_train=0,
                    max_openpii_validation=0,
                    synth_train=64,
                    synth_validation=32,
                    synth_holdout=32,
                    noise_fraction=0.35,
                    negative_ratio=0.15,
                    include_openpii=False,
                )
            )
            self.assertEqual(manifest["files"]["train"]["records"], 64)
            self.assertEqual(manifest["files"]["validation"]["records"], 32)
            self.assertEqual(manifest["files"]["holdout"]["records"], 32)
            measured = json.loads((output / "data_manifest.json").read_text())
            self.assertEqual(
                measured["files"]["train"]["sha256"],
                manifest["files"]["train"]["sha256"],
            )
            for split in ("train", "validation", "holdout"):
                for record in iter_jsonl(output / f"{split}.jsonl"):
                    validate_record(record)

    def test_attaches_never_trained_ocr_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source-ocr.jsonl"
            source_manifest = root / "source-ocr-manifest.json"
            record = {
                "id": "ocr-holdout:fixture:0",
                "text": "never-trained-only",
                "spans": [
                    {
                        "start": 0,
                        "end": 18,
                        "label": "PASSWORD",
                        "value": "never-trained-only",
                    }
                ],
                "split": "holdout",
                "source": "ocr_stack_holdout",
                "template_id": "ocr.fixture",
                "seed": 1,
                "language": "en",
                "region": "SYNTHETIC_OCR",
                "noisy": True,
                "provenance": {"never_train": True},
            }
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")
            source_manifest.write_text("{}\n", encoding="utf-8")
            output = root / "prepared"
            manifest = prepare(
                argparse.Namespace(
                    output_dir=output,
                    seed=1311,
                    dataset_revision=DATASET_REVISION,
                    max_openpii_train=0,
                    max_openpii_validation=0,
                    synth_train=16,
                    synth_validation=16,
                    synth_holdout=16,
                    noise_fraction=0.35,
                    negative_ratio=0.15,
                    include_openpii=False,
                    ocr_holdout=source,
                    ocr_holdout_manifest=source_manifest,
                )
            )
            self.assertTrue(manifest["files"]["ocr_holdout"]["never_train"])
            attached = list(iter_jsonl(output / "ocr_holdout.jsonl"))
            self.assertEqual(attached, [record])


if __name__ == "__main__":
    unittest.main()
