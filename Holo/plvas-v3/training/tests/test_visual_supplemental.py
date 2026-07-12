from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from training.visual.compose_dataset import compose
from training.visual.prepare_supplemental import (
    CONTENT_HASH_ALGORITHM,
    EXPLICIT_HARD_NEGATIVE_KEYS,
    PII_SOURCE_MAPPING,
    aggregate_content_hash,
    prepare,
    source_disposition,
)
from training.visual.prepare_webpii import DETECTOR_CLASSES, clamp_box


def element(
    key: str,
    value: str,
    *,
    x: float = 10,
    y: float = 10,
    width: float = 20,
    height: float = 10,
    visible: bool = True,
    clipped: bool = False,
) -> dict[str, object]:
    return {
        "key": key,
        "value": value,
        "bbox_x": x,
        "bbox_y": y,
        "bbox_width": width,
        "bbox_height": height,
        "visible": visible,
        "clipped": clipped,
        "element_type": "input",
    }


def metadata_row(
    source_id: str,
    split: str,
    image: str,
    pii: list[dict[str, object]],
    misc: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    groups = {
        "pii": pii,
        "misc": misc or [],
        "product": [],
        "order": [],
        "search": [],
    }
    return {
        "source_id": source_id,
        "variant": "full",
        "split": split,
        "image": image,
        "image_width": 100,
        "image_height": 80,
        "company": "fixture-company",
        "page_type": "job-application",
        "capture_mode": "tile",
        "generator_version": "2.0.0-test",
        "fillable_count": len(pii),
        **{
            f"{name}_elements_json": json.dumps(items)
            for name, items in groups.items()
        },
        **{f"num_{name}_elements": len(items) for name, items in groups.items()},
    }


def write_source(root: Path, rows: list[dict[str, object]]) -> None:
    (root / "images").mkdir(parents=True)
    for row in rows:
        image_path = root / str(row["image"])
        Image.new("RGB", (100, 80), "white").save(image_path)
        row["image_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
    metadata = root / "metadata.jsonl"
    metadata.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    source_keys = {
        item["key"]
        for row in rows
        for field in (
            "pii_elements_json",
            "misc_elements_json",
            "product_elements_json",
            "order_elements_json",
            "search_elements_json",
        )
        for item in json.loads(str(row[field]))
    }
    source_keys.add("MISC_COMPANY")
    policy = {
        "schema_version": 1,
        "fail_on_unmapped": True,
        "source_dispositions": {
            key: {
                "disposition": source_disposition(str(key))[0],
                "class": source_disposition(str(key))[1],
            }
            for key in sorted(source_keys)
        },
    }
    policy_path = root / "plva-label-policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    metadata_sha = hashlib.sha256(metadata.read_bytes()).hexdigest()
    policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    image_set = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["image"])):
        image_set.update(f"{row['image']} {row['image_sha256']}\n".encode())
    split_counts = {
        split: sum(row["split"] == split for row in rows) for split in ("train", "test")
    }
    manifest = {
        "schema_version": 2,
        "generation": {"rendered_images": len(rows)},
        "splits": {"identity_disjoint": True, "counts": split_counts},
        "capture": {
            "mode": "tiles",
            "full_page_downsampling_avoided": True,
        },
        "label_policy": {
            "path": "plva-label-policy.json",
            "fail_on_unmapped": True,
            "company_is_hard_negative": True,
        },
        "audit": {
            "passed": True,
            "records": len(rows),
            "identities": len({str(row["source_id"]) for row in rows}),
            "identity_disjoint": True,
            "split_counts": split_counts,
            "metadata_sha256": metadata_sha,
        },
        "artifacts": {
            "metadata.jsonl": {"sha256": metadata_sha},
            "plva-label-policy.json": {"sha256": policy_sha},
        },
        "image_set_sha256": image_set.hexdigest(),
        "failures": 0,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def set_nested(mapping: dict[str, object], path: tuple[str, ...], value: object) -> None:
    current = mapping
    for key in path[:-1]:
        child = current[key]
        assert isinstance(child, dict)
        current = child
    current[path[-1]] = value


class SupplementalMappingTests(unittest.TestCase):
    def test_mapping_is_explicit_and_company_is_a_hard_negative(self) -> None:
        self.assertEqual(source_disposition("PII_FULLNAME"), ("map", "NAME"))
        self.assertEqual(source_disposition("PII_LINKEDIN"), ("map", "SENSITIVE_FIELD"))
        self.assertEqual(source_disposition("PII_COMPANY"), ("hard-negative", None))
        self.assertEqual(source_disposition("MISC_COMPANY"), ("hard-negative", None))
        self.assertEqual(source_disposition("PII_NEW_UNREVIEWED"), ("unmapped", None))
        self.assertEqual(
            set(PII_SOURCE_MAPPING) | EXPLICIT_HARD_NEGATIVE_KEYS,
            {
                "PII_FULLNAME",
                "PII_FULLNAME2",
                "PII_EMAIL",
                "PII_EMAIL2",
                "PII_EMAIL3",
                "PII_EMAIL4",
                "PII_PHONE",
                "PII_CITY_STATE",
                "PII_STREET",
                "PII_LINKEDIN",
                "PII_RESUME_FILENAME",
                "PII_SALARY",
                "PII_START_DATE",
                "PII_US_BASED",
                "PII_VISA_SPONSORSHIP",
                "PII_VOICE_FILENAME",
                "PII_WEBSITE",
                "PII_PRONOUNS",
                "PII_COMPANY",
                "MISC_APPLIED_BEFORE",
                "MISC_COMPANY",
                "MISC_NOTICE_PERIOD",
                "MISC_SOURCE",
            },
        )

    def test_clamp_uses_the_raw_far_edge(self) -> None:
        self.assertEqual(
            clamp_box(
                element(
                    "PII_EMAIL",
                    "private@example.test",
                    x=-10,
                    width=20,
                    clipped=True,
                ),
                100,
                80,
            ),
            [0.0, 10.0, 10.0, 20.0],
        )

    def test_aggregate_content_hash_is_canonical_and_order_independent(self) -> None:
        entries = [
            ("images/z.png", "b" * 64),
            ("images/a.png", "a" * 64),
        ]
        canonical = (
            b"images/a.png\0" + b"a" * 64 + b"\n"
            + b"images/z.png\0" + b"b" * 64 + b"\n"
        )
        expected = hashlib.sha256(canonical).hexdigest()
        self.assertEqual(aggregate_content_hash(entries), expected)
        self.assertEqual(aggregate_content_hash(list(reversed(entries))), expected)
        self.assertEqual(
            CONTENT_HASH_ALGORITHM["name"],
            "sha256-sorted-path-nul-file-sha256-lf-v1",
        )


class SupplementalPrepareTests(unittest.TestCase):
    def _valid_rows(self) -> list[dict[str, object]]:
        return [
            metadata_row(
                "identity-train",
                "train",
                "images/train.png",
                [
                    element(
                        "PII_FULLNAME",
                        "Sensitive Train Name",
                        x=-10,
                        width=20,
                        clipped=True,
                    ),
                    element("PII_EMAIL", "private-train@example.test", x=25),
                ],
                [
                    element("MISC_COMPANY", "Public Employer", x=50),
                    element("MISC_SOURCE", "Friend", x=75),
                ],
            ),
            metadata_row(
                "identity-test",
                "test",
                "images/test.png",
                [
                    element("PII_PHONE", "+1 555 010 9000"),
                    element("PII_LINKEDIN", "https://example.test/private-profile", x=40),
                ],
            ),
        ]

    def test_prepare_writes_private_value_free_yolo_data_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "prepared"
            rows = self._valid_rows()
            write_source(source, rows)
            manifest = prepare(
                argparse.Namespace(source_root=source, output_dir=output, force=False)
            )

            self.assertTrue(manifest["identity_split"]["verified"])
            self.assertEqual(manifest["identity_split"]["source_id_overlap"], 0)
            self.assertFalse(manifest["values_persisted"])
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["content_hash_algorithm"], CONTENT_HASH_ALGORITHM)
            self.assertEqual(
                manifest["source"]["safety_verification"]["capture_mode"], "tiles"
            )
            self.assertTrue(
                manifest["source"]["safety_verification"][
                    "full_page_downsampling_avoided"
                ]
            )
            self.assertEqual(manifest["splits"]["train"]["records"], 1)
            self.assertEqual(manifest["splits"]["test"]["records"], 1)
            self.assertEqual(manifest["splits"]["train"]["class_counts"]["NAME"], 1)
            self.assertEqual(
                manifest["splits"]["test"]["class_counts"]["SENSITIVE_FIELD"], 1
            )

            record = json.loads((output / "records/train.jsonl").read_text())
            self.assertEqual(record["annotations"][0]["bbox_xyxy"], [0.0, 10.0, 10.0, 20.0])
            self.assertEqual(
                record["hard_negative_boxes"][0]["source_key"], "MISC_COMPANY"
            )
            label = (output / record["label_file"]).read_text().splitlines()[0]
            self.assertEqual(label.split()[0], str(DETECTOR_CLASSES.index("NAME")))

            persisted_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output.rglob("*")
                if path.is_file() and path.suffix in {".json", ".jsonl", ".txt"}
            )
            for raw_value in (
                "Sensitive Train Name",
                "private-train@example.test",
                "Public Employer",
                "+1 555 010 9000",
                "https://example.test/private-profile",
            ):
                self.assertNotIn(raw_value, persisted_text)

    def test_v2_source_manifest_safety_gates_fail_closed(self) -> None:
        cases = (
            ("schema", ("schema_version",), 1, "schema_version"),
            ("render failures", ("failures",), 1, "failures"),
            ("audit", ("audit", "passed"), False, "audit must be passed"),
            (
                "audit identity",
                ("audit", "identity_disjoint"),
                False,
                "audit must verify identity disjointness",
            ),
            (
                "split identity",
                ("splits", "identity_disjoint"),
                False,
                "identity-disjoint splits",
            ),
            ("capture mode", ("capture", "mode"), "full-page", "mode must be tiles"),
            (
                "downsampling",
                ("capture", "full_page_downsampling_avoided"),
                False,
                "avoid full-page downsampling",
            ),
            (
                "unmapped policy",
                ("label_policy", "fail_on_unmapped"),
                False,
                "fail_on_unmapped",
            ),
            (
                "metadata artifact hash",
                ("artifacts", "metadata.jsonl", "sha256"),
                "0" * 64,
                "metadata hash does not match manifest",
            ),
            (
                "metadata audit hash",
                ("audit", "metadata_sha256"),
                "0" * 64,
                "metadata hash does not match audit",
            ),
        )
        for name, path, value, error in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "source"
                write_source(source, self._valid_rows())
                manifest_path = source / "manifest.json"
                source_manifest = json.loads(manifest_path.read_text())
                set_nested(source_manifest, path, value)
                manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    prepare(
                        argparse.Namespace(
                            source_root=source,
                            output_dir=root / "prepared",
                            force=False,
                        )
                    )

    def test_missing_v2_source_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            write_source(source, self._valid_rows())
            (source / "manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "requires a v2 manifest"):
                prepare(
                    argparse.Namespace(
                        source_root=source,
                        output_dir=root / "prepared",
                        force=False,
                    )
                )

    def test_identity_split_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            rows = self._valid_rows()
            rows[1]["source_id"] = rows[0]["source_id"]
            rows[1]["variant"] = "partial_00"
            write_source(source, rows)
            with self.assertRaisesRegex(ValueError, "both train and test"):
                prepare(
                    argparse.Namespace(
                        source_root=source,
                        output_dir=root / "prepared",
                        force=False,
                    )
                )

    def test_unknown_source_key_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            rows = self._valid_rows()
            unknown = element("PII_UNREVIEWED", "do-not-guess")
            pii = json.loads(str(rows[0]["pii_elements_json"]))
            pii.append(unknown)
            rows[0]["pii_elements_json"] = json.dumps(pii)
            rows[0]["num_pii_elements"] = len(pii)
            write_source(source, rows)
            with self.assertRaisesRegex(ValueError, "unmapped supplemental key"):
                prepare(
                    argparse.Namespace(
                        source_root=source,
                        output_dir=root / "prepared",
                        force=False,
                    )
                )

    def test_clipped_flag_must_match_box_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            rows = self._valid_rows()
            pii = json.loads(str(rows[0]["pii_elements_json"]))
            pii[0]["clipped"] = False
            rows[0]["pii_elements_json"] = json.dumps(pii)
            write_source(source, rows)
            with self.assertRaisesRegex(ValueError, "clipped flag disagrees"):
                prepare(
                    argparse.Namespace(
                        source_root=source,
                        output_dir=root / "prepared",
                        force=False,
                    )
                )

    def test_compose_adds_only_the_supplemental_train_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            supplemental = root / "supplemental"
            write_source(source, self._valid_rows())
            prepare(
                argparse.Namespace(
                    source_root=source,
                    output_dir=supplemental,
                    force=False,
                )
            )
            webpii = root / "webpii"
            synthetic = root / "synthetic"
            webpii.mkdir()
            synthetic.mkdir()
            (webpii / "manifest.json").write_text(
                json.dumps(
                    {
                        "published_splits_preserved": True,
                        "test_used_for_checkpoint_selection": False,
                        "classes": list(DETECTOR_CLASSES),
                        "revision": "fixture",
                        "license": "fixture",
                    }
                ),
                encoding="utf-8",
            )
            (synthetic / "manifest.json").write_text(
                json.dumps(
                    {
                        "synthetic_only": True,
                        "classes": list(DETECTOR_CLASSES),
                        "license": "fixture",
                    }
                ),
                encoding="utf-8",
            )
            composed = compose(
                argparse.Namespace(
                    webpii_root=webpii,
                    synthetic_root=synthetic,
                    supplemental_root=supplemental,
                    output_dir=root / "composed",
                )
            )
            yaml_text = (root / "composed/dataset.yaml").read_text()
            self.assertIn(str(supplemental / "images/train"), yaml_text)
            self.assertNotIn(str(supplemental / "images/test"), yaml_text)
            self.assertEqual(composed["schema_version"], 2)
            self.assertTrue(
                composed["sources"]["supplemental_ats"][
                    "test_excluded_from_training_and_selection"
                ]
            )
            supplemental_manifest_path = supplemental / "manifest.json"
            stale = json.loads(supplemental_manifest_path.read_text())
            stale["schema_version"] = 1
            supplemental_manifest_path.write_text(json.dumps(stale), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verified v2 dataset"):
                compose(
                    argparse.Namespace(
                        webpii_root=webpii,
                        synthetic_root=synthetic,
                        supplemental_root=supplemental,
                        output_dir=root / "stale-composed",
                    )
                )


if __name__ == "__main__":
    unittest.main()
