from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError


ROOT = Path(__file__).resolve().parents[2]


class LockAndGateTests(unittest.TestCase):
    def test_model_lock_has_complete_default_bootstrap(self) -> None:
        lock = json.loads((ROOT / "models.lock.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["schema_version"], 2)
        default_assets = {
            name for name, asset in lock["assets"].items() if asset["default_fetch"]
        }
        bootstrap_assets = set(
            lock["model_sets"]["full_hybrid_node_bootstrap"]["assets"]
        )
        self.assertEqual(default_assets, bootstrap_assets)
        self.assertEqual(
            sum(lock["assets"][name]["bytes"] for name in default_assets),
            lock["model_sets"]["full_hybrid_node_bootstrap"]["runtime_bytes"],
        )

    def test_replacement_requires_a_frozen_baseline(self) -> None:
        from training.baseline_gate import (
            RAMPART_REVISION,
            require_frozen_rampart_baseline,
        )

        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "baseline.json"
            document = {
                "bootstrap_model": {"revision": RAMPART_REVISION},
                "baseline_frozen": False,
                "evaluation": {"all_required_gates_passed": False},
            }
            manifest.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not frozen"):
                require_frozen_rampart_baseline(manifest)
            document["baseline_frozen"] = True
            document["evaluation"]["all_required_gates_passed"] = True
            manifest.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(
                require_frozen_rampart_baseline(manifest)["bootstrap_model"][
                    "revision"
                ],
                RAMPART_REVISION,
            )

    def test_release_checks_fail_closed(self) -> None:
        from training.export_onnx import release_checks

        quantization = {
            "mean_argmax_agreement": 1.0,
            "mean_probability_absolute_error": 0.0,
            "holdout_recall_delta": 0.0,
            "minimum_holdout_recall_delta": 0.0,
        }
        cross_runtime = {
            "passed": True,
            "backends": {
                name: {"status": "passed"} for name in ("node", "wasm", "webgpu")
            },
        }
        passing = release_checks(
            training_run={
                "baseline_gate": {
                    "status": "frozen",
                    "baseline_frozen": True,
                    "replacement_release_allowed": True,
                }
            },
            evaluation={"gates": {"passed": True}},
            quantization=quantization,
            cross_runtime=cross_runtime,
            artifact_license="Apache-2.0",
        )
        self.assertTrue(passing["passed"])
        failing = release_checks(
            training_run={"baseline_gate": {"status": "development-override"}},
            evaluation={"gates": {"passed": True}},
            quantization=quantization,
            cross_runtime=cross_runtime,
            artifact_license="Apache-2.0",
        )
        self.assertFalse(failing["passed"])

    def test_cross_runtime_report_validates_runtime_and_hashes(self) -> None:
        from training.export_onnx import cross_runtime_status

        model_hash = "a" * 64
        golden_hash = "b" * 64
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "model_sha256": model_hash,
                        "golden_vectors_sha256": golden_hash,
                        "backends": {
                            "node": {
                                "passed": True,
                                "package": "wrong-package",
                                "package_version": "1.27.0",
                                "vectors_checked": 2,
                                "failed_vector_ids": [],
                                "execution_providers_requested": ["cpu"],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = cross_runtime_status(
                report_path,
                model_sha256=model_hash,
                golden_vectors_sha256=golden_hash,
                vector_count=2,
            )
            self.assertFalse(result["passed"])
            self.assertEqual(result["backends"]["node"]["status"], "failed")

    def test_tensor_signature_mismatch_is_rejected(self) -> None:
        from training.inspect_models import validate_contract

        description = {
            "opsets": {"ai.onnx": 17},
            "inputs": [
                {"name": "input_ids", "dtype": "tensor(int64)", "shape": [1, 8]}
            ],
            "outputs": [
                {"name": "logits", "dtype": "tensor(float)", "shape": [1, 8, 27]}
            ],
        }
        with self.assertRaisesRegex(ValueError, "missing tensor attention_mask"):
            validate_contract(
                description,
                {
                    "opset": 17,
                    "inputs": [
                        {
                            "name": "attention_mask",
                            "dtype": "tensor(int64)",
                            "rank": 2,
                        }
                    ],
                    "outputs": [
                        {"name": "logits", "dtype": "tensor(float)", "rank": 3}
                    ],
                },
                "fixture",
            )

    def test_detector_training_requires_an_explicit_base_license(self) -> None:
        from training.visual.train_detector import train

        with self.assertRaisesRegex(RuntimeError, "explicit license"):
            train(argparse.Namespace(base_license="NOASSERTION"))

    def test_webpii_source_mapping_covers_observed_variants(self) -> None:
        from training.visual.prepare_webpii import source_disposition

        expected = {
            "PII_NAME_FULL_DERIVED": ("map", "NAME"),
            "PII_LOCATION12_NAME": ("map", "NAME"),
            "PII_GIFT_EMAIL": ("map", "EMAIL"),
            "PII_ALT_PHONE": ("map", "PHONE"),
            "PII_LOCATION19_POSTCODE_EXT": ("map", "ADDRESS"),
            "PII_LOGIN_PASSWORD_CONFIRM": ("map", "SECRET"),
            "PII_GIFT_PIN": ("map", "SECRET"),
            "PII_DOB": ("map", "SENSITIVE_FIELD"),
            "PII_BILLING_ADDRESS": ("map", "ADDRESS"),
            "PII_BILLING_POSTCODE": ("map", "ADDRESS"),
            "PII_BILLING_STREET2": ("map", "ADDRESS"),
            "PII_CARD_NAME": ("map", "NAME"),
            "PII_DOB_MONTH_FORMATTED_COMMA": ("map", "SENSITIVE_FIELD"),
            "PII_BILLING_COMPANY": ("hard-negative", None),
            "PII_JOB_CODE": ("hard-negative", None),
            "PII_GIFT_MESSAGE3": ("map", "SENSITIVE_FIELD"),
            "PII_MAP": ("map", "SENSITIVE_IMAGE"),
            "PII_PROMO_CODE": ("hard-negative", None),
            "PII_CARD_TYPE": ("hard-negative", None),
        }
        self.assertEqual(
            {key: source_disposition(key) for key in expected}, expected
        )

    def test_webpii_parallel_prefetch_preserves_row_order(self) -> None:
        from training.visual.prepare_webpii import rows_with_images

        source_rows = iter(
            [
                {"source_id": str(index), "image": {"src": f"https://image/{index}"}}
                for index in range(5)
            ]
        )
        with patch(
            "training.visual.prepare_webpii.download_image",
            side_effect=lambda url: url.encode("utf-8"),
        ):
            result = list(
                rows_with_images(
                    source_rows,
                    metadata_only=False,
                    download_workers=2,
                    batch_size=2,
                )
            )
        self.assertEqual([row["source_id"] for row, _ in result], list("01234"))
        self.assertEqual(
            [content for _, content in result],
            [f"https://image/{index}".encode("utf-8") for index in range(5)],
        )

    def test_webpii_image_download_retries_transient_503(self) -> None:
        from training.visual.prepare_webpii import download_image

        transient = HTTPError(
            "https://datasets-server.huggingface.co/image.png",
            503,
            "unavailable",
            {},
            None,
        )
        with (
            patch(
                "training.visual.prepare_webpii.urlopen",
                side_effect=[transient, io.BytesIO(b"png-bytes")],
            ),
            patch("training.visual.prepare_webpii.time.sleep"),
        ):
            self.assertEqual(
                download_image(
                    "https://datasets-server.huggingface.co/image.png"
                ),
                b"png-bytes",
            )

    def test_detector_metrics_align_when_validation_omits_a_class(self) -> None:
        from training.visual.train_detector import align_class_metrics

        names = {0: "NAME", 4: "CARD_NUMBER", 8: "SENSITIVE_IMAGE"}
        box = SimpleNamespace(
            p=[0.75, 0.5],
            r=[0.8, 0.6],
            ap_class_index=[0, 4],
        )
        recalls, precisions, missing = align_class_metrics(box, names)
        self.assertEqual(recalls, {0: 0.8, 4: 0.6, 8: 0.0})
        self.assertEqual(precisions, {0: 0.75, 4: 0.5, 8: 0.0})
        self.assertEqual(missing, ["SENSITIVE_IMAGE"])

    def test_detector_test_split_check_uses_exact_path_components(self) -> None:
        from training.visual.train_detector import validation_uses_test_split

        self.assertFalse(
            validation_uses_test_split(
                "/vol/runs/plva-visual-agpl-test-v2/synthetic/images/validation"
            )
        )
        self.assertTrue(validation_uses_test_split("/vol/webpii/images/test"))

    def test_modal_training_asset_uses_locked_hash(self) -> None:
        from training.locked_asset import ensure_locked_training_asset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            content = b"pinned-training-checkpoint"
            destination = root / "checkpoint.pt"
            destination.write_bytes(content)
            lock = root / "models.lock.json"
            lock.write_text(
                json.dumps(
                    {
                        "assets": {
                            "fixture": {
                                "url": "https://example.test/checkpoint.pt",
                                "allowed_redirect_hosts": ["example.test"],
                                "bytes": len(content),
                                "sha256": hashlib.sha256(content).hexdigest(),
                                "license": "AGPL-3.0-only",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = ensure_locked_training_asset(
                "fixture",
                destination,
                lock_path=lock,
            )
            self.assertEqual(result["status"], "verified-existing")


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
class NumericContractTests(unittest.TestCase):
    def test_overlapping_windows_merge_by_source_offset(self) -> None:
        import numpy as np

        from training.modeling import labels_from_merged_windows
        from training.schema import LABEL_CONFIG

        logits = np.full((2, 2, len(LABEL_CONFIG.labels)), -10.0, dtype=np.float32)
        outside = LABEL_CONFIG.label_to_id["O"]
        begin_name = LABEL_CONFIG.label_to_id["B-NAME"]
        inside_name = LABEL_CONFIG.label_to_id["I-NAME"]
        logits[0, 0, outside] = 10.0
        logits[0, 1, outside] = 10.0
        logits[1, 0, begin_name] = 10.0
        logits[1, 1, inside_name] = 10.0
        labels = np.asarray(
            [[outside, begin_name], [begin_name, inside_name]], dtype=np.int64
        )
        truth, predicted = labels_from_merged_windows(
            logits,
            labels,
            [0, 0],
            [[[0, 4], [5, 9]], [[5, 9], [10, 14]]],
            [[[-1, -1], [5, 14]], [[5, 14], [5, 14]]],
            thresholds={name: 0.5 for name in LABEL_CONFIG.entity_classes},
        )
        self.assertEqual(truth, [["O", "B-NAME", "I-NAME"]])
        self.assertEqual(predicted, [["O", "B-NAME", "I-NAME"]])

    def test_rapidocr_ctc_dictionary_has_97_classes(self) -> None:
        import numpy as np

        from training.ocr.reference import character_list, decode_probabilities

        with tempfile.TemporaryDirectory() as temporary:
            dictionary = Path(temporary) / "dict.txt"
            dictionary.write_text(
                "\n".join(chr(code) for code in range(32, 127)) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(len(character_list(dictionary)), 97)
            probabilities = np.zeros((1, 4, 97), dtype=np.float32)
            probabilities[0, 0, 1] = 1.0
            probabilities[0, 1, 1] = 1.0
            probabilities[0, 2, 0] = 1.0
            probabilities[0, 3, 2] = 1.0
            decoded = decode_probabilities(probabilities, dictionary)
            self.assertEqual(decoded[0]["text"], " !")

    def test_webredact_decode_is_class_aware(self) -> None:
        import numpy as np

        from training.visual.reference_webredact import decode

        output = np.zeros((1, 6, 3), dtype=np.float32)
        output[0, :, 0] = [100, 100, 80, 40, 0.9, 0.1]
        output[0, :, 1] = [102, 100, 80, 40, 0.8, 0.1]
        output[0, :, 2] = [100, 100, 80, 40, 0.1, 0.95]
        boxes, scores, classes = decode(
            output,
            threshold=0.25,
            iou_threshold=0.7,
            maximum_detections=10,
        )
        self.assertEqual(len(boxes), 2)
        self.assertEqual(classes.tolist(), [1, 0])
        self.assertEqual(scores.tolist(), [0.949999988079071, 0.8999999761581421])


class VisualEvaluationTests(unittest.TestCase):
    def test_secret_proposal_gate(self) -> None:
        from training.visual.evaluate_detector import evaluate

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            truth = root / "truth.jsonl"
            predictions = root / "predictions.jsonl"
            output = root / "evaluation.json"
            truth.write_text(
                json.dumps(
                    {
                        "id": "synthetic-checkout",
                        "page_type": "checkout",
                        "annotations": [
                            {
                                "class": "CARD_NUMBER",
                                "bbox_xyxy": [10, 10, 110, 30],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            predictions.write_text(
                json.dumps(
                    {
                        "id": "synthetic-checkout",
                        "proposals": [
                            {
                                "class": "text",
                                "confidence": 0.9,
                                "xyxy": [8, 8, 112, 32],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = evaluate(
                argparse.Namespace(
                    truth=truth,
                    predictions=predictions,
                    output=output,
                    coverage_threshold=0.9,
                )
            )
            self.assertTrue(result["gates"]["passed"])
            self.assertEqual(result["metrics"]["secret_misses"], 0)


if __name__ == "__main__":
    unittest.main()
