from __future__ import annotations

import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.schema import sha256_file
from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.quantize_detector_onnx import (
    _evenly_spaced,
    gate_report,
    quantize_and_validate,
)
from training.visual.export_detector_onnx import write_artifact_metadata


def passing_report() -> dict:
    per_class = {
        name: {
            "support": 20,
            "matched": 16,
            "detections": 20,
            "recall": 0.8,
            "precision": 0.8,
        }
        for name in DETECTOR_CLASSES
    }
    model_metrics = {
        "support": 20 * len(DETECTOR_CLASSES),
        "matched": 16 * len(DETECTOR_CLASSES),
        "detections": 20 * len(DETECTOR_CLASSES),
        "recall": 0.8,
        "precision": 0.8,
        "per_class": per_class,
    }
    candidate_hash = "a" * 64
    return {
        "source": {"bytes": 1_000, "sha256": "b" * 64},
        "candidate": {"bytes": 400, "sha256": candidate_hash},
        "graph_contract": {"passed": True},
        "data": {
            "calibration_images": 256,
            "evaluation_images": 400,
            "calibration_evaluation_hash_overlap": 0,
            "class_support": {name: 20 for name in DETECTOR_CLASSES},
        },
        "comparison": {
            "raw_output": {
                "non_finite_candidate_values": 0,
                "class_argmax_agreement": 0.995,
                "mean_class_score_absolute_error": 0.001,
                "maximum_class_score_absolute_error": 0.1,
            },
            "profiles": {
                profile: {
                    "fp32": copy.deepcopy(model_metrics),
                    "candidate": copy.deepcopy(model_metrics),
                    "fp32_proposal_coverage": 0.995,
                }
                for profile in ("high-recall", "balanced")
            },
        },
        "cross_runtime": {
            "model_sha256": candidate_hash,
            "backends": {
                "node": {
                    "passed": True,
                    "package": "onnxruntime-node",
                    "package_version": "1.27.0",
                    "vectors_checked": 104,
                },
                "wasm": {
                    "passed": True,
                    "package": "onnxruntime-web",
                    "package_version": "1.27.0",
                    "vectors_checked": 104,
                },
            },
        },
    }


class DetectorQuantizationGateTests(unittest.TestCase):
    def test_fp32_export_declares_quantization_gated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pt"
            destination = root / "detector.onnx"
            source.write_bytes(b"source")
            destination.write_bytes(b"fp32")
            args = argparse.Namespace(
                source=source,
                source_license="AGPL-3.0-only",
                training_manifest=None,
                output_dir=root,
                opset=17,
            )
            write_artifact_metadata(
                args,
                {
                    "created_at": "2026-07-11T00:00:00+00:00",
                    "source_sha256": sha256_file(source),
                    "output_sha256": sha256_file(destination),
                },
                destination,
            )
            import json

            manifest = json.loads(
                (root / "detector_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["quantization"]["status"], "gated-not-produced"
            )
            self.assertTrue(manifest["quantization"]["fp32_remains_authoritative"])
            self.assertIn(
                "detector INT8 quantization not yet approved",
                manifest["publication_blockers"],
            )

    def test_gate_requires_size_quality_and_browser_parity(self) -> None:
        report = passing_report()
        result = gate_report(report)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["size_reduction_fraction"], 0.6)

        report["candidate"]["bytes"] = 900
        del report["cross_runtime"]["backends"]["wasm"]
        result = gate_report(report)
        self.assertFalse(result["passed"])
        self.assertTrue(any("model bytes" in value for value in result["failures"]))
        self.assertTrue(any("wasm parity" in value for value in result["failures"]))

    def test_gate_allows_no_secret_recall_regression(self) -> None:
        report = passing_report()
        report["comparison"]["profiles"]["high-recall"]["candidate"][
            "per_class"
        ]["SECRET"] = {
            "support": 20,
            "matched": 15,
            "detections": 20,
            "recall": 0.75,
            "precision": 0.75,
        }
        result = gate_report(report)
        self.assertFalse(result["passed"])
        self.assertIn(
            "high-recall SECRET recall regresses beyond its allowed bound",
            result["failures"],
        )

    def test_sampling_is_deterministic_and_spans_the_input(self) -> None:
        paths = [Path(f"{index:03d}.png") for index in range(10)]
        self.assertEqual(
            _evenly_spaced(paths, 4),
            [Path("000.png"), Path("002.png"), Path("005.png"), Path("007.png")],
        )
        self.assertEqual(_evenly_spaced(paths, 0), paths)

    def test_failed_candidate_is_not_promoted_or_left_behind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "detector.onnx"
            source.write_bytes(b"s" * 1_000)
            calibration_root = root / "calibration"
            evaluation_root = root / "evaluation"
            labels_root = root / "labels"
            output_dir = root / "output"
            for directory in (calibration_root, evaluation_root, labels_root):
                directory.mkdir()
            output_dir.mkdir()
            previous = output_dir / "detector.int8.onnx"
            previous.write_bytes(b"previous-approved-model")
            calibration_paths = [Path(f"cal-{index}.png") for index in range(256)]
            evaluation_paths = [Path(f"eval-{index}.png") for index in range(400)]
            support = {index: 20 for index in range(len(DETECTOR_CLASSES))}

            def fake_quantize(_source, destination, _images) -> None:
                destination.write_bytes(b"q" * 400)

            rejected_comparison = passing_report()["comparison"]
            rejected_comparison["raw_output"]["class_argmax_agreement"] = 0.50

            args = argparse.Namespace(
                source=source,
                calibration_images=calibration_root,
                evaluation_images=evaluation_root,
                evaluation_labels=labels_root,
                output_dir=output_dir,
                runtime_parity_dir=root,
                max_calibration_images=512,
                max_evaluation_images=0,
                force=True,
                require_output=False,
            )
            with (
                patch(
                    "training.visual.quantize_detector_onnx._images",
                    side_effect=[calibration_paths, evaluation_paths],
                ),
                patch(
                    "training.visual.quantize_detector_onnx._class_support",
                    return_value=support,
                ),
                patch(
                    "training.visual.quantize_detector_onnx._content_hashes",
                    side_effect=[{"calibration"}, {"evaluation"}],
                ),
                patch(
                    "training.visual.quantize_detector_onnx.quantize_qdq",
                    side_effect=fake_quantize,
                ),
                patch(
                    "training.visual.quantize_detector_onnx.graph_contract",
                    return_value={"passed": True},
                ),
                patch(
                    "training.visual.quantize_detector_onnx.compare_models",
                    return_value=rejected_comparison,
                ),
                patch(
                    "training.visual.quantize_detector_onnx.run_cross_runtime_parity",
                    side_effect=lambda candidate, *_: {
                        **passing_report()["cross_runtime"],
                        "model_sha256": sha256_file(candidate),
                    },
                ),
            ):
                result = quantize_and_validate(args)

            self.assertEqual(result["status"], "rejected")
            self.assertIsNone(result["output"])
            self.assertEqual(previous.read_bytes(), b"previous-approved-model")
            self.assertEqual(
                result["preserved_previous_output"]["sha256"], sha256_file(previous)
            )
            self.assertFalse(any(output_dir.glob("*candidate*.onnx")))
            self.assertTrue((output_dir / "detector_quantization_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
