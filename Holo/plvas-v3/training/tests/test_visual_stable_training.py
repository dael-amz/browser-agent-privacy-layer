from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.schema import sha256_file
from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.train_detector import (
    OTHER_SENSITIVE_CLASSES,
    SECRET_VISUAL_CLASSES,
    STABLE_OPTIMIZER_POLICY,
    consider_safety_candidate,
    require_valid_safety_decision,
)
from training.visual.training_attempt import (
    attempt_artifact_dir,
    attempt_output_dir,
    initialize_seeded_attempt,
    load_seeded_attempt,
    resolve_frozen_seed,
)


def class_map(
    secret: float, other: float, overrides: dict[str, float] | None = None
) -> dict[str, float]:
    values = {
        name: secret if name in SECRET_VISUAL_CLASSES else other
        for name in DETECTOR_CLASSES
    }
    values.update(overrides or {})
    return values


def initialize_state(
    recalls: dict[str, float] | None = None,
    precisions: dict[str, float] | None = None,
) -> dict:
    decision, state = consider_safety_candidate(
        None,
        epoch=-1,
        recall_by_class=recalls or class_map(0.5, 0.8),
        precision_by_class=precisions
        or {name: 0.8 for name in DETECTOR_CLASSES},
        missing_classes=[],
    )
    if not decision["selected"] or state is None:
        raise AssertionError(decision)
    return state


class ConstrainedSafetySelectionTests(unittest.TestCase):
    def test_human_epoch_five_cannot_displace_epoch_two(self) -> None:
        epoch_two_recalls = class_map(0.5830700553, 0.9273324774)
        state = initialize_state(
            epoch_two_recalls,
            {name: 0.8556403503 for name in DETECTOR_CLASSES},
        )
        decision, next_state = consider_safety_candidate(
            state,
            epoch=5,
            recall_by_class=class_map(0.5924606303, 0.7004927363),
            precision_by_class={
                name: 0.8975670572 for name in DETECTOR_CLASSES
            },
            missing_classes=[],
        )
        self.assertFalse(decision["selected"])
        self.assertIs(next_state, state)
        self.assertAlmostEqual(
            decision["deltas"]["secret_recall_minimum"], 0.009390575
        )
        self.assertAlmostEqual(
            decision["deltas"]["other_sensitive_recall_mean"], -0.2268397411
        )

    def test_exact_meaningful_gain_and_drop_boundaries_pass(self) -> None:
        state = initialize_state()
        decision, next_state = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=class_map(0.51, 0.77),
            precision_by_class={name: 0.75 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertTrue(decision["selected"])
        self.assertEqual(decision["reason"], "selected-meaningful-secret-improvement")
        self.assertIsNot(next_state, state)

        decision, _ = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=class_map(0.51, 0.769999),
            precision_by_class={name: 0.75 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertFalse(decision["selected"])
        self.assertEqual(decision["reason"], "rejected-regression-guard")

    def test_tie_band_and_precision_paths_require_recall_safety(self) -> None:
        state = initialize_state()
        decision, _ = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=class_map(0.509, 0.81),
            precision_by_class={name: 0.8 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertTrue(decision["selected"])
        self.assertEqual(
            decision["reason"], "selected-tie-band-other-sensitive-improvement"
        )

        decision, _ = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=class_map(0.5, 0.8),
            precision_by_class={name: 0.81 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertTrue(decision["selected"])
        self.assertEqual(
            decision["reason"],
            "selected-precision-tiebreak-without-recall-regression",
        )

        regressed = class_map(0.5, 0.8, {OTHER_SENSITIVE_CLASSES[0]: 0.799})
        decision, _ = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=regressed,
            precision_by_class={name: 0.81 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertFalse(decision["selected"])

    def test_individual_secret_guard_and_high_water_prevent_ratcheting(self) -> None:
        seed = class_map(
            0.5,
            0.8,
            {"CVC": 0.8, "SECRET": 0.8},
        )
        state = initialize_state(seed)
        candidate = class_map(
            0.51,
            0.8,
            {"CVC": 0.78, "SECRET": 0.8},
        )
        decision, _ = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=candidate,
            precision_by_class={name: 0.8 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertFalse(decision["selected"])
        self.assertFalse(decision["gates"]["individual_secret_drop"])

        state = initialize_state()
        first, state = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=class_map(0.51, 0.77),
            precision_by_class={name: 0.8 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertTrue(first["selected"])
        assert state is not None
        second, unchanged = consider_safety_candidate(
            state,
            epoch=1,
            recall_by_class=class_map(0.52, 0.74),
            precision_by_class={name: 0.8 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertFalse(second["selected"])
        self.assertFalse(second["gates"]["other_mean_drop_from_high_water"])
        self.assertIs(unchanged, state)

    def test_missing_and_nonfinite_metrics_abort_fail_closed(self) -> None:
        state = initialize_state()
        decision, unchanged = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=class_map(0.6, 0.8),
            precision_by_class={name: 0.8 for name in DETECTOR_CLASSES},
            missing_classes=["CVC"],
        )
        self.assertFalse(decision["valid"])
        self.assertIs(unchanged, state)
        with self.assertRaisesRegex(RuntimeError, "incomplete or non-finite"):
            require_valid_safety_decision(decision)

        invalid = class_map(0.6, 0.8, {"SECRET": float("nan")})
        decision, unchanged = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=invalid,
            precision_by_class={name: 0.8 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertFalse(decision["valid"])
        self.assertIs(unchanged, state)


class FrozenSeedAttemptTests(unittest.TestCase):
    def test_attempt_paths_are_isolated(self) -> None:
        root = Path("/vol/runs/example")
        self.assertEqual(attempt_output_dir(root, "default"), root / "visual/training")
        self.assertEqual(
            attempt_output_dir(root, "stable"), root / "visual/training-stable"
        )
        self.assertEqual(
            attempt_artifact_dir(root, "stable"), root / "visual/artifacts-stable"
        )

    def test_frozen_seed_is_pinned_copied_and_resume_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot_root = root / "snapshots"
            snapshot_root.mkdir()
            source = snapshot_root / "aggregate-best-epoch2.pt"
            source.write_bytes(b"frozen-epoch-two")
            license_path = root / "detector-base-license.txt"
            license_path.write_text("AGPL-3.0-only\n", encoding="utf-8")
            inherited = root / "detector-base-provenance.json"
            inherited.write_text('{"source":"v2"}\n', encoding="utf-8")
            resolved = resolve_frozen_seed(
                str(source), volume_root=root, snapshot_root=snapshot_root
            )
            attempt_root = root / "runs/run/visual/training-stable"
            result = initialize_seeded_attempt(
                attempt_root=attempt_root,
                run_id="run",
                attempt_id="stable",
                source_checkpoint=resolved,
                expected_sha256=sha256_file(source),
                seed_provenance={"snapshot_id": "epoch2-safety"},
                optimizer_policy=STABLE_OPTIMIZER_POLICY,
                license_path=license_path,
                inherited_provenance_path=inherited,
            )
            self.assertEqual(result["checkpoint"].read_bytes(), source.read_bytes())
            self.assertEqual(result["checkpoint_sha256"], sha256_file(source))
            self.assertTrue(result["manifest"]["resume_allowed"])
            with self.assertRaises(FileExistsError):
                initialize_seeded_attempt(
                    attempt_root=attempt_root,
                    run_id="run",
                    attempt_id="stable",
                    source_checkpoint=resolved,
                    expected_sha256=sha256_file(source),
                    seed_provenance={},
                    optimizer_policy=STABLE_OPTIMIZER_POLICY,
                    license_path=license_path,
                )

            (attempt_root / "frozen-seed.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "byte count|SHA-256"):
                load_seeded_attempt(
                    attempt_root=attempt_root,
                    run_id="run",
                    attempt_id="stable",
                    optimizer_policy=STABLE_OPTIMIZER_POLICY,
                    license_path=license_path,
                )

    def test_seed_hash_mismatch_fails_before_attempt_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            seed = snapshots / "seed.pt"
            seed.write_bytes(b"seed")
            license_path = root / "license.txt"
            license_path.write_text("AGPL-3.0-only\n", encoding="utf-8")
            attempt = root / "runs/run/visual/training-stable"
            with self.assertRaisesRegex(RuntimeError, "CLI pin"):
                initialize_seeded_attempt(
                    attempt_root=attempt,
                    run_id="run",
                    attempt_id="stable",
                    source_checkpoint=seed,
                    expected_sha256="0" * 64,
                    seed_provenance={},
                    optimizer_policy=STABLE_OPTIMIZER_POLICY,
                    license_path=license_path,
                )
            self.assertFalse(attempt.exists())


if __name__ == "__main__":
    unittest.main()
