from __future__ import annotations

import argparse
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.train_detector import (
    SAFETY_SELECTION_POLICY,
    SCREENSHOT_AUGMENTATION_POLICY,
    SEED_INCUMBENT_NAME,
    STABLE_OPTIMIZER_POLICY,
    build_training_resume_contract,
    consider_safety_candidate,
    evaluation_report,
    load_visual_resume_state,
    prepare_supplemental_test_yaml,
    sha256_file,
    train,
    validate_composed_dataset_contract,
)


NAMES = {index: name for index, name in enumerate(DETECTOR_CLASSES)}


def write_supplemental(root: Path) -> dict[str, object]:
    records: list[str] = []
    for index in range(200):
        image = root / f"images/test/{index:03d}.png"
        label = root / f"labels/test/{index:03d}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"image-{index}".encode())
        label.write_text("", encoding="utf-8")
        records.append(
            json.dumps(
                {
                    "id": f"test-{index}",
                    "split": "test",
                    "image": image.relative_to(root).as_posix(),
                    "image_sha256": sha256_file(image),
                    "label_file": label.relative_to(root).as_posix(),
                    "label_sha256": sha256_file(label),
                },
                sort_keys=True,
            )
        )
    (root / "images/train").mkdir(parents=True)
    records_path = root / "records/test.jsonl"
    records_path.parent.mkdir(parents=True)
    records_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    content_hashes = {
        "images_aggregate_sha256": "1" * 64,
        "labels_aggregate_sha256": "2" * 64,
    }
    manifest = {
        "schema_version": 2,
        "supplemental_only": True,
        "training_use": "train-split-only",
        "test_used_for_checkpoint_selection": False,
        "values_persisted": False,
        "unmapped_source_labels_fail_closed": True,
        "classes": list(DETECTOR_CLASSES),
        "splits": {
            "train": {"records": 800},
            "test": {
                "records": 200,
                "records_path": "records/test.jsonl",
                "records_sha256": sha256_file(records_path),
            },
        },
        "content_hashes": content_hashes,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "training_split_records": 800,
        "test_excluded_from_training_and_selection": True,
        **content_hashes,
    }


def composed_fixture(
    root: Path, supplemental_source: dict[str, object] | None = None
) -> tuple[dict[str, object], dict[str, object], Path]:
    webpii = root / "webpii"
    synthetic = root / "synthetic"
    composed = root / "composed"
    for source_root in (webpii, synthetic, composed):
        source_root.mkdir()
        (source_root / "manifest.json").write_text("{}\n", encoding="utf-8")
    dataset = {
        "path": "/",
        "train": [str(webpii / "images/train")],
        "val": str(synthetic / "images/validation"),
        "test": str(webpii / "images/test"),
        "nc": len(DETECTOR_CLASSES),
        "names": list(DETECTOR_CLASSES),
    }
    sources: dict[str, object] = {
        "webpii": {"manifest": str(webpii / "manifest.json")},
        "synthetic": {"manifest": str(synthetic / "manifest.json")},
    }
    if supplemental_source is not None:
        sources["supplemental_ats"] = supplemental_source
    source_manifest = {
        "schema_version": 2 if supplemental_source is not None else 1,
        "published_splits_preserved": True,
        "test_used_for_checkpoint_selection": False,
        "selection_data": "screen-native-synthetic-validation",
        "test_source": "WebPII/test",
        "classes": list(DETECTOR_CLASSES),
        "sources": sources,
    }
    return dataset, source_manifest, composed / "manifest.json"


def resume_fixture(
    root: Path, *, progress_schema: int = 3, checkpoint_epochs: int = 4
) -> tuple[argparse.Namespace, dict[str, object], object]:
    run_id = "resume-run"
    local_stage = root / "ephemeral" / run_id
    local_stage.mkdir(parents=True)
    dataset_yaml = local_stage / "dataset.yaml"
    source_manifest = local_stage / "source_manifest.json"
    staging_manifest = local_stage / "staging_manifest.json"
    dataset_yaml.write_text("names: fixture\n", encoding="utf-8")
    source_manifest.write_text('{"classes": "fixture"}\n', encoding="utf-8")
    staging_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_contract_sha256": "a" * 64,
                "local_dataset_yaml_sha256": sha256_file(dataset_yaml),
                "local_source_manifest_sha256": sha256_file(source_manifest),
            }
        ),
        encoding="utf-8",
    )

    output = root / run_id / "visual/training"
    output.mkdir(parents=True)
    base_checkpoint = output / "frozen-seed.pt"
    base_checkpoint.write_bytes(b"frozen-seed")
    persisted = {
        "dataset_yaml": output / "staged-dataset.yaml",
        "source_manifest": output / "staged-source-manifest.json",
        "staging_manifest": output / "dataset-staging-manifest.json",
    }
    for source, destination in (
        (dataset_yaml, persisted["dataset_yaml"]),
        (source_manifest, persisted["source_manifest"]),
        (staging_manifest, persisted["staging_manifest"]),
    ):
        destination.write_bytes(source.read_bytes())
    data_staging = {
        f"{name}_{field}": (
            str(path) if field == "path" else sha256_file(path)
        )
        for name, path in persisted.items()
        for field in ("path", "sha256")
    }
    args = argparse.Namespace(
        dataset_yaml=dataset_yaml,
        source_manifest=source_manifest,
        local_staging_manifest=staging_manifest,
        data_staging=data_staging,
        output_dir=output,
        base_checkpoint=base_checkpoint,
        run_id=run_id,
        attempt_id="default",
        attempt_manifest_sha256=None,
        epochs=4,
        image_size=640,
        batch_size=64,
        seed=1311,
    )
    contract = build_training_resume_contract(args, NAMES)
    weights = output / "ultralytics-run/weights"
    weights.mkdir(parents=True)
    last = weights / "last.pt"
    best = weights / "best.pt"
    safety_best = output / "safety-best.pt"
    last.write_bytes(b"durable-last")
    best.write_bytes(b"durable-best")
    safety_best.write_bytes(b"durable-safety-best")

    def checkpoint_record(path: Path) -> dict[str, object]:
        return {
            "path": str(path),
            "exists": True,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    def class_values(secret: float, other: float) -> dict[str, float]:
        return {
            name: secret if name in {"CARD_NUMBER", "CVC", "SECRET"} else other
            for name in DETECTOR_CLASSES
        }

    seed_recalls = class_values(0.7, 0.6)
    seed_precisions = {name: 0.8 for name in DETECTOR_CLASSES}
    seed_decision, selection_state = consider_safety_candidate(
        None,
        epoch=-1,
        recall_by_class=seed_recalls,
        precision_by_class=seed_precisions,
        missing_classes=[],
    )
    assert selection_state is not None
    seed_incumbent = {
        **seed_decision["candidate"],
        "missing_validation_classes": [],
        "checkpoint": str(base_checkpoint),
        "checkpoint_sha256": sha256_file(base_checkpoint),
    }
    history: list[dict[str, object]] = []
    for epoch, recalls, precisions in (
        (0, class_values(0.72, 0.61), {name: 0.8 for name in DETECTOR_CLASSES}),
        (1, class_values(0.60, 0.90), {name: 0.9 for name in DETECTOR_CLASSES}),
    ):
        decision, next_state = consider_safety_candidate(
            selection_state,
            epoch=epoch,
            recall_by_class=recalls,
            precision_by_class=precisions,
            missing_classes=[],
        )
        if next_state is not None:
            selection_state = next_state
        candidate = decision["candidate"]
        history.append(
            {
                "epoch": epoch,
                "secret_recall_minimum": candidate["secret_recall_minimum"],
                "other_sensitive_recall_mean": candidate[
                    "other_sensitive_recall_mean"
                ],
                "non_secret_recall_mean": candidate[
                    "other_sensitive_recall_mean"
                ],
                "precision_mean": candidate["precision_mean"],
                "per_class_recall": recalls,
                "per_class_precision": precisions,
                "missing_validation_classes": [],
                "selected": decision["selected"],
                "selection_reason": decision["reason"],
                "selection_deltas": decision["deltas"],
                "selection_gates": decision["gates"],
            }
        )
    incumbent = selection_state["incumbent"]
    progress = {
        "schema_version": progress_schema,
        "epoch": 1,
        "selection_policy": SAFETY_SELECTION_POLICY,
        "best_score": [
            incumbent["secret_recall_minimum"],
            incumbent["other_sensitive_recall_mean"],
            incumbent["precision_mean"],
        ],
        "seed_incumbent": seed_incumbent,
        "selection_state": selection_state,
        "history": history,
        "checkpoints": {
            "last": checkpoint_record(last),
            "best": checkpoint_record(best),
            "safety_best": checkpoint_record(safety_best),
        },
    }
    if progress_schema == 2:
        progress["resume_contract"] = contract
    if progress_schema == 3:
        progress["resume_contract"] = contract
    (output / "safety-selection-progress.json").write_text(
        json.dumps(progress), encoding="utf-8"
    )
    (output / SEED_INCUMBENT_NAME).write_text(
        json.dumps(seed_incumbent), encoding="utf-8"
    )
    train_args = {
        "epochs": checkpoint_epochs,
        "data": str(dataset_yaml),
        "save_dir": str(output / "ultralytics-run"),
        "imgsz": args.image_size,
        "batch": args.batch_size,
        "seed": args.seed,
        **STABLE_OPTIMIZER_POLICY,
        **SCREENSHOT_AUGMENTATION_POLICY,
    }

    class FakeTorch:
        @staticmethod
        def load(
            path: str, *, map_location: str, weights_only: bool
        ) -> dict[str, object]:
            if (
                Path(path) != last
                or map_location != "cpu"
                or weights_only is not False
            ):
                raise AssertionError("resume loaded an unexpected checkpoint")
            return {
                "epoch": 1,
                "optimizer": {"state": {}, "param_groups": []},
                "scaler": {},
                "ema": object(),
                "train_args": train_args,
            }

    return args, contract, FakeTorch


class DetectorEvaluationContractTests(unittest.TestCase):
    def test_modal_visual_resume_flag_uses_spawn_and_get(self) -> None:
        modal_source = (
            Path(__file__).resolve().parents[1] / "modal_app.py"
        ).read_text(encoding="utf-8")
        self.assertIn("visual_resume: bool = False", modal_source)
        self.assertIn('visual_attempt: str = "default"', modal_source)
        self.assertIn('visual_seed_checkpoint: str = ""', modal_source)
        self.assertIn('visual_seed_sha256: str = ""', modal_source)
        self.assertIn("attempt_output_dir(paths[\"root\"], attempt_id)", modal_source)
        self.assertIn("copy_workers=VISUAL_STAGE_COPY_WORKERS", modal_source)
        self.assertIn("resume: bool = False", modal_source)
        self.assertIn("visual_train_remote.spawn(", modal_source)
        self.assertIn("result[\"visual_train\"] = visual_call.get()", modal_source)

    def test_resume_restores_safety_history_from_only_persisted_last(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, contract, torch_module = resume_fixture(Path(temporary))
            state = load_visual_resume_state(args, contract, torch_module)
            self.assertEqual(
                state["last_path"],
                args.output_dir / "ultralytics-run/weights/last.pt",
            )
            self.assertEqual(len(state["history"]), 2)
            self.assertEqual(state["best_key"], (0.72, 0.61, 0.8))
            self.assertEqual(state["selection_state"]["incumbent"]["epoch"], 0)

    def test_resume_rejects_tampered_checkpoint_and_epoch_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, contract, torch_module = resume_fixture(Path(temporary))
            last = args.output_dir / "ultralytics-run/weights/last.pt"
            last.write_bytes(b"tampered-xxx")
            with self.assertRaisesRegex(RuntimeError, "last.*SHA-256"):
                load_visual_resume_state(args, contract, torch_module)

        with tempfile.TemporaryDirectory() as temporary:
            args, contract, torch_module = resume_fixture(
                Path(temporary), progress_schema=3, checkpoint_epochs=5
            )
            with self.assertRaisesRegex(RuntimeError, "total epochs"):
                load_visual_resume_state(args, contract, torch_module)

    def test_resume_rejects_jointly_tampered_seed_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, contract, torch_module = resume_fixture(Path(temporary))
            seed_path = args.output_dir / SEED_INCUMBENT_NAME
            progress_path = args.output_dir / "safety-selection-progress.json"
            seed = json.loads(seed_path.read_text(encoding="utf-8"))
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            seed["checkpoint_sha256"] = "f" * 64
            progress["seed_incumbent"] = seed
            seed_path.write_text(json.dumps(seed), encoding="utf-8")
            progress_path.write_text(json.dumps(progress), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "immutable seed"):
                load_visual_resume_state(args, contract, torch_module)

    def test_legacy_progress_is_rejected_after_policy_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, contract, torch_module = resume_fixture(
                Path(temporary), progress_schema=1
            )
            with self.assertRaisesRegex(RuntimeError, "legacy visual progress"):
                load_visual_resume_state(args, contract, torch_module)

    def test_train_resumes_with_optimizer_state_and_restored_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _contract, torch_loader = resume_fixture(root)
            commits: list[str] = []
            args.base_license = "AGPL-3.0-only"
            args.base_source = "fixture"
            args.commercial_license_approved = False
            args.resume = True
            args.device = "cpu"
            args.workers = 0
            args.patience = args.epochs
            args.checkpoint_commit_hook = lambda: commits.append("commit")

            class FakeYOLO:
                loaded: list[str] = []
                training_calls: list[dict[str, object]] = []

                def __init__(self, source: str) -> None:
                    self.source = source
                    self.callback = None
                    self.loaded.append(source)

                def add_callback(self, event: str, callback: object) -> None:
                    if event != "on_model_save":
                        raise AssertionError(event)
                    self.callback = callback

                def train(self, **kwargs: object) -> SimpleNamespace:
                    self.training_calls.append(kwargs)
                    weights = args.output_dir / "ultralytics-run/weights"
                    last = weights / "last.pt"
                    best = weights / "best.pt"
                    last.write_bytes(b"resumed-last")
                    best.write_bytes(b"resumed-aggregate-best")
                    trainer = SimpleNamespace(
                        epoch=2,
                        last=str(last),
                        best=str(best),
                        validator=SimpleNamespace(
                            metrics=SimpleNamespace(
                                box=SimpleNamespace(
                                    p=[0.75] * len(NAMES),
                                    r=[0.65] * len(NAMES),
                                    ap_class_index=list(NAMES),
                                )
                            )
                        ),
                        model=SimpleNamespace(names=NAMES),
                    )
                    if self.callback is None:
                        raise AssertionError("resume callback was not installed")
                    self.callback(trainer)
                    return SimpleNamespace(results_dict={"train/box_loss": 0.25})

                def val(self, **_kwargs: object) -> SimpleNamespace:
                    return SimpleNamespace(
                        names=NAMES,
                        box=SimpleNamespace(
                            p=[0.9] * len(NAMES),
                            r=[0.85] * len(NAMES),
                            ap_class_index=list(NAMES),
                        ),
                        results_dict={"metrics/mAP50(B)": 0.5},
                    )

            numpy = types.ModuleType("numpy")
            numpy.mean = lambda values: sum(values) / len(values)
            torch = types.ModuleType("torch")
            torch.__version__ = "fixture-torch"
            torch.cuda = SimpleNamespace(is_available=lambda: False)
            torch.load = torch_loader.load
            yaml = types.ModuleType("yaml")
            yaml.safe_load = lambda _content: {}
            ultralytics = types.ModuleType("ultralytics")
            ultralytics.YOLO = FakeYOLO

            with (
                patch.dict(
                    sys.modules,
                    {
                        "numpy": numpy,
                        "torch": torch,
                        "yaml": yaml,
                        "ultralytics": ultralytics,
                    },
                ),
                patch(
                    "training.visual.train_detector.validate_composed_dataset_contract",
                    return_value=NAMES,
                ),
                patch(
                    "training.visual.train_detector.prepare_supplemental_test_yaml",
                    return_value=None,
                ),
                patch(
                    "training.visual.train_detector.importlib.metadata.version",
                    return_value="fixture-ultralytics",
                ),
            ):
                manifest = train(args)

            self.assertEqual(len(FakeYOLO.training_calls), 1)
            training_call = FakeYOLO.training_calls[0]
            self.assertTrue(training_call["resume"])
            self.assertTrue(training_call["save"])
            self.assertEqual(training_call["save_period"], 1)
            self.assertEqual(training_call["epochs"], 4)
            expected_last = args.output_dir / "ultralytics-run/weights/last.pt"
            self.assertIn(str(expected_last), FakeYOLO.loaded)
            self.assertEqual(len(manifest["history"]), 3)
            self.assertFalse(manifest["history"][-1]["selected"])
            self.assertEqual(manifest["best_checkpoint"]["score"], [0.72, 0.61, 0.8])
            persistence = manifest["checkpoint_persistence"]
            self.assertTrue(persistence["resume_supported"])
            self.assertTrue(persistence["resumed"])
            self.assertEqual(persistence["resumed_from_epoch"], 1)
            self.assertEqual(persistence["committed_epochs"], 3)
            self.assertEqual(persistence["committed_epochs_this_invocation"], 1)
            self.assertEqual(commits, ["commit"])
            progress = json.loads(
                (args.output_dir / "safety-selection-progress.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(progress["schema_version"], 3)
            self.assertEqual(progress["resume_contract"], persistence["resume_contract"])

    def test_composed_contract_requires_exact_published_test_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, source_manifest, manifest_path = composed_fixture(root)
            self.assertEqual(
                validate_composed_dataset_contract(
                    dataset, source_manifest, manifest_path
                ),
                NAMES,
            )

            missing_test = dict(dataset)
            del missing_test["test"]
            with self.assertRaisesRegex(RuntimeError, "must include.*published test"):
                validate_composed_dataset_contract(
                    missing_test, source_manifest, manifest_path
                )

            wrong_test = dict(dataset)
            wrong_test["test"] = str(root / "other/images/test")
            with self.assertRaisesRegex(RuntimeError, "composed WebPII"):
                validate_composed_dataset_contract(
                    wrong_test, source_manifest, manifest_path
                )

    def test_evaluation_report_aligns_and_reports_missing_test_classes(self) -> None:
        result = SimpleNamespace(
            names=NAMES,
            box=SimpleNamespace(
                p=[0.9, 0.8],
                r=[0.7, 0.6],
                ap_class_index=[0, 4],
            ),
            results_dict={"metrics/mAP50(B)": 0.25},
        )
        report = evaluation_report(result, NAMES, "missing_test_classes")
        self.assertEqual(report["aggregate"]["metrics/mAP50(B)"], 0.25)
        self.assertEqual(report["per_class"]["NAME"]["recall"], 0.7)
        self.assertEqual(report["per_class"]["CVC"]["precision"], 0.0)
        self.assertIn("CVC", report["missing_test_classes"])
        self.assertEqual(report["missing_classes"], report["missing_test_classes"])
        self.assertEqual(report["minimum_secret_class_recall"], 0.0)

    def test_supplemental_test_requires_exactly_200_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supplemental = root / "supplemental"
            supplemental.mkdir()
            source = write_supplemental(supplemental)
            manifest_path = supplemental / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["splits"]["test"]["records"] = 199
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            source["manifest_sha256"] = sha256_file(manifest_path)
            output = root / "output"
            output.mkdir()
            with self.assertRaisesRegex(RuntimeError, "exactly 200 records"):
                prepare_supplemental_test_yaml(
                    {"sources": {"supplemental_ats": source}},
                    root / "composed/manifest.json",
                    output,
                    NAMES,
                )

    def test_train_compares_frozen_and_selected_checkpoints_on_both_tests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supplemental = root / "supplemental"
            supplemental.mkdir()
            supplemental_source = write_supplemental(supplemental)
            dataset, source_manifest, source_manifest_path = composed_fixture(
                root, supplemental_source
            )
            source_manifest_path.write_text(
                json.dumps(source_manifest), encoding="utf-8"
            )
            dataset_yaml = root / "composed/dataset.yaml"
            dataset_yaml.write_text("fixture: true\n", encoding="utf-8")
            base_checkpoint = root / "base.pt"
            base_checkpoint.write_bytes(b"base-checkpoint")
            output = root / "output"
            last_checkpoint = output / "ultralytics-run/weights/last.pt"
            best_checkpoint = output / "ultralytics-run/weights/best.pt"
            commit_calls: list[str] = []

            class FakeYOLO:
                loaded: list[str] = []
                validation_calls: list[dict[str, object]] = []
                training_calls: list[dict[str, object]] = []

                def __init__(self, source: str) -> None:
                    self.source = source
                    self.callback = None
                    self.loaded.append(source)

                def add_callback(self, event: str, callback: object) -> None:
                    self.assert_event(event)
                    self.callback = callback

                @staticmethod
                def assert_event(event: str) -> None:
                    if event != "on_model_save":
                        raise AssertionError(event)

                def train(self, **kwargs: object) -> SimpleNamespace:
                    self.training_calls.append(kwargs)
                    last_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    last_checkpoint.write_bytes(b"selected-checkpoint")
                    best_checkpoint.write_bytes(b"aggregate-best-checkpoint")
                    metrics = SimpleNamespace(
                        box=SimpleNamespace(
                            p=[0.7] * len(NAMES),
                            r=[0.8] * len(NAMES),
                            ap_class_index=list(NAMES),
                        )
                    )
                    trainer = SimpleNamespace(
                        epoch=0,
                        last=str(last_checkpoint),
                        best=str(best_checkpoint),
                        validator=SimpleNamespace(metrics=metrics),
                        model=SimpleNamespace(names=NAMES),
                    )
                    if self.callback is None:
                        raise AssertionError("training callback was not installed")
                    self.callback(trainer)
                    return SimpleNamespace(results_dict={"train/box_loss": 0.5})

                def val(self, **kwargs: object) -> SimpleNamespace:
                    self.validation_calls.append(kwargs)
                    name = kwargs["name"]
                    precision = 0.9
                    recall = 0.85
                    if name == "frozen-seed-validation":
                        indices = list(NAMES)
                        aggregate = {"metrics/mAP50(B)": 0.3}
                        precision = 0.6
                        recall = 0.6
                    elif name == "frozen-base-webpii-test":
                        indices = [index for index in NAMES if index != 5]
                        aggregate = {"metrics/mAP50(B)": 0.1}
                    elif name == "frozen-base-supplemental-test":
                        indices = [index for index in NAMES if index != 6]
                        aggregate = {"metrics/mAP50(B)": 0.2}
                    elif name == "safety-best-validation":
                        indices = list(NAMES)
                        aggregate = {"metrics/mAP50(B)": 0.4}
                    elif name == "safety-best-webpii-test":
                        indices = [index for index in NAMES if index != 5]
                        aggregate = {"metrics/mAP50(B)": 0.5}
                    elif name == "safety-best-supplemental-test":
                        indices = [index for index in NAMES if index != 6]
                        aggregate = {"metrics/mAP50(B)": 0.6}
                    else:
                        raise AssertionError(name)
                    return SimpleNamespace(
                        names=NAMES,
                        box=SimpleNamespace(
                            p=[precision] * len(indices),
                            r=[recall] * len(indices),
                            ap_class_index=indices,
                        ),
                        results_dict=aggregate,
                    )

            numpy = types.ModuleType("numpy")
            numpy.mean = lambda values: sum(values) / len(values)
            torch = types.ModuleType("torch")
            torch.__version__ = "fixture-torch"
            torch.cuda = SimpleNamespace(is_available=lambda: False)
            yaml = types.ModuleType("yaml")
            yaml.safe_load = lambda _content: dataset
            ultralytics = types.ModuleType("ultralytics")
            ultralytics.YOLO = FakeYOLO

            args = argparse.Namespace(
                dataset_yaml=dataset_yaml,
                source_manifest=source_manifest_path,
                base_checkpoint=base_checkpoint,
                base_license="AGPL-3.0-only",
                base_source="fixture",
                commercial_license_approved=False,
                output_dir=output,
                epochs=1,
                image_size=640,
                batch_size=4,
                device="cpu",
                workers=0,
                patience=1,
                seed=1311,
                data_staging={"fixture": True},
                checkpoint_commit_hook=lambda: commit_calls.append("commit"),
            )
            with (
                patch.dict(
                    sys.modules,
                    {
                        "numpy": numpy,
                        "torch": torch,
                        "yaml": yaml,
                        "ultralytics": ultralytics,
                    },
                ),
                patch(
                    "training.visual.train_detector.importlib.metadata.version",
                    return_value="fixture-ultralytics",
                ),
            ):
                manifest = train(args)

            self.assertEqual(
                [call["name"] for call in FakeYOLO.validation_calls],
                [
                    "frozen-seed-validation",
                    "frozen-base-webpii-test",
                    "frozen-base-supplemental-test",
                    "safety-best-validation",
                    "safety-best-webpii-test",
                    "safety-best-supplemental-test",
                ],
            )
            self.assertEqual(
                [call["split"] for call in FakeYOLO.validation_calls],
                ["val", "test", "test", "val", "test", "test"],
            )
            self.assertEqual(manifest["schema_version"], 3)
            self.assertEqual(commit_calls, ["commit", "commit"])
            self.assertEqual(manifest["checkpoint_persistence"]["committed_epochs"], 1)
            self.assertTrue(manifest["checkpoint_persistence"]["commit_hook_enabled"])
            self.assertEqual(manifest["data"]["staging"], {"fixture": True})
            progress = json.loads(
                (output / "safety-selection-progress.json").read_text(encoding="utf-8")
            )
            self.assertTrue(progress["checkpoints"]["last"]["exists"])
            self.assertTrue(progress["checkpoints"]["best"]["exists"])
            self.assertTrue(progress["checkpoints"]["safety_best"]["exists"])
            self.assertAlmostEqual(manifest["best_checkpoint"]["score"][0], 0.8)
            self.assertEqual(len(FakeYOLO.training_calls), 1)
            training_call = FakeYOLO.training_calls[0]
            self.assertEqual(
                {
                    key: training_call[key]
                    for key in SCREENSHOT_AUGMENTATION_POLICY
                },
                SCREENSHOT_AUGMENTATION_POLICY,
            )
            self.assertEqual(
                {
                    key: training_call[key] for key in STABLE_OPTIMIZER_POLICY
                },
                STABLE_OPTIMIZER_POLICY,
            )
            self.assertEqual(training_call["fliplr"], 0.0)
            self.assertEqual(training_call["flipud"], 0.0)
            self.assertEqual(training_call["mosaic"], 0.0)
            self.assertEqual(training_call["degrees"], 0.0)
            self.assertEqual(training_call["translate"], 0.02)
            self.assertEqual(training_call["scale"], 0.05)
            self.assertTrue(training_call["save"])
            self.assertEqual(training_call["save_period"], 1)
            self.assertFalse(training_call["resume"])
            self.assertEqual(
                manifest["training_config"]["augmentation_policy"][
                    "ultralytics_overrides"
                ],
                SCREENSHOT_AUGMENTATION_POLICY,
            )
            baseline = manifest["baseline_published_test_evaluation"]
            self.assertEqual(baseline["timing"], "pre-training")
            self.assertEqual(baseline["checkpoint_status"], "frozen")
            self.assertFalse(baseline["used_for_checkpoint_selection"])
            self.assertEqual(baseline["aggregate"]["metrics/mAP50(B)"], 0.1)
            baseline_supplemental = manifest["baseline_supplemental_test_evaluation"]
            self.assertFalse(baseline_supplemental["used_for_checkpoint_selection"])
            self.assertEqual(
                baseline_supplemental["aggregate"]["metrics/mAP50(B)"], 0.2
            )
            published = manifest["published_test_evaluation"]
            self.assertEqual(published["timing"], "post-selection")
            self.assertFalse(published["used_for_checkpoint_selection"])
            self.assertEqual(published["aggregate"]["metrics/mAP50(B)"], 0.5)
            self.assertIn("CVC", published["missing_test_classes"])
            self.assertEqual(published["per_class"]["CVC"]["recall"], 0.0)
            supplemental_result = manifest["supplemental_test_evaluation"]
            self.assertFalse(supplemental_result["used_for_checkpoint_selection"])
            self.assertIn("SECRET", supplemental_result["missing_test_classes"])
            self.assertTrue((output / "supplemental-test.yaml").is_file())
            persisted = json.loads(
                (output / "training_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted, manifest)


if __name__ == "__main__":
    unittest.main()
