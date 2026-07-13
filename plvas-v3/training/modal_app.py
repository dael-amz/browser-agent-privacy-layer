from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import modal


APP_NAME = "plva-model-engineering"
VOLUME_NAME = "plva-model-engineering"
CACHE_VOLUME_NAME = "plva-hf-cache"
REMOTE_ROOT = Path("/vol/runs")
VISUAL_STAGE_COPY_WORKERS = 64
LOCAL_TRAINING_DIR = Path(__file__).resolve().parent
LOCAL_REPO_ROOT = LOCAL_TRAINING_DIR.parent

REMOTE_PACKAGES = [
    "accelerate==1.14.0",
    "datasets==5.0.0",
    "huggingface-hub==1.23.0",
    "onnx==1.22.0",
    "onnxruntime==1.27.0",
    "onnxscript==0.7.1",
    "safetensors==0.8.0",
    "scikit-learn==1.9.0",
    "seqeval==1.2.2",
    "torch==2.13.0",
    "tokenizers==0.22.2",
    "transformers==5.13.1",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*REMOTE_PACKAGES)
    .add_local_dir(
        str(LOCAL_TRAINING_DIR),
        remote_path="/root/training",
        copy=True,
        ignore=[
            ".venv",
            ".cache",
            "data",
            "runs",
            "artifacts",
            "__pycache__",
            "tests",
        ],
    )
    .add_local_file(
        str(LOCAL_REPO_ROOT / "models.lock.json"),
        remote_path="/root/models.lock.json",
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/root",
            "HF_HOME": "/cache/huggingface",
            "HF_DATASETS_CACHE": "/cache/datasets",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)
visual_image = image.apt_install("libgl1", "libglib2.0-0").uv_pip_install(
    "pillow==12.3.0",
    "pyyaml==6.0.3",
    "torchvision==0.28.0",
    "ultralytics==8.4.92",
)
app = modal.App(APP_NAME, image=image)
run_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id):
        raise ValueError("run_id must be 1-64 safe filename characters")
    return run_id


def run_paths(run_id: str) -> dict[str, Path]:
    root = REMOTE_ROOT / validate_run_id(run_id)
    return {
        "root": root,
        "inputs": root / "inputs",
        "baseline": root / "baseline",
        "data": root / "data",
        "tokenizer": root / "tokenizer",
        "training": root / "training",
        "evaluation": root / "evaluation",
        "artifacts": root / "artifacts",
        "visual_webpii": root / "visual" / "webpii",
        "visual_synthetic": root / "visual" / "synthetic",
        "visual_supplemental_input": root / "inputs" / "visual-supplemental-ats",
        "visual_webpii_audit_input": root / "inputs" / "webpii-full-label-audit.json",
        "visual_supplemental": root / "visual" / "supplemental",
        "visual_dataset": root / "visual" / "dataset",
        "visual_training": root / "visual" / "training",
        "visual_artifacts": root / "visual" / "artifacts",
    }


@app.function(
    cpu=4.0,
    memory=16384,
    timeout=60 * 60,
    volumes={"/vol": run_volume, "/cache": cache_volume},
)
def baseline_remote(
    run_id: str,
    force: bool = False,
    evaluation_json: str | None = None,
) -> dict[str, Any]:
    import argparse
    import shutil

    from huggingface_hub import snapshot_download
    from training.freeze_rampart_baseline import freeze
    from training.semantic.rampart_reference import DEFAULT_CONTRACT, DEFAULT_FIXTURES

    paths = run_paths(run_id)
    if force and paths["baseline"].exists():
        shutil.rmtree(paths["baseline"])
    model_dir = Path(
        snapshot_download(
            repo_id="nationaldesignstudio/rampart",
            revision="b1993e4e68b082835b80ffc65acc03325ea2e501",
            allow_patterns=[
                "onnx/model_q4.onnx",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.txt",
            ],
        )
    )
    evaluation_path = None
    if evaluation_json:
        evaluation_path = paths["baseline"] / "baseline_evaluation.json"
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        parsed = json.loads(evaluation_json)
        evaluation_path.write_text(
            json.dumps(parsed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    result = freeze(
        argparse.Namespace(
            model_dir=model_dir,
            contract=DEFAULT_CONTRACT,
            fixtures=DEFAULT_FIXTURES,
            evaluation=evaluation_path,
            output_dir=paths["baseline"],
            freeze=evaluation_path is not None,
        )
    )
    run_volume.commit()
    cache_volume.commit()
    return result


@app.function(
    cpu=8.0,
    memory=32768,
    timeout=4 * 60 * 60,
    volumes={"/vol": run_volume, "/cache": cache_volume},
)
def prepare_remote(
    run_id: str, quick: bool = False, force: bool = False
) -> dict[str, Any]:
    import argparse
    import shutil

    from training.build_tokenizer import build
    from training.prepare_data import DATASET_REVISION, prepare
    from training.tokenizer_goldens import generate

    paths = run_paths(run_id)
    if force and paths["data"].exists():
        shutil.rmtree(paths["data"])
    if force and paths["tokenizer"].exists():
        shutil.rmtree(paths["tokenizer"])
    if paths["data"].exists() and (paths["data"] / "data_manifest.json").exists():
        manifest = json.loads((paths["data"] / "data_manifest.json").read_text())
    else:
        ocr_holdout = paths["inputs"] / "ocr_holdout.jsonl"
        ocr_holdout_manifest = paths["inputs"] / "ocr_holdout_manifest.json"
        manifest = prepare(
            argparse.Namespace(
                output_dir=paths["data"],
                seed=1311,
                dataset_revision=DATASET_REVISION,
                max_openpii_train=5_000 if quick else 250_000,
                max_openpii_validation=1_000 if quick else 50_000,
                synth_train=5_000 if quick else 100_000,
                synth_validation=1_000 if quick else 20_000,
                synth_holdout=2_000 if quick else 20_000,
                noise_fraction=0.35,
                negative_ratio=0.15,
                include_openpii=True,
                ocr_holdout=ocr_holdout if ocr_holdout.exists() else None,
                ocr_holdout_manifest=(
                    ocr_holdout_manifest if ocr_holdout_manifest.exists() else None
                ),
            )
        )
    tokenizer_manifest = build(
        argparse.Namespace(
            train=paths["data"] / "train.jsonl",
            validation=paths["data"] / "validation.jsonl",
            output_dir=paths["tokenizer"],
            vocab_size=8192,
            min_frequency=2,
            model_max_length=192,
            validation_records=1_000 if quick else 20_000,
            maximum_unknown_rate=0.001,
            base_model=manifest["base_model"]["id"],
            base_revision=manifest["base_model"]["revision"],
        )
    )
    golden_count = min(10_000, manifest["files"]["holdout"]["records"])
    tokenizer_goldens = generate(
        paths["data"] / "holdout.jsonl",
        paths["data"] / "tokenizer_goldens.jsonl",
        count=golden_count,
        model_id=str(paths["tokenizer"]),
        revision="local",
        max_length=192,
    )
    run_volume.commit()
    cache_volume.commit()
    return {
        "data": manifest,
        "tokenizer": tokenizer_manifest,
        "tokenizer_goldens": tokenizer_goldens,
    }


@app.function(
    gpu="L4",
    cpu=8.0,
    memory=32768,
    timeout=12 * 60 * 60,
    volumes={"/vol": run_volume, "/cache": cache_volume},
)
def train_remote(
    run_id: str, quick: bool = False, force: bool = False
) -> dict[str, Any]:
    import argparse
    import shutil

    from training.calibrate import calibrate
    from training.evaluate import evaluate
    from training.prepare_data import BASE_MODEL_ID, BASE_MODEL_REVISION
    from training.train import run

    paths = run_paths(run_id)
    if force and paths["training"].exists():
        shutil.rmtree(paths["training"])
    if force and paths["evaluation"].exists():
        shutil.rmtree(paths["evaluation"])
    if not (paths["data"] / "data_manifest.json").exists():
        raise RuntimeError("prepare stage has not completed")
    if not (paths["data"] / "ocr_holdout.jsonl").exists():
        raise RuntimeError(
            "replacement training requires a never-trained OCR-output holdout"
        )
    baseline_manifest = paths["baseline"] / "rampart_baseline.json"
    if not baseline_manifest.exists():
        raise RuntimeError(
            "a frozen Rampart baseline manifest must be uploaded before training"
        )

    training_result = run(
        argparse.Namespace(
            data_dir=paths["data"],
            output_dir=paths["training"],
            base_model=BASE_MODEL_ID,
            base_revision=BASE_MODEL_REVISION,
            tokenizer_dir=paths["tokenizer"],
            seed=1311,
            max_length=192,
            stride=32,
            epochs=0.25 if quick else 2.0,
            max_steps=100 if quick else -1,
            learning_rate=4e-5,
            weight_decay=0.01,
            warmup_ratio=0.06,
            train_batch_size=128,
            eval_batch_size=256,
            gradient_accumulation_steps=1,
            logging_steps=10 if quick else 25,
            dataloader_workers=4,
            num_proc=4,
            max_train_records=10_000 if quick else None,
            max_validation_records=2_000 if quick else None,
            resume_from_checkpoint=None,
            full_determinism=False,
            baseline_manifest=baseline_manifest,
            development_allow_unfrozen_baseline=False,
        )
    )
    threshold_path = paths["training"] / "thresholds.json"
    threshold_result = calibrate(
        argparse.Namespace(
            data_dir=paths["data"],
            model_dir=paths["training"] / "final",
            output=threshold_path,
            max_records=2_000 if quick else 20_000,
            max_length=192,
            stride=32,
            batch_size=256,
            num_proc=4,
            dataloader_workers=4,
            secret_target_recall=1.0,
            contextual_target_recall=0.95,
            default_target_recall=0.95,
        )
    )
    evaluation_result = evaluate(
        argparse.Namespace(
            data_dir=paths["data"],
            model_dir=paths["training"] / "final",
            thresholds=threshold_path,
            output_dir=paths["evaluation"],
            max_records_per_slice=2_000 if quick else 30_000,
            max_length=192,
            stride=32,
            batch_size=256,
            num_proc=4,
            dataloader_workers=4,
            enforce_gates=not quick,
        )
    )
    run_volume.commit()
    cache_volume.commit()
    return {
        "validation_metrics": training_result["validation_metrics"],
        "thresholds": threshold_result["thresholds"],
        "gates": evaluation_result["gates"],
    }


@app.function(
    cpu=8.0,
    memory=32768,
    timeout=6 * 60 * 60,
    volumes={"/vol": run_volume, "/cache": cache_volume},
)
def export_remote(
    run_id: str,
    quick: bool = False,
    force: bool = False,
    cross_runtime_json: str | None = None,
    artifact_license: str = "pending-maintainer-review",
) -> dict[str, Any]:
    import argparse
    import shutil

    from training.export_onnx import export
    from training.validate_artifacts import validate

    paths = run_paths(run_id)
    if force and paths["artifacts"].exists():
        shutil.rmtree(paths["artifacts"])
    if not (paths["training"] / "final" / "config.json").exists():
        raise RuntimeError("training stage has not completed")
    cross_runtime_path = None
    if cross_runtime_json:
        cross_runtime_path = paths["evaluation"] / "cross_runtime_report.json"
        cross_runtime_path.parent.mkdir(parents=True, exist_ok=True)
        parsed = json.loads(cross_runtime_json)
        cross_runtime_path.write_text(
            json.dumps(parsed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest = export(
        argparse.Namespace(
            model_dir=paths["training"] / "final",
            data_dir=paths["data"],
            thresholds=paths["training"] / "thresholds.json",
            output_dir=paths["artifacts"],
            training_run=paths["training"] / "training_run.json",
            evaluation=paths["evaluation"] / "evaluation.json",
            cross_runtime_report=cross_runtime_path,
            artifact_license=artifact_license,
            allow_development_artifact=quick,
            max_length=192,
            stride=32,
            agreement_records=128 if quick else 512,
            onnx_eval_records=2_000 if quick else 10_000,
            onnx_batch_size=256,
            num_proc=4,
            runtime_budget_bytes=20_000_000,
            minimum_argmax_agreement=0.98 if quick else 0.99,
            maximum_probability_error=0.05 if quick else 0.03,
            maximum_metric_drop=0.02 if quick else 0.01,
        )
    )
    source_goldens = paths["data"] / "tokenizer_goldens.jsonl"
    if source_goldens.exists():
        shutil.copy2(source_goldens, paths["artifacts"] / "tokenizer_goldens.jsonl")
    validation = validate(paths["artifacts"], release=not quick)
    run_volume.commit()
    cache_volume.commit()
    return {
        "manifest": manifest,
        "validation": validation,
    }


@app.function(
    image=visual_image,
    cpu=16.0,
    memory=32768,
    timeout=12 * 60 * 60,
    volumes={"/vol": run_volume, "/cache": cache_volume},
)
def visual_prepare_remote(
    run_id: str,
    quick: bool = False,
    force: bool = False,
    require_supplemental: bool = False,
) -> dict[str, Any]:
    import argparse
    import shutil

    from training.schema import sha256_file
    from training.visual.compose_dataset import compose
    from training.visual.prepare_synthetic import prepare as prepare_synthetic
    from training.visual.prepare_webpii import (
        EXPECTED_SPLIT_RECORDS,
        SOURCE_MAPPING_VERSION,
        WEBPII_ID,
        WEBPII_REVISION,
        prepare as prepare_webpii,
    )
    from training.visual.stage_supplemental import stage_prepared_supplemental

    paths = run_paths(run_id)
    webpii_audit_path = paths["visual_webpii_audit_input"]
    if not webpii_audit_path.is_file():
        raise RuntimeError(
            f"full WebPII label audit is required at {webpii_audit_path}"
        )
    try:
        webpii_audit = json.loads(webpii_audit_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("full WebPII label audit must be valid UTF-8 JSON") from exc
    expected_audit = {
        "dataset": WEBPII_ID,
        "revision": WEBPII_REVISION,
        "source_mapping_version": SOURCE_MAPPING_VERSION,
        "metadata_only": True,
        "values_persisted": False,
        "passed": True,
        "unmapped": {},
    }
    if not isinstance(webpii_audit, dict) or any(
        webpii_audit.get(key) != value for key, value in expected_audit.items()
    ):
        raise RuntimeError("full WebPII label audit does not match the pinned contract")
    measured_records = {
        split: webpii_audit.get("splits", {}).get(split, {}).get("records")
        for split in EXPECTED_SPLIT_RECORDS
    }
    if measured_records != EXPECTED_SPLIT_RECORDS:
        raise RuntimeError(
            "full WebPII label audit split counts do not match the pinned revision"
        )
    supplemental_input = paths["visual_supplemental_input"]
    supplemental_staging = paths["visual_supplemental"]
    if require_supplemental and not supplemental_input.exists():
        raise RuntimeError(
            "required prepared ATS supplement is missing at "
            f"{supplemental_input}; refusing to prepare a dataset without it"
        )
    if supplemental_staging.exists() and not supplemental_input.exists():
        raise RuntimeError(
            "a staged ATS supplement exists without its immutable uploaded input; "
            "refusing to use unverifiable stale data"
        )
    supplemental = (
        stage_prepared_supplemental(
            supplemental_input,
            supplemental_staging,
            force=force,
        )
        if supplemental_input.exists()
        else None
    )
    for name in ("visual_webpii", "visual_synthetic", "visual_dataset"):
        if force and paths[name].exists():
            shutil.rmtree(paths[name])
    webpii = prepare_webpii(
        argparse.Namespace(
            output_dir=paths["visual_webpii"],
            max_train=500 if quick else None,
            max_test=100 if quick else None,
            metadata_only=False,
            fail_on_unmapped=True,
            download_workers=12,
            force=force,
        )
    )
    synthetic = prepare_synthetic(
        argparse.Namespace(
            output_dir=paths["visual_synthetic"],
            train_records=1_000 if quick else 20_000,
            validation_records=250 if quick else 4_000,
            seed=1311,
            force=force,
        )
    )
    composed = compose(
        argparse.Namespace(
            webpii_root=paths["visual_webpii"],
            synthetic_root=paths["visual_synthetic"],
            supplemental_root=(
                supplemental_staging if supplemental is not None else None
            ),
            output_dir=paths["visual_dataset"],
        )
    )
    if supplemental is not None:
        composed["sources"]["supplemental_ats"]["remote_staging"] = supplemental
        (paths["visual_dataset"] / "manifest.json").write_text(
            json.dumps(composed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    composed["sources"]["webpii"]["full_label_audit"] = {
        "path": str(webpii_audit_path),
        "sha256": sha256_file(webpii_audit_path),
        "mapping_version": SOURCE_MAPPING_VERSION,
        "split_records": EXPECTED_SPLIT_RECORDS,
        "passed": True,
    }
    (paths["visual_dataset"] / "manifest.json").write_text(
        json.dumps(composed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_volume.commit()
    return {
        "webpii": webpii,
        "synthetic": synthetic,
        "supplemental": supplemental,
        "composed": composed,
    }


@app.function(
    image=visual_image,
    cpu=2.0,
    memory=8192,
    timeout=3 * 60 * 60,
    volumes={"/vol": run_volume},
)
def visual_audit_remote(run_id: str) -> dict[str, Any]:
    """Audit the complete pinned WebPII label vocabulary before downloading images."""
    import argparse

    from training.visual.audit_webpii_labels import audit

    paths = run_paths(run_id)
    output = paths["root"] / "visual" / "webpii-label-audit.json"
    result = audit(
        argparse.Namespace(
            output=output,
            max_train=None,
            max_test=None,
            fail_on_unmapped=False,
        )
    )
    run_volume.commit()
    return result


@app.function(
    image=visual_image,
    gpu="H200",
    cpu=16.0,
    memory=65536,
    timeout=24 * 60 * 60,
    volumes={"/vol": run_volume, "/cache": cache_volume},
)
def visual_train_remote(
    run_id: str,
    quick: bool = False,
    force: bool = False,
    commercial_license_approved: bool = False,
    epochs: int = 0,
    resume: bool = False,
    attempt_id: str = "default",
    seed_checkpoint: str = "",
    seed_sha256: str = "",
    seed_provenance_json: str = "",
) -> dict[str, Any]:
    import argparse
    import shutil

    from training.locked_asset import ensure_locked_training_asset
    from training.schema import sha256_file
    from training.visual.stage_training_dataset import (
        persist_stage_metadata,
        stage_composed_dataset,
        verify_persisted_stage_metadata,
    )
    from training.visual.train_detector import STABLE_OPTIMIZER_POLICY, train
    from training.visual.training_attempt import (
        DEFAULT_ATTEMPT_ID,
        attempt_output_dir,
        initialize_seeded_attempt,
        load_seeded_attempt,
        resolve_frozen_seed,
        validate_attempt_id,
    )

    if resume and force:
        raise ValueError("visual resume and force are mutually exclusive")
    attempt_id = validate_attempt_id(attempt_id)
    if (seed_checkpoint or seed_sha256 or seed_provenance_json) and resume:
        raise ValueError("a resumed visual attempt cannot replace its frozen seed")
    if seed_checkpoint and attempt_id == DEFAULT_ATTEMPT_ID:
        raise ValueError("a frozen-seed retry requires --visual-attempt")
    if seed_checkpoint and force:
        raise ValueError("a frozen-seed retry never permits force deletion")
    if attempt_id != DEFAULT_ATTEMPT_ID and not resume and not seed_checkpoint:
        raise ValueError("a new named visual attempt requires a frozen seed checkpoint")
    if bool(seed_checkpoint) != bool(seed_sha256):
        raise ValueError("visual seed checkpoint and SHA-256 must be supplied together")
    if resume:
        # A newly launched resume container must see the latest per-epoch commit.
        run_volume.reload()
    paths = run_paths(run_id)
    attempt_root = attempt_output_dir(paths["root"], attempt_id)
    checkpoint = paths["inputs"] / "detector-base.pt"
    license_path = paths["inputs"] / "detector-base-license.txt"
    provenance_path = paths["inputs"] / "detector-base-provenance.json"
    if not (paths["visual_dataset"] / "dataset.yaml").exists():
        raise RuntimeError("visual-prepare stage has not completed")
    preserved_provenance: dict[str, Any] | None = None
    attempt_contract: dict[str, Any] | None = None
    if attempt_id != DEFAULT_ATTEMPT_ID:
        if license_path.is_symlink() or not license_path.is_file():
            raise RuntimeError("named visual attempt requires the detector license input")
        if resume:
            attempt_contract = load_seeded_attempt(
                attempt_root=attempt_root,
                run_id=run_id,
                attempt_id=attempt_id,
                optimizer_policy=STABLE_OPTIMIZER_POLICY,
                license_path=license_path,
            )
        else:
            try:
                declared_seed_provenance = (
                    json.loads(seed_provenance_json)
                    if seed_provenance_json
                    else {
                        "snapshot": seed_checkpoint,
                        "purpose": "stable-retry-after-auto-MuSGD-instability",
                    }
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError("visual seed provenance must be valid JSON") from exc
            if not isinstance(declared_seed_provenance, dict):
                raise RuntimeError("visual seed provenance must be a JSON object")
            attempt_contract = initialize_seeded_attempt(
                attempt_root=attempt_root,
                run_id=run_id,
                attempt_id=attempt_id,
                source_checkpoint=resolve_frozen_seed(seed_checkpoint),
                expected_sha256=seed_sha256,
                seed_provenance=declared_seed_provenance,
                optimizer_policy=STABLE_OPTIMIZER_POLICY,
                license_path=license_path,
                inherited_provenance_path=(
                    provenance_path if provenance_path.exists() else None
                ),
            )
            # Preserve the immutable seed and attempt contract before the long
            # Volume-to-local dataset copy begins.
            run_volume.commit()
        checkpoint = Path(attempt_contract["checkpoint"])
        base_source = {
            "mode": "frozen-seed-training-attempt",
            "attempt_id": attempt_id,
            "attempt_manifest": str(attempt_contract["manifest_path"]),
            "attempt_manifest_sha256": attempt_contract["manifest_sha256"],
            "checkpoint_sha256": attempt_contract["checkpoint_sha256"],
            "provenance": attempt_contract["provenance"],
            "provenance_path": str(attempt_contract["provenance_path"]),
            "provenance_sha256": attempt_contract["provenance_sha256"],
        }
    else:
        supplied_override_parts = {
            "checkpoint": checkpoint.exists(),
            "license": license_path.exists(),
            "provenance": provenance_path.exists(),
        }
        if any(supplied_override_parts.values()) and not (
            supplied_override_parts["checkpoint"]
            and supplied_override_parts["license"]
        ):
            raise RuntimeError(
                "detector checkpoint override is incomplete; detector-base.pt and "
                "detector-base-license.txt are both required, and provenance cannot "
                "be supplied without them"
            )
        if checkpoint.exists() and license_path.exists():
            base_source = {
                "mode": "user-supplied-override",
                "path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
            if provenance_path.exists():
                try:
                    preserved_provenance = json.loads(
                        provenance_path.read_text(encoding="utf-8")
                    )
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        "detector-base-provenance.json must be valid UTF-8 JSON"
                    ) from exc
                if not isinstance(preserved_provenance, dict):
                    raise RuntimeError(
                        "detector-base-provenance.json must contain a JSON object"
                    )
                declared_checkpoint_sha = preserved_provenance.get(
                    "checkpoint_sha256"
                )
                if (
                    declared_checkpoint_sha is not None
                    and declared_checkpoint_sha != base_source["checkpoint_sha256"]
                ):
                    raise RuntimeError(
                        "detector-base-provenance.json checkpoint_sha256 does not "
                        "match detector-base.pt"
                    )
                base_source["provenance"] = preserved_provenance
                base_source["provenance_path"] = str(provenance_path)
                base_source["provenance_sha256"] = sha256_file(provenance_path)
        else:
            locked = ensure_locked_training_asset(
                "visual.ultralytics.yolo11n",
                checkpoint,
            )
            license_path.parent.mkdir(parents=True, exist_ok=True)
            license_path.write_text(locked["license"] + "\n", encoding="utf-8")
            base_source = {
                "mode": "pinned-agpl-development",
                "url": locked["url"],
                "revision": locked["source_revision"],
                "sha256": locked["sha256"],
            }
    progress_path = attempt_root / "safety-selection-progress.json"
    if resume and not progress_path.is_file():
        raise RuntimeError("visual resume requires persisted safety selection progress")
    if not resume and progress_path.exists() and not force:
        raise RuntimeError(
            "visual training progress already exists; pass --visual-resume to "
            "continue or --force to start over"
        )
    if force and attempt_root.exists():
        if attempt_id != DEFAULT_ATTEMPT_ID:
            raise RuntimeError("named visual attempts cannot be force-deleted")
        shutil.rmtree(attempt_root)
    if preserved_provenance is not None:
        attempt_root.mkdir(parents=True, exist_ok=True)
        persisted_provenance = attempt_root / "base_checkpoint_provenance.json"
        if resume:
            if (
                not persisted_provenance.is_file()
                or sha256_file(persisted_provenance) != sha256_file(provenance_path)
            ):
                raise RuntimeError(
                    "visual resume base-checkpoint provenance has changed"
                )
        else:
            shutil.copy2(provenance_path, persisted_provenance)
    source_roots = {
        "webpii": paths["visual_webpii"],
        "synthetic": paths["visual_synthetic"],
    }
    if paths["visual_supplemental"].exists():
        source_roots["supplemental_ats"] = paths["visual_supplemental"]
    local_stage = stage_composed_dataset(
        paths["visual_dataset"] / "dataset.yaml",
        paths["visual_dataset"] / "manifest.json",
        source_roots,
        Path("/tmp/plva-visual-training-data")
        / f"{validate_run_id(run_id)}-{attempt_id}",
        force=force,
        copy_workers=VISUAL_STAGE_COPY_WORKERS,
    )
    persisted_stage = (
        verify_persisted_stage_metadata(local_stage, attempt_root)
        if resume
        else persist_stage_metadata(local_stage, attempt_root)
    )

    def commit_visual_checkpoints() -> None:
        # The callback runs only after safety selection progress and the
        # last/best/safety checkpoints have been updated on the Volume.
        run_volume.commit()

    selected_epochs = epochs or (2 if quick else 100)
    if not 1 <= selected_epochs <= 500:
        raise ValueError("visual epochs must be between 1 and 500")
    result = train(
        argparse.Namespace(
            dataset_yaml=Path(local_stage["dataset_yaml"]),
            source_manifest=Path(local_stage["source_manifest"]),
            base_checkpoint=checkpoint,
            base_license=license_path.read_text(encoding="utf-8").strip(),
            base_source=base_source,
            commercial_license_approved=commercial_license_approved,
            output_dir=attempt_root,
            epochs=selected_epochs,
            image_size=640,
            batch_size=64,
            device="0",
            workers=16,
            # Ultralytics' built-in early stop follows its aggregate fitness,
            # while PLVA selects on minimum secret-class recall first.  Let the
            # requested schedule finish so aggregate fitness cannot stop a run
            # before the safety objective improves.
            patience=selected_epochs,
            seed=1311,
            data_staging=persisted_stage,
            local_staging_manifest=Path(local_stage["staging_manifest"]),
            run_id=run_id,
            attempt_id=attempt_id,
            attempt_manifest_sha256=(
                attempt_contract["manifest_sha256"]
                if attempt_contract is not None
                else None
            ),
            resume=resume,
            checkpoint_commit_hook=commit_visual_checkpoints,
        )
    )
    run_volume.commit()
    cache_volume.commit()
    return result


@app.function(
    image=visual_image,
    cpu=8.0,
    memory=32768,
    timeout=4 * 60 * 60,
    volumes={"/vol": run_volume, "/cache": cache_volume},
)
def visual_export_remote(
    run_id: str,
    force: bool = False,
    attempt_id: str = "default",
) -> dict[str, Any]:
    import argparse
    import shutil

    from training.visual.export_detector_onnx import export
    from training.visual.training_attempt import (
        attempt_artifact_dir,
        attempt_output_dir,
        validate_attempt_id,
    )

    paths = run_paths(run_id)
    attempt_id = validate_attempt_id(attempt_id)
    training_root = attempt_output_dir(paths["root"], attempt_id)
    artifact_root = attempt_artifact_dir(paths["root"], attempt_id)
    source = training_root / "safety-best.pt"
    license_path = paths["inputs"] / "detector-base-license.txt"
    if not source.exists() or not license_path.exists():
        raise RuntimeError("visual training or detector license input is missing")
    if force and artifact_root.exists():
        shutil.rmtree(artifact_root)
    result = export(
        argparse.Namespace(
            source=source,
            source_license=license_path.read_text(encoding="utf-8").strip(),
            training_manifest=training_root / "training_manifest.json",
            output_dir=artifact_root,
            image_size=640,
            opset=17,
            require_output=True,
        )
    )
    run_volume.commit()
    return result


def _read_volume_file(path: str) -> bytes:
    return b"".join(run_volume.read_file(path))


def download_run(run_id: str, destination: Path) -> dict[str, Any]:
    remote_root = f"runs/{validate_run_id(run_id)}/artifacts"
    manifest = json.loads(_read_volume_file(f"{remote_root}/model_manifest.json"))
    files = {
        "model_manifest.json",
        "MODEL_CARD.md",
        "tokenizer_goldens.jsonl",
        *[item["path"] for item in manifest["artifacts"].values()],
        "provenance/data_manifest.json",
        "provenance/training_run.json",
        "provenance/evaluation.json",
    }
    downloaded: list[str] = []
    for relative in sorted(files):
        try:
            content = _read_volume_file(f"{remote_root}/{relative}")
        except Exception:
            continue
        local_path = destination / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(content)
        downloaded.append(relative)
    from training.sync_model_lock import sync

    candidate_lock = sync(destination, LOCAL_REPO_ROOT / "models.lock.json")
    return {
        "destination": str(destination),
        "downloaded": downloaded,
        "semantic_candidate_lock": candidate_lock,
    }


def download_visual_run(
    run_id: str, destination: Path, attempt_id: str = "default"
) -> dict[str, Any]:
    from training.visual.training_attempt import (
        attempt_artifact_dir,
        attempt_output_dir,
        validate_attempt_id,
    )

    validate_run_id(run_id)
    attempt_id = validate_attempt_id(attempt_id)
    paths = run_paths(run_id)
    training_name = attempt_output_dir(paths["root"], attempt_id).name
    artifact_name = attempt_artifact_dir(paths["root"], attempt_id).name
    files = (
        f"visual/{artifact_name}/conversion_report.json",
        f"visual/{artifact_name}/detector.onnx",
        f"visual/{artifact_name}/detector_manifest.json",
        f"visual/{artifact_name}/MODEL_CARD.md",
        f"visual/{training_name}/training_manifest.json",
        f"visual/{training_name}/base_checkpoint_provenance.json",
        f"visual/{training_name}/frozen-seed-provenance.json",
        f"visual/{training_name}/training-attempt.json",
        f"visual/{training_name}/seed-incumbent.json",
    )
    downloaded = []
    for relative in files:
        try:
            content = _read_volume_file(f"runs/{run_id}/{relative}")
        except Exception:
            continue
        local_path = destination / Path(relative).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(content)
        downloaded.append(relative)
    if not downloaded:
        raise RuntimeError("no visual artifacts are available for this run")
    return {"destination": str(destination), "downloaded": downloaded}


@app.local_entrypoint()
def main(
    run_id: str = "plva-tagger-v1",
    stage: str = "baseline",
    quick: bool = False,
    force: bool = False,
    download: bool = False,
    baseline_evaluation: str = "",
    cross_runtime_report: str = "",
    artifact_license: str = "pending-maintainer-review",
    visual_commercial_license_approved: bool = False,
    visual_epochs: int = 0,
    visual_require_supplemental: bool = False,
    visual_resume: bool = False,
    visual_attempt: str = "default",
    visual_seed_checkpoint: str = "",
    visual_seed_sha256: str = "",
    visual_seed_provenance: str = "",
):
    from training.visual.training_attempt import validate_attempt_id

    visual_attempt = validate_attempt_id(visual_attempt)
    valid_stages = {
        "all",
        "baseline",
        "prepare",
        "train",
        "export",
        "download",
        "visual-prepare",
        "visual-audit",
        "visual-train",
        "visual-export",
        "visual-download",
    }
    if stage not in valid_stages:
        raise ValueError(f"stage must be one of {sorted(valid_stages)}")
    if visual_resume and force:
        raise ValueError("--visual-resume and --force are mutually exclusive")
    if visual_resume and stage != "visual-train":
        raise ValueError("--visual-resume is valid only with --stage visual-train")
    if (
        visual_seed_checkpoint or visual_seed_sha256 or visual_seed_provenance
    ) and stage != "visual-train":
        raise ValueError(
            "visual seed options are valid only with --stage visual-train"
        )
    seed_provenance_json = (
        Path(visual_seed_provenance).read_text(encoding="utf-8")
        if visual_seed_provenance
        else ""
    )
    validate_run_id(run_id)
    result: dict[str, Any] = {}
    baseline_json = (
        Path(baseline_evaluation).read_text(encoding="utf-8")
        if baseline_evaluation
        else None
    )
    cross_runtime_json = (
        Path(cross_runtime_report).read_text(encoding="utf-8")
        if cross_runtime_report
        else None
    )
    if stage in {"all", "baseline"}:
        result["baseline"] = baseline_remote.remote(
            run_id,
            force,
            baseline_json,
        )
    if stage in {"all", "prepare"}:
        result["prepare"] = prepare_remote.remote(run_id, quick, force)
    if stage in {"all", "train"}:
        result["train"] = train_remote.remote(run_id, quick, force)
    if stage in {"all", "export"}:
        result["export"] = export_remote.remote(
            run_id,
            quick,
            force,
            cross_runtime_json,
            artifact_license,
        )
    if stage == "visual-prepare":
        result["visual_prepare"] = visual_prepare_remote.remote(
            run_id,
            quick,
            force,
            visual_require_supplemental,
        )
    if stage == "visual-audit":
        result["visual_audit"] = visual_audit_remote.remote(run_id)
    if stage == "visual-train":
        # Long GPU work uses an asynchronous Modal call while this entrypoint
        # still waits and returns the structured result to the caller.
        visual_call = visual_train_remote.spawn(
            run_id,
            quick,
            force,
            visual_commercial_license_approved,
            visual_epochs,
            visual_resume,
            visual_attempt,
            visual_seed_checkpoint,
            visual_seed_sha256,
            seed_provenance_json,
        )
        result["visual_train"] = visual_call.get()
    if stage == "visual-export":
        result["visual_export"] = visual_export_remote.remote(
            run_id, force, visual_attempt
        )
    if stage == "visual-download":
        visual_directory = (
            "visual"
            if visual_attempt == "default"
            else f"visual-{visual_attempt}"
        )
        destination = LOCAL_TRAINING_DIR / "artifacts" / run_id / visual_directory
        result["visual_download"] = download_visual_run(
            run_id, destination, visual_attempt
        )
    if stage == "download" or download:
        destination = LOCAL_TRAINING_DIR / "artifacts" / run_id
        result["download"] = download_run(run_id, destination)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
