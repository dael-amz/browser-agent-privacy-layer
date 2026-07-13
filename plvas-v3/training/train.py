from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .baseline_gate import require_frozen_rampart_baseline
from .modeling import (
    choose_subset,
    load_raw_datasets,
    tokenize_dataset,
    trainer_metrics,
)
from .prepare_data import BASE_MODEL_ID, BASE_MODEL_REVISION
from .schema import LABEL_CONFIG, LABEL_CONFIG_PATH, sha256_file


def git_commit() -> str | None:
    explicit = os.environ.get("PLVA_GIT_COMMIT")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def remap_embeddings(model, base_tokenizer, tokenizer) -> dict[str, Any]:
    import torch

    old_embeddings = model.get_input_embeddings().weight.detach().clone()
    old_vocab = base_tokenizer.get_vocab()
    new_vocab = tokenizer.get_vocab()
    old_unknown_id = base_tokenizer.unk_token_id
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    new_embeddings = model.get_input_embeddings().weight

    copied = 0
    composed = 0
    random_kept = 0
    with torch.no_grad():
        for token, new_id in sorted(new_vocab.items(), key=lambda item: item[1]):
            old_id = old_vocab.get(token)
            if old_id is not None:
                new_embeddings[new_id].copy_(old_embeddings[old_id])
                copied += 1
                continue
            surface = token[2:] if token.startswith("##") else token
            source_ids = [
                token_id
                for token_id in base_tokenizer(
                    surface,
                    add_special_tokens=False,
                )["input_ids"]
                if token_id != old_unknown_id
            ]
            if source_ids:
                source = old_embeddings[torch.tensor(source_ids, dtype=torch.long)]
                new_embeddings[new_id].copy_(source.mean(dim=0))
                composed += 1
            else:
                random_kept += 1
    model.config.vocab_size = len(tokenizer)
    return {
        "base_vocab_size": len(base_tokenizer),
        "runtime_vocab_size": len(tokenizer),
        "copied_tokens": copied,
        "composed_tokens": composed,
        "random_tokens": random_kept,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import accelerate
    import datasets
    import torch
    import transformers
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    development_override = bool(
        getattr(args, "development_allow_unfrozen_baseline", False)
    )
    baseline_path = getattr(args, "baseline_manifest", None)
    if development_override:
        baseline_gate: dict[str, Any] = {
            "status": "development-override",
            "baseline_frozen": False,
            "replacement_release_allowed": False,
        }
    else:
        baseline = require_frozen_rampart_baseline(baseline_path)
        baseline_gate = {
            "status": "frozen",
            "baseline_frozen": True,
            "replacement_release_allowed": True,
            "manifest": str(baseline_path),
            "manifest_sha256": sha256_file(baseline_path),
            "bootstrap_revision": baseline["bootstrap_model"]["revision"],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    raw = load_raw_datasets(args.data_dir)
    train_raw = choose_subset(raw["train"], args.max_train_records)
    validation_raw = choose_subset(raw["validation"], args.max_validation_records)

    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, revision=args.base_revision, use_fast=True
    )
    tokenizer = (
        AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
        if args.tokenizer_dir
        else base_tokenizer
    )
    if not tokenizer.is_fast:
        raise RuntimeError("PLVA requires a fast tokenizer with character offsets")
    train_data = tokenize_dataset(
        train_raw,
        tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        num_proc=args.num_proc,
        desc="Tokenizing train",
    )
    validation_data = tokenize_dataset(
        validation_raw,
        tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        num_proc=args.num_proc,
        desc="Tokenizing validation",
    )

    label_to_id = LABEL_CONFIG.label_to_id
    id_to_label = dict(enumerate(LABEL_CONFIG.labels))
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        num_labels=len(LABEL_CONFIG.labels),
        label2id=label_to_id,
        id2label=id_to_label,
        ignore_mismatched_sizes=True,
    )
    embedding_remap = (
        remap_embeddings(model, base_tokenizer, tokenizer)
        if args.tokenizer_dir
        else {
            "base_vocab_size": len(base_tokenizer),
            "runtime_vocab_size": len(tokenizer),
            "copied_tokens": len(tokenizer),
            "composed_tokens": 0,
            "random_tokens": 0,
        }
    )

    using_cuda = torch.cuda.is_available()
    using_bf16 = bool(using_cuda and torch.cuda.is_bf16_supported())
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="safety_score",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=args.full_determinism,
        fp16=bool(using_cuda and not using_bf16),
        bf16=using_bf16,
        tf32=bool(using_cuda and not args.full_determinism),
        optim="adamw_torch_fused" if using_cuda else "adamw_torch",
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=using_cuda,
        eval_accumulation_steps=16,
        remove_unused_columns=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
        compute_metrics=trainer_metrics(),
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    validation_metrics = trainer.evaluate()

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    shutil.copy2(LABEL_CONFIG_PATH, final_dir / "labels.json")
    if args.tokenizer_dir and (args.tokenizer_dir / "tokenizer_manifest.json").exists():
        shutil.copy2(
            args.tokenizer_dir / "tokenizer_manifest.json",
            final_dir / "tokenizer_manifest.json",
        )

    manifest_path = args.data_dir / "data_manifest.json"
    data_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    )
    run_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at.isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "git_commit": git_commit(),
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "data_manifest": data_manifest,
        "data_manifest_sha256": sha256_file(manifest_path)
        if manifest_path.exists()
        else None,
        "baseline_gate": baseline_gate,
        "invocation": sys.argv,
        "arguments": vars(args)
        | {
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "accelerate": accelerate.__version__,
            "cuda_available": using_cuda,
            "cuda_device": torch.cuda.get_device_name(0) if using_cuda else None,
            "cuda_device_count": torch.cuda.device_count() if using_cuda else 0,
            "cuda_device_memory_bytes": (
                torch.cuda.get_device_properties(0).total_memory if using_cuda else None
            ),
            "bf16": using_bf16,
            "compute_note": "Trainer runtime is recorded; energy and carbon were not measured.",
        },
        "counts": {
            "train_records": len(train_raw),
            "validation_records": len(validation_raw),
            "train_windows": len(train_data),
            "validation_windows": len(validation_data),
        },
        "embedding_remap": embedding_remap,
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_metrics,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
    }
    (args.output_dir / "training_run.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the PLVA compact BIO tagger"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--base-revision", default=BASE_MODEL_REVISION)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1311)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-validation-records", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--full-determinism", action="store_true")
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        help="frozen Rampart-backed screen/OCR baseline required for replacement training",
    )
    parser.add_argument(
        "--development-allow-unfrozen-baseline",
        action="store_true",
        help="exercise training code only; produced artifacts are not release candidates",
    )
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
