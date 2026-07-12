from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .prepare_data import BASE_MODEL_ID, BASE_MODEL_REVISION
from .schema import iter_jsonl, sha256_file


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def text_iterator(path: Path) -> Iterable[str]:
    for record in iter_jsonl(path):
        yield record["text"]


def build(args: argparse.Namespace) -> dict:
    from tokenizers import BertWordPieceTokenizer
    from transformers import AutoTokenizer, BertTokenizerFast

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer_tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=True,
        lowercase=True,
    )
    record_count = sum(1 for _ in iter_jsonl(args.train))
    trainer_tokenizer.train_from_iterator(
        text_iterator(args.train),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        length=record_count,
    )
    trainer_tokenizer.save_model(str(args.output_dir))
    trainer_tokenizer.save(str(args.output_dir / "tokenizer.json"))

    tokenizer = BertTokenizerFast(
        tokenizer_file=str(args.output_dir / "tokenizer.json"),
        do_lower_case=True,
        strip_accents=True,
        model_max_length=args.model_max_length,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokenizer.save_pretrained(args.output_dir)
    if len(tokenizer) > args.vocab_size:
        raise RuntimeError(
            f"trained vocabulary {len(tokenizer)} exceeds {args.vocab_size}"
        )

    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        use_fast=True,
    )
    base_vocab = base_tokenizer.get_vocab()
    new_vocab = tokenizer.get_vocab()
    overlapping = set(base_vocab) & set(new_vocab)

    unknown_tokens = 0
    evaluated_tokens = 0
    validation_records = 0
    for record in iter_jsonl(args.validation):
        encoded = tokenizer(record["text"], add_special_tokens=False)
        ids = encoded["input_ids"]
        unknown_tokens += sum(token_id == tokenizer.unk_token_id for token_id in ids)
        evaluated_tokens += len(ids)
        validation_records += 1
        if validation_records >= args.validation_records:
            break
    unknown_rate = unknown_tokens / max(1, evaluated_tokens)
    if unknown_rate > args.maximum_unknown_rate:
        raise RuntimeError(
            f"tokenizer unknown rate {unknown_rate:.4%} exceeds {args.maximum_unknown_rate:.4%}"
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "BertWordPieceTokenizer",
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": len(tokenizer),
        "min_frequency": args.min_frequency,
        "model_max_length": args.model_max_length,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "tokens_shared_with_base": len(overlapping),
        "tokens_new_to_base": len(new_vocab) - len(overlapping),
        "validation_records": validation_records,
        "validation_tokens": evaluated_tokens,
        "unknown_tokens": unknown_tokens,
        "unknown_rate": unknown_rate,
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(args.output_dir.iterdir())
            if path.is_file()
        },
    }
    (args.output_dir / "tokenizer_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the compact PLVA WordPiece tokenizer"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--model-max-length", type=int, default=192)
    parser.add_argument("--validation-records", type=int, default=20_000)
    parser.add_argument("--maximum-unknown-rate", type=float, default=0.001)
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--base-revision", default=BASE_MODEL_REVISION)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
