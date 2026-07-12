from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .prepare_data import BASE_MODEL_ID, BASE_MODEL_REVISION
from .schema import iter_jsonl


def generate(
    input_path: Path,
    output_path: Path,
    *,
    count: int,
    model_id: str,
    revision: str,
    max_length: int,
) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("tokenizer golden generation requires a fast tokenizer")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(input_path):
            encoded = tokenizer(
                record["text"],
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            golden = {
                "id": record["id"],
                "text": record["text"],
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "token_type_ids": encoded.get("token_type_ids"),
                "offset_mapping": [
                    list(offset) for offset in encoded["offset_mapping"]
                ],
                "special_tokens_mask": encoded["special_tokens_mask"],
            }
            line = json.dumps(
                golden, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            handle.write(line + "\n")
            digest.update((line + "\n").encode())
            written += 1
            if written >= count:
                break
    if written < count:
        raise RuntimeError(
            f"requested {count} goldens but input supplied only {written}"
        )
    return {
        "path": output_path.name,
        "records": written,
        "sha256": digest.hexdigest(),
        "model_id": model_id,
        "revision": revision,
        "max_length": max_length,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Python tokenizer parity vectors"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--model-id", default=BASE_MODEL_ID)
    parser.add_argument("--revision", default=BASE_MODEL_REVISION)
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()
    print(
        json.dumps(
            generate(
                args.input,
                args.output,
                count=args.count,
                model_id=args.model_id,
                revision=args.revision,
                max_length=args.max_length,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
