from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.schema import sha256_file


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT.parents[1] / "models" / "semantic" / "rampart"
DEFAULT_CONTRACT = ROOT / "rampart_contract.json"
DEFAULT_FIXTURES = ROOT / "rampart_fixtures.jsonl"


@dataclass
class RawSpan:
    source_label: str
    start: int
    end: int
    confidence: float


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def split_bio(label: str) -> tuple[str | None, str]:
    if label.startswith("B-") or label.startswith("I-"):
        return label[0], label[2:]
    return None, label


def decode_window(
    *,
    tokens: list[str],
    offsets: list[tuple[int, int]],
    label_ids: np.ndarray,
    probabilities: np.ndarray,
    id_to_label: dict[int, str],
) -> list[RawSpan]:
    spans: list[RawSpan] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            spans.append(
                RawSpan(
                    source_label=current["source_label"],
                    start=current["start"],
                    end=current["end"],
                    confidence=float(np.mean(current["scores"])),
                )
            )
        current = None

    for token, (start, end), label_id, token_probabilities in zip(
        tokens, offsets, label_ids.tolist(), probabilities
    ):
        label = id_to_label[int(label_id)]
        if start == end or label == "O":
            flush()
            continue
        prefix, source_label = split_bio(label)
        score = float(token_probabilities[int(label_id)])
        continues = (
            current is not None
            and current["source_label"] == source_label
            and (prefix == "I" or token.startswith("##"))
        )
        if continues:
            current["end"] = end
            current["scores"].append(score)
        else:
            flush()
            current = {
                "source_label": source_label,
                "start": start,
                "end": end,
                "scores": [score],
            }
    flush()
    return spans


def merge_overflow_spans(spans: list[RawSpan]) -> list[RawSpan]:
    merged: list[RawSpan] = []
    for span in sorted(
        spans, key=lambda item: (item.start, item.end, item.source_label)
    ):
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if existing.source_label == span.source_label
                and span.start < existing.end
                and existing.start < span.end
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(span)
            continue
        existing = merged[duplicate_index]
        merged[duplicate_index] = RawSpan(
            source_label=span.source_label,
            start=min(existing.start, span.start),
            end=max(existing.end, span.end),
            confidence=max(existing.confidence, span.confidence),
        )
    return sorted(merged, key=lambda item: (item.start, item.end, item.source_label))


class RampartReference:
    """Low-level Python contract runner for the pinned Rampart ONNX artifact.

    Production semantic behavior still comes from the pinned Rampart runtime,
    including its deterministic pre-mask and span-repair code. This class exists
    to freeze tokenizer inputs, raw model outputs, offsets, and PLVA label mapping.
    """

    def __init__(self, model_dir: Path, contract_path: Path = DEFAULT_CONTRACT):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = model_dir
        self.contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.config = json.loads(
            (model_dir / "config.json").read_text(encoding="utf-8")
        )
        self.id_to_label = {
            int(index): label for index, label in self.config["id2label"].items()
        }
        source_labels = {split_bio(label)[1] for label in self.id_to_label.values()} - {
            "O"
        }
        mapped_labels = set(self.contract["label_mapping"])
        if source_labels != mapped_labels:
            raise ValueError(
                "Rampart mapping must explicitly cover every source label: "
                f"missing={sorted(source_labels - mapped_labels)}, "
                f"extra={sorted(mapped_labels - source_labels)}"
            )
        model_contract = self.contract["model"]
        model_path = model_dir / "onnx" / "model_q4.onnx"
        if sha256_file(model_path) != model_contract["onnx_sha256"]:
            raise ValueError("Rampart ONNX SHA-256 does not match the contract")
        if len(self.id_to_label) != model_contract["label_count"]:
            raise ValueError("Rampart label count does not match the contract")
        if self.config["vocab_size"] != model_contract["vocab_size"]:
            raise ValueError("Rampart vocabulary size does not match the contract")

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(
            max_length=model_contract["max_sequence_length"],
            stride=model_contract["window_token_overlap"],
        )
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        if [item.name for item in self.session.get_inputs()] != [
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ]:
            raise ValueError("Rampart ONNX input contract changed")
        output = self.session.get_outputs()[0]
        if output.name != "logits" or output.shape[-1] != model_contract["label_count"]:
            raise ValueError("Rampart ONNX output contract changed")

    def infer(self, text: str) -> dict[str, Any]:
        inference_text = text.replace("-", " ")
        first = self.tokenizer.encode(inference_text)
        encodings = [first, *first.overflowing]
        window_results: list[dict[str, Any]] = []
        raw_spans: list[RawSpan] = []
        for encoding in encodings:
            ids = np.asarray([encoding.ids], dtype=np.int64)
            attention = np.asarray([encoding.attention_mask], dtype=np.int64)
            type_ids = np.asarray([encoding.type_ids], dtype=np.int64)
            logits = self.session.run(
                ["logits"],
                {
                    "input_ids": ids,
                    "attention_mask": attention,
                    "token_type_ids": type_ids,
                },
            )[0][0]
            probabilities = softmax(logits)
            label_ids = np.argmax(probabilities, axis=-1)
            raw_spans.extend(
                decode_window(
                    tokens=encoding.tokens,
                    offsets=encoding.offsets,
                    label_ids=label_ids,
                    probabilities=probabilities,
                    id_to_label=self.id_to_label,
                )
            )
            window_results.append(
                {
                    "input_ids": encoding.ids,
                    "attention_mask": encoding.attention_mask,
                    "token_type_ids": encoding.type_ids,
                    "tokens": encoding.tokens,
                    "offsets": [list(value) for value in encoding.offsets],
                    "predicted_label_ids": label_ids.tolist(),
                    "maximum_probabilities": [
                        round(float(value), 7)
                        for value in np.max(probabilities, axis=-1)
                    ],
                }
            )

        threshold = self.contract["model"]["minimum_anchor_score"]
        spans = []
        for span in merge_overflow_spans(raw_spans):
            mapping = self.contract["label_mapping"][span.source_label]
            if span.confidence < threshold:
                continue
            spans.append(
                {
                    "source_label": span.source_label,
                    "plva_class": mapping.get("plva_class"),
                    "disposition": mapping["disposition"],
                    "start": span.start,
                    "end": span.end,
                    "text": text[span.start : span.end],
                    "confidence": round(span.confidence, 7),
                }
            )
        return {"text": text, "windows": window_results, "model_spans": spans}


def generate_golden_vectors(
    model_dir: Path,
    contract_path: Path,
    fixture_path: Path,
    output: Path,
) -> dict[str, Any]:
    runner = RampartReference(model_dir, contract_path)
    fixtures = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vectors = [{"id": item["id"]} | runner.infer(item["text"]) for item in fixtures]
    result = {
        "schema_version": 1,
        "kind": "rampart-model-only",
        "warning": (
            "These vectors cover tokenizer, ONNX, offsets, and label adaptation only. "
            "Use the pinned Rampart package for deterministic recognizers and span repair."
        ),
        "model_revision": runner.contract["model"]["revision"],
        "model_sha256": runner.contract["model"]["onnx_sha256"],
        "contract_sha256": sha256_file(contract_path),
        "vectors": vectors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Rampart model-only golden vectors"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_golden_vectors(
        args.model_dir, args.contract, args.fixtures, args.output
    )
    print(
        json.dumps(
            {"vectors": len(result["vectors"]), "output": str(args.output)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
