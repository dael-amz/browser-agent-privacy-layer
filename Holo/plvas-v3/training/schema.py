from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parent
LABEL_CONFIG_PATH = ROOT / "labels.json"
SPLITS = ("train", "validation", "holdout")


@dataclass(frozen=True)
class LabelConfig:
    labels: tuple[str, ...]
    entity_classes: tuple[str, ...]
    secret_classes: frozenset[str]
    contextual_classes: frozenset[str]

    @property
    def label_to_id(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.labels)}


def load_label_config(path: Path = LABEL_CONFIG_PATH) -> LabelConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    labels = tuple(raw["labels"])
    classes = tuple(raw["entity_classes"])
    expected = ["O"]
    for entity_class in classes:
        expected.extend((f"B-{entity_class}", f"I-{entity_class}"))
    if list(labels) != expected:
        raise ValueError(
            "labels.json order must be O followed by B-/I- for every entity class"
        )
    return LabelConfig(
        labels=labels,
        entity_classes=classes,
        secret_classes=frozenset(raw["secret_classes"]),
        contextual_classes=frozenset(raw["contextual_classes"]),
    )


LABEL_CONFIG = load_label_config()


def make_span(text: str, start: int, end: int, label: str) -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "label": label,
        "value": text[start:end],
    }


def canonical_value(label: str, value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = " ".join(normalized.split())
    if label in {"API_KEY", "AUTH_TOKEN", "PRIVATE_KEY", "PASSWORD"}:
        return normalized
    return normalized.casefold()


def validate_record(record: dict[str, Any]) -> None:
    required = {
        "id",
        "text",
        "spans",
        "split",
        "source",
        "template_id",
        "seed",
        "language",
        "region",
        "noisy",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(
            f"record {record.get('id', '<unknown>')} missing fields: {missing}"
        )
    if record["split"] not in SPLITS:
        raise ValueError(f"invalid split: {record['split']}")
    if not isinstance(record["text"], str) or not record["text"]:
        raise ValueError("text must be a non-empty string")
    if not isinstance(record["spans"], list):
        raise ValueError("spans must be a list")

    previous_end = -1
    for span in sorted(record["spans"], key=lambda item: (item["start"], item["end"])):
        start, end = int(span["start"]), int(span["end"])
        label = span["label"]
        if label not in LABEL_CONFIG.entity_classes:
            raise ValueError(f"unknown entity class {label}")
        if not (0 <= start < end <= len(record["text"])):
            raise ValueError(
                f"invalid span bounds {start}:{end} for {len(record['text'])}"
            )
        if start < previous_end:
            raise ValueError(f"overlapping spans in {record['id']}")
        actual = record["text"][start:end]
        if actual != span.get("value"):
            raise ValueError(
                f"span mismatch in {record['id']}: {actual!r} != {span.get('value')!r}"
            )
        previous_end = end


def record_values(record: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (span["label"], canonical_value(span["label"], span["value"]))
        for span in record["spans"]
    }


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()
    noisy_counts: Counter[str] = Counter()
    digest = hashlib.sha256()

    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                validate_record(record)
                encoded = (
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                handle.write(encoded.decode("utf-8"))
                digest.update(encoded)
                counts["records"] += 1
                counts["entities"] += len(record["spans"])
                source_counts[record["source"]] += 1
                region_counts[record["region"]] += 1
                noisy_counts[str(bool(record["noisy"])).lower()] += 1
                for span in record["spans"]:
                    class_counts[span["label"]] += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "records": counts["records"],
        "entities": counts["entities"],
        "class_counts": dict(sorted(class_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "region_counts": dict(sorted(region_counts.items())),
        "noisy_counts": dict(sorted(noisy_counts.items())),
    }


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "records": 0,
        "entities": 0,
        "classes": defaultdict(int),
        "sources": defaultdict(int),
        "templates": defaultdict(int),
    }
    for record in records:
        validate_record(record)
        summary["records"] += 1
        summary["entities"] += len(record["spans"])
        summary["sources"][record["source"]] += 1
        summary["templates"][record["template_id"]] += 1
        for span in record["spans"]:
            summary["classes"][span["label"]] += 1
    for key in ("classes", "sources", "templates"):
        summary[key] = dict(sorted(summary[key].items()))
    return summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
