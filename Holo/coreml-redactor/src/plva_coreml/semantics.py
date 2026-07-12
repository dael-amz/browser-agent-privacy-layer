"""Structured rules and Core ML Rampart NER over in-memory OCR findings."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
from tokenizers import Tokenizer

from plva_coreml.coreml_session import create_ane_session
from plva_coreml.model_cache import prepare_fixed_model
from plva_coreml.ocr import OCRFinding, PIIValue

SEQUENCE_LENGTH: Final = 128
RAMPART_MIN_SCORE: Final = 0.4
KEEP_LABELS: Final = frozenset({"CITY", "STATE", "ZIP_CODE"})
SEMANTIC_ENGINES: Final = ("rampart", "gliner2", "openai-pf")
# Alternative engines load frozen local copies only; no runtime downloads.
SEMANTIC_MODELS_ROOT: Final = Path(__file__).resolve().parents[2] / "models" / "semantic"
GLINER2_MODEL_DIR: Final = SEMANTIC_MODELS_ROOT / "gliner2-privacy-filter-PII-multi"
OPENAI_PF_MODEL_DIR: Final = SEMANTIC_MODELS_ROOT / "openai-privacy-filter"
# Alternative-engine vocabularies mapped onto the Rampart taxonomy so fusion,
# policy, and vault logic keep seeing one label space. Unmapped labels pass
# through uppercased and stay blocked by the unknown-class policy default.
_GLINER2_LABELS: Final = {
    "person": "GIVEN_NAME",
    "full_name": "GIVEN_NAME",
    "first_name": "GIVEN_NAME",
    "middle_name": "GIVEN_NAME",
    "last_name": "SURNAME",
    "date_of_birth": "DATE_OF_BIRTH",
    "email": "EMAIL",
    "phone_number": "PHONE",
    "address": "STREET_NAME",
    "street_address": "STREET_NAME",
    "postal_code": "ZIP_CODE",
    "city": "CITY",
    "state_or_region": "STATE",
    "government_id": "GOVERNMENT_ID",
    "national_id_number": "GOVERNMENT_ID",
    "passport_number": "PASSPORT",
    "drivers_license_number": "DRIVERS_LICENSE",
    "license_number": "DRIVERS_LICENSE",
    "tax_id": "TAX_ID",
    "tax_number": "TAX_ID",
    "bank_account": "BANK_ACCOUNT",
    "account_number": "BANK_ACCOUNT",
    "routing_number": "ROUTING_NUMBER",
    "iban": "BANK_ACCOUNT",
    "payment_card": "CREDIT_CARD",
    "card_number": "CREDIT_CARD",
    "card_expiry": "CREDIT_CARD",
    "card_cvv": "CREDIT_CARD",
    "username": "USERNAME",
    "ip_address": "IP_ADDRESS",
    "password": "PASSWORD",
    "secret": "SECRET",
    "api_key": "API_KEY",
    "access_token": "AUTH_TOKEN",
    "recovery_code": "SECRET",
}
_OPENAI_PF_LABELS: Final = {
    "private_person": "GIVEN_NAME",
    "private_name": "GIVEN_NAME",
    "private_email": "EMAIL",
    "private_phone": "PHONE",
    "private_address": "STREET_NAME",
    "private_id": "GOVERNMENT_ID",
    "private_financial": "BANK_ACCOUNT",
    "private_credential": "SECRET",
    "private_date": "DATE_OF_BIRTH",
}
ID_TO_LABEL: Final = (
    "O",
    "B-GIVEN_NAME",
    "I-GIVEN_NAME",
    "B-SURNAME",
    "I-SURNAME",
    "B-EMAIL",
    "I-EMAIL",
    "B-PHONE",
    "I-PHONE",
    "B-URL",
    "I-URL",
    "B-TAX_ID",
    "I-TAX_ID",
    "B-BANK_ACCOUNT",
    "I-BANK_ACCOUNT",
    "B-ROUTING_NUMBER",
    "I-ROUTING_NUMBER",
    "B-GOVERNMENT_ID",
    "I-GOVERNMENT_ID",
    "B-PASSPORT",
    "I-PASSPORT",
    "B-DRIVERS_LICENSE",
    "I-DRIVERS_LICENSE",
    "B-BUILDING_NUMBER",
    "I-BUILDING_NUMBER",
    "B-STREET_NAME",
    "I-STREET_NAME",
    "B-SECONDARY_ADDRESS",
    "I-SECONDARY_ADDRESS",
    "B-CITY",
    "I-CITY",
    "B-STATE",
    "I-STATE",
    "B-ZIP_CODE",
    "I-ZIP_CODE",
)


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    label: str
    score: float
    source: str
    text: str


@dataclass(frozen=True, slots=True)
class SemanticResult:
    findings: tuple[OCRFinding, ...]
    sensitive_count: int
    rampart_ms: float
    total_ms: float


class SemanticPipeline:
    """Warm Rampart session plus deterministic high-confidence PII rules."""

    def __init__(self, baseline: Path, cache: Path, engine: str = "rampart") -> None:
        if engine not in SEMANTIC_ENGINES:
            raise ValueError(f"engine must be one of: {', '.join(SEMANTIC_ENGINES)}")
        self._engine = engine
        self._alt: _Gliner2NER | _OpenAIPrivacyFilterNER | None = None
        if engine == "gliner2":
            self._alt = _Gliner2NER(GLINER2_MODEL_DIR)
            return
        if engine == "openai-pf":
            self._alt = _OpenAIPrivacyFilterNER(OPENAI_PF_MODEL_DIR)
            return
        model = prepare_fixed_model(
            baseline / "dist/semantic/rampart/onnx/model_q4.onnx",
            cache / "models/rampart-1x128.onnx",
            {
                "input_ids": (1, SEQUENCE_LENGTH),
                "attention_mask": (1, SEQUENCE_LENGTH),
                "token_type_ids": (1, SEQUENCE_LENGTH),
            },
        )
        self._session = create_ane_session(model, cache_directory=cache / "compiled/rampart")
        self._tokenizer = Tokenizer.from_file(
            str(baseline / "dist/semantic/rampart/tokenizer.json")
        )
        self._tokenizer.enable_truncation(max_length=SEQUENCE_LENGTH, stride=24)
        self._tokenizer.enable_padding(
            length=SEQUENCE_LENGTH,
            pad_id=0,
            pad_type_id=0,
            pad_token="[PAD]",
        )

    def warm(self) -> None:
        if self._alt is not None:
            self._alt.warm()
            return
        zeros = np.zeros((1, SEQUENCE_LENGTH), dtype=np.int64)
        self._run_rampart(zeros, zeros, zeros)

    def classify(self, findings: tuple[OCRFinding, ...]) -> SemanticResult:
        total_started = time.perf_counter()
        document, entries = _build_document(findings)
        if not document:
            return SemanticResult(findings, 0, 0.0, 0.0)
        heuristic = _detect_heuristics(document)
        masked, raw_starts, raw_ends = _premask(document, heuristic)
        started = time.perf_counter()
        contextual = self._detect_ner(masked, document, raw_starts, raw_ends)
        rampart_ms = (time.perf_counter() - started) * 1000
        spans = _merge_spans([*heuristic, *contextual])
        enriched: list[OCRFinding] = []
        sensitive_count = 0
        for index, finding in enumerate(findings):
            entry = entries.get(index)
            labels: list[str] = []
            sources: list[str] = []
            values: list[PIIValue] = []
            if entry is not None:
                entry_start, entry_end = entry
                hits = [span for span in spans if span.start < entry_end and span.end > entry_start]
                hits = _filter_contextual_hits(hits, finding.text)
                for hit in hits:
                    if hit.label in KEEP_LABELS:
                        continue
                    labels.append(hit.label)
                    sources.append("OCR+RAMPART" if hit.source == "ner" else "OCR+RULE")
                    local_start = max(0, hit.start - entry_start)
                    local_end = min(len(finding.text), hit.end - entry_start)
                    if local_end > local_start:
                        values.append(
                            PIIValue(
                                label=hit.label,
                                value=finding.text[local_start:local_end],
                                start=local_start,
                                end=local_end,
                                score=hit.score,
                                source=hit.source,
                            )
                        )
            cue_labels = _detect_sensitive_cues(finding.text)
            labels.extend(cue_labels)
            sources.extend("OCR+CUE" for _ in cue_labels)
            is_sensitive = finding.uncertain or bool(labels)
            if is_sensitive:
                sensitive_count += 1
            enriched.append(
                replace(
                    finding,
                    labels=tuple(dict.fromkeys((*finding.labels, *labels))),
                    sources=tuple(dict.fromkeys((*finding.sources, *sources))),
                    values=tuple(values),
                    sensitive=is_sensitive,
                )
            )
        return SemanticResult(
            findings=tuple(enriched),
            sensitive_count=sensitive_count,
            rampart_ms=rampart_ms,
            total_ms=(time.perf_counter() - total_started) * 1000,
        )

    def _detect_ner(
        self, masked: str, raw: str, raw_starts: list[int], raw_ends: list[int]
    ) -> list[Span]:
        if self._alt is not None:
            spans: list[Span] = []
            for start, end, label, score in self._alt.detect(masked):
                start = max(0, start)
                end = min(end, len(raw_ends))
                if end <= start or score < RAMPART_MIN_SCORE:
                    continue
                raw_start = raw_starts[start]
                raw_end = raw_ends[end - 1]
                if raw_end > raw_start:
                    spans.append(
                        Span(raw_start, raw_end, label, score, "ner", raw[raw_start:raw_end])
                    )
            return _merge_spans(spans)
        encoding = self._tokenizer.encode(masked)
        encodings = [encoding, *encoding.overflowing]
        spans: list[Span] = []
        for window in encodings:
            ids = np.asarray(window.ids, dtype=np.int64)[None]
            attention = np.asarray(window.attention_mask, dtype=np.int64)[None]
            type_ids = np.asarray(window.type_ids, dtype=np.int64)[None]
            logits = self._run_rampart(ids, attention, type_ids)[0]
            probabilities = _softmax(logits)
            spans.extend(
                _aggregate_tokens(
                    probabilities,
                    window.offsets,
                    masked,
                    raw,
                    raw_starts,
                    raw_ends,
                )
            )
        return _merge_spans(spans)

    def _run_rampart(
        self, ids: np.ndarray, attention: np.ndarray, type_ids: np.ndarray
    ) -> np.ndarray:
        output = self._session.run(
            None,
            {
                "input_ids": ids,
                "attention_mask": attention,
                "token_type_ids": type_ids,
            },
        )[0]
        if not isinstance(output, np.ndarray) or output.shape != (1, SEQUENCE_LENGTH, 35):
            raise RuntimeError("Rampart returned an unexpected output")
        return output


def _build_document(
    findings: tuple[OCRFinding, ...],
) -> tuple[str, dict[int, tuple[int, int]]]:
    text = ""
    entries: dict[int, tuple[int, int]] = {}
    for index, finding in enumerate(findings):
        recognized = finding.text.strip()
        if not recognized:
            continue
        if text:
            text += "\n"
        start = len(text)
        text += recognized
        entries[index] = (start, len(text))
    return text, entries


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _aggregate_tokens(
    probabilities: np.ndarray,
    offsets: list[tuple[int, int]],
    masked: str,
    raw: str,
    raw_starts: list[int],
    raw_ends: list[int],
) -> list[Span]:
    spans: list[Span] = []
    current: tuple[str, int, int, float, int] | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        label, start, end, score_sum, count = current
        score = score_sum / count
        if score >= RAMPART_MIN_SCORE and end > start and end <= len(raw_ends):
            raw_start = raw_starts[start]
            raw_end = raw_ends[end - 1]
            spans.append(Span(raw_start, raw_end, label, score, "ner", raw[raw_start:raw_end]))
        current = None

    for index, (start, end) in enumerate(offsets):
        if start == end or start >= len(masked):
            continue
        label_id = int(np.argmax(probabilities[index]))
        raw_label = ID_TO_LABEL[label_id]
        if raw_label == "O":
            flush()
            continue
        prefix, label = raw_label.split("-", 1)
        score = float(probabilities[index, label_id])
        if score < 0.15:
            flush()
            continue
        if current is not None and current[0] == label and prefix == "I":
            current = (label, current[1], end, current[3] + score, current[4] + 1)
        else:
            flush()
            current = (label, start, end, score, 1)
    flush()
    return spans


def _detect_heuristics(text: str) -> list[Span]:
    spans: list[Span] = []
    for match in re.finditer(r"\d(?:[ .-]?\d)*", text):
        digits = re.sub(r"\D", "", match.group())
        label = None
        if len(digits) in {14, 15, 16} and _luhn(digits):
            label = "CREDIT_CARD"
        elif len(digits) == 9 and _valid_ssn(digits):
            label = "SSN"
        if label:
            spans.append(Span(match.start(), match.end(), label, 1.0, "heuristic", match.group()))
    patterns = (
        ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        ("URL", r"\bhttps?://[^\s<>\"'\])}]+"),
        ("URL", r"\bwww\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s<>\"'\])}]*)?"),
        ("IP_ADDRESS", r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
        ("IP_ADDRESS", r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
    )
    for label, pattern in patterns:
        for match in re.finditer(pattern, text):
            spans.append(Span(match.start(), match.end(), label, 1.0, "heuristic", match.group()))
    return spans


def _premask(text: str, spans: list[Span]) -> tuple[str, list[int], list[int]]:
    ordered = sorted(_merge_spans(spans), key=lambda span: span.start)
    masked = ""
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for span in ordered:
        if span.start < cursor:
            continue
        for index in range(cursor, span.start):
            masked += text[index]
            starts.append(index)
            ends.append(index + 1)
        sentinel = f"[{span.label}]"
        masked += sentinel
        starts.extend([span.start] * len(sentinel))
        ends.extend([span.end] * len(sentinel))
        cursor = span.end
    for index in range(cursor, len(text)):
        masked += text[index]
        starts.append(index)
        ends.append(index + 1)
    return masked, starts, ends


def _merge_spans(spans: list[Span]) -> list[Span]:
    merged: list[Span] = []
    for span in sorted(spans, key=lambda item: (item.start, -item.end)):
        if not merged or span.start >= merged[-1].end:
            merged.append(span)
            continue
        previous = merged[-1]
        winner = _preferred(previous, span)
        if previous.end >= span.end:
            merged[-1] = winner
        else:
            merged[-1] = replace(
                winner,
                start=min(previous.start, span.start),
                end=max(previous.end, span.end),
            )
    return merged


def _preferred(left: Span, right: Span) -> Span:
    if left.score != right.score:
        return left if left.score > right.score else right
    if left.end - left.start != right.end - right.start:
        return left if left.end - left.start > right.end - right.start else right
    return left if left.source == "heuristic" else right


def _filter_contextual_hits(hits: list[Span], text: str) -> list[Span]:
    address_labels = {"BUILDING_NUMBER", "STREET_NAME", "SECONDARY_ADDRESS"}
    kinds = {hit.label for hit in hits if hit.label in address_labels}
    has_context = (
        bool(
            re.search(
                r"\b(?:address|street|st\.?|road|rd\.?|avenue|ave\.?|lane|ln\.?|boulevard|blvd\.?|drive|dr\.?|way|court|ct\.?|terrace|suite|apt\.?|apartment|ship\s+to|deliver\s+to)\b",
                text,
                re.IGNORECASE,
            )
        )
        or len(kinds) >= 2
    )
    return [hit for hit in hits if hit.label not in address_labels or has_context]


def _detect_sensitive_cues(text: str) -> list[str]:
    rules = (
        ("CVC", r"\b(?:cvv2?|cvc2?|security\s+c[o0]de)\b"),
        ("PRIVATE_KEY", r"\b(?:private\s+key|begin\s+(?:rsa\s+)?private\s+key)\b"),
        ("API_KEY", r"\b(?:api[\s_-]*key|client\s+secret|access[\s_-]*key)\b"),
        ("AUTH_TOKEN", r"\b(?:authorization|bearer|auth[\s_-]*token|access[\s_-]*token)\b"),
        ("PASSWORD", r"\b(?:password|passcode|passphrase)\b"),
        ("CARD_NUMBER", r"\b(?:card\s+(?:number|no\.?|#)|credit\s+card|debit\s+card|pan)\b"),
        (
            "GOVERNMENT_ID",
            r"\b(?:social\s+security|ssn|passport|driver'?s?\s+licen[cs]e|tax\s+id)\b",
        ),
        ("DOB", r"\b(?:date\s+of\s+birth|d[.\s/-]*o[.\s/-]*b[.]?)\b"),
        (
            "BANK_ACCOUNT",
            r"\b(?:routing\s+(?:number|no\.?|#)|bank\s+account|account\s+(?:number|no\.?|#)|iban)\b",
        ),
        ("SECRET", r"\b(?:secret\s+(?:key|value)|recovery\s+(?:code|phrase)|seed\s+phrase)\b"),
        (
            "PHONE",
            r"\b(?:phone|mobile|telephone|tel|call|sms)\b[^\n]{0,28}\+?\d[\d\s().-]{6,}\d(?:\s*(?:ext|x|extension)\s*\d+)?",
        ),
        (
            "API_KEY",
            r"\b(?:sk[-_](?:test|live)[-_][a-z0-9_-]{8,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{20,})\b",
        ),
        ("AUTH_TOKEN", r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]*"),
        ("BANK_ACCOUNT", r"\b[A-Z]{2}\d{2}(?:[\s-]?[A-Z0-9]){11,30}\b"),
    )
    return [label for label, pattern in rules if re.search(pattern, text, re.IGNORECASE)]


def _luhn(digits: str) -> bool:
    total = 0
    double = False
    for character in reversed(digits):
        value = int(character)
        if double:
            value *= 2
            if value > 9:
                value -= 9
        total += value
        double = not double
    return total % 10 == 0


def _valid_ssn(digits: str) -> bool:
    return not (
        len(digits) != 9
        or digits[:3] in {"000", "666"}
        or int(digits[:3]) >= 900
        or digits[3:5] == "00"
        or digits[5:] == "0000"
    )


def _missing_model_error(engine: str, repo: str, model_dir: Path, extra: str) -> RuntimeError:
    return RuntimeError(
        f"{engine} semantic model is not installed. Download a frozen copy with:\n"
        f"  huggingface-cli download {repo} --local-dir {model_dir}\n"
        f"{extra}Runtime downloads are never attempted; the worker fails closed instead."
    )


class _Gliner2NER:
    """GLiNER2 PII extractor behind the SemanticPipeline NER seam."""

    def __init__(self, model_dir: Path) -> None:
        if not model_dir.is_dir():
            raise _missing_model_error(
                "gliner2",
                "fastino/gliner2-privacy-filter-PII-multi",
                model_dir,
                "  (and install its runtime: uv add gliner2 in coreml-redactor)\n",
            )
        try:
            from gliner2 import GLiNER2
        except ImportError as exc:
            raise RuntimeError(
                "the gliner2 package is not installed in the Vision worker "
                "environment; run `uv add gliner2` in coreml-redactor"
            ) from exc
        self._model = GLiNER2.from_pretrained(str(model_dir))
        self._labels = list(_GLINER2_LABELS)

    def warm(self) -> None:
        self.detect("warm up john.smith@example.com")

    def detect(self, text: str) -> list[tuple[int, int, str, float]]:
        result = self._model.extract_entities(
            text,
            self._labels,
            threshold=0.5,
            include_confidence=True,
            include_spans=True,
        )
        spans: list[tuple[int, int, str, float]] = []
        for entity in _walk_entities(result):
            label = str(entity.get("label") or entity.get("type") or "")
            if not label:
                continue
            try:
                start = int(entity["start"])
                end = int(entity["end"])
            except (KeyError, TypeError, ValueError):
                continue
            score = float(entity.get("confidence") or entity.get("score") or 1.0)
            spans.append((start, end, _GLINER2_LABELS.get(label.lower(), label.upper()), score))
        return spans


def _walk_entities(node: object) -> list[dict]:
    """Collect span dicts from the nested gliner2 result shape defensively."""

    found: list[dict] = []
    if isinstance(node, dict):
        if "start" in node and "end" in node:
            found.append(node)
        else:
            for value in node.values():
                found.extend(_walk_entities(value))
    elif isinstance(node, (list, tuple)):
        for item in node:
            found.extend(_walk_entities(item))
    return found


class _OpenAIPrivacyFilterNER:
    """openai/privacy-filter token classifier behind the SemanticPipeline NER seam."""

    def __init__(self, model_dir: Path) -> None:
        if not model_dir.is_dir():
            raise _missing_model_error(
                "openai-pf",
                "openai/privacy-filter",
                model_dir,
                "  (and install its runtime: uv add transformers torch in coreml-redactor)\n",
            )
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "the transformers package is not installed in the Vision worker "
                "environment; run `uv add transformers torch` in coreml-redactor"
            ) from exc
        self._classifier = pipeline(
            task="token-classification",
            model=str(model_dir),
            aggregation_strategy="simple",
        )

    def warm(self) -> None:
        self.detect("My name is Alice Smith")

    def detect(self, text: str) -> list[tuple[int, int, str, float]]:
        spans: list[tuple[int, int, str, float]] = []
        for entity in self._classifier(text):
            group = str(entity.get("entity_group") or entity.get("entity") or "")
            start = entity.get("start")
            end = entity.get("end")
            if not group or start is None or end is None:
                continue
            score = float(entity.get("score", 0.0))
            spans.append(
                (int(start), int(end), _OPENAI_PF_LABELS.get(group.lower(), group.upper()), score)
            )
        return spans
