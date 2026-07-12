from __future__ import annotations

import numpy as np
import pytest

from plva_coreml.ocr import OCRFinding
from plva_coreml.semantics import (
    SemanticPipeline,
    Span,
    _aggregate_tokens,
    _detect_heuristics,
    _detect_sensitive_cues,
    _filter_contextual_hits,
    _premask,
)


def test_structured_rules_emit_exact_values_for_future_vault_entries() -> None:
    text = "Email alice@example.com and card 4111-1111-1111-1111"

    spans = _detect_heuristics(text)

    assert [(span.label, span.text) for span in spans] == [
        ("CREDIT_CARD", "4111-1111-1111-1111"),
        ("EMAIL", "alice@example.com"),
    ]


def test_premask_preserves_projection_to_raw_value() -> None:
    raw = "Send to alice@example.com now"
    spans = _detect_heuristics(raw)

    masked, starts, ends = _premask(raw, spans)

    sentinel_start = masked.index("[EMAIL]")
    assert raw[starts[sentinel_start] : ends[sentinel_start]] == "alice@example.com"


def test_sensitive_cues_do_not_redact_bare_field_labels() -> None:
    assert _detect_sensitive_cues("Account password") == []
    assert _detect_sensitive_cues("API key") == []


def test_sensitive_cues_require_a_value_for_password_fields() -> None:
    assert _detect_sensitive_cues("Password: hunter2") == ["PASSWORD"]


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("phone +1 (415) 555-0136", "PHONE"),
        ("sk-live-abcdefgh1234", "API_KEY"),
        ("eyJabcd.efghijkl.signature", "AUTH_TOKEN"),
        ("GB82 WEST 1234 5698 7654 32", "BANK_ACCOUNT"),
    ],
)
def test_sensitive_cues_match_original_secret_patterns(text: str, label: str) -> None:
    assert label in _detect_sensitive_cues(text)


def test_semantic_findings_emit_exact_sensitive_value_for_vault() -> None:
    pipeline = object.__new__(SemanticPipeline)
    pipeline._detect_ner = lambda *args: []
    finding = OCRFinding(0, 0, 100, 20, "alice@example.com", 0.9, 0.95)

    result = pipeline.classify((finding,))

    assert result.findings[0].sensitive is True
    assert result.findings[0].labels == ("EMAIL",)
    assert [(value.label, value.value) for value in result.findings[0].values] == [
        ("EMAIL", "alice@example.com")
    ]


def test_name_filter_rejects_isolated_low_confidence_word_fragments() -> None:
    hits = [
        Span(0, 3, "GIVEN_NAME", 0.551, "ner", "Mar"),
        Span(3, 6, "GIVEN_NAME", 0.529, "ner", "gul"),
    ]

    assert _filter_contextual_hits(hits, "Margulis") == []


def test_name_filter_keeps_complete_high_confidence_name() -> None:
    hits = [
        Span(0, 4, "GIVEN_NAME", 0.83, "ner", "John"),
        Span(5, 10, "SURNAME", 0.82, "ner", "Smith"),
    ]

    assert _filter_contextual_hits(hits, "John Smith") == hits


def test_name_filter_keeps_complete_moderate_confidence_name() -> None:
    hits = [
        Span(0, 4, "GIVEN_NAME", 0.462, "ner", "John"),
        Span(5, 10, "SURNAME", 0.85, "ner", "Smith"),
    ]

    assert _filter_contextual_hits(hits, "John Smith") == hits


def test_name_filter_keeps_contextual_single_name() -> None:
    hits = [Span(8, 12, "GIVEN_NAME", 0.6, "ner", "Jane")]

    assert _filter_contextual_hits(hits, "Contact Jane") == hits


def test_name_filter_keeps_high_confidence_multiword_name_from_alt_engine() -> None:
    hit = Span(0, 10, "GIVEN_NAME", 0.91, "ner", "John Smith")

    assert _filter_contextual_hits([hit], "John Smith") == [hit]


def test_name_filter_does_not_treat_generic_ui_words_as_name_context() -> None:
    hit = Span(11, 14, "SURNAME", 0.68, "ner", "int")

    assert _filter_contextual_hits([hit], "Account settings integer") == []
    assert _filter_contextual_hits([hit], "Scroll to integer") == []


def test_aggregate_tokens_does_not_continue_entity_across_ocr_findings() -> None:
    raw = "Alice\nMargulis"
    probabilities = np.zeros((2, 35), dtype=np.float32)
    probabilities[0, 1] = 0.9  # B-GIVEN_NAME
    probabilities[1, 2] = 0.8  # I-GIVEN_NAME

    spans = _aggregate_tokens(
        probabilities,
        [(0, 5), (6, 14)],
        raw,
        raw,
        list(range(len(raw))),
        list(range(1, len(raw) + 1)),
    )

    assert [(span.text, span.label) for span in spans] == [
        ("Alice", "GIVEN_NAME"),
        ("Margulis", "GIVEN_NAME"),
    ]
