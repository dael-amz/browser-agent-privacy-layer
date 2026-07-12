from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from plva_proxy import mediator as mediator_module
from plva_proxy.local_llm import LLMConfig, LoopbackLLMClient
from plva_proxy.mediator import (
    ApprovalRequest,
    Mediator,
    MediatorCriteria,
    TraceEvent,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _llm_config() -> LLMConfig:
    return LLMConfig(
        base_url="http://127.0.0.1:8555/v1",
        model="test-model",
        timeout_seconds=5.0,
        max_tokens=64,
    )


def _client(handler: Handler) -> LoopbackLLMClient:
    return LoopbackLLMClient(_llm_config(), transport=httpx.MockTransport(handler))


def _completion(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}]},
    )


def _forbidden(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"no HTTP request expected in this fail-closed path: {request.url}")


def _request(**overrides: Any) -> ApprovalRequest:
    fields: dict[str, Any] = {
        "placeholder": "API_KEY_1_ab12",
        "pii_class": "API_KEY",
        "tool_name": "write",
        "argument_path": "args.content",
        "target": "https://service.test",
        "task_context": "Configure the demo service.",
        "cleartext_value": "sk-synthetic-1234567890",
    }
    fields.update(overrides)
    return ApprovalRequest(**fields)


# --- MediatorCriteria.load ----------------------------------------------------


def test_criteria_load_round_trips_a_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "criteria.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "classes": {"api_key": " Use only for the issuing service. "},
                "global_rules": [" When in doubt, deny. "],
                "trace_review": {"denied_resolution_threshold": 2, "window_steps": 5},
            }
        ),
        encoding="utf-8",
    )

    criteria = MediatorCriteria.load(path)

    assert criteria.class_guidance == {"API_KEY": "Use only for the issuing service."}
    assert criteria.global_rules == ("When in doubt, deny.",)
    assert criteria.denied_resolution_threshold == 2
    assert criteria.window_steps == 5


@pytest.mark.parametrize(
    "raw",
    [
        {"version": 1, "classes": {"NOT_A_CLASS": "guidance"}},
        {"version": 1, "classes": {}, "trace_review": {"denied_resolution_threshold": 0}},
        {"version": 1, "classes": {}, "trace_review": {"denied_resolution_threshold": "3"}},
        {"version": 1, "classes": {}, "trace_review": {"denied_resolution_threshold": 101}},
        {"version": 1, "classes": {}, "trace_review": {"window_steps": 0}},
        {"version": 2, "classes": {}},
        {"version": 1, "classes": {"API_KEY": "x" * 2001}},
    ],
)
def test_criteria_load_rejects_invalid_files(tmp_path: Path, raw: dict[str, Any]) -> None:
    path = tmp_path / "criteria.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError):
        MediatorCriteria.load(path)


# --- decide_approval ------------------------------------------------------------


def test_decide_approval_denies_without_guidance_and_makes_no_request() -> None:
    mediator = Mediator(_client(_forbidden), MediatorCriteria.defaults())

    verdict = mediator.decide_approval(_request(pii_class="SSN"))

    assert verdict.decision == "deny"
    assert "no user criteria exist" in verdict.reason
    assert verdict.scope is None


def test_decide_approval_approves_and_builds_the_grant_scope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _completion(
            {
                "decision": "approve",
                "reason": "matches the rule",
                "ttl_seconds": 120,
                "use_count": 2,
            }
        )

    mediator = Mediator(_client(handler), MediatorCriteria.defaults())
    request = _request()

    verdict = mediator.decide_approval(request)

    assert verdict.decision == "approve"
    assert verdict.scope is not None
    assert verdict.scope.ttl_seconds == 120.0
    assert verdict.scope.use_count == 2
    assert verdict.scope.target == request.target
    assert verdict.grant_kwargs(request) == {
        "tool_name": "write",
        "argument_path": "args.content",
        "target": "https://service.test",
        "ttl_seconds": 120.0,
        "use_count": 2,
    }


def test_decide_approval_denies_on_malformed_decision() -> None:
    mediator = Mediator(
        _client(lambda request: _completion({"decision": "yes"})), MediatorCriteria.defaults()
    )

    verdict = mediator.decide_approval(_request())

    assert verdict.decision == "deny"
    assert "malformed decision" in verdict.reason


def test_decide_approval_clamps_out_of_range_scope_to_defaults() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _completion(
            {"decision": "approve", "reason": "ok", "ttl_seconds": 999999, "use_count": 999}
        )

    mediator = Mediator(_client(handler), MediatorCriteria.defaults())

    verdict = mediator.decide_approval(_request())

    assert verdict.scope is not None
    assert verdict.scope.ttl_seconds == 60.0
    assert verdict.scope.use_count == 1


def test_decide_approval_denies_when_reason_echoes_the_hidden_value() -> None:
    secret = "sk-synthetic-echo-value"
    request = _request(cleartext_value=secret)

    def handler(_: httpx.Request) -> httpx.Response:
        return _completion({"decision": "approve", "reason": f"ok because {secret}"})

    mediator = Mediator(_client(handler), MediatorCriteria.defaults())

    verdict = mediator.decide_approval(request)

    assert verdict.decision == "deny"
    assert "withheld" in verdict.reason
    assert secret not in verdict.reason


def test_decide_approval_denies_and_never_raises_when_llm_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    mediator = Mediator(_client(handler), MediatorCriteria.defaults())
    request = _request()

    verdict = mediator.decide_approval(request)

    assert verdict.decision == "deny"
    assert verdict.scope is None
    with pytest.raises(ValueError, match="only approving verdicts"):
        verdict.grant_kwargs(request)


def test_decide_approval_applies_the_scrubber_hook_to_reason_and_steering() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _completion(
            {"decision": "approve", "reason": "looks fine", "steering": "be careful"}
        )

    scrubbed_calls: list[str] = []

    def scrubber(text: str) -> str:
        scrubbed_calls.append(text)
        return f"[scrubbed] {text}"

    mediator = Mediator(_client(handler), MediatorCriteria.defaults(), scrubber=scrubber)

    verdict = mediator.decide_approval(_request())

    assert verdict.reason == "[scrubbed] looks fine"
    assert verdict.steering == "[scrubbed] be careful"
    assert scrubbed_calls == ["looks fine", "be careful"]


# --- should_review ---------------------------------------------------------------


def test_should_review_is_false_below_threshold() -> None:
    mediator = Mediator(_client(_forbidden), MediatorCriteria.defaults())
    events = [TraceEvent(1, "resolution_denied", "x"), TraceEvent(2, "resolution_denied", "x")]

    assert mediator.should_review(events) is False


def test_should_review_is_true_at_threshold_within_window() -> None:
    mediator = Mediator(_client(_forbidden), MediatorCriteria.defaults())
    events = [
        TraceEvent(1, "resolution_denied", "x"),
        TraceEvent(2, "resolution_denied", "x"),
        TraceEvent(3, "blocked_class_attempt", "x"),
    ]

    assert mediator.should_review(events) is True


def test_should_review_ignores_suspicious_events_outside_the_window() -> None:
    mediator = Mediator(_client(_forbidden), MediatorCriteria.defaults())
    events = [
        TraceEvent(1, "resolution_denied", "x"),
        TraceEvent(2, "resolution_denied", "x"),
        TraceEvent(3, "resolution_denied", "x"),
        TraceEvent(20, "action", "unrelated"),
    ]

    assert mediator.should_review(events) is False


# --- review_trace -----------------------------------------------------------------


def test_review_trace_returns_halt_on_a_valid_reply() -> None:
    mediator = Mediator(
        _client(lambda request: _completion({"action": "halt", "reason": "repeated probing"})),
        MediatorCriteria.defaults(),
    )

    verdict = mediator.review_trace([TraceEvent(1, "resolution_denied", "x")])

    assert verdict.action == "halt"
    assert verdict.reason == "repeated probing"


def test_review_trace_halts_on_a_malformed_action() -> None:
    mediator = Mediator(
        _client(lambda request: _completion({"action": "pause"})), MediatorCriteria.defaults()
    )

    verdict = mediator.review_trace([TraceEvent(1, "resolution_denied", "x")])

    assert verdict.action == "halt"
    assert "malformed action" in verdict.reason


def test_review_trace_halts_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    mediator = Mediator(_client(handler), MediatorCriteria.defaults())

    verdict = mediator.review_trace([TraceEvent(1, "resolution_denied", "x")])

    assert verdict.action == "halt"


def test_review_trace_continues_without_http_for_an_empty_trace() -> None:
    mediator = Mediator(_client(_forbidden), MediatorCriteria.defaults())

    verdict = mediator.review_trace([])

    assert verdict.action == "continue"


# --- CLI ----------------------------------------------------------------------------


def test_cli_probe_reports_unreachable_and_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = mediator_module.main(["--url", "http://127.0.0.1:1/v1", "probe"])

    assert exit_code == 1
    assert "UNREACHABLE" in capsys.readouterr().out
