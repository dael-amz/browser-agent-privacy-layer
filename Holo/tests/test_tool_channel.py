"""Tests for the Step 13/Step 7 tool channel bridge (src/plva_proxy/tool_channel.py).

These cover the marker scan/execute/queue contract, the deterministic and
semantic-executor routing, the bounded pending queue, the Step 7 approval and
watchdog automation, and the two privacy-hook integration points
(``privacy_request_hook``/``privacy_response_hook`` with ``tool_channel=``).
Fail-closed behavior is the spec: ``run_op`` must never raise, and every
denial or halt path is asserted explicitly. No network is used; the local LLM
and mediator are faked via ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from plva_proxy.local_llm import LLMConfig, LoopbackLLMClient
from plva_proxy.mediator import Mediator, MediatorCriteria
from plva_proxy.privacy import (
    PLACEHOLDER_MANIFEST_KEY,
    HistoryScrubber,
    PrivacyError,
    SafetyPolicy,
    SessionVault,
    privacy_request_hook,
    privacy_response_hook,
)
from plva_proxy.semantic_executor import SemanticExecutor
from plva_proxy.tool_channel import ToolChannel

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


def _no_op_classifier(texts: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{"sensitive": False, "values": []} for _ in texts]


def _scrubber(vault: SessionVault) -> HistoryScrubber:
    return HistoryScrubber(vault, _no_op_classifier)


# --- marker scan: deterministic sort + dedupe --------------------------------------


def test_scan_completion_text_deterministic_sort_and_dedupe() -> None:
    vault = SessionVault(nonce="ab12")
    token_charlie = vault.store("NAME", "Charlie Rivera")
    token_alice = vault.store("NAME", "Alice Zhang")
    channel = ToolChannel(vault, _scrubber(vault))
    text = f"Sorting now ⟦PLVA_TOOL:sort:«{token_charlie}»,«{token_alice}»:req1⟧ done"

    channel.scan_completion_text(text)

    pending = channel.drain_pending()
    assert pending == (
        f"⟦PLVA_TOOL_RESULT:sort:req1⟧ tokens in answer order: «{token_alice}», «{token_charlie}»",
    )

    # Second scan with the same request id is a no-op (dedupe).
    channel.scan_completion_text(text)
    assert channel.drain_pending() == ()


# --- marker with instruction routes to the semantic executor -----------------------


def test_scan_completion_text_marker_with_instruction_routes_to_semantic_executor() -> None:
    vault = SessionVault(nonce="ab12")
    token_charlie = vault.store("NAME", "Charlie Rivera")
    token_alice = vault.store("NAME", "Alice Zhang")

    def handler(request: httpx.Request) -> httpx.Response:
        return _completion({"answer": ["Alice Zhang", "Charlie Rivera"]})

    executor = SemanticExecutor(_client(handler), resolver=vault.resolve)
    channel = ToolChannel(vault, _scrubber(vault), executor=executor)
    text = f"⟦PLVA_TOOL:sort:«{token_charlie}»,«{token_alice}»:req2:most recent first⟧"

    channel.scan_completion_text(text)

    pending = channel.drain_pending()
    assert len(pending) == 1
    assert "⟦PLVA_TOOL_RESULT:sort:req2⟧" in pending[0]
    assert f"«{token_alice}»" in pending[0]
    assert f"«{token_charlie}»" in pending[0]


def test_scan_completion_text_marker_denied_without_executor() -> None:
    vault = SessionVault(nonce="ab12")
    token_charlie = vault.store("NAME", "Charlie Rivera")
    token_alice = vault.store("NAME", "Alice Zhang")
    channel = ToolChannel(vault, _scrubber(vault), executor=None)
    text = f"⟦PLVA_TOOL:sort:«{token_charlie}»,«{token_alice}»:req3:most recent first⟧"

    channel.scan_completion_text(text)

    pending = channel.drain_pending()
    assert len(pending) == 1
    assert "denied" in pending[0]
    assert "no local executor" in pending[0]


# --- select without instruction, unknown verb, marker cap --------------------------


def test_select_without_instruction_is_denied() -> None:
    vault = SessionVault(nonce="ab12")
    token_charlie = vault.store("NAME", "Charlie Rivera")
    token_alice = vault.store("NAME", "Alice Zhang")
    channel = ToolChannel(vault, _scrubber(vault))
    text = f"⟦PLVA_TOOL:select:«{token_charlie}»,«{token_alice}»:req4⟧"

    channel.scan_completion_text(text)

    pending = channel.drain_pending()
    assert len(pending) == 1
    assert "denied" in pending[0]
    assert "instruction" in pending[0]


def test_unknown_verb_inside_a_valid_marker_is_denied() -> None:
    vault = SessionVault(nonce="ab12")
    token_charlie = vault.store("NAME", "Charlie Rivera")
    token_alice = vault.store("NAME", "Alice Zhang")
    channel = ToolChannel(vault, _scrubber(vault))
    text = f"⟦PLVA_TOOL:leak:«{token_charlie}»,«{token_alice}»:req5⟧"

    channel.scan_completion_text(text)

    pending = channel.drain_pending()
    assert len(pending) == 1
    assert "unknown tool verb" in pending[0]


def test_more_than_two_markers_only_the_first_two_run() -> None:
    vault = SessionVault(nonce="ab12")
    token_charlie = vault.store("NAME", "Charlie Rivera")
    token_alice = vault.store("NAME", "Alice Zhang")
    channel = ToolChannel(vault, _scrubber(vault))
    text = " ".join(
        f"⟦PLVA_TOOL:sort:«{token_charlie}»,«{token_alice}»:req{n}⟧" for n in range(1, 4)
    )

    channel.scan_completion_text(text)

    pending = channel.drain_pending()
    assert len(pending) == 2
    assert any("req1" in line for line in pending)
    assert any("req2" in line for line in pending)
    assert not any("req3" in line for line in pending)


# --- deterministic sort failure never raises ----------------------------------------


def test_deterministic_sort_failure_is_swallowed_into_denied_notice() -> None:
    vault = SessionVault(nonce="ab12")
    token_charlie = vault.store("NAME", "Charlie Rivera")
    forged_token = "NAME_9_ab12"  # well-shaped but never issued by this vault
    channel = ToolChannel(vault, _scrubber(vault))
    text = f"⟦PLVA_TOOL:sort:«{token_charlie}»,«{forged_token}»:req6⟧"

    channel.scan_completion_text(text)  # must not raise

    pending = channel.drain_pending()
    assert len(pending) == 1
    assert "denied" in pending[0]


# --- bounded pending queue ------------------------------------------------------------


def test_pending_queue_is_bounded_at_eight_entries() -> None:
    vault = SessionVault(nonce="ab12")
    token_charlie = vault.store("NAME", "Charlie Rivera")
    token_alice = vault.store("NAME", "Alice Zhang")
    channel = ToolChannel(vault, _scrubber(vault))

    for index in range(10):
        channel.run_op("sort", (token_charlie, token_alice), None, f"req{index}")

    assert len(channel.drain_pending()) == 8


# --- consult_approval integration -----------------------------------------------------


def test_consult_approval_mints_a_grant_on_approve_and_resolve_action_then_succeeds() -> None:
    vault = SessionVault(nonce="ab12", policy=SafetyPolicy({"API_KEY": "approval"}))
    token = vault.store("API_KEY", "sk-synthetic-secret")
    mediator = Mediator(
        _client(lambda request: _completion({"decision": "approve", "reason": "ok"})),
        MediatorCriteria.defaults(),
    )
    channel = ToolChannel(vault, _scrubber(vault), mediator=mediator)
    call = {"tool_name": "write", "text": f"«{token}»", "origin": "https://x.test"}

    granted = channel.consult_approval(call)

    assert granted is True
    resolved = vault.resolve_action(((token, "text"),), tool_name="write", target="https://x.test")
    assert resolved[(token, "text")] == "sk-synthetic-secret"


def test_consult_approval_denies_and_resolve_action_still_raises() -> None:
    vault = SessionVault(nonce="ab12", policy=SafetyPolicy({"API_KEY": "approval"}))
    token = vault.store("API_KEY", "sk-synthetic-secret")
    mediator = Mediator(
        _client(lambda request: _completion({"decision": "deny", "reason": "no rule matches"})),
        MediatorCriteria.defaults(),
    )
    channel = ToolChannel(vault, _scrubber(vault), mediator=mediator)
    call = {"tool_name": "write", "text": f"«{token}»", "origin": "https://x.test"}

    granted = channel.consult_approval(call)

    assert granted is False
    with pytest.raises(PrivacyError, match="local approval"):
        vault.resolve_action(((token, "text"),), tool_name="write", target="https://x.test")


# --- watchdog ---------------------------------------------------------------------------


def test_watchdog_halts_after_threshold_denials_within_the_window() -> None:
    vault = SessionVault(nonce="ab12")

    def handler(request: httpx.Request) -> httpx.Response:
        return _completion({"action": "halt", "reason": "probing"})

    mediator = Mediator(_client(handler), MediatorCriteria.defaults())
    channel = ToolChannel(vault, _scrubber(vault), mediator=mediator)
    for _ in range(3):  # MediatorCriteria.defaults(): denied_resolution_threshold=3, window=8
        channel.record_denial(PrivacyError("synthetic denial"))

    channel.maybe_review()

    with pytest.raises(PrivacyError, match="halted"):
        channel.ensure_not_halted()


# --- privacy_request_hook integration ---------------------------------------------------


def test_privacy_request_hook_with_tool_channel_injects_teaching_pending_and_task_context() -> None:
    vault = SessionVault(nonce="a3f9")
    token_charlie = vault.store("NAME", "Charlie Rivera")
    token_alice = vault.store("NAME", "Alice Zhang")
    scrubber = _scrubber(vault)
    channel = ToolChannel(vault, scrubber)
    channel.run_op("sort", (token_charlie, token_alice), None, "req1")

    hook = privacy_request_hook(scrubber, tool_channel=channel)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Sort these two contacts"},
        {"role": "user", "content": [{"type": "text", "text": "Continue"}]},
    ]
    document, _ = hook(
        {"messages": messages, PLACEHOLDER_MANIFEST_KEY: {"message_index": 1, "items": []}},
        {},
    )

    serialized = json.dumps(document)
    assert "[PLVA_TOOLS]" in serialized
    target_content = document["messages"][1]["content"]
    assert "Placeholders visible" in target_content[-2]["text"]
    assert "⟦PLVA_TOOL_RESULT:sort:req1⟧" in target_content[-1]["text"]
    assert channel.task_context == "Sort these two contacts"


def test_privacy_request_hook_with_tool_channel_raises_after_a_halt() -> None:
    vault = SessionVault(nonce="a3f9")
    scrubber = _scrubber(vault)

    def handler(request: httpx.Request) -> httpx.Response:
        return _completion({"action": "halt", "reason": "probing"})

    mediator = Mediator(_client(handler), MediatorCriteria.defaults())
    channel = ToolChannel(vault, scrubber, mediator=mediator)
    for _ in range(3):
        channel.record_denial(PrivacyError("synthetic denial"))

    hook = privacy_request_hook(scrubber, tool_channel=channel)

    with pytest.raises(PrivacyError, match="halted"):
        hook({"messages": [{"role": "user", "content": "Continue"}]}, {})


# --- privacy_response_hook integration --------------------------------------------------


def test_privacy_response_hook_with_tool_channel_executes_marker_and_queues_result() -> None:
    vault = SessionVault(nonce="a3f9")
    vault.store("NAME", "Charlie Rivera")  # NAME_1_a3f9
    vault.store("NAME", "Alice Zhang")  # NAME_2_a3f9
    channel = ToolChannel(vault, _scrubber(vault))
    content = (
        '{"thought": "Let me sort ⟦PLVA_TOOL:sort:«NAME_1_a3f9»,«NAME_2_a3f9»:req1⟧ these", '
        '"tool_call": {"tool_name": "write", "text": "ok"}}'
    )
    document = {"choices": [{"message": {"content": content}}]}

    result = privacy_response_hook(vault, tool_channel=channel)(document)

    pending = channel.drain_pending()
    assert len(pending) == 1
    assert "⟦PLVA_TOOL_RESULT:sort:req1⟧" in pending[0]
    assert pending[0].index("NAME_2_a3f9") < pending[0].index("NAME_1_a3f9")  # Alice < Charlie
    action = json.loads(result["choices"][0]["message"]["content"])
    assert action["tool_call"]["text"] == "ok"


def test_privacy_response_hook_approval_denial_resolves_on_retry_when_approved() -> None:
    vault = SessionVault(nonce="a3f9", policy=SafetyPolicy({"API_KEY": "approval"}))
    token = vault.store("API_KEY", "sk-synthetic-secret")
    scrubber = _scrubber(vault)

    def approving(request: httpx.Request) -> httpx.Response:
        return _completion({"decision": "approve", "reason": "ok"})

    mediator = Mediator(_client(approving), MediatorCriteria.defaults())
    channel = ToolChannel(vault, scrubber, mediator=mediator)
    call = {"tool_name": "write", "text": token, "origin": "https://x.test"}
    document = {"choices": [{"message": {"content": json.dumps({"tool_call": call})}}]}

    result = privacy_response_hook(vault, tool_channel=channel)(document)

    action = json.loads(result["choices"][0]["message"]["content"])
    assert action["tool_call"]["text"] == "sk-synthetic-secret"


def test_privacy_response_hook_approval_denial_raises_and_records_event_with_a_denying_mediator() -> (
    None
):
    vault = SessionVault(nonce="a3f9", policy=SafetyPolicy({"API_KEY": "approval"}))
    token = vault.store("API_KEY", "sk-synthetic-secret")
    scrubber = _scrubber(vault)

    def denying(request: httpx.Request) -> httpx.Response:
        return _completion({"decision": "deny", "reason": "no rule matches"})

    mediator = Mediator(_client(denying), MediatorCriteria.defaults())
    channel = ToolChannel(vault, scrubber, mediator=mediator)
    call = {"tool_name": "write", "text": token, "origin": "https://x.test"}
    document = {"choices": [{"message": {"content": json.dumps({"tool_call": call})}}]}

    with pytest.raises(PrivacyError, match="local approval"):
        privacy_response_hook(vault, tool_channel=channel)(document)

    # No public accessor exposes the value-free trace; per the spec this is the
    # documented fallback to the private event log rather than a behavior-level assert.
    assert any(event.kind == "resolution_denied" for event in channel._events)
