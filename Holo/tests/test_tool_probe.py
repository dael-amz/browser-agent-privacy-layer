"""Step 6.5: privacy-safe live tool-channel probe."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from plva_proxy.tool_probe import (
    MARKER_TEACHING,
    STRUCTURED_TEACHING,
    build_probe_request,
    evaluate_channel,
    summarize_grammar,
)


def test_summarize_grammar_lists_enums_and_admission() -> None:
    snapshot = {
        "structured_outputs": {
            "json_schema": {"properties": {"tool_name": {"enum": ["click", "write", "answer"]}}}
        }
    }
    summary = summarize_grammar(snapshot)
    assert ["answer", "click", "write"] in summary["enum_sets"]
    assert summary["admits_plva_tool"] is False


def test_summarize_grammar_admits_when_unconstrained() -> None:
    assert summarize_grammar({"structured_outputs": {"type": "object"}})["admits_plva_tool"] is True


def test_build_probe_request_is_synthetic_and_single_system() -> None:
    request = build_probe_request("Hcompany/Holo3-35B-A3B", STRUCTURED_TEACHING, None)
    roles = [m["role"] for m in request["messages"]]
    assert roles == ["system", "user"]
    assert request["stream"] is False
    assert "structured_outputs" not in request
    with_grammar = build_probe_request("m", STRUCTURED_TEACHING, {"type": "object"})
    assert with_grammar["structured_outputs"] == {"type": "object"}


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(base_url="https://upstream.test/v1", transport=httpx.MockTransport(handler))


def test_evaluate_channel_full_round_trip() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        content = (
            json.dumps(
                {
                    "tool_calls": [
                        {"tool_name": "plva_tool", "name": "add", "args": {"a": 17, "b": 25}}
                    ]
                }
            )
            if len(calls) == 1
            else "The sum is 42."
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": content}}]},
        )

    with _client(handler) as client:
        row = evaluate_channel(client, "m", "structured", STRUCTURED_TEACHING, None)
    assert row == {
        "channel": "structured",
        "grammar_attached": False,
        "status": 200,
        "emits": True,
        "parses": True,
        "round_trip": True,
    }


def test_evaluate_channel_records_non_emission() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "I cannot."}}]},
        )

    with _client(handler) as client:
        row = evaluate_channel(client, "m", "marker", MARKER_TEACHING, None)
    assert (row["emits"], row["parses"], row["round_trip"]) == (False, False, False)


def test_evaluate_channel_records_provider_rejection() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad template"})

    with _client(handler) as client:
        row = evaluate_channel(client, "m", "structured", STRUCTURED_TEACHING, None)
    assert row["status"] == 400
    assert row["emits"] is False
