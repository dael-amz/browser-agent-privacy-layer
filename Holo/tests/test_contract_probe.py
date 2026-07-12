from __future__ import annotations

import base64
import json
import sys

import httpx
import pytest

import plva_proxy.contract_probe as contract_probe
from plva_proxy.contract_probe import (
    COMPLETIONS_URL,
    MODEL_ID,
    CompletionSummary,
    ContractError,
    SSESummary,
    _probe,
    build_chat_payload,
    find_ready_model,
    summarize_completion,
    summarize_sse,
)

SYNTHETIC_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_build_chat_payload_uses_exact_model_and_synthetic_data_url() -> None:
    payload = build_chat_payload(SYNTHETIC_PNG, stream=False)

    assert payload["model"] == MODEL_ID == "Hcompany/Holo3-35B-A3B"
    assert payload["stream"] is False
    image_url = payload["messages"][-1]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    assert base64.b64decode(image_url.partition(",")[2]) == SYNTHETIC_PNG
    assert "api_key" not in repr(payload).lower()


def test_contract_constants_use_current_overshoot_v1_endpoint() -> None:
    assert COMPLETIONS_URL == "https://api.overshoot.ai/v1/chat/completions"


def test_find_ready_model_requires_exact_ready_entry() -> None:
    document = {
        "data": [
            {"id": MODEL_ID, "status": "ready", "object": "model"},
            {"id": "nearby-model", "status": "ready", "object": "model"},
        ]
    }

    assert find_ready_model(document) == document["data"][0]

    with pytest.raises(ContractError, match="not advertised"):
        find_ready_model({"data": []})

    with pytest.raises(ContractError, match="not ready"):
        find_ready_model({"data": [{"id": MODEL_ID, "status": "loading"}]})


@pytest.mark.parametrize(
    ("message", "expected_mode"),
    [
        ({"role": "assistant", "content": '{"tool_call": {}}'}, "content"),
        ({"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]}, "tool_calls"),
    ],
)
def test_summarize_completion_records_shape_not_content(
    message: dict[str, object], expected_mode: str
) -> None:
    sensitive_content = "synthetic-sensitive-marker"
    message = dict(message)
    if expected_mode == "content":
        message["content"] = sensitive_content
    document = {
        "id": "completion-id",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }

    summary = summarize_completion(document)

    assert summary.response_keys == ("choices", "id", "model", "object", "usage")
    assert summary.message_mode == expected_mode
    assert summary.model == MODEL_ID
    assert sensitive_content not in repr(summary)


def test_summarize_sse_reports_delta_shape_and_done_without_retaining_text() -> None:
    raw = b"".join(
        [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"synthetic-sensitive-marker"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )

    summary = summarize_sse(raw)

    assert summary.event_count == 2
    assert summary.done is True
    assert summary.delta_keys == ("content", "role")
    assert summary.has_tool_call_delta is False
    assert "synthetic-sensitive-marker" not in repr(summary)


def test_summarize_sse_fails_closed_on_malformed_data_event() -> None:
    with pytest.raises(ContractError, match="invalid SSE JSON"):
        summarize_sse(b"data: {not-json}\n\n")


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"", "no JSON data events"),
        (b'data: {"choices":[]}\n\n', "before \\[DONE\\]"),
    ],
)
def test_summarize_sse_fails_closed_on_empty_or_truncated_stream(raw: bytes, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        summarize_sse(raw)


@pytest.mark.parametrize("stream", [False, True])
def test_probe_checks_models_and_uses_bearer_auth_only_for_completion(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    client_type = httpx.Client
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path.endswith("/models"):
            assert "authorization" not in request.headers
            return httpx.Response(
                200,
                json={"data": [{"id": MODEL_ID, "status": "ready"}]},
            )

        assert request.url == httpx.URL(COMPLETIONS_URL)
        assert request.headers["authorization"] == "Bearer test-only-key"
        payload = json.loads(request.content)
        assert payload["model"] == MODEL_ID
        assert payload["stream"] is stream
        if stream:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
                    b'data: {"choices":[{"delta":{"content":"{}"}}]}\n\n'
                    b"data: [DONE]\n\n"
                ),
            )
        return httpx.Response(
            200,
            json={
                "model": MODEL_ID,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "plva_proxy.contract_probe.httpx.Client",
        lambda **kwargs: client_type(transport=transport, timeout=kwargs["timeout"]),
    )

    summary = _probe(api_key="test-only-key", stream=stream)

    assert isinstance(summary, SSESummary if stream else CompletionSummary)
    assert seen_paths == ["/v1/models", "/v1/chat/completions"]


def test_main_requires_key_without_printing_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["plva-probe"])

    with pytest.raises(SystemExit) as exc_info:
        contract_probe.main()

    assert exc_info.value.code == 2
    assert "API_KEY is required" in capsys.readouterr().err


def test_main_prints_only_summary_metadata(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = CompletionSummary(
        response_keys=("choices",),
        choice_keys=("message",),
        message_keys=("content",),
        message_mode="content",
        model=MODEL_ID,
        finish_reason="stop",
    )
    monkeypatch.setenv("API_KEY", "sensitive-test-key")
    monkeypatch.setattr(sys, "argv", ["plva-probe"])
    monkeypatch.setattr(contract_probe, "_probe", lambda **kwargs: summary)

    contract_probe.main()

    output = capsys.readouterr().out
    assert json.loads(output)["message_mode"] == "content"
    assert "sensitive-test-key" not in output


def test_main_reports_only_failure_type(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(**kwargs: object) -> CompletionSummary:
        raise ContractError("sensitive-upstream-detail")

    monkeypatch.setenv("API_KEY", "sensitive-test-key")
    monkeypatch.setattr(sys, "argv", ["plva-probe"])
    monkeypatch.setattr(contract_probe, "_probe", fail)

    with pytest.raises(SystemExit) as exc_info:
        contract_probe.main()

    assert exc_info.value.code == 1
    error = capsys.readouterr().err
    assert "ContractError" in error
    assert "sensitive" not in error
