from __future__ import annotations

import json
import socket
from typing import Any

import httpx
import pytest

from plva_proxy.local_llm import (
    LLMConfig,
    LocalLLMError,
    LocalLLMUnavailableError,
    LoopbackLLMClient,
    _flagged_sockets,
    extract_json_object,
    find_cleartext_leak,
    port_from_base_url,
    verify_no_egress,
)


def _config(**overrides: Any) -> LLMConfig:
    base = LLMConfig(
        base_url="http://127.0.0.1:8555/v1",
        model="test-model",
        timeout_seconds=5.0,
        max_tokens=64,
    )
    return base.replace(**overrides) if overrides else base


def _completion_body(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _client(handler: Any, **config_overrides: Any) -> LoopbackLLMClient:
    return LoopbackLLMClient(_config(**config_overrides), transport=httpx.MockTransport(handler))


# --- _require_loopback via LoopbackLLMClient -------------------------------


def test_non_loopback_host_is_rejected() -> None:
    with pytest.raises(ValueError, match="loopback"):
        LoopbackLLMClient(_config(base_url="http://10.0.0.5:8555/v1"))


def test_https_scheme_is_rejected() -> None:
    with pytest.raises(ValueError, match="plain http"):
        LoopbackLLMClient(_config(base_url="https://127.0.0.1:8555/v1"))


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:8555/v1",
        "http://localhost:8555/v1",
        "http://[::1]:8555/v1",
    ],
)
def test_loopback_hosts_are_accepted(base_url: str) -> None:
    client = LoopbackLLMClient(_config(base_url=base_url))
    try:
        assert client.config.base_url == base_url
    finally:
        client.close()


def test_loopback_client_rejects_out_of_range_timeout() -> None:
    with pytest.raises(ValueError, match="timeout"):
        LoopbackLLMClient(_config(timeout_seconds=0))


def test_loopback_client_rejects_out_of_range_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        LoopbackLLMClient(_config(max_tokens=5000))


def test_llm_config_from_env_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLVA_LOCAL_LLM_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("PLVA_LOCAL_LLM_MODEL", "custom-model")

    config = LLMConfig.from_env()

    assert config.base_url == "http://127.0.0.1:9999/v1"
    assert config.model == "custom-model"


# --- extract_json_object ----------------------------------------------------


def test_extract_json_object_from_plain_object() -> None:
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_object_wrapped_in_prose_and_code_fences() -> None:
    content = 'Here is my answer:\n```json\n{"a": 1}\n```\nDone.'
    assert extract_json_object(content) == {"a": 1}


def test_extract_json_object_takes_last_object_after_reasoning() -> None:
    content = 'Reasoning: {"foo": 1} is a draft. Actually the answer is {"decision": "approve"}.'
    assert extract_json_object(content) == {"decision": "approve"}


def test_extract_json_object_returns_none_when_absent() -> None:
    assert extract_json_object("no json object anywhere here") is None


# --- complete_json -----------------------------------------------------------


def test_complete_json_retries_once_and_parses_the_valid_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json=_completion_body("not json at all"))
        return httpx.Response(200, json=_completion_body('{"ok": true}'))

    client = _client(handler)

    parsed = client.complete_json(system="s", user="u")

    assert parsed == {"ok": True}
    assert calls == 2


def test_complete_json_raises_when_both_attempts_are_unparseable() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_completion_body("still not json"))

    client = _client(handler)

    with pytest.raises(LocalLLMError, match="parseable"):
        client.complete_json(system="s", user="u")
    assert calls == 2


# --- _chat 400-fallback -------------------------------------------------------


def test_chat_retries_without_response_format_after_400() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(400)
        return httpx.Response(200, json=_completion_body('{"ok": true}'))

    client = _client(handler)

    content = client._chat(system="s", user="u")

    assert content == '{"ok": true}'
    assert len(bodies) == 2
    assert "response_format" in bodies[0]
    assert "response_format" not in bodies[1]


def test_chat_with_schema_falls_back_json_schema_then_json_object_then_none() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) < 3:
            return httpx.Response(400)
        return httpx.Response(200, json=_completion_body('{"ok": true}'))

    client = _client(handler)
    schema = {"type": "object", "properties": {"answer": {"type": "array"}}}

    content = client._chat(system="s", user="u", schema=schema)

    assert content == '{"ok": true}'
    assert len(bodies) == 3
    assert bodies[0]["response_format"]["type"] == "json_schema"
    assert bodies[1]["response_format"]["type"] == "json_object"
    assert "response_format" not in bodies[2]


# --- timeout / transport failure ---------------------------------------------


def test_chat_connect_timeout_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("boom")

    client = _client(handler)

    with pytest.raises(LocalLLMUnavailableError):
        client._chat(system="s", user="u")


def test_chat_http_500_raises_unavailable() -> None:
    client = _client(lambda request: httpx.Response(500))

    with pytest.raises(LocalLLMUnavailableError):
        client._chat(system="s", user="u")


def test_chat_raises_on_unexpected_completion_shape() -> None:
    client = _client(lambda request: httpx.Response(200, json={"unexpected": "shape"}))

    with pytest.raises(LocalLLMError, match="unexpected completion shape"):
        client._chat(system="s", user="u")


def test_chat_raises_on_empty_completion_content() -> None:
    client = _client(lambda request: httpx.Response(200, json=_completion_body("   ")))

    with pytest.raises(LocalLLMError, match="empty completion"):
        client._chat(system="s", user="u")


# --- probe --------------------------------------------------------------------


def test_probe_happy_path_returns_model_ids() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"data": [{"id": "model-a"}, {"id": "model-b"}]})

    client = _client(handler)

    result = client.probe()

    assert result == {"endpoint": "http://127.0.0.1:8555/v1", "models": ["model-a", "model-b"]}


def test_probe_connection_error_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = _client(handler)

    with pytest.raises(LocalLLMUnavailableError):
        client.probe()


# --- find_cleartext_leak -------------------------------------------------------


def test_find_cleartext_leak_exact_match_never_repeats_the_value() -> None:
    leak = find_cleartext_leak(["The secret is sk-ABCDEF1234"], ["sk-ABCDEF1234"])

    assert leak is not None
    assert "sk-ABCDEF1234" not in leak


def test_find_cleartext_leak_matches_case_and_whitespace_reformatting() -> None:
    leak = find_cleartext_leak(["contains SK-ABCDEF 1234 duplicated"], ["sk-abcdef1234"])

    assert leak is not None
    assert "sk-abcdef1234" not in leak


def test_find_cleartext_leak_matches_reformatted_digit_sequence() -> None:
    leak = find_cleartext_leak(["card ending 4111-1111-1111-1111 stored"], ["4111111111111111"])

    assert leak is not None
    assert "digit-sequence" in leak
    assert "4111111111111111" not in leak


def test_find_cleartext_leak_ignores_short_values() -> None:
    assert find_cleartext_leak(["ab is present here"], ["ab"]) is None


def test_find_cleartext_leak_returns_none_for_clean_output() -> None:
    assert find_cleartext_leak(["nothing sensitive here"], ["super-secret-value"]) is None


# --- _flagged_sockets ------------------------------------------------------------


def test_flagged_sockets_allows_loopback_listen_and_established() -> None:
    output = "\n".join(
        [
            "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME",
            "holo 111 user 5u IPv4 0x1 0t0 TCP 127.0.0.1:8555 (LISTEN)",
            "holo 111 user 6u IPv4 0x2 0t0 TCP 127.0.0.1:54111->127.0.0.1:8555 (ESTABLISHED)",
        ]
    )

    assert _flagged_sockets(output) == ()


def test_flagged_sockets_flags_wildcard_listen_bind() -> None:
    output = "holo 111 user 5u IPv4 0x1 0t0 TCP *:8555 (LISTEN)"

    flagged = _flagged_sockets(output)

    assert len(flagged) == 1
    assert "non-loopback bind" in flagged[0]


def test_flagged_sockets_flags_remote_established_connection() -> None:
    output = "holo 111 user 6u IPv4 0x2 0t0 TCP 127.0.0.1:54111->93.184.216.34:443 (ESTABLISHED)"

    flagged = _flagged_sockets(output)

    assert len(flagged) == 1
    assert "remote connection" in flagged[0]


# --- verify_no_egress / port_from_base_url ---------------------------------------


def test_port_from_base_url_extracts_the_port() -> None:
    assert port_from_base_url("http://127.0.0.1:8555/v1") == 8555


def test_port_from_base_url_requires_an_explicit_port() -> None:
    with pytest.raises(ValueError, match="explicit port"):
        port_from_base_url("http://127.0.0.1/v1")


def test_verify_no_egress_reports_unchecked_when_nothing_listens() -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    finally:
        probe.close()

    report = verify_no_egress(port)

    assert report.checked is False
    assert report.flagged == ()
    assert report.clean is False
