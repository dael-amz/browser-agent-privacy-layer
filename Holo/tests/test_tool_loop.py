"""Step 6.5: relay tool-loop integration and grammar capture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from plva_proxy.proxy import HookError, Hooks, ProxyConfig, create_app, grammar_capture_hook
from plva_proxy.tools import ToolLoop, ToolRegistry, tool_teaching_request_hook


def test_grammar_capture_writes_schema_only(tmp_path: Path) -> None:
    out = tmp_path / "grammar.json"
    hook = grammar_capture_hook(out)
    document = {
        "model": "Hcompany/Holo3-35B-A3B",
        "messages": [{"role": "user", "content": "SECRET TEXT"}],
        "structured_outputs": {"json_schema": {"properties": {"tool_name": {"enum": ["click"]}}}},
        "chat_template_kwargs": {"reasoning": True},
    }
    returned, headers = hook(document, {"x-h": "1"})
    assert returned == document
    assert headers == {"x-h": "1"}
    snapshot = json.loads(out.read_text())
    assert snapshot["model"] == "Hcompany/Holo3-35B-A3B"
    assert snapshot["structured_outputs"]["json_schema"]["properties"]["tool_name"]["enum"] == [
        "click"
    ]
    assert "messages" not in snapshot
    assert "SECRET TEXT" not in out.read_text()
    assert sorted(snapshot["request_keys"]) == snapshot["request_keys"]


def test_grammar_capture_only_first_request(tmp_path: Path) -> None:
    out = tmp_path / "grammar.json"
    hook = grammar_capture_hook(out)
    hook({"model": "first", "structured_outputs": {}}, {})
    hook({"model": "second", "structured_outputs": {}}, {})
    assert json.loads(out.read_text())["model"] == "first"


def test_grammar_capture_write_failure_raises_hook_error(tmp_path: Path) -> None:
    hook = grammar_capture_hook(tmp_path)  # a directory: write_text raises OSError
    with pytest.raises(HookError):
        hook({"model": "m", "structured_outputs": {}}, {})


def _completion_body(content: str) -> dict[str, Any]:
    return {
        "id": "c1",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _tool_call_content() -> str:
    return json.dumps(
        {"tool_calls": [{"tool_name": "plva_tool", "name": "add", "args": {"a": 17, "b": 25}}]}
    )


def _final_action_content() -> str:
    return json.dumps({"tool_calls": [{"tool_name": "write", "text": "42"}]})


def _app_with_scripted_upstream(
    contents: list[str], seen: list[dict[str, Any]], *, max_rounds: int = 4
) -> FastAPI:
    def handler(request: httpx.Request) -> httpx.Response:
        document = json.loads(request.content)
        seen.append(document)
        index = min(len(seen), len(contents)) - 1
        return httpx.Response(200, json=_completion_body(contents[index]))

    return create_app(
        ProxyConfig(upstream_base_url="https://upstream.test/v1", api_key="key"),
        hooks=Hooks(on_request=tool_teaching_request_hook()),
        tool_loop=ToolLoop(ToolRegistry(), max_rounds=max_rounds),
        transport=httpx.MockTransport(handler),
    )


def test_tool_loop_round_trips_and_forwards_only_final_action() -> None:
    seen: list[dict[str, Any]] = []
    app = _app_with_scripted_upstream([_tool_call_content(), _final_action_content()], seen)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "add the numbers"}]},
        )
    assert response.status_code == 200
    assert len(seen) == 2
    follow_messages = seen[1]["messages"]
    assert follow_messages[-2]["role"] == "assistant"
    assert follow_messages[-1]["role"] == "user"
    assert follow_messages[-1]["content"].startswith("[PLVA_TOOL_RESULT] add returned: 42")
    assert seen[1]["stream"] is False
    final = json.loads(response.json()["choices"][0]["message"]["content"])
    assert final["tool_calls"][0]["tool_name"] == "write"


def test_tool_loop_exceeding_max_rounds_fails_closed() -> None:
    seen: list[dict[str, Any]] = []
    app = _app_with_scripted_upstream([_tool_call_content()], seen, max_rounds=2)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "loop forever"}]},
        )
    assert response.status_code == 502


def test_tool_loop_handles_sse_upstream_and_reemits_sse() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            chunks = (
                b"".join(
                    f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode()
                    for event in (
                        {
                            "id": "c1",
                            "created": 1,
                            "model": "m",
                            "choices": [
                                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                            ],
                        },
                        {
                            "id": "c1",
                            "created": 1,
                            "model": "m",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": _tool_call_content()},
                                    "finish_reason": "stop",
                                }
                            ],
                        },
                    )
                )
                + b"data: [DONE]\n\n"
            )
            return httpx.Response(
                200, content=chunks, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=_completion_body(_final_action_content()))

    app = create_app(
        ProxyConfig(upstream_base_url="https://upstream.test/v1", api_key="key"),
        tool_loop=ToolLoop(ToolRegistry()),
        transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "go"}]},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert len(calls) == 2
    assert b"write" in response.content
    assert b"plva_tool" not in response.content


def test_marker_channel_round_trips_through_relay() -> None:
    seen: list[dict[str, Any]] = []
    marker = '⟦TOOL⟧{"name": "echo", "args": {"text": "pong"}}⟦/TOOL⟧'
    app = _app_with_scripted_upstream([marker, _final_action_content()], seen)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "ping"}]},
        )
    assert response.status_code == 200
    assert len(seen) == 2
    assert "echo returned: pong" in seen[1]["messages"][-1]["content"]
