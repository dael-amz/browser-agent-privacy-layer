from __future__ import annotations

import base64
import json
import logging
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

import plva_proxy.proxy as proxy
from plva_proxy.proxy import (
    TEST_HOOKS,
    Hooks,
    ProxyConfig,
    _chain_request_hooks,
    create_app,
    image_replacement_hook,
)
from plva_proxy.runtime_capture import LOOPBACK_HOST

UPSTREAM_KEY = "unit-test-hooks-upstream-key"
CONFIG = ProxyConfig(upstream_base_url="https://upstream.invalid/v1beta", api_key=UPSTREAM_KEY)

PLACEHOLDER_TOKEN = "{{PLACEHOLDER}}"
REPLACEMENT_VALUE = "resolved-value"

ACTION = {"action": "click", "target": "button"}
ACTION_JSON = json.dumps(ACTION, separators=(",", ":"))
_SPLIT = len(ACTION_JSON) // 2
ACTION_JSON_PART_ONE = ACTION_JSON[:_SPLIT]
ACTION_JSON_PART_TWO = ACTION_JSON[_SPLIT:]


def sse_event(document: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(document)}\n\n".encode()


def _parse_sse_events(raw: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for chunk in raw.split(b"\n\n"):
        for line in chunk.splitlines():
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:") :].strip()
            if payload == b"[DONE]":
                continue
            events.append(json.loads(payload))
    return events


SSE_ACTION_EVENTS = (
    sse_event({"id": "sse-1", "choices": [{"delta": {"role": "assistant"}}]}),
    sse_event({"id": "sse-1", "choices": [{"delta": {"content": ACTION_JSON_PART_ONE}}]}),
    sse_event({"id": "sse-1", "choices": [{"delta": {"content": ACTION_JSON_PART_TWO}}]}),
    sse_event({"id": "sse-1", "choices": [{"delta": {}, "finish_reason": "stop"}]}),
    b"data: [DONE]\n\n",
)


def chat_payload(*, stream: bool = False) -> dict[str, Any]:
    return {
        "model": "Hcompany/Holo3-35B-A3B",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
    }


def make_recording_json_upstream(
    response_body: dict[str, Any],
) -> tuple[FastAPI, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    app = FastAPI()

    @app.post("/v1beta/chat/completions")
    async def chat(request: Request) -> Response:
        body = await request.body()
        seen.append({"path": request.url.path, "headers": dict(request.headers), "body": body})
        return JSONResponse(response_body)

    return app, seen


def make_recording_sse_upstream(
    events: tuple[bytes, ...],
) -> tuple[FastAPI, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    app = FastAPI()

    @app.post("/v1beta/chat/completions")
    async def chat(request: Request) -> Response:
        body = await request.body()
        seen.append({"path": request.url.path, "headers": dict(request.headers), "body": body})
        return StreamingResponse(iter(events), media_type="text/event-stream")

    return app, seen


def make_proxy_client(
    transport: httpx.AsyncBaseTransport, *, hooks: Hooks | None = None
) -> httpx.AsyncClient:
    app = create_app(CONFIG, hooks=hooks, transport=transport)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")


def _add_marker_key(
    document: dict[str, Any], headers: dict[str, str]
) -> tuple[dict[str, Any], dict[str, str]]:
    return {**document, "injected_by_hook": True}, headers


def _replace_placeholder(document: dict[str, Any]) -> dict[str, Any]:
    mutated: dict[str, Any] = json.loads(json.dumps(document))
    for choice in mutated["choices"]:
        message = choice["message"]
        message["content"] = message["content"].replace(PLACEHOLDER_TOKEN, REPLACEMENT_VALUE)
    return mutated


CUSTOM_HOOKS = Hooks(on_request=_add_marker_key, on_response=_replace_placeholder)


# --- 1. request hook tags the upstream request, body otherwise unchanged ---


async def test_request_hook_tags_upstream_request_and_forwards_body_unchanged() -> None:
    upstream_body = {
        "id": "upstream-json",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}}],
    }
    upstream_app, seen = make_recording_json_upstream(upstream_body)
    payload = chat_payload()

    async with make_proxy_client(httpx.ASGITransport(app=upstream_app), hooks=TEST_HOOKS) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert len(seen) == 1
    assert seen[0]["headers"]["x-plva-hook"] == "request"
    assert json.loads(seen[0]["body"]) == payload


# --- 2. response hook rewrites JSON action content compactly ---


async def test_response_hook_rewrites_json_content_compactly_and_tags_header() -> None:
    upstream_body = {
        "id": "upstream-json",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"tool_calls": [ ] }'},
                "finish_reason": "stop",
            }
        ],
    }
    upstream_app, seen = make_recording_json_upstream(upstream_body)

    async with make_proxy_client(httpx.ASGITransport(app=upstream_app), hooks=TEST_HOOKS) as client:
        response = await client.post("/v1/chat/completions", json=chat_payload())

    assert response.status_code == 200
    assert response.headers["x-plva-hook"] == "response"
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["choices"][0]["message"]["content"] == '{"tool_calls":[]}'
    assert len(seen) == 1


# --- 3. response hook rewrites a buffered SSE stream ---


async def test_response_hook_rewrites_sse_stream_and_tags_header() -> None:
    upstream_app, seen = make_recording_sse_upstream(SSE_ACTION_EVENTS)

    async with make_proxy_client(httpx.ASGITransport(app=upstream_app), hooks=TEST_HOOKS) as client:
        response = await client.post("/v1/chat/completions", json=chat_payload(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-plva-hook"] == "response"
    assert response.content.endswith(b"data: [DONE]\n\n")
    reassembled = "".join(
        event["choices"][0]["delta"].get("content", "")
        for event in _parse_sse_events(response.content)
    )
    assert reassembled == ACTION_JSON
    assert len(seen) == 1


# --- 4. custom (non-TEST_HOOKS) hooks mutate both directions arbitrarily ---


async def test_custom_hooks_mutate_request_and_response_bodies() -> None:
    upstream_body = {
        "id": "upstream-json",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"click on {PLACEHOLDER_TOKEN} now"},
                "finish_reason": "stop",
            }
        ],
    }
    upstream_app, seen = make_recording_json_upstream(upstream_body)

    async with make_proxy_client(
        httpx.ASGITransport(app=upstream_app), hooks=CUSTOM_HOOKS
    ) as client:
        response = await client.post("/v1/chat/completions", json=chat_payload())

    assert response.status_code == 200
    assert len(seen) == 1
    assert json.loads(seen[0]["body"])["injected_by_hook"] is True
    assert (
        response.json()["choices"][0]["message"]["content"] == f"click on {REPLACEMENT_VALUE} now"
    )


# --- 5. fail-closed paths ---


async def test_request_hook_fails_closed_on_invalid_json_body_and_forwards_nothing() -> None:
    upstream_app, seen = make_recording_json_upstream({"id": "unused"})

    async with make_proxy_client(httpx.ASGITransport(app=upstream_app), hooks=TEST_HOOKS) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=b"not-valid-json",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 502
    assert response.json() == {"detail": "request hook failed"}
    assert seen == []


async def test_response_hook_fails_closed_when_action_content_is_not_json() -> None:
    upstream_body = {
        "id": "upstream-json",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "not-json-content"}}],
    }
    upstream_app, seen = make_recording_json_upstream(upstream_body)

    async with make_proxy_client(httpx.ASGITransport(app=upstream_app), hooks=TEST_HOOKS) as client:
        response = await client.post("/v1/chat/completions", json=chat_payload())

    assert response.status_code == 502
    assert response.json() == {"detail": "response hook failed"}
    assert len(seen) == 1


async def test_response_hook_fails_closed_on_sse_missing_done_terminal() -> None:
    events = (
        b'data: {"id":"sse-x","choices":[{"delta":{"role":"assistant"}}]}\n\n',
        b'data: {"id":"sse-x","choices":[{"delta":{"content":"partial"}}]}\n\n',
    )
    upstream_app, seen = make_recording_sse_upstream(events)

    async with make_proxy_client(httpx.ASGITransport(app=upstream_app), hooks=TEST_HOOKS) as client:
        response = await client.post("/v1/chat/completions", json=chat_payload(stream=True))

    assert response.status_code == 502
    assert response.json() == {"detail": "response hook failed"}
    assert len(seen) == 1


async def test_response_hook_fails_closed_on_native_tool_call_delta() -> None:
    events = (
        b'data: {"id":"sse-y","choices":[{"delta":{"role":"assistant"}}]}\n\n',
        b'data: {"id":"sse-y","choices":[{"delta":{"tool_calls":[{"id":"call-1"}]}}]}\n\n',
        b'data: {"id":"sse-y","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    )
    upstream_app, seen = make_recording_sse_upstream(events)

    async with make_proxy_client(httpx.ASGITransport(app=upstream_app), hooks=TEST_HOOKS) as client:
        response = await client.post("/v1/chat/completions", json=chat_payload(stream=True))

    assert response.status_code == 502
    assert response.json() == {"detail": "response hook failed"}
    assert len(seen) == 1


# --- 6. non-200 upstream is relayed verbatim, hooks or not ---


async def test_upstream_error_status_with_hooks_active_is_relayed_verbatim() -> None:
    def unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid key"})

    async with make_proxy_client(httpx.MockTransport(unauthorized), hooks=TEST_HOOKS) as client:
        response = await client.post("/v1/chat/completions", json=chat_payload())

    assert response.status_code == 401
    assert response.json() == {"error": "invalid key"}
    assert "x-plva-hook" not in response.headers


# --- 7. main() wires --hook through to create_app ---


def _run_main_with_spy(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> tuple[dict[str, Any], list[Hooks | None]]:
    monkeypatch.setattr(sys, "argv", argv)
    captured: dict[str, Any] = {}

    def fake_run(app: Any, **kwargs: Any) -> None:
        captured["app"] = app
        captured.update(kwargs)

    hooks_seen: list[Hooks | None] = []
    real_create_app = proxy.create_app

    def spy_create_app(
        config: ProxyConfig,
        *,
        hooks: Hooks | None = None,
        frame_store: proxy.FrameStore | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> FastAPI:
        hooks_seen.append(hooks)
        return real_create_app(config, hooks=hooks, frame_store=frame_store, transport=transport)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(proxy, "create_app", spy_create_app)

    proxy.main()
    return captured, hooks_seen


def test_main_passes_test_hooks_when_hook_flag_is_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "unit-test-main-hooks-key")

    captured, hooks_seen = _run_main_with_spy(monkeypatch, ["plva-proxy", "--hook", "test"])

    assert hooks_seen == [TEST_HOOKS]
    assert hooks_seen[0] is TEST_HOOKS
    assert captured["host"] == LOOPBACK_HOST


def test_main_passes_no_hooks_when_hook_flag_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "unit-test-main-hooks-key")

    captured, hooks_seen = _run_main_with_spy(monkeypatch, ["plva-proxy", "--hook", "none"])

    assert hooks_seen == [None]
    assert captured["host"] == LOOPBACK_HOST


# --- 8. sensitive content never appears in logs when hooks are active ---


async def test_hooks_never_log_sensitive_request_or_response_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "hook-test-sensitive-marker-9f3a"
    upstream_body = {
        "id": "upstream-json",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps({"note": marker})},
                "finish_reason": "stop",
            }
        ],
    }
    upstream_app, seen = make_recording_json_upstream(upstream_body)
    payload = chat_payload()
    payload["messages"].append({"role": "user", "content": marker})

    with caplog.at_level(logging.INFO, logger="plva_proxy.proxy"):
        async with make_proxy_client(
            httpx.ASGITransport(app=upstream_app), hooks=TEST_HOOKS
        ) as client:
            response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert len(seen) == 1
    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert marker not in joined


# --- 9. image replacement hook swaps every image_url for the static image ---


def make_png(path: Path) -> bytes:
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(path, format="PNG")
    return path.read_bytes()


def image_chat_payload() -> dict[str, Any]:
    original_data_url = "data:image/png;base64," + base64.b64encode(b"original-frame").decode()
    return {
        "model": "Hcompany/Holo3-35B-A3B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "click the button"},
                    {"type": "image_url", "image_url": {"url": original_data_url}},
                ],
            }
        ],
        "stream": False,
        "chat_template_kwargs": {"custom": True},
    }


async def test_image_hook_replaces_image_url_and_leaves_everything_else_intact(
    tmp_path: Path,
) -> None:
    png_path = tmp_path / "static.png"
    png_bytes = make_png(png_path)
    expected_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

    upstream_body = {
        "id": "upstream-json",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}}],
    }
    upstream_app, seen = make_recording_json_upstream(upstream_body)
    hooks = Hooks(on_request=image_replacement_hook(png_path))
    payload = image_chat_payload()

    async with make_proxy_client(httpx.ASGITransport(app=upstream_app), hooks=hooks) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert len(seen) == 1
    forwarded = json.loads(seen[0]["body"])
    parts = forwarded["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "click the button"}
    assert parts[1] == {"type": "image_url", "image_url": {"url": expected_url}}
    assert forwarded["model"] == payload["model"]
    assert forwarded["chat_template_kwargs"] == {"custom": True}
    assert forwarded["stream"] is False


async def test_image_hook_fails_closed_when_request_has_no_images(tmp_path: Path) -> None:
    png_path = tmp_path / "static.png"
    make_png(png_path)
    upstream_app, seen = make_recording_json_upstream({"id": "unused"})
    hooks = Hooks(on_request=image_replacement_hook(png_path))

    async with make_proxy_client(httpx.ASGITransport(app=upstream_app), hooks=hooks) as client:
        response = await client.post("/v1/chat/completions", json=chat_payload())

    assert response.status_code == 502
    assert response.json() == {"detail": "request hook failed"}
    assert seen == []


def test_image_hook_factory_rejects_missing_and_non_image_files(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        image_replacement_hook(tmp_path / "absent.png")

    not_an_image = tmp_path / "not-an-image.png"
    not_an_image.write_bytes(b"definitely not pixels")
    with pytest.raises(OSError):
        image_replacement_hook(not_an_image)


# --- 10. request hook chaining ---


def test_chain_request_hooks_returns_single_hook_when_other_side_is_none() -> None:
    assert _chain_request_hooks(None, None) is None
    assert _chain_request_hooks(_add_marker_key, None) is _add_marker_key
    assert _chain_request_hooks(None, _add_marker_key) is _add_marker_key


def test_chain_request_hooks_applies_in_order_with_first_output_fed_to_second() -> None:
    def first(
        document: dict[str, Any], headers: dict[str, str]
    ) -> tuple[dict[str, Any], dict[str, str]]:
        return {**document, "trace": ["first"]}, {**headers, "x-order": "first"}

    def second(
        document: dict[str, Any], headers: dict[str, str]
    ) -> tuple[dict[str, Any], dict[str, str]]:
        return {**document, "trace": [*document["trace"], "second"]}, {
            **headers,
            "x-order": headers["x-order"] + ",second",
        }

    chained = _chain_request_hooks(first, second)
    assert chained is not None

    document, headers = chained({"seed": True}, {"accept": "application/json"})
    assert document == {"seed": True, "trace": ["first", "second"]}
    assert headers == {"accept": "application/json", "x-order": "first,second"}


# --- 11. main() composes --hook-image with --hook test ---


def test_main_chains_image_hook_onto_test_hooks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("API_KEY", "unit-test-main-hooks-key")
    png_path = tmp_path / "static.png"
    make_png(png_path)

    _, hooks_seen = _run_main_with_spy(
        monkeypatch, ["plva-proxy", "--hook", "test", "--hook-image", str(png_path)]
    )

    assert len(hooks_seen) == 1
    hooks = hooks_seen[0]
    assert hooks is not None
    assert hooks is not TEST_HOOKS
    assert hooks.on_response is TEST_HOOKS.on_response
    assert hooks.on_request is not TEST_HOOKS.on_request
    assert hooks.on_request is not None


def test_main_rejects_unusable_hook_image_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("API_KEY", "unit-test-main-hooks-key")
    monkeypatch.setattr(sys, "argv", ["plva-proxy", "--hook-image", str(tmp_path / "missing.png")])
    with pytest.raises(SystemExit):
        proxy.main()
