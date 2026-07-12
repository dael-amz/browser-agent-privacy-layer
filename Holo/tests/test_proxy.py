from __future__ import annotations

import json
import logging
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

import plva_proxy.proxy as proxy
from plva_proxy.contract_probe import API_BASE_URL
from plva_proxy.proxy import ProxyConfig, _env_file_value, create_app

SENSITIVE_MARKER = "proxy-request-sensitive-marker"
UPSTREAM_KEY = "unit-test-upstream-key"
CONFIG = ProxyConfig(upstream_base_url="https://upstream.invalid/v1beta", api_key=UPSTREAM_KEY)
SSE_EVENTS = (
    b'data: {"id":"x","choices":[{"delta":{"role":"assistant"}}]}\n\n',
    b'data: {"id":"x","choices":[{"delta":{"content":"chunk"}}]}\n\n',
    b"data: [DONE]\n\n",
)


def chat_payload(*, stream: bool) -> dict[str, Any]:
    # Mirrors the captured runtime request: unknown keys must survive verbatim.
    return {
        "model": "Hcompany/Holo3-35B-A3B",
        "messages": [{"role": "system", "content": SENSITIVE_MARKER}],
        "stream": stream,
        "chat_template_kwargs": {"custom": True},
        "logit_bias": {},
        "structured_outputs": {"json": {"type": "object"}},
    }


def make_recording_upstream() -> tuple[FastAPI, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    app = FastAPI()

    @app.post("/v1beta/chat/completions")
    async def chat(request: Request) -> Response:
        body = await request.body()
        seen.append({"path": request.url.path, "headers": dict(request.headers), "body": body})
        if json.loads(body).get("stream"):
            return StreamingResponse(iter(SSE_EVENTS), media_type="text/event-stream")
        return JSONResponse(
            {"id": "upstream-json", "marker": "upstream-response-marker"},
            headers={"x-upstream-secret": "internal"},
        )

    @app.get("/v1beta/models")
    async def models(request: Request) -> Response:
        seen.append({"path": request.url.path, "headers": dict(request.headers), "body": b""})
        return JSONResponse({"object": "list", "data": []})

    return app, seen


def make_proxy_client(transport: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
    app = create_app(CONFIG, transport=transport)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")


async def test_health_endpoints_answer_locally_without_upstream_traffic() -> None:
    upstream_app, seen = make_recording_upstream()
    async with make_proxy_client(httpx.ASGITransport(app=upstream_app)) as client:
        for path in ("/health", "/v1/health"):
            response = await client.get(path)
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
    assert seen == []


async def test_chat_relay_forwards_body_verbatim_and_injects_credential() -> None:
    upstream_app, seen = make_recording_upstream()
    sent = json.dumps(chat_payload(stream=False)).encode()
    async with make_proxy_client(httpx.ASGITransport(app=upstream_app)) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=sent,
            headers={
                "content-type": "application/json",
                "authorization": "Bearer runtime-local-token",
                "user-agent": "holo-runtime/0.1.8",
            },
        )

    assert response.status_code == 200
    assert response.json()["marker"] == "upstream-response-marker"
    assert "x-upstream-secret" not in response.headers

    assert len(seen) == 1
    assert seen[0]["path"] == "/v1beta/chat/completions"
    assert seen[0]["body"] == sent
    assert seen[0]["headers"]["authorization"] == f"Bearer {UPSTREAM_KEY}"
    assert seen[0]["headers"]["host"] == "upstream.invalid"
    assert "holo" not in seen[0]["headers"].get("user-agent", "")


async def test_chat_relay_streams_sse_through() -> None:
    upstream_app, _ = make_recording_upstream()
    async with make_proxy_client(httpx.ASGITransport(app=upstream_app)) as client:
        response = await client.post("/v1/chat/completions", json=chat_payload(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == b"".join(SSE_EVENTS)


async def test_models_relay_forwards_with_credential() -> None:
    upstream_app, seen = make_recording_upstream()
    async with make_proxy_client(httpx.ASGITransport(app=upstream_app)) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}
    assert seen[0]["path"] == "/v1beta/models"
    assert seen[0]["headers"]["authorization"] == f"Bearer {UPSTREAM_KEY}"


async def test_upstream_error_status_is_relayed_verbatim() -> None:
    def unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid key"})

    async with make_proxy_client(httpx.MockTransport(unauthorized)) as client:
        response = await client.post("/v1/chat/completions", json=chat_payload(stream=False))

    assert response.status_code == 401
    assert response.json() == {"error": "invalid key"}


async def test_upstream_connection_failure_fails_closed_with_safe_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with caplog.at_level(logging.INFO, logger="plva_proxy.proxy"):
        async with make_proxy_client(httpx.MockTransport(refuse)) as client:
            response = await client.post("/v1/chat/completions", json=chat_payload(stream=False))

    assert response.status_code == 502
    assert response.json() == {"detail": "upstream request failed"}
    assert any("ConnectError" in record.message for record in caplog.records)
    joined = "\n".join(record.message for record in caplog.records)
    assert SENSITIVE_MARKER not in joined
    assert UPSTREAM_KEY not in joined


class _AbortingStream(httpx.AsyncByteStream):
    def __init__(self, *, content_type: str) -> None:
        self.content_type = content_type

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"id":"x"}\n\n'
        raise httpx.ReadError("mid-stream failure")


class _AbortingTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, content_type: str) -> None:
        self._content_type = content_type

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": self._content_type},
            stream=_AbortingStream(content_type=self._content_type),
            request=request,
        )


async def test_sse_failure_truncates_stream_instead_of_fabricating(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="plva_proxy.proxy"):
        async with make_proxy_client(
            _AbortingTransport(content_type="text/event-stream")
        ) as client:
            response = await client.post("/v1/chat/completions", json=chat_payload(stream=True))

    assert response.status_code == 200
    assert response.content == b'data: {"id":"x"}\n\n'
    assert b"[DONE]" not in response.content
    assert any("ReadError" in record.message for record in caplog.records)


async def test_json_body_read_failure_fails_closed() -> None:
    async with make_proxy_client(_AbortingTransport(content_type="application/json")) as client:
        response = await client.post("/v1/chat/completions", json=chat_payload(stream=False))

    assert response.status_code == 502
    assert response.json() == {"detail": "upstream response failed"}


async def test_request_logs_contain_only_size_and_timing_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    upstream_app, _ = make_recording_upstream()
    with caplog.at_level(logging.INFO, logger="plva_proxy.proxy"):
        async with make_proxy_client(httpx.ASGITransport(app=upstream_app)) as client:
            await client.post("/v1/chat/completions", json=chat_payload(stream=False))

    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "request_bytes=" in joined
    assert SENSITIVE_MARKER not in joined
    assert UPSTREAM_KEY not in joined


def test_env_file_value_reads_quoted_values_and_skips_other_lines(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nOTHER=1\nAPI_KEY="quoted-key"\n', "utf-8")

    assert _env_file_value(env_file, "API_KEY") == "quoted-key"
    assert _env_file_value(env_file, "MISSING") is None
    assert _env_file_value(tmp_path / "absent.env", "API_KEY") is None

    env_file.write_text("API_KEY=\n", "utf-8")
    assert _env_file_value(env_file, "API_KEY") is None


def test_main_runs_uvicorn_on_loopback_with_env_file_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text("API_KEY=from-env-file\n", "utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["plva-proxy", "--port", "18099"])

    captured: dict[str, Any] = {}

    def fake_run(app: Any, **kwargs: Any) -> None:
        captured["app"] = app
        captured.update(kwargs)

    configs: list[ProxyConfig] = []
    real_create_app = proxy.create_app

    def spy_create_app(
        config: ProxyConfig,
        *,
        hooks: proxy.Hooks | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> FastAPI:
        configs.append(config)
        return real_create_app(config, hooks=hooks, transport=transport)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(proxy, "create_app", spy_create_app)

    proxy.main()

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 18099
    assert captured["access_log"] is False
    assert configs == [ProxyConfig(upstream_base_url=API_BASE_URL, api_key="from-env-file")]


@pytest.mark.parametrize(
    "argv",
    [
        ["plva-proxy", "--port", "0"],
        ["plva-proxy", "--port", "70000"],
        ["plva-proxy", "--upstream", "ftp://provider.invalid"],
    ],
)
def test_main_rejects_invalid_arguments(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setenv("API_KEY", "set-but-unused")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        proxy.main()


def test_main_fails_closed_without_a_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["plva-proxy"])
    with pytest.raises(SystemExit):
        proxy.main()
