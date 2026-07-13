from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from plva_proxy.privacy import (
    PLACEHOLDER_MANIFEST_PREFIX,
    HistoryScrubber,
    SafetyPolicy,
    SessionVault,
    StubRedactor,
    StubSpan,
    VaultRedactor,
    privacy_request_hook,
    privacy_response_hook,
)
from plva_proxy.proxy import (
    FrameStore,
    Hooks,
    ProxyConfig,
    _chain_request_hooks,
    create_app,
    frame_redaction_hook,
)

VALUE = "alice@example.com"
TOKEN = "EMAIL_1_a3f9"
CONFIG = ProxyConfig("https://upstream.invalid/v1", "integration-test-key")


async def test_loopback_approval_api_grants_lists_and_revokes_without_raw_value() -> None:
    vault = SessionVault(nonce="a3f9", policy=SafetyPolicy({"API_KEY": "approval"}))
    token = vault.store("API_KEY", "synthetic-secret-key")
    proxy = create_app(CONFIG, frame_store=FrameStore(), vault=vault)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy), base_url="http://proxy.test"
    ) as client:
        body = {"token": token, "tool_name": "write", "argument_path": "content"}
        created = await client.post("/viewer/approvals", json=body)
        listed = await client.get("/viewer/approvals")
        rejected_origin = await client.post(
            "/viewer/approvals", json=body, headers={"origin": "https://attacker.test"}
        )
        revoked = await client.request("DELETE", "/viewer/approvals", json=body)
        after = await client.get("/viewer/approvals")

    assert created.status_code == 201
    assert listed.json()["approvals"][0]["token"] == token
    assert "synthetic-secret-key" not in "".join(
        (created.text, listed.text, rejected_origin.text, after.text)
    )
    assert rejected_origin.status_code == 403
    assert revoked.status_code == 204
    assert after.json() == {"approvals": []}


def source_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (260, 100), "white").save(output, format="PNG")
    return output.getvalue()


def cua_payload(*, history_value: str | None = None) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Synthetic Holo runtime instructions"}
    ]
    if history_value is not None:
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"tool_call": {"tool_name": "write", "content": history_value}}
                ),
            }
        )
    url = "data:image/png;base64," + base64.b64encode(source_png()).decode()
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Use the email chip"},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    )
    return {"model": "synthetic", "messages": messages, "stream": False}


def privacy_stack(
    *, classifier: Any | None = None
) -> tuple[Hooks, SessionVault, FrameStore, VaultRedactor]:
    vault = SessionVault(nonce="a3f9")
    vaulted = VaultRedactor(StubRedactor((StubSpan("EMAIL", VALUE, (20, 20, 240, 65)),)), vault)
    classify = classifier or (lambda texts: [{"sensitive": False, "values": []} for _ in texts])
    scrubber = HistoryScrubber(vault, classify)
    store = FrameStore()
    request = _chain_request_hooks(
        frame_redaction_hook(vaulted, store, include_placeholder_manifest=True),
        privacy_request_hook(scrubber),
    )
    return Hooks(request, privacy_response_hook(vault)), vault, store, vaulted


class SequencedManifestRedactor:
    def __init__(self, manifests: list[tuple[dict[str, str], ...]]) -> None:
        self._manifests = iter(manifests)
        self.latest_analysis: dict[str, Any] = {}

    def __call__(self, png: bytes) -> bytes:
        return png

    def redact_with_manifest(self, png: bytes) -> tuple[bytes, tuple[dict[str, str], ...]]:
        return png, next(self._manifests)


class SequencedFindingRedactor:
    def __init__(self, spans: list[tuple[StubSpan, ...]]) -> None:
        self._spans = iter(spans)
        self._active = StubRedactor()

    @property
    def latest_analysis(self) -> dict[str, Any]:
        return self._active.latest_analysis

    def start(self) -> None:
        return

    def close(self) -> None:
        self._active.close()

    def __call__(self, png: bytes) -> bytes:
        self._active = StubRedactor(next(self._spans))
        return self._active(png)


def test_manifest_tracks_only_latest_frame_and_explicitly_clears_stale_tokens() -> None:
    encoded = base64.b64encode(source_png()).decode()
    redactor = SequencedManifestRedactor([({"token": TOKEN, "class": "EMAIL"},), ()])
    redact = frame_redaction_hook(redactor, include_placeholder_manifest=True)
    scrubber = HistoryScrubber(
        SessionVault(nonce="a3f9"),
        lambda texts: [{"sensitive": False, "values": []} for _ in texts],
    )
    document = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Current observation"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                ],
            },
        ]
    }

    redacted, headers = redact(document, {})
    injected, _ = privacy_request_hook(scrubber)(redacted, headers)
    serialized = json.dumps(injected)

    assert TOKEN not in serialized
    manifest = injected["messages"][-1]["content"][-2]["text"]
    assert manifest.startswith(PLACEHOLDER_MANIFEST_PREFIX)
    assert "visible in the current screenshot: none" in manifest


def upstream_app(seen: list[dict[str, Any]], *, token: str = TOKEN) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completion(request: Request) -> JSONResponse:
        body = await request.body()
        seen.append(json.loads(body))
        action = {
            "note": token,
            "thought": f"Use {token}",
            "tool_call": {"tool_name": "write", "content": token},
        }
        return JSONResponse(
            {"choices": [{"message": {"role": "assistant", "content": json.dumps(action)}}]}
        )

    return app


async def test_full_store_paint_resolve_and_history_scrub_loop(
    caplog: Any,
) -> None:
    seen: list[dict[str, Any]] = []
    hooks, vault, store, vaulted = privacy_stack()
    transport = httpx.ASGITransport(app=upstream_app(seen))
    app = create_app(CONFIG, hooks=hooks, frame_store=store, transport=transport)
    caplog.set_level(logging.INFO)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        first = await client.post("/v1/chat/completions", json=cua_payload())
        second = await client.post("/v1/chat/completions", json=cua_payload(history_value=VALUE))

    assert first.status_code == second.status_code == 200
    assert vault.resolve(TOKEN) == VALUE
    assert VALUE not in json.dumps(seen)
    assert TOKEN in json.dumps(seen)
    assert seen[0]["messages"][0]["role"] == "system"
    manifest_parts = [
        part["text"]
        for part in seen[0]["messages"][-1]["content"]
        if isinstance(part, dict)
        and isinstance(part.get("text"), str)
        and part["text"].startswith("[PLVA_VISIBLE_PLACEHOLDERS]")
    ]
    assert manifest_parts == [
        "[PLVA_VISIBLE_PLACEHOLDERS] Placeholders visible in the current screenshot: "
        "«EMAIL_1_a3f9» (email · hidden, use allowed). "
        "Use only exact tokens listed in this manifest; never invent one."
    ]
    assert TOKEN in seen[1]["messages"][1]["content"]
    result = json.loads(first.json()["choices"][0]["message"]["content"])
    assert result["tool_call"]["content"] == VALUE
    assert result["thought"] == f"Use {TOKEN}"
    assert store.stats()["frames_seen"] == 2
    assert vaulted.latest_analysis["findings"][0]["placeholders"] == [TOKEN]
    assert VALUE not in "\n".join(record.getMessage() for record in caplog.records)


async def test_token_remains_usable_after_private_value_leaves_current_frame() -> None:
    seen: list[dict[str, Any]] = []
    vault = SessionVault(nonce="a3f9")
    detector = SequencedFindingRedactor([(StubSpan("EMAIL", VALUE, (20, 20, 240, 65)),), ()])
    vaulted = VaultRedactor(detector, vault, cache_entries=0)
    scrubber = HistoryScrubber(
        vault,
        lambda texts: [{"sensitive": False, "values": []} for _ in texts],
    )
    store = FrameStore()
    hooks = Hooks(
        _chain_request_hooks(
            frame_redaction_hook(vaulted, store, include_placeholder_manifest=True),
            privacy_request_hook(scrubber),
        ),
        privacy_response_hook(vault),
    )
    proxy = create_app(
        CONFIG,
        hooks=hooks,
        frame_store=store,
        transport=httpx.ASGITransport(app=upstream_app(seen)),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy), base_url="http://proxy.test"
    ) as client:
        first = await client.post("/v1/chat/completions", json=cua_payload())
        second = await client.post("/v1/chat/completions", json=cua_payload())

    second_manifest = next(
        part["text"]
        for part in seen[1]["messages"][-1]["content"]
        if isinstance(part, dict)
        and isinstance(part.get("text"), str)
        and part["text"].startswith(PLACEHOLDER_MANIFEST_PREFIX)
    )
    second_action = json.loads(second.json()["choices"][0]["message"]["content"])
    assert first.status_code == second.status_code == 200
    assert "visible in the current screenshot: none" in second_manifest
    assert f"Active private-session tokens from earlier observations: «{TOKEN}»" in second_manifest
    assert VALUE not in json.dumps(seen)
    assert second_action["tool_call"]["content"] == VALUE


async def test_history_classifier_failure_forwards_nothing() -> None:
    def fail(_: tuple[str, ...]) -> list[dict[str, Any]]:
        raise RuntimeError("synthetic failure")

    seen: list[dict[str, Any]] = []
    hooks, _, store, _ = privacy_stack(classifier=fail)
    app = create_app(
        CONFIG,
        hooks=hooks,
        frame_store=store,
        transport=httpx.ASGITransport(app=upstream_app(seen)),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post("/v1/chat/completions", json=cua_payload())

    assert response.status_code == 502
    assert seen == []


async def test_unknown_response_placeholder_is_not_forwarded() -> None:
    seen: list[dict[str, Any]] = []
    hooks, _, store, _ = privacy_stack()
    app = create_app(
        CONFIG,
        hooks=hooks,
        frame_store=store,
        transport=httpx.ASGITransport(app=upstream_app(seen, token="EMAIL_99_a3f9")),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post("/v1/chat/completions", json=cua_payload())

    assert response.status_code == 502
    assert len(seen) == 1


async def test_streamed_action_is_buffered_resolved_and_reemitted() -> None:
    seen: list[dict[str, Any]] = []
    action = json.dumps(
        {
            "note": TOKEN,
            "thought": f"Use {TOKEN}",
            "tool_call": {"tool_name": "write", "content": TOKEN},
        },
        separators=(",", ":"),
    )
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completion(request: Request) -> StreamingResponse:
        seen.append(await request.json())
        events = (
            b'data: {"id":"s1","choices":[{"delta":{"role":"assistant"}}]}\n\n',
            (
                "data: "
                + json.dumps({"id": "s1", "choices": [{"delta": {"content": action}}]})
                + "\n\n"
            ).encode(),
            b'data: {"id":"s1","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        )
        return StreamingResponse(iter(events), media_type="text/event-stream")

    hooks, _, store, _ = privacy_stack()
    proxy = create_app(
        CONFIG,
        hooks=hooks,
        frame_store=store,
        transport=httpx.ASGITransport(app=app),
    )
    payload = cua_payload()
    payload["stream"] = True

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy), base_url="http://proxy.test"
    ) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    content = ""
    for line in response.content.splitlines():
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        event = json.loads(line.removeprefix(b"data: "))
        content += event["choices"][0]["delta"].get("content", "")
    resolved = json.loads(content)
    assert response.status_code == 200
    assert resolved["tool_call"]["content"] == VALUE
    assert resolved["thought"] == f"Use {TOKEN}"
    assert len(seen) == 1
