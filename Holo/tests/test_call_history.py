"""Model-call history: proxy CallStore, viewer routes, and demo mirroring."""

from __future__ import annotations

import base64
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

import plva_proxy.demo as demo
from plva_proxy.proxy import CallStore, FrameStore, Hooks, ProxyConfig, create_app

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
CONFIG = ProxyConfig(upstream_base_url="https://upstream.invalid/v1beta", api_key="unit-key")


def call_request() -> dict[str, Any]:
    return {
        "model": "Hcompany/Holo3-35B-A3B",
        "messages": [
            {"role": "system", "content": "placeholder scheme prompt"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "click EMAIL_1_ab12 in the form"},
                    {"type": "image_url", "image_url": {"url": DATA_URL}},
                ],
            },
        ],
        "stream": False,
    }


def test_call_store_strips_images_and_serves_summaries() -> None:
    store = CallStore(capacity=2)
    completion = {"choices": [{"message": {"role": "assistant", "content": "{}"}}]}
    call_id = store.record(
        call_request(), status=200, response=completion, duration_ms=42, state="sent"
    )

    summary = store.entries()[0]
    assert call_id == 1
    assert summary["preview"] == "click EMAIL_1_ab12 in the form"
    assert summary["images"] == ["image/png"] and summary["messages"] == 2
    assert summary["status"] == 200 and summary["state"] == "sent"

    full = store.full(call_id)
    assert full is not None
    image_part = full["request"]["messages"][1]["content"][1]
    assert image_part["image_url"] == {"url": "plva:image/0"}
    assert DATA_URL not in str(full)
    assert full["response"] == completion
    assert store.image(call_id, 0) == ("image/png", PNG_BYTES)
    assert store.image(call_id, 1) is None and store.full(99) is None

    # Ring behavior: oldest record falls out at capacity.
    store.record(call_request(), status=200, response=None, duration_ms=1, state="sent")
    store.record(call_request(), status=None, response=None, duration_ms=1, state="failed")
    assert [entry["id"] for entry in store.entries()] == [2, 3]
    assert store.full(1) is None


def test_call_store_rejects_bad_capacity_and_tolerates_odd_shapes() -> None:
    with pytest.raises(ValueError):
        CallStore(capacity=0)
    store = CallStore()
    call_id = store.record(
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://cdn.example/x.png"}}]}]},
        status=502,
        response=None,
        duration_ms=0,
        state="failed",
    )
    full = store.full(call_id)
    assert full is not None
    part = full["request"]["messages"][0]["content"][0]
    assert part["image_url"] == {"url": "(external image)"}
    assert full["images"] == [] and full["preview"] == ""


async def test_relay_records_call_and_viewer_serves_it() -> None:
    upstream = FastAPI()

    @upstream.post("/v1beta/chat/completions")
    async def chat(request: Request) -> Response:
        await request.body()
        return JSONResponse(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    call_store = CallStore()
    app = create_app(
        CONFIG,
        hooks=Hooks(on_request=lambda document, headers: (document, headers)),
        frame_store=FrameStore(),
        call_store=call_store,
        transport=httpx.ASGITransport(app=upstream),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        relayed = await client.post("/v1/chat/completions", json=call_request())
        index = await client.get("/viewer/calls")
        full = await client.get("/viewer/call/1")
        image = await client.get("/viewer/call/1/image/0")
        missing = await client.get("/viewer/call/9")

    assert relayed.status_code == 200
    entries = index.json()["calls"]
    assert len(entries) == 1 and entries[0]["state"] == "sent" and entries[0]["status"] == 200
    record = full.json()
    assert record["response"]["choices"][0]["message"]["content"] == "done"
    assert record["request"]["messages"][1]["content"][1]["image_url"] == {"url": "plva:image/0"}
    assert image.status_code == 200 and image.content == PNG_BYTES
    assert image.headers["content-type"] == "image/png"
    assert missing.status_code == 404


async def test_relay_records_failed_upstream_call() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    call_store = CallStore()
    app = create_app(
        CONFIG,
        hooks=Hooks(on_request=lambda document, headers: (document, headers)),
        frame_store=FrameStore(),
        call_store=call_store,
        transport=httpx.MockTransport(refuse),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        relayed = await client.post("/v1/chat/completions", json=call_request())

    assert relayed.status_code == 502
    (entry,) = call_store.entries()
    assert entry["state"] == "failed" and entry["status"] is None


def seeded_controller() -> demo.DemoController:
    controller = demo.DemoController()
    record = CallStore()
    call_id = record.record(
        call_request(),
        status=200,
        response={"choices": [{"message": {"role": "assistant", "content": "done"}}]},
        duration_ms=7,
        state="sent",
    )
    controller._calls[call_id] = record.full(call_id)
    controller._call_images[(call_id, 0)] = ("image/png", PNG_BYTES)
    return controller


async def test_demo_serves_mirrored_call_history() -> None:
    app = demo.create_demo_app(seeded_controller())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://demo.test"
    ) as client:
        calls = await client.get("/api/calls")
        call = await client.get("/api/call/1")
        image = await client.get("/api/call/1/image/0")
        missing_call = await client.get("/api/call/2")
        missing_image = await client.get("/api/call/1/image/5")

    (summary,) = calls.json()["calls"]
    assert summary["id"] == 1 and "request" not in summary and "response" not in summary
    record = call.json()
    assert record["response"]["choices"][0]["message"]["content"] == "done"
    assert image.content == PNG_BYTES and image.headers["content-type"] == "image/png"
    assert missing_call.status_code == 404 and missing_image.status_code == 404


def test_mirror_calls_copies_new_records_once_images_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = demo.DemoController()
    store = CallStore()
    store.record(
        call_request(),
        status=200,
        response={"choices": [{"message": {"role": "assistant", "content": "done"}}]},
        duration_ms=7,
        state="sent",
    )

    def fake_json(path: str) -> dict[str, Any] | None:
        if path == "/viewer/calls":
            return {"calls": store.entries()}
        if path == "/viewer/call/1":
            return store.full(1)
        return None

    blob_available = {"ready": False}

    def fake_bytes(path: str) -> bytes | None:
        if path == "/viewer/call/1/image/0" and blob_available["ready"]:
            return PNG_BYTES
        return None

    monkeypatch.setattr(demo, "_fetch_json", fake_json)
    monkeypatch.setattr(demo, "_fetch_bytes", fake_bytes)

    controller._mirror_calls()  # Image missing: record must wait.
    assert controller.calls() == []

    blob_available["ready"] = True
    controller._mirror_calls()
    (summary,) = controller.calls()
    assert summary["id"] == 1 and summary["images"] == ["image/png"]
    assert controller.call_image(1, 0) == ("image/png", PNG_BYTES)

    # A second pass must not duplicate or refetch the mirrored record.
    controller._mirror_calls()
    assert len(controller.calls()) == 1
