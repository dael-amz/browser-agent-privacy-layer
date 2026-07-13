from __future__ import annotations

import base64
import json
import sys
from typing import Any

import httpx
import pytest

import plva_proxy.runtime_capture as runtime_capture
from plva_proxy.contract_probe import MODEL_ID
from plva_proxy.runtime_capture import CaptureError, CaptureState, create_app, summarize_request

SYNTHETIC_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
SENSITIVE_MARKER = "runtime-request-sensitive-marker"


def runtime_payload(*, stream: bool = False) -> dict[str, Any]:
    encoded = base64.b64encode(SYNTHETIC_PNG).decode("ascii")
    return {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SENSITIVE_MARKER},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SENSITIVE_MARKER},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            },
        ],
        "structured_outputs": {"json": {"type": "object"}},
        "stream": stream,
    }


def test_summarize_request_records_schema_and_image_metadata_only() -> None:
    payload = runtime_payload()

    summary = summarize_request(payload)

    assert summary.model == MODEL_ID
    assert summary.stream is False
    assert summary.message_roles == ("system", "user")
    assert summary.image_count == 1
    assert summary.image_media_types == ("image/png",)
    assert summary.image_byte_lengths == (len(SYNTHETIC_PNG),)
    assert summary.image_dimensions == ((1, 1),)
    assert summary.has_structured_outputs is True
    assert summary.has_tools is False
    assert SENSITIVE_MARKER not in repr(summary)
    assert base64.b64encode(SYNTHETIC_PNG).decode("ascii") not in repr(summary)


@pytest.mark.parametrize(
    "payload",
    [
        {"model": MODEL_ID, "messages": [], "stream": False},
        {"model": "wrong-model", "messages": [], "stream": False},
        {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,not-valid-base64"},
                        }
                    ],
                }
            ],
            "stream": False,
        },
    ],
)
def test_summarize_request_rejects_an_unusable_runtime_request(payload: dict[str, Any]) -> None:
    with pytest.raises(CaptureError):
        summarize_request(payload)


async def test_capture_app_returns_non_executable_answer_and_metadata_status() -> None:
    state = CaptureState()
    transport = httpx.ASGITransport(app=create_app(state))
    async with httpx.AsyncClient(transport=transport, base_url="http://probe.test") as client:
        response = await client.post("/v1/chat/completions", json=runtime_payload())
        status = await client.get("/_probe/status")

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    action = json.loads(message["content"])
    assert action == {
        "note": None,
        "thought": "The local transport contract is verified.",
        "tool_calls": [
            {
                "tool_name": "answer",
                "content": "PLVA local transport probe complete.",
            }
        ],
    }
    assert message["tool_calls"] is None
    metadata = status.json()
    assert metadata["captured"] is True
    assert metadata["capture_count"] == 1
    assert metadata["summary"]["image_dimensions"] == [[1, 1]]
    assert SENSITIVE_MARKER not in status.text


async def test_capture_app_supports_sse_without_echoing_request_content() -> None:
    state = CaptureState()
    transport = httpx.ASGITransport(app=create_app(state))
    async with httpx.AsyncClient(transport=transport, base_url="http://probe.test") as client:
        response = await client.post("/v1/chat/completions", json=runtime_payload(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in response.text
    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    content = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
    assert json.loads(content)["tool_calls"][0]["tool_name"] == "answer"
    assert SENSITIVE_MARKER not in response.text
    assert state.snapshot()["capture_count"] == 1


async def test_capture_app_rejects_missing_image_without_recording_or_echoing() -> None:
    state = CaptureState()
    transport = httpx.ASGITransport(app=create_app(state))
    payload = runtime_payload()
    payload["messages"] = [{"role": "user", "content": SENSITIVE_MARKER}]
    async with httpx.AsyncClient(transport=transport, base_url="http://probe.test") as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 422
    assert state.snapshot() == {"captured": False, "capture_count": 0, "summary": None}
    assert SENSITIVE_MARKER not in response.text


@pytest.mark.parametrize("path", ["/health", "/v1/health"])
async def test_capture_app_health_returns_ok(path: str) -> None:
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://probe.test") as client:
        response = await client.get(path)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_capture_app_advertises_only_the_selected_model() -> None:
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://probe.test") as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "status": "ready"}],
    }


def test_main_binds_only_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(sys, "argv", ["plva-runtime-capture", "--port", "18081"])
    monkeypatch.setattr(
        "plva_proxy.runtime_capture.uvicorn.run", lambda app, **kwargs: calls.append(kwargs)
    )

    runtime_capture.main()

    assert calls == [
        {
            "host": "127.0.0.1",
            "port": 18081,
            "access_log": False,
            "log_level": "warning",
        }
    ]
