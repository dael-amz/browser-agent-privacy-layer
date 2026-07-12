from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from PIL import Image

import plva_proxy.proxy as proxy
from plva_proxy.proxy import FrameStore, Hooks, ProxyConfig, create_app, frame_redaction_hook
from plva_proxy.redactor import RedactionError, RedactorConfig, redact_png

UPSTREAM_KEY = "unit-test-redaction-upstream-key"
CONFIG = ProxyConfig(upstream_base_url="https://upstream.invalid/v1beta", api_key=UPSTREAM_KEY)

FAKE_CLI_OK = """\
import json
import shutil
import sys

args = sys.argv[1:]
source = args[0]
output = args[args.index("--output") + 1]
report = args[args.index("--report") + 1]
shutil.copyfile(source, output)
with open(report, "w", encoding="utf-8") as fh:
    json.dump({"counts": {"fused": 2}}, fh)
"""

FAKE_CLI_EXIT_1 = """\
import sys

sys.exit(1)
"""

FAKE_CLI_NO_OUTPUT = """\
# Exits 0 without writing the output or report files.
"""


def make_image_bytes(fmt: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(0, 128, 255)).save(buffer, format=fmt)
    return buffer.getvalue()


def image_chat_payload(url: str) -> dict[str, Any]:
    return {
        "model": "Hcompany/Holo3-35B-A3B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "click the button"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
        "stream": False,
    }


def make_recording_json_upstream() -> tuple[FastAPI, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    app = FastAPI()

    @app.post("/v1beta/chat/completions")
    async def chat(request: Request) -> Response:
        body = await request.body()
        seen.append({"path": request.url.path, "headers": dict(request.headers), "body": body})
        return JSONResponse({"id": "upstream-json"})

    return app, seen


def make_proxy_client(
    transport: httpx.AsyncBaseTransport,
    *,
    hooks: Hooks | None = None,
    frame_store: FrameStore | None = None,
) -> httpx.AsyncClient:
    app = create_app(CONFIG, hooks=hooks, frame_store=frame_store, transport=transport)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")


def unused_upstream() -> httpx.MockTransport:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no upstream traffic expected")

    return httpx.MockTransport(refuse)


# --- frame_redaction_hook through the relay ---


async def test_redaction_hook_converts_jpeg_to_png_and_swaps_in_redacted_frame() -> None:
    jpeg_bytes = make_image_bytes("JPEG")
    redacted_png = make_image_bytes("PNG")
    received: list[bytes] = []

    def fake_redact(png: bytes) -> bytes:
        received.append(png)
        return redacted_png

    store = FrameStore()
    upstream_app, seen = make_recording_json_upstream()
    hooks = Hooks(on_request=frame_redaction_hook(fake_redact, store))
    jpeg_url = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")
    payload = image_chat_payload(jpeg_url)

    async with make_proxy_client(httpx.ASGITransport(app=upstream_app), hooks=hooks) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert len(received) == 1
    with Image.open(io.BytesIO(received[0])) as image:
        assert image.format == "PNG"

    assert len(seen) == 1
    forwarded = json.loads(seen[0]["body"])
    parts = forwarded["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "click the button"}
    expected_url = "data:image/png;base64," + base64.b64encode(redacted_png).decode("ascii")
    assert parts[1] == {"type": "image_url", "image_url": {"url": expected_url}}
    assert store.stats().items() >= {"frames_seen": 1, "buffered": 1}.items()


async def test_redaction_hook_passes_imageless_requests_through_unchanged() -> None:
    def fake_redact(png: bytes) -> bytes:
        raise AssertionError("redact must not be called without images")

    upstream_app, seen = make_recording_json_upstream()
    hooks = Hooks(on_request=frame_redaction_hook(fake_redact, FrameStore()))
    payload = {
        "model": "Hcompany/Holo3-35B-A3B",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }

    async with make_proxy_client(httpx.ASGITransport(app=upstream_app), hooks=hooks) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert len(seen) == 1
    assert json.loads(seen[0]["body"]) == payload


async def test_redaction_hook_fails_closed_when_redact_raises() -> None:
    def fake_redact(png: bytes) -> bytes:
        raise RuntimeError("model blew up")

    upstream_app, seen = make_recording_json_upstream()
    hooks = Hooks(on_request=frame_redaction_hook(fake_redact, FrameStore()))
    png_url = "data:image/png;base64," + base64.b64encode(make_image_bytes("PNG")).decode("ascii")

    async with make_proxy_client(httpx.ASGITransport(app=upstream_app), hooks=hooks) as client:
        response = await client.post("/v1/chat/completions", json=image_chat_payload(png_url))

    assert response.status_code == 502
    assert response.json() == {"detail": "request hook failed"}
    assert seen == []


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://screenshots.invalid/frame.png",
        "data:image/png;base64,@@not-base64@@",
    ],
)
async def test_redaction_hook_fails_closed_on_invalid_data_urls(bad_url: str) -> None:
    def fake_redact(png: bytes) -> bytes:
        raise AssertionError("redact must not be reached for invalid URLs")

    upstream_app, seen = make_recording_json_upstream()
    hooks = Hooks(on_request=frame_redaction_hook(fake_redact, FrameStore()))

    async with make_proxy_client(httpx.ASGITransport(app=upstream_app), hooks=hooks) as client:
        response = await client.post("/v1/chat/completions", json=image_chat_payload(bad_url))

    assert response.status_code == 502
    assert response.json() == {"detail": "request hook failed"}
    assert seen == []


# --- FrameStore ring buffer ---


def test_frame_store_evicts_oldest_beyond_capacity_and_tracks_totals() -> None:
    store = FrameStore(capacity=3)
    assert store.latest() is None
    assert store.stats().items() >= {"frames_seen": 0, "buffered": 0}.items()

    for index in range(4):
        store.add(f"frame-{index}".encode())

    assert store.stats().items() >= {"frames_seen": 4, "buffered": 3}.items()
    assert store.latest() == b"frame-3"


# --- viewer endpoints ---


async def test_viewer_endpoints_serve_html_latest_frame_and_stats() -> None:
    store = FrameStore()

    async with make_proxy_client(unused_upstream(), frame_store=store) as client:
        page = await client.get("/viewer")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert "PLVA" in page.text

        empty = await client.get("/viewer/frame")
        assert empty.status_code == 404

        stats_before = await client.get("/viewer/stats")
        assert stats_before.json().items() >= {"frames_seen": 0, "buffered": 0}.items()

        store.add(b"redacted-png-bytes")
        frame = await client.get("/viewer/frame")
        assert frame.status_code == 200
        assert frame.headers["content-type"] == "image/png"
        assert frame.headers["cache-control"] == "no-store"
        assert frame.content == b"redacted-png-bytes"

        stats_after = await client.get("/viewer/stats")
        assert stats_after.json().items() >= {"frames_seen": 1, "buffered": 1}.items()


async def test_viewer_is_absent_without_a_frame_store() -> None:
    async with make_proxy_client(unused_upstream()) as client:
        response = await client.get("/viewer")

    assert response.status_code == 404


# --- redact_png against a fake CLI (no node, no real pipeline) ---


def write_fake_cli(tmp_path: Path, body: str) -> Path:
    # Nested two levels deep: redact_png sets cwd=cli_path.parent.parent.
    cli_dir = tmp_path / "plva-v2-baseline" / "bin"
    cli_dir.mkdir(parents=True, exist_ok=True)
    cli_path = cli_dir / "fake_cli.py"
    cli_path.write_text(body, "utf-8")
    return cli_path


def fake_cli_config(cli_path: Path) -> RedactorConfig:
    return RedactorConfig(cli_path=cli_path, node_path=sys.executable, timeout_s=30.0)


def test_redact_png_returns_cli_output_bytes(tmp_path: Path) -> None:
    cli_path = write_fake_cli(tmp_path, FAKE_CLI_OK)

    result = redact_png(fake_cli_config(cli_path), b"fake-png-input")

    assert result == b"fake-png-input"


def test_redact_png_raises_on_nonzero_exit(tmp_path: Path) -> None:
    cli_path = write_fake_cli(tmp_path, FAKE_CLI_EXIT_1)

    with pytest.raises(RedactionError, match="redactor exited 1"):
        redact_png(fake_cli_config(cli_path), b"fake-png-input")


def test_redact_png_raises_when_cli_writes_no_output(tmp_path: Path) -> None:
    cli_path = write_fake_cli(tmp_path, FAKE_CLI_NO_OUTPUT)

    with pytest.raises(RedactionError, match="no readable output"):
        redact_png(fake_cli_config(cli_path), b"fake-png-input")


def test_redact_png_raises_when_interpreter_is_missing(tmp_path: Path) -> None:
    cli_path = write_fake_cli(tmp_path, FAKE_CLI_OK)
    config = RedactorConfig(cli_path=cli_path, node_path="/nonexistent-binary", timeout_s=30.0)

    with pytest.raises(RedactionError, match="did not run"):
        redact_png(config, b"fake-png-input")


# --- main() wiring ---


def test_main_rejects_redact_directory_without_bundled_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("API_KEY", "unit-test-redaction-key")
    monkeypatch.setattr(sys, "argv", ["plva-proxy", "--redact", str(tmp_path)])

    with pytest.raises(SystemExit):
        proxy.main()
