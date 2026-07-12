"""Loopback pass-through proxy between the Holo runtime and Overshoot.

Step 1 scope: a transparent relay that is the runtime's only model endpoint.
It binds to loopback, injects the provider credential from the environment,
forwards request bodies verbatim (unknown keys included, per the Step 0
contract findings), and relays JSON or SSE responses. On any upstream failure
it fails closed: nothing is fabricated, streams are truncated rather than
completed. Logs carry only privacy-safe metadata — byte counts, statuses,
durations, exception class names — never bodies, frames, or key material.
Steps 3-4 extend this seam with mutation hooks, redaction, and placeholder
resolution.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from plva_proxy.contract_probe import API_BASE_URL
from plva_proxy.runtime_capture import LOOPBACK_HOST

DEFAULT_PORT: Final = 18081
_FORWARDED_REQUEST_HEADERS: Final = frozenset({"accept", "content-type"})
_UPSTREAM_TIMEOUT: Final = httpx.Timeout(10.0, read=300.0, write=60.0, pool=10.0)

_LOGGER: Final = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProxyConfig:
    """Static proxy settings; the key never appears in logs or responses."""

    upstream_base_url: str
    api_key: str


def _upstream_headers(request: Request, api_key: str) -> dict[str, str]:
    """Build upstream headers from an allowlist; inbound auth is never forwarded."""

    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() in _FORWARDED_REQUEST_HEADERS
    }
    headers["authorization"] = f"Bearer {api_key}"
    return headers


async def _relay_stream(upstream: httpx.Response, started: float) -> AsyncIterator[bytes]:
    """Relay SSE bytes as they arrive; truncate (never fabricate) on failure."""

    relayed = 0
    try:
        async for chunk in upstream.aiter_raw():
            relayed += len(chunk)
            yield chunk
    except httpx.HTTPError as exc:
        _LOGGER.warning("upstream stream aborted: %s", type(exc).__name__)
    finally:
        await upstream.aclose()
        _LOGGER.info(
            "relay stream done status=%d response_bytes=%d duration_ms=%d",
            upstream.status_code,
            relayed,
            int((time.monotonic() - started) * 1000),
        )


def create_app(
    config: ProxyConfig, *, transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    """Create the loopback relay application around one upstream client."""

    client = httpx.AsyncClient(
        base_url=config.upstream_base_url,
        timeout=_UPSTREAM_TIMEOUT,
        transport=transport,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(title="PLVA pass-through proxy", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.upstream_client = client

    async def _relay(request: Request, method: str, path: str) -> Response:
        started = time.monotonic()
        body = await request.body()
        upstream_request = client.build_request(
            method,
            path,
            content=body or None,
            headers=_upstream_headers(request, config.api_key),
        )
        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            _LOGGER.warning("upstream request failed: %s", type(exc).__name__)
            raise HTTPException(status_code=502, detail="upstream request failed") from exc

        content_type = upstream.headers.get("content-type", "application/octet-stream")
        if content_type.lower().startswith("text/event-stream"):
            return StreamingResponse(
                _relay_stream(upstream, started),
                status_code=upstream.status_code,
                media_type=content_type,
            )
        try:
            payload = await upstream.aread()
        except httpx.HTTPError as exc:
            _LOGGER.warning("upstream read failed: %s", type(exc).__name__)
            raise HTTPException(status_code=502, detail="upstream response failed") from exc
        finally:
            await upstream.aclose()
        _LOGGER.info(
            "relay %s status=%d request_bytes=%d response_bytes=%d duration_ms=%d",
            path,
            upstream.status_code,
            len(body),
            len(payload),
            int((time.monotonic() - started) * 1000),
        )
        return Response(content=payload, status_code=upstream.status_code, media_type=content_type)

    @app.get("/health")
    @app.get("/v1/health")
    async def health() -> dict[str, str]:
        # The closed runtime health-checks <base-host>/health before POSTing;
        # answer locally so a slow provider cannot block the loop.
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        return await _relay(request, "GET", "/models")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _relay(request, "POST", "/chat/completions")

    return app


def _env_file_value(path: Path, key: str) -> str | None:
    """Read ``KEY=value`` from a dotenv-style file without echoing its contents."""

    try:
        lines = path.read_text("utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith(f"{key}="):
            continue
        value = stripped.removeprefix(f"{key}=").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value or None
    return None


def main() -> None:
    """Run the pass-through proxy on a fixed loopback interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--upstream", default=API_BASE_URL, help="provider base URL")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not args.upstream.startswith(("http://", "https://")):
        parser.error("--upstream must be an http(s) URL")
    api_key = os.environ.get("API_KEY") or _env_file_value(Path(".env"), "API_KEY")
    if not api_key:
        parser.error("API_KEY is required (export it, or fill .env next to pyproject.toml)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    uvicorn.run(
        create_app(ProxyConfig(upstream_base_url=args.upstream, api_key=api_key)),
        host=LOOPBACK_HOST,
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
