"""Loopback interception proxy between the Holo runtime and Overshoot.

Step 1 gave this proxy its pass-through role: the runtime's only model
endpoint, loopback-bound, injecting the provider credential and relaying
bodies verbatim (unknown keys included, per the Step 0 contract findings).
Step 3 adds the interception seam: optional hooks may mutate the outbound
request (body + upstream headers) and the inbound completion, for JSON and
SSE responses alike. A streamed response under a response hook is buffered,
reconstructed, mutated, and re-emitted so nothing unresolved is ever
forwarded (§8.7); any hook or parse failure forwards nothing at all (§8.1).
Logs carry only privacy-safe metadata — byte counts, statuses, durations,
exception class names — never bodies, frames, or key material. Step 4 plugs
redaction and placeholder resolution into these hooks.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from PIL import Image

from plva_proxy.contract_probe import API_BASE_URL
from plva_proxy.runtime_capture import LOOPBACK_HOST

DEFAULT_PORT: Final = 18081
_FORWARDED_REQUEST_HEADERS: Final = frozenset({"accept", "content-type"})
_UPSTREAM_TIMEOUT: Final = httpx.Timeout(10.0, read=300.0, write=60.0, pool=10.0)

_LOGGER: Final = logging.getLogger(__name__)

RequestHook = Callable[[dict[str, Any], dict[str, str]], tuple[dict[str, Any], dict[str, str]]]
ResponseHook = Callable[[dict[str, Any]], dict[str, Any]]


class HookError(RuntimeError):
    """Raised when traffic cannot be safely parsed or mutated; fails closed."""


@dataclass(frozen=True, slots=True)
class ProxyConfig:
    """Static proxy settings; the key never appears in logs or responses."""

    upstream_base_url: str
    api_key: str


@dataclass(frozen=True, slots=True)
class Hooks:
    """Mutation seam for both traffic directions; a None hook is pass-through."""

    on_request: RequestHook | None = None
    on_response: ResponseHook | None = None


def _tag_request(
    document: dict[str, Any], headers: dict[str, str]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Step 3 test hook: observably tag the upstream request."""

    return document, {**headers, "x-plva-hook": "request"}


def _noop_rewrite_actions(document: dict[str, Any]) -> dict[str, Any]:
    """Step 3 test hook: decode and re-encode each action payload unchanged.

    Exercises the exact parse → mutate → re-serialize path that Step 4 will
    use for placeholder resolution. Unparseable action content fails closed.
    """

    choices = document.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HookError("completion has no choices to rewrite")
    rewritten: dict[str, Any] = json.loads(json.dumps(document))
    for choice in rewritten["choices"]:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            action = json.loads(content)
        except ValueError as exc:
            raise HookError("action content is not JSON") from exc
        message["content"] = json.dumps(action, separators=(",", ":"))
    return rewritten


TEST_HOOKS: Final = Hooks(on_request=_tag_request, on_response=_noop_rewrite_actions)

_IMAGE_MEDIA_TYPES: Final = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def image_replacement_hook(image_path: Path) -> RequestHook:
    """Build a request hook replacing every outbound screenshot with one static image.

    The replacement file is read and validated once, at startup. If a hooked
    request contains no replaceable screenshot, the hook raises so a request
    that was meant to be scrubbed can never leave with its original frame
    (§8.1/§8.2 rehearsal for Step 4 redaction).
    """

    data = image_path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        media_type = _IMAGE_MEDIA_TYPES.get(image.format or "")
        image.verify()
    if media_type is None:
        allowed = ", ".join(sorted(_IMAGE_MEDIA_TYPES.values()))
        raise ValueError(f"replacement image must be one of: {allowed}")
    data_url = f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"

    def replace(
        document: dict[str, Any], headers: dict[str, str]
    ) -> tuple[dict[str, Any], dict[str, str]]:
        rewritten: dict[str, Any] = json.loads(json.dumps(document))
        replaced = 0
        for message in rewritten.get("messages") or []:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    part["image_url"] = {"url": data_url}
                    replaced += 1
        if replaced == 0:
            raise HookError("no screenshot found to replace")
        _LOGGER.info("image hook replaced %d screenshot(s)", replaced)
        return rewritten, headers

    return replace


def _chain_request_hooks(
    first: RequestHook | None, second: RequestHook | None
) -> RequestHook | None:
    """Compose two optional request hooks, applying them in order."""

    if first is None:
        return second
    if second is None:
        return first

    def chained(
        document: dict[str, Any], headers: dict[str, str]
    ) -> tuple[dict[str, Any], dict[str, str]]:
        document, headers = first(document, headers)
        return second(document, headers)

    return chained


def _upstream_headers(request: Request, api_key: str) -> dict[str, str]:
    """Build upstream headers from an allowlist; inbound auth is never forwarded."""

    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() in _FORWARDED_REQUEST_HEADERS
    }
    headers["authorization"] = f"Bearer {api_key}"
    return headers


def _assemble_sse_completion(raw: bytes) -> dict[str, Any]:
    """Reconstruct one completion document from a fully buffered SSE stream.

    Only complete streams (terminal ``[DONE]`` seen) are accepted; a truncated
    or exotic stream raises so it is never re-emitted to the executor (§8.7).
    """

    envelope: dict[str, Any] | None = None
    role = "assistant"
    parts: list[str] = []
    finish_reason: str | None = None
    done = False

    for event in raw.replace(b"\r\n", b"\n").split(b"\n\n"):
        data_lines = [line[5:].lstrip() for line in event.splitlines() if line.startswith(b"data:")]
        if not data_lines:
            continue
        payload = b"\n".join(data_lines)
        if payload == b"[DONE]":
            done = True
            continue
        try:
            document = json.loads(payload)
        except (ValueError, UnicodeDecodeError) as exc:
            raise HookError("invalid SSE JSON data event") from exc
        if not isinstance(document, dict):
            raise HookError("SSE data event is not an object")
        if envelope is None:
            envelope = {key: document.get(key) for key in ("id", "created", "model")}
        for choice in document.get("choices") or []:
            if not isinstance(choice, dict):
                raise HookError("SSE choice is not an object")
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            if "tool_calls" in delta:
                raise HookError("native tool_call deltas are not supported by the hook seam")
            if isinstance(delta.get("role"), str):
                role = delta["role"]
            if isinstance(delta.get("content"), str):
                parts.append(delta["content"])
            if isinstance(choice.get("finish_reason"), str):
                finish_reason = choice["finish_reason"]

    if envelope is None or not done:
        raise HookError("SSE stream ended without a complete completion")
    return {
        **envelope,
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": role, "content": "".join(parts)},
                "finish_reason": finish_reason or "stop",
            }
        ],
    }


def _sse_bytes(document: dict[str, Any]) -> Iterator[bytes]:
    """Re-emit a (possibly mutated) completion as a minimal SSE stream."""

    common = {
        "id": document.get("id"),
        "object": "chat.completion.chunk",
        "created": document.get("created"),
        "model": document.get("model"),
    }
    choice = document["choices"][0]
    message = choice["message"]
    deltas: tuple[tuple[dict[str, Any], str | None], ...] = (
        ({"role": message["role"]}, None),
        ({"content": message["content"]}, None),
        ({}, choice.get("finish_reason") or "stop"),
    )
    for delta, finish_reason in deltas:
        event = {
            **common,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode()
    yield b"data: [DONE]\n\n"


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
    config: ProxyConfig,
    *,
    hooks: Hooks | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Create the loopback relay application around one upstream client."""

    active_hooks = hooks if hooks is not None else Hooks()
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

    app = FastAPI(title="PLVA interception proxy", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.upstream_client = client

    async def _relay(
        request: Request, method: str, path: str, *, use_hooks: bool = False
    ) -> Response:
        started = time.monotonic()
        body = await request.body()
        headers = _upstream_headers(request, config.api_key)

        request_hook = active_hooks.on_request if use_hooks else None
        if request_hook is not None:
            try:
                document = json.loads(body)
                if not isinstance(document, dict):
                    raise HookError("request body is not a JSON object")
                document, headers = request_hook(document, headers)
                body = json.dumps(document, separators=(",", ":")).encode()
            except (HookError, ValueError) as exc:
                _LOGGER.warning("request hook failed closed: %s", type(exc).__name__)
                raise HTTPException(status_code=502, detail="request hook failed") from exc

        upstream_request = client.build_request(method, path, content=body or None, headers=headers)
        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            _LOGGER.warning("upstream request failed: %s", type(exc).__name__)
            raise HTTPException(status_code=502, detail="upstream request failed") from exc

        content_type = upstream.headers.get("content-type", "application/octet-stream")
        is_sse = content_type.lower().startswith("text/event-stream")
        response_hook = active_hooks.on_response if use_hooks else None
        hook_applies = response_hook is not None and upstream.status_code == 200

        if is_sse and not hook_applies:
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

        if response_hook is not None and upstream.status_code == 200:
            try:
                document = _assemble_sse_completion(payload) if is_sse else json.loads(payload)
                if not isinstance(document, dict):
                    raise HookError("completion body is not a JSON object")
                mutated = response_hook(document)
            except (HookError, ValueError) as exc:
                _LOGGER.warning("response hook failed closed: %s", type(exc).__name__)
                raise HTTPException(status_code=502, detail="response hook failed") from exc
            _LOGGER.info(
                "relay %s status=200 request_bytes=%d response_bytes=%d duration_ms=%d hooks=on",
                path,
                len(body),
                len(payload),
                int((time.monotonic() - started) * 1000),
            )
            hook_header = {"x-plva-hook": "response"}
            if is_sse:
                return StreamingResponse(
                    _sse_bytes(mutated), media_type="text/event-stream", headers=hook_header
                )
            return Response(
                content=json.dumps(mutated, separators=(",", ":")).encode(),
                status_code=200,
                media_type="application/json",
                headers=hook_header,
            )

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
        return await _relay(request, "POST", "/chat/completions", use_hooks=True)

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
    """Run the interception proxy on a fixed loopback interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--upstream", default=API_BASE_URL, help="provider base URL")
    parser.add_argument(
        "--hook",
        choices=("none", "test"),
        default="none",
        help="traffic mutation hooks: none = pass-through, test = Step 3 test hooks",
    )
    parser.add_argument(
        "--hook-image",
        type=Path,
        default=None,
        help="replace every outbound screenshot with this static PNG/JPEG/WebP "
        "(fails closed if a request has no screenshot)",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not args.upstream.startswith(("http://", "https://")):
        parser.error("--upstream must be an http(s) URL")
    api_key = os.environ.get("API_KEY") or _env_file_value(Path(".env"), "API_KEY")
    if not api_key:
        parser.error("API_KEY is required (export it, or fill .env next to pyproject.toml)")

    image_hook: RequestHook | None = None
    if args.hook_image is not None:
        try:
            image_hook = image_replacement_hook(args.hook_image)
        except (OSError, ValueError) as exc:
            parser.error(f"--hook-image is unusable: {exc}")
    hooks = TEST_HOOKS if args.hook == "test" else None
    if image_hook is not None:
        prior = hooks if hooks is not None else Hooks()
        hooks = Hooks(
            on_request=_chain_request_hooks(prior.on_request, image_hook),
            on_response=prior.on_response,
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    uvicorn.run(
        create_app(
            ProxyConfig(upstream_base_url=args.upstream, api_key=api_key),
            hooks=hooks,
        ),
        host=LOOPBACK_HOST,
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
