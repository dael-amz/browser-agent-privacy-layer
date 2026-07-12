"""Loopback interception proxy between the Holo runtime and its selected provider.

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
import binascii
import copy
import functools
import hashlib
import io
import json
import logging
import os
import re
import shutil
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

from plva_proxy.privacy import (
    PLACEHOLDER_MANIFEST_KEY,
    HistoryScrubber,
    PrivacyError,
    SessionVault,
    VaultRedactor,
    privacy_request_hook,
    privacy_response_hook,
)
from plva_proxy.providers import PROVIDERS
from plva_proxy.redactor import (
    BACKENDS,
    PROFILES,
    AcceleratedRedactor,
    AcceleratedRedactorConfig,
    RedactorConfig,
    redact_png,
)
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

BANANA_TEXT: Final = "banana"
# Holo3's exact type-tool name lives only in the closed runtime's structured
# output schema, so match text-entry verbs loosely and let the log reveal the
# real name on first use. "answer" is excluded — that is the CUA's reply, not
# text it types into the computer.
_TYPING_TOOL: Final = re.compile(r"type|write|input|fill|keyboard")


def _is_typing_tool(name: str) -> bool:
    low = name.lower()
    return "answer" not in low and _TYPING_TOOL.search(low) is not None


def _bananafy_call(call: dict[str, Any]) -> list[str]:
    """Replace every string text argument on one typing tool-call with 'banana'.

    Handles both wire shapes: args inlined beside ``tool_name`` (the shape the
    Step 0 capture showed) and args nested under an ``args`` object. Returns the
    keys changed, for a privacy-safe log — key names only, never values.
    """

    changed: list[str] = []
    for key, value in list(call.items()):
        if key in {"tool_name", "id"}:
            continue
        if isinstance(value, str):
            call[key] = BANANA_TEXT
            changed.append(key)
        elif key == "args" and isinstance(value, dict):
            for arg_key, arg_value in list(value.items()):
                if isinstance(arg_value, str):
                    value[arg_key] = BANANA_TEXT
                    changed.append(f"args.{arg_key}")
    return changed


def _banana_rewrite_actions(document: dict[str, Any]) -> dict[str, Any]:
    """Test hook: replace whatever text the CUA would type with 'banana'.

    Proves the response-leg action-rewrite seam on live Holo3 output — the same
    seam real placeholder->value resolution will use. Deliberately tolerant: any
    step that is not a JSON action, or not a text-entry tool (click, scroll,
    answer, ...), passes through untouched so a whole task can still run. Logs
    only tool names and rewritten arg keys, never the original (possibly
    private) text the model tried to type.
    """

    choices = document.get("choices")
    if not isinstance(choices, list):
        return document
    rewritten: dict[str, Any] = json.loads(json.dumps(document))
    seen: list[str] = []
    hit: list[str] = []
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
        except ValueError:
            continue  # plain-text answer or non-JSON step — leave untouched
        if not isinstance(action, dict):
            continue
        calls = action.get("tool_calls")
        if not isinstance(calls, list):
            continue
        changed = False
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get("tool_name")
            if not isinstance(name, str):
                continue
            seen.append(name)
            if _is_typing_tool(name) and _bananafy_call(call):
                hit.append(name)
                changed = True
        if changed:
            message["content"] = json.dumps(action, separators=(",", ":"))
    if seen:
        _LOGGER.info("banana hook: tools=%s rewrote=%s", sorted(set(seen)), sorted(set(hit)))
    return rewritten


BANANA_HOOKS: Final = Hooks(on_response=_banana_rewrite_actions)

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


class FrameStore:
    """Memory-only ring buffer of the redacted frames sent upstream.

    Feeds the loopback operator viewer. Never persisted anywhere; dropped
    with the process (§8.6). Holds only post-redaction pixels — exactly what
    the model sees.
    """

    def __init__(self, capacity: int = 8) -> None:
        self._lock = threading.Lock()
        self._frames: deque[bytes] = deque(maxlen=capacity)
        self._total = 0
        self._latest_sha12 = ""
        self._latest_at = 0
        self._analysis: dict[str, Any] = {}

    def add(self, png: bytes, analysis: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._frames.append(png)
            self._total += 1
            self._latest_sha12 = hashlib.sha256(png).hexdigest()[:12]
            self._latest_at = int(time.time())
            self._analysis = copy.deepcopy(analysis) if analysis is not None else {}

    def latest(self) -> bytes | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def findings(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._analysis)

    def stats(self) -> dict[str, int | str]:
        with self._lock:
            counts = self._analysis.get("counts")
            timings = self._analysis.get("timings")
            findings = self._analysis.get("findings")
            duration = (
                timings.get("workerTotalMs", timings.get("total_ms", 0))
                if isinstance(timings, dict)
                else 0
            )
            return {
                "frames_seen": self._total,
                "buffered": len(self._frames),
                "latest_sha12": self._latest_sha12,
                "latest_at": self._latest_at,
                "backend": str(self._analysis.get("backend", "")),
                "regions": int(counts.get("fused", 0)) if isinstance(counts, dict) else 0,
                "ocr_findings": len(findings) if isinstance(findings, list) else 0,
                "total_ms": round(duration) if isinstance(duration, int | float) else 0,
            }


_VIEWER_HTML: Final = """<!doctype html>
<html><head><title>PLVA — what the model sees</title><style>
body{background:#111;color:#ddd;font:14px system-ui;margin:2rem;text-align:center}
img{max-width:96vw;max-height:80vh;border:1px solid #444;margin-top:1rem}
#meta{color:#8b8}
</style></head><body>
<h2>PLVA viewer — redacted frames the model sees</h2>
<p id="meta">waiting for the first redacted frame…</p>
<img id="frame" alt="">
<script>
let lastSha = '';
async function tick(){
  try{
    const s = await fetch('/viewer/stats'); const st = await s.json();
    if(st.frames_seen > 0){
      const at = st.latest_at ? new Date(st.latest_at * 1000).toLocaleTimeString() : '';
      document.getElementById('meta').textContent =
        'frame #' + st.frames_seen + ' · ' + (st.backend || 'baseline') +
        ' · ' + st.total_ms + ' ms · ' + st.regions + ' masks · ' +
        st.ocr_findings + ' OCR findings · sha ' + st.latest_sha12 + ' · at ' + at;
      if(st.latest_sha12 !== lastSha){
        lastSha = st.latest_sha12;
        const r = await fetch('/viewer/frame?t=' + Date.now());
        if(r.ok){
          const img = document.getElementById('frame');
          const old = img.src;
          img.src = URL.createObjectURL(await r.blob());
          if(old) URL.revokeObjectURL(old);
        }
      }
    }
  }catch(e){}
  setTimeout(tick, 250);
}
tick();
</script></body></html>
"""


def _to_png(image_bytes: bytes) -> bytes:
    """Return the image as PNG bytes, converting only when necessary."""

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if (image.format or "") == "PNG":
                return image_bytes
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            return buffer.getvalue()
    except (OSError, ValueError) as exc:
        raise HookError("screenshot bytes are not a decodable image") from exc


def _redact_data_url(
    image_url: Any,
    redact: Callable[[bytes], bytes],
    store: FrameStore | None,
    *,
    capture_manifest: bool = False,
) -> tuple[str, tuple[dict[str, str], ...]]:
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str):
        raise HookError("screenshot has no URL")
    header, separator, encoded = url.partition(",")
    if not separator or not header.startswith("data:") or not header.endswith(";base64"):
        raise HookError("screenshot is not an inline base64 data URL")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HookError("screenshot base64 is invalid") from exc
    try:
        source = _to_png(raw)
        if capture_manifest:
            redact_with_manifest = getattr(redact, "redact_with_manifest", None)
            if not callable(redact_with_manifest):
                raise HookError("privacy redactor has no placeholder manifest")
            redacted, manifest = redact_with_manifest(source)
        else:
            redacted = redact(source)
            manifest = ()
    except HookError:
        raise
    except Exception as exc:
        # Whatever went wrong in the redactor, the raw frame must not leave.
        raise HookError("frame redaction failed") from exc
    if store is not None:
        analysis = getattr(redact, "latest_analysis", None)
        store.add(redacted, analysis if isinstance(analysis, dict) else None)
    _LOGGER.info("frame in_bytes=%d out_bytes=%d", len(raw), len(redacted))
    return (
        "data:image/png;base64," + base64.b64encode(redacted).decode("ascii"),
        manifest,
    )


def frame_redaction_hook(
    redact: Callable[[bytes], bytes],
    store: FrameStore | None = None,
    *,
    include_placeholder_manifest: bool = False,
) -> RequestHook:
    """Build a request hook that redacts every outbound screenshot (§8.2).

    Each screenshot is decoded, converted to PNG when needed, run through
    ``redact``, and swapped for the redacted PNG (a copy — input pixels are
    never mutated, §8.3). Requests without a screenshot pass through
    untouched; any redaction failure raises so no raw frame can ever leave.
    """

    def apply(
        document: dict[str, Any], headers: dict[str, str]
    ) -> tuple[dict[str, Any], dict[str, str]]:
        rewritten: dict[str, Any] = json.loads(json.dumps(document))
        redacted = 0
        manifests: dict[int, list[dict[str, str]]] = {}
        for message_index, message in enumerate(rewritten.get("messages") or []):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    redacted_url, manifest = _redact_data_url(
                        part.get("image_url"),
                        redact,
                        store,
                        capture_manifest=include_placeholder_manifest,
                    )
                    part["image_url"] = {"url": redacted_url}
                    if include_placeholder_manifest:
                        target = manifests.setdefault(message_index, [])
                        known = {item["token"] for item in target}
                        if not isinstance(manifest, (list, tuple)):
                            raise HookError("privacy redactor returned an invalid manifest")
                        for item in manifest:
                            if (
                                not isinstance(item, dict)
                                or not isinstance(item.get("token"), str)
                                or not isinstance(item.get("class"), str)
                            ):
                                raise HookError("privacy redactor returned an invalid manifest")
                            token = item["token"]
                            if token not in known:
                                target.append({"token": token, "class": item["class"]})
                                known.add(token)
                    redacted += 1
        if redacted:
            _LOGGER.info("redaction hook processed %d screenshot(s)", redacted)
        if include_placeholder_manifest and manifests:
            current_index = max(manifests)
            rewritten[PLACEHOLDER_MANIFEST_KEY] = {
                "message_index": current_index,
                "items": manifests[current_index],
            }
        return rewritten, headers

    return apply


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


def _chain_response_hooks(
    first: ResponseHook | None, second: ResponseHook | None
) -> ResponseHook | None:
    """Compose two optional response hooks in order."""

    if first is None:
        return second
    if second is None:
        return first

    def chained(document: dict[str, Any]) -> dict[str, Any]:
        return second(first(document))

    return chained


def _combine_hooks(first: Hooks | None, second: Hooks | None) -> Hooks | None:
    if first is None:
        return second
    if second is None:
        return first
    return Hooks(
        on_request=_chain_request_hooks(first.on_request, second.on_request),
        on_response=_chain_response_hooks(first.on_response, second.on_response),
    )


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
    frame_store: FrameStore | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    startup_callbacks: tuple[Callable[[], None], ...] = (),
    cleanup_callbacks: tuple[Callable[[], None], ...] = (),
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
            for callback in startup_callbacks:
                await run_in_threadpool(callback)
            yield
        finally:
            await client.aclose()
            for callback in cleanup_callbacks:
                await run_in_threadpool(callback)

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
                # Threadpool keeps the loop responsive while slow hooks
                # (e.g. frame redaction) work on the request.
                document, headers = await run_in_threadpool(request_hook, document, headers)
                body = json.dumps(document, separators=(",", ":")).encode()
            except (HookError, PrivacyError, ValueError) as exc:
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
            except (HookError, PrivacyError, ValueError) as exc:
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

    if frame_store is not None:
        add_viewer_routes(app, frame_store)

    return app


def add_viewer_routes(app: FastAPI, store: FrameStore) -> None:
    """Attach the loopback-only obscured-frame viewer to an application."""

    @app.get("/viewer")
    async def viewer_page() -> HTMLResponse:
        return HTMLResponse(_VIEWER_HTML)

    @app.get("/viewer/frame")
    async def viewer_frame() -> Response:
        png = store.latest()
        if png is None:
            raise HTTPException(status_code=404, detail="no redacted frame yet")
        return Response(content=png, media_type="image/png", headers={"cache-control": "no-store"})

    @app.get("/viewer/stats")
    async def viewer_stats() -> dict[str, int | str]:
        return store.stats()

    @app.get("/viewer/findings")
    async def viewer_findings() -> Response:
        return Response(
            content=json.dumps(store.findings(), separators=(",", ":")),
            media_type="application/json",
            headers={"cache-control": "no-store"},
        )


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
    parser.add_argument(
        "--provider",
        choices=tuple(PROVIDERS),
        default=os.environ.get("PLVA_PROVIDER", "overshoot"),
        help="inference-provider preset (default: overshoot)",
    )
    parser.add_argument(
        "--upstream",
        default=None,
        help="override the selected provider's base URL",
    )
    parser.add_argument(
        "--hook",
        choices=("none", "test", "banana"),
        default="none",
        help="traffic mutation hooks: none = pass-through, test = Step 3 test hooks, "
        "banana = replace every text the CUA types with 'banana'",
    )
    parser.add_argument(
        "--hook-image",
        type=Path,
        default=None,
        help="replace every outbound screenshot with this static PNG/JPEG/WebP "
        "(fails closed if a request has no screenshot)",
    )
    parser.add_argument(
        "--redact",
        type=Path,
        default=None,
        help="redact every outbound screenshot through this plva-v2-baseline "
        "directory (or its bin/plva-v2.mjs); enables the /viewer page",
    )
    parser.add_argument(
        "--redact-profile",
        choices=PROFILES,
        default="high-recall",
        help="detector profile for --redact",
    )
    parser.add_argument(
        "--redact-engine",
        choices=("accelerated", "vision", "baseline"),
        default="accelerated",
        help="accelerated browser worker, native Vision/Core ML worker, or frozen baseline",
    )
    parser.add_argument(
        "--redact-backend",
        choices=BACKENDS,
        default="auto",
        help="accelerated inference backend (default: auto prefers WebGPU)",
    )
    parser.add_argument(
        "--redact-worker",
        type=Path,
        default=Path("redactor-worker"),
        help="accelerated redactor worker directory",
    )
    parser.add_argument(
        "--vision-worker",
        type=Path,
        default=Path("coreml-redactor"),
        help="native Vision/Core ML worker package directory",
    )
    parser.add_argument(
        "--vision-mode",
        choices=("fast", "cascade", "accurate"),
        default="cascade",
        help="Vision OCR strategy (default: fast full frame + accurate sensitive regions)",
    )
    parser.add_argument(
        "--visual-model",
        type=Path,
        default=None,
        help="visual detector ONNX override; OCR and Rampart still come from --redact",
    )
    parser.add_argument(
        "--redact-lifecycle",
        choices=("adaptive", "eager", "cold"),
        default="adaptive",
        help="adaptive starts on demand and releases after idle; "
        "eager stays warm; cold exits per frame",
    )
    parser.add_argument(
        "--redact-idle-seconds",
        type=float,
        default=60.0,
        help="adaptive worker idle timeout (default: 60 seconds)",
    )
    parser.add_argument(
        "--privacy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable the Step 5 vault, placeholder chips, resolution, and history scrub",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    provider = PROVIDERS.get(args.provider)
    if provider is None:
        parser.error("--provider must be overshoot or hcompany")
    upstream = args.upstream or provider.base_url
    if not upstream.startswith(("http://", "https://")):
        parser.error("--upstream must be an http(s) URL")
    if args.redact_idle_seconds < 0:
        parser.error("--redact-idle-seconds cannot be negative")
    if args.privacy and (args.redact is None or args.redact_engine != "vision"):
        parser.error("--privacy requires --redact with --redact-engine vision")
    api_key = next(
        (
            value
            for key in provider.key_names
            if (value := os.environ.get(key) or _env_file_value(Path(".env"), key))
        ),
        None,
    )
    if not api_key:
        names = " or ".join(provider.key_names)
        parser.error(f"{names} is required for provider {args.provider}")

    image_hook: RequestHook | None = None
    if args.hook_image is not None:
        try:
            image_hook = image_replacement_hook(args.hook_image)
        except (OSError, ValueError) as exc:
            parser.error(f"--hook-image is unusable: {exc}")

    redact_hook: RequestHook | None = None
    privacy_hooks: Hooks | None = None
    frame_store: FrameStore | None = None
    cleanup_callbacks: tuple[Callable[[], None], ...] = ()
    startup_callbacks: tuple[Callable[[], None], ...] = ()
    if args.redact is not None:
        cli_path = args.redact / "bin" / "plva-v2.mjs" if args.redact.is_dir() else args.redact
        if not cli_path.is_file():
            parser.error(f"--redact CLI not found: {cli_path}")
        if args.redact_engine != "vision" and shutil.which("node") is None:
            parser.error("--redact requires node on PATH")
        frame_store = FrameStore()
        if args.redact_engine in {"accelerated", "vision"}:
            lifecycle = {
                "adaptive": args.redact_idle_seconds,
                "eager": None,
                "cold": 0.0,
            }[args.redact_lifecycle]
            if args.redact_engine == "vision":
                vision_root = args.vision_worker.resolve()
                python = vision_root / ".venv" / "bin" / "python"
                module = vision_root / "src" / "plva_coreml" / "worker.py"
                if not python.is_file() or not module.is_file():
                    parser.error(
                        "Vision worker is not installed; run `uv sync --group dev` "
                        f"in {args.vision_worker}"
                    )
                accelerated = AcceleratedRedactor(
                    AcceleratedRedactorConfig(
                        baseline_root=cli_path.parent.parent,
                        worker_script=module,
                        node_path=str(python),
                        profile=args.redact_profile,
                        idle_timeout_s=lifecycle,
                        worker_kind="vision",
                        worker_root=vision_root,
                        cache_root=vision_root / ".cache",
                        vision_mode=args.vision_mode,
                        visual_model=args.visual_model,
                    )
                )
            else:
                worker_script = args.redact_worker / "bin" / "redactor-worker.mjs"
                if not worker_script.is_file():
                    parser.error(f"accelerated redactor worker not found: {worker_script}")
                if not (args.redact_worker / "dist" / "index.html").is_file():
                    parser.error(
                        "accelerated redactor is not built; run npm install && npm run build "
                        f"in {args.redact_worker}"
                    )
                accelerated = AcceleratedRedactor(
                    AcceleratedRedactorConfig(
                        baseline_root=cli_path.parent.parent,
                        worker_script=worker_script,
                        backend=args.redact_backend,
                        profile=args.redact_profile,
                        idle_timeout_s=lifecycle,
                    )
                )
            active_redactor: Callable[[bytes], bytes] = accelerated
            lifecycle_owner: AcceleratedRedactor | VaultRedactor = accelerated
            if args.privacy:
                vault = SessionVault()
                vaulted = VaultRedactor(accelerated, vault)
                active_redactor = vaulted
                lifecycle_owner = vaulted
                privacy_hooks = Hooks(
                    on_request=privacy_request_hook(
                        HistoryScrubber(vault, accelerated.classify_texts)
                    ),
                    on_response=privacy_response_hook(vault),
                )
            redact_hook = frame_redaction_hook(
                active_redactor,
                frame_store,
                include_placeholder_manifest=args.privacy,
            )
            if args.redact_lifecycle == "eager":
                startup_callbacks = (lifecycle_owner.start,)
            cleanup_callbacks = (lifecycle_owner.close,)
        else:
            redactor_config = RedactorConfig(cli_path=cli_path, profile=args.redact_profile)
            redact_hook = frame_redaction_hook(
                functools.partial(redact_png, redactor_config), frame_store
            )

    hooks = {"test": TEST_HOOKS, "banana": BANANA_HOOKS}.get(args.hook)
    for extra_hook in (image_hook, redact_hook):
        if extra_hook is not None:
            hooks = _combine_hooks(hooks, Hooks(on_request=extra_hook))
    hooks = _combine_hooks(hooks, privacy_hooks)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if frame_store is not None:
        _LOGGER.info("viewer: http://127.0.0.1:%d/viewer", args.port)
    uvicorn.run(
        create_app(
            ProxyConfig(upstream_base_url=upstream, api_key=api_key),
            hooks=hooks,
            frame_store=frame_store,
            startup_callbacks=startup_callbacks,
            cleanup_callbacks=cleanup_callbacks,
        ),
        host=LOOPBACK_HOST,
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
