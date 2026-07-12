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

from plva_proxy.credentials import (
    CREDENTIAL_SOURCES,
    CredentialSource,
    env_file_value,
    resolve_provider_key,
)
from plva_proxy.privacy import (
    PLACEHOLDER_MANIFEST_KEY,
    HistoryScrubber,
    PrivacyError,
    SafetyPolicy,
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
from plva_proxy.tools import (
    ToolError,
    ToolLoop,
    ToolRegistry,
    tool_teaching_request_hook,
)

DEFAULT_PORT: Final = 18081
FRAME_AUDIT_IDS_KEY: Final = "_plva_frame_audit_ids"
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
    instance_token: str | None = None


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
    rewritten: dict[str, Any] = copy.deepcopy(document)
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
    rewritten: dict[str, Any] = copy.deepcopy(document)
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
        rewritten: dict[str, Any] = copy.deepcopy(document)
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


def grammar_capture_hook(out_path: Path) -> RequestHook:
    """Snapshot the first request's action grammar to a file — schema only, never messages.

    Step 6.5 needs the runtime's ``structured_outputs`` value to judge whether a
    novel tool action is even grammatically admissible. The snapshot records
    request keys, model id, ``structured_outputs``, and ``chat_template_kwargs``
    and deliberately nothing else (§8.5): the message history — the only part
    that can carry frames or values — is never written.
    """

    lock = threading.Lock()
    captured = False

    def capture(
        document: dict[str, Any], headers: dict[str, str]
    ) -> tuple[dict[str, Any], dict[str, str]]:
        nonlocal captured
        with lock:
            first = not captured
            captured = True
        if first:
            request_keys = sorted(document)
            snapshot = {
                "request_keys": request_keys,
                "model": document.get("model"),
                "structured_outputs": document.get("structured_outputs"),
                "chat_template_kwargs": document.get("chat_template_kwargs"),
            }
            try:
                out_path.write_text(json.dumps(snapshot, indent=2) + "\n")
            except OSError as exc:
                raise HookError("grammar snapshot could not be written") from exc
            _LOGGER.info("grammar snapshot written: keys=%d", len(request_keys))
        return document, headers

    return capture


@dataclass(slots=True)
class FrameRecord:
    frame_id: int
    png: bytes
    sha12: str
    created_at: int
    analysis: dict[str, Any]
    state: str = "prepared"
    upstream_status: int | None = None
    sent_at: int = 0


class FrameStore:
    """Memory-only ring buffer of the redacted frames sent upstream.

    Feeds the loopback operator viewer. Never persisted anywhere; dropped
    with the process (§8.6). Holds only post-redaction pixels — exactly what
    the model sees.
    """

    def __init__(self, capacity: int = 32) -> None:
        if capacity < 1:
            raise ValueError("frame store capacity must be positive")
        self._lock = threading.Lock()
        self._capacity = capacity
        self._frames: deque[FrameRecord] = deque(maxlen=capacity)
        self._total = 0

    def add(self, png: bytes, analysis: dict[str, Any] | None = None) -> int:
        with self._lock:
            self._total += 1
            record = FrameRecord(
                frame_id=self._total,
                png=png,
                sha12=hashlib.sha256(png).hexdigest()[:12],
                created_at=int(time.time()),
                analysis=copy.deepcopy(analysis) if analysis is not None else {},
            )
            self._frames.append(record)
            return record.frame_id

    def latest(self) -> bytes | None:
        with self._lock:
            return self._frames[-1].png if self._frames else None

    def frame(self, frame_id: int) -> bytes | None:
        with self._lock:
            record = next((item for item in self._frames if item.frame_id == frame_id), None)
            return record.png if record is not None else None

    def mark_sent(self, frame_ids: tuple[int, ...], status_code: int) -> None:
        now = int(time.time())
        targets = set(frame_ids)
        with self._lock:
            for record in self._frames:
                if record.frame_id in targets:
                    record.state = "sent"
                    record.upstream_status = status_code
                    record.sent_at = now

    def mark_failed(self, frame_ids: tuple[int, ...]) -> None:
        targets = set(frame_ids)
        with self._lock:
            for record in self._frames:
                if record.frame_id in targets:
                    record.state = "failed"

    def entries(self) -> list[dict[str, int | str | None]]:
        with self._lock:
            return [self._metadata(record) for record in self._frames]

    def findings(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._frames[-1].analysis) if self._frames else {}

    def stats(self) -> dict[str, int | str]:
        with self._lock:
            latest = self._frames[-1] if self._frames else None
            metadata = self._metadata(latest) if latest is not None else {}
            return {
                "frames_seen": self._total,
                "buffered": len(self._frames),
                "capacity": self._capacity,
                "latest_sha12": str(metadata.get("sha12", "")),
                "latest_at": int(metadata.get("created_at", 0)),
                "backend": str(metadata.get("backend", "")),
                "regions": int(metadata.get("regions", 0)),
                "ocr_findings": int(metadata.get("ocr_findings", 0)),
                "total_ms": int(metadata.get("total_ms", 0)),
                "latest_state": str(metadata.get("state", "waiting")),
            }

    @staticmethod
    def _metadata(record: FrameRecord) -> dict[str, Any]:
        counts = record.analysis.get("counts")
        timings = record.analysis.get("timings")
        findings = record.analysis.get("findings")
        duration = (
            timings.get("workerTotalMs", timings.get("total_ms", 0))
            if isinstance(timings, dict)
            else 0
        )
        return {
            "id": record.frame_id,
            "sha12": record.sha12,
            "created_at": record.created_at,
            "sent_at": record.sent_at,
            "state": record.state,
            "upstream_status": record.upstream_status,
            "backend": str(record.analysis.get("backend", "")),
            "regions": int(counts.get("fused", 0)) if isinstance(counts, dict) else 0,
            "ocr_findings": len(findings) if isinstance(findings, list) else 0,
            "total_ms": round(duration) if isinstance(duration, int | float) else 0,
        }


def _preview_text(document: dict[str, Any]) -> str:
    """Short value-free-enough preview: the last user text, post-scrub."""

    preview = ""
    for message in document.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            preview = content
        elif isinstance(content, list):
            texts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
            if texts:
                preview = " ".join(texts)
    return " ".join(preview.split())[:160]


def _decode_data_url(url: Any) -> tuple[str, bytes] | None:
    """Return (media_type, bytes) for an inline base64 data URL, else None."""

    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    header, separator, encoded = url.partition(",")
    if not separator or not header.endswith(";base64"):
        return None
    media_type = header[len("data:") : -len(";base64")] or "application/octet-stream"
    try:
        return media_type, base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None


@dataclass(slots=True)
class CallRecord:
    call_id: int
    at: int
    duration_ms: int
    status: int | None
    state: str
    request: dict[str, Any]
    response: dict[str, Any] | None
    images: tuple[tuple[str, bytes], ...]


class CallStore:
    """Memory-only ring buffer of upstream model calls for the history viewer.

    Each record holds the request exactly as it was sent upstream — after the
    hook chain, so frames are redacted, history is scrubbed, and injected
    system prompts are visible — and the completion exactly as the model
    returned it, before placeholder resolution reinserts local values. Inline
    screenshots are pulled out of the document and kept as bytes so the
    viewer can render them without re-shipping base64 on every poll. Never
    persisted anywhere; dropped with the process (§8.6).
    """

    def __init__(self, capacity: int = 32) -> None:
        if capacity < 1:
            raise ValueError("call store capacity must be positive")
        self._lock = threading.Lock()
        self._calls: deque[CallRecord] = deque(maxlen=capacity)
        self._total = 0

    def record(
        self,
        document: dict[str, Any],
        *,
        status: int | None,
        response: dict[str, Any] | None,
        duration_ms: int,
        state: str,
    ) -> int:
        request: dict[str, Any] = copy.deepcopy(document)
        images: list[tuple[str, bytes]] = []
        for message in request.get("messages") or []:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                decoded = _decode_data_url(url)
                if decoded is None:
                    part["image_url"] = {"url": "(external image)"}
                    continue
                part["image_url"] = {"url": f"plva:image/{len(images)}"}
                images.append(decoded)
        with self._lock:
            self._total += 1
            self._calls.append(
                CallRecord(
                    call_id=self._total,
                    at=int(time.time()),
                    duration_ms=duration_ms,
                    status=status,
                    state=state,
                    request=request,
                    response=json.loads(json.dumps(response)) if response is not None else None,
                    images=tuple(images),
                )
            )
            return self._total

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._summary(record) for record in self._calls]

    def full(self, call_id: int) -> dict[str, Any] | None:
        with self._lock:
            record = self._find(call_id)
            if record is None:
                return None
            return {
                **self._summary(record),
                "request": copy.deepcopy(record.request),
                "response": copy.deepcopy(record.response),
            }

    def image(self, call_id: int, index: int) -> tuple[str, bytes] | None:
        with self._lock:
            record = self._find(call_id)
            if record is None or not 0 <= index < len(record.images):
                return None
            return record.images[index]

    def _find(self, call_id: int) -> CallRecord | None:
        return next((item for item in self._calls if item.call_id == call_id), None)

    @staticmethod
    def _summary(record: CallRecord) -> dict[str, Any]:
        messages = record.request.get("messages")
        return {
            "id": record.call_id,
            "at": record.at,
            "duration_ms": record.duration_ms,
            "status": record.status,
            "state": record.state,
            "model": str(record.request.get("model", "")),
            "messages": len(messages) if isinstance(messages, list) else 0,
            "images": [media_type for media_type, _ in record.images],
            "preview": _preview_text(record.request),
        }


_VIEWER_HTML: Final = """<!doctype html>
<html><head>
<meta name="viewport" content="width=device-width">
<title>PLVA — sent-frame audit</title>
<style>
:root {
  color-scheme:dark; font:14px system-ui; background:#0d0f12; color:#e6e8eb;
}
* { box-sizing:border-box; }
body { margin:0; display:grid; grid-template-columns:290px 1fr; min-height:100vh; }
aside { border-right:1px solid #2b3038; padding:18px; overflow:auto; background:#12151a; }
main { padding:22px; text-align:center; min-width:0; }
h1 { font-size:18px; margin:0 0 8px; }
p { color:#a7adb7; line-height:1.45; }
#safe { color:#8ed4a6; font-size:12px; }
#frames { display:grid; gap:8px; margin-top:18px; }
button {
  appearance:none; text-align:left; border:1px solid #343b45; background:#1a1f26;
  color:#e6e8eb; border-radius:8px; padding:10px; cursor:pointer;
}
button:hover, button.active { border-color:#77c990; background:#202a25; }
.sub { display:block; color:#9da5b0; font-size:11px; margin-top:4px; }
#meta { color:#9fdaa9; min-height:22px; }
img {
  display:none; max-width:100%; max-height:82vh; border:1px solid #343b45;
  background:#08090b;
}
#empty { margin-top:30vh; color:#7d8590; }
@media(max-width:760px) {
  body { display:block; }
  aside { border-right:0; border-bottom:1px solid #2b3038; max-height:38vh; }
}
</style></head><body><aside>
<h1>PLVA sent-frame audit</h1>
<p id="safe">Memory-only. This page loads only redacted pixels and value-free metadata.</p>
<p>Each row is one image included in an upstream request. <b>sent</b> means the provider
returned an HTTP response; <b>prepared</b> means it never left the hook chain.</p>
<div id="frames"></div>
</aside><main>
<p id="meta">waiting for the first redacted frame…</p>
<div id="empty">No frames captured yet.</div>
<img id="frame" alt="Selected redacted frame sent to the model">
</main><script>
let selected = null, followLatest = true, rendered = '';
const esc = value => String(value ?? '').replace(
  /[&<>"']/g,
  char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]),
);
function label(item) {
  const http = item.upstream_status === null ? '' : ` · HTTP ${item.upstream_status}`;
  return `<b>#${item.id} · ${esc(item.state)}</b><span class="sub">` +
    `${esc(item.backend || 'baseline')} · ${item.regions} masks · ` +
    `${item.total_ms} ms${http}<br>sha ${esc(item.sha12)}</span>`;
}
async function show(item) {
  selected = item.id;
  document.querySelectorAll('button').forEach(button => button.classList.toggle(
    'active', Number(button.dataset.id) === selected,
  ));
  const at = item.sent_at || item.created_at;
  document.getElementById('meta').textContent =
    `frame #${item.id} · ${item.state} · ${item.regions} masks · ` +
    `${item.ocr_findings} OCR findings · sha ${item.sha12} · ` +
    new Date(at * 1000).toLocaleTimeString();
  const key = `${item.id}:${item.state}:${item.upstream_status}`;
  if (key !== rendered) {
    rendered = key;
    const image = document.getElementById('frame');
    image.src = `/viewer/frame/${item.id}?t=${Date.now()}`;
    image.style.display = 'inline-block';
    document.getElementById('empty').style.display = 'none';
  }
}
async function tick() {
  try {
    const response = await fetch('/viewer/frames', {cache:'no-store'});
    const data = await response.json();
    const items = data.frames || [];
    const list = document.getElementById('frames');
    list.innerHTML = items.slice().reverse().map(
      item => `<button data-id="${item.id}">${label(item)}</button>`,
    ).join('');
    list.querySelectorAll('button').forEach(button => button.onclick = () => {
      followLatest = false;
      const item = items.find(value => value.id === Number(button.dataset.id));
      if (item) show(item);
    });
    if (items.length) {
      const selectedItem = items.find(value => value.id === selected);
      show(followLatest ? items[items.length - 1] : selectedItem || items[items.length - 1]);
    }
  } catch (error) {}
  setTimeout(tick, 500);
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
) -> tuple[str, tuple[dict[str, str], ...], int | None]:
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
    frame_id: int | None = None
    if store is not None:
        analysis = getattr(redact, "latest_analysis", None)
        frame_id = store.add(redacted, analysis if isinstance(analysis, dict) else None)
    _LOGGER.info("frame in_bytes=%d out_bytes=%d", len(raw), len(redacted))
    return (
        "data:image/png;base64," + base64.b64encode(redacted).decode("ascii"),
        manifest,
        frame_id,
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
        rewritten: dict[str, Any] = copy.deepcopy(document)
        if rewritten.pop(FRAME_AUDIT_IDS_KEY, None) is not None:
            raise HookError("request forged internal frame audit metadata")
        redacted = 0
        frame_ids: list[int] = []
        manifests: dict[int, list[dict[str, str]]] = {}
        for message_index, message in enumerate(rewritten.get("messages") or []):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    redacted_url, manifest, frame_id = _redact_data_url(
                        part.get("image_url"),
                        redact,
                        store,
                        capture_manifest=include_placeholder_manifest,
                    )
                    part["image_url"] = {"url": redacted_url}
                    if frame_id is not None:
                        frame_ids.append(frame_id)
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
                                or not isinstance(item.get("safety_level", "hide_use"), str)
                            ):
                                raise HookError("privacy redactor returned an invalid manifest")
                            token = item["token"]
                            if token not in known:
                                target.append(
                                    {
                                        "token": token,
                                        "class": item["class"],
                                        "safety_level": item.get("safety_level", "hide_use"),
                                    }
                                )
                                known.add(token)
                    redacted += 1
        if redacted:
            _LOGGER.info("redaction hook processed %d screenshot(s)", redacted)
        if frame_ids:
            rewritten[FRAME_AUDIT_IDS_KEY] = frame_ids
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
    call_store: CallStore | None = None,
    vault: SessionVault | None = None,
    scrubber: HistoryScrubber | None = None,
    tool_loop: ToolLoop | None = None,
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
        frame_ids: tuple[int, ...] = ()
        call_document: dict[str, Any] | None = None

        def record_call(
            status: int | None, response_document: dict[str, Any] | None, state: str
        ) -> None:
            if call_store is None or call_document is None:
                return
            call_store.record(
                call_document,
                status=status,
                response=response_document,
                duration_ms=int((time.monotonic() - started) * 1000),
                state=state,
            )

        request_hook = active_hooks.on_request if use_hooks else None
        if request_hook is not None:
            try:
                document = json.loads(body)
                if not isinstance(document, dict):
                    raise HookError("request body is not a JSON object")
                # Threadpool keeps the loop responsive while slow hooks
                # (e.g. frame redaction) work on the request.
                document, headers = await run_in_threadpool(request_hook, document, headers)
                raw_frame_ids = document.pop(FRAME_AUDIT_IDS_KEY, [])
                if not isinstance(raw_frame_ids, list) or any(
                    not isinstance(frame_id, int) for frame_id in raw_frame_ids
                ):
                    raise HookError("frame audit metadata is invalid")
                frame_ids = tuple(raw_frame_ids)
                body = json.dumps(document, separators=(",", ":")).encode()
                # Post-hook only: what leaves here is redacted and scrubbed,
                # so it is the only version the history viewer may hold.
                call_document = document
            except (HookError, PrivacyError, ToolError, ValueError) as exc:
                # Hook/Privacy errors are deliberately value-free fixed messages;
                # include them so live fail-closed loops are diagnosable without
                # ever logging request content or vault values.
                _LOGGER.warning("request hook failed closed: %s: %s", type(exc).__name__, str(exc))
                raise HTTPException(status_code=502, detail="request hook failed") from exc

        upstream_request = client.build_request(method, path, content=body or None, headers=headers)
        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            if frame_store is not None and frame_ids:
                frame_store.mark_failed(frame_ids)
            record_call(None, None, "failed")
            _LOGGER.warning("upstream request failed: %s", type(exc).__name__)
            raise HTTPException(status_code=502, detail="upstream request failed") from exc
        if frame_store is not None and frame_ids:
            frame_store.mark_sent(frame_ids, upstream.status_code)

        content_type = upstream.headers.get("content-type", "application/octet-stream")
        is_sse = content_type.lower().startswith("text/event-stream")
        response_hook = active_hooks.on_response if use_hooks else None
        active_tool_loop = tool_loop if use_hooks else None
        hook_applies = (
            response_hook is not None or active_tool_loop is not None
        ) and upstream.status_code == 200

        async def _run_tool_loop(loop: ToolLoop, completion: dict[str, Any]) -> dict[str, Any]:
            # The original request may have asked for text/event-stream; tool
            # continuations are always plain JSON, so force that explicitly
            # rather than reusing the caller's Accept header verbatim.
            follow_headers = {**headers, "accept": "application/json"}
            request_document = json.loads(body)
            if not isinstance(request_document, dict):
                raise HookError("request body is not a JSON object")
            rounds = 0
            call = loop.detect(completion)
            while call is not None:
                rounds += 1
                if rounds > loop.max_rounds:
                    raise HookError("tool loop exceeded max rounds")
                result = await run_in_threadpool(loop.execute, call)
                request_document = loop.continuation(request_document, completion, call, result)
                continuation_body = json.dumps(request_document, separators=(",", ":")).encode()
                follow_request = client.build_request(
                    "POST", path, content=continuation_body, headers=follow_headers
                )
                try:
                    follow = await client.send(follow_request)
                except httpx.HTTPError as exc:
                    raise HookError("tool continuation request failed") from exc
                if follow.status_code != 200:
                    raise HookError(f"tool continuation status {follow.status_code}")
                try:
                    parsed = json.loads(follow.content)
                except ValueError as exc:
                    raise HookError("tool continuation response is not JSON") from exc
                if not isinstance(parsed, dict):
                    raise HookError("tool continuation response is not an object")
                completion = parsed
                call = loop.detect(completion)
            if rounds:
                _LOGGER.info("tool loop completed: rounds=%d", rounds)
            return completion

        if is_sse and not hook_applies:
            record_call(upstream.status_code, None, "streamed")
            return StreamingResponse(
                _relay_stream(upstream, started),
                status_code=upstream.status_code,
                media_type=content_type,
            )
        try:
            payload = await upstream.aread()
        except httpx.HTTPError as exc:
            record_call(upstream.status_code, None, "failed")
            _LOGGER.warning("upstream read failed: %s", type(exc).__name__)
            raise HTTPException(status_code=502, detail="upstream response failed") from exc
        finally:
            await upstream.aclose()

        if hook_applies:
            recorded = False
            try:
                document = _assemble_sse_completion(payload) if is_sse else json.loads(payload)
                if not isinstance(document, dict):
                    raise HookError("completion body is not a JSON object")
                if active_tool_loop is not None:
                    document = await _run_tool_loop(active_tool_loop, document)
                # Pre-resolution: the model's own output still carries
                # placeholders, never the restored local values. Recorded
                # after the tool loop so the history viewer holds the final
                # completion actually forwarded to the runtime, not an
                # intermediate plva_tool call.
                record_call(200, document, "sent")
                recorded = True
                mutated = response_hook(document) if response_hook is not None else document
            except (HookError, PrivacyError, ToolError, ValueError) as exc:
                if not recorded:
                    record_call(200, None, "hook_failed")
                _LOGGER.warning("response hook failed closed: %s: %s", type(exc).__name__, str(exc))
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

        if call_store is not None and call_document is not None:
            response_document: dict[str, Any] | None = None
            try:
                parsed = _assemble_sse_completion(payload) if is_sse else json.loads(payload)
                if isinstance(parsed, dict):
                    response_document = parsed
            except (HookError, ValueError, UnicodeDecodeError):
                response_document = None
            record_call(upstream.status_code, response_document, "sent")
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
        payload = {"status": "ok"}
        if config.instance_token is not None:
            payload["instance"] = config.instance_token
        return payload

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        return await _relay(request, "GET", "/models")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _relay(request, "POST", "/chat/completions", use_hooks=True)

    if frame_store is not None:
        add_viewer_routes(app, frame_store, calls=call_store, vault=vault, scrubber=scrubber)

    return app


def add_viewer_routes(
    app: FastAPI,
    store: FrameStore,
    *,
    calls: CallStore | None = None,
    vault: SessionVault | None = None,
    scrubber: HistoryScrubber | None = None,
) -> None:
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

    @app.get("/viewer/frame/{frame_id}")
    async def viewer_frame_by_id(frame_id: int) -> Response:
        png = store.frame(frame_id)
        if png is None:
            raise HTTPException(status_code=404, detail="redacted frame is no longer buffered")
        return Response(content=png, media_type="image/png", headers={"cache-control": "no-store"})

    @app.get("/viewer/frames")
    async def viewer_frames() -> Response:
        return Response(
            content=json.dumps({"frames": store.entries()}, separators=(",", ":")),
            media_type="application/json",
            headers={"cache-control": "no-store"},
        )

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

    @app.get("/viewer/vault")
    async def viewer_vault() -> Response:
        payload = (
            {"entries": list(vault.entries()), "policy": vault.policy_snapshot()}
            if vault is not None
            else {"entries": [], "policy": {}}
        )
        return Response(
            content=json.dumps(payload, separators=(",", ":")),
            media_type="application/json",
            headers={"cache-control": "no-store"},
        )

    def require_local_approval_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if origin is not None and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(status_code=403, detail="approval origin is not this proxy")

    async def approval_payload(request: Request) -> dict[str, Any]:
        require_local_approval_origin(request)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="approval body is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="approval body must be an object")
        return payload

    @app.get("/viewer/approvals")
    async def viewer_approvals() -> Response:
        payload = {"approvals": list(vault.approvals()) if vault is not None else []}
        return Response(
            content=json.dumps(payload, separators=(",", ":")),
            media_type="application/json",
            headers={"cache-control": "no-store"},
        )

    @app.post("/viewer/approvals", status_code=201)
    async def create_viewer_approval(request: Request) -> Response:
        if vault is None:
            raise HTTPException(status_code=404, detail="privacy vault is disabled")
        payload = await approval_payload(request)
        try:
            grant = vault.grant_approval(
                payload["token"],
                tool_name=payload["tool_name"],
                argument_path=payload["argument_path"],
                target=payload.get("target"),
                ttl_seconds=payload.get("ttl_seconds"),
                use_count=payload.get("use_count"),
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail="approval context is incomplete") from exc
        except (PrivacyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(
            content=json.dumps(grant, separators=(",", ":")),
            status_code=201,
            media_type="application/json",
            headers={"cache-control": "no-store"},
        )

    @app.delete("/viewer/approvals")
    async def delete_viewer_approval(request: Request) -> Response:
        if vault is None:
            raise HTTPException(status_code=404, detail="privacy vault is disabled")
        payload = await approval_payload(request)
        try:
            revoked = vault.revoke_approval(
                payload["token"],
                tool_name=payload["tool_name"],
                argument_path=payload["argument_path"],
                target=payload.get("target"),
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail="approval context is incomplete") from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not revoked:
            raise HTTPException(status_code=404, detail="approval grant was not found")
        return Response(status_code=204, headers={"cache-control": "no-store"})

    @app.get("/viewer/filter")
    async def viewer_filter() -> Response:
        payload = scrubber.diagnostics() if scrubber is not None else {"status": "disabled"}
        return Response(
            content=json.dumps(payload, separators=(",", ":")),
            media_type="application/json",
            headers={"cache-control": "no-store"},
        )

    @app.get("/viewer/calls")
    async def viewer_calls() -> Response:
        payload = {"calls": calls.entries() if calls is not None else []}
        return Response(
            content=json.dumps(payload, separators=(",", ":")),
            media_type="application/json",
            headers={"cache-control": "no-store"},
        )

    @app.get("/viewer/call/{call_id}")
    async def viewer_call(call_id: int) -> Response:
        record = calls.full(call_id) if calls is not None else None
        if record is None:
            raise HTTPException(status_code=404, detail="call is no longer buffered")
        return Response(
            content=json.dumps(record, separators=(",", ":")),
            media_type="application/json",
            headers={"cache-control": "no-store"},
        )

    @app.get("/viewer/call/{call_id}/image/{index}")
    async def viewer_call_image(call_id: int, index: int) -> Response:
        image = calls.image(call_id, index) if calls is not None else None
        if image is None:
            raise HTTPException(status_code=404, detail="call image is no longer buffered")
        media_type, data = image
        return Response(content=data, media_type=media_type, headers={"cache-control": "no-store"})


_env_file_value = env_file_value


def main() -> None:
    """Run the interception proxy on a fixed loopback interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--instance-token",
        default=None,
        help="opaque launcher token echoed only by /health to prevent stale-proxy attachment",
    )
    parser.add_argument(
        "--provider",
        choices=tuple(PROVIDERS),
        default=os.environ.get("PLVA_PROVIDER", "overshoot"),
        help="inference-provider preset (default: overshoot)",
    )
    parser.add_argument(
        "--credential-source",
        choices=CREDENTIAL_SOURCES,
        default=os.environ.get("PLVA_CREDENTIAL_SOURCE", "auto"),
        help="where to load the provider API key (default: auto)",
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
        "--ocr-engine",
        choices=("apple", "rapidocr"),
        default="apple",
        help="OCR engine for the Vision worker (default: apple)",
    )
    parser.add_argument(
        "--semantic-engine",
        choices=("rampart", "gliner2", "openai-pf"),
        default="rampart",
        help="local semantic PII engine for the Vision worker (default: rampart)",
    )
    parser.add_argument(
        "--visual-detector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable the visual detector alongside OCR (default: enabled)",
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
    parser.add_argument(
        "--privacy-history-scrub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scrub outbound text history through the vault and Rampart",
    )
    parser.add_argument(
        "--privacy-chips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="vault current-frame findings and paint their placeholder chips",
    )
    parser.add_argument(
        "--privacy-scheme",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="inject the static placeholder scheme system message",
    )
    parser.add_argument(
        "--privacy-duplicate-warning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="inject the static duplicate-placeholder warning",
    )
    parser.add_argument(
        "--privacy-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="attach the current frame's token/class manifest",
    )
    parser.add_argument(
        "--privacy-resolution",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resolve issued placeholders in executed response action fields",
    )
    parser.add_argument(
        "--privacy-policy-teaching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="teach the model the active per-class security policy",
    )
    parser.add_argument(
        "--privacy-policy-json",
        default=os.environ.get("PLVA_POLICY_JSON", ""),
        help="JSON object mapping PII classes to hide_use, approval, or blocked",
    )
    parser.add_argument(
        "--privacy-approval-ttl-seconds",
        type=float,
        default=float(os.environ.get("PLVA_APPROVAL_TTL_SECONDS", "60")),
        help="default lifetime of a local approval grant (default: 60 seconds)",
    )
    parser.add_argument(
        "--privacy-approval-use-count",
        type=int,
        default=int(os.environ.get("PLVA_APPROVAL_USE_COUNT", "1")),
        help="default number of exact action uses per local approval (default: 1)",
    )
    parser.add_argument(
        "--audit-capacity",
        type=int,
        default=int(os.environ.get("PLVA_AUDIT_CAPACITY", "32")),
        help="maximum redacted sent-frame records kept in memory (default: 32)",
    )
    parser.add_argument(
        "--capture-grammar",
        type=Path,
        default=None,
        help="write the first request's structured_outputs schema (never messages) to this file",
    )
    parser.add_argument(
        "--tools",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable the Step 6.5 local tool channel: teaching, detection, bounded tool loop",
    )
    parser.add_argument(
        "--tools-max-rounds",
        type=int,
        default=int(os.environ.get("PLVA_TOOLS_MAX_ROUNDS", "4")),
        help="maximum model-tool exchanges per runtime step (default: 4)",
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
    if args.audit_capacity < 1:
        parser.error("--audit-capacity must be positive")
    if not 0 < args.privacy_approval_ttl_seconds <= 3600:
        parser.error("--privacy-approval-ttl-seconds must be between 0 and 3600")
    if not 1 <= args.privacy_approval_use_count <= 100:
        parser.error("--privacy-approval-use-count must be between 1 and 100")
    if not 1 <= args.tools_max_rounds <= 8:
        parser.error("--tools-max-rounds must be between 1 and 8")
    if args.privacy and (args.redact is None or args.redact_engine != "vision"):
        parser.error("--privacy requires --redact with --redact-engine vision")
    try:
        raw_policy = json.loads(args.privacy_policy_json) if args.privacy_policy_json else {}
        if not isinstance(raw_policy, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in raw_policy.items()
        ):
            raise ValueError("policy must be an object of string values")
        privacy_policy = SafetyPolicy(raw_policy)
    except (ValueError, TypeError) as exc:
        parser.error(f"--privacy-policy-json is invalid: {exc}")
    credential_source: CredentialSource = args.credential_source
    api_key, _ = resolve_provider_key(
        provider=args.provider,
        source=credential_source,
        project_root=Path.cwd(),
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
    vault_store: SessionVault | None = None
    history_scrubber: HistoryScrubber | None = None
    frame_store: FrameStore | None = None
    call_store: CallStore | None = None
    cleanup_callbacks: tuple[Callable[[], None], ...] = ()
    startup_callbacks: tuple[Callable[[], None], ...] = ()
    if args.redact is not None:
        cli_path = args.redact / "bin" / "plva-v2.mjs" if args.redact.is_dir() else args.redact
        if not cli_path.is_file():
            parser.error(f"--redact CLI not found: {cli_path}")
        if args.redact_engine != "vision" and shutil.which("node") is None:
            parser.error("--redact requires node on PATH")
        frame_store = FrameStore(capacity=args.audit_capacity)
        call_store = CallStore(capacity=args.audit_capacity)
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
                        ocr_engine=args.ocr_engine,
                        visual_enabled=args.visual_detector,
                        semantic_engine=args.semantic_engine,
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
            detached_vault_cleanup: Callable[[], None] | None = None
            if args.privacy:
                vault = SessionVault(
                    policy=privacy_policy,
                    approval_ttl_seconds=args.privacy_approval_ttl_seconds,
                    approval_use_count=args.privacy_approval_use_count,
                )
                vault_store = vault
                if args.privacy_chips:
                    vaulted = VaultRedactor(accelerated, vault)
                    active_redactor = vaulted
                    lifecycle_owner = vaulted
                else:
                    detached_vault_cleanup = vault.dispose
                history_scrubber = HistoryScrubber(vault, accelerated.classify_texts)
                privacy_hooks = Hooks(
                    on_request=privacy_request_hook(
                        history_scrubber,
                        policy=privacy_policy,
                        history_scrub=args.privacy_history_scrub,
                        inject_scheme=args.privacy_scheme,
                        inject_duplicate_warning=args.privacy_duplicate_warning,
                        inject_manifest=args.privacy_manifest,
                        inject_policy=args.privacy_policy_teaching,
                    ),
                    on_response=(privacy_response_hook(vault) if args.privacy_resolution else None),
                )
            redact_hook = frame_redaction_hook(
                active_redactor,
                frame_store,
                include_placeholder_manifest=(
                    args.privacy and args.privacy_chips and args.privacy_manifest
                ),
            )
            if args.redact_lifecycle == "eager":
                startup_callbacks = (lifecycle_owner.start,)
            cleanup_callbacks = (lifecycle_owner.close,)
            if detached_vault_cleanup is not None:
                cleanup_callbacks += (detached_vault_cleanup,)
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
    tool_loop_instance: ToolLoop | None = None
    if args.tools:
        tool_loop_instance = ToolLoop(ToolRegistry(), max_rounds=args.tools_max_rounds)
        hooks = _combine_hooks(
            hooks, Hooks(on_request=tool_teaching_request_hook(tool_loop_instance))
        )
    if args.capture_grammar is not None:
        hooks = _combine_hooks(Hooks(on_request=grammar_capture_hook(args.capture_grammar)), hooks)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.privacy:
        _LOGGER.info(
            "privacy features chips=%s history_scrub=%s scheme=%s duplicate_warning=%s "
            "manifest=%s resolution=%s policy_teaching=%s",
            args.privacy_chips,
            args.privacy_history_scrub,
            args.privacy_scheme,
            args.privacy_duplicate_warning,
            args.privacy_manifest,
            args.privacy_resolution,
            args.privacy_policy_teaching,
        )
    if frame_store is not None:
        _LOGGER.info("viewer: http://127.0.0.1:%d/viewer", args.port)
    app_options: dict[str, Any] = {
        "hooks": hooks,
        "frame_store": frame_store,
        "call_store": call_store,
        "startup_callbacks": startup_callbacks,
        "cleanup_callbacks": cleanup_callbacks,
    }
    if vault_store is not None:
        app_options["vault"] = vault_store
    if history_scrubber is not None:
        app_options["scrubber"] = history_scrubber
    if tool_loop_instance is not None:
        app_options["tool_loop"] = tool_loop_instance
    uvicorn.run(
        create_app(
            ProxyConfig(
                upstream_base_url=upstream,
                api_key=api_key,
                instance_token=args.instance_token,
            ),
            **app_options,
        ),
        host=LOOPBACK_HOST,
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
