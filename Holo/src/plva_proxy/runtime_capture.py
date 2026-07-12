"""Loopback-only capture stub for HoloDesktop's closed runtime contract.

The stub validates that a request contains a decodable screenshot, retains only
non-sensitive schema metadata, and returns a non-executable ``answer`` action.
It never forwards, logs, or persists the request body or image.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import threading
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Final

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError

from plva_proxy.contract_probe import MODEL_ID

LOOPBACK_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 18080
MAX_IMAGE_BYTES: Final = 64 * 1024 * 1024
_ALLOWED_IMAGE_MEDIA_TYPES: Final = frozenset({"image/jpeg", "image/png", "image/webp"})
_ANSWER_CONTENT: Final = json.dumps(
    {
        "note": None,
        "thought": "The local transport contract is verified.",
        "tool_calls": [
            {
                "tool_name": "answer",
                "content": "PLVA local transport probe complete.",
            }
        ],
    },
    separators=(",", ":"),
)


class CaptureError(ValueError):
    """Raised when a runtime request cannot prove the interception contract."""


@dataclass(frozen=True, slots=True)
class CaptureSummary:
    """Non-sensitive metadata proving that an image traversed the base URL."""

    model: str
    stream: bool
    request_keys: tuple[str, ...]
    message_count: int
    message_roles: tuple[str, ...]
    image_count: int
    image_media_types: tuple[str, ...]
    image_byte_lengths: tuple[int, ...]
    image_dimensions: tuple[tuple[int, int], ...]
    has_structured_outputs: bool
    has_tools: bool


class CaptureState:
    """Thread-safe, memory-only capture state; the first summary is retained."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._first_summary: CaptureSummary | None = None
        self._capture_count = 0

    def record(self, summary: CaptureSummary) -> None:
        """Count a valid request while retaining metadata from only the first."""

        with self._lock:
            self._capture_count += 1
            if self._first_summary is None:
                self._first_summary = summary

    def snapshot(self) -> dict[str, object]:
        """Return JSON-safe metadata without request content or image bytes."""

        with self._lock:
            summary = self._first_summary
            return {
                "captured": summary is not None,
                "capture_count": self._capture_count,
                "summary": asdict(summary) if summary is not None else None,
            }


def _image_metadata(part: Mapping[str, Any]) -> tuple[str, int, tuple[int, int]]:
    image_url = part.get("image_url")
    url = image_url.get("url") if isinstance(image_url, Mapping) else image_url
    if not isinstance(url, str):
        raise CaptureError("image_url has no URL")

    header, separator, encoded = url.partition(",")
    if not separator or not header.startswith("data:") or not header.endswith(";base64"):
        raise CaptureError("screenshot is not an inline base64 data URL")
    media_type = header.removeprefix("data:").removesuffix(";base64").lower()
    if media_type not in _ALLOWED_IMAGE_MEDIA_TYPES:
        raise CaptureError("screenshot media type is not allowed")
    if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise CaptureError("screenshot exceeds the capture limit")

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CaptureError("screenshot base64 is invalid") from exc
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise CaptureError("screenshot size is invalid")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            dimensions = image.size
            image.verify()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise CaptureError("screenshot bytes are not a valid image") from exc
    if dimensions[0] < 1 or dimensions[1] < 1:
        raise CaptureError("screenshot dimensions are invalid")
    return media_type, len(image_bytes), dimensions


def summarize_request(document: Mapping[str, Any]) -> CaptureSummary:
    """Validate a Holo model request and discard all content after summarizing."""

    model = document.get("model")
    if model != MODEL_ID:
        raise CaptureError("runtime request did not use the selected model")
    stream = document.get("stream", False)
    if not isinstance(stream, bool):
        raise CaptureError("stream must be a boolean")
    messages = document.get("messages")
    if not isinstance(messages, list) or not messages:
        raise CaptureError("runtime request has no messages")

    roles: list[str] = []
    images: list[tuple[str, int, tuple[int, int]]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise CaptureError("runtime message is not an object")
        role = message.get("role")
        if not isinstance(role, str):
            raise CaptureError("runtime message has no role")
        roles.append(role)
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "image_url":
                images.append(_image_metadata(part))

    if not images:
        raise CaptureError("runtime request contains no decodable screenshot")
    return CaptureSummary(
        model=model,
        stream=stream,
        request_keys=tuple(sorted(str(key) for key in document)),
        message_count=len(messages),
        message_roles=tuple(roles),
        image_count=len(images),
        image_media_types=tuple(item[0] for item in images),
        image_byte_lengths=tuple(item[1] for item in images),
        image_dimensions=tuple(item[2] for item in images),
        has_structured_outputs="structured_outputs" in document,
        has_tools="tools" in document,
    )


def _completion(model: str) -> dict[str, Any]:
    return {
        "id": "plva-runtime-contract-probe",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": _ANSWER_CONTENT,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse_events(model: str) -> Iterator[bytes]:
    common = {
        "id": "plva-runtime-contract-probe",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
    }
    deltas: tuple[tuple[dict[str, Any], str | None], ...] = (
        ({"role": "assistant"}, None),
        ({"content": _ANSWER_CONTENT}, None),
        ({}, "stop"),
    )
    for delta, finish_reason in deltas:
        event = {
            **common,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode()
    yield b"data: [DONE]\n\n"


def create_app(state: CaptureState | None = None) -> FastAPI:
    """Create the local-only capture application with memory-only state."""

    capture_state = state if state is not None else CaptureState()
    app = FastAPI(title="PLVA runtime contract capture", docs_url=None, redoc_url=None)

    @app.get("/health")
    @app.get("/v1/health")
    async def health() -> dict[str, str]:
        # The closed vLLM provider health-checks <base-host>/health (any trailing
        # /v1 stripped) before the chat POST; answer 200 so the run does not block.
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "status": "ready"}],
        }

    @app.get("/_probe/status")
    async def status() -> dict[str, object]:
        return capture_state.snapshot()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        try:
            document = await request.json()
        except (UnicodeDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="invalid JSON request") from exc
        if not isinstance(document, Mapping):
            raise HTTPException(status_code=422, detail="runtime request rejected")
        try:
            summary = summarize_request(document)
        except CaptureError as exc:
            raise HTTPException(status_code=422, detail="runtime request rejected") from exc

        capture_state.record(summary)
        if summary.stream:
            return StreamingResponse(_sse_events(summary.model), media_type="text/event-stream")
        return JSONResponse(_completion(summary.model))

    return app


def main() -> None:
    """Run the capture stub on a fixed loopback interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    uvicorn.run(
        create_app(),
        host=LOOPBACK_HOST,
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
