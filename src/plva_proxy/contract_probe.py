"""Privacy-safe probes for Overshoot's external API contract.

The probe deliberately uses a one-pixel synthetic image and emits only schema
metadata. Model output is never printed or persisted.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Final

import httpx

from plva_proxy.providers import PROVIDERS

API_BASE_URL: Final = PROVIDERS["overshoot"].base_url
MODELS_URL: Final = f"{API_BASE_URL}/models"
COMPLETIONS_URL: Final = f"{API_BASE_URL}/chat/completions"
MODEL_ID: Final = "Hcompany/Holo3-35B-A3B"

_SYNTHETIC_PNG: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class ContractError(RuntimeError):
    """Raised when an external response violates the expected safe contract."""


@dataclass(frozen=True, slots=True)
class CompletionSummary:
    """Non-sensitive description of a chat-completion response shape."""

    response_keys: tuple[str, ...]
    choice_keys: tuple[str, ...]
    message_keys: tuple[str, ...]
    message_mode: str
    model: str | None
    finish_reason: str | None


@dataclass(frozen=True, slots=True)
class SSESummary:
    """Non-sensitive description of a streamed chat-completion response."""

    event_count: int
    done: bool
    delta_keys: tuple[str, ...]
    has_tool_call_delta: bool


def build_chat_payload(image_bytes: bytes, *, stream: bool) -> dict[str, Any]:
    """Build a synthetic Holo3 request without including credentials or real data."""

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": (
                    "This is a transport contract probe using a synthetic one-pixel image. "
                    "Return a single computer-use action as JSON; do not infer private data."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Return a no-op wait action."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            },
        ],
        "stream": stream,
        "temperature": 0,
        "max_tokens": 256,
    }


def find_ready_model(document: Mapping[str, Any], model_id: str = MODEL_ID) -> Mapping[str, Any]:
    """Return the exact advertised model only when its status is ready."""

    models = document.get("data")
    if not isinstance(models, list):
        raise ContractError("models response has no data list")
    for candidate in models:
        if isinstance(candidate, Mapping) and candidate.get("id") == model_id:
            if candidate.get("status") != "ready":
                raise ContractError(f"model {model_id} is not ready")
            return candidate
    raise ContractError(f"model {model_id} is not advertised")


def summarize_completion(document: Mapping[str, Any]) -> CompletionSummary:
    """Summarize a completion schema without retaining content or arguments."""

    choices = document.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ContractError("completion response has no first choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ContractError("completion response has no message object")

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        mode = "tool_calls"
    elif message.get("content") is not None:
        mode = "content"
    else:
        mode = "empty"

    model = document.get("model")
    finish_reason = choice.get("finish_reason")
    return CompletionSummary(
        response_keys=tuple(sorted(str(key) for key in document)),
        choice_keys=tuple(sorted(str(key) for key in choice)),
        message_keys=tuple(sorted(str(key) for key in message)),
        message_mode=mode,
        model=model if isinstance(model, str) else None,
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
    )


def summarize_sse(raw: bytes) -> SSESummary:
    """Summarize SSE delta keys, rejecting malformed JSON before reporting success."""

    event_count = 0
    done = False
    delta_keys: set[str] = set()
    has_tool_call_delta = False

    normalized = raw.replace(b"\r\n", b"\n")
    for event in normalized.split(b"\n\n"):
        data_lines = [line[5:].lstrip() for line in event.splitlines() if line.startswith(b"data:")]
        if not data_lines:
            continue
        payload = b"\n".join(data_lines)
        if payload == b"[DONE]":
            done = True
            continue
        try:
            document = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ContractError("invalid SSE JSON data event") from exc
        if not isinstance(document, Mapping):
            raise ContractError("SSE data event is not an object")
        event_count += 1
        choices = document.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                continue
            delta_keys.update(str(key) for key in delta)
            if "tool_calls" in delta:
                has_tool_call_delta = True

    if event_count == 0:
        raise ContractError("SSE response has no JSON data events")
    if not done:
        raise ContractError("SSE response ended before [DONE]")

    return SSESummary(
        event_count=event_count,
        done=done,
        delta_keys=tuple(sorted(delta_keys)),
        has_tool_call_delta=has_tool_call_delta,
    )


def _probe(*, api_key: str, stream: bool) -> CompletionSummary | SSESummary:
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=httpx.Timeout(60.0, read=120.0)) as client:
        models_response = client.get(MODELS_URL)
        models_response.raise_for_status()
        find_ready_model(models_response.json())

        response = client.post(
            COMPLETIONS_URL,
            headers=headers,
            json=build_chat_payload(_SYNTHETIC_PNG, stream=stream),
        )
        response.raise_for_status()
        if stream:
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" not in content_type:
                raise ContractError("streamed response is not text/event-stream")
            return summarize_sse(response.content)
        return summarize_completion(response.json())


def main() -> None:
    """Run a live synthetic probe and print only its non-sensitive shape summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", action="store_true", help="probe SSE instead of JSON")
    args = parser.parse_args()

    api_key = os.environ.get("API_KEY")
    if not api_key:
        parser.error("API_KEY is required in the environment")
    try:
        summary = _probe(api_key=api_key, stream=args.stream)
    except (ContractError, httpx.HTTPError, ValueError) as exc:
        print(f"contract probe failed: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(asdict(summary), sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
