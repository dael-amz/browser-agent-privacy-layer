"""Live tool-channel probe for Step 6.5 — records booleans and schema, never content.

Talks to the provider directly (no runtime, no real screen): a synthetic
screenshot and a synthetic arithmetic task exercise each candidate invocation
channel, with and without the captured runtime grammar attached. Output is a
channel x (grammar, status, emits, parses, round_trip) matrix. Response text
is checked programmatically and never printed or persisted (§8.5).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import httpx
from PIL import Image

from plva_proxy.providers import PROVIDERS
from plva_proxy.tools import ToolError, ToolLoop, ToolRegistry, find_tool_call

PROBE_PROMPT: Final = (
    "This screen is blank. Use the add tool to compute 17 plus 25. After you "
    "receive the tool result, answer with only the sum."
)
EXPECTED_ANSWER: Final = "42"

STRUCTURED_TEACHING: Final = (
    "[PLVA_TOOLS] A local tool add(a, b) returns the sum of two numbers. To call "
    'it, emit exactly one action of the form {"tool_calls": [{"tool_name": '
    '"plva_tool", "name": "add", "args": {"a": <number>, "b": <number>}}]} and '
    "nothing else in that step. The next user message will begin with "
    "[PLVA_TOOL_RESULT] and carry the result; then answer with only the sum."
)
MARKER_TEACHING: Final = (
    "[PLVA_TOOLS] A local tool add(a, b) returns the sum of two numbers. To call "
    'it, write the single line ⟦TOOL⟧{"name": "add", "args": {"a": <number>, '
    '"b": <number>}}⟦/TOOL⟧ inside your answer text. The next user message will '
    "begin with [PLVA_TOOL_RESULT] and carry the result; then answer with only "
    "the sum."
)
CHANNELS: Final = (("structured", STRUCTURED_TEACHING), ("marker", MARKER_TEACHING))


def _synthetic_screenshot_data_url() -> str:
    image = Image.new("RGB", (64, 64), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def summarize_grammar(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """List enum constraints in the captured grammar and whether plva_tool fits."""

    enums: list[list[str]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            enum = node.get("enum")
            if isinstance(enum, list) and enum and all(isinstance(v, str) for v in enum):
                enums.append(sorted(enum))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(snapshot.get("structured_outputs"))
    admits = all("plva_tool" in enum for enum in enums) if enums else True
    return {"enum_sets": enums, "admits_plva_tool": admits}


def build_probe_request(
    model: str, teaching: str, grammar: Mapping[str, Any] | None
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": teaching},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROBE_PROMPT},
                    {"type": "image_url", "image_url": {"url": _synthetic_screenshot_data_url()}},
                ],
            },
        ],
    }
    if grammar is not None:
        request["structured_outputs"] = dict(grammar)
    return request


def _content_of(completion: Mapping[str, Any]) -> str | None:
    choices = completion.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
    return None


def evaluate_channel(
    client: httpx.Client,
    model: str,
    channel: str,
    teaching: str,
    grammar: Mapping[str, Any] | None,
) -> dict[str, Any]:
    request = build_probe_request(model, teaching, grammar)
    row: dict[str, Any] = {
        "channel": channel,
        "grammar_attached": grammar is not None,
        "status": 0,
        "emits": False,
        "parses": False,
        "round_trip": False,
    }
    response = client.post("/chat/completions", json=request)
    row["status"] = response.status_code
    if response.status_code != 200:
        return row
    completion = response.json()
    try:
        call = find_tool_call(completion)
    except ToolError:
        row["emits"] = True  # tool-shaped but unparseable
        return row
    if call is None or call.name != "add" or call.channel != channel:
        return row
    row["emits"] = True
    row["parses"] = True
    loop = ToolLoop(ToolRegistry())
    result = loop.execute(call)
    follow_request = loop.continuation(request, completion, call, result)
    follow = client.post("/chat/completions", json=follow_request)
    if follow.status_code != 200:
        return row
    final = follow.json()
    final_content = _content_of(final)
    try:
        residual = find_tool_call(final)
    except ToolError:
        return row
    row["round_trip"] = (
        final_content is not None and EXPECTED_ANSWER in final_content and residual is None
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=tuple(PROVIDERS), default="overshoot")
    parser.add_argument("--upstream", default=None, help="override the provider base URL")
    parser.add_argument("--grammar", type=Path, default=None, help="captured grammar snapshot")
    parser.add_argument("--out", type=Path, default=None, help="write the matrix JSON here")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="only summarize the grammar snapshot; no provider traffic",
    )
    args = parser.parse_args()

    snapshot: dict[str, Any] | None = None
    if args.grammar is not None:
        snapshot = json.loads(args.grammar.read_text())
        print(json.dumps(summarize_grammar(snapshot), indent=2))
    if args.analyze_only:
        return

    provider = PROVIDERS[args.provider]
    api_key = next((value for name in provider.key_names if (value := os.environ.get(name))), None)
    if not api_key:
        print(f"ERROR: {' or '.join(provider.key_names)} is required", file=sys.stderr)
        raise SystemExit(2)

    rows: list[dict[str, Any]] = []
    with httpx.Client(
        base_url=args.upstream or provider.base_url,
        headers={"authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(10.0, read=300.0),
    ) as client:
        grammars: tuple[Mapping[str, Any] | None, ...] = (None,)
        if snapshot is not None and snapshot.get("structured_outputs") is not None:
            grammars = (None, snapshot["structured_outputs"])
        for grammar in grammars:
            for channel, teaching in CHANNELS:
                row = evaluate_channel(client, provider.model, channel, teaching, grammar)
                rows.append(row)
                print(
                    f"channel={row['channel']} grammar={row['grammar_attached']} "
                    f"status={row['status']} emits={row['emits']} parses={row['parses']} "
                    f"round_trip={row['round_trip']}"
                )
    if args.out is not None:
        args.out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":  # pragma: no cover
    main()
