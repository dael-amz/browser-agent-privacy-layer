"""Privacy-safe empirical probe for Holo's model-to-local-tool channel.

The probe captures one authentic HoloDesktop request on loopback, keeps it only
in memory, replaces every user message and image with synthetic fixtures, and
tests three invocation conventions against the selected Holo provider.  It
retains only schema metadata and boolean outcomes; never request/response text,
desktop pixels, credentials, or tool arguments beyond the fixed synthetic
``add(3, 4)`` fixture.
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw

from plva_proxy.providers import PROVIDERS, ProviderSpec
from plva_proxy.runtime_capture import LOOPBACK_HOST

ROOT: Final = Path(__file__).resolve().parents[2]
PROBE_SKILL: Final = ROOT / "holo-skills" / "plva-tool-probe" / "SKILL.md"
REQUEST_ID: Final = "p65"
CHANNELS: Final = ("native_skill", "structured_action", "free_text_marker")
_MARKER: Final = re.compile(r"⟦PLVA_TOOL:add:(?P<a>\d+):(?P<b>\d+):(?P<id>[a-z0-9]+)⟧")
_ACK: Final = f"PLVA_TOOL_ACK:{REQUEST_ID}:7"
_SKILL_BEGIN: Final = "[PLVA_TOOL_PROBE_SKILL_BEGIN]"
_SKILL_END: Final = "[PLVA_TOOL_PROBE_SKILL_END]"


@dataclass(frozen=True, slots=True)
class Invocation:
    name: str
    a: int
    b: int
    request_id: str


@dataclass(frozen=True, slots=True)
class ChannelResult:
    grammar_permits: bool
    model_emitted: bool
    parser_accepted: bool
    round_trip: bool
    first_status: int | None
    second_status: int | None
    error_code: str


@dataclass(frozen=True, slots=True)
class ProbeReport:
    provider: str
    model: str
    runtime_request_captured: bool
    runtime_exit: int | None
    skill_loaded: bool
    schema: dict[str, Any]
    channels: dict[str, ChannelResult]
    recommendation: str
    fallback: str

    def json_value(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "channels": {name: asdict(result) for name, result in self.channels.items()},
        }


def _synthetic_image_url() -> str:
    image = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((28, 28, 612, 332), outline="black", width=3)
    draw.text((60, 105), "PLVA TOOL CHANNEL PROBE", fill="black")
    draw.text((60, 155), "Synthetic operation: add 3 and 4", fill="black")
    draw.text((60, 205), "No desktop data is present in this frame.", fill="black")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def schema_summary(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return a value-free structural summary of the runtime's output grammar."""

    schema = document.get("structured_outputs")
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":"), default=str).encode()
    property_paths: set[str] = set()
    string_fields: set[str] = set()
    action_enums: set[str] = set()
    tool_name_enums: set[str] = set()
    tool_name_unconstrained = False

    def walk(value: Any, path: str = "") -> None:
        nonlocal tool_name_unconstrained
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                for name, child in properties.items():
                    child_path = f"{path}.{name}" if path else str(name)
                    property_paths.add(child_path)
                    if isinstance(child, dict) and child.get("type") == "string":
                        string_fields.add(child_path)
                        if name == "tool_name":
                            choices = child.get("enum")
                            if isinstance(choices, list):
                                tool_name_enums.update(
                                    item
                                    for item in choices
                                    if isinstance(item, str) and len(item) <= 64
                                )
                            else:
                                tool_name_unconstrained = True
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key == "enum" and isinstance(child, list):
                    action_enums.update(
                        item for item in child if isinstance(item, str) and len(item) <= 64
                    )
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(schema)
    declared_tools = _declared_tool_names(document.get("tools"))
    return {
        "request_keys": sorted(str(key) for key in document),
        "has_structured_outputs": schema is not None,
        "has_tools": "tools" in document,
        "declared_tools": sorted(declared_tools),
        "schema_sha12": sha256(encoded).hexdigest()[:12],
        "property_paths": sorted(property_paths),
        "string_fields": sorted(string_fields),
        "action_enums": sorted(action_enums),
        "tool_name_enums": sorted(tool_name_enums),
        "tool_name_unconstrained": tool_name_unconstrained,
    }


def grammar_permits(channel: str, summary: Mapping[str, Any]) -> bool:
    if channel == "native_skill":
        return bool(summary.get("has_tools")) and "plva_add" in summary.get(
            "declared_tools", []
        )
    if channel == "structured_action":
        return bool(summary.get("tool_name_unconstrained")) or "plva_add" in summary.get(
            "tool_name_enums", []
        )
    if channel == "free_text_marker":
        return any(
            path.rsplit(".", 1)[-1]
            in {"thought", "note", "reasoning", "content", "text", "answer"}
            for path in summary.get("string_fields", [])
            if isinstance(path, str)
        )
    raise ValueError("unknown probe channel")


def parse_invocation(channel: str, response: Mapping[str, Any]) -> Invocation | None:
    message = _response_message(response)
    if message is None:
        return None
    if channel == "native_skill":
        candidates = message.get("tool_calls")
        calls = candidates if isinstance(candidates, list) else []
        function_call = message.get("function_call")
        if isinstance(function_call, dict):
            calls = [*calls, {"function": function_call}]
        return _parse_call_dicts(calls)
    content = message.get("content")
    if not isinstance(content, str):
        return None
    if channel == "free_text_marker":
        match = _MARKER.search(content)
        if match is None:
            return None
        return Invocation(
            "plva_add", int(match.group("a")), int(match.group("b")), match.group("id")
        )
    try:
        payload = json.loads(content)
    except ValueError:
        return None
    return _parse_call_dicts(_all_dicts(payload))


def model_emitted(channel: str, response: Mapping[str, Any]) -> bool:
    message = _response_message(response)
    if message is None:
        return False
    if channel == "native_skill":
        return bool(message.get("tool_calls") or message.get("function_call"))
    content = message.get("content")
    if not isinstance(content, str):
        return False
    if channel == "free_text_marker":
        return "⟦PLVA_TOOL" in content
    try:
        json.loads(content)
    except ValueError:
        return False
    return "plva_add" in content


def consumed_result(response: Mapping[str, Any]) -> bool:
    message = _response_message(response)
    return message is not None and _ACK in json.dumps(message, ensure_ascii=False)


class ProbeRunner:
    def __init__(
        self,
        *,
        provider_name: str,
        provider: ProviderSpec,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.provider = provider
        self.api_key = api_key
        self.transport = transport

    async def run(self, captured: Mapping[str, Any]) -> ProbeReport:
        summary = schema_summary(captured)
        skill_loaded = _SKILL_BEGIN in _system_text(captured)
        results: dict[str, ChannelResult] = {}
        headers = {"authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        async with httpx.AsyncClient(
            base_url=self.provider.base_url,
            headers=headers,
            timeout=httpx.Timeout(120.0),
            transport=self.transport,
        ) as client:
            for channel in CHANNELS:
                results[channel] = await self._run_channel(client, captured, summary, channel)
        recommendation = next(
            (name for name in CHANNELS if results[name].round_trip),
            "proxy_app_pseudo_tool",
        )
        return ProbeReport(
            provider=self.provider_name,
            model=self.provider.model,
            runtime_request_captured=True,
            runtime_exit=None,
            skill_loaded=skill_loaded,
            schema=summary,
            channels=results,
            recommendation=recommendation,
            fallback=(
                "proxy executes locally and injects a value-free result into the next observation"
            ),
        )

    async def _run_channel(
        self,
        client: httpx.AsyncClient,
        captured: Mapping[str, Any],
        summary: Mapping[str, Any],
        channel: str,
    ) -> ChannelResult:
        request = _synthetic_request(captured, self.provider.model, channel)
        first_status: int | None = None
        try:
            first = await client.post("/chat/completions", json=request)
            first_status = first.status_code
            if first.status_code != 200:
                return ChannelResult(
                    grammar_permits(channel, summary), False, False, False, first_status, None,
                    "first_http_error",
                )
            response = first.json()
            if not isinstance(response, dict):
                raise ValueError
        except (httpx.HTTPError, ValueError):
            return ChannelResult(
                grammar_permits(channel, summary), False, False, False, first_status, None,
                "first_response_invalid",
            )
        emitted = model_emitted(channel, response)
        invocation = parse_invocation(channel, response)
        accepted = invocation == Invocation("plva_add", 3, 4, REQUEST_ID)
        if not accepted:
            return ChannelResult(
                grammar_permits(channel, summary), emitted, False, False, first_status, None,
                "invocation_not_parseable",
            )
        assert invocation is not None
        followup = _round_trip_request(request, response, invocation.a + invocation.b)
        second_status: int | None = None
        try:
            second = await client.post("/chat/completions", json=followup)
            second_status = second.status_code
            if second.status_code != 200:
                return ChannelResult(
                    grammar_permits(channel, summary), emitted, True, False, first_status,
                    second_status, "second_http_error",
                )
            second_response = second.json()
            if not isinstance(second_response, dict):
                raise ValueError
        except (httpx.HTTPError, ValueError):
            return ChannelResult(
                grammar_permits(channel, summary), emitted, True, False, first_status,
                second_status, "second_response_invalid",
            )
        return ChannelResult(
            grammar_permits(channel, summary), emitted, True,
            consumed_result(second_response), first_status, second_status, "none",
        )


class ProbeState:
    def __init__(self, runner: ProbeRunner, runtime_model: str) -> None:
        self.runner = runner
        self.runtime_model = runtime_model
        self.complete = threading.Event()
        self.lock = threading.Lock()
        self.started = False
        self.report: ProbeReport | None = None
        self.error_code = "none"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "complete": self.complete.is_set(),
                "error_code": self.error_code,
                "report": self.report.json_value() if self.report is not None else None,
            }


def create_app(state: ProbeState) -> FastAPI:
    app = FastAPI(title="PLVA tool channel probe", docs_url=None, redoc_url=None)

    @app.get("/health")
    @app.get("/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": state.runtime_model, "object": "model"}]}

    @app.get("/_probe/status")
    async def status() -> dict[str, Any]:
        return state.snapshot()

    @app.post("/v1/chat/completions")
    async def completion(request: Request) -> JSONResponse:
        try:
            document = await request.json()
        except (UnicodeDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="invalid JSON request") from exc
        if not isinstance(document, dict):
            raise HTTPException(status_code=422, detail="runtime request rejected")
        with state.lock:
            if state.started:
                raise HTTPException(status_code=409, detail="probe already captured")
            state.started = True
        try:
            state.report = await state.runner.run(document)
        except Exception:
            # The public report deliberately exposes only a stable, value-free code.
            state.error_code = "probe_execution_failed"
        finally:
            state.complete.set()
        return JSONResponse(_runtime_answer(state.runtime_model))

    return app


def _synthetic_request(captured: Mapping[str, Any], model: str, channel: str) -> dict[str, Any]:
    request = copy.deepcopy(dict(captured))
    messages = request.get("messages")
    source_messages = messages if isinstance(messages, list) else []
    systems = [
        copy.deepcopy(message)
        for message in source_messages
        if isinstance(message, dict) and message.get("role") == "system"
    ]
    instruction = {
        "native_skill": (
            "Use the native PLVA Tool Probe skill to invoke plva_add with a=3, b=4, and "
            f"request_id={REQUEST_ID}. Do not calculate or answer directly."
        ),
        "structured_action": (
            "Emit a structured Holo action call named plva_add with integer arguments a=3, b=4, "
            f"and request_id={REQUEST_ID}. Do not calculate or answer directly."
        ),
        "free_text_marker": (
            "Emit the exact marker ⟦PLVA_TOOL:add:3:4:p65⟧ in an allowed thought, note, or "
            "content field. Do not calculate or answer directly."
        ),
    }[channel]
    request["messages"] = [
        *systems,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[PLVA_TOOL_PROBE] " + instruction},
                {"type": "image_url", "image_url": {"url": _synthetic_image_url()}},
            ],
        },
    ]
    request["model"] = model
    request["stream"] = False
    request["max_tokens"] = min(int(request.get("max_tokens", 1024)), 2048)
    return request


def _round_trip_request(
    request: Mapping[str, Any], response: Mapping[str, Any], result: int
) -> dict[str, Any]:
    followup = copy.deepcopy(dict(request))
    messages = followup.get("messages")
    if not isinstance(messages, list):
        raise ValueError("probe request omitted messages")
    message = _response_message(response)
    if message is None:
        raise ValueError("probe response omitted message")
    messages.append(copy.deepcopy(message))
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"[PLVA_TOOL_RESULT request_id={REQUEST_ID}] result={result}. "
                        f"Consume this result and emit the exact acknowledgement {_ACK}."
                    ),
                },
                {"type": "image_url", "image_url": {"url": _synthetic_image_url()}},
            ],
        }
    )
    return followup


def _response_message(response: Mapping[str, Any]) -> dict[str, Any] | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    return message if isinstance(message, dict) else None


def _all_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        found.append(value)
        for child in value.values():
            found.extend(_all_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_all_dicts(child))
    return found


def _parse_call_dicts(calls: list[Any]) -> Invocation | None:
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        body = function if isinstance(function, dict) else call
        name = body.get("name", call.get("tool_name", call.get("name")))
        if name != "plva_add":
            continue
        arguments = body.get("arguments", call.get("arguments", call.get("args", call)))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                continue
        if not isinstance(arguments, dict):
            continue
        try:
            return Invocation(
                "plva_add",
                int(arguments["a"]),
                int(arguments["b"]),
                str(arguments.get("request_id", "")),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _declared_tool_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            function = item.get("function")
            candidate = function.get("name") if isinstance(function, dict) else item.get("name")
            if isinstance(candidate, str):
                names.add(candidate)
    return names


def _system_text(document: Mapping[str, Any]) -> str:
    messages = document.get("messages")
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "system"
    )


def _runtime_answer(model: str) -> dict[str, Any]:
    content = json.dumps(
        {
            "note": None,
            "thought": "The synthetic tool-channel probe is complete.",
            "tool_calls": [{"tool_name": "answer", "content": "Probe complete."}],
        },
        separators=(",", ":"),
    )
    return {
        "id": "plva-tool-probe",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "tool_calls": None},
                "finish_reason": "stop",
            }
        ],
    }


def _env_value(path: Path, names: tuple[str, ...]) -> str | None:
    for name in names:
        if value := os.environ.get(name):
            return value
    try:
        lines = path.read_text("utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        for name in names:
            if line.startswith(name + "="):
                value = line.split("=", 1)[1].strip().strip("\"'")
                if value:
                    return value
    return None


@contextmanager
def _temporary_skill() -> Iterator[None]:
    target = Path.home() / ".holo" / "skills" / "plva-tool-probe" / "SKILL.md"
    previous = target.read_bytes() if target.is_file() else None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(PROBE_SKILL.read_bytes())
    try:
        yield
    finally:
        if previous is None:
            shutil.rmtree(target.parent, ignore_errors=True)
        else:
            target.write_bytes(previous)


def _run_runtime(port: int, model: str) -> int:
    uv = os.environ.get("UV") or shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    with tempfile.TemporaryDirectory(prefix="holo-tool-probe-runs.") as runs:
        command = [
            uv,
            "tool",
            "run",
            "--from",
            "holo-desktop-cli",
            "holo",
            "run",
            (
                "Use the PLVA Tool Probe skill for the synthetic add operation. "
                "Do not touch the desktop."
            ),
            "--base-url",
            f"http://{LOOPBACK_HOST}:{port}/v1",
            "--model",
            model,
            "--max-steps",
            "1",
            "--max-time-s",
            "180",
            "--no-kill-switch",
            "--runs-dir",
            runs,
        ]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
            return 124


def _recommendation(report: ProbeReport) -> str:
    return next(
        (channel for channel in CHANNELS if report.channels[channel].round_trip),
        "proxy_app_pseudo_tool",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=tuple(PROVIDERS), default="hcompany")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    provider = PROVIDERS[args.provider]
    key = _env_value(ROOT / ".env", provider.key_names)
    if key is None:
        parser.error("the selected provider key is unavailable")
    runner = ProbeRunner(
        provider_name=args.provider, provider=provider, api_key=key
    )
    state = ProbeState(runner, provider.model)
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(state), host=LOOPBACK_HOST, port=args.port, access_log=False,
            log_level="warning"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        parser.error("loopback probe server did not start")
    try:
        with _temporary_skill():
            runtime_exit = _run_runtime(args.port, provider.model)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    report = state.report
    if report is None:
        parser.error(f"runtime probe failed ({state.error_code})")
    report = ProbeReport(
        provider=report.provider,
        model=report.model,
        runtime_request_captured=report.runtime_request_captured,
        runtime_exit=runtime_exit,
        skill_loaded=report.skill_loaded,
        schema=report.schema,
        channels=report.channels,
        recommendation=_recommendation(report),
        fallback=report.fallback,
    )
    rendered = json.dumps(report.json_value(), sort_keys=True, separators=(",", ":"))
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":  # pragma: no cover
    main()
