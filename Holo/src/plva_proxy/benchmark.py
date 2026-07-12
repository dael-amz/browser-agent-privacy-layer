"""Controlled CUA privacy benchmark and independently verified live acceptance run.

The default benchmark is entirely local and deterministic.  It exercises the same request and
response hooks as the proxy, executes the resulting actions against a tiny form state machine,
and verifies the final state independently of the simulated agent.  ``--live`` serves an
equivalent form on loopback and invokes the real Holo launcher; it is deliberately spend-gated.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import statistics
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final
from urllib.parse import parse_qs

from PIL import Image, ImageDraw

from plva_proxy.privacy import (
    HistoryScrubber,
    SessionVault,
    StubRedactor,
    StubSpan,
    VaultRedactor,
    privacy_request_hook,
    privacy_response_hook,
)
from plva_proxy.proxy import frame_redaction_hook

SYNTHETIC_VALUE: Final = "alice.benchmark@example.com"
SYNTHETIC_TOKEN: Final = "EMAIL_1_a3f9"
_SOURCE_BOX: Final = (36, 54, 430, 94)
_TARGET_BOX: Final = (36, 142, 430, 182)


@dataclass(slots=True)
class FormState:
    """Minimal browser-form state owned by the local executor, never the agent."""

    source_email: str = SYNTHETIC_VALUE
    confirm_email: str = ""
    page: int = 1
    submitted: bool = False


@dataclass(frozen=True, slots=True)
class Verification:
    passed: bool
    assertions: dict[str, bool]


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    mode: str
    task_completed: bool
    runtime_reported_success: bool
    private_frame_exposures: int
    private_text_exposures: int
    actions_attempted: int
    actions_executed: int
    placeholder_actions_resolved: int
    median_interception_ms: float
    wall_time_ms: float
    verification: Verification
    live: bool = False
    error: str | None = None


def verify_form(state: FormState) -> Verification:
    """Verify the requested UI postcondition without consulting agent/runtime status."""

    assertions = {
        "submitted": state.submitted,
        "destination_matches_source": state.confirm_email == state.source_email,
        "destination_nonempty": bool(state.confirm_email),
    }
    return Verification(all(assertions.values()), assertions)


def _render_form(state: FormState) -> bytes:
    image = Image.new("RGB", (480, 240), "white")
    draw = ImageDraw.Draw(image)
    if state.page == 1:
        draw.text((36, 28), "Account email", fill="black")
        draw.rectangle(_SOURCE_BOX, outline="black")
        draw.text((44, 66), state.source_email, fill="black")
        draw.rectangle((36, 142, 130, 174), fill="#eeeeee", outline="black")
        draw.text((54, 152), "Continue", fill="black")
    else:
        draw.text((36, 28), "Confirm the email from the previous step", fill="black")
        draw.rectangle(_TARGET_BOX, outline="black")
        draw.text((44, 154), state.confirm_email, fill="black")
        draw.rectangle((36, 200, 130, 228), fill="#eeeeee", outline="black")
        draw.text((58, 208), "Submit", fill="black")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _request(frame: bytes, history: list[dict[str, Any]]) -> dict[str, Any]:
    encoded = base64.b64encode(frame).decode("ascii")
    return {
        "model": "local-deterministic-cua",
        "messages": [
            {"role": "system", "content": "Complete the visible form."},
            *history,
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Copy account email, then submit."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            },
        ],
        "stream": False,
    }


def _image_bytes(document: dict[str, Any]) -> bytes:
    part = document["messages"][-1]["content"][-1]
    url = part["image_url"]["url"]
    return base64.b64decode(url.partition(",")[2], validate=True)


def _agent_completion(step: int, *, privacy_on: bool) -> dict[str, Any]:
    action: dict[str, Any]
    if step == 0:
        action = {"tool_call": {"tool_name": "click", "selector": "#continue"}}
    else:
        content = SYNTHETIC_TOKEN if privacy_on else SYNTHETIC_VALUE
        action = {
            "thought": "Copy the observed account email into the confirmation field.",
            "tool_calls": [
                {
                    "tool_name": "write",
                    "selector": "#confirm-email",
                    "content": content,
                },
                {"tool_name": "click", "selector": "#submit"},
            ],
        }
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps(action)}}]}


def _execute(document: dict[str, Any], state: FormState) -> int:
    content = document.get("choices", [{}])[0].get("message", {}).get("content")
    if not isinstance(content, str):
        return 0
    try:
        payload = json.loads(content)
    except (KeyError, TypeError, json.JSONDecodeError):
        return 0
    actions = payload.get("tool_calls")
    if not isinstance(actions, list):
        single = payload.get("tool_call")
        actions = [single] if isinstance(single, dict) else []
    executed = 0
    for action in actions:
        if not isinstance(action, dict):
            continue
        if action.get("tool_name") == "click" and action.get("selector") == "#continue":
            state.page = 2
            executed += 1
        elif action.get("tool_name") == "write" and action.get("selector") == "#confirm-email":
            value = action.get("content")
            if isinstance(value, str):
                state.confirm_email = value
                executed += 1
        elif action.get("tool_name") == "click" and action.get("selector") == "#submit":
            state.submitted = True
            executed += 1
    return executed


def run_controlled(*, privacy_on: bool) -> BenchmarkResult:
    """Run one synthetic CUA task through the PLVA interception seam."""

    run_started = time.perf_counter()
    state = FormState()
    vault = SessionVault(nonce="a3f9")
    spans = (StubSpan("EMAIL", SYNTHETIC_VALUE, _SOURCE_BOX),)
    scrubber = HistoryScrubber(
        vault, lambda texts: [{"sensitive": False, "values": []} for _ in texts]
    )
    response_hook = privacy_response_hook(vault)
    history: list[dict[str, Any]] = []
    latencies: list[float] = []
    frame_exposures = 0
    text_exposures = 0
    executed = 0
    resolved = 0

    for step in range(2):
        raw_frame = _render_form(state)
        outbound = _request(raw_frame, history)
        started = time.perf_counter()
        if privacy_on:
            detector = StubRedactor(spans if state.page == 1 else ())
            vaulted = VaultRedactor(detector, vault, cache_entries=0)
            redact_hook = frame_redaction_hook(vaulted, include_placeholder_manifest=True)
            outbound, headers = redact_hook(outbound, {})
            outbound, _ = privacy_request_hook(scrubber)(outbound, headers)
        latencies.append((time.perf_counter() - started) * 1_000)
        provider_frame = _image_bytes(outbound)
        frame_exposures += int(state.page == 1 and provider_frame == raw_frame)
        text_exposures += int(SYNTHETIC_VALUE in json.dumps(outbound))

        completion = _agent_completion(step, privacy_on=privacy_on)
        if privacy_on:
            started = time.perf_counter()
            completion = response_hook(completion)
            latencies.append((time.perf_counter() - started) * 1_000)
            resolved += int(
                step == 1 and SYNTHETIC_VALUE in completion["choices"][0]["message"]["content"]
            )
        executed += _execute(completion, state)
        history.append(completion["choices"][0]["message"])

    verification = verify_form(state)
    return BenchmarkResult(
        mode="plva-on" if privacy_on else "plva-off",
        task_completed=verification.passed,
        runtime_reported_success=True,
        private_frame_exposures=frame_exposures,
        private_text_exposures=text_exposures,
        actions_attempted=3,
        actions_executed=executed,
        placeholder_actions_resolved=resolved,
        median_interception_ms=round(statistics.median(latencies), 3),
        wall_time_ms=round((time.perf_counter() - run_started) * 1_000, 3),
        verification=verification,
    )


def run_controlled_suite() -> list[BenchmarkResult]:
    return [run_controlled(privacy_on=False), run_controlled(privacy_on=True)]


def _fixture_html() -> bytes:
    return f"""<!doctype html><html><head><title>PLVA CUA acceptance</title></head><body>
<main><h1>Account verification</h1><p>Account email:</p>
<output id="account-email">{SYNTHETIC_VALUE}</output>
<a id="continue" href="/confirm">Continue</a></main></body></html>""".encode()


def _confirm_html() -> bytes:
    return b"""<!doctype html><html><head><title>Confirm account</title></head><body>
<main><h1>Confirm the email from the previous step</h1>
<form method="post" action="/submit"><label>Confirm email
<input id="confirm-email" name="confirm_email" type="email" autocomplete="off" required></label>
<button id="submit" type="submit">Submit</button></form></main></body></html>"""


class _Receipt:
    def __init__(self) -> None:
        self.submitted = threading.Event()
        self.visited_confirm = False
        self.matches = False


def _handler(receipt: _Receipt) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                body = _fixture_html()
            elif self.path == "/confirm":
                receipt.visited_confirm = True
                body = _confirm_html()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/submit":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            values = parse_qs(self.rfile.read(length).decode(), keep_blank_values=True)
            receipt.matches = values.get("confirm_email") == [SYNTHETIC_VALUE]
            receipt.submitted.set()
            body = b"<h1 id='success'>Accepted</h1>" if receipt.matches else b"Invalid"
            self.send_response(200 if receipt.matches else 422)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def run_live(*, privacy_on: bool, runner: Path, timeout_s: int = 300) -> BenchmarkResult:
    """Invoke Holo against a loopback fixture and verify its POST independently."""

    receipt = _Receipt()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(receipt))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    prompt = (
        f"Open {url} in the browser. Remember the visible Account email, click Continue, enter "
        "that email on the next page, click Submit, and stop only after Accepted is visible."
    )
    env = dict(os.environ)
    env.update(
        {
            "PLVA_PRIVACY": "1" if privacy_on else "0",
            "PLVA_REDACT": "1" if privacy_on else "0",
            "PLVA_AUDIT": "0",
        }
    )
    error: str | None = None
    return_code = -1
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(runner), prompt],
            cwd=runner.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return_code = completed.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        error = type(exc).__name__
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    submitted = receipt.submitted.is_set()
    verification = Verification(
        receipt.visited_confirm and submitted and receipt.matches,
        {
            "visited_private-free_step": receipt.visited_confirm,
            "submitted": submitted,
            "destination_matches_source": receipt.matches,
        },
    )
    return BenchmarkResult(
        mode="plva-on" if privacy_on else "plva-off",
        task_completed=verification.passed,
        runtime_reported_success=return_code == 0,
        private_frame_exposures=-1,
        private_text_exposures=-1,
        actions_attempted=-1,
        actions_executed=-1,
        placeholder_actions_resolved=-1,
        median_interception_ms=-1,
        wall_time_ms=round((time.perf_counter() - started) * 1_000, 3),
        verification=verification,
        live=True,
        error=error,
    )


def _serialize(results: Sequence[BenchmarkResult]) -> str:
    return json.dumps(
        {"schema_version": 1, "results": [asdict(item) for item in results]}, indent=2
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="invoke the real provider-backed CUA")
    parser.add_argument(
        "--allow-provider-spend",
        action="store_true",
        help="required acknowledgement for --live",
    )
    parser.add_argument(
        "--mode",
        choices=("on", "off", "both"),
        help="default: both for controlled runs, on for live runs",
    )
    parser.add_argument("--runner", type=Path, default=Path("run_step1.sh"))
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_s <= 0:
        raise SystemExit("--timeout-s must be positive")
    if args.mode is None:
        selected = [True] if args.live else [False, True]
    else:
        selected = [False, True] if args.mode == "both" else [args.mode == "on"]
    if args.live:
        if not args.allow_provider_spend:
            raise SystemExit("--live requires --allow-provider-spend")
        runner = args.runner.resolve()
        if not runner.is_file():
            raise SystemExit(f"runner not found: {runner}")
        results = [
            run_live(privacy_on=mode, runner=runner, timeout_s=args.timeout_s) for mode in selected
        ]
    else:
        results = [run_controlled(privacy_on=mode) for mode in selected]
    rendered = _serialize(results)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return int(any(not result.task_completed for result in results))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
