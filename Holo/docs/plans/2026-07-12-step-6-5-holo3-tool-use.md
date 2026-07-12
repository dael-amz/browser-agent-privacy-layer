# Holo3 Tool Use (BLUEPRINT Step 6.5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Empirically settle *whether and how* the upstream `Hcompany/Holo3-35B-A3B` model can emit a tool call the PLVA proxy can parse, execute locally, and round-trip back so the model consumes the result — and ship the working invocation channel that Steps 7/8/9/10/13 will adopt.

**Architecture:** Holo3 has **no native `tools` support** — the runtime's request carries `structured_outputs` (a grammar constraint) and actions come back as structured JSON in `message.content` with a `tool_calls` array of `{tool_name, ...args}` objects (Step 0 evidence). The closed `hai-agent-runtime` executes actions and cannot execute a novel tool. Therefore the backbone of this plan is a **proxy inner loop**: when the model's completion contains a PLVA tool call (any emission channel), the proxy executes the tool locally, appends the model's turn plus a `[PLVA_TOOL_RESULT]` user message to the request it just sent, re-queries the provider, and repeats (bounded, fail-closed) until the model emits a runtime-executable action. The runtime only ever sees the final completion. Three *emission channels* are probed on top of that backbone, per the blueprint: (1) native skill-declared tool, (2) structured-action tool call, (3) free-text marker. A live probe records the blueprint's channels × (a) grammar-permits / (b) model-emits / (c) proxy-parses / (d) round-trip matrix and an ADR records the recommendation.

**Tech Stack:** Python 3 (uv-managed), FastAPI/httpx proxy (`Holo/src/plva_proxy/proxy.py`), pytest + httpx.MockTransport, HoloDesktop native skills (`~/.holo/skills/*/SKILL.md`), Overshoot / H Company OpenAI-compatible providers.

## Global Constraints

- Model is exactly `Hcompany/Holo3-35B-A3B` on Overshoot (preset `overshoot`) or `holo3-1-35b-a3b` on H Company (preset `hcompany`); no other model or harness (BLUEPRINT §3).
- **Fail closed** (§8.1): any hook/parse/tool failure forwards nothing; no raw fallback.
- **Privacy-safe logs** (§8.5): logs may carry tool names, channel names, arg *keys*, counts, statuses, durations — never values, frames, transcripts, or response content. Spike tool inputs are synthetic and non-sensitive, but logs stay value-free anyway.
- **Single system message**: the Holo chat template rejects two consecutive system messages (HTTP 400, Step 5a evidence). All injected teaching merges into the existing system message.
- Forward unknown request keys verbatim (`chat_template_kwargs`, `logit_bias`, …) — Step 0 contract.
- Nothing sensitive is recorded by the spike: the grammar snapshot contains schema only (never `messages`); verification docs carry booleans/schema, never content (BLUEPRINT Step 6.5).
- Quality gates (run from `Holo/`): `.venv/bin/pytest -q` (≥80% coverage), `.venv/bin/ruff format --check src tests`, `.venv/bin/ruff check .`, `.venv/bin/mypy src tests`, `~/.local/bin/uv lock --check`.
- Never commit API keys, screenshots, transcripts, or vault contents. Real runs relocate `--runs-dir` to an ephemeral path and shred it; `~/.holo/runs` is off-limits.
- Update `PROJECT_MAP.md` in the same change as any structural work.

**Working directory for all commands: `Holo/`.**

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/plva_proxy/tools.py` | Create | Tool registry (`echo`/`add`/`sort`), channel detection, teaching injection, bounded `ToolLoop` |
| `src/plva_proxy/proxy.py` | Modify | Grammar-snapshot hook, `tool_loop` relay integration, `--tools*` / `--capture-grammar` flags |
| `src/plva_proxy/tool_probe.py` | Create | Privacy-safe live channel probe CLI (`plva-tool-probe`) |
| `holo-skills/plva-tools/SKILL.md` | Create | Native skill-declared tool teaching (channel 1) |
| `run_step1.sh` | Modify | `PLVA_TOOLS`, `PLVA_TOOLS_SKILL`, `PLVA_CAPTURE_GRAMMAR` wiring |
| `pyproject.toml` | Modify | `plva-tool-probe` console script |
| `tests/test_tools.py` | Create | Registry, detection, teaching, loop-mechanics tests |
| `tests/test_tool_loop.py` | Create | Relay integration + grammar-capture tests |
| `tests/test_tool_probe.py` | Create | Probe pure-function and MockTransport tests |
| `docs/decisions/0002-tool-call-channel.md` | Create (Task 8) | Channel recommendation ADR |
| `verification/step-6-5-tool-channel.md` | Create (Task 8) | Recorded channels × (a–d) matrix |
| `BLUEPRINT.md`, `PROJECT_MAP.md` | Modify (Task 8) | Status + structure updates |

## Design Decisions (locked in)

1. **Round-trip = proxy inner loop.** The runtime cannot execute or relay a tool result; the proxy re-queries the provider itself. Continuation requests force `"stream": false` and append `{"role": "assistant", "content": <model's raw content>}` + `{"role": "user", "content": "[PLVA_TOOL_RESULT] …"}`. This is the only shape that makes (d) independent of the closed runtime.
2. **Structured channel wire shape:** `{"tool_calls": [{"tool_name": "plva_tool", "name": "<tool>", "args": {…}}]}` — reuses the exact envelope the runtime/model already produce, with one reserved `tool_name`.
3. **Marker channel wire shape:** `⟦TOOL⟧{"name": "<tool>", "args": {…}}⟦/TOOL⟧` with an explicit close marker (no brace-balancing), scanned in every string of the action document (thought, answer, or raw non-JSON content).
4. **Cross-step amnesia is handled by result memory.** The runtime rebuilds history from its own record, so inner-loop turns vanish from the next runtime step. The teaching hook re-injects a bounded "results already computed this session" list so multi-step tasks keep tool outcomes. (Spike results are non-sensitive; Step 13 must apply its token-only contract to this memory before sensitive tools exist.)
5. **Malformed-but-identifiable tool intents fold into an error result** fed back to the model (bounded by max rounds); *unidentifiable* tool-shaped output (`plva_tool` with no string name, unparseable marker payload) raises `ToolError` and fails closed — it must never reach the runtime's executor.
6. **Hook order:** grammar-capture hook first (sees the runtime's raw request), then redaction/privacy hooks, then tool teaching (privacy's system-message merge guarantees at most one system message exists for teaching to append to). Tool loop runs **before** `privacy_response_hook` so placeholder resolution applies only to the final completion the runtime will execute.

---

### Task 1: Grammar snapshot hook (`--capture-grammar`)

The `structured_outputs` value in the runtime's request *is* the action grammar. Capturing it answers channel question (a) — e.g. whether `tool_name` is enum-constrained — without recording anything sensitive.

**Files:**
- Modify: `src/plva_proxy/proxy.py` (new function after `image_replacement_hook`, ~line 270; new flag in `main()`)
- Test: `tests/test_tool_loop.py` (new file)

**Interfaces:**
- Produces: `grammar_capture_hook(out_path: Path) -> RequestHook` — writes `{"request_keys", "model", "structured_outputs", "chat_template_kwargs"}` JSON to `out_path` on the first request only; passes the document through untouched. Task 7's `summarize_grammar` consumes the file.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tool_loop.py`:

```python
"""Step 6.5: relay tool-loop integration and grammar capture."""

from __future__ import annotations

import json
from pathlib import Path

from plva_proxy.proxy import grammar_capture_hook


def test_grammar_capture_writes_schema_only(tmp_path: Path) -> None:
    out = tmp_path / "grammar.json"
    hook = grammar_capture_hook(out)
    document = {
        "model": "Hcompany/Holo3-35B-A3B",
        "messages": [{"role": "user", "content": "SECRET TEXT"}],
        "structured_outputs": {"json_schema": {"properties": {"tool_name": {"enum": ["click"]}}}},
        "chat_template_kwargs": {"reasoning": True},
    }
    returned, headers = hook(document, {"x-h": "1"})
    assert returned == document
    assert headers == {"x-h": "1"}
    snapshot = json.loads(out.read_text())
    assert snapshot["model"] == "Hcompany/Holo3-35B-A3B"
    assert snapshot["structured_outputs"]["json_schema"]["properties"]["tool_name"]["enum"] == [
        "click"
    ]
    assert "messages" not in snapshot
    assert "SECRET TEXT" not in out.read_text()
    assert sorted(snapshot["request_keys"]) == snapshot["request_keys"]


def test_grammar_capture_only_first_request(tmp_path: Path) -> None:
    out = tmp_path / "grammar.json"
    hook = grammar_capture_hook(out)
    hook({"model": "first", "structured_outputs": {}}, {})
    hook({"model": "second", "structured_outputs": {}}, {})
    assert json.loads(out.read_text())["model"] == "first"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_tool_loop.py -v`
Expected: FAIL — `ImportError: cannot import name 'grammar_capture_hook'`

- [ ] **Step 3: Implement the hook**

In `src/plva_proxy/proxy.py`, after `image_replacement_hook` (below its closing `return replace` / before `@dataclass class FrameRecord`):

```python
def grammar_capture_hook(out_path: Path) -> RequestHook:
    """Snapshot the first request's action grammar to a file — schema only, never messages.

    Step 6.5 needs the runtime's ``structured_outputs`` value to judge whether a
    novel tool action is even grammatically admissible. The snapshot records
    request keys, model id, ``structured_outputs``, and ``chat_template_kwargs``
    and deliberately nothing else (§8.5): the message history — the only part
    that can carry frames or values — is never written.
    """

    captured = threading.Event()

    def capture(
        document: dict[str, Any], headers: dict[str, str]
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if not captured.is_set():
            captured.set()
            snapshot = {
                "request_keys": sorted(document),
                "model": document.get("model"),
                "structured_outputs": document.get("structured_outputs"),
                "chat_template_kwargs": document.get("chat_template_kwargs"),
            }
            out_path.write_text(json.dumps(snapshot, indent=2) + "\n")
            _LOGGER.info("grammar snapshot written: keys=%d", len(snapshot["request_keys"]))
        return document, headers

    return capture
```

(`threading`, `json`, `Path`, `_LOGGER`, `RequestHook` are already imported/defined in this module.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tool_loop.py -v`
Expected: 2 passed

- [ ] **Step 5: Wire the CLI flag**

In `main()` add after the `--audit-capacity` argument:

```python
    parser.add_argument(
        "--capture-grammar",
        type=Path,
        default=None,
        help="write the first request's structured_outputs schema (never messages) to this file",
    )
```

And immediately after the line `hooks = _combine_hooks(hooks, privacy_hooks)`:

```python
    if args.capture_grammar is not None:
        hooks = _combine_hooks(Hooks(on_request=grammar_capture_hook(args.capture_grammar)), hooks)
```

(Placing the capture hook *first* in the chain records the runtime's raw request shape, pre-redaction — safe because only schema keys are written.)

- [ ] **Step 6: Run gates and commit**

Run: `.venv/bin/pytest -q && .venv/bin/ruff format src tests && .venv/bin/ruff check . && .venv/bin/mypy src tests`
Expected: all pass

```bash
git add src/plva_proxy/proxy.py tests/test_tool_loop.py
git commit -m "feat(step6.5): capture the runtime's structured_outputs grammar, schema-only"
```

---

### Task 2: Tool registry (`tools.py` — `echo`, `add`, `sort`)

**Files:**
- Create: `src/plva_proxy/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: `ToolError(RuntimeError)`; `ToolCall(name: str, args: Mapping[str, Any], channel: str)` frozen dataclass; `ToolRegistry` with `names() -> tuple[str, ...]` and `execute(call: ToolCall) -> str`.
- Constants: `STRUCTURED_TOOL_NAME = "plva_tool"`, `TOOL_MARKER_BEGIN = "⟦TOOL⟧"`, `TOOL_MARKER_END = "⟦/TOOL⟧"`, `TOOL_RESULT_PREFIX = "[PLVA_TOOL_RESULT]"`, `TOOL_SYSTEM_BEGIN = "[PLVA_TOOLS_BEGIN]"`, `TOOL_SYSTEM_END = "[PLVA_TOOLS_END]"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tools.py`:

```python
"""Step 6.5: local tool registry, channel detection, teaching, and loop mechanics."""

from __future__ import annotations

import pytest

from plva_proxy.tools import ToolCall, ToolError, ToolRegistry


def _call(name: str, args: dict[str, object]) -> ToolCall:
    return ToolCall(name=name, args=args, channel="structured")


def test_echo_returns_text() -> None:
    assert ToolRegistry().execute(_call("echo", {"text": "hello"})) == "hello"


def test_add_returns_integer_sum_as_string() -> None:
    assert ToolRegistry().execute(_call("add", {"a": 17, "b": 25})) == "42"


def test_add_returns_float_sum_when_fractional() -> None:
    assert ToolRegistry().execute(_call("add", {"a": 1.5, "b": 1})) == "2.5"


def test_sort_returns_comma_joined_ascending() -> None:
    result = ToolRegistry().execute(_call("sort", {"items": ["pear", "apple", "mango"]}))
    assert result == "apple, mango, pear"


def test_registry_names_are_stable() -> None:
    assert ToolRegistry().names() == ("add", "echo", "sort")


@pytest.mark.parametrize(
    "name,args",
    [
        ("echo", {}),
        ("echo", {"text": 5}),
        ("add", {"a": "1", "b": 2}),
        ("add", {"a": True, "b": 2}),
        ("sort", {"items": "abc"}),
        ("sort", {"items": ["a", 1]}),
        ("launch_missiles", {}),
    ],
)
def test_invalid_invocations_raise_tool_error(name: str, args: dict[str, object]) -> None:
    with pytest.raises(ToolError):
        ToolRegistry().execute(_call(name, args))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'plva_proxy.tools'`

- [ ] **Step 3: Implement the module**

Create `src/plva_proxy/tools.py`:

```python
"""Local tool channel for the Step 6.5 spike: registry, detection, teaching, and loop.

Holo3 exposes no native ``tools`` support (Step 0), so a "tool call" here is a
convention the proxy teaches and parses: either a reserved ``plva_tool`` entry
inside the model's existing structured ``tool_calls`` action envelope, or an
explicit ``⟦TOOL⟧…⟦/TOOL⟧`` marker inside free text. Execution is local and
deterministic; results are fed back through a bounded proxy inner loop. Logs
carry tool names, channels, and argument *keys* only — never values (§8.5).
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final

STRUCTURED_TOOL_NAME: Final = "plva_tool"
TOOL_MARKER_BEGIN: Final = "⟦TOOL⟧"
TOOL_MARKER_END: Final = "⟦/TOOL⟧"
TOOL_RESULT_PREFIX: Final = "[PLVA_TOOL_RESULT]"
TOOL_SYSTEM_BEGIN: Final = "[PLVA_TOOLS_BEGIN]"
TOOL_SYSTEM_END: Final = "[PLVA_TOOLS_END]"
_MARKER_PATTERN: Final = re.compile(
    re.escape(TOOL_MARKER_BEGIN) + r"(?P<payload>.*?)" + re.escape(TOOL_MARKER_END),
    re.DOTALL,
)

_LOGGER: Final = logging.getLogger(__name__)


class ToolError(RuntimeError):
    """Raised when a tool invocation is malformed or cannot run; fails closed."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One parsed tool invocation and the channel it arrived on."""

    name: str
    args: Mapping[str, Any]
    channel: str


def _echo(args: Mapping[str, Any]) -> str:
    text = args.get("text")
    if not isinstance(text, str):
        raise ToolError("echo requires a string 'text'")
    return text


def _add(args: Mapping[str, Any]) -> str:
    a = args.get("a")
    b = args.get("b")
    if (
        isinstance(a, bool)
        or isinstance(b, bool)
        or not isinstance(a, (int, float))
        or not isinstance(b, (int, float))
    ):
        raise ToolError("add requires numeric 'a' and 'b'")
    total = a + b
    return str(int(total)) if float(total).is_integer() else str(total)


def _sort(args: Mapping[str, Any]) -> str:
    items = args.get("items")
    if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
        raise ToolError("sort requires a list of strings 'items'")
    return ", ".join(sorted(items))


class ToolRegistry:
    """Deterministic local tools; execution never touches the network."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[[Mapping[str, Any]], str]] = {
            "echo": _echo,
            "add": _add,
            "sort": _sort,
        }

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def execute(self, call: ToolCall) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            raise ToolError(f"unknown tool: {call.name}")
        return tool(call.args)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: 13 passed

- [ ] **Step 5: Run gates and commit**

Run: `.venv/bin/pytest -q && .venv/bin/ruff format src tests && .venv/bin/ruff check . && .venv/bin/mypy src tests`
Expected: all pass

```bash
git add src/plva_proxy/tools.py tests/test_tools.py
git commit -m "feat(step6.5): deterministic local tool registry (echo, add, sort)"
```

---

### Task 3: Channel detection (`find_tool_call`)

**Files:**
- Modify: `src/plva_proxy/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: `find_tool_call(completion: dict[str, Any]) -> ToolCall | None` — scans an assembled completion document. Returns the first structured `plva_tool` call, else the first `⟦TOOL⟧…⟦/TOOL⟧` marker (in JSON-action strings or raw text content). Raises `ToolError` on tool-shaped output it cannot parse (fail closed). Returns `None` for ordinary actions/answers.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tools.py`:

```python
import json

from plva_proxy.tools import find_tool_call


def _completion(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_detects_structured_plva_tool_call() -> None:
    action = {
        "thought": "I need the sum.",
        "tool_calls": [{"tool_name": "plva_tool", "name": "add", "args": {"a": 17, "b": 25}}],
    }
    call = find_tool_call(_completion(json.dumps(action)))
    assert call is not None
    assert (call.name, call.channel) == ("add", "structured")
    assert call.args == {"a": 17, "b": 25}


def test_ignores_ordinary_runtime_actions() -> None:
    action = {"tool_calls": [{"tool_name": "click", "x": 10, "y": 20}]}
    assert find_tool_call(_completion(json.dumps(action))) is None


def test_ignores_plain_text_answers() -> None:
    assert find_tool_call(_completion("The sum is 42.")) is None


def test_detects_marker_inside_action_thought() -> None:
    action = {
        "thought": 'Delegating: ⟦TOOL⟧{"name": "sort", "args": {"items": ["b", "a"]}}⟦/TOOL⟧',
        "tool_calls": [{"tool_name": "wait"}],
    }
    call = find_tool_call(_completion(json.dumps(action)))
    assert call is not None
    assert (call.name, call.channel) == ("sort", "marker")


def test_detects_marker_in_raw_text_content() -> None:
    call = find_tool_call(_completion('⟦TOOL⟧{"name": "echo", "args": {"text": "hi"}}⟦/TOOL⟧'))
    assert call is not None
    assert (call.name, call.channel) == ("echo", "marker")


def test_structured_call_without_name_fails_closed() -> None:
    action = {"tool_calls": [{"tool_name": "plva_tool", "args": {"a": 1}}]}
    with pytest.raises(ToolError):
        find_tool_call(_completion(json.dumps(action)))


def test_marker_with_invalid_payload_fails_closed() -> None:
    with pytest.raises(ToolError):
        find_tool_call(_completion("⟦TOOL⟧not json⟦/TOOL⟧"))


def test_unknown_tool_name_is_returned_for_error_folding() -> None:
    action = {"tool_calls": [{"tool_name": "plva_tool", "name": "nope", "args": {}}]}
    call = find_tool_call(_completion(json.dumps(action)))
    assert call is not None and call.name == "nope"


def test_non_dict_args_default_to_empty() -> None:
    action = {"tool_calls": [{"tool_name": "plva_tool", "name": "echo", "args": "hi"}]}
    call = find_tool_call(_completion(json.dumps(action)))
    assert call is not None and call.args == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_tools.py -v -k "detect or ignores or fails_closed or unknown_tool or non_dict"`
Expected: FAIL — `ImportError: cannot import name 'find_tool_call'`

- [ ] **Step 3: Implement detection**

Append to `src/plva_proxy/tools.py`:

```python
def _iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)


def _find_structured(action: Mapping[str, Any]) -> ToolCall | None:
    calls = action.get("tool_calls")
    if not isinstance(calls, list):
        return None
    for call in calls:
        if not isinstance(call, dict) or call.get("tool_name") != STRUCTURED_TOOL_NAME:
            continue
        name = call.get("name")
        if not isinstance(name, str) or not name:
            raise ToolError("plva_tool call has no tool name")
        args = call.get("args", {})
        return ToolCall(
            name=name, args=args if isinstance(args, dict) else {}, channel="structured"
        )
    return None


def _find_marker(source: Any) -> ToolCall | None:
    for text in _iter_strings(source):
        match = _MARKER_PATTERN.search(text)
        if match is None:
            continue
        try:
            payload = json.loads(match.group("payload"))
        except ValueError as exc:
            raise ToolError("tool marker payload is not JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            raise ToolError("tool marker payload has no tool name")
        args = payload.get("args", {})
        return ToolCall(
            name=payload["name"], args=args if isinstance(args, dict) else {}, channel="marker"
        )
    return None


def find_tool_call(completion: dict[str, Any]) -> ToolCall | None:
    """Return the first PLVA tool invocation in a completion, if any.

    Structured ``plva_tool`` entries win over free-text markers. Tool-shaped
    output that cannot be parsed raises so it never reaches the runtime's
    executor (§8.1); ordinary actions and answers return ``None`` untouched.
    """

    choices = completion.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            continue
        action: Any = None
        try:
            action = json.loads(content)
        except ValueError:
            action = None
        if isinstance(action, dict):
            structured = _find_structured(action)
            if structured is not None:
                return structured
        marker = _find_marker(action if isinstance(action, dict) else content)
        if marker is not None:
            return marker
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: all pass (22 tests)

- [ ] **Step 5: Run gates and commit**

Run: `.venv/bin/pytest -q && .venv/bin/ruff format src tests && .venv/bin/ruff check . && .venv/bin/mypy src tests`
Expected: all pass

```bash
git add src/plva_proxy/tools.py tests/test_tools.py
git commit -m "feat(step6.5): detect structured and marker tool-call channels, fail-closed"
```

---

### Task 4: `ToolLoop` mechanics (execute with error-folding, memory, continuation)

**Files:**
- Modify: `src/plva_proxy/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: `ToolLoop(registry: ToolRegistry, *, max_rounds: int = 4, memory_capacity: int = 8)` with:
  - `max_rounds: int` attribute (read by the relay),
  - `detect(completion: dict[str, Any]) -> ToolCall | None` (delegates to `find_tool_call`),
  - `execute(call: ToolCall) -> str` — runs the tool; a `ToolError` from the registry folds into the string `"error: <reason>"` (fed back to the model, bounded by rounds); records `"<name>: <result>"` in bounded memory; logs name/channel/arg-keys only,
  - `memory() -> tuple[str, ...]`,
  - `continuation(request_document, completion, call, result) -> dict[str, Any]` — deep-copies the request, forces `"stream": False`, appends the assistant turn verbatim and a `[PLVA_TOOL_RESULT]` user message.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tools.py`:

```python
from plva_proxy.tools import ToolLoop


def test_loop_execute_returns_result_and_records_memory() -> None:
    loop = ToolLoop(ToolRegistry())
    result = loop.execute(_call("add", {"a": 17, "b": 25}))
    assert result == "42"
    assert loop.memory() == ("add: 42",)


def test_loop_execute_folds_tool_errors_into_result() -> None:
    loop = ToolLoop(ToolRegistry())
    result = loop.execute(_call("nope", {}))
    assert result.startswith("error: ")
    assert "unknown tool" in result


def test_loop_memory_is_bounded() -> None:
    loop = ToolLoop(ToolRegistry(), memory_capacity=2)
    for value in ("a", "b", "c"):
        loop.execute(_call("echo", {"text": value}))
    assert loop.memory() == ("echo: b", "echo: c")


def test_continuation_appends_turns_and_disables_streaming() -> None:
    loop = ToolLoop(ToolRegistry())
    request = {
        "model": "m",
        "stream": True,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": "task"}],
    }
    completion = _completion('{"tool_calls": [{"tool_name": "plva_tool", "name": "add"}]}')
    call = _call("add", {"a": 17, "b": 25})
    follow = loop.continuation(request, completion, call, "42")
    assert request["messages"] == [{"role": "user", "content": "task"}]  # original untouched
    assert follow["stream"] is False
    assert follow["temperature"] == 0.1  # unknown keys preserved verbatim
    assert follow["messages"][1]["role"] == "assistant"
    assert follow["messages"][1]["content"] == completion["choices"][0]["message"]["content"]
    assert follow["messages"][2]["role"] == "user"
    assert follow["messages"][2]["content"].startswith("[PLVA_TOOL_RESULT] add returned: 42")


def test_continuation_without_assistant_content_fails_closed() -> None:
    loop = ToolLoop(ToolRegistry())
    with pytest.raises(ToolError):
        loop.continuation({"messages": []}, {"choices": []}, _call("echo", {"text": "x"}), "x")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_tools.py -v -k loop_or_continuation or true` — use: `.venv/bin/pytest tests/test_tools.py -v -k "loop or continuation"`
Expected: FAIL — `ImportError: cannot import name 'ToolLoop'`

- [ ] **Step 3: Implement `ToolLoop`**

Append to `src/plva_proxy/tools.py`:

```python
class ToolLoop:
    """Bounded local execution loop shared by the relay and the live probe.

    Holds no vault data: spike tools are synthetic and non-sensitive. The
    memory list re-teaches results across runtime steps (the runtime rebuilds
    history from its own record, so inner-loop turns otherwise vanish).
    Step 13 must gate this memory through its token-only contract before any
    tool can touch a real value.
    """

    def __init__(
        self, registry: ToolRegistry, *, max_rounds: int = 4, memory_capacity: int = 8
    ) -> None:
        self._registry = registry
        self.max_rounds = max_rounds
        self._memory: deque[str] = deque(maxlen=memory_capacity)
        self._lock = threading.Lock()

    def detect(self, completion: dict[str, Any]) -> ToolCall | None:
        return find_tool_call(completion)

    def execute(self, call: ToolCall) -> str:
        try:
            result = self._registry.execute(call)
        except ToolError as exc:
            result = f"error: {exc}"
        with self._lock:
            self._memory.append(f"{call.name}: {result}")
        _LOGGER.info(
            "tool executed: name=%s channel=%s arg_keys=%s",
            call.name,
            call.channel,
            sorted(call.args),
        )
        return result

    def memory(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._memory)

    def continuation(
        self,
        request_document: dict[str, Any],
        completion: dict[str, Any],
        call: ToolCall,
        result: str,
    ) -> dict[str, Any]:
        content: str | None = None
        choices = completion.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                content = message["content"]
        if content is None:
            raise ToolError("tool continuation has no assistant content")
        document = copy.deepcopy(request_document)
        document["stream"] = False
        messages = document.get("messages")
        if not isinstance(messages, list):
            raise ToolError("tool continuation has no message history")
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{TOOL_RESULT_PREFIX} {call.name} returned: {result}\n"
                    "Use this result to continue the task. Do not repeat the same tool call."
                ),
            }
        )
        return document
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: all pass (27 tests)

- [ ] **Step 5: Run gates and commit**

Run: `.venv/bin/pytest -q && .venv/bin/ruff format src tests && .venv/bin/ruff check . && .venv/bin/mypy src tests`
Expected: all pass

```bash
git add src/plva_proxy/tools.py tests/test_tools.py
git commit -m "feat(step6.5): bounded ToolLoop with error-folding, memory, and continuations"
```

---

### Task 5: Teaching injection + native skill + `run_step1.sh` wiring

**Files:**
- Modify: `src/plva_proxy/tools.py`
- Create: `holo-skills/plva-tools/SKILL.md`
- Modify: `run_step1.sh`
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: `TOOL_TEACHING: Final[str]` and `tool_teaching_request_hook(loop: ToolLoop | None = None) -> RequestHook-compatible callable` — strips any stale `[PLVA_TOOLS_BEGIN]…[PLVA_TOOLS_END]` block, then merges fresh teaching (plus the loop's result memory, when non-empty) into the **existing** system message; falls back to the last user message; never creates a second system message. Task 6 chains it after the privacy hook.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tools.py`:

```python
from plva_proxy.tools import (
    TOOL_SYSTEM_BEGIN,
    TOOL_SYSTEM_END,
    TOOL_TEACHING,
    tool_teaching_request_hook,
)


def test_teaching_merges_into_existing_system_message() -> None:
    hook = tool_teaching_request_hook()
    document = {
        "messages": [
            {"role": "system", "content": "You are Holo."},
            {"role": "user", "content": "task"},
        ]
    }
    rewritten, _ = hook(document, {})
    system_messages = [m for m in rewritten["messages"] if m["role"] == "system"]
    assert len(system_messages) == 1
    assert system_messages[0]["content"].startswith("You are Holo.")
    assert TOOL_SYSTEM_BEGIN in system_messages[0]["content"]
    assert "plva_tool" in system_messages[0]["content"]
    assert document["messages"][0]["content"] == "You are Holo."  # input untouched


def test_teaching_falls_back_to_last_user_message() -> None:
    hook = tool_teaching_request_hook()
    document = {"messages": [{"role": "user", "content": [{"type": "text", "text": "task"}]}]}
    rewritten, _ = hook(document, {})
    parts = rewritten["messages"][0]["content"]
    assert any(TOOL_SYSTEM_BEGIN in p["text"] for p in parts if isinstance(p, dict))


def test_teaching_replaces_stale_blocks() -> None:
    hook = tool_teaching_request_hook()
    stale = f"You are Holo.\n\n{TOOL_SYSTEM_BEGIN}\nold teaching\n{TOOL_SYSTEM_END}"
    document = {"messages": [{"role": "system", "content": stale}]}
    rewritten, _ = hook(document, {})
    content = rewritten["messages"][0]["content"]
    assert "old teaching" not in content
    assert content.count(TOOL_SYSTEM_BEGIN) == 1


def test_teaching_includes_session_memory() -> None:
    loop = ToolLoop(ToolRegistry())
    loop.execute(_call("add", {"a": 17, "b": 25}))
    hook = tool_teaching_request_hook(loop)
    document = {"messages": [{"role": "system", "content": "You are Holo."}]}
    rewritten, _ = hook(document, {})
    assert "add: 42" in rewritten["messages"][0]["content"]


def test_teaching_without_messages_fails_closed() -> None:
    with pytest.raises(ToolError):
        tool_teaching_request_hook()({"model": "m"}, {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_tools.py -v -k teaching`
Expected: FAIL — `ImportError: cannot import name 'TOOL_TEACHING'`

- [ ] **Step 3: Implement teaching**

Append to `src/plva_proxy/tools.py`:

```python
TOOL_TEACHING: Final = (
    "[PLVA_TOOLS] This private session provides local tools the desktop cannot see: "
    "echo(text) repeats text; add(a, b) returns the sum of two numbers; sort(items) "
    "returns a list of strings in ascending order. Tools run locally and return "
    "instantly. To call one, emit exactly one action of the form "
    '{"tool_calls": [{"tool_name": "plva_tool", "name": "<tool>", "args": {...}}]} '
    "and nothing else in that step. If your output format rejects that action, "
    'instead include the single line ⟦TOOL⟧{"name": "<tool>", "args": {...}}⟦/TOOL⟧ '
    "inside your thought or answer text. After a call, the next user message begins "
    "with [PLVA_TOOL_RESULT] and carries the result; continue the task using it and "
    "do not repeat an identical call."
)


def _strip_tool_teaching(text: str) -> str:
    while TOOL_SYSTEM_BEGIN in text:
        start = text.find(TOOL_SYSTEM_BEGIN)
        end = text.find(TOOL_SYSTEM_END, start)
        if end < 0:
            raise ToolError("tool teaching block is incomplete")
        text = text[:start] + text[end + len(TOOL_SYSTEM_END) :]
    return text.rstrip()


def _remove_old_tool_teaching(messages: list[Any]) -> None:
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _strip_tool_teaching(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    part["text"] = _strip_tool_teaching(part["text"])


def _merge_teaching(messages: list[Any], wrapped: str) -> None:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            content = message.get("content")
            if not isinstance(content, str):
                raise ToolError("system prompt is not text")
            message["content"] = content.rstrip() + "\n\n" + wrapped
            return
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = content.rstrip() + "\n\n" + wrapped
                return
            if isinstance(content, list):
                content.append({"type": "text", "text": wrapped})
                return
    raise ToolError("tool teaching has no compatible message")


def tool_teaching_request_hook(
    loop: ToolLoop | None = None,
) -> Callable[[dict[str, Any], dict[str, str]], tuple[dict[str, Any], dict[str, str]]]:
    """Merge tool teaching (and session results) into the request, single-system-safe."""

    def apply(
        document: dict[str, Any], headers: dict[str, str]
    ) -> tuple[dict[str, Any], dict[str, str]]:
        rewritten: dict[str, Any] = copy.deepcopy(document)
        messages = rewritten.get("messages")
        if not isinstance(messages, list):
            raise ToolError("request has no message history")
        _remove_old_tool_teaching(messages)
        teaching = TOOL_TEACHING
        if loop is not None:
            memory = loop.memory()
            if memory:
                teaching += (
                    " Results already computed this session: " + "; ".join(memory) + "."
                )
        wrapped = f"{TOOL_SYSTEM_BEGIN}\n{teaching}\n{TOOL_SYSTEM_END}"
        _merge_teaching(messages, wrapped)
        return rewritten, headers

    return apply
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: all pass (32 tests)

- [ ] **Step 5: Author the native skill (channel 1)**

Create `holo-skills/plva-tools/SKILL.md`:

```markdown
---
name: plva-tools
description: Call fast local computation tools (echo, add, sort) through this private session when a task needs a computed result instead of a desktop action.
---

# Local computation tools

This session provides local tools that run on this computer, outside your view,
and return instantly. Available tools:

- `echo(text)` — repeats `text` back.
- `add(a, b)` — returns the sum of the numbers `a` and `b`.
- `sort(items)` — returns the strings in `items` in ascending order.

To call a tool, emit exactly one action of the form
`{"tool_calls": [{"tool_name": "plva_tool", "name": "<tool>", "args": {...}}]}`
and nothing else in that step. If your output format rejects that action, instead
write the single line `⟦TOOL⟧{"name": "<tool>", "args": {...}}⟦/TOOL⟧` inside your
thought or answer text.

After a call, the next user message begins with `[PLVA_TOOL_RESULT]` and carries
the result. Continue the task using that result and do not repeat an identical
call. The live session instructions are authoritative if they differ from this
document.
```

- [ ] **Step 6: Wire `run_step1.sh`**

In `run_step1.sh`, next to the existing `parse_on_off PRIVACY_*` block (around line 131), add (outside the `PLVA_REDACT` conditional — tools do not require redaction):

```bash
parse_on_off TOOLS_ENABLED "${PLVA_TOOLS:-0}" PLVA_TOOLS
parse_on_off TOOLS_SKILL "${PLVA_TOOLS_SKILL:-$TOOLS_ENABLED}" PLVA_TOOLS_SKILL
[[ "$TOOLS_ENABLED" == 1 ]] && HOOK_ARGS+=(--tools)
if [[ -n "${PLVA_CAPTURE_GRAMMAR:-}" ]]; then
  HOOK_ARGS+=(--capture-grammar "$PLVA_CAPTURE_GRAMMAR")
fi
```

Then mirror the `plva-placeholders` skill install/disable block (lines ~220–231) with a parallel `plva-tools` block, including the same restore-on-exit handling in the cleanup trap:

```bash
if [[ "$TOOLS_SKILL" == 1 ]]; then
  mkdir -p "$HOME/.holo/skills/plva-tools"
  cp "holo-skills/plva-tools/SKILL.md" "$HOME/.holo/skills/plva-tools/SKILL.md"
elif [[ -f "$HOME/.holo/skills/plva-tools/SKILL.md" ]]; then
  TOOLS_SKILL_DISABLED_FILE="$HOME/.holo/skills/plva-tools/SKILL.md.disabled.$$"
  mv "$HOME/.holo/skills/plva-tools/SKILL.md" "$TOOLS_SKILL_DISABLED_FILE"
fi
```

and in the existing cleanup trap where `SKILL_DISABLED_FILE` is restored:

```bash
if [[ -n "${TOOLS_SKILL_DISABLED_FILE:-}" && -f "$TOOLS_SKILL_DISABLED_FILE" ]]; then
  mv "$TOOLS_SKILL_DISABLED_FILE" "$HOME/.holo/skills/plva-tools/SKILL.md"
fi
```

(Note: `--tools` and `--capture-grammar` do not exist on the proxy until Task 6; commit this task and Task 6 before running the script with `PLVA_TOOLS=1`.)

- [ ] **Step 7: Verify script syntax, run gates, commit**

Run: `bash -n run_step1.sh && .venv/bin/pytest -q && .venv/bin/ruff format src tests && .venv/bin/ruff check . && .venv/bin/mypy src tests`
Expected: all pass

```bash
git add src/plva_proxy/tools.py tests/test_tools.py holo-skills/plva-tools/SKILL.md run_step1.sh
git commit -m "feat(step6.5): tool teaching injection, native plva-tools skill, launcher wiring"
```

---

### Task 6: Relay integration (`tool_loop` in `_relay`) + CLI flags

**Files:**
- Modify: `src/plva_proxy/proxy.py`
- Test: `tests/test_tool_loop.py`

**Interfaces:**
- Consumes: `ToolLoop`, `ToolRegistry`, `ToolError`, `tool_teaching_request_hook` from `plva_proxy.tools` (Tasks 2–5).
- Produces: `create_app(..., tool_loop: ToolLoop | None = None)`; `plva-proxy --tools [--tools-max-rounds N]`. The loop runs inside `_relay` after SSE assembly and **before** the response hook; continuations re-use the upstream client and headers; every failure raises `HookError`/`ToolError` → 502.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tool_loop.py`:

```python
import httpx
from fastapi.testclient import TestClient

from plva_proxy.proxy import Hooks, ProxyConfig, create_app
from plva_proxy.tools import ToolLoop, ToolRegistry, tool_teaching_request_hook


def _completion_body(content: str) -> dict[str, object]:
    return {
        "id": "c1",
        "created": 1,
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
    }


def _tool_call_content() -> str:
    return json.dumps(
        {"tool_calls": [{"tool_name": "plva_tool", "name": "add", "args": {"a": 17, "b": 25}}]}
    )


def _final_action_content() -> str:
    return json.dumps({"tool_calls": [{"tool_name": "write", "text": "42"}]})


def _app_with_scripted_upstream(
    contents: list[str], seen: list[dict[str, object]], *, max_rounds: int = 4
):
    def handler(request: httpx.Request) -> httpx.Response:
        document = json.loads(request.content)
        seen.append(document)
        index = min(len(seen), len(contents)) - 1
        return httpx.Response(200, json=_completion_body(contents[index]))

    return create_app(
        ProxyConfig(upstream_base_url="https://upstream.test/v1", api_key="key"),
        hooks=Hooks(on_request=tool_teaching_request_hook()),
        tool_loop=ToolLoop(ToolRegistry(), max_rounds=max_rounds),
        transport=httpx.MockTransport(handler),
    )


def test_tool_loop_round_trips_and_forwards_only_final_action() -> None:
    seen: list[dict[str, object]] = []
    app = _app_with_scripted_upstream([_tool_call_content(), _final_action_content()], seen)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "add the numbers"}]},
        )
    assert response.status_code == 200
    assert len(seen) == 2
    follow_messages = seen[1]["messages"]
    assert follow_messages[-2]["role"] == "assistant"
    assert follow_messages[-1]["role"] == "user"
    assert follow_messages[-1]["content"].startswith("[PLVA_TOOL_RESULT] add returned: 42")
    assert seen[1]["stream"] is False
    final = json.loads(response.json()["choices"][0]["message"]["content"])
    assert final["tool_calls"][0]["tool_name"] == "write"


def test_tool_loop_exceeding_max_rounds_fails_closed() -> None:
    seen: list[dict[str, object]] = []
    app = _app_with_scripted_upstream([_tool_call_content()], seen, max_rounds=2)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "loop forever"}]},
        )
    assert response.status_code == 502


def test_tool_loop_handles_sse_upstream_and_reemits_sse() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            chunks = b"".join(
                f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode()
                for event in (
                    {
                        "id": "c1",
                        "created": 1,
                        "model": "m",
                        "choices": [
                            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                        ],
                    },
                    {
                        "id": "c1",
                        "created": 1,
                        "model": "m",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": _tool_call_content()},
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            ) + b"data: [DONE]\n\n"
            return httpx.Response(
                200, content=chunks, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=_completion_body(_final_action_content()))

    app = create_app(
        ProxyConfig(upstream_base_url="https://upstream.test/v1", api_key="key"),
        tool_loop=ToolLoop(ToolRegistry()),
        transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "go"}]},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert len(calls) == 2
    assert b"write" in response.content
    assert b"plva_tool" not in response.content


def test_marker_channel_round_trips_through_relay() -> None:
    seen: list[dict[str, object]] = []
    marker = '⟦TOOL⟧{"name": "echo", "args": {"text": "pong"}}⟦/TOOL⟧'
    app = _app_with_scripted_upstream([marker, _final_action_content()], seen)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "ping"}]},
        )
    assert response.status_code == 200
    assert len(seen) == 2
    assert "echo returned: pong" in seen[1]["messages"][-1]["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_tool_loop.py -v`
Expected: grammar tests pass; new tests FAIL — `TypeError: create_app() got an unexpected keyword argument 'tool_loop'`

- [ ] **Step 3: Implement relay integration**

In `src/plva_proxy/proxy.py`:

3a. Extend the privacy import block's neighbors with:

```python
from plva_proxy.tools import (
    ToolError,
    ToolLoop,
    ToolRegistry,
    tool_teaching_request_hook,
)
```

3b. Add the parameter to `create_app` (after `scrubber`):

```python
    tool_loop: ToolLoop | None = None,
```

3c. Inside `_relay`, replace:

```python
        response_hook = active_hooks.on_response if use_hooks else None
        hook_applies = response_hook is not None and upstream.status_code == 200
```

with:

```python
        response_hook = active_hooks.on_response if use_hooks else None
        active_tool_loop = tool_loop if use_hooks else None
        hook_applies = (
            response_hook is not None or active_tool_loop is not None
        ) and upstream.status_code == 200
```

3d. Define the loop runner inside `_relay` (before the `if is_sse and not hook_applies:` block), capturing `body`, `headers`, `path`, `client`:

```python
        async def _run_tool_loop(loop: ToolLoop, completion: dict[str, Any]) -> dict[str, Any]:
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
                    "POST", path, content=continuation_body, headers=headers
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
```

3e. Change the response-mutation branch from `if response_hook is not None and upstream.status_code == 200:` to `if hook_applies:` and inside it replace `mutated = response_hook(document)` with:

```python
                if active_tool_loop is not None:
                    document = await _run_tool_loop(active_tool_loop, document)
                mutated = response_hook(document) if response_hook is not None else document
```

3f. Widen both fail-closed except tuples in `_relay` that read `(HookError, PrivacyError, ValueError)` to `(HookError, PrivacyError, ToolError, ValueError)`.

3g. In `main()`, add the flags (after `--capture-grammar` from Task 1):

```python
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
```

validation next to the other checks:

```python
    if not 1 <= args.tools_max_rounds <= 8:
        parser.error("--tools-max-rounds must be between 1 and 8")
```

assembly, after `hooks = _combine_hooks(hooks, privacy_hooks)` and **before** the Task 1 grammar-capture chaining:

```python
    tool_loop_instance: ToolLoop | None = None
    if args.tools:
        tool_loop_instance = ToolLoop(ToolRegistry(), max_rounds=args.tools_max_rounds)
        hooks = _combine_hooks(
            hooks, Hooks(on_request=tool_teaching_request_hook(tool_loop_instance))
        )
```

and pass it to the app: in the `app_options` dict add `"tool_loop": tool_loop_instance,`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tool_loop.py tests/test_proxy.py tests/test_proxy_hooks.py -v`
Expected: all pass (existing proxy suites must stay green — the seam change is additive)

- [ ] **Step 5: Run gates and commit**

Run: `.venv/bin/pytest -q && .venv/bin/ruff format src tests && .venv/bin/ruff check . && .venv/bin/mypy src tests`
Expected: all pass, coverage ≥80%

```bash
git add src/plva_proxy/proxy.py tests/test_tool_loop.py
git commit -m "feat(step6.5): bounded proxy inner loop executes local tools and re-queries the model"
```

---

### Task 7: Live probe CLI (`plva-tool-probe`)

A Step-0-style, non-sensitive probe that talks to the provider directly (no runtime) and records the (b)/(c)/(d) booleans per channel, with and without the captured grammar attached — plus an offline grammar summary for (a). Never prints or stores response content.

**Files:**
- Create: `src/plva_proxy/tool_probe.py`
- Modify: `pyproject.toml` (`[project.scripts]`: `plva-tool-probe = "plva_proxy.tool_probe:main"`)
- Test: `tests/test_tool_probe.py`

**Interfaces:**
- Consumes: `ToolLoop`, `ToolRegistry`, `ToolError`, `find_tool_call`, `TOOL_TEACHING` (Tasks 2–5); `PROVIDERS` presets.
- Produces:
  - `summarize_grammar(snapshot: Mapping[str, Any]) -> dict[str, Any]` → `{"enum_sets": list[list[str]], "admits_plva_tool": bool}`,
  - `build_probe_request(model: str, teaching: str, grammar: Mapping[str, Any] | None) -> dict[str, Any]`,
  - `evaluate_channel(client: httpx.Client, model: str, channel: str, teaching: str, grammar: Mapping[str, Any] | None) -> dict[str, Any]` → row with `channel`, `grammar_attached`, `status`, `emits`, `parses`, `round_trip` (bools/ints only),
  - `main() -> None` CLI: `--provider {overshoot,hcompany}`, `--upstream URL`, `--grammar PATH`, `--out PATH`, `--analyze-only`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tool_probe.py`:

```python
"""Step 6.5: privacy-safe live tool-channel probe."""

from __future__ import annotations

import json

import httpx

from plva_proxy.tool_probe import (
    MARKER_TEACHING,
    STRUCTURED_TEACHING,
    build_probe_request,
    evaluate_channel,
    summarize_grammar,
)


def test_summarize_grammar_lists_enums_and_admission() -> None:
    snapshot = {
        "structured_outputs": {
            "json_schema": {
                "properties": {"tool_name": {"enum": ["click", "write", "answer"]}}
            }
        }
    }
    summary = summarize_grammar(snapshot)
    assert ["answer", "click", "write"] in summary["enum_sets"]
    assert summary["admits_plva_tool"] is False


def test_summarize_grammar_admits_when_unconstrained() -> None:
    assert summarize_grammar({"structured_outputs": {"type": "object"}})["admits_plva_tool"] is True


def test_build_probe_request_is_synthetic_and_single_system() -> None:
    request = build_probe_request("Hcompany/Holo3-35B-A3B", STRUCTURED_TEACHING, None)
    roles = [m["role"] for m in request["messages"]]
    assert roles == ["system", "user"]
    assert request["stream"] is False
    assert "structured_outputs" not in request
    with_grammar = build_probe_request("m", STRUCTURED_TEACHING, {"type": "object"})
    assert with_grammar["structured_outputs"] == {"type": "object"}


def _client(handler) -> httpx.Client:
    return httpx.Client(
        base_url="https://upstream.test/v1", transport=httpx.MockTransport(handler)
    )


def test_evaluate_channel_full_round_trip() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        content = (
            json.dumps(
                {
                    "tool_calls": [
                        {"tool_name": "plva_tool", "name": "add", "args": {"a": 17, "b": 25}}
                    ]
                }
            )
            if len(calls) == 1
            else "The sum is 42."
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": content}}]},
        )

    with _client(handler) as client:
        row = evaluate_channel(client, "m", "structured", STRUCTURED_TEACHING, None)
    assert row == {
        "channel": "structured",
        "grammar_attached": False,
        "status": 200,
        "emits": True,
        "parses": True,
        "round_trip": True,
    }


def test_evaluate_channel_records_non_emission() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "I cannot."}}]},
        )

    with _client(handler) as client:
        row = evaluate_channel(client, "m", "marker", MARKER_TEACHING, None)
    assert (row["emits"], row["parses"], row["round_trip"]) == (False, False, False)


def test_evaluate_channel_records_provider_rejection() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad template"})

    with _client(handler) as client:
        row = evaluate_channel(client, "m", "structured", STRUCTURED_TEACHING, None)
    assert row["status"] == 400
    assert row["emits"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_tool_probe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'plva_proxy.tool_probe'`

- [ ] **Step 3: Implement the probe**

Create `src/plva_proxy/tool_probe.py`:

```python
"""Live tool-channel probe for Step 6.5 — records booleans and schema, never content.

Talks to the provider directly (no runtime, no real screen): a synthetic
screenshot and a synthetic arithmetic task exercise each candidate invocation
channel, with and without the captured runtime grammar attached. Output is a
channel × (grammar, status, emits, parses, round_trip) matrix. Response text
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
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
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
    api_key = next(
        (value for name in provider.key_names if (value := os.environ.get(name))), None
    )
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
```

3b. In `pyproject.toml` under `[project.scripts]`, next to the existing entries:

```toml
plva-tool-probe = "plva_proxy.tool_probe:main"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tool_probe.py -v`
Expected: 6 passed

- [ ] **Step 5: Reinstall entry points, run gates, commit**

Run: `~/.local/bin/uv sync --frozen 2>/dev/null || ~/.local/bin/uv sync && .venv/bin/plva-tool-probe --help && .venv/bin/pytest -q && .venv/bin/ruff format src tests && .venv/bin/ruff check . && .venv/bin/mypy src tests && ~/.local/bin/uv lock --check`
Expected: help text prints; all gates pass

```bash
git add src/plva_proxy/tool_probe.py tests/test_tool_probe.py pyproject.toml uv.lock
git commit -m "feat(step6.5): privacy-safe live tool-channel probe (plva-tool-probe)"
```

---

### Task 8: Live spike — record the matrix, decide the channel, update docs

This is the empirical heart of Step 6.5. Everything recorded is booleans/schema; nothing sensitive. Requires an operator with `OVERSHOOT_API_KEY` (or `HAI_API_KEY`).

**Files:**
- Create: `verification/step-6-5-tool-channel.md`
- Create: `docs/decisions/0002-tool-call-channel.md`
- Modify: `BLUEPRINT.md` (Step 6.5 status), `PROJECT_MAP.md` (new files + checkpoint)

- [ ] **Step 1: Capture the runtime grammar (one authorized single-step run)**

```bash
PLVA_TOOLS=0 PLVA_CAPTURE_GRAMMAR=/tmp/plva-grammar.json PLVA_MAX_STEPS=1 \
  ./run_step1.sh "Return an answer immediately without acting."
jq 'keys' /tmp/plva-grammar.json     # expect: request_keys, model, structured_outputs, chat_template_kwargs
jq 'has("messages")' /tmp/plva-grammar.json   # expect: false
```

Shred the run's ephemeral `--runs-dir` per the standing Step 0 note.

- [ ] **Step 2: Analyze the grammar offline — answers (a) per channel**

```bash
.venv/bin/plva-tool-probe --grammar /tmp/plva-grammar.json --analyze-only
```

Record: does any `enum` constrain `tool_name`? If `admits_plva_tool` is false, the structured channel is grammar-blocked (a=no) and the marker channel's (a) depends on whether free-text fields (`thought`/`answer`) are unconstrained strings — read the schema and record it.

- [ ] **Step 3: Probe emission and round-trip direct-to-provider — answers (b)(c)(d) for structured + marker**

```bash
.venv/bin/plva-tool-probe --provider overshoot --grammar /tmp/plva-grammar.json \
  --out /tmp/plva-tool-matrix.json
```

Expected output: one `channel=… emits=… parses=… round_trip=…` line per channel × grammar variant. Re-run 3× to check stability; record the worst case.

- [ ] **Step 4: Probe the skill channel through the real runtime — (b)(c)(d) for channel 1, plus end-to-end proof of channels 2/3 under the runtime**

```bash
PLVA_TOOLS=1 PLVA_TOOLS_SKILL=1 PLVA_MAX_STEPS=6 \
  ./run_step1.sh "Use the local add tool to compute 17 plus 25, then answer with only the sum."
```

Watch the proxy log for `tool executed: name=add channel=…` and `tool loop completed: rounds=…`. The channel named in the log answers *which* mechanism the model chose with the skill installed; a correct final answer of 42 proves (d) end-to-end through the closed runtime. Also run once with `PLVA_TOOLS_SKILL=0` to isolate proxy-injection-only behavior. Shred runs-dirs.

- [ ] **Step 5: Write the verification record**

Create `verification/step-6-5-tool-channel.md` with exactly this structure (filled with the recorded values — booleans and schema facts only, no content):

```markdown
# Step 6.5 verification — tool-call channel spike

Date: <run date>. All inputs synthetic; recorded data is booleans and schema only.

## Captured grammar

- structured_outputs present: yes/no; tool_name enum: <sorted list or "unconstrained">
- free-text fields available to the marker channel: <field names>

## Channel matrix

| Channel | (a) grammar permits | (b) model emits | (c) proxy parses | (d) round-trip |
|---|---|---|---|---|
| 1. skill-declared | … | … | … | … |
| 2. structured action | … | … | … | … |
| 3. free-text marker | … | … | … | … |

(b)-(d) recorded from N probe runs each; worst case shown. Provider status codes: …

## Recommendation

<one channel + why>, fallback: <proxy pseudo-tool via next-observation injection (Task 9)
or point-and-flag>. Steps 7/8/9/10/13 adopt this channel.

## Commands

<the exact Task 8 commands used>
```

- [ ] **Step 6: Write the ADR**

Create `docs/decisions/0002-tool-call-channel.md`:

```markdown
# ADR 0002 — Tool-call invocation channel for Holo3

Status: accepted. Date: <date>.

## Context
Holo3 exposes structured_outputs and no native tools (Step 0). Steps 7/8/9/10/13
need the CUA to invoke a local tool and consume its result. Step 6.5 probed three
emission channels over a proxy inner loop (execute locally, append
[PLVA_TOOL_RESULT], re-query, forward only the final action).

## Decision
<channel> is the invocation channel; the proxy inner loop is the round-trip
mechanism; max rounds default 4; failures fail closed (502).

## Evidence
verification/step-6-5-tool-channel.md (matrix, N runs, provider statuses).

## Consequences
- Step 8 layers the SPEAK payload/reference question on this channel.
- Step 13 adopts it for deterministic ops; its token-only contract must gate the
  ToolLoop result memory before any tool touches vault values.
- Tool interactions are invisible to the closed runtime; cross-step continuity
  relies on the teaching hook's bounded result memory.
```

- [ ] **Step 7: Update BLUEPRINT and PROJECT_MAP**

In `BLUEPRINT.md`, change the Step 6.5 heading marker from `### 🔲 Step 6.5` to `### ✅ Step 6.5` and append a dated completion note pointing at the verification doc and ADR (mirror the format of the Step 5a/6 completion notes). In `PROJECT_MAP.md`: add `tools.py` and `tool_probe.py` under `src/plva_proxy/`, the three new test files under `tests/`, `plva-tools/` under a `holo-skills/` entry, `0002-tool-call-channel.md` under `docs/decisions/`, `step-6-5-tool-channel.md` under `verification/`, the `plva-tool-probe` command in the commands table, and a checkpoint paragraph stating the recorded recommendation.

- [ ] **Step 8: Commit**

```bash
git add verification/step-6-5-tool-channel.md docs/decisions/0002-tool-call-channel.md BLUEPRINT.md PROJECT_MAP.md
git commit -m "docs(step6.5): record tool-channel matrix, ADR 0002 recommendation"
```

---

### Task 9 (CONDITIONAL — build only if Task 8 shows no channel round-trips): pseudo-tool fallback via next-observation delivery

**Trigger:** every channel fails (d) — e.g. the chat template rejects appended assistant/user turns (continuation HTTP 400) or the model never consumes results. The blueprint's fallback: the proxy runs the op and folds the result into the *next* runtime step's observation instead of re-querying inline.

**Files:**
- Modify: `src/plva_proxy/tools.py`, `src/plva_proxy/proxy.py`
- Test: `tests/test_tool_loop.py`

**Design (complete, so a fresh engineer can build it):**
1. Add `--tools-noop-action JSON` (default `'{"tool_calls": [{"tool_name": "wait", "seconds": 1}]}'` — confirm the actual no-op action name from the Task 8 grammar snapshot and set the recorded one as the default).
2. In `ToolLoop`, add `neutralize(completion: dict[str, Any], noop_action: dict[str, Any]) -> dict[str, Any]`: deep-copy the completion and replace `choices[0].message.content` with `json.dumps(noop_action)` — the runtime executes a harmless step and takes the next screenshot.
3. In `_run_tool_loop`, on `HookError("tool continuation status 400")` from the **first** continuation attempt: execute the tool (already done), record the result in memory, and return `loop.neutralize(completion, noop_action)` instead of raising. The Task 5 teaching hook already injects `Results already computed this session: add: 42` into the next request, which is exactly the delivery mechanism — the model re-plans with the result in context on the following step.
4. Tests: scripted MockTransport where the continuation POST returns 400 → assert the relay responds 200 with the no-op action content, and that a subsequent request through `tool_teaching_request_hook(loop)` carries `add: 42`.
5. Record the fallback's activation in the verification doc and flip ADR 0002's decision to the pseudo-tool shape.

- [ ] Build per the design above (TDD, same gates), commit as `feat(step6.5): pseudo-tool fallback via next-observation result delivery`.

---

## Self-Review Notes

- **Spec coverage:** Blueprint Step 6.5 asks for (1) a Step-0-style non-sensitive capture → Task 1 + Task 8 Step 1; (2) a trivial round-trip tool exercised across three ranked channels → Tasks 2–7 (skill channel in Task 5 + Task 8 Step 4); (3) a recorded matrix of channels × (a–d) → Task 8 Step 5; (4) a recommendation + fallback → Task 8 Step 6 + Task 9. §8 constraints: fail-closed (ToolError → 502 paths, tested), value-free logs (names/keys/counts only), single-system-message rule (merge logic, tested), streaming safety (SSE test in Task 6), unknown-key forwarding (continuation deep-copies the full request; tested via `temperature` preservation).
- **Type consistency:** `ToolCall(name, args, channel)`, `ToolRegistry.execute(call) -> str`, `find_tool_call(completion) -> ToolCall | None`, `ToolLoop.max_rounds/.detect/.execute/.memory/.continuation`, `create_app(..., tool_loop=...)`, `grammar_capture_hook(out_path)` are used with the same signatures across Tasks 1–9.
- **Known judgment calls:** unknown tool names fold into `error:` results (round-trip-testable) while structurally unparseable tool intents fail closed; result memory deliberately carries spike results in cleartext because spike tools are synthetic — Step 13 must gate this before sensitive tools exist (noted in code docstring and ADR).
