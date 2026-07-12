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
