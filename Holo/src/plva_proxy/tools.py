"""Local tool channel for the Step 6.5 spike: registry, detection, teaching, and loop.

Holo3 exposes no native ``tools`` support (Step 0), so a "tool call" here is a
convention the proxy teaches and parses: either a reserved ``plva_tool`` entry
inside the model's existing structured ``tool_calls`` action envelope, or an
explicit ``⟦TOOL⟧…⟦/TOOL⟧`` marker inside free text. Execution is local and
deterministic; results are fed back through a bounded proxy inner loop. Logs
carry tool names, channels, and argument *keys* only — never values (§8.5).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
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
