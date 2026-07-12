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
