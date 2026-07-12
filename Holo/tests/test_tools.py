"""Step 6.5: local tool registry, channel detection, teaching, and loop mechanics."""

from __future__ import annotations

import json

import pytest

from plva_proxy.tools import ToolCall, ToolError, ToolLoop, ToolRegistry, find_tool_call


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
    choices = completion["choices"]
    assert isinstance(choices, list) and choices
    message = choices[0]
    assert isinstance(message, dict)
    assert follow["messages"][1]["content"] == message["message"]["content"]
    assert follow["messages"][2]["role"] == "user"
    assert follow["messages"][2]["content"].startswith("[PLVA_TOOL_RESULT] add returned: 42")


def test_continuation_without_assistant_content_fails_closed() -> None:
    loop = ToolLoop(ToolRegistry())
    with pytest.raises(ToolError):
        loop.continuation({"messages": []}, {"choices": []}, _call("echo", {"text": "x"}), "x")
