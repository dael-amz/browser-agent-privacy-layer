from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import httpx
import pytest

from plva_proxy.local_llm import LLMConfig, LocalLLMError, LoopbackLLMClient
from plva_proxy.semantic_executor import SemanticExecutor, SemanticOpRequest

Handler = Callable[[httpx.Request], httpx.Response]

VALUES: dict[str, str] = {
    "NAME_1_ab12": "Charlie Rivera",
    "NAME_2_ab12": "Alice Zhang",
    "NAME_3_ab12": "Bob O'Neill",
}


def _llm_config() -> LLMConfig:
    return LLMConfig(
        base_url="http://127.0.0.1:8555/v1",
        model="test-model",
        timeout_seconds=5.0,
        max_tokens=128,
    )


def _client(handler: Handler) -> LoopbackLLMClient:
    return LoopbackLLMClient(_llm_config(), transport=httpx.MockTransport(handler))


def _reply(answer: Sequence[object]) -> httpx.Response:
    """Build a completion whose content is the model's item-text answer.

    The current contract (see ``semantic_executor._answer_schema``) has the
    model answer with the exact resolved item texts, grammar-constrained to
    an enum of those exact strings; the executor maps each string back to its
    token via ``token_by_value`` and the mapped result is tokens-only.
    """
    content = json.dumps({"answer": answer})
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": content}}]},
    )


def _forbidden(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"no HTTP request expected in this fail-closed path: {request.url}")


def _resolver(values: dict[str, str]) -> Callable[[str], str]:
    return values.__getitem__


def _executor(handler: Handler, *, values: dict[str, str] | None = None) -> SemanticExecutor:
    return SemanticExecutor(_client(handler), resolver=_resolver(values or VALUES))


# --- sort happy path ---------------------------------------------------------------


def test_sort_happy_path_returns_mapped_tokens_and_value_free_observation() -> None:
    executor = _executor(lambda request: _reply(["Alice Zhang", "Bob O'Neill", "Charlie Rivera"]))
    request = SemanticOpRequest(
        kind="sort",
        instruction="alphabetical order of first names",
        tokens=tuple(VALUES),
        request_id="req_1",
    )

    result = executor.execute(request)

    assert result.tokens == ("NAME_2_ab12", "NAME_3_ab12", "NAME_1_ab12")
    observation = result.observation_text()
    assert "⟦PLVA_TOOL_RESULT:sort:req_1⟧" in observation
    for token in result.tokens:
        assert f"«{token}»" in observation
    for value in VALUES.values():
        assert value not in observation


def test_reply_item_texts_with_surrounding_whitespace_are_cleaned_and_accepted() -> None:
    executor = _executor(lambda request: _reply([f" {value} " for value in VALUES.values()]))
    request = SemanticOpRequest(kind="sort", instruction="any order", tokens=tuple(VALUES))

    result = executor.execute(request)

    assert set(result.tokens) == set(VALUES)


def test_sort_missing_item_retries_then_succeeds() -> None:
    calls = 0
    all_values = list(VALUES.values())

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _reply(all_values[:2])  # missing one item
        return _reply(all_values)

    executor = _executor(handler)
    request = SemanticOpRequest(kind="sort", instruction="any order", tokens=tuple(VALUES))

    result = executor.execute(request)

    assert calls == 2
    assert set(result.tokens) == set(VALUES)


def test_sort_missing_item_on_retry_too_raises() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _reply(list(VALUES.values())[:2])  # always missing one item

    executor = _executor(handler)
    request = SemanticOpRequest(kind="sort", instruction="any order", tokens=tuple(VALUES))

    with pytest.raises(LocalLLMError):
        executor.execute(request)
    assert calls == 2


def test_reply_with_an_item_text_that_does_not_match_exactly_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # "Alice" is a shortened, non-exact copy of the resolved value.
        return _reply(["Alice", "Bob O'Neill", "Charlie Rivera"])

    executor = _executor(handler)
    request = SemanticOpRequest(kind="sort", instruction="any order", tokens=tuple(VALUES))

    with pytest.raises(LocalLLMError):
        executor.execute(request)


def test_reply_with_a_non_string_entry_fails_validation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _reply([1, "Bob O'Neill", "Charlie Rivera"])

    executor = _executor(handler)
    request = SemanticOpRequest(kind="sort", instruction="any order", tokens=tuple(VALUES))

    with pytest.raises(LocalLLMError):
        executor.execute(request)


# --- select ------------------------------------------------------------------------


def test_select_count_mismatch_is_retried_then_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _reply(["Charlie Rivera"])  # always 1 item, but select_count=2

    executor = _executor(handler)
    request = SemanticOpRequest(
        kind="select", instruction="match something", tokens=tuple(VALUES), select_count=2
    )

    with pytest.raises(LocalLLMError):
        executor.execute(request)


def test_select_happy_path_returns_the_matching_subset() -> None:
    executor = _executor(lambda request: _reply(["Charlie Rivera", "Bob O'Neill"]))
    request = SemanticOpRequest(
        kind="select", instruction="match something", tokens=tuple(VALUES), select_count=2
    )

    result = executor.execute(request)

    assert result.tokens == ("NAME_1_ab12", "NAME_3_ab12")


def test_select_empty_result_is_allowed_when_select_count_is_none() -> None:
    executor = _executor(lambda request: _reply([]))
    request = SemanticOpRequest(kind="select", instruction="match nothing", tokens=tuple(VALUES))

    result = executor.execute(request)

    assert result.tokens == ()
    assert "no tokens match" in result.observation_text()


# --- request validation (must fail before any HTTP call) ---------------------------


def test_validation_rejects_too_few_tokens_before_http() -> None:
    executor = _executor(_forbidden)
    request = SemanticOpRequest(kind="sort", instruction="order", tokens=("NAME_1_ab12",))

    with pytest.raises(LocalLLMError, match="tokens"):
        executor.execute(request)


def test_validation_rejects_duplicate_tokens_before_http() -> None:
    executor = _executor(_forbidden)
    request = SemanticOpRequest(
        kind="sort", instruction="order", tokens=("NAME_1_ab12", "NAME_1_ab12")
    )

    with pytest.raises(LocalLLMError, match="unique"):
        executor.execute(request)


def test_validation_rejects_malformed_token_shape_before_http() -> None:
    executor = _executor(_forbidden)
    lowercase = SemanticOpRequest(
        kind="sort", instruction="order", tokens=("name_1_ab12", "NAME_2_ab12")
    )
    with pytest.raises(LocalLLMError, match="malformed token"):
        executor.execute(lowercase)

    missing_nonce = SemanticOpRequest(
        kind="sort", instruction="order", tokens=("NAME_1", "NAME_2_ab12")
    )
    with pytest.raises(LocalLLMError, match="malformed token"):
        executor.execute(missing_nonce)


def test_validation_rejects_oversized_instruction_before_http() -> None:
    executor = _executor(_forbidden)
    request = SemanticOpRequest(
        kind="sort", instruction="x" * 501, tokens=("NAME_1_ab12", "NAME_2_ab12")
    )

    with pytest.raises(LocalLLMError, match="too long"):
        executor.execute(request)


def test_validation_rejects_select_count_on_sort_before_http() -> None:
    executor = _executor(_forbidden)
    request = SemanticOpRequest(
        kind="sort",
        instruction="order",
        tokens=("NAME_1_ab12", "NAME_2_ab12"),
        select_count=1,
    )

    with pytest.raises(LocalLLMError, match="select_count is only valid"):
        executor.execute(request)


# --- resolver and value guards (must fail before any HTTP call) ----------------------


def test_resolver_error_propagates() -> None:
    def failing_resolver(token: str) -> str:
        raise RuntimeError("synthetic vault denial")

    executor = SemanticExecutor(_client(_forbidden), resolver=failing_resolver)
    request = SemanticOpRequest(
        kind="sort", instruction="order", tokens=("NAME_1_ab12", "NAME_2_ab12")
    )

    with pytest.raises(RuntimeError, match="synthetic vault denial"):
        executor.execute(request)


def test_resolved_value_too_long_fails_before_http() -> None:
    values = {"NAME_1_ab12": "x" * 501, "NAME_2_ab12": "short"}
    executor = _executor(_forbidden, values=values)
    request = SemanticOpRequest(kind="sort", instruction="order", tokens=tuple(values))

    with pytest.raises(LocalLLMError, match="empty or too long"):
        executor.execute(request)


def test_resolved_value_empty_fails_before_http() -> None:
    values = {"NAME_1_ab12": "   ", "NAME_2_ab12": "short"}
    executor = _executor(_forbidden, values=values)
    request = SemanticOpRequest(kind="sort", instruction="order", tokens=tuple(values))

    with pytest.raises(LocalLLMError, match="empty or too long"):
        executor.execute(request)


def test_two_tokens_resolving_to_the_same_value_fails_before_http() -> None:
    values = {"NAME_1_ab12": "Same Value", "NAME_2_ab12": "Same Value"}
    executor = _executor(_forbidden, values=values)
    request = SemanticOpRequest(kind="sort", instruction="order", tokens=tuple(values))

    with pytest.raises(LocalLLMError, match="two tokens resolved to the same value"):
        executor.execute(request)
