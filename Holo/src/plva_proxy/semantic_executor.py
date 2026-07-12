"""Placeholder-preserving semantic operations via the local LLM (Step 13 B).

The executor resolves tokens to cleartext through an injected resolver (later
the session vault, so per-class policy still gates what it may see), asks the
zero-egress local model to compute a fuzzy answer, and returns ONLY tokens the
CUA already holds — a reordering (``sort``) or a subset (``select``). The model
answers with the exact item texts (grammar-constrained to that enum where the
server supports ``json_schema``; small models reason far better over values
than over label indirection), the executor maps each answer string back to its
token, and the completion is then discarded. The completion may carry cleartext
— that is allowed for a zero-egress local executor and it never leaves this
process — while the result folded into the CUA's next observation is tokens-only
by construction: mapped, membership-validated, never free text (§8.12).

Invocation arrives later via the Step 6.5 marker channel
(``⟦PLVA_TOOL:sort:request_42⟧`` → proxy allowlist → this executor → value-free
result injected into the next observation). ``kind`` names align with the
marker verbs and ``request_id`` is carried through so the injected observation
can reference the originating marker; the same entry point also serves the
proxy/app-initiated fallback when the model never emits a marker.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from .local_llm import LocalLLMError, LoopbackLLMClient

OpKind = Literal["sort", "select"]
TokenResolver = Callable[[str], str]

OP_KINDS: Final = frozenset({"sort", "select"})
_TOKEN_SHAPE: Final = re.compile(r"[A-Z][A-Z0-9_]*_\d+_[0-9a-f]{4}")
_MIN_TOKENS: Final = 2
_MAX_TOKENS: Final = 40
_MAX_VALUE_CHARS: Final = 500
_MAX_INSTRUCTION_CHARS: Final = 500

_SYSTEM: Final = (
    "You are PLVA's local computation helper. You will see a list of items. Compute the "
    'requested answer, then reply with ONLY a JSON object: {"answer": [...]} whose entries '
    "are item texts copied exactly, character for character, from the list. Never invent, "
    "shorten, or rewrite an item."
)


@dataclass(frozen=True, slots=True)
class SemanticOpRequest:
    """One content-dependent operation over issued tokens.

    ``sort`` returns every token, reordered per the instruction; ``select``
    returns the matching subset (exactly ``select_count`` tokens when given).
    """

    kind: OpKind
    instruction: str
    tokens: tuple[str, ...]
    select_count: int | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticOpResult:
    """A value-free answer: nothing but tokens the model already holds."""

    kind: OpKind
    tokens: tuple[str, ...]
    request_id: str | None

    def observation_text(self) -> str:
        """The line the proxy injects into the CUA's next observation."""
        marker = f"⟦PLVA_TOOL_RESULT:{self.kind}"
        if self.request_id:
            marker += f":{self.request_id}"
        marker += "⟧"
        if not self.tokens:
            return f"{marker} no tokens match"
        rendered = ", ".join(f"«{token}»" for token in self.tokens)
        label = "tokens in answer order" if self.kind == "sort" else "matching tokens"
        return f"{marker} {label}: {rendered}"


def _clean_returned_item(raw: object) -> str:
    if not isinstance(raw, str):
        raise ValueError("answer entries must be item texts")
    return raw.strip()


def _validate_request(request: SemanticOpRequest) -> None:
    if request.kind not in OP_KINDS:
        raise LocalLLMError(f"unsupported semantic operation: {request.kind}")
    if not request.instruction.strip() or len(request.instruction) > _MAX_INSTRUCTION_CHARS:
        raise LocalLLMError("semantic operation instruction is empty or too long")
    if not _MIN_TOKENS <= len(request.tokens) <= _MAX_TOKENS:
        raise LocalLLMError(f"semantic operations accept {_MIN_TOKENS}..{_MAX_TOKENS} tokens")
    if len(set(request.tokens)) != len(request.tokens):
        raise LocalLLMError("semantic operation tokens must be unique")
    for token in request.tokens:
        if _TOKEN_SHAPE.fullmatch(token) is None:
            raise LocalLLMError("semantic operation received a malformed token")
    if request.select_count is not None and (
        request.kind != "select" or not 0 <= request.select_count <= len(request.tokens)
    ):
        raise LocalLLMError("select_count is only valid for select and must fit the token list")


def _answer_schema(request: SemanticOpRequest, values: Sequence[str]) -> dict[str, object]:
    """A json_schema whose only vocabulary is the exact item texts.

    Grammar-capable servers enforce this at decode time, so every answer entry
    is a byte-exact copy of one input value and the local value→token mapping
    can never be ambiguous; ``_validate_answer`` stays as the
    server-independent backstop for servers without grammar support.
    """
    count = len(values)
    items: dict[str, object] = {"type": "string", "enum": list(values)}
    answer: dict[str, object] = {"type": "array", "items": items, "maxItems": count}
    if request.kind == "sort":
        answer["minItems"] = count
    elif request.select_count is not None:
        answer["minItems"] = request.select_count
        answer["maxItems"] = request.select_count
    return {"type": "object", "properties": {"answer": answer}, "required": ["answer"]}


def _validate_answer(
    request: SemanticOpRequest, token_by_value: Mapping[str, str], returned: object
) -> tuple[str, ...]:
    if not isinstance(returned, list):
        raise ValueError('the reply must be {"answer": [...]}')
    items = tuple(_clean_returned_item(entry) for entry in returned)
    unknown = [item for item in items if item not in token_by_value]
    if unknown:
        raise ValueError("every entry must be one of the item texts, copied exactly")
    if len(set(items)) != len(items):
        raise ValueError("each item may appear at most once")
    if request.kind == "sort" and len(items) != len(request.tokens):
        raise ValueError("sort must return every item exactly once")
    if (
        request.kind == "select"
        and request.select_count is not None
        and len(items) != request.select_count
    ):
        raise ValueError(f"select must return exactly {request.select_count} items")
    return tuple(token_by_value[item] for item in items)


class SemanticExecutor:
    """Fail-closed executor for fuzzy operations the deterministic library can't cover."""

    def __init__(self, client: LoopbackLLMClient, *, resolver: TokenResolver) -> None:
        self._client = client
        self._resolver = resolver

    def execute(self, request: SemanticOpRequest) -> SemanticOpResult:
        """Run one operation; raises ``LocalLLMError`` (or the resolver's own
        policy error) instead of ever returning a doubtful answer."""
        _validate_request(request)
        token_by_value: dict[str, str] = {}
        values: list[str] = []
        for token in request.tokens:
            value = self._resolver(token).strip()
            if not value or len(value) > _MAX_VALUE_CHARS:
                raise LocalLLMError("a resolved value is empty or too long for local reasoning")
            if value in token_by_value:
                # The vault maps one value to one placeholder, so this signals a
                # broken resolver; an ambiguous value→token mapping cannot be safe.
                raise LocalLLMError("two tokens resolved to the same value; failing closed")
            token_by_value[value] = token
            values.append(value)
        prompt = self._prompt(request, values)
        schema = _answer_schema(request, values)
        max_tokens = min(2000, 100 + sum(len(value) for value in values))
        payload = self._client.complete_json(
            system=_SYSTEM, user=prompt, max_tokens=max_tokens, schema=schema
        )
        try:
            answer = _validate_answer(request, token_by_value, payload.get("answer"))
        except ValueError as first_error:
            retry_prompt = (
                f"{prompt}\n\nYour previous reply was invalid: {first_error}. "
                'Reply again with ONLY {"answer": [...]} using exact item texts.'
            )
            payload = self._client.complete_json(
                system=_SYSTEM, user=retry_prompt, max_tokens=max_tokens, schema=schema
            )
            try:
                answer = _validate_answer(request, token_by_value, payload.get("answer"))
            except ValueError as error:
                raise LocalLLMError(
                    f"local model did not produce a valid token-only answer: {error}"
                ) from error
        return SemanticOpResult(kind=request.kind, tokens=answer, request_id=request.request_id)

    def _prompt(self, request: SemanticOpRequest, values: Sequence[str]) -> str:
        if request.kind == "sort":
            task = "sort — return ALL items, reordered by the instruction."
            contract = (
                'Reply with JSON {"answer": [...]} listing every item exactly once, '
                "in answer order."
            )
        else:
            task = "select — return ONLY the items that match the instruction."
            contract = 'Reply with JSON {"answer": [...]} containing only the matching items.'
            if request.select_count is not None:
                contract += f" Return exactly {request.select_count} items."
        lines = [f"Task: {task}", f"Instruction: {request.instruction.strip()}", "Items:"]
        lines.extend(f"- {value}" for value in values)
        lines.append(contract)
        return "\n".join(lines)
