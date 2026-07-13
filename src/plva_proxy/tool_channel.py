"""Token tool channel + mediator automation over the proxy seams.

The CUA invokes a local operation by
emitting a strict, bounded free-text marker
(``⟦PLVA_TOOL:<verb>:<tokens>:<id>[:<instruction>]⟧``); the proxy validates it
against a fixed verb allowlist, executes locally, and injects a value-free
``⟦PLVA_TOOL_RESULT:…⟧`` line into the next observation. Because marker
compliance varies run to run, ``run_op`` is also callable directly by the
proxy/app (the mandatory fallback path).

Routing: ``sort`` with no instruction is the deterministic library path:
resolve locally, order lexicographically, return tokens; ``sort`` with an
instruction and ``select`` delegate to the sandboxed local model through
:class:`SemanticExecutor`, whose return is tokens-only by construction.

Approval automation: on an approval-gated resolution denial the channel consults
the :class:`Mediator` and mints an exact vault grant on approval; value-free
trace events feed the deterministic watchdog trigger, and a ``halt`` verdict
flips a fail-closed flag that blocks every further forwarded request.

Everything here fails closed: unknown verbs, forged tokens, policy denials,
executor/mediator failures, and the halt state all end in "no real value
moves" plus at most a value-free notice to the CUA.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Mapping
from typing import Any, Final

from .local_llm import LocalLLMError
from .mediator import ApprovalRequest, Mediator, TraceEvent
from .privacy import (
    HistoryScrubber,
    PrivacyError,
    SessionVault,
    action_references,
    call_target,
)
from .semantic_executor import SemanticExecutor, SemanticOpRequest

logger = logging.getLogger("plva.tools")

_TOKEN: Final = r"«?[A-Z][A-Z0-9_]*_\d+_[0-9a-f]{4}»?"
_MARKER: Final = re.compile(
    rf"⟦PLVA_TOOL:(?P<verb>[a-z_]{{2,16}}):(?P<tokens>{_TOKEN}(?:\s*,\s*{_TOKEN}){{0,39}})"
    r":(?P<id>[a-z0-9_]{1,24})(?::(?P<instruction>[^⟧]{1,200}))?⟧"
)
_ALLOWED_VERBS: Final = frozenset({"sort", "select"})
_MAX_MARKERS_PER_COMPLETION: Final = 2
_MAX_PENDING: Final = 8
_APPROVAL_DENIED_MESSAGE: Final = "placeholder requires matching local approval"

TOOL_TEACHING: Final = (
    "[PLVA_TOOLS] Local private tools can compute over hidden values for you. To run one, "
    "write a marker in your thought text exactly like "
    "⟦PLVA_TOOL:sort:«EMAIL_1_ab12»,«EMAIL_2_ab12»:req1⟧ (alphabetical sort) or "
    "⟦PLVA_TOOL:sort:«NAME_1_ab12»,«NAME_2_ab12»:req2:most recent first⟧ (custom order) or "
    "⟦PLVA_TOOL:select:«EMAIL_1_ab12»,«EMAIL_2_ab12»:req3:the personal address⟧. "
    "Verbs: sort (optional :instruction) and select (:instruction required). Use only exact "
    "tokens from the manifest and a fresh short lowercase id each time. The answer arrives in "
    "your next observation as ⟦PLVA_TOOL_RESULT:…⟧ listing tokens in answer order. Tokens stay "
    "hidden values you cannot see; never guess them."
)


class ToolChannel:
    """Session-scoped bridge between the proxy hooks and the local LLM component."""

    def __init__(
        self,
        vault: SessionVault,
        scrubber: HistoryScrubber,
        *,
        executor: SemanticExecutor | None = None,
        mediator: Mediator | None = None,
    ) -> None:
        self._vault = vault
        self._scrubber = scrubber
        self._executor = executor
        self._mediator = mediator
        self._lock = threading.RLock()
        self._pending: list[str] = []
        self._seen_ids: set[str] = set()
        self._events: list[TraceEvent] = []
        self._step = 0
        self._last_review_step = -(10**9)
        self._halted: str | None = None
        self.task_context = ""

    # ------------------------------------------------------------- request leg

    def ensure_not_halted(self) -> None:
        with self._lock:
            if self._halted is not None:
                raise PrivacyError(f"CUA halted by local watchdog: {self._halted}")

    def teaching_text(self) -> str:
        return TOOL_TEACHING

    def drain_pending(self) -> tuple[str, ...]:
        with self._lock:
            drained = tuple(self._pending)
            self._pending.clear()
            return drained

    def note_step(self) -> None:
        with self._lock:
            self._step += 1

    # ------------------------------------------------------------ response leg

    def scan_completion_text(self, text: str) -> None:
        """Find, validate, and execute tool markers; queue value-free results."""
        for count, match in enumerate(_MARKER.finditer(text)):
            if count >= _MAX_MARKERS_PER_COMPLETION:
                break
            request_id = match.group("id")
            with self._lock:
                if request_id in self._seen_ids:
                    continue
                self._seen_ids.add(request_id)
            verb = match.group("verb")
            tokens = tuple(
                part.strip().strip("«»").strip() for part in match.group("tokens").split(",")
            )
            self.run_op(verb, tokens, match.group("instruction"), request_id)

    def run_op(
        self,
        verb: str,
        tokens: tuple[str, ...],
        instruction: str | None,
        request_id: str | None,
    ) -> str:
        """Execute one operation (marker-invoked or proxy/app-initiated fallback).

        Always returns (and queues) a value-free observation line; failures
        become a denial notice, never an exception to the caller.
        """
        label = f"{verb}:{request_id or 'direct'}"
        try:
            text = self._execute(verb, tokens, instruction, request_id)
            self._record("tool_executed", f"{label} over {len(tokens)} tokens")
        except (PrivacyError, LocalLLMError) as error:
            reason = _safe_reason(error)
            self._record("tool_denied", f"{label}: {reason}")
            marker = f"⟦PLVA_TOOL_RESULT:{verb}"
            if request_id:
                marker += f":{request_id}"
            text = f"{marker}⟧ denied: {reason}"
        self._queue(text)
        return text

    def _execute(
        self,
        verb: str,
        tokens: tuple[str, ...],
        instruction: str | None,
        request_id: str | None,
    ) -> str:
        if verb not in _ALLOWED_VERBS:
            raise PrivacyError("unknown tool verb")
        instruction = instruction.strip() if instruction and instruction.strip() else None
        if verb == "sort" and instruction is None:
            return self._deterministic_sort(tokens, request_id)
        if verb == "select" and instruction is None:
            raise PrivacyError("select requires an instruction")
        if self._executor is None:
            raise PrivacyError("no local executor is configured")
        result = self._executor.execute(
            SemanticOpRequest(
                kind="sort" if verb == "sort" else "select",
                instruction=instruction or "",
                tokens=tokens,
                request_id=request_id,
            )
        )
        return result.observation_text()

    def _deterministic_sort(self, tokens: tuple[str, ...], request_id: str | None) -> str:
        """Deterministic library path: resolve locally, sort lexicographically, emit tokens only."""
        if not 2 <= len(tokens) <= 40 or len(set(tokens)) != len(tokens):
            raise PrivacyError("sort takes 2..40 unique tokens")
        pairs = [(self._vault.resolve(token).casefold(), token) for token in tokens]
        ordered = tuple(token for _value, token in sorted(pairs))
        rendered = ", ".join(f"«{token}»" for token in ordered)
        marker = "⟦PLVA_TOOL_RESULT:sort"
        if request_id:
            marker += f":{request_id}"
        return f"{marker}⟧ tokens in answer order: {rendered}"

    # ---------------------------------------------------- approval automation

    def consult_approval(self, call: Mapping[str, Any]) -> bool:
        """Ask the mediator to mint grants for approval-gated tokens in one call.

        Returns True when at least one grant was minted (the caller retries the
        resolution once); on any other outcome the original denial stands.
        """
        if self._mediator is None:
            return False
        try:
            tool_name = str(call.get("tool_name") or call.get("name") or call.get("action") or "")
            references = action_references(call)
            active = {item["token"]: item for item in self._scrubber.active_manifest()}
            target = call_target(call)
            granted = False
            for token, path in references:
                info = active.get(token)
                if info is None or info.get("safety_level") != "approval":
                    continue
                request = ApprovalRequest(
                    placeholder=token,
                    pii_class=info.get("class", ""),
                    tool_name=tool_name,
                    argument_path=path,
                    target=target,
                    task_context=self.task_context,
                )
                verdict = self._mediator.decide_approval(request)
                if verdict.decision in ("approve", "modify"):
                    self._vault.grant_approval(token, **verdict.grant_kwargs(request))
                    self._record("approval_granted", f"{info.get('class')} token for {tool_name}")
                    granted = True
                else:
                    self._record("approval_denied", f"{info.get('class')} token for {tool_name}")
                if verdict.steering:
                    self._queue(f"[PLVA_MEDIATOR] {verdict.steering}")
            return granted
        except (PrivacyError, ValueError) as error:
            logger.warning("mediator consult failed closed: %s", _safe_reason(error))
            return False

    def record_denial(self, error: PrivacyError) -> None:
        self._record("resolution_denied", _safe_reason(error))

    def is_approval_denial(self, error: PrivacyError) -> bool:
        return _APPROVAL_DENIED_MESSAGE in str(error)

    def maybe_review(self) -> None:
        """Deterministic trigger → one watchdog review per window; halt fails closed."""
        if self._mediator is None:
            return
        with self._lock:
            events = tuple(self._events)
            step = self._step
            window = self._mediator.criteria.window_steps
            if self._halted is not None or step - self._last_review_step < window:
                return
        if not self._mediator.should_review(events):
            return
        with self._lock:
            self._last_review_step = step
        verdict = self._mediator.review_trace(events, task_context=self.task_context)
        logger.warning("watchdog verdict action=%s", verdict.action)
        if verdict.action == "halt":
            with self._lock:
                self._halted = verdict.reason
        elif verdict.action == "warn":
            self._queue(f"[PLVA_MEDIATOR] warning: {verdict.reason}")

    # ------------------------------------------------------------------ shared

    def _record(self, kind: str, detail: str) -> None:
        with self._lock:
            self._events.append(TraceEvent(self._step, kind, detail))
            del self._events[:-200]

    def _queue(self, text: str) -> None:
        with self._lock:
            self._pending.append(text)
            del self._pending[:-_MAX_PENDING]


def _safe_reason(error: Exception) -> str:
    """Error text kept short; our own error strings are value-free by policy."""
    return re.sub(r"\s+", " ", str(error))[:160] or type(error).__name__
