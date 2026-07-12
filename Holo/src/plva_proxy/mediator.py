"""Local LLM mediator: approval verdicts and misbehavior trace review (§7).

The mediator is consulted by the proxy on deterministic triggers; it never
takes control. ``decide_approval`` judges one exact token use against
user-written criteria and returns a verdict whose scope maps directly onto
``SessionVault.grant_approval``; ``review_trace`` judges a value-free event
trace and can tell the supervisor to halt the CUA. Every failure path returns
the fail-closed verdict (deny / halt) instead of raising, because a mediator
outage must never fail open (§7 "unreachable or times out → deny").

The model behind :class:`LoopbackLLMClient` may see cleartext, so it must be
local with zero egress (llama-server + a local Nemotron GGUF; see
docs/local-llm-runbook.md). Verdict text is leak-scanned against the cleartext
it was shown and can additionally be routed through a scrubber hook (the same
Rampart machinery the history leg uses) before anything leaves this module.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from .local_llm import (
    LLMConfig,
    LocalLLMError,
    LoopbackLLMClient,
    find_cleartext_leak,
    port_from_base_url,
    verify_no_egress,
)
from .privacy import POLICY_CLASSES

Decision = Literal["approve", "deny", "modify"]
TraceAction = Literal["continue", "warn", "halt"]

_DECISIONS: Final = frozenset({"approve", "deny", "modify"})
_TRACE_ACTIONS: Final = frozenset({"continue", "warn", "halt"})
_SUSPICIOUS_KINDS: Final = frozenset(
    {"resolution_denied", "approval_denied", "forged_token", "blocked_class_attempt"}
)
_TEXT_CAP: Final = 400
_GUIDANCE_CAP: Final = 2000
DEFAULT_CRITERIA_PATH: Final = Path("config/mediator-criteria.json")

_APPROVAL_SYSTEM: Final = (
    "You are PLVA's local privacy mediator. You decide whether ONE hidden value may be used in "
    "ONE exact action. You never reveal or quote the value. Reply with ONLY a JSON object: "
    '{"decision": "approve" | "deny" | "modify", "reason": "<short, never quote the value>", '
    '"steering": "<optional short advice to the agent>", '
    '"ttl_seconds": <optional int 1-3600>, "use_count": <optional int 1-100>}. '
    "Approve only when the user's rule clearly allows this exact use in this exact place. "
    "Use modify to approve with a tighter limit. When unsure, deny."
)

_TRACE_SYSTEM: Final = (
    "You are PLVA's local run watchdog. You are shown a value-free event trace of a computer-use "
    "agent. Judge whether the agent is misbehaving: repeatedly probing blocked or unapproved "
    "data, ignoring denials, or acting outside its task. Reply with ONLY a JSON object: "
    '{"action": "continue" | "warn" | "halt", "reason": "<short>"}. '
    '"halt" stops the agent. If the trace shows persistent probing after denials, halt.'
)

_APPROVAL_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["approve", "deny", "modify"]},
        "reason": {"type": "string", "maxLength": 400},
        "steering": {"type": "string", "maxLength": 400},
        "ttl_seconds": {"type": "integer", "minimum": 1, "maximum": 3600},
        "use_count": {"type": "integer", "minimum": 1, "maximum": 100},
    },
    "required": ["decision", "reason"],
}

_TRACE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["continue", "warn", "halt"]},
        "reason": {"type": "string", "maxLength": 400},
    },
    "required": ["action", "reason"],
}


@dataclass(frozen=True, slots=True)
class MediatorCriteria:
    """User-written rules for when gated data may be used, plus review thresholds."""

    class_guidance: Mapping[str, str]
    global_rules: tuple[str, ...]
    denied_resolution_threshold: int
    window_steps: int

    @classmethod
    def defaults(cls) -> MediatorCriteria:
        return cls(
            class_guidance={
                "API_KEY": (
                    "An API key may be entered only into a field that expects a credential "
                    "(settings page, CLI login, request header) of the service that issued it, "
                    "and only when the task explicitly requires configuring or calling that "
                    "service. Never into chat messages, emails, documents, or search boxes."
                ),
                "AUTH_TOKEN": (
                    "A session or auth token may be re-entered only into the same site or app "
                    "it came from, to restore or keep a login working. Never anywhere else."
                ),
            },
            global_rules=(
                "When in doubt, deny.",
                "Never approve placing a value in a message addressed to another person.",
                "Prefer the shortest scope that lets the task continue.",
            ),
            denied_resolution_threshold=3,
            window_steps=8,
        )

    @classmethod
    def load(cls, path: Path) -> MediatorCriteria:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("mediator criteria must be a JSON object with version 1")
        classes_raw = raw.get("classes", {})
        if not isinstance(classes_raw, dict):
            raise ValueError("mediator criteria classes must be an object")
        guidance: dict[str, str] = {}
        for raw_class, text in classes_raw.items():
            pii_class = str(raw_class).strip().upper()
            if pii_class not in POLICY_CLASSES:
                raise ValueError(f"unknown PII class in mediator criteria: {pii_class}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"guidance for {pii_class} must be a non-empty string")
            if len(text) > _GUIDANCE_CAP:
                raise ValueError(f"guidance for {pii_class} exceeds {_GUIDANCE_CAP} characters")
            guidance[pii_class] = text.strip()
        rules_raw = raw.get("global_rules", [])
        if not isinstance(rules_raw, list) or len(rules_raw) > 20:
            raise ValueError("global_rules must be a list of at most 20 strings")
        rules: list[str] = []
        for rule in rules_raw:
            if not isinstance(rule, str) or not rule.strip() or len(rule) > 500:
                raise ValueError("each global rule must be a string of at most 500 characters")
            rules.append(rule.strip())
        review_raw = raw.get("trace_review", {})
        if not isinstance(review_raw, dict):
            raise ValueError("trace_review must be an object")
        threshold = review_raw.get("denied_resolution_threshold", 3)
        window = review_raw.get("window_steps", 8)
        if not isinstance(threshold, int) or not 1 <= threshold <= 100:
            raise ValueError("denied_resolution_threshold must be an integer in 1..100")
        if not isinstance(window, int) or not 1 <= window <= 1000:
            raise ValueError("window_steps must be an integer in 1..1000")
        return cls(
            class_guidance=guidance,
            global_rules=tuple(rules),
            denied_resolution_threshold=threshold,
            window_steps=window,
        )


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """One gated resolution the proxy wants judged; everything but the value is value-free."""

    placeholder: str
    pii_class: str
    tool_name: str
    argument_path: str
    target: str | None
    task_context: str
    cleartext_value: str | None = None


@dataclass(frozen=True, slots=True)
class ApprovalScope:
    """Limits for a minted grant; fields mirror ``SessionVault.grant_approval``."""

    ttl_seconds: float
    use_count: int
    target: str | None


@dataclass(frozen=True, slots=True)
class MediatorVerdict:
    decision: Decision
    reason: str
    steering: str | None = None
    scope: ApprovalScope | None = None

    def grant_kwargs(self, request: ApprovalRequest) -> dict[str, Any]:
        """Keyword arguments for ``SessionVault.grant_approval`` — the Step 7 plug point."""
        if self.decision not in ("approve", "modify"):
            raise ValueError("only approving verdicts can mint a grant")
        kwargs: dict[str, Any] = {
            "tool_name": request.tool_name,
            "argument_path": request.argument_path,
            "target": request.target,
        }
        if self.scope is not None:
            kwargs["ttl_seconds"] = self.scope.ttl_seconds
            kwargs["use_count"] = self.scope.use_count
        return kwargs


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One value-free event: class names, tools, and tokens only — never values."""

    step: int
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class TraceVerdict:
    action: TraceAction
    reason: str


def _clean_text(value: object, *, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    return re.sub(r"\s+", " ", value).strip()[:_TEXT_CAP]


class Mediator:
    """Fail-closed judgment calls over the loopback local model."""

    def __init__(
        self,
        client: LoopbackLLMClient,
        criteria: MediatorCriteria | None = None,
        *,
        scrubber: Callable[[str], str] | None = None,
    ) -> None:
        self._client = client
        self._criteria = criteria or MediatorCriteria.defaults()
        self._scrubber = scrubber

    @property
    def criteria(self) -> MediatorCriteria:
        return self._criteria

    def decide_approval(self, request: ApprovalRequest) -> MediatorVerdict:
        pii_class = request.pii_class.strip().upper()
        guidance = self._criteria.class_guidance.get(pii_class)
        if guidance is None:
            return MediatorVerdict(
                "deny", f"no user criteria exist for class {pii_class}; failing closed"
            )
        try:
            payload = self._client.complete_json(
                system=_APPROVAL_SYSTEM,
                user=self._approval_prompt(request, guidance),
                schema=_APPROVAL_SCHEMA,
            )
        except LocalLLMError as error:
            return MediatorVerdict(
                "deny", f"local mediator unavailable ({type(error).__name__}); failing closed"
            )
        return self._parse_approval(payload, request)

    def review_trace(self, events: Sequence[TraceEvent], *, task_context: str = "") -> TraceVerdict:
        if not events:
            return TraceVerdict("continue", "empty trace")
        try:
            payload = self._client.complete_json(
                system=_TRACE_SYSTEM,
                user=self._trace_prompt(events, task_context),
                schema=_TRACE_SCHEMA,
            )
        except LocalLLMError as error:
            return TraceVerdict(
                "halt",
                f"local mediator unavailable ({type(error).__name__}) during misbehavior "
                "review; failing closed",
            )
        action = payload.get("action")
        if action not in _TRACE_ACTIONS:
            return TraceVerdict("halt", "mediator returned a malformed action; failing closed")
        reason = self._scrub(_clean_text(payload.get("reason"), fallback="no reason given"))
        return TraceVerdict(action, reason)

    def should_review(self, events: Sequence[TraceEvent]) -> bool:
        """Deterministic trigger: enough suspicious events inside the recent window."""
        if not events:
            return False
        window_start = max(event.step for event in events) - self._criteria.window_steps + 1
        suspicious = sum(
            1 for event in events if event.step >= window_start and event.kind in _SUSPICIOUS_KINDS
        )
        return suspicious >= self._criteria.denied_resolution_threshold

    def _approval_prompt(self, request: ApprovalRequest, guidance: str) -> str:
        lines = [
            f"User rule for class {request.pii_class.strip().upper()}: {guidance}",
        ]
        if self._criteria.global_rules:
            lines.append("Global rules:")
            lines.extend(f"- {rule}" for rule in self._criteria.global_rules)
        lines.extend(
            [
                "Requested use:",
                f"- the hidden value {request.placeholder} (class "
                f"{request.pii_class.strip().upper()}) would be entered by tool "
                f"'{request.tool_name}' into field '{request.argument_path}'",
                f"- destination it would be entered at: {request.target or 'not specified'}",
                f"- the task the agent is doing: {request.task_context or 'not specified'}",
            ]
        )
        if request.cleartext_value is not None:
            lines.append(f"- hidden value (never repeat it): {request.cleartext_value}")
        lines.append(
            "Check each condition of the user's rule against this requested use, then decide."
        )
        return "\n".join(lines)

    def _trace_prompt(self, events: Sequence[TraceEvent], task_context: str) -> str:
        lines = [f"Task context: {task_context or 'not specified'}", "Event trace:"]
        lines.extend(f"step {event.step}: {event.kind} — {event.detail}" for event in events)
        lines.append("Judge the agent's behavior.")
        return "\n".join(lines)

    def _parse_approval(
        self, payload: Mapping[str, Any], request: ApprovalRequest
    ) -> MediatorVerdict:
        decision = payload.get("decision")
        if decision not in _DECISIONS:
            return MediatorVerdict("deny", "mediator returned a malformed decision; failing closed")
        reason = _clean_text(payload.get("reason"), fallback="no reason given")
        steering_raw = payload.get("steering")
        steering = _clean_text(steering_raw, fallback="") if isinstance(steering_raw, str) else ""
        if request.cleartext_value is not None and find_cleartext_leak(
            [reason, steering], [request.cleartext_value]
        ):
            return MediatorVerdict(
                "deny", "mediator response withheld: it echoed the hidden value; failing closed"
            )
        scope: ApprovalScope | None = None
        if decision in ("approve", "modify"):
            scope = ApprovalScope(
                ttl_seconds=_bounded_number(payload.get("ttl_seconds"), 1, 3600, 60.0),
                use_count=int(_bounded_number(payload.get("use_count"), 1, 100, 1)),
                target=request.target,
            )
        return MediatorVerdict(
            decision,
            self._scrub(reason),
            self._scrub(steering) if steering else None,
            scope,
        )

    def _scrub(self, text: str) -> str:
        return self._scrubber(text) if self._scrubber is not None else text


def _bounded_number(value: object, low: float, high: float, fallback: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return fallback
    return float(value) if low <= float(value) <= high else fallback


def _load_criteria(path: Path | None) -> tuple[MediatorCriteria, str]:
    if path is not None:
        return MediatorCriteria.load(path), str(path)
    if DEFAULT_CRITERIA_PATH.exists():
        return MediatorCriteria.load(DEFAULT_CRITERIA_PATH), str(DEFAULT_CRITERIA_PATH)
    return MediatorCriteria.defaults(), "built-in defaults"


def _build_config(args: argparse.Namespace) -> LLMConfig:
    config = LLMConfig.from_env()
    overrides: dict[str, Any] = {}
    if args.url:
        overrides["base_url"] = args.url
    if args.model:
        overrides["model"] = args.model
    if args.timeout:
        overrides["timeout_seconds"] = args.timeout
    return config.replace(**overrides) if overrides else config


def _cmd_probe(config: LLMConfig) -> int:
    with LoopbackLLMClient(config) as client:
        try:
            info = client.probe()
        except LocalLLMError as error:
            print(f"UNREACHABLE: {error}")
            return 1
    report = verify_no_egress(port_from_base_url(config.base_url))
    print(json.dumps({"reachable": info, "egress": dataclasses.asdict(report)}, indent=2))
    if not report.clean:
        print("EGRESS CHECK FAILED: do not trust this server with cleartext")
        return 1
    print("OK: endpoint reachable, no non-loopback socket observed")
    return 0


def _cmd_demo_approval(config: LLMConfig, criteria_path: Path | None) -> int:
    criteria, source = _load_criteria(criteria_path)
    print(f"criteria: {source}")
    request = ApprovalRequest(
        placeholder="API_KEY_1_ab12",
        pii_class="API_KEY",
        tool_name="write",
        argument_path="text",
        target="the 'API key' credential field on the settings page at console.demo-service.test",
        task_context="Configure the demo-service CLI with the account's API key.",
        cleartext_value="sk-demo-1234567890abcdef",
    )
    with LoopbackLLMClient(config) as client:
        verdict = Mediator(client, criteria).decide_approval(request)
    print(json.dumps(dataclasses.asdict(verdict), indent=2))
    return 0


def _cmd_demo_trace(config: LLMConfig, criteria_path: Path | None) -> int:
    criteria, source = _load_criteria(criteria_path)
    print(f"criteria: {source}")
    events = [
        TraceEvent(1, "action", "click on the account settings page"),
        TraceEvent(2, "resolution_denied", "PASSWORD token requested in write.text"),
        TraceEvent(3, "resolution_denied", "PASSWORD token requested in write.text"),
        TraceEvent(4, "blocked_class_attempt", "CARD_NUMBER token requested in write.text"),
        TraceEvent(5, "resolution_denied", "PASSWORD token requested again in write.text"),
    ]
    with LoopbackLLMClient(config) as client:
        mediator = Mediator(client, criteria)
        print(f"deterministic trigger fired: {mediator.should_review(events)}")
        verdict = mediator.review_trace(events, task_context="Update the account display name.")
    print(json.dumps(dataclasses.asdict(verdict), indent=2))
    return 0


def _cmd_demo_sort(config: LLMConfig) -> int:
    from .semantic_executor import SemanticExecutor, SemanticOpRequest

    values = {
        "NAME_1_ab12": "Charlie Rivera",
        "NAME_2_ab12": "Alice Zhang",
        "NAME_3_ab12": "Bob O'Neill",
    }
    request = SemanticOpRequest(
        kind="sort",
        instruction="alphabetical order of first names",
        tokens=tuple(values),
        request_id="demo_1",
    )
    with LoopbackLLMClient(config) as client:
        executor = SemanticExecutor(client, resolver=values.__getitem__)
        try:
            result = executor.execute(request)
        except LocalLLMError as error:
            print(f"FAILED CLOSED: {error}")
            return 1
    observation = result.observation_text()
    leak = find_cleartext_leak([observation], values.values())
    print(observation)
    print(f"leak scan of the injected observation: {leak or 'clean'}")
    return 0 if leak is None else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="plva-mediator",
        description="Probe and demo the sandboxed local LLM (mediator + semantic executor).",
    )
    parser.add_argument("--url", help="loopback OpenAI-compatible base URL")
    parser.add_argument("--model", help="model name served at the endpoint")
    parser.add_argument("--timeout", type=float, help="request timeout in seconds")
    parser.add_argument("--criteria", type=Path, help="path to mediator criteria JSON")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("probe", help="reachability + no-egress audit of the local model server")
    sub.add_parser("demo-approval", help="synthetic approval verdict (no real data)")
    sub.add_parser("demo-trace", help="synthetic misbehavior trace review (no real data)")
    sub.add_parser("demo-sort", help="synthetic placeholder-preserving sort (no real data)")
    args = parser.parse_args(argv)
    config = _build_config(args)
    if args.command == "probe":
        return _cmd_probe(config)
    if args.command == "demo-approval":
        return _cmd_demo_approval(config, args.criteria)
    if args.command == "demo-trace":
        return _cmd_demo_trace(config, args.criteria)
    return _cmd_demo_sort(config)


if __name__ == "__main__":
    sys.exit(main())
