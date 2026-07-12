"""Session vault, placeholder painting, action resolution, and history scrubbing."""

from __future__ import annotations

import copy
import io
import json
import re
import secrets
import threading
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol

from PIL import Image, ImageDraw, ImageFont

PLACEHOLDER_SCHEME: Final = (
    "[PLVA_PLACEHOLDERS] Some sensitive values on screen are hidden behind chips written like "
    "«EMAIL_1_ab12» or «PHONE_2_ab12». Each chip stands for a real value you cannot see; its "
    "class label tells you what kind of value it is. Treat it as that real value when planning. "
    "To use it, copy only the inner token (for example EMAIL_1_ab12), exactly and with its "
    "session suffix, into the executed action field. Do not include the decorative « » marks, "
    "change its spelling or case, invent a token, guess the hidden value, or expose a token only "
    "in notes/reasoning. If a hidden sensitive value you need is not listed as a visible PLVA "
    "token in the current observation, do not fabricate or reuse it. Ordinary unmarked UI text "
    "and non-sensitive actions are unaffected; continue the requested task normally."
)
PLACEHOLDER_DUPLICATE_WARNING: Final = (
    "[PLVA_PLACEHOLDERS] Occasionally one real value may have more than one token "
    "across steps; treat each token independently and use the token shown on the field you are "
    "acting on."
)
PLACEHOLDER_INSTRUCTIONS: Final = (
    PLACEHOLDER_SCHEME + " " + PLACEHOLDER_DUPLICATE_WARNING.removeprefix("[PLVA_PLACEHOLDERS] ")
)
PLACEHOLDER_MANIFEST_KEY: Final = "_plva_current_placeholder_manifest"
PLACEHOLDER_MANIFEST_PREFIX: Final = "[PLVA_VISIBLE_PLACEHOLDERS]"
PLACEHOLDER_SYSTEM_BEGIN: Final = "[PLVA_PLACEHOLDERS_BEGIN]"
PLACEHOLDER_SYSTEM_END: Final = "[PLVA_PLACEHOLDERS_END]"
_CREDENTIAL_CLASSES: Final = frozenset(
    {"API_KEY", "AUTH_TOKEN", "PASSWORD", "CVC", "CARD_NUMBER", "PRIVATE_KEY", "SECRET"}
)
_PLACEHOLDER_SHAPE: Final = re.compile(r"\b[A-Z][A-Z0-9_]*_[1-9]\d*_[0-9a-f]{4}\b")
_TOOL_NAME_KEYS: Final = frozenset({"tool_name", "name", "id"})


class PrivacyError(RuntimeError):
    """Raised when privacy transformation cannot be completed safely."""


class RedactorWithAnalysis(Protocol):
    @property
    def latest_analysis(self) -> dict[str, Any]: ...

    def __call__(self, png: bytes) -> bytes: ...

    def start(self) -> None: ...

    def close(self) -> None: ...


TextClassifier = Callable[[tuple[str, ...]], list[dict[str, Any]]]

SAFETY_LEVELS: Final = ("hide_use", "approval", "blocked")
POLICY_CLASSES: Final = (
    "NAME",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "DOB",
    "GOV_ID",
    "SSN",
    "BANK_ACCOUNT",
    "CARD_NUMBER",
    "CVC",
    "PASSWORD",
    "API_KEY",
    "AUTH_TOKEN",
    "PRIVATE_KEY",
    "SECRET",
)
DEFAULT_SAFETY_LEVELS: Final[dict[str, str]] = {
    "NAME": "hide_use",
    "EMAIL": "hide_use",
    "PHONE": "hide_use",
    "ADDRESS": "hide_use",
    "DOB": "hide_use",
    "API_KEY": "approval",
    "AUTH_TOKEN": "approval",
    "GOV_ID": "blocked",
    "SSN": "blocked",
    "BANK_ACCOUNT": "blocked",
    "CARD_NUMBER": "blocked",
    "CVC": "blocked",
    "PASSWORD": "blocked",
    "PRIVATE_KEY": "blocked",
    "SECRET": "blocked",
}


class SafetyPolicy:
    """Validated per-class safety levels; unknown classes fail closed as blocked."""

    def __init__(self, levels: Mapping[str, str] | None = None) -> None:
        selected = dict(DEFAULT_SAFETY_LEVELS)
        for raw_class, raw_level in (levels or {}).items():
            pii_class = _normalize_class(raw_class)
            level = str(raw_level).strip().lower()
            if level not in SAFETY_LEVELS:
                raise ValueError(f"invalid safety level for {pii_class}")
            selected[pii_class] = level
        self._levels = selected

    def level_for(self, pii_class: str) -> str:
        return self._levels.get(_normalize_class(pii_class), "blocked")

    def snapshot(self) -> dict[str, str]:
        return {pii_class: self.level_for(pii_class) for pii_class in POLICY_CLASSES}

    def prompt(self) -> str:
        grouped = {
            level: [
                pii_class for pii_class, selected in self.snapshot().items() if selected == level
            ]
            for level in SAFETY_LEVELS
        }
        return (
            "[PLVA_SECURITY_POLICY] Active placeholder security levels: "
            f"hide_use={','.join(grouped['hide_use']) or 'none'}; "
            f"approval={','.join(grouped['approval']) or 'none'}; "
            f"blocked={','.join(grouped['blocked']) or 'none'}. "
            "hide_use tokens may be copied verbatim into executed actions; approval tokens require "
            "an explicit local approval and must not be used before approval; blocked classes are "
            "opaque, have no usable token, and must never be guessed or requested from the model. "
            "These restrictions apply only to PLVA-marked sensitive content. They do not restrict "
            "ordinary visible UI text or unrelated actions; continue the user's task normally."
        )


@dataclass(frozen=True, slots=True)
class VaultEntry:
    placeholder: str
    pii_class: str
    value: str
    canonical: str


class SessionVault:
    """Thread-safe, memory-only placeholder map scoped to one proxy process."""

    def __init__(self, *, nonce: str | None = None, policy: SafetyPolicy | None = None) -> None:
        selected_nonce = nonce or secrets.token_hex(2)
        if re.fullmatch(r"[0-9a-f]{4}", selected_nonce) is None:
            raise ValueError("vault nonce must be four lowercase hexadecimal characters")
        self._nonce = selected_nonce
        self._policy = policy or SafetyPolicy()
        self._lock = threading.RLock()
        self._by_key: dict[tuple[str, str], VaultEntry] = {}
        self._by_placeholder: dict[str, VaultEntry] = {}
        self._variants: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    @property
    def nonce(self) -> str:
        return self._nonce

    def store(self, pii_class: str, value: str) -> str:
        normalized_class = _normalize_class(pii_class)
        if self._policy.level_for(normalized_class) == "blocked":
            raise PrivacyError("blocked PII cannot be stored")
        exact = unicodedata.normalize("NFKC", value)
        if not exact.strip():
            raise PrivacyError("cannot vault an empty value")
        canonical = _canonical_value(normalized_class, exact)
        key = (normalized_class, canonical)
        with self._lock:
            existing = self._by_key.get(key)
            if existing is not None:
                if len(re.sub(r"[^\w]", "", exact, flags=re.UNICODE)) >= 3:
                    self._variants[exact] = existing.placeholder
                return existing.placeholder
            counter = self._counters.get(normalized_class, 0) + 1
            self._counters[normalized_class] = counter
            placeholder = f"{normalized_class}_{counter}_{self._nonce}"
            if placeholder in self._by_placeholder:
                raise PrivacyError("placeholder collision")
            entry = VaultEntry(placeholder, normalized_class, exact, canonical)
            self._by_key[key] = entry
            self._by_placeholder[placeholder] = entry
            if len(re.sub(r"[^\w]", "", exact, flags=re.UNICODE)) >= 3:
                self._variants[exact] = placeholder
            return placeholder

    def resolve(self, placeholder: str) -> str:
        with self._lock:
            entry = self._by_placeholder.get(placeholder)
            if entry is None:
                raise PrivacyError("placeholder was not issued by this session")
            level = self._policy.level_for(entry.pii_class)
            if level == "blocked":
                raise PrivacyError("placeholder is blocked by policy")
            if level == "approval":
                raise PrivacyError("placeholder requires local approval")
            return entry.value

    def safety_level(self, pii_class: str) -> str:
        return self._policy.level_for(pii_class)

    def policy_snapshot(self) -> dict[str, str]:
        return self._policy.snapshot()

    def entries(self) -> tuple[dict[str, str], ...]:
        with self._lock:
            return tuple(
                {
                    "placeholder": entry.placeholder,
                    "class": entry.pii_class,
                    "value": entry.value,
                    "safety_level": self._policy.level_for(entry.pii_class),
                }
                for entry in self._by_placeholder.values()
            )

    def resolve_text(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            return self.resolve(match.group(0))

        return _PLACEHOLDER_SHAPE.sub(replace, text)

    def scrub_plain(self, text: str) -> str:
        with self._lock:
            variants = sorted(self._variants.items(), key=lambda item: len(item[0]), reverse=True)
        protected: list[str] = []
        cursor = 0
        for match in _PLACEHOLDER_SHAPE.finditer(text):
            protected.append(_replace_variants(text[cursor : match.start()], variants))
            protected.append(match.group(0))
            cursor = match.end()
        protected.append(_replace_variants(text[cursor:], variants))
        return "".join(protected)

    def dispose(self) -> None:
        with self._lock:
            self._by_key.clear()
            self._by_placeholder.clear()
            self._variants.clear()
            self._counters.clear()

    def placeholders(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._by_placeholder)


class VaultRedactor:
    """Wrap a findings-capable redactor and paint vault-owned placeholder chips."""

    def __init__(self, redactor: RedactorWithAnalysis, vault: SessionVault) -> None:
        self._redactor = redactor
        self._vault = vault
        self._analysis: dict[str, Any] = {}
        self._lock = threading.RLock()

    @property
    def latest_analysis(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._analysis)

    def start(self) -> None:
        self._redactor.start()

    def close(self) -> None:
        try:
            self._redactor.close()
        finally:
            self._vault.dispose()
            with self._lock:
                self._analysis = {}

    def __call__(self, png: bytes) -> bytes:
        painted, _ = self.redact_with_manifest(png)
        return painted

    def redact_with_manifest(self, png: bytes) -> tuple[bytes, tuple[dict[str, str], ...]]:
        """Redact one frame and atomically return its value-free visible-token manifest."""

        with self._lock:
            return self._redact_with_manifest(png)

    def _redact_with_manifest(self, png: bytes) -> tuple[bytes, tuple[dict[str, str], ...]]:
        redacted = self._redactor(png)
        analysis = self._redactor.latest_analysis
        findings = analysis.get("findings")
        if not isinstance(findings, list):
            raise PrivacyError("redactor did not emit OCR findings")
        chips: list[tuple[tuple[float, float, float, float], tuple[str, ...]]] = []
        enriched: list[dict[str, Any]] = []
        manifest: list[dict[str, str]] = []
        manifested: set[str] = set()
        for raw in findings:
            if not isinstance(raw, dict):
                raise PrivacyError("redactor emitted an invalid finding")
            finding = copy.deepcopy(raw)
            placeholders: list[str] = []
            values = _coalesce_finding_values(finding)
            finding["values"] = values
            if values:
                for raw_value in values:
                    if not isinstance(raw_value, dict):
                        raise PrivacyError("redactor emitted an invalid PII value")
                    label = raw_value.get("label")
                    value = raw_value.get("value")
                    if not isinstance(label, str) or not isinstance(value, str):
                        raise PrivacyError("redactor omitted a recognized PII value")
                    safety_level = self._vault.safety_level(label)
                    raw_value["safety_level"] = safety_level
                    if safety_level == "blocked":
                        continue
                    placeholder = self._vault.store(label, value)
                    raw_value["placeholder"] = placeholder
                    placeholders.append(placeholder)
                    if placeholder not in manifested:
                        manifest.append(
                            {
                                "token": placeholder,
                                "class": _normalize_class(label),
                                "safety_level": safety_level,
                            }
                        )
                        manifested.add(placeholder)
            finding["placeholders"] = list(dict.fromkeys(placeholders))
            if placeholders:
                try:
                    box = tuple(float(finding[key]) for key in ("x1", "y1", "x2", "y2"))
                except (KeyError, TypeError, ValueError) as exc:
                    raise PrivacyError("redactor omitted a PII bounding box") from exc
                chips.append((box, tuple(dict.fromkeys(placeholders))))  # type: ignore[arg-type]
            enriched.append(finding)
        painted = _paint_chips(redacted, chips)
        analysis["findings"] = enriched
        analysis["vault_placeholders"] = sum(len(placeholders) for _, placeholders in chips)
        analysis["policy"] = self._vault.policy_snapshot()
        self._analysis = analysis
        return painted, tuple(manifest)


@dataclass(frozen=True, slots=True)
class StubSpan:
    pii_class: str
    value: str
    bounding_box: tuple[int, int, int, int]


class StubRedactor:
    """Deterministic §5 detector stub with configurable fixture spans or no detections."""

    def __init__(self, spans: tuple[StubSpan, ...] = ()) -> None:
        self._spans = spans
        self._analysis: dict[str, Any] = {}

    @property
    def latest_analysis(self) -> dict[str, Any]:
        return copy.deepcopy(self._analysis)

    def start(self) -> None:
        return

    def close(self) -> None:
        self._analysis = {}

    def __call__(self, png: bytes) -> bytes:
        try:
            with Image.open(io.BytesIO(png)) as loaded:
                image = loaded.convert("RGB")
        except (OSError, ValueError) as exc:
            raise PrivacyError("stub received an invalid image") from exc
        draw = ImageDraw.Draw(image)
        findings: list[dict[str, Any]] = []
        for span in self._spans:
            x1, y1, x2, y2 = span.bounding_box
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > image.width or y2 > image.height:
                raise PrivacyError("stub span is outside the image")
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=(5, 8, 7))
            findings.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "text": span.value,
                    "detector_score": 1.0,
                    "ocr_confidence": 1.0,
                    "labels": [span.pii_class],
                    "sources": ["STUB"],
                    "values": [
                        {
                            "label": span.pii_class,
                            "value": span.value,
                            "start": 0,
                            "end": len(span.value),
                            "score": 1.0,
                            "source": "stub",
                        }
                    ],
                    "sensitive": True,
                    "uncertain": False,
                }
            )
        output = io.BytesIO()
        image.save(output, format="PNG")
        self._analysis = {
            "backend": "stub",
            "counts": {"fused": len(self._spans), "ocr_sensitive": len(self._spans)},
            "timings": {"workerTotalMs": 0},
            "findings": findings,
        }
        return output.getvalue()


class HistoryScrubber:
    """Plain-vault scrub followed by accelerated Rampart classification."""

    def __init__(self, vault: SessionVault, classify: TextClassifier) -> None:
        self._vault = vault
        self._classify = classify
        self._lock = threading.Lock()
        self._diagnostics: dict[str, int | str] = {
            "status": "idle",
            "texts_scanned": 0,
            "plain_vault_hits": 0,
            "semantic_hits": 0,
            "blocked_hits": 0,
            "error_code": "none",
        }

    def diagnostics(self) -> dict[str, int | str]:
        with self._lock:
            return dict(self._diagnostics)

    def scrub(self, texts: tuple[str, ...]) -> tuple[str, ...]:
        plain = tuple(self._vault.scrub_plain(text) for text in texts)
        plain_hits = sum(
            max(0, len(_PLACEHOLDER_SHAPE.findall(after)) - len(_PLACEHOLDER_SHAPE.findall(before)))
            for before, after in zip(texts, plain, strict=True)
        )
        classifier_inputs = tuple(
            _PLACEHOLDER_SHAPE.sub(lambda match: " " * len(match.group(0)), text) for text in plain
        )
        try:
            classifications = self._classify(classifier_inputs)
        except Exception as exc:
            self._set_diagnostics("failed", len(texts), plain_hits, 0, 0, "classifier_failed")
            raise PrivacyError("history classifier failed") from exc
        if len(classifications) != len(plain):
            self._set_diagnostics("failed", len(texts), plain_hits, 0, 0, "result_count")
            raise PrivacyError("history classifier returned the wrong result count")
        semantic_hits = sum(
            len(classification.get("values", []))
            for classification in classifications
            if isinstance(classification.get("values"), list)
        )
        scrubbed: list[str] = []
        blocked_hits = 0
        try:
            for text, classified, classification in zip(
                plain, classifier_inputs, classifications, strict=True
            ):
                rewritten, blocked = self._apply_classification(
                    text, classified, classification
                )
                scrubbed.append(rewritten)
                blocked_hits += blocked
        except PrivacyError:
            self._set_diagnostics(
                "failed",
                len(texts),
                plain_hits,
                semantic_hits,
                blocked_hits,
                "transform_failed",
            )
            raise
        self._set_diagnostics(
            "passed", len(texts), plain_hits, semantic_hits, blocked_hits, "none"
        )
        return tuple(scrubbed)

    def _set_diagnostics(
        self,
        status: str,
        texts: int,
        plain_hits: int,
        semantic_hits: int,
        blocked_hits: int,
        error_code: str,
    ) -> None:
        with self._lock:
            self._diagnostics = {
                "status": status,
                "texts_scanned": texts,
                "plain_vault_hits": plain_hits,
                "semantic_hits": semantic_hits,
                "blocked_hits": blocked_hits,
                "error_code": error_code,
            }

    def _apply_classification(
        self, text: str, classified: str, classification: dict[str, Any]
    ) -> tuple[str, int]:
        values = classification.get("values")
        sensitive = bool(classification.get("sensitive"))
        if not isinstance(values, list):
            raise PrivacyError("history classifier returned invalid values")
        replacements: list[tuple[int, int, str]] = []
        blocked_hits = 0
        for raw in values:
            if not isinstance(raw, dict):
                raise PrivacyError("history classifier returned an invalid value")
            try:
                label = str(raw["label"])
                value = str(raw["value"])
                start = int(raw["start"])
                end = int(raw["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PrivacyError("history classifier omitted span metadata") from exc
            if start < 0 or end <= start or end > len(text) or classified[start:end] != value:
                raise PrivacyError("history classifier span did not project to source text")
            normalized_class = _normalize_class(label)
            if self._vault.safety_level(normalized_class) == "blocked":
                replacement = f"[PLVA_BLOCKED_{normalized_class}]"
                blocked_hits += 1
            else:
                replacement = self._vault.store(normalized_class, text[start:end])
            replacements.append((start, end, replacement))
        if sensitive and not replacements:
            raise PrivacyError("history contains sensitive text without an exact local value")
        scrubbed = text
        for start, end, placeholder in sorted(replacements, reverse=True):
            scrubbed = scrubbed[:start] + placeholder + scrubbed[end:]
        return scrubbed, blocked_hits


def privacy_request_hook(
    scrubber: HistoryScrubber,
    *,
    policy: SafetyPolicy | None = None,
    history_scrub: bool = True,
    inject_scheme: bool = True,
    inject_duplicate_warning: bool = True,
    inject_manifest: bool = True,
    inject_policy: bool = True,
) -> Callable[[dict[str, Any], dict[str, str]], tuple[dict[str, Any], dict[str, str]]]:
    def apply(
        document: dict[str, Any], headers: dict[str, str]
    ) -> tuple[dict[str, Any], dict[str, str]]:
        rewritten: dict[str, Any] = json.loads(json.dumps(document))
        raw_manifest = rewritten.pop(PLACEHOLDER_MANIFEST_KEY, None)
        messages = rewritten.get("messages")
        if not isinstance(messages, list):
            raise PrivacyError("request has no message history")
        manifest_target, manifest_items = (
            _manifest_target(messages, raw_manifest) if inject_manifest else (None, ())
        )
        messages[:] = _remove_old_placeholder_teaching(messages)
        locations: list[tuple[dict[str, Any], str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                locations.append((message, "content"))
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        locations.append((part, "text"))
        texts = tuple(container[key] for container, key in locations)
        scrubbed = scrubber.scrub(texts) if texts and history_scrub else texts
        for (container, key), value in zip(locations, scrubbed, strict=True):
            container[key] = value
        instructions = _placeholder_instructions(
            scheme=inject_scheme,
            duplicate_warning=inject_duplicate_warning,
            policy_prompt=(policy or SafetyPolicy()).prompt() if inject_policy else None,
        )
        if instructions is not None:
            _inject_placeholder_instructions(messages, instructions, manifest_target)
        if manifest_target is not None:
            _attach_manifest(manifest_target, manifest_items)
        return rewritten, headers

    return apply


def _placeholder_instructions(
    *, scheme: bool, duplicate_warning: bool, policy_prompt: str | None
) -> str | None:
    parts: list[str] = []
    if scheme:
        parts.append(PLACEHOLDER_SCHEME)
    if duplicate_warning:
        warning = PLACEHOLDER_DUPLICATE_WARNING
        if parts:
            warning = warning.removeprefix("[PLVA_PLACEHOLDERS] ")
        parts.append(warning)
    if policy_prompt is not None:
        parts.append(policy_prompt)
    return " ".join(parts) if parts else None


def _inject_placeholder_instructions(
    messages: list[Any],
    instructions: str,
    observation: dict[str, Any] | None,
) -> None:
    wrapped = f"{PLACEHOLDER_SYSTEM_BEGIN}\n{instructions}\n{PLACEHOLDER_SYSTEM_END}"
    system_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "system"
    ]
    if system_messages:
        contents: list[str] = []
        for message in system_messages:
            content = message.get("content")
            if not isinstance(content, str):
                raise PrivacyError("system prompt is not text")
            if content.strip():
                contents.append(content.rstrip())
        primary = system_messages[0]
        primary["content"] = "\n\n".join((*contents, wrapped))
        messages[:] = [
            message
            for message in messages
            if not (
                isinstance(message, dict)
                and message.get("role") == "system"
                and message is not primary
            )
        ]
        if messages[0] is not primary:
            messages.remove(primary)
            messages.insert(0, primary)
        return
    target = observation or next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        None,
    )
    if target is None:
        raise PrivacyError("placeholder instructions have no compatible message")
    _attach_observation_text(target, wrapped)


def _manifest_target(
    messages: list[Any], raw_manifest: Any
) -> tuple[dict[str, Any] | None, tuple[tuple[str, str, str], ...]]:
    if raw_manifest is None:
        return None, ()
    if not isinstance(raw_manifest, dict):
        raise PrivacyError("placeholder manifest metadata is invalid")
    try:
        message_index = int(raw_manifest["message_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PrivacyError("placeholder manifest omitted its observation") from exc
    items = raw_manifest.get("items")
    if not isinstance(items, list):
        raise PrivacyError("placeholder manifest items are invalid")
    if message_index < 0 or message_index >= len(messages):
        raise PrivacyError("placeholder manifest observation is out of range")
    target = messages[message_index]
    if not isinstance(target, dict):
        raise PrivacyError("placeholder manifest observation is invalid")
    parsed: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise PrivacyError("placeholder manifest entry is invalid")
        token = item.get("token")
        pii_class = item.get("class")
        safety_level = item.get("safety_level", "hide_use")
        if not isinstance(token, str) or not isinstance(pii_class, str):
            raise PrivacyError("placeholder manifest entry omitted token metadata")
        if not isinstance(safety_level, str) or safety_level not in SAFETY_LEVELS:
            raise PrivacyError("placeholder manifest safety level is invalid")
        normalized_class = _normalize_class(pii_class)
        if _PLACEHOLDER_SHAPE.fullmatch(token) is None or not token.startswith(
            normalized_class + "_"
        ):
            raise PrivacyError("placeholder manifest token is invalid")
        if token not in seen:
            parsed.append((token, normalized_class, safety_level))
            seen.add(token)
    return target, tuple(parsed)


def _remove_old_placeholder_teaching(messages: list[Any]) -> list[Any]:
    cleaned: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            cleaned.append(message)
            continue
        content = message.get("content")
        if (
            message.get("role") == "system"
            and isinstance(content, str)
            and content.startswith("[PLVA_PLACEHOLDERS]")
        ):
            continue
        if isinstance(content, str):
            message["content"] = _strip_placeholder_text(content)
        elif isinstance(content, list):
            message["content"] = [
                part
                for part in content
                if not (
                    isinstance(part, dict)
                    and isinstance(part.get("text"), str)
                    and (
                        part["text"].startswith(PLACEHOLDER_MANIFEST_PREFIX)
                        or part["text"].startswith(PLACEHOLDER_SYSTEM_BEGIN)
                        or part["text"].startswith("[PLVA_PLACEHOLDERS]")
                    )
                )
            ]
        cleaned.append(message)
    return cleaned


def _strip_placeholder_text(text: str) -> str:
    while PLACEHOLDER_SYSTEM_BEGIN in text:
        start = text.find(PLACEHOLDER_SYSTEM_BEGIN)
        end = text.find(PLACEHOLDER_SYSTEM_END, start)
        if end < 0:
            raise PrivacyError("placeholder instruction block is incomplete")
        text = text[:start] + text[end + len(PLACEHOLDER_SYSTEM_END) :]
    kept = [line for line in text.splitlines() if not line.startswith(PLACEHOLDER_MANIFEST_PREFIX)]
    return "\n".join(kept).rstrip()


def _attach_manifest(message: dict[str, Any], items: tuple[tuple[str, str, str], ...]) -> None:
    if items:
        visible = ", ".join(
            f"«{token}» ({_class_hint(pii_class)} · {_level_hint(level)})"
            for token, pii_class, level in items
        )
        text = (
            f"{PLACEHOLDER_MANIFEST_PREFIX} Placeholders visible in the current screenshot: "
            f"{visible}. Use only the exact inner tokens shown here for this step."
        )
    else:
        text = (
            f"{PLACEHOLDER_MANIFEST_PREFIX} Placeholders visible in the current screenshot: "
            "none. Do not reuse a placeholder merely because it appeared in an earlier step."
        )
    _attach_observation_text(message, text)


def _attach_observation_text(message: dict[str, Any], text: str) -> None:
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = content.rstrip() + "\n\n" + text
        return
    if not isinstance(content, list):
        raise PrivacyError("placeholder manifest target has no text observation")
    insertion = next(
        (
            index
            for index, part in enumerate(content)
            if isinstance(part, dict) and part.get("type") == "image_url"
        ),
        len(content),
    )
    content.insert(insertion, {"type": "text", "text": text})


def _class_hint(pii_class: str) -> str:
    return {
        "ADDRESS": "postal address",
        "CARD_NUMBER": "payment card number",
        "CVC": "card security code",
        "GOV_ID": "government identifier",
        "SSN": "social security number",
    }.get(pii_class, pii_class.lower().replace("_", " "))


def _level_hint(level: str) -> str:
    return {
        "hide_use": "hidden, use allowed",
        "approval": "local approval required",
        "blocked": "blocked",
    }[level]


def privacy_response_hook(vault: SessionVault) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def apply(document: dict[str, Any]) -> dict[str, Any]:
        rewritten: dict[str, Any] = json.loads(json.dumps(document))
        choices = rewritten.get("choices")
        if not isinstance(choices, list):
            raise PrivacyError("completion has no choices")
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            try:
                action = json.loads(content)
            except ValueError:
                continue
            if not isinstance(action, dict):
                continue
            calls = _executed_calls(action)
            for call in calls:
                name = call.get("tool_name", call.get("name"))
                if not isinstance(name, str):
                    raise PrivacyError("tool call has no name")
                if "answer" in name.lower():
                    continue
                for key, value in list(call.items()):
                    if key not in _TOOL_NAME_KEYS:
                        call[key] = _resolve_structure(value, vault)
            if calls:
                message["content"] = json.dumps(action, separators=(",", ":"))
        return rewritten

    return apply


def _executed_calls(action: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    plural = action.get("tool_calls")
    if isinstance(plural, list):
        calls.extend(call for call in plural if isinstance(call, dict))
    singular = action.get("tool_call")
    if isinstance(singular, dict):
        calls.append(singular)
    if isinstance(action.get("tool_name"), str):
        calls.append(action)
    return calls


def _resolve_structure(value: Any, vault: SessionVault) -> Any:
    if isinstance(value, str):
        return vault.resolve_text(value)
    if isinstance(value, list):
        return [_resolve_structure(item, vault) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_structure(item, vault) for key, item in value.items()}
    return value


def _paint_chips(
    png: bytes,
    chips: list[tuple[tuple[float, float, float, float], tuple[str, ...]]],
) -> bytes:
    try:
        with Image.open(io.BytesIO(png)) as loaded:
            image = loaded.convert("RGB")
    except (OSError, ValueError) as exc:
        raise PrivacyError("redactor returned an invalid image") from exc
    for raw_box, placeholders in chips:
        x1, y1, x2, y2 = raw_box
        left = max(0, min(image.width, int(x1)))
        top = max(0, min(image.height, int(y1)))
        right = max(left, min(image.width, int(x2 + 0.999)))
        bottom = max(top, min(image.height, int(y2 + 0.999)))
        if right <= left or bottom <= top:
            raise PrivacyError("placeholder chip has an empty bounding box")
        chip = Image.new("RGB", (right - left, bottom - top), (5, 8, 7))
        draw = ImageDraw.Draw(chip)
        text = "«" + "+".join(placeholders) + "»"
        size = max(6, min(18, chip.height - 2))
        font = ImageFont.load_default(size=size)
        while size > 6 and draw.textbbox((0, 0), text, font=font)[2] > chip.width - 2:
            size -= 1
            font = ImageFont.load_default(size=size)
        draw.text((1, max(0, (chip.height - size) // 2)), text, fill=(255, 255, 255), font=font)
        image.paste(chip, (left, top))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _normalize_class(value: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")
    aliases = {
        "GIVEN_NAME": "NAME",
        "SURNAME": "NAME",
        "CREDIT_CARD": "CARD_NUMBER",
        "CREDIT_CARD_NUMBER": "CARD_NUMBER",
        "CREDIT_CARD_CVV": "CVC",
        "GOVERNMENT_ID": "GOV_ID",
        "SOCIAL_SECURITY_NUMBER": "SSN",
        "BUILDING_NUMBER": "ADDRESS",
        "STREET_NAME": "ADDRESS",
        "SECONDARY_ADDRESS": "ADDRESS",
    }
    normalized = aliases.get(normalized, normalized)
    if not normalized:
        raise PrivacyError("PII class is empty")
    return normalized


def _canonical_value(pii_class: str, value: str) -> str:
    if pii_class in _CREDENTIAL_CLASSES:
        return value
    return " ".join(value.split()).casefold()


def _replace_variants(text: str, variants: list[tuple[str, str]]) -> str:
    scrubbed = text
    for value, placeholder in variants:
        scrubbed = scrubbed.replace(value, placeholder)
    return scrubbed


def _coalesce_finding_values(finding: dict[str, Any]) -> list[dict[str, Any]]:
    raw_values = finding.get("values")
    text = finding.get("text")
    if raw_values is None:
        return []
    if not isinstance(raw_values, list) or not isinstance(text, str):
        raise PrivacyError("redactor emitted invalid finding values")
    parsed: list[dict[str, Any]] = []
    for raw in raw_values:
        if not isinstance(raw, dict):
            raise PrivacyError("redactor emitted an invalid PII value")
        try:
            label = str(raw["label"])
            value = str(raw["value"])
            start = int(raw["start"])
            end = int(raw["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PrivacyError("redactor omitted recognized PII span metadata") from exc
        if start < 0 or end <= start or end > len(text) or text[start:end] != value:
            raise PrivacyError("redactor PII span did not project to OCR text")
        parsed.append({**raw, "label": label, "value": value, "start": start, "end": end})
    if not parsed:
        return []
    parsed.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    merged: list[dict[str, Any]] = []
    for current in parsed:
        if merged:
            previous = merged[-1]
            gap = text[int(previous["end"]) : int(current["start"])]
            if (
                _normalize_class(str(previous["label"])) == _normalize_class(str(current["label"]))
                and re.fullmatch(r"[\s_.-]{0,3}", gap) is not None
            ):
                previous["end"] = current["end"]
                previous["value"] = text[int(previous["start"]) : int(current["end"])]
                previous["score"] = max(
                    float(previous.get("score", 0.0)), float(current.get("score", 0.0))
                )
                continue
        merged.append(copy.deepcopy(current))
    return merged
