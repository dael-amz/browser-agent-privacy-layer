"""Fail-closed loopback client for the sandboxed local LLM.

This is the pluggable client for the §7 mediator and the Step 13(B) semantic
executor: a small local model (Nemotron via llama-server, or an equivalent
OpenAI-compatible loopback server) that may see vault cleartext and therefore
must have zero network egress. The client refuses any non-loopback endpoint,
never follows redirects, and turns every transport or format failure into an
exception the callers translate into a fail-closed decision (deny / halt / no
result). See docs/local-llm-runbook.md for how to launch and verify the server.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import json
import os
import re
import subprocess
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urlparse

import httpx

DEFAULT_BASE_URL: Final = "http://127.0.0.1:8555/v1"
DEFAULT_MODEL: Final = "nemotron-mini"
_LSOF_TIMEOUT_SECONDS: Final = 10.0


class LocalLLMError(RuntimeError):
    """The local model could not produce a usable, safe answer."""


class LocalLLMUnavailableError(LocalLLMError):
    """The local model endpoint could not be reached safely in time."""


def _require_loopback(base_url: str) -> str:
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if parsed.scheme != "http":
        raise ValueError("local LLM base URL must use plain http on loopback")
    if host != "localhost":
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            raise ValueError("local LLM base URL must resolve to loopback")
    return base_url.rstrip("/")


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Where and how to reach the local model; the URL must stay on loopback."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 45.0
    max_tokens: int = 400
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            base_url=os.environ.get("PLVA_LOCAL_LLM_URL", DEFAULT_BASE_URL),
            model=os.environ.get("PLVA_LOCAL_LLM_MODEL", DEFAULT_MODEL),
        )

    def replace(self, **overrides: Any) -> LLMConfig:
        return dataclasses.replace(self, **overrides)


def extract_json_object(content: str) -> dict[str, Any] | None:
    """Return the last complete JSON object in a completion, or None.

    Small local models often wrap the answer in prose, code fences, or
    reasoning text; the final well-formed object is taken as the answer.
    """
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    index = content.find("{")
    while index != -1:
        try:
            candidate, end = decoder.raw_decode(content, index)
        except ValueError:
            index = content.find("{", index + 1)
            continue
        if isinstance(candidate, dict):
            found = candidate
            index = content.find("{", end)
        else:
            index = content.find("{", index + 1)
    return found


class LoopbackLLMClient:
    """Synchronous OpenAI-compatible chat client pinned to a loopback endpoint."""

    def __init__(
        self, config: LLMConfig | None = None, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._config = config or LLMConfig.from_env()
        self._base_url = _require_loopback(self._config.base_url)
        if not 0 < self._config.timeout_seconds <= 600:
            raise ValueError("local LLM timeout must be between 0 and 600 seconds")
        if not 1 <= self._config.max_tokens <= 4096:
            raise ValueError("local LLM max_tokens must be between 1 and 4096")
        self._client = httpx.Client(
            timeout=self._config.timeout_seconds, follow_redirects=False, transport=transport
        )

    @property
    def config(self) -> LLMConfig:
        return self._config

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LoopbackLLMClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def probe(self) -> dict[str, Any]:
        """Value-free reachability check: the endpoint and its advertised models."""
        try:
            response = self._client.get(f"{self._base_url}/models")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise LocalLLMUnavailableError(
                f"local LLM endpoint unreachable: {type(error).__name__}"
            ) from error
        data = payload.get("data") if isinstance(payload, dict) else None
        models = (
            [str(item.get("id", "?")) for item in data if isinstance(item, dict)]
            if isinstance(data, list)
            else []
        )
        return {"endpoint": self._base_url, "models": models}

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One chat completion that must come back as a single JSON object.

        When ``schema`` is given it is sent as a ``json_schema`` response format,
        which grammar-capable servers (llama-server) enforce at decode time —
        the strongest form of the token-only output contract. Servers that
        reject it fall back to ``json_object`` and then to plain text; callers
        must still validate, which they do.
        """
        content = self._chat(system=system, user=user, max_tokens=max_tokens, schema=schema)
        parsed = extract_json_object(content)
        if parsed is None:
            nudge = f"{user}\n\nYour reply MUST be exactly one JSON object with no other text."
            content = self._chat(system=system, user=nudge, max_tokens=max_tokens, schema=schema)
            parsed = extract_json_object(content)
        if parsed is None:
            raise LocalLLMError("local LLM did not return a parseable JSON object")
        return parsed

    def _chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._config.temperature,
            "max_tokens": max_tokens or self._config.max_tokens,
        }
        # Preference order; servers that reject one format (HTTP 400) get the next.
        formats: list[dict[str, Any] | None] = [{"type": "json_object"}, None]
        if schema is not None:
            formats.insert(
                0,
                {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "strict": True, "schema": schema},
                },
            )
        url = f"{self._base_url}/chat/completions"

        def _post(response_format: dict[str, Any] | None) -> httpx.Response:
            if response_format is None:
                body.pop("response_format", None)
            else:
                body["response_format"] = response_format
            return self._client.post(url, json=body)

        try:
            response = _post(formats[0])
            for fallback in formats[1:]:
                if response.status_code != 400:
                    break
                response = _post(fallback)
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as error:
            raise LocalLLMUnavailableError("local LLM timed out; failing closed") from error
        except httpx.HTTPError as error:
            raise LocalLLMUnavailableError(
                f"local LLM request failed: {type(error).__name__}"
            ) from error
        except ValueError as error:
            raise LocalLLMError("local LLM returned a non-JSON transport payload") from error
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LocalLLMError("local LLM returned an unexpected completion shape") from error
        if not isinstance(content, str) or not content.strip():
            raise LocalLLMError("local LLM returned an empty completion")
        return content


def _normalize_for_scan(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def find_cleartext_leak(outputs: Iterable[str], values: Iterable[str]) -> str | None:
    """Scan model outputs for any of the hidden values; describe the first hit.

    The returned description never contains the value itself. Values shorter
    than three normalized characters are skipped (they match everything);
    digit-bearing values are additionally matched on their bare digit stream so
    reformatting (spaces, dashes) cannot hide a leak.
    """
    normalized_outputs = [_normalize_for_scan(text) for text in outputs if text]
    digit_outputs = [re.sub(r"\D", "", text) for text in outputs if text]
    for value in values:
        needle = _normalize_for_scan(value)
        if len(needle) >= 3 and any(needle in haystack for haystack in normalized_outputs):
            return "output contained a hidden value (normalized match)"
        digits = re.sub(r"\D", "", value)
        if len(digits) >= 6 and any(digits in haystack for haystack in digit_outputs):
            return "output contained a hidden value (digit-sequence match)"
    return None


@dataclass(frozen=True, slots=True)
class EgressReport:
    """Best-effort, point-in-time socket audit of the local model server."""

    port: int
    pids: tuple[int, ...]
    flagged: tuple[str, ...]
    checked: bool
    detail: str

    @property
    def clean(self) -> bool:
        return self.checked and not self.flagged


def port_from_base_url(base_url: str) -> int:
    parsed = urlparse(base_url)
    if parsed.port is None:
        raise ValueError("local LLM base URL must carry an explicit port")
    return parsed.port


def _is_loopback_host(host: str) -> bool:
    stripped = host.strip("[]")
    if stripped == "localhost":
        return True
    try:
        return ipaddress.ip_address(stripped).is_loopback
    except ValueError:
        return False


def _flagged_sockets(lsof_output: str) -> tuple[str, ...]:
    flagged: list[str] = []
    for line in lsof_output.splitlines():
        parts = line.split()
        if len(parts) < 9 or parts[0] == "COMMAND":
            continue
        name = " ".join(parts[8:])
        if "->" in name:
            remote = name.split("->", 1)[1].split(" ", 1)[0]
            host = remote.rsplit(":", 1)[0]
            if not _is_loopback_host(host):
                flagged.append(f"remote connection: {name}")
        else:
            local = name.split(" ", 1)[0]
            host = local.rsplit(":", 1)[0]
            if not _is_loopback_host(host):
                flagged.append(f"non-loopback bind: {name}")
    return tuple(flagged)


def verify_no_egress(port: int) -> EgressReport:
    """Audit the process serving the given loopback port for non-loopback sockets.

    This is the macOS substitute for OpenShell's enforcing namespace
    (ADR-0001): a point-in-time `lsof` audit, not a proof. On hosts where
    OpenShell enforcement is available, prefer `openshell policy prove`.
    """
    try:
        listeners = subprocess.run(
            ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=_LSOF_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return EgressReport(port, (), (), False, f"lsof unavailable: {type(error).__name__}")
    pids = tuple(int(line) for line in listeners.stdout.split() if line.isdigit())
    if not pids:
        return EgressReport(port, (), (), False, "no process is listening on the port")
    flagged: list[str] = []
    for pid in pids:
        try:
            sockets = subprocess.run(
                ["lsof", "-nP", "-a", "-p", str(pid), "-i"],
                capture_output=True,
                text=True,
                timeout=_LSOF_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return EgressReport(
                port, pids, (), False, f"socket audit failed: {type(error).__name__}"
            )
        flagged.extend(f"pid {pid}: {entry}" for entry in _flagged_sockets(sockets.stdout))
    return EgressReport(port, pids, tuple(flagged), True, "point-in-time lsof audit")
