"""Layered provider credential resolution shared by the proxy, demo, and probes.

Precedence for ``auto`` matches Holo Desktop CLI: process environment, then
``~/.holo/.env``, then the project ``.env`` beside the Holo workbench.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final, Literal

from plva_proxy.providers import PROVIDERS

CredentialSource = Literal["auto", "holo_cli", "project", "environment"]
CREDENTIAL_SOURCES: Final[tuple[CredentialSource, ...]] = (
    "auto",
    "holo_cli",
    "project",
    "environment",
)

HOLO_DIR: Final = Path.home() / ".holo"
HOLO_USER_ENV: Final = HOLO_DIR / ".env"
HOLO_PROFILE: Final = HOLO_DIR / "profile.json"


def env_file_value(path: Path, key: str) -> str | None:
    """Read ``KEY=value`` from a dotenv-style file without echoing its contents."""

    try:
        raw = path.read_text("utf-8")
    except OSError:
        return None
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").strip()
        if not stripped.startswith(f"{key}="):
            continue
        value = stripped.removeprefix(f"{key}=").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value or None
    return None


def _read_holo_profile() -> dict[str, Any]:
    try:
        raw = json.loads(HOLO_PROFILE.read_text("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _key_from_environment(key_names: tuple[str, ...], environ: dict[str, str]) -> str | None:
    for name in key_names:
        if value := environ.get(name):
            return value
    return None


def _key_from_file(path: Path, key_names: tuple[str, ...]) -> str | None:
    for name in key_names:
        if value := env_file_value(path, name):
            return value
    return None


def resolve_provider_key(
    *,
    provider: str,
    source: CredentialSource = "auto",
    project_root: Path | None = None,
    environ: dict[str, str] | os._Environ[str] | None = None,
) -> tuple[str | None, CredentialSource | None]:
    """Return the resolved key and which layer supplied it."""

    spec = PROVIDERS[provider]
    env = dict(environ) if environ is not None else dict(os.environ)
    project_env = (project_root or Path.cwd()) / ".env"

    layers: tuple[tuple[CredentialSource, str], ...] = (
        ("environment", _key_from_environment(spec.key_names, env) or ""),
        ("holo_cli", _key_from_file(HOLO_USER_ENV, spec.key_names) or ""),
        ("project", _key_from_file(project_env, spec.key_names) or ""),
    )
    if source == "auto":
        selected = layers
    else:
        selected = tuple(layer for layer in layers if layer[0] == source)
    for resolved_label, value in selected:
        if value:
            return value, resolved_label
    return None, None


def inject_provider_keys(
    environment: dict[str, str],
    *,
    provider: str,
    source: CredentialSource = "auto",
    project_root: Path | None = None,
) -> None:
    """Copy the resolved provider key into ``environment`` under its canonical name."""

    key, _ = resolve_provider_key(
        provider=provider, source=source, project_root=project_root, environ=environment
    )
    if key is None:
        return
    environment[PROVIDERS[provider].key_names[0]] = key


def credential_status(
    *,
    provider: str,
    source: CredentialSource = "auto",
    project_root: Path | None = None,
    environ: dict[str, str] | os._Environ[str] | None = None,
) -> dict[str, Any]:
    """Privacy-safe credential summary for UI and preflight checks."""

    env = dict(environ) if environ is not None else dict(os.environ)
    project_env = (project_root or Path.cwd()) / ".env"
    spec = PROVIDERS[provider]
    key, resolved = resolve_provider_key(
        provider=provider, source=source, project_root=project_root, environ=env
    )
    profile = _read_holo_profile()
    holo_cli_key = _key_from_file(HOLO_USER_ENV, spec.key_names)
    project_key = _key_from_file(project_env, spec.key_names)
    env_key = _key_from_environment(spec.key_names, env)
    key_label = profile.get("key_label")
    account_email = profile.get("email")
    return {
        "configured": key is not None,
        "provider": provider,
        "preference": source,
        "source": resolved,
        "key_label": key_label if isinstance(key_label, str) and holo_cli_key is not None else None,
        "account_email": account_email if isinstance(account_email, str) else None,
        "holo_cli_available": holo_cli_key is not None,
        "project_env_available": project_key is not None,
        "environment_available": env_key is not None,
        "holo_cli_path": str(HOLO_USER_ENV.expanduser()),
        "project_env_path": str(project_env),
        "can_connect_holo_cli": holo_cli_key is not None and resolved != "holo_cli",
    }
