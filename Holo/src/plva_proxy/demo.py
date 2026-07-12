"""PLVA consumer demo and task launcher.

Live session state stays memory-only, but each run's post-boundary audit
trail (redacted frames, scrubbed text, placeholder-form replies) is also
recorded to a local history directory so past runs can be replayed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import secrets
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Final

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse

from plva_proxy.credentials import (
    CREDENTIAL_SOURCES,
    CredentialSource,
    credential_status,
    inject_provider_keys,
)
from plva_proxy.privacy import SafetyPolicy
from plva_proxy.providers import PROVIDERS, ProviderSpec
from plva_proxy.runtime_capture import LOOPBACK_HOST

ROOT: Final = Path(__file__).resolve().parents[2]
UI_PATH: Final = Path(__file__).with_name("demo_ui.html")
LANDING_PATH: Final = Path(__file__).with_name("landing_ui.html")
HISTORY_ROOT: Final = ROOT / "history"
_RUN_ID: Final = re.compile(r"^run-[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")
_IMAGE_SUFFIXES: Final = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
DEFAULT_POLICY_PATH: Final = ROOT / "config" / "privacy-policy.json"
MODEL_LIST_TTL: Final = 300.0


def _provider_api_key(spec: ProviderSpec) -> str | None:
    for name in spec.key_names:
        value = os.environ.get(name)
        if value:
            return value
    try:
        for line in (ROOT / ".env").read_text("utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() in spec.key_names and value.strip():
                return value.strip()
    except OSError:
        pass
    return None


def _fetch_provider_models(spec: ProviderSpec) -> list[dict[str, str]]:
    """Query the provider's live /models list; raises when unreachable or empty."""

    headers: dict[str, str] = {}
    key = _provider_api_key(spec)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(spec.base_url.rstrip("/") + "/models", headers=headers)
    with urllib.request.urlopen(request, timeout=8.0) as response:
        payload = json.loads(response.read())
    models: list[dict[str, str]] = []
    entries = payload.get("data") if isinstance(payload, dict) else None
    for item in entries if isinstance(entries, list) else []:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        if item.get("is_ready") is False or item.get("is_active") is False:
            continue
        name = item.get("name")
        models.append(
            {"id": model_id, "label": name if isinstance(name, str) and name else model_id}
        )
    if not models:
        raise ValueError("provider returned no models")
    return models


PROXY_BASE: Final = "http://127.0.0.1:18081"
_ANSI_ESCAPE: Final = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_INLINE_SECRET: Final = re.compile(
    r"(?i)\b([A-Z_]*(?:API_KEY|AUTH_TOKEN|PASSWORD|SECRET|TOKEN|KEY))\s*=\s*([^\s]+)"
)
FEATURE_ENV: Final = {
    "chips": "PLVA_PRIVACY_CHIPS",
    "history_scrub": "PLVA_PRIVACY_HISTORY_SCRUB",
    "scheme": "PLVA_PRIVACY_SCHEME",
    "duplicate_warning": "PLVA_PRIVACY_DUPLICATE_WARNING",
    "manifest": "PLVA_PRIVACY_MANIFEST",
    "resolution": "PLVA_PRIVACY_RESOLUTION",
    "policy_teaching": "PLVA_PRIVACY_POLICY_TEACHING",
    "skill": "PLVA_PRIVACY_SKILL",
}


def _load_policy() -> SafetyPolicy:
    """Load the editable local defaults, failing closed on malformed policy data."""

    selected = Path(os.environ.get("PLVA_POLICY_FILE", DEFAULT_POLICY_PATH))
    if not selected.is_absolute():
        selected = ROOT / selected
    try:
        raw = json.loads(selected.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not load privacy policy: {selected}") from exc
    if not isinstance(raw, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in raw.items()
    ):
        raise RuntimeError("privacy policy must be an object of string values")
    try:
        return SafetyPolicy(raw)
    except ValueError as exc:
        raise RuntimeError("privacy policy contains an invalid safety level") from exc


class HistoryStore:
    """Local, file-based audit history of runs and their model calls.

    Persists only post-boundary artifacts: the redacted frames, scrubbed
    text, and placeholder-form replies that the live viewers already show.
    Raw values and unredacted pixels never reach this store. Layout:
    ``<root>/<run-id>/run.json`` plus ``calls/NNNN.json`` and image files.
    All disk failures are swallowed: history must never break a live run.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = threading.Lock()

    def create_run(
        self, prompt: str, settings: dict[str, Any], policy: dict[str, str]
    ) -> str | None:
        run_id = time.strftime("run-%Y%m%d-%H%M%S") + f"-{secrets.token_hex(2)}"
        record = {
            "id": run_id,
            "prompt": prompt,
            "started_at": time.time(),
            "finished_at": 0.0,
            "status": "running",
            "settings": settings,
            "policy": policy,
            "events": [],
        }
        with self._lock:
            try:
                (self._root / run_id / "calls").mkdir(parents=True, exist_ok=True)
                self._write_run(run_id, record)
            except OSError:
                return None
        return run_id

    def update_run(self, run_id: str | None, **fields: Any) -> None:
        if run_id is None:
            return
        with self._lock:
            record = self._read_run(run_id)
            if record is None:
                return
            record.update(fields)
            try:
                self._write_run(run_id, record)
            except OSError:
                return

    def save_call(
        self,
        run_id: str | None,
        record: dict[str, Any],
        images: list[tuple[str, bytes]],
    ) -> None:
        if run_id is None or not isinstance(record.get("id"), int):
            return
        calls_dir = self._root / run_id / "calls"
        name = f"{record['id']:04d}"
        with self._lock:
            try:
                for index, (media_type, blob) in enumerate(images):
                    suffix = _IMAGE_SUFFIXES.get(media_type, ".bin")
                    (calls_dir / f"{name}-img-{index}{suffix}").write_bytes(blob)
                (calls_dir / f"{name}.json").write_text(json.dumps(record, indent=1), "utf-8")
            except OSError:
                return

    def runs(self) -> list[dict[str, Any]]:
        with self._lock:
            try:
                children = sorted(
                    (path for path in self._root.iterdir() if path.is_dir()),
                    key=lambda path: path.name,
                    reverse=True,
                )
            except OSError:
                return []
            out: list[dict[str, Any]] = []
            for child in children:
                if _RUN_ID.match(child.name) is None:
                    continue
                record = self._read_run(child.name)
                if record is None:
                    continue
                try:
                    count = len(list((child / "calls").glob("*.json")))
                except OSError:
                    count = 0
                summary = {
                    key: record.get(key)
                    for key in ("id", "prompt", "started_at", "finished_at", "status")
                }
                summary["calls"] = count
                out.append(summary)
            return out

    def run(self, run_id: str) -> dict[str, Any] | None:
        if _RUN_ID.match(run_id) is None:
            return None
        with self._lock:
            record = self._read_run(run_id)
            if record is None:
                return None
            calls: list[dict[str, Any]] = []
            try:
                paths = sorted((self._root / run_id / "calls").glob("[0-9]*.json"))
            except OSError:
                paths = []
            for path in paths:
                try:
                    data = json.loads(path.read_text("utf-8"))
                except (OSError, ValueError):
                    continue
                if isinstance(data, dict):
                    calls.append(
                        {k: v for k, v in data.items() if k not in {"request", "response"}}
                    )
            return {"run": record, "calls": calls}

    def call(self, run_id: str, call_id: int) -> dict[str, Any] | None:
        if _RUN_ID.match(run_id) is None or call_id < 1:
            return None
        path = self._root / run_id / "calls" / f"{call_id:04d}.json"
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def image(self, run_id: str, call_id: int, index: int) -> tuple[str, bytes] | None:
        if _RUN_ID.match(run_id) is None or call_id < 1 or index < 0:
            return None
        for media_type, suffix in _IMAGE_SUFFIXES.items():
            path = self._root / run_id / "calls" / f"{call_id:04d}-img-{index}{suffix}"
            try:
                return media_type, path.read_bytes()
            except OSError:
                continue
        return None

    def _write_run(self, run_id: str, record: dict[str, Any]) -> None:
        (self._root / run_id / "run.json").write_text(json.dumps(record, indent=1), "utf-8")

    def _read_run(self, run_id: str) -> dict[str, Any] | None:
        try:
            data = json.loads((self._root / run_id / "run.json").read_text("utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None


class DemoController:
    """Own one local CUA run; live state is memory-only, audits go to history."""

    def __init__(self, history_root: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._status = "idle"
        self._started_at = 0.0
        self._finished_at = 0.0
        self._events: deque[dict[str, str]] = deque(maxlen=24)
        self._traces: deque[dict[str, str]] = deque(maxlen=300)
        self._traces_dropped = 0
        self._frame: bytes | None = None
        self._vault: dict[str, Any] = {"entries": [], "policy": {}}
        self._approvals: list[dict[str, Any]] = []
        self._findings: dict[str, Any] = {}
        self._filter: dict[str, Any] = {"status": "idle"}
        self._stats: dict[str, Any] = {}
        self._calls: dict[int, dict[str, Any]] = {}
        self._call_images: dict[tuple[int, int], tuple[str, bytes]] = {}
        env_root = os.environ.get("PLVA_HISTORY_DIR")
        self._history = HistoryStore(
            history_root
            if history_root is not None
            else Path(env_root)
            if env_root
            else HISTORY_ROOT
        )
        self._run_id: str | None = None
        self._model_lists: dict[str, tuple[float, list[dict[str, str]], bool]] = {}
        self._policy = _load_policy().snapshot()
        self._settings: dict[str, Any] = {
            "plva_enabled": True,
            "provider": "hcompany",
            "model": PROVIDERS["hcompany"].model,
            "credential_source": "auto",
            "vision_mode": "fast",
            "lifecycle": "eager",
            "detector_version": "v2",
            "ocr_engine": "apple",
            # The current visual checkpoint over-masks ordinary BrowserUse UI.
            # OCR + semantic rules are the calibrated default; the visual lab
            # path remains opt-in for detector evaluation.
            "visual_detector": "off",
            "semantic_engine": "rampart",
            "tools": "off",
            "features": {name: True for name in FEATURE_ENV},
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            running = process is not None and process.poll() is None
            return {
                "status": self._status,
                "running": running,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "events": list(self._events),
                "trace_count": len(self._traces),
                "policy": dict(self._policy),
                "settings": json.loads(json.dumps(self._settings)),
                "credentials": self._credentials_snapshot(),
                "stats": dict(self._stats),
                "has_frame": self._frame is not None,
                "vault_count": len(self._vault.get("entries", [])),
                "approval_count": len(self._approvals),
                "call_count": len(self._calls),
                "filter": dict(self._filter),
            }

    def _credential_source(self) -> CredentialSource:
        source = self._settings.get("credential_source", "auto")
        return source if source in CREDENTIAL_SOURCES else "auto"

    def _credentials_snapshot(self) -> dict[str, Any]:
        return credential_status(
            provider=str(self._settings["provider"]),
            source=self._credential_source(),
            project_root=ROOT,
        )

    def connect_holo_cli(self) -> dict[str, Any]:
        with self._lock:
            self._require_idle()
            probe = credential_status(
                provider=str(self._settings["provider"]),
                source="holo_cli",
                project_root=ROOT,
            )
            if not probe["holo_cli_available"]:
                raise RuntimeError(
                    "No Holo Desktop CLI key found. Run holo login in a terminal first."
                )
            self._settings["credential_source"] = "auto"
            self._event(
                "Holo Desktop CLI connected",
                "Using the API key saved by holo login.",
            )
            return self._credentials_snapshot()

    def set_policy(self, raw: Any) -> dict[str, str]:
        if not isinstance(raw, dict):
            raise ValueError("policy must be an object")
        policy = SafetyPolicy(raw)
        with self._lock:
            self._require_idle()
            self._policy = policy.snapshot()
            self._event("Policy updated", "Your choices will apply to the next task.")
            return dict(self._policy)

    def set_settings(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("settings must be an object")
        provider = raw.get("provider")
        model = raw.get("model")
        vision_mode = raw.get("vision_mode")
        lifecycle = raw.get("lifecycle")
        plva_enabled = raw.get("plva_enabled")
        features = raw.get("features")
        if provider not in PROVIDERS:
            raise ValueError("provider is invalid")
        if model is not None and not isinstance(model, str):
            raise ValueError("model is invalid")
        # A model from another provider falls back to the new provider's default
        # so switching providers never strands an incompatible model id.
        allowed = set(PROVIDERS[provider].allowed_models())
        cached = self._model_lists.get(provider)
        if cached is not None:
            allowed.update(entry["id"] for entry in cached[1])
        if model not in allowed:
            model = PROVIDERS[provider].model
        if vision_mode not in {"fast", "cascade", "accurate"}:
            raise ValueError("vision mode is invalid")
        if lifecycle not in {"adaptive", "eager", "cold"}:
            raise ValueError("lifecycle is invalid")
        # Absent keys fall back to defaults so older clients keep working.
        detector_version = raw.get("detector_version") or "v2"
        ocr_engine = raw.get("ocr_engine") or "apple"
        visual_detector = raw.get("visual_detector") or "off"
        if detector_version not in {"v2", "v3"}:
            raise ValueError("detector version is invalid")
        v3_detector = ROOT / "plva-v3" / "dist" / "visual" / "detector.onnx"
        if detector_version == "v3" and visual_detector == "on" and not v3_detector.is_file():
            raise ValueError(
                "the v3 detector is not installed; extract the frozen plva-v3 "
                "bundle to Holo/plva-v3 or keep v2"
            )
        if ocr_engine not in {"apple", "rapidocr"}:
            raise ValueError("OCR engine is invalid")
        if visual_detector not in {"on", "off"}:
            raise ValueError("visual detector setting is invalid")
        semantic_engine = raw.get("semantic_engine") or "rampart"
        if semantic_engine not in {"rampart", "gliner2", "openai-pf", "openai-pf-4bit"}:
            raise ValueError("semantic engine is invalid")
        credential_source = raw.get("credential_source") or "auto"
        if credential_source not in CREDENTIAL_SOURCES:
            raise ValueError("credential source is invalid")
        tools = raw.get("tools") or "off"
        if tools not in {"on", "off"}:
            raise ValueError("tools setting is invalid")
        if not isinstance(plva_enabled, bool) or not isinstance(features, dict):
            raise ValueError("settings are invalid")
        selected_features: dict[str, bool] = {}
        for name in FEATURE_ENV:
            value = features.get(name)
            if not isinstance(value, bool):
                raise ValueError(f"feature {name} is invalid")
            selected_features[name] = value
        with self._lock:
            self._require_idle()
            self._settings = {
                "plva_enabled": plva_enabled,
                "provider": provider,
                "model": model,
                "credential_source": credential_source,
                "vision_mode": vision_mode,
                "lifecycle": lifecycle,
                "detector_version": detector_version,
                "ocr_engine": ocr_engine,
                "visual_detector": visual_detector,
                "semantic_engine": semantic_engine,
                "tools": tools,
                "features": selected_features,
            }
            return copy.deepcopy(self._settings)

    def start(self, prompt: Any) -> None:
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 10_000:
            raise ValueError("prompt must contain 1-10,000 characters")
        if "\x00" in prompt:
            raise ValueError("prompt contains an invalid character")
        with self._lock:
            self._require_idle()
            self._status = "starting"
            self._started_at = time.time()
            self._finished_at = 0.0
            self._events.clear()
            self._traces.clear()
            self._traces_dropped = 0
            self._frame = None
            self._vault = {"entries": [], "policy": dict(self._policy)}
            self._approvals = []
            self._findings = {}
            self._filter = {"status": "idle"}
            self._stats = {}
            self._calls = {}
            self._call_images = {}
            self._run_id = self._history.create_run(
                prompt.strip(),
                json.loads(json.dumps(self._settings)),
                dict(self._policy),
            )
            self._event("Preparing private session", "Nothing has left the device yet.")
            environment = self._run_environment()
            try:
                process = subprocess.Popen(
                    [str(ROOT / "run_step1.sh"), prompt.strip()],
                    cwd=ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    start_new_session=True,
                )
            except OSError as exc:
                self._status = "failed"
                raise RuntimeError("could not start the local task runner") from exc
            self._process = process
            threading.Thread(target=self._read_process, args=(process,), daemon=True).start()
            threading.Thread(target=self._monitor_proxy, args=(process,), daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return
            self._status = "stopping"
            self._event("Stopping task", "Closing the agent and clearing its private session.")
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return

    def close(self) -> None:
        self.stop()

    def frame(self) -> bytes | None:
        with self._lock:
            return self._frame

    def vault(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._vault)

    def approvals(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._approvals)

    def approve_once(self, token: Any) -> dict[str, Any]:
        if not isinstance(token, str) or not token:
            raise ValueError("approval token is invalid")
        with self._lock:
            entry = next(
                (
                    item
                    for item in self._vault.get("entries", [])
                    if isinstance(item, dict) and item.get("placeholder") == token
                ),
                None,
            )
            if entry is None or entry.get("safety_level") != "approval":
                raise ValueError("token is not approval-gated in the active session")
        grant = _proxy_json_request(
            "/viewer/approvals",
            method="POST",
            payload={
                "token": token,
                "tool_name": "write_desktop",
                "argument_path": "content",
                "ttl_seconds": 60,
                "use_count": 1,
            },
        )
        if grant is None:
            raise RuntimeError("the local approval service is unavailable")
        with self._lock:
            self._approvals = [item for item in self._approvals if item.get("token") != token] + [
                grant
            ]
            self._event("Local approval granted", "One exact private write is authorized.")
        return copy.deepcopy(grant)

    def findings(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._findings)

    def filter_diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._filter)

    def calls(self) -> list[dict[str, Any]]:
        """Value-light call summaries, newest last, for the History tab list."""

        with self._lock:
            return [
                {key: value for key, value in record.items() if key not in {"request", "response"}}
                for _, record in sorted(self._calls.items())
            ]

    def call(self, call_id: int) -> dict[str, Any] | None:
        with self._lock:
            record = self._calls.get(call_id)
            return copy.deepcopy(record) if record is not None else None

    def call_image(self, call_id: int, index: int) -> tuple[str, bytes] | None:
        with self._lock:
            return self._call_images.get((call_id, index))

    def traces(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": list(self._traces),
                "dropped": self._traces_dropped,
                "memory_only": True,
            }

    def history_runs(self) -> list[dict[str, Any]]:
        runs = self._history.runs()
        with self._lock:
            active_run = self._run_id if self._status in {"starting", "running"} else None
        for summary in runs:
            # A run left in "running" by a dead process was interrupted; only
            # the controller's live run may truthfully claim to be running.
            if summary.get("status") == "running" and summary.get("id") != active_run:
                summary["status"] = "interrupted"
        return runs

    def history_run(self, run_id: str) -> dict[str, Any] | None:
        return self._history.run(run_id)

    def history_call(self, run_id: str, call_id: int) -> dict[str, Any] | None:
        return self._history.call(run_id, call_id)

    def history_image(self, run_id: str, call_id: int, index: int) -> tuple[str, bytes] | None:
        return self._history.image(run_id, call_id, index)

    def model_catalog(self) -> dict[str, Any]:
        """Per-provider model lists from the live /models endpoints.

        Results (including failures) are cached for MODEL_LIST_TTL; on failure
        the frozen presets from providers.py answer instead, marked live=false.
        """

        catalog: dict[str, Any] = {}
        for name, spec in PROVIDERS.items():
            cached = self._model_lists.get(name)
            if cached is None or time.time() - cached[0] >= MODEL_LIST_TTL:
                try:
                    models, live = _fetch_provider_models(spec), True
                except (OSError, ValueError, urllib.error.URLError):
                    models = [{"id": m, "label": m} for m in spec.allowed_models()]
                    live = False
                cached = (time.time(), models, live)
                self._model_lists[name] = cached
            catalog[name] = {"models": cached[1], "live": cached[2]}
        return catalog

    def _require_idle(self) -> None:
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("stop the active task before changing this setting")

    def _run_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        enabled = bool(self._settings["plva_enabled"])
        environment.update(
            {
                "PLVA_PROVIDER": str(self._settings["provider"]),
                "PLVA_MODEL": str(self._settings["model"]),
                "PLVA_REDACT": "1" if enabled else "0",
                "PLVA_REDACT_ENGINE": "vision",
                "PLVA_VISION_MODE": str(self._settings["vision_mode"]),
                "PLVA_REDACT_LIFECYCLE": str(self._settings["lifecycle"]),
                "PLVA_DETECTOR_VERSION": str(self._settings["detector_version"]),
                "PLVA_OCR_ENGINE": str(self._settings["ocr_engine"]),
                "PLVA_VISUAL_DETECTOR": "1" if self._settings["visual_detector"] == "on" else "0",
                "PLVA_SEMANTIC_ENGINE": str(self._settings["semantic_engine"]),
                "PLVA_TOOLS": "1" if self._settings["tools"] == "on" else "0",
                "PLVA_PRIVACY": "1" if enabled else "0",
                "PLVA_POLICY_JSON": json.dumps(self._policy, separators=(",", ":")),
            }
        )
        for name, variable in FEATURE_ENV.items():
            environment[variable] = "1" if self._settings["features"][name] else "0"
        source = self._credential_source()
        environment["PLVA_CREDENTIAL_SOURCE"] = source
        inject_provider_keys(
            environment,
            provider=str(self._settings["provider"]),
            source=source,
            project_root=ROOT,
        )
        return environment

    def _read_process(self, process: subprocess.Popen[str]) -> None:
        stdout = process.stdout
        if stdout is not None:
            for line in stdout:
                trace = _safe_agent_trace(line)
                event = _safe_runner_event(line)
                if trace is not None or event is not None:
                    with self._lock:
                        if trace is not None:
                            if self._traces and all(
                                self._traces[-1][key] == trace[key] for key in ("channel", "text")
                            ):
                                trace = None
                            if trace is not None:
                                if len(self._traces) == self._traces.maxlen:
                                    self._traces_dropped += 1
                                self._traces.append(trace)
                        if event is not None:
                            self._event(*event)
                            if event[0] in {"Privacy engine ready", "Provider connected"}:
                                self._status = "running"
        exit_code = process.wait()
        # The proxy can outlive the runner by only a moment; sweep any call
        # records from the run's final seconds before the viewer disappears.
        for _ in range(3):
            self._mirror_calls()
            time.sleep(0.25)
        with self._lock:
            if self._process is process:
                self._process = None
                self._finished_at = time.time()
                if self._status == "stopping":
                    self._status = "stopped"
                    self._event("Task stopped", "The private session was closed.")
                elif exit_code == 0:
                    self._status = "completed"
                    self._event(
                        "Task complete",
                        "The run is saved to your local audit history for replay.",
                    )
                else:
                    self._status = "failed"
                    self._event("Task needs attention", "Open Advanced lab for safe diagnostics.")
                self._history.update_run(
                    self._run_id,
                    status=self._status,
                    finished_at=self._finished_at,
                    events=list(self._events),
                )

    def _monitor_proxy(self, process: subprocess.Popen[str]) -> None:
        while process.poll() is None:
            frame = _fetch_bytes("/viewer/frame")
            stats = _fetch_json("/viewer/stats")
            findings = _fetch_json("/viewer/findings")
            vault = _fetch_json("/viewer/vault")
            approvals = _fetch_json("/viewer/approvals")
            filter_report = _fetch_json("/viewer/filter")
            with self._lock:
                if frame is not None:
                    self._frame = frame
                if stats is not None:
                    self._stats = stats
                if findings is not None:
                    self._findings = findings
                if vault is not None:
                    self._vault = vault
                if approvals is not None and isinstance(approvals.get("approvals"), list):
                    self._approvals = approvals["approvals"]
                if filter_report is not None:
                    self._filter = filter_report
            self._mirror_calls()
            time.sleep(0.35)

    def _mirror_calls(self) -> None:
        """Copy new proxy call records into memory so history outlives the run.

        The proxy buffer is the source of truth while it lives; a call is
        mirrored only once its images are all fetched, so a partially copied
        record is retried on the next tick instead of appearing broken.
        """

        index = _fetch_json("/viewer/calls")
        if index is None:
            return
        with self._lock:
            known = set(self._calls)
        for item in index.get("calls", []) if isinstance(index.get("calls"), list) else []:
            call_id = item.get("id")
            if not isinstance(call_id, int) or call_id in known:
                continue
            record = _fetch_json(f"/viewer/call/{call_id}")
            if record is None:
                continue
            image_types = record.get("images")
            images: list[tuple[str, bytes]] = []
            complete = True
            for image_index, media_type in enumerate(
                image_types if isinstance(image_types, list) else []
            ):
                blob = _fetch_bytes(f"/viewer/call/{call_id}/image/{image_index}")
                if blob is None:
                    complete = False
                    break
                images.append((str(media_type), blob))
            if not complete:
                continue
            with self._lock:
                self._calls[call_id] = record
                for image_index, image in enumerate(images):
                    self._call_images[(call_id, image_index)] = image
                while len(self._calls) > 48:
                    oldest = min(self._calls)
                    del self._calls[oldest]
                    self._call_images = {
                        key: value for key, value in self._call_images.items() if key[0] != oldest
                    }
                run_id = self._run_id
            self._history.save_call(run_id, record, images)

    def _event(self, title: str, detail: str) -> None:
        self._events.append({"title": title, "detail": detail, "time": time.strftime("%H:%M:%S")})
        self._history.update_run(self._run_id, events=list(self._events))


def _safe_runner_event(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if text.startswith("--- redaction ON"):
        return "Privacy engine ready", "Vision, Core ML, OCR, and the vault are active."
    if text.startswith("--- redaction OFF"):
        return "PLVA bypassed", "This diagnostic task is running without redaction."
    if text.startswith("--- preflight:"):
        return "Checking provider", "Verifying the selected model without sending a frame."
    if text.endswith("advertised: True"):
        return "Provider connected", "The selected Holo model is available."
    if text.startswith(("--- runs dir shredded", "--- ephemeral runs dir removed")):
        return "Private artifacts cleared", "Temporary runtime files were removed."
    if text.startswith("--- holo exit: 0"):
        return "Agent finished", "The requested task completed end-to-end."
    if "ERROR" in text.upper():
        return "Runner reported an error", "Sensitive command output is intentionally hidden here."
    return None


def _safe_agent_trace(line: str) -> dict[str, str] | None:
    """Keep useful Holo output in memory without retaining the user's task or credentials."""

    text = _ANSI_ESCAPE.sub("", line).strip()
    if not text or text.startswith(("user:", "--- task:")):
        return None
    text = text.strip("│ ")
    if not text or all(character in "─━═╭╮╰╯┌┐└┘┄┈ " for character in text):
        return None
    text = _INLINE_SECRET.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    text = "".join(character for character in text if character >= " " or character == "\t")
    if len(text) > 2_000:
        text = text[:1_997] + "…"
    lowered = text.lower()
    if text.startswith("---"):
        channel = "runtime"
    elif "💭" in text or lowered.startswith(("thought", "reasoning")):
        channel = "reasoning"
    elif "⚡" in text or any(
        marker in lowered
        for marker in ("click", "write", "scroll", "keypress", "tool_call", "action")
    ):
        channel = "action"
    elif "📝" in text or "answer" in lowered:
        channel = "answer"
    elif "error" in lowered or "failed" in lowered:
        channel = "error"
    else:
        return None
    return {"time": time.strftime("%H:%M:%S"), "channel": channel, "text": text}


def _fetch_bytes(path: str) -> bytes | None:
    try:
        with urllib.request.urlopen(PROXY_BASE + path, timeout=0.3) as response:
            return bytes(response.read())
    except (OSError, urllib.error.URLError):
        return None


def _fetch_json(path: str) -> dict[str, Any] | None:
    raw = _fetch_bytes(path)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _proxy_json_request(
    path: str, *, method: str, payload: dict[str, Any]
) -> dict[str, Any] | None:
    request = urllib.request.Request(
        PROXY_BASE + path,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        method=method,
        headers={"content-type": "application/json", "origin": PROXY_BASE},
    )
    try:
        with urllib.request.urlopen(request, timeout=1.0) as response:
            value = json.loads(response.read())
    except (OSError, UnicodeDecodeError, ValueError, urllib.error.URLError):
        return None
    return value if isinstance(value, dict) else None


def create_demo_app(controller: DemoController | None = None) -> FastAPI:
    active = controller or DemoController()

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
        try:
            yield
        finally:
            active.close()

    app = FastAPI(title="PLVA", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(LANDING_PATH.read_text("utf-8"), headers={"cache-control": "no-store"})

    @app.get("/app")
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(UI_PATH.read_text("utf-8"), headers={"cache-control": "no-store"})

    @app.get("/api/state")
    async def state() -> Response:
        return _json_response(active.snapshot())

    @app.get("/api/models")
    def models() -> Response:
        # Sync handler on purpose: FastAPI runs it in a worker thread, keeping
        # the upstream /models fetch off the event loop.
        return _json_response({"providers": active.model_catalog()})

    @app.put("/api/policy")
    async def policy(request: Request) -> Response:
        try:
            selected = active.set_policy(await request.json())
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _json_response({"policy": selected})

    @app.put("/api/settings")
    async def settings(request: Request) -> Response:
        try:
            selected = active.set_settings(await request.json())
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _json_response({"settings": selected, "credentials": active._credentials_snapshot()})

    @app.post("/api/credentials/connect-holo-cli")
    async def connect_holo_cli() -> Response:
        try:
            credentials = active.connect_holo_cli()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _json_response(
            {
                "settings": active.snapshot()["settings"],
                "credentials": credentials,
            }
        )

    @app.post("/api/run")
    async def run(request: Request) -> Response:
        body = await request.json()
        try:
            active.start(body.get("prompt") if isinstance(body, dict) else None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _json_response({"started": True}, status=202)

    @app.post("/api/stop")
    async def stop() -> Response:
        active.stop()
        return _json_response({"stopping": True}, status=202)

    @app.get("/api/frame")
    async def frame() -> Response:
        png = active.frame()
        if png is None:
            raise HTTPException(status_code=404, detail="no redacted frame yet")
        return Response(png, media_type="image/png", headers={"cache-control": "no-store"})

    @app.get("/api/vault")
    async def vault() -> Response:
        return _json_response(active.vault())

    @app.get("/api/approvals")
    async def approvals() -> Response:
        return _json_response({"approvals": active.approvals()})

    @app.post("/api/approve")
    async def approve(request: Request) -> Response:
        body = await request.json()
        try:
            grant = active.approve_once(body.get("token") if isinstance(body, dict) else None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _json_response({"approval": grant}, status=201)

    @app.get("/api/findings")
    async def findings() -> Response:
        return _json_response(active.findings())

    @app.get("/api/filter")
    async def filter_report() -> Response:
        return _json_response(active.filter_diagnostics())

    @app.get("/api/calls")
    async def calls() -> Response:
        return _json_response({"calls": active.calls()})

    @app.get("/api/call/{call_id}")
    async def call(call_id: int) -> Response:
        record = active.call(call_id)
        if record is None:
            raise HTTPException(status_code=404, detail="call not found")
        return _json_response(record)

    @app.get("/api/call/{call_id}/image/{index}")
    async def call_image(call_id: int, index: int) -> Response:
        image = active.call_image(call_id, index)
        if image is None:
            raise HTTPException(status_code=404, detail="call image not found")
        media_type, data = image
        return Response(data, media_type=media_type, headers={"cache-control": "no-store"})

    @app.get("/api/traces")
    async def traces() -> Response:
        return _json_response(active.traces())

    @app.get("/api/history/runs")
    async def history_runs() -> Response:
        return _json_response({"runs": active.history_runs()})

    @app.get("/api/history/runs/{run_id}")
    async def history_run(run_id: str) -> Response:
        record = active.history_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return _json_response(record)

    @app.get("/api/history/runs/{run_id}/calls/{call_id}")
    async def history_call(run_id: str, call_id: int) -> Response:
        record = active.history_call(run_id, call_id)
        if record is None:
            raise HTTPException(status_code=404, detail="call not found")
        return _json_response(record)

    @app.get("/api/history/runs/{run_id}/calls/{call_id}/image/{index}")
    async def history_image(run_id: str, call_id: int, index: int) -> Response:
        image = active.history_image(run_id, call_id, index)
        if image is None:
            raise HTTPException(status_code=404, detail="image not found")
        media_type, data = image
        return Response(data, media_type=media_type, headers={"cache-control": "no-store"})

    return app


def _json_response(value: Any, *, status: int = 200) -> Response:
    return Response(
        json.dumps(value, separators=(",", ":")),
        status_code=status,
        media_type="application/json",
        headers={"cache-control": "no-store"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.port == 18081:
        parser.error("--port must be 1-65535 and cannot be the proxy port 18081")
    uvicorn.run(
        create_demo_app(),
        host=LOOPBACK_HOST,
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
