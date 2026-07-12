"""Memory-only PLVA consumer demo and task launcher."""

from __future__ import annotations

import argparse
import copy
import json
import os
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

from plva_proxy.privacy import SafetyPolicy
from plva_proxy.runtime_capture import LOOPBACK_HOST

ROOT: Final = Path(__file__).resolve().parents[2]
UI_PATH: Final = Path(__file__).with_name("demo_ui.html")
DEFAULT_POLICY_PATH: Final = ROOT / "config" / "privacy-policy.json"
PROXY_BASE: Final = "http://127.0.0.1:18081"
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


class DemoController:
    """Own one local CUA run and retain only memory-resident demo artifacts."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._status = "idle"
        self._started_at = 0.0
        self._finished_at = 0.0
        self._events: deque[dict[str, str]] = deque(maxlen=24)
        self._frame: bytes | None = None
        self._vault: dict[str, Any] = {"entries": [], "policy": {}}
        self._findings: dict[str, Any] = {}
        self._filter: dict[str, Any] = {"status": "idle"}
        self._stats: dict[str, Any] = {}
        self._policy = _load_policy().snapshot()
        self._settings: dict[str, Any] = {
            "plva_enabled": True,
            "provider": "hcompany",
            "vision_mode": "cascade",
            "lifecycle": "eager",
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
                "policy": dict(self._policy),
                "settings": json.loads(json.dumps(self._settings)),
                "stats": dict(self._stats),
                "has_frame": self._frame is not None,
                "vault_count": len(self._vault.get("entries", [])),
                "filter": dict(self._filter),
            }

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
        vision_mode = raw.get("vision_mode")
        lifecycle = raw.get("lifecycle")
        plva_enabled = raw.get("plva_enabled")
        features = raw.get("features")
        if provider not in {"hcompany", "overshoot"}:
            raise ValueError("provider is invalid")
        if vision_mode not in {"fast", "cascade", "accurate"}:
            raise ValueError("vision mode is invalid")
        if lifecycle not in {"adaptive", "eager", "cold"}:
            raise ValueError("lifecycle is invalid")
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
                "vision_mode": vision_mode,
                "lifecycle": lifecycle,
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
            self._frame = None
            self._vault = {"entries": [], "policy": dict(self._policy)}
            self._findings = {}
            self._filter = {"status": "idle"}
            self._stats = {}
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

    def findings(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._findings)

    def filter_diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._filter)

    def _require_idle(self) -> None:
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("stop the active task before changing this setting")

    def _run_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        enabled = bool(self._settings["plva_enabled"])
        environment.update(
            {
                "PLVA_PROVIDER": str(self._settings["provider"]),
                "PLVA_REDACT": "1" if enabled else "0",
                "PLVA_REDACT_ENGINE": "vision",
                "PLVA_VISION_MODE": str(self._settings["vision_mode"]),
                "PLVA_REDACT_LIFECYCLE": str(self._settings["lifecycle"]),
                "PLVA_PRIVACY": "1" if enabled else "0",
                "PLVA_POLICY_JSON": json.dumps(self._policy, separators=(",", ":")),
            }
        )
        for name, variable in FEATURE_ENV.items():
            environment[variable] = "1" if self._settings["features"][name] else "0"
        return environment

    def _read_process(self, process: subprocess.Popen[str]) -> None:
        stdout = process.stdout
        if stdout is not None:
            for line in stdout:
                event = _safe_runner_event(line)
                if event is not None:
                    with self._lock:
                        self._event(*event)
                        if event[0] in {"Privacy engine ready", "Provider connected"}:
                            self._status = "running"
        exit_code = process.wait()
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
                        "The runner closed; the demo snapshot stays in memory until the next run.",
                    )
                else:
                    self._status = "failed"
                    self._event("Task needs attention", "Open Advanced lab for safe diagnostics.")

    def _monitor_proxy(self, process: subprocess.Popen[str]) -> None:
        while process.poll() is None:
            frame = _fetch_bytes("/viewer/frame")
            stats = _fetch_json("/viewer/stats")
            findings = _fetch_json("/viewer/findings")
            vault = _fetch_json("/viewer/vault")
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
                if filter_report is not None:
                    self._filter = filter_report
            time.sleep(0.35)

    def _event(self, title: str, detail: str) -> None:
        self._events.append({"title": title, "detail": detail, "time": time.strftime("%H:%M:%S")})


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
    if text.startswith("--- runs dir shredded"):
        return "Private artifacts cleared", "Temporary runtime files were removed."
    if text.startswith("--- holo exit: 0"):
        return "Agent finished", "The requested task completed end-to-end."
    if "ERROR" in text.upper():
        return "Runner reported an error", "Sensitive command output is intentionally hidden here."
    return None


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
        return HTMLResponse(UI_PATH.read_text("utf-8"), headers={"cache-control": "no-store"})

    @app.get("/api/state")
    async def state() -> Response:
        return _json_response(active.snapshot())

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
        return _json_response({"settings": selected})

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

    @app.get("/api/findings")
    async def findings() -> Response:
        return _json_response(active.findings())

    @app.get("/api/filter")
    async def filter_report() -> Response:
        return _json_response(active.filter_diagnostics())

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
