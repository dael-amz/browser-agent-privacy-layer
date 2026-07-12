"""Persistent accelerated redaction plus the frozen one-shot baseline fallback.

``AcceleratedRedactor`` owns one warm browser worker, concurrent visual/OCR
branches, WebGPU selection, and a redacted-output-only memory cache. The
legacy ``redact_png`` wrapper remains as a correctness oracle and writes only
to a private temporary directory. Both paths raise ``RedactionError`` on any
failure so callers fail closed; neither has a raw-frame fallback.
"""

from __future__ import annotations

import binascii
import copy
import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
from base64 import b64decode, b64encode
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from itertools import count
from pathlib import Path
from typing import Any, Final, TextIO

_LOGGER: Final = logging.getLogger(__name__)

PROFILES: Final = ("high-recall", "balanced")
BACKENDS: Final = ("auto", "webgpu", "wasm")


class RedactionError(RuntimeError):
    """Raised when a frame cannot be redacted; the caller must fail closed."""


@dataclass(frozen=True, slots=True)
class RedactorConfig:
    """Location and behavior of the frozen v2 CLI."""

    cli_path: Path
    node_path: str = "node"
    profile: str = "high-recall"
    timeout_s: float = 180.0


@dataclass(frozen=True, slots=True)
class AcceleratedRedactorConfig:
    """Persistent browser worker configuration."""

    baseline_root: Path
    worker_script: Path
    node_path: str = "node"
    backend: str = "auto"
    profile: str = "high-recall"
    startup_timeout_s: float = 180.0
    frame_timeout_s: float = 180.0
    # A CUA request can carry many historical frames. Four entries caused an
    # oldest-to-newest LRU thrash once trajectories grew past four steps.
    cache_entries: int = 32
    idle_timeout_s: float | None = 60.0
    worker_kind: str = "browser"
    worker_root: Path | None = None
    cache_root: Path | None = None
    vision_mode: str = "cascade"
    visual_model: Path | None = None
    ocr_engine: str = "apple"
    visual_enabled: bool = True
    semantic_engine: str = "rampart"


class AcceleratedRedactor:
    """Persistent, parallel, hardware-accelerated frame redactor.

    One browser process owns warm ONNX sessions during an active CUA burst.
    Calls are serialized to avoid CPU/GPU oversubscription, while the worker
    runs its independent visual and OCR branches concurrently. By default the
    process starts on demand and exits after an idle minute; ``None`` keeps it
    warm indefinitely and zero selects cold-per-call operation. Only redacted
    output bytes are cached; cache keys and values are memory-only and cleared
    on close.
    """

    def __init__(self, config: AcceleratedRedactorConfig) -> None:
        if config.worker_kind not in {"browser", "vision"}:
            raise ValueError("worker_kind must be browser or vision")
        if config.worker_kind == "browser" and config.backend not in BACKENDS:
            raise ValueError(f"backend must be one of: {', '.join(BACKENDS)}")
        if config.worker_kind == "vision" and config.vision_mode not in {
            "fast",
            "cascade",
            "accurate",
        }:
            raise ValueError("vision_mode must be fast, cascade, or accurate")
        if config.worker_kind == "vision" and config.ocr_engine not in {"apple", "rapidocr"}:
            raise ValueError("ocr_engine must be apple or rapidocr")
        if config.worker_kind == "vision" and config.semantic_engine not in {
            "rampart",
            "gliner2",
            "openai-pf",
        }:
            raise ValueError("semantic_engine must be rampart, gliner2, or openai-pf")
        if config.profile not in PROFILES:
            raise ValueError(f"profile must be one of: {', '.join(PROFILES)}")
        if config.cache_entries < 0:
            raise ValueError("cache_entries cannot be negative")
        if config.idle_timeout_s is not None and config.idle_timeout_s < 0:
            raise ValueError("idle_timeout_s cannot be negative")
        self._config = config
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._ids = count(1)
        self._cache: OrderedDict[bytes, tuple[bytes, dict[str, Any]]] = OrderedDict()
        self._text_cache: OrderedDict[bytes, list[dict[str, Any]]] = OrderedDict()
        self._backend = "not-started"
        self._analysis: dict[str, Any] = {}
        self._idle_timer: threading.Timer | None = None

    @property
    def backend(self) -> str:
        """Return the active backend after startup."""

        return self._backend

    @property
    def latest_analysis(self) -> dict[str, Any]:
        """Return a detached copy of the latest worker metadata and OCR findings."""

        with self._lock:
            return copy.deepcopy(self._analysis)

    def start(self) -> None:
        """Start and warm the worker before the first frame arrives."""

        with self._lock:
            self._cancel_idle_timer()
            self._ensure_started()
            self._arm_idle_timer()

    def __call__(self, png: bytes) -> bytes:
        """Redact one PNG, failing closed on every worker/protocol error."""

        key = sha256(png).digest()
        with self._lock:
            try:
                self._cancel_idle_timer()
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    redacted, analysis = cached
                    self._analysis = copy.deepcopy(analysis)
                    _LOGGER.info("redacted frame: memory cache hit")
                    return redacted

                self._ensure_started()
                process = self._process
                if process is None or process.stdin is None:
                    raise RedactionError("accelerated worker is unavailable")
                request_id = str(next(self._ids))
                request = {
                    "id": request_id,
                    "profile": self._config.profile,
                    "image": b64encode(png).decode("ascii"),
                }
                try:
                    process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                    process.stdin.flush()
                except (BrokenPipeError, OSError) as exc:
                    self._stop_process()
                    raise RedactionError("accelerated worker stopped") from exc

                response = self._wait_response(self._config.frame_timeout_s)
                if response.get("id") != request_id:
                    self._stop_process()
                    raise RedactionError("accelerated worker protocol mismatch")
                if response.get("ok") is not True:
                    raise RedactionError("accelerated worker rejected frame")
                try:
                    redacted = b64decode(str(response["image"]), validate=True)
                except (KeyError, ValueError, binascii.Error) as exc:
                    self._stop_process()
                    raise RedactionError("accelerated worker returned invalid output") from exc
                if not redacted.startswith(b"\x89PNG\r\n\x1a\n"):
                    self._stop_process()
                    raise RedactionError("accelerated worker returned a non-PNG output")

                self._backend = _safe_backend(response.get("backend"))
                counts = response.get("counts")
                timings = response.get("timings")
                regions = counts.get("fused", "?") if isinstance(counts, dict) else "?"
                duration = timings.get("workerTotalMs", "?") if isinstance(timings, dict) else "?"
                self._analysis = _safe_analysis(response, self._backend)
                _LOGGER.info(
                    "redacted frame: %s region(s) masked backend=%s duration_ms=%s",
                    regions,
                    self._backend,
                    duration,
                )
                self._remember(key, redacted, self._analysis)
                return redacted
            finally:
                if self._process is not None:
                    self._arm_idle_timer()

    def close(self) -> None:
        """Stop the worker and erase its memory-only frame cache."""

        with self._lock:
            self._cache.clear()
            self._text_cache.clear()
            self._analysis = {}
            self._cancel_idle_timer()
            self._stop_process()
            self._backend = "closed"

    def classify_texts(self, texts: tuple[str, ...]) -> list[dict[str, Any]]:
        """Classify history through the warm Vision worker's Core ML Rampart session."""

        if self._config.worker_kind != "vision":
            raise RedactionError("history classification requires the Vision worker")
        if not texts:
            return []
        encoded = json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        key = sha256(encoded).digest()
        with self._lock:
            try:
                self._cancel_idle_timer()
                cached = self._text_cache.get(key)
                if cached is not None:
                    self._text_cache.move_to_end(key)
                    return copy.deepcopy(cached)
                self._ensure_started()
                process = self._process
                if process is None or process.stdin is None:
                    raise RedactionError("Vision worker is unavailable")
                request_id = str(next(self._ids))
                request = {
                    "id": request_id,
                    "operation": "classify_texts",
                    "texts": list(texts),
                }
                try:
                    process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                    process.stdin.flush()
                except (BrokenPipeError, OSError) as exc:
                    self._stop_process()
                    raise RedactionError("Vision worker stopped") from exc
                response = self._wait_response(self._config.frame_timeout_s)
                classifications = response.get("classifications")
                if (
                    response.get("id") != request_id
                    or response.get("ok") is not True
                    or not isinstance(classifications, list)
                    or len(classifications) != len(texts)
                    or any(not isinstance(item, dict) for item in classifications)
                ):
                    raise RedactionError("Vision history-classification protocol mismatch")
                result: list[dict[str, Any]] = copy.deepcopy(classifications)
                self._text_cache[key] = copy.deepcopy(result)
                self._text_cache.move_to_end(key)
                while len(self._text_cache) > max(1, self._config.cache_entries):
                    self._text_cache.popitem(last=False)
                return result
            finally:
                if self._process is not None:
                    self._arm_idle_timer()

    def _ensure_started(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            return
        baseline = self._config.baseline_root.resolve()
        if not (baseline / "snapshot.json").is_file():
            raise RedactionError("frozen baseline assets are missing")
        environment = os.environ.copy()
        environment["PLVA_BASELINE_ROOT"] = str(baseline)
        if self._config.worker_kind == "browser":
            script = self._config.worker_script.resolve()
            if not script.is_file():
                raise RedactionError("accelerated worker script is missing")
            if not (script.parent.parent / "dist" / "index.html").is_file():
                raise RedactionError(
                    "accelerated worker is not built (run npm install && npm run build)"
                )
            command = [self._config.node_path, str(script), "--backend", self._config.backend]
            working_directory = script.parent.parent
        else:
            root = self._config.worker_root
            cache = self._config.cache_root
            if root is None or cache is None:
                raise RedactionError("Vision worker paths are not configured")
            root = root.resolve()
            module = root / "src" / "plva_coreml" / "worker.py"
            python = Path(self._config.node_path).expanduser().absolute()
            if not module.is_file() or not python.is_file():
                raise RedactionError(
                    "Vision worker is not installed (run uv sync in coreml-redactor)"
                )
            existing_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = str(root / "src") + (
                os.pathsep + existing_pythonpath if existing_pythonpath else ""
            )
            command = [
                str(python),
                "-m",
                "plva_coreml.worker",
                "--baseline",
                str(baseline),
                "--cache",
                str(cache.resolve()),
                "--profile",
                self._config.profile,
                "--mode",
                self._config.vision_mode,
                "--engine",
                self._config.ocr_engine,
                "--semantic-engine",
                self._config.semantic_engine,
            ]
            if not self._config.visual_enabled:
                command.append("--no-visual")
            elif self._config.visual_model is not None:
                visual_model = self._config.visual_model.resolve()
                if not visual_model.is_file():
                    raise RedactionError("configured visual detector is missing")
                command.extend(("--visual-model", str(visual_model)))
            working_directory = root
        try:
            responses: queue.Queue[dict[str, Any] | None] = queue.Queue()
            self._responses = responses
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=working_directory,
                env=environment,
            )
        except OSError as exc:
            self._process = None
            raise RedactionError("accelerated worker did not start") from exc
        stdout = self._process.stdout
        if stdout is None:
            self._stop_process()
            raise RedactionError("accelerated worker has no protocol output")
        self._reader = threading.Thread(
            target=self._read_responses,
            args=(stdout, responses),
            daemon=True,
        )
        self._reader.start()
        ready = self._wait_response(self._config.startup_timeout_s)
        if ready.get("ready") is not True:
            self._stop_process()
            raise RedactionError("accelerated worker initialization failed")
        self._backend = _safe_backend(ready.get("backend"))
        _LOGGER.info(
            "accelerated redactor ready backend=%s threaded=%s",
            self._backend,
            ready.get("threaded") is True,
        )

    @staticmethod
    def _read_responses(stdout: TextIO, responses: queue.Queue[dict[str, Any] | None]) -> None:
        try:
            for line in stdout:
                try:
                    value = json.loads(line)
                except ValueError:
                    responses.put(None)
                    return
                responses.put(value if isinstance(value, dict) else None)
        finally:
            responses.put(None)

    def _wait_response(self, timeout: float) -> dict[str, Any]:
        try:
            response = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            self._stop_process()
            raise RedactionError("accelerated worker timed out") from exc
        if response is None:
            self._stop_process()
            raise RedactionError("accelerated worker protocol ended")
        return response

    def _remember(self, key: bytes, redacted: bytes, analysis: dict[str, Any]) -> None:
        if self._config.cache_entries == 0:
            return
        self._cache[key] = (redacted, copy.deepcopy(analysis))
        self._cache.move_to_end(key)
        while len(self._cache) > self._config.cache_entries:
            self._cache.popitem(last=False)

    def _arm_idle_timer(self) -> None:
        self._cancel_idle_timer()
        timeout = self._config.idle_timeout_s
        if timeout is None:
            return
        if timeout == 0:
            self._stop_process()
            self._backend = "idle"
            return
        timer = threading.Timer(timeout, self._expire_idle_worker)
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _cancel_idle_timer(self) -> None:
        timer = self._idle_timer
        self._idle_timer = None
        if timer is not None:
            timer.cancel()

    def _expire_idle_worker(self) -> None:
        with self._lock:
            if self._idle_timer is not threading.current_thread():
                return
            self._idle_timer = None
            self._stop_process()
            self._backend = "idle"
            _LOGGER.info("accelerated redactor released after idle timeout")

    def _stop_process(self) -> None:
        self._cancel_idle_timer()
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            with suppress(OSError):
                process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()


def _safe_backend(value: Any) -> str:
    backend = str(value)
    return backend if backend in {"webgpu", "wasm"} or backend.startswith("vision-") else "unknown"


def _safe_analysis(response: dict[str, Any], backend: str) -> dict[str, Any]:
    counts = response.get("counts")
    timings = response.get("timings")
    findings = response.get("findings")
    return {
        "backend": backend,
        "counts": copy.deepcopy(counts) if isinstance(counts, dict) else {},
        "timings": copy.deepcopy(timings) if isinstance(timings, dict) else {},
        "findings": copy.deepcopy(findings) if isinstance(findings, list) else [],
    }


def redact_png(config: RedactorConfig, png: bytes) -> bytes:
    """Run one PNG frame through the v2 pipeline and return the redacted PNG."""

    cli_path = config.cli_path.resolve()
    with tempfile.TemporaryDirectory(prefix="plva-redact-") as tmp:
        tmp_dir = Path(tmp)
        source = tmp_dir / "frame.png"
        output = tmp_dir / "frame.redacted.png"
        report = tmp_dir / "frame.report.json"
        source.write_bytes(png)
        command = [
            config.node_path,
            str(cli_path),
            str(source),
            "--output",
            str(output),
            "--report",
            str(report),
            "--profile",
            config.profile,
            "--force",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                timeout=config.timeout_s,
                check=False,
                cwd=cli_path.parent.parent,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RedactionError(f"redactor did not run: {type(exc).__name__}") from exc
        if completed.returncode != 0:
            raise RedactionError(f"redactor exited {completed.returncode}")
        try:
            redacted = output.read_bytes()
            counts = json.loads(report.read_text("utf-8")).get("counts", {})
        except (OSError, ValueError) as exc:
            raise RedactionError("redactor produced no readable output") from exc
    _LOGGER.info("redacted frame: %s region(s) masked", counts.get("fused", "?"))
    return redacted
