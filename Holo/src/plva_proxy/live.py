"""Continuous local screen redaction viewer — nothing leaves the machine.

``plva-live`` captures the screen in a loop with macOS ``screencapture``,
redacts each frame through the persistent accelerated worker, and serves
the obscured result at ``http://127.0.0.1:<port>/viewer``. There is no
upstream and no key: frames exist only in a memory ring buffer and in a
per-cycle temp file that is deleted immediately after redaction. This shows,
live and continuously, exactly what a model behind the PLVA proxy would see.
The worker stays warm for the process lifetime; ``--scale`` trades detector
detail for throughput in this viewer-only mode.
"""

from __future__ import annotations

import argparse
import io
import logging
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

import uvicorn
from fastapi import FastAPI
from PIL import Image

from plva_proxy.proxy import FrameStore, add_viewer_routes
from plva_proxy.redactor import (
    BACKENDS,
    PROFILES,
    AcceleratedRedactor,
    AcceleratedRedactorConfig,
    RedactionError,
    RedactorConfig,
    redact_png,
)
from plva_proxy.runtime_capture import LOOPBACK_HOST

DEFAULT_PORT: Final = 18082
_LOGGER: Final = logging.getLogger(__name__)


def capture_screen_png(scale: float) -> bytes:
    """Capture the main display via ``screencapture``; optionally downscale."""

    with tempfile.TemporaryDirectory(prefix="plva-live-") as tmp:
        shot = Path(tmp) / "capture.png"
        completed = subprocess.run(
            ["screencapture", "-x", "-t", "png", str(shot)],
            capture_output=True,
            timeout=20,
            check=False,
        )
        if completed.returncode != 0 or not shot.is_file():
            raise RedactionError("screencapture failed (is Screen Recording permitted?)")
        data = shot.read_bytes()
    if scale >= 1.0:
        return data
    with Image.open(io.BytesIO(data)) as image:
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        buffer = io.BytesIO()
        image.resize(size).convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()


def run_capture_loop(  # pragma: no cover - endless loop, exercised manually
    store: FrameStore,
    redact: Callable[[bytes], bytes],
    stop: threading.Event,
    *,
    scale: float,
    interval: float,
) -> None:
    """Capture, redact, and publish frames until the process exits."""

    while not stop.is_set():
        started = time.monotonic()
        try:
            store.add(redact(capture_screen_png(scale)))
        except RedactionError as exc:
            _LOGGER.warning("live cycle skipped: %s", exc)
            stop.wait(2.0)
        remaining = interval - (time.monotonic() - started)
        if remaining > 0:
            stop.wait(remaining)


def main() -> None:  # pragma: no cover - thin CLI wiring, exercised manually
    """Run the continuous local redaction viewer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--redact",
        type=Path,
        default=Path("plva-v2-baseline"),
        help="plva-v2-baseline directory (or its bin/plva-v2.mjs)",
    )
    parser.add_argument("--redact-profile", choices=PROFILES, default="high-recall")
    parser.add_argument(
        "--redact-engine",
        choices=("accelerated", "baseline"),
        default="accelerated",
        help="persistent parallel worker (default) or one-process-per-frame baseline",
    )
    parser.add_argument("--redact-backend", choices=BACKENDS, default="auto")
    parser.add_argument("--redact-worker", type=Path, default=Path("redactor-worker"))
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="downscale factor for captures; smaller is faster (default 0.5)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="minimum seconds between capture cycles (default: back-to-back)",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 0.05 <= args.scale <= 1.0:
        parser.error("--scale must be between 0.05 and 1.0")
    cli_path = args.redact / "bin" / "plva-v2.mjs" if args.redact.is_dir() else args.redact
    if not cli_path.is_file():
        parser.error(f"--redact CLI not found: {cli_path}")
    if shutil.which("node") is None:
        parser.error("redaction requires node on PATH")

    accelerated: AcceleratedRedactor | None = None
    if args.redact_engine == "accelerated":
        worker_script = args.redact_worker / "bin" / "redactor-worker.mjs"
        if not worker_script.is_file() or not (args.redact_worker / "dist/index.html").is_file():
            parser.error(
                f"accelerated worker is not built in {args.redact_worker}; "
                "run npm install && npm run build there"
            )
        accelerated = AcceleratedRedactor(
            AcceleratedRedactorConfig(
                baseline_root=cli_path.parent.parent,
                worker_script=worker_script,
                backend=args.redact_backend,
                profile=args.redact_profile,
                idle_timeout_s=None,
            )
        )
        accelerated.start()
        redact: Callable[[bytes], bytes] = accelerated
    else:
        config = RedactorConfig(cli_path=cli_path, profile=args.redact_profile)

        def redact(png: bytes) -> bytes:
            return redact_png(config, png)

    store = FrameStore()
    app = FastAPI(title="PLVA live viewer", docs_url=None, redoc_url=None)
    add_viewer_routes(app, store)
    stop = threading.Event()
    capture_thread = threading.Thread(
        target=run_capture_loop,
        args=(store, redact, stop),
        kwargs={"scale": args.scale, "interval": args.interval},
        daemon=True,
    )
    capture_thread.start()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    _LOGGER.info("live viewer: http://127.0.0.1:%d/viewer", args.port)
    try:
        uvicorn.run(app, host=LOOPBACK_HOST, port=args.port, access_log=False, log_level="warning")
    finally:
        stop.set()
        if accelerated is not None:
            accelerated.close()
        capture_thread.join(timeout=5)


if __name__ == "__main__":  # pragma: no cover
    main()
