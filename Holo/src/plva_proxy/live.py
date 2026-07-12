"""Continuous local screen redaction viewer — nothing leaves the machine.

``plva-live`` captures the screen in a loop with macOS ``screencapture``,
redacts each frame through the frozen plva-v2-baseline pipeline, and serves
the obscured result at ``http://127.0.0.1:<port>/viewer``. There is no
upstream and no key: frames exist only in a memory ring buffer and in a
per-cycle temp file that is deleted immediately after redaction. This shows,
live and continuously, exactly what a model behind the PLVA proxy would see.
Cycle time is dominated by the v2 pipeline (roughly 4-10 s per frame
depending on resolution); ``--scale`` trades detail for speed.
"""

from __future__ import annotations

import argparse
import io
import logging
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Final

import uvicorn
from fastapi import FastAPI
from PIL import Image

from plva_proxy.proxy import FrameStore, add_viewer_routes
from plva_proxy.redactor import PROFILES, RedactionError, RedactorConfig, redact_png
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
    store: FrameStore, config: RedactorConfig, *, scale: float, interval: float
) -> None:
    """Capture, redact, and publish frames until the process exits."""

    while True:
        started = time.monotonic()
        try:
            store.add(redact_png(config, capture_screen_png(scale)))
        except RedactionError as exc:
            _LOGGER.warning("live cycle skipped: %s", exc)
            time.sleep(2.0)
        remaining = interval - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)


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

    store = FrameStore()
    config = RedactorConfig(cli_path=cli_path, profile=args.redact_profile)
    app = FastAPI(title="PLVA live viewer", docs_url=None, redoc_url=None)
    add_viewer_routes(app, store)
    threading.Thread(
        target=run_capture_loop,
        args=(store, config),
        kwargs={"scale": args.scale, "interval": args.interval},
        daemon=True,
    ).start()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    _LOGGER.info("live viewer: http://127.0.0.1:%d/viewer", args.port)
    uvicorn.run(app, host=LOOPBACK_HOST, port=args.port, access_log=False, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
