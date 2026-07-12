"""Frame redaction through the frozen plva-v2-baseline CLI.

Wraps ``node bin/plva-v2.mjs`` — the bundled headless-Chrome + local-ONNX
pipeline that burns detected-PII masks into a PNG. Frames touch disk only
inside a private temporary directory that is deleted before returning, and
the CLI's geometry-only JSON report is read for a region count and discarded.
Any failure raises RedactionError so the caller can fail closed (§8.1);
there is no raw-frame fallback (§8.2). Logs carry region counts and exit
codes only — never pixels, recognized text, or report contents.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_LOGGER: Final = logging.getLogger(__name__)

PROFILES: Final = ("high-recall", "balanced")


class RedactionError(RuntimeError):
    """Raised when a frame cannot be redacted; the caller must fail closed."""


@dataclass(frozen=True, slots=True)
class RedactorConfig:
    """Location and behavior of the frozen v2 CLI."""

    cli_path: Path
    node_path: str = "node"
    profile: str = "high-recall"
    timeout_s: float = 180.0


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
