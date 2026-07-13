"""Persistent native Apple Vision OCR client and geometry conversion."""

from __future__ import annotations

import base64
import io
import json
import os
import select
import subprocess
import threading
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

from PIL import Image

from plva_coreml.ocr import OCRFinding

VISION_MODES: Final = ("fast", "accurate")


class VisionError(RuntimeError):
    """Raised when native Vision compilation, execution, or protocol handling fails."""


@dataclass(frozen=True, slots=True)
class VisionROI:
    """Normalized top-left-origin region of interest."""

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class VisionResult:
    findings: tuple[OCRFinding, ...]
    duration_ms: float


class VisionOCRClient:
    """Own one native Vision subprocess and exchange frame data only through memory pipes."""

    def __init__(self, cache: Path, *, timeout_s: float = 30.0) -> None:
        self._binary = _prepare_binary(cache)
        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._next_id = 1
        try:
            self._process = subprocess.Popen(
                [str(self._binary)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise VisionError("native Vision worker did not start") from exc

    def warm(self) -> None:
        # A tiny in-memory PNG pays Vision's one-time model initialization before a CUA frame.
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), "white").save(buffer, format="PNG")
        tiny_png = buffer.getvalue()
        self.recognize(tiny_png, 16, 16, mode="fast")
        self.recognize(tiny_png, 16, 16, mode="accurate")

    def recognize(
        self,
        png: bytes,
        width: int,
        height: int,
        *,
        mode: str,
        rois: tuple[VisionROI, ...] = (),
    ) -> VisionResult:
        if mode not in VISION_MODES:
            raise ValueError(f"Vision mode must be one of: {', '.join(VISION_MODES)}")
        if width < 1 or height < 1:
            raise ValueError("image dimensions must be positive")
        with self._lock:
            identifier = str(self._next_id)
            self._next_id += 1
            request = {
                "id": identifier,
                "image": base64.b64encode(png).decode("ascii"),
                "mode": mode,
                "rois": [
                    {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height}
                    for roi in rois
                ],
            }
            process = self._process
            if process.stdin is None or process.stdout is None or process.poll() is not None:
                raise VisionError("native Vision worker is unavailable")
            try:
                process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise VisionError("native Vision worker stopped") from exc
            ready, _, _ = select.select([process.stdout], [], [], self._timeout_s)
            if not ready:
                raise VisionError("native Vision worker timed out")
            line = process.stdout.readline()
            if not line:
                raise VisionError("native Vision worker protocol ended")
            try:
                response = json.loads(line)
            except ValueError as exc:
                raise VisionError("native Vision worker returned invalid JSON") from exc
            if not isinstance(response, dict) or response.get("ok") is not True:
                raise VisionError("native Vision worker rejected the frame")
            if response.get("id") != identifier:
                raise VisionError("native Vision worker protocol mismatch")
            observations = response.get("observations")
            if not isinstance(observations, list):
                raise VisionError("native Vision worker returned no observations")
            findings = _parse_observations(observations, width, height, mode=mode)
            return VisionResult(findings, float(response.get("duration_ms", 0.0)))

    def close(self) -> None:
        with self._lock:
            process = self._process
            if process.stdin is not None:
                with suppress(OSError):
                    process.stdin.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            if process.stdout is not None:
                process.stdout.close()


def _prepare_binary(cache: Path) -> Path:
    source = Path(__file__).with_name("native") / "vision_ocr_worker.swift"
    if not source.is_file():
        raise VisionError("native Vision source is missing")
    compiler = "/usr/bin/swiftc"
    if not Path(compiler).is_file():
        raise VisionError("Swift compiler is unavailable")
    digest = sha256(source.read_bytes() + os.uname().machine.encode()).hexdigest()[:16]
    output = cache / "bin" / f"plva-vision-{digest}"
    if output.is_file():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    try:
        completed = subprocess.run(
            [compiler, "-O", str(source), "-o", str(temporary)],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise VisionError("native Vision worker did not compile") from exc
    if completed.returncode != 0 or not temporary.is_file():
        raise VisionError("native Vision worker did not compile")
    temporary.chmod(0o700)
    temporary.replace(output)
    return output


def _parse_observations(
    observations: list[Any], width: int, height: int, *, mode: str
) -> tuple[OCRFinding, ...]:
    findings: list[OCRFinding] = []
    for raw in observations:
        if not isinstance(raw, dict):
            continue
        try:
            text = str(raw["text"]).strip()
            confidence = min(1.0, max(0.0, float(raw["confidence"])))
            x = float(raw["x"])
            y = float(raw["y"])
            box_width = float(raw["width"])
            box_height = float(raw["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if not text or box_width <= 0 or box_height <= 0:
            continue
        x1 = min(float(width), max(0.0, x * width))
        x2 = min(float(width), max(x1, (x + box_width) * width))
        y1 = min(float(height), max(0.0, (1.0 - y - box_height) * height))
        y2 = min(float(height), max(y1, (1.0 - y) * height))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        uncertain = confidence < 0.35
        findings.append(
            OCRFinding(
                x1,
                y1,
                x2,
                y2,
                text,
                confidence,
                confidence,
                labels=("UNREADABLE",) if uncertain else (),
                sources=(f"OCR+VISION_{mode.upper()}",),
                sensitive=uncertain,
                uncertain=uncertain,
            )
        )
    return tuple(sorted(findings, key=lambda finding: (finding.y1, finding.x1)))
