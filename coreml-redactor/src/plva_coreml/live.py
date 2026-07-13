"""Local-only live viewer for the experimental ANE visual redactor."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final

from PIL import Image

from plva_coreml.hybrid import HybridANERedactor, HybridResult
from plva_coreml.ocr import OCRFinding
from plva_coreml.visual_ane import ANEError
from plva_coreml.visual_redactor import THRESHOLD_PROFILES

_LOGGER: Final = logging.getLogger(__name__)
_HOST: Final = "127.0.0.1"
_DEFAULT_PORT: Final = 18083

_VIEWER_HTML: Final = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>PLVA accelerated hybrid redactor</title><style>
:root{color-scheme:dark;font-family:ui-sans-serif,system-ui,sans-serif;background:#090b0a;color:#e9eeeb}
body{margin:0;padding:28px;display:grid;gap:18px;min-height:100vh;box-sizing:border-box}
header{display:flex;gap:18px;align-items:end;justify-content:space-between;flex-wrap:wrap}
h1{font-size:24px;margin:0}.sub{color:#98a29c;margin-top:7px}.warning{color:#ffcb6b}
#status{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:10px}
.metric{background:#141815;border:1px solid #2a332d;border-radius:12px;padding:12px}
.label{color:#91a097;font-size:12px;text-transform:uppercase;letter-spacing:.08em}
.value{font:600 21px ui-monospace,SFMono-Regular,monospace;margin-top:5px}
.stage{grid-column:1/-1;color:#aeb8b2;font:13px ui-monospace,SFMono-Regular,monospace}
.frame{background:#111512;border:1px solid #303a34;border-radius:16px;padding:10px;display:grid;place-items:center;min-height:260px}
img{display:block;max-width:100%;max-height:70vh;border-radius:9px}.empty{color:#728078}
@media(max-width:700px){#status{grid-template-columns:repeat(2,1fr)}body{padding:16px}}
</style></head><body><header><div><h1>PLVA · Accelerated hybrid redactor</h1>
<div class="sub">Visual + RapidOCR + Rampart · <span class="warning">local evaluation; parity validation pending</span></div>
</div><div id="state">Starting…</div></header><section id="status">
<div class="metric"><div class="label">Frames</div><div class="value" id="frames">0</div></div>
<div class="metric"><div class="label">Total</div><div class="value" id="total">—</div></div>
<div class="metric"><div class="label">OCR</div><div class="value" id="ocr">—</div></div>
<div class="metric"><div class="label">Rampart</div><div class="value" id="rampart">—</div></div>
<div class="metric"><div class="label">Regions</div><div class="value" id="regions">—</div></div>
<div class="stage" id="detail">Initializing fixed Core ML models…</div></section>
<div class="frame"><span class="empty" id="empty">Waiting for the first frame…</span><img id="frame" alt="Latest hybrid-redacted frame"></div>
<script>
let version=0;async function tick(){try{const r=await fetch('/stats',{cache:'no-store'});const s=await r.json();
document.getElementById('state').textContent=s.status;document.getElementById('frames').textContent=s.frames;
document.getElementById('total').textContent=s.total_ms==null?'—':s.total_ms.toFixed(1)+' ms';
document.getElementById('ocr').textContent=s.ocr_ms==null?'—':s.ocr_ms.toFixed(1)+' ms';
document.getElementById('rampart').textContent=s.rampart_ms==null?'—':s.rampart_ms.toFixed(1)+' ms';
document.getElementById('regions').textContent=s.regions==null?'—':s.regions;
document.getElementById('detail').textContent=s.detail;
if(s.version!==version&&s.frames>0){version=s.version;const image=document.getElementById('frame');
const old=image.src;image.src='/frame?v='+version;if(old&&old.startsWith('blob:'))URL.revokeObjectURL(old);
document.getElementById('empty').style.display='none';}}catch(e){}setTimeout(tick,200)}tick();
</script></body></html>"""


class ViewerState:
    """Thread-safe, memory-only latest frame and timing metadata."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._png: bytes | None = None
        self._findings: tuple[dict[str, Any], ...] = ()
        self._stats: dict[str, Any] = {
            "status": "initializing",
            "detail": "Preparing fixed Core ML model…",
            "frames": 0,
            "version": 0,
            "regions": None,
            "total_ms": None,
            "ocr_ms": None,
            "rampart_ms": None,
        }

    def update_status(self, status: str, detail: str) -> None:
        with self._lock:
            self._stats.update(status=status, detail=detail)

    def publish(self, png: bytes, *, regions: int, total_ms: float, inference_ms: float) -> None:
        with self._lock:
            self._png = png
            self._stats.update(
                status="running",
                detail="CPUAndNeuralEngine · visual + RapidOCR + Rampart",
                frames=int(self._stats["frames"]) + 1,
                version=int(self._stats["version"]) + 1,
                regions=regions,
                total_ms=total_ms,
                ocr_ms=inference_ms,
                rampart_ms=0.0,
            )

    def publish_result(self, result: HybridResult) -> None:
        findings = tuple(_serialize_finding(finding) for finding in result.findings)
        with self._lock:
            self._png = result.png
            self._findings = findings
            self._stats.update(
                status="running",
                detail="CPUAndNeuralEngine · visual + RapidOCR + Rampart · findings in /findings",
                frames=int(self._stats["frames"]) + 1,
                version=int(self._stats["version"]) + 1,
                regions=result.counts["fused"],
                total_ms=result.timings["total_ms"],
                ocr_ms=result.timings["ocr_ms"],
                rampart_ms=result.timings["rampart_ms"],
                counts=result.counts,
            )

    def snapshot(self) -> tuple[bytes | None, dict[str, Any], tuple[dict[str, Any], ...]]:
        with self._lock:
            return self._png, dict(self._stats), self._findings


def _serialize_finding(finding: OCRFinding) -> dict[str, Any]:
    return {
        "bounds": {"x1": finding.x1, "y1": finding.y1, "x2": finding.x2, "y2": finding.y2},
        "text": finding.text,
        "detector_score": finding.detector_score,
        "ocr_confidence": finding.ocr_confidence,
        "labels": list(finding.labels),
        "sources": list(finding.sources),
        "sensitive": finding.sensitive,
        "uncertain": finding.uncertain,
        "values": [
            {
                "label": value.label,
                "value": value.value,
                "start": value.start,
                "end": value.end,
                "score": value.score,
                "source": value.source,
            }
            for value in finding.values
        ],
    }


def _handler_for(state: ViewerState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            png, stats, findings = state.snapshot()
            if self.path == "/" or self.path.startswith("/?"):
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", _VIEWER_HTML.encode())
            elif self.path == "/health":
                self._send(HTTPStatus.OK, "application/json", b'{"ok":true}')
            elif self.path == "/stats":
                self._send(HTTPStatus.OK, "application/json", json.dumps(stats).encode())
            elif self.path == "/findings":
                body = {"version": stats["version"], "findings": findings}
                self._send(HTTPStatus.OK, "application/json", json.dumps(body).encode())
            elif self.path == "/frame" or self.path.startswith("/frame?"):
                if png is None:
                    self._send(HTTPStatus.NOT_FOUND, "application/json", b'{"detail":"no frame"}')
                else:
                    self._send(HTTPStatus.OK, "image/png", png)
            else:
                self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found")

        def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self'; connect-src 'self'; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def _capture_screen(scale: float) -> Image.Image:
    executable = shutil.which("screencapture") or "/usr/sbin/screencapture"
    with tempfile.TemporaryDirectory(prefix="plva-ane-live-") as directory:
        path = Path(directory) / "screen.png"
        completed = subprocess.run(
            [executable, "-x", "-t", "png", str(path)],
            capture_output=True,
            timeout=20,
            check=False,
        )
        if completed.returncode != 0 or not path.is_file():
            raise ANEError("screen capture failed; grant Screen Recording permission")
        with Image.open(path) as captured:
            image = captured.convert("RGB")
    if scale < 1:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
    return image


def _run_pipeline(
    state: ViewerState,
    stop: threading.Event,
    *,
    baseline: Path,
    cache: Path,
    visual_model: Path | None,
    fixture: Path | None,
    profile: str,
    scale: float,
    interval: float,
) -> None:
    try:
        state.update_status("initializing", "Compiling visual, OCR, and Rampart Core ML graphs…")
        pipeline = HybridANERedactor(
            baseline, cache, profile=profile, visual_model=visual_model
        )
        pipeline.warm()
        state.update_status("running", "Core ML ready; capturing the first frame…")
        while not stop.is_set():
            started = time.monotonic()
            if fixture is None:
                source = _capture_screen(scale)
            else:
                with Image.open(fixture) as loaded:
                    source = loaded.convert("RGB")
            result = pipeline.process(source)
            state.publish_result(result)
            stop.wait(max(0.0, interval - (time.monotonic() - started)))
    except Exception as exc:
        _LOGGER.exception("ANE viewer pipeline stopped")
        state.update_status("failed", f"{type(exc).__name__}: {exc}")
    finally:
        if "pipeline" in locals():
            pipeline.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    parser.add_argument("--baseline", type=Path, default=Path("../plva-v2-baseline"))
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--visual-model", type=Path, default=None)
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument("--profile", choices=tuple(THRESHOLD_PROFILES), default="high-recall")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--interval", type=float, default=0.25)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 0.05 <= args.scale <= 1:
        parser.error("--scale must be between 0.05 and 1")
    if args.interval < 0:
        parser.error("--interval cannot be negative")
    if args.fixture is not None and not args.fixture.is_file():
        parser.error(f"fixture not found: {args.fixture}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    state = ViewerState()
    stop = threading.Event()
    worker = threading.Thread(
        target=_run_pipeline,
        args=(state, stop),
        kwargs={
            "baseline": args.baseline.resolve(),
            "cache": args.cache.resolve(),
            "visual_model": (
                args.visual_model.resolve() if args.visual_model is not None else None
            ),
            "fixture": args.fixture.resolve() if args.fixture is not None else None,
            "profile": args.profile,
            "scale": args.scale,
            "interval": args.interval,
        },
        daemon=True,
    )
    worker.start()
    server = ThreadingHTTPServer((_HOST, args.port), _handler_for(state))
    _LOGGER.info("ANE viewer: http://%s:%d/", _HOST, args.port)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        worker.join(timeout=5)


if __name__ == "__main__":
    main()
