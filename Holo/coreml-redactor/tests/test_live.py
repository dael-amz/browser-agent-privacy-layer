from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from plva_coreml.hybrid import HybridResult
from plva_coreml.live import ViewerState, _handler_for
from plva_coreml.ocr import OCRFinding, PIIValue


def test_viewer_state_publishes_only_redacted_output_and_metrics() -> None:
    state = ViewerState()
    state.publish(b"redacted-png", regions=2, total_ms=31.5, inference_ms=9.4)

    png, stats, findings = state.snapshot()

    assert png == b"redacted-png"
    assert findings == ()
    assert (
        json.loads(json.dumps(stats)).items()
        >= {
            "status": "running",
            "frames": 1,
            "version": 1,
            "regions": 2,
            "total_ms": 31.5,
            "ocr_ms": 9.4,
        }.items()
    )


def test_viewer_serves_page_stats_and_latest_redacted_frame() -> None:
    state = ViewerState()
    finding = OCRFinding(
        0,
        0,
        100,
        20,
        "alice@example.com",
        0.9,
        0.95,
        labels=("EMAIL",),
        values=(PIIValue("EMAIL", "alice@example.com", 0, 17, 1.0, "heuristic"),),
        sensitive=True,
    )
    state.publish_result(
        HybridResult(
            png=b"redacted-png",
            regions=(),
            findings=(finding,),
            counts={"fused": 1},
            timings={"total_ms": 25, "ocr_ms": 20, "rampart_ms": 5},
        )
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/", timeout=2) as response:
            page = response.read().decode()
            assert response.status == 200
            assert "Accelerated hybrid redactor" in page
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["Content-Security-Policy"]
        with urllib.request.urlopen(base + "/stats", timeout=2) as response:
            assert json.load(response)["regions"] == 1
        with urllib.request.urlopen(base + "/frame", timeout=2) as response:
            assert response.headers["Content-Type"] == "image/png"
            assert response.read() == b"redacted-png"
        with urllib.request.urlopen(base + "/findings", timeout=2) as response:
            emitted = json.load(response)
            assert emitted["findings"][0]["text"] == "alice@example.com"
            assert emitted["findings"][0]["values"][0]["label"] == "EMAIL"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
