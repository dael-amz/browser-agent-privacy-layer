from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from plva_proxy.redactor import (
    AcceleratedRedactor,
    AcceleratedRedactorConfig,
    RedactionError,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

FAKE_WORKER = r"""
import argparse
import base64
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--backend")
args = parser.parse_args()
print(json.dumps({"ready": True, "backend": "webgpu", "threaded": False}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    output = b"\x89PNG\r\n\x1a\n" + request["id"].encode()
    print(json.dumps({
        "id": request["id"],
        "ok": True,
        "image": base64.b64encode(output).decode(),
        "backend": "webgpu",
        "counts": {"fused": 2},
        "timings": {"workerTotalMs": 12},
    }), flush=True)
"""

FAKE_REJECTING_WORKER = r"""
import argparse
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--backend")
parser.parse_args()
print(json.dumps({"ready": True, "backend": "wasm", "threaded": False}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({"id": request["id"], "ok": False, "error": "FrameError"}), flush=True)
"""


def make_config(tmp_path: Path, source: str = FAKE_WORKER) -> AcceleratedRedactorConfig:
    worker_root = tmp_path / "redactor-worker"
    worker_script = worker_root / "bin" / "fake_worker.py"
    worker_script.parent.mkdir(parents=True)
    worker_script.write_text(source, "utf-8")
    dist = worker_root / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("worker", "utf-8")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "snapshot.json").write_text("{}", "utf-8")
    return AcceleratedRedactorConfig(
        baseline_root=baseline,
        worker_script=worker_script,
        node_path=sys.executable,
        startup_timeout_s=5,
        frame_timeout_s=5,
    )


def test_accelerated_redactor_stays_warm_and_caches_exact_frames(tmp_path: Path) -> None:
    redactor = AcceleratedRedactor(make_config(tmp_path))
    frame = PNG_SIGNATURE + b"raw-frame"

    redactor.start()
    first = redactor(frame)
    cached = redactor(frame)
    second = redactor(frame + b"-changed")
    redactor.close()

    assert first == PNG_SIGNATURE + b"1"
    assert cached == first
    assert second == PNG_SIGNATURE + b"2"
    assert redactor.backend == "closed"


def test_accelerated_redactor_fails_closed_when_worker_rejects_frame(tmp_path: Path) -> None:
    redactor = AcceleratedRedactor(make_config(tmp_path, FAKE_REJECTING_WORKER))
    try:
        with pytest.raises(RedactionError, match="rejected frame"):
            redactor(PNG_SIGNATURE + b"raw-frame")
    finally:
        redactor.close()


def test_accelerated_redactor_requires_built_worker(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    (config.worker_script.parent.parent / "dist" / "index.html").unlink()
    redactor = AcceleratedRedactor(config)

    with pytest.raises(RedactionError, match="not built"):
        redactor.start()


def test_accelerated_redactor_cold_mode_restarts_between_distinct_frames(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), idle_timeout_s=0)
    redactor = AcceleratedRedactor(config)

    try:
        first = redactor(PNG_SIGNATURE + b"first")
        assert redactor.backend == "idle"
        second = redactor(PNG_SIGNATURE + b"second")
        assert redactor.backend == "idle"
    finally:
        redactor.close()

    assert first == PNG_SIGNATURE + b"1"
    assert second == PNG_SIGNATURE + b"2"


def test_accelerated_redactor_rejects_negative_idle_timeout(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), idle_timeout_s=-1)

    with pytest.raises(ValueError, match="idle_timeout_s"):
        AcceleratedRedactor(config)


def test_accelerated_redactor_releases_worker_after_idle_timeout(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), idle_timeout_s=0.02)
    redactor = AcceleratedRedactor(config)

    try:
        redactor.start()
        deadline = time.monotonic() + 1
        while redactor.backend != "idle" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert redactor.backend == "idle"
    finally:
        redactor.close()
