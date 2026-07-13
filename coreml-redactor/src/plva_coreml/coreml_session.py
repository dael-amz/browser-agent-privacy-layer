"""Shared Core ML session construction with GPU explicitly excluded."""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort


class CoreMLSessionError(RuntimeError):
    """Raised when a Core ML session is unavailable or invalid."""


def create_ane_session(model: Path, *, cache_directory: Path | None = None) -> ort.InferenceSession:
    """Create a static Core ML NeuralNetwork session eligible for ANE execution."""

    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        raise CoreMLSessionError("this ONNX Runtime build has no Core ML execution provider")
    options = {
        "ModelFormat": "NeuralNetwork",
        "MLComputeUnits": "CPUAndNeuralEngine",
        "RequireStaticInputShapes": "1",
        "EnableOnSubgraphs": "0",
    }
    if cache_directory is not None:
        cache_directory.mkdir(parents=True, exist_ok=True)
        options["ModelCacheDirectory"] = str(cache_directory.resolve())
    try:
        session = ort.InferenceSession(
            str(model.resolve()),
            providers=[("CoreMLExecutionProvider", options), "CPUExecutionProvider"],
        )
    except Exception as exc:
        raise CoreMLSessionError(
            f"Core ML session initialization failed: {type(exc).__name__}"
        ) from exc
    if session.get_providers()[0] != "CoreMLExecutionProvider":
        raise CoreMLSessionError("Core ML was not selected as the primary execution provider")
    return session
