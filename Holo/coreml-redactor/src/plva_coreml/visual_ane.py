"""Fixed-shape visual inference through ONNX Runtime's Core ML provider."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from plva_coreml.coreml_session import CoreMLSessionError, create_ane_session
from plva_coreml.model_cache import prepare_fixed_model

INPUT_SHAPE: Final = (1, 3, 640, 640)


class ANEError(RuntimeError):
    """Raised when the ANE-enabled backend cannot safely initialize or run."""


def prepare_fixed_visual_model(source: Path, destination: Path) -> Path:
    """Create the static-shape ONNX derivative required for ANE specialization."""

    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise ANEError(f"visual model not found: {source}")
    try:
        return prepare_fixed_model(source, destination, {"images": INPUT_SHAPE})
    except (OSError, ValueError) as exc:
        raise ANEError(f"could not prepare fixed visual model: {type(exc).__name__}") from exc


class VisualANESession:
    """Warm visual session with GPU excluded and Core ML/ANE enabled.

    Core ML may retain CPU fallback for unsupported graph nodes. This class
    therefore reports an ANE-enabled path, not a guarantee that every node ran
    on the Neural Engine.
    """

    def __init__(self, model: Path, *, cache_directory: Path | None = None) -> None:
        try:
            self._session = create_ane_session(model, cache_directory=cache_directory)
        except CoreMLSessionError as exc:
            raise ANEError(str(exc)) from exc

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        """Run one normalized NCHW detector tensor."""

        if tensor.shape != INPUT_SHAPE or tensor.dtype != np.float32:
            raise ANEError(f"expected float32 tensor shaped {INPUT_SHAPE}")
        try:
            output = self._session.run(None, {"images": tensor})[0]
        except Exception as exc:
            raise ANEError(f"Core ML inference failed: {type(exc).__name__}") from exc
        if not isinstance(output, np.ndarray) or output.shape != (1, 13, 8400):
            raise ANEError("Core ML returned an unexpected detector output")
        return output

    def warm(self) -> None:
        """Compile and specialize the fixed graph before a real frame arrives."""

        self.infer(np.zeros(INPUT_SHAPE, dtype=np.float32))
