"""Fixed-shape visual inference through ONNX Runtime's Core ML provider."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Final

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_input_shape_fixed

INPUT_SHAPE: Final = (1, 3, 640, 640)


class ANEError(RuntimeError):
    """Raised when the ANE-enabled backend cannot safely initialize or run."""


def prepare_fixed_visual_model(source: Path, destination: Path) -> Path:
    """Create the static-shape ONNX derivative required for ANE specialization."""

    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise ANEError(f"visual model not found: {source}")
    digest = sha256(source.read_bytes()).hexdigest()
    digest_file = destination.with_suffix(destination.suffix + ".source-sha256")
    if (
        destination.is_file()
        and digest_file.is_file()
        and digest_file.read_text("ascii").strip() == digest
    ):
        return destination
    model = onnx.load_model(source)
    make_input_shape_fixed(model.graph, "images", list(INPUT_SHAPE))
    fix_output_shapes(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, destination)
    digest_file.write_text(digest + "\n", "ascii")
    return destination


class VisualANESession:
    """Warm visual session with GPU excluded and Core ML/ANE enabled.

    Core ML may retain CPU fallback for unsupported graph nodes. This class
    therefore reports an ANE-enabled path, not a guarantee that every node ran
    on the Neural Engine.
    """

    def __init__(self, model: Path, *, cache_directory: Path | None = None) -> None:
        if "CoreMLExecutionProvider" not in ort.get_available_providers():
            raise ANEError("this ONNX Runtime build has no Core ML execution provider")
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
            self._session = ort.InferenceSession(
                str(model.resolve()),
                providers=[("CoreMLExecutionProvider", options), "CPUExecutionProvider"],
            )
        except Exception as exc:
            raise ANEError(f"Core ML session initialization failed: {type(exc).__name__}") from exc
        if self._session.get_providers()[0] != "CoreMLExecutionProvider":
            raise ANEError("Core ML was not selected as the primary execution provider")

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
