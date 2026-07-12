"""Content-addressed fixed-shape model derivatives for Core ML specialization."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import onnx
from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_input_shape_fixed


def prepare_fixed_model(
    source: Path, destination: Path, input_shapes: dict[str, tuple[int, ...]]
) -> Path:
    """Create or reuse a fixed-input ONNX derivative keyed by source content and shapes."""

    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    signature = {
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "input_shapes": {name: list(shape) for name, shape in sorted(input_shapes.items())},
    }
    signature_file = destination.with_suffix(destination.suffix + ".source.json")
    if destination.is_file() and signature_file.is_file():
        try:
            if json.loads(signature_file.read_text("utf-8")) == signature:
                return destination
        except (OSError, ValueError):
            pass
    model = onnx.load_model(source)
    for name, shape in input_shapes.items():
        make_input_shape_fixed(model.graph, name, list(shape))
    fix_output_shapes(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, destination)
    signature_file.write_text(json.dumps(signature, sort_keys=True) + "\n", "utf-8")
    return destination
