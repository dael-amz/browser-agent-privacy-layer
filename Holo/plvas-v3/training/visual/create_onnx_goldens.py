from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(args.model), providers=["CPUExecutionProvider"]
    )
    model_input = session.get_inputs()[0]
    shape = (1, 3, args.image_size, args.image_size)
    size = int(np.prod(shape))
    values = (np.arange(size, dtype=np.uint32) % 256).astype(np.float32)
    image = (values / np.float32(255.0)).reshape(shape)
    output = session.run(None, {model_input.name: image})[0]
    if output.ndim != 3 or output.shape[0] != 1:
        raise RuntimeError(f"unexpected detector output shape: {output.shape}")
    channels, anchors = int(output.shape[1]), int(output.shape[2])
    anchor_samples = sorted(
        {
            0,
            1,
            7,
            63,
            511,
            1023,
            anchors // 2,
            anchors - 1,
        }
    )
    samples = [
        {
            "flat_index": channel * anchors + anchor,
            "channel": channel,
            "anchor": anchor,
            "value": float(output[0, channel, anchor]),
        }
        for channel in range(channels)
        for anchor in anchor_samples
    ]
    document = {
        "schema_version": 1,
        "model_sha256": sha256_file(args.model),
        "reference_runtime": {
            "package": "onnxruntime",
            "version": importlib.metadata.version("onnxruntime"),
            "provider": "CPUExecutionProvider",
        },
        "input": {
            "name": model_input.name,
            "dtype": "float32",
            "shape": list(shape),
            "algorithm": "float32((flat_index mod 256) / 255)",
        },
        "output": {
            "name": session.get_outputs()[0].name,
            "shape": list(output.shape),
            "samples": samples,
            "minimum": float(output.min()),
            "maximum": float(output.max()),
            "mean": float(output.mean()),
        },
        "tolerance": {"absolute": 1e-4, "relative": 1e-4},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic ONNX Runtime goldens for a visual detector"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(create(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
