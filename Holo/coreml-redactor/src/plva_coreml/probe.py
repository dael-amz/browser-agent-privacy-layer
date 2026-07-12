"""Benchmark the separate ANE-enabled visual model on the synthetic fixture."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from plva_coreml.visual_ane import VisualANESession, prepare_fixed_visual_model
from plva_coreml.visual_redactor import prepare_tensor


def _detector_tensor(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as source:
        tensor, _ = prepare_tensor(source)
    return tensor


def _cpu_output(model: Path, tensor: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    output: Any = session.run(None, {"images": tensor})[0]
    if not isinstance(output, np.ndarray):
        raise RuntimeError("CPU reference returned an invalid output")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("../plva-v2-baseline"))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    source_model = args.baseline / "dist/visual/detector.onnx"
    fixture = args.baseline / "fixtures/ats-smoke.png"
    fixed_model = prepare_fixed_visual_model(source_model, args.cache / "visual-fixed.onnx")
    tensor = _detector_tensor(fixture).astype(np.float32, copy=False)

    started = time.perf_counter()
    session = VisualANESession(fixed_model, cache_directory=args.cache / "compiled")
    session.warm()
    startup_ms = (time.perf_counter() - started) * 1000
    samples = []
    output = None
    for _ in range(args.runs):
        started = time.perf_counter()
        output = session.infer(tensor)
        samples.append((time.perf_counter() - started) * 1000)
    assert output is not None
    reference = _cpu_output(fixed_model, tensor)
    difference = np.abs(output - reference)
    print(
        json.dumps(
            {
                "backend": "CoreMLExecutionProvider/CPUAndNeuralEngine",
                "model_format": "NeuralNetwork",
                "startup_and_warm_ms": round(startup_ms, 2),
                "runs": args.runs,
                "median_ms": round(statistics.median(samples), 2),
                "minimum_ms": round(min(samples), 2),
                "maximum_ms": round(max(samples), 2),
                "cpu_reference_max_abs_difference": float(difference.max()),
                "accepted_fixture_anchors": int(
                    np.any(
                        output[0, 4:]
                        >= np.array([0.1, 0.1, 0.1, 0.1, 0.08, 0.01, 0.08, 0.08, 0.35])[:, None],
                        axis=0,
                    ).sum()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
