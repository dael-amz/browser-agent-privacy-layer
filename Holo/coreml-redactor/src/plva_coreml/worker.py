"""Persistent stdin/stdout worker for the native Vision hybrid redactor."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from plva_coreml.hybrid import HybridANERedactor
from plva_coreml.ocr import OCRFinding
from plva_coreml.semantics import SEMANTIC_ENGINES
from plva_coreml.vision_hybrid import VISION_PIPELINE_MODES, HybridVisionRedactor


def _finding_json(finding: OCRFinding) -> dict[str, Any]:
    return {
        "x1": finding.x1,
        "y1": finding.y1,
        "x2": finding.x2,
        "y2": finding.y2,
        "text": finding.text,
        "detector_score": finding.detector_score,
        "ocr_confidence": finding.ocr_confidence,
        "labels": list(finding.labels),
        "sources": list(finding.sources),
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
        "sensitive": finding.sensitive,
        "uncertain": finding.uncertain,
    }


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, separators=(",", ":")), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--visual-model", type=Path, default=None)
    parser.add_argument("--profile", choices=("high-recall", "balanced"), default="high-recall")
    parser.add_argument("--mode", choices=VISION_PIPELINE_MODES, default="cascade")
    parser.add_argument("--engine", choices=("apple", "rapidocr"), default="apple")
    parser.add_argument("--no-visual", action="store_true")
    parser.add_argument("--semantic-engine", choices=SEMANTIC_ENGINES, default="rampart")
    args = parser.parse_args()
    visual_model = args.visual_model.resolve() if args.visual_model is not None else None
    visual_enabled = not args.no_visual
    if args.engine == "rapidocr":
        pipeline: HybridANERedactor | HybridVisionRedactor = HybridANERedactor(
            args.baseline.resolve(),
            args.cache.resolve(),
            profile=args.profile,
            visual_model=visual_model,
            visual_enabled=visual_enabled,
            semantic_engine=args.semantic_engine,
        )
        backend = "vision-rapidocr"
    else:
        pipeline = HybridVisionRedactor(
            args.baseline.resolve(),
            args.cache.resolve(),
            profile=args.profile,
            mode=args.mode,
            visual_model=visual_model,
            visual_enabled=visual_enabled,
            semantic_engine=args.semantic_engine,
        )
        backend = f"vision-{args.mode}"
    if not visual_enabled:
        backend += "-novisual"
    try:
        pipeline.warm()
        _emit({"ready": True, "backend": backend, "threaded": True})
        for line in sys.stdin:
            identifier = ""
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("request")
                identifier = str(request.get("id", ""))
                if request.get("operation") == "classify_texts":
                    texts = request.get("texts")
                    if (
                        not identifier
                        or not isinstance(texts, list)
                        or len(texts) > 256
                        or any(not isinstance(text, str) for text in texts)
                        or sum(len(text) for text in texts) > 2_000_000
                    ):
                        raise ValueError("request")
                    classified = pipeline.classify_texts(tuple(texts))
                    _emit(
                        {
                            "id": identifier,
                            "ok": True,
                            "classifications": [_finding_json(finding) for finding in classified],
                        }
                    )
                    continue
                encoded = request.get("image")
                if not identifier or not isinstance(encoded, str):
                    raise ValueError("request")
                png = base64.b64decode(encoded, validate=True)
                if isinstance(pipeline, HybridANERedactor):
                    with Image.open(io.BytesIO(png)) as loaded:
                        source = loaded.convert("RGB")
                        source.load()
                    result = pipeline.process(source)
                else:
                    result = pipeline.process(png)
                timings = dict(result.timings)
                timings["workerTotalMs"] = timings.get("total_ms", 0.0)
                _emit(
                    {
                        "id": identifier,
                        "ok": True,
                        "image": base64.b64encode(result.png).decode("ascii"),
                        "backend": backend,
                        "counts": result.counts,
                        "timings": timings,
                        "findings": [_finding_json(finding) for finding in result.findings],
                    }
                )
            except (ValueError, TypeError, binascii.Error, RuntimeError, OSError):
                _emit({"id": identifier, "ok": False, "error": "FrameError"})
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
