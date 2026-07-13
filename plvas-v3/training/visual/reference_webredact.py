from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "visual" / "webredact" / "nano640.xml"
DEFAULT_CONTRACT = ROOT / "webredact_contract.json"


def model_metadata(xml_path: Path) -> dict[str, str]:
    root = ElementTree.parse(xml_path).getroot()
    model_info = root.find("./rt_info/model_info")
    if model_info is None:
        raise ValueError("WebRedact IR is missing model_info")
    return {
        child.tag: child.attrib["value"]
        for child in model_info
        if "value" in child.attrib
    }


def letterbox(
    image: Image.Image, size: int, pad_value: int
) -> tuple[np.ndarray, float, float, float]:
    image = image.convert("RGB")
    source_width, source_height = image.size
    ratio = min(size / source_width, size / source_height)
    resized_width = round(source_width * ratio)
    resized_height = round(source_height * ratio)
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    delta_width = size - resized_width
    delta_height = size - resized_height
    left = round(delta_width / 2 - 0.1)
    top = round(delta_height / 2 - 0.1)
    canvas = Image.new("RGB", (size, size), (pad_value, pad_value, pad_value))
    canvas.paste(resized, (left, top))
    tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return tensor, ratio, float(left), float(top)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.copy()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    top_left = np.maximum(one[:2], many[:, :2])
    bottom_right = np.minimum(one[2:], many[:, 2:])
    intersection = np.prod(np.maximum(0.0, bottom_right - top_left), axis=1)
    one_area = max(0.0, one[2] - one[0]) * max(0.0, one[3] - one[1])
    many_area = np.maximum(0.0, many[:, 2] - many[:, 0]) * np.maximum(
        0.0, many[:, 3] - many[:, 1]
    )
    union = one_area + many_area - intersection
    return np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    *,
    iou_threshold: float,
    maximum_detections: int,
) -> list[int]:
    kept: list[int] = []
    for class_id in sorted(set(class_ids.tolist())):
        remaining = np.where(class_ids == class_id)[0]
        remaining = remaining[np.argsort(-scores[remaining], kind="stable")]
        while len(remaining) and len(kept) < maximum_detections:
            selected = int(remaining[0])
            kept.append(selected)
            if len(remaining) == 1:
                break
            overlaps = box_iou(boxes[selected], boxes[remaining[1:]])
            remaining = remaining[1:][overlaps <= iou_threshold]
    return sorted(
        kept, key=lambda index: (-float(scores[index]), int(class_ids[index]), index)
    )[:maximum_detections]


def decode(
    output: np.ndarray,
    *,
    threshold: float,
    iou_threshold: float,
    maximum_detections: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if output.shape[0] != 1 or output.shape[1] != 6:
        raise ValueError(f"unexpected WebRedact output shape: {output.shape}")
    rows = output[0].transpose(1, 0)
    class_scores = rows[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)
    accepted = scores >= threshold
    boxes = xywh_to_xyxy(rows[accepted, :4])
    scores = scores[accepted]
    class_ids = class_ids[accepted]
    if not len(boxes):
        return boxes, scores, class_ids
    kept = class_aware_nms(
        boxes,
        scores,
        class_ids,
        iou_threshold=iou_threshold,
        maximum_detections=maximum_detections,
    )
    return boxes[kept], scores[kept], class_ids[kept]


def run(args: argparse.Namespace) -> dict[str, Any]:
    from openvino import Core

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    metadata = model_metadata(args.model)
    expected_metadata = {
        "iou_threshold": str(contract["postprocessing"]["nms_iou_threshold"]),
        "labels": " ".join(contract["model"]["classes"]),
        "model_type": "YOLO",
        "pad_value": str(contract["preprocessing"]["pad_value"]),
        "resize_type": contract["preprocessing"]["resize"],
        "reverse_input_channels": "YES",
        "scale_values": str(contract["preprocessing"]["scale"]),
    }
    if metadata != expected_metadata:
        raise ValueError(
            f"WebRedact metadata changed: {metadata} != {expected_metadata}"
        )
    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
    if fixtures.get("synthetic_only") is not True:
        raise ValueError("WebRedact goldens must use synthetic-only fixtures")

    core = Core()
    model = core.read_model(str(args.model))
    if list(model.input(0).shape) != contract["model"]["input_shape"]:
        raise ValueError("WebRedact input shape changed")
    if list(model.output(0).shape) != contract["model"]["output_shape"]:
        raise ValueError("WebRedact output shape changed")
    compiled = core.compile_model(model, "CPU")
    size = contract["model"]["resolution"][0]
    records = []
    for fixture in fixtures["fixtures"]:
        image_path = args.fixtures.parent / fixture["image"]
        image_bytes = image_path.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != fixture["sha256"]:
            raise ValueError(f"fixture hash mismatch: {fixture['id']}")
        image = Image.open(image_path).convert("RGB")
        tensor, ratio, pad_x, pad_y = letterbox(
            image, size, contract["preprocessing"]["pad_value"]
        )
        raw = compiled([tensor])[compiled.output(0)]
        boxes, scores, class_ids = decode(
            raw,
            threshold=args.threshold,
            iou_threshold=contract["postprocessing"]["nms_iou_threshold"],
            maximum_detections=contract["postprocessing"]["maximum_detections"],
        )
        proposals = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            source_box = [
                (float(box[0]) - pad_x) / ratio,
                (float(box[1]) - pad_y) / ratio,
                (float(box[2]) - pad_x) / ratio,
                (float(box[3]) - pad_y) / ratio,
            ]
            source_box[0] = min(max(source_box[0], 0.0), image.width)
            source_box[2] = min(max(source_box[2], 0.0), image.width)
            source_box[1] = min(max(source_box[1], 0.0), image.height)
            source_box[3] = min(max(source_box[3], 0.0), image.height)
            proposals.append(
                {
                    "class": contract["model"]["classes"][int(class_id)],
                    "confidence": round(float(score), 7),
                    "xyxy": [round(value, 4) for value in source_box],
                }
            )
        records.append(
            {
                "id": fixture["id"],
                "image_sha256": fixture["sha256"],
                "source_size": [image.width, image.height],
                "preprocess": {"ratio": ratio, "pad_x": pad_x, "pad_y": pad_y},
                "proposals": proposals,
            }
        )
    report = {
        "schema_version": 1,
        "model_revision": contract["source"]["revision"],
        "model_xml_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "threshold": args.threshold,
        "threshold_status": "smoke-only-not-frozen",
        "metadata": metadata,
        "fixtures": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pinned WebRedact OpenVINO reference"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "fixtures": len(report["fixtures"]),
                "proposals": sum(len(item["proposals"]) for item in report["fixtures"]),
                "threshold_status": report["threshold_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
