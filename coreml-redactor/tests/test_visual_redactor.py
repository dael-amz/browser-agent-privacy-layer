from __future__ import annotations

import io

import numpy as np
from PIL import Image

from plva_coreml.visual_ane import visual_model_cache_key
from plva_coreml.visual_redactor import (
    REDACTION_RGB,
    Region,
    decode_detections,
    prepare_tensor,
    render_masks,
)


def test_visual_cache_key_changes_with_checkpoint_content(tmp_path) -> None:
    model = tmp_path / "detector.onnx"
    model.write_bytes(b"v2 checkpoint")
    first = visual_model_cache_key(model)

    model.write_bytes(b"v3 checkpoint")

    assert visual_model_cache_key(model) != first


def test_prepare_tensor_letterboxes_to_fixed_nchw_shape() -> None:
    image = Image.new("RGB", (800, 400), "white")

    tensor, transform = prepare_tensor(image)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert (transform.scaled_width, transform.scaled_height) == (640, 320)
    assert (transform.pad_left, transform.pad_top) == (0, 160)


def test_decode_detections_applies_threshold_mapping_and_nms() -> None:
    image = Image.new("RGB", (640, 640), "white")
    _, transform = prepare_tensor(image)
    output = np.zeros((1, 13, 8400), dtype=np.float32)
    for anchor, score, center_x in ((0, 0.02, 320.0), (1, 0.015, 321.0)):
        output[0, 0, anchor] = center_x
        output[0, 1, anchor] = 320
        output[0, 2, anchor] = 100
        output[0, 3, anchor] = 50
        output[0, 9, anchor] = score  # class 5, CVC

    regions = decode_detections(output, transform)

    assert len(regions) == 1
    assert regions[0].label == "CVC"
    assert regions[0].score == np.float32(0.02)
    assert (regions[0].x1, regions[0].y1, regions[0].x2, regions[0].y2) == (
        268.0,
        293.0,
        372.0,
        347.0,
    )


def test_render_masks_irreversibly_paints_region() -> None:
    source = Image.new("RGB", (20, 20), "white")
    region = Region(4.2, 5.1, 10.8, 12.2, 0, "NAME", 0.9)

    png = render_masks(source, (region,))

    with Image.open(io.BytesIO(png)) as output:
        assert output.getpixel((5, 6)) == REDACTION_RGB
        assert output.getpixel((0, 0)) == (255, 255, 255)
