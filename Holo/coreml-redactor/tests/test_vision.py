from __future__ import annotations

import pytest

from plva_coreml.ocr import OCRFinding
from plva_coreml.vision import VisionROI, _parse_observations
from plva_coreml.vision_hybrid import _fallback_rois, _merge_accurate


def test_vision_observations_convert_from_lower_left_to_pixel_bounds() -> None:
    findings = _parse_observations(
        [
            {
                "text": "person@example.com",
                "confidence": 0.8,
                "x": 0.1,
                "y": 0.2,
                "width": 0.3,
                "height": 0.1,
            }
        ],
        1000,
        500,
        mode="fast",
    )

    assert len(findings) == 1
    assert (findings[0].x1, findings[0].y1, findings[0].x2, findings[0].y2) == pytest.approx((
        100,
        350,
        400,
        400,
    ))
    assert findings[0].sources == ("OCR+VISION_FAST",)


def test_cascade_builds_padded_rois_only_for_sensitive_findings() -> None:
    safe = OCRFinding(0, 0, 100, 20, "safe", 0.8, 0.8)
    sensitive = OCRFinding(
        200,
        100,
        300,
        130,
        "person@example.com",
        0.8,
        0.8,
        labels=("EMAIL",),
        sensitive=True,
    )

    rois = _fallback_rois((safe, sensitive), 1000, 500)

    assert len(rois) == 1
    assert rois[0].x < 0.2
    assert rois[0].y < 0.2
    assert rois[0].width > 0.1


def test_accurate_findings_replace_fast_findings_inside_roi() -> None:
    fast = (
        OCRFinding(100, 100, 200, 130, "Pers0n", 0.5, 0.5),
        OCRFinding(10, 10, 50, 30, "Keep", 0.5, 0.5),
    )
    accurate = (OCRFinding(100, 100, 200, 130, "Person", 0.9, 0.9),)
    roi = VisionROI(0.09, 0.09, 0.12, 0.05)

    merged = _merge_accurate(fast, accurate, (roi,), 1000, 1000)

    assert [finding.text for finding in merged] == ["Keep", "Person"]
