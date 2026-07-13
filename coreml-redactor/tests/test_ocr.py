from __future__ import annotations

import numpy as np

from plva_coreml.ocr import _decode_ctc


def test_decode_ctc_collapses_blanks_and_repeated_classes() -> None:
    dictionary = ("<blank>", "a", "b", " ")
    probabilities = np.zeros((7, 4), dtype=np.float32)
    for step, class_id in enumerate((0, 1, 1, 0, 2, 2, 3)):
        probabilities[step, class_id] = 0.9

    text, confidence = _decode_ctc(probabilities, dictionary)

    assert text == "ab "
    assert confidence == np.float32(0.9)
