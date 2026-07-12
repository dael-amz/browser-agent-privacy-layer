from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
class AlignmentTests(unittest.TestCase):
    def test_aligns_wordpieces_to_character_spans(self) -> None:
        from training.modeling import align_offsets
        from training.schema import LABEL_CONFIG

        offsets = [(0, 0), (0, 5), (5, 6), (7, 11), (11, 14), (0, 0)]
        spans = [{"start": 7, "end": 14, "label": "NAME", "value": "JaneDoe"}]
        labels = align_offsets(offsets, spans, LABEL_CONFIG.label_to_id)
        names = [
            LABEL_CONFIG.labels[label] if label >= 0 else "IGNORE" for label in labels
        ]
        self.assertEqual(
            names,
            ["IGNORE", "O", "O", "B-NAME", "I-NAME", "IGNORE"],
        )


if __name__ == "__main__":
    unittest.main()
