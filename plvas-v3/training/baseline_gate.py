from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RAMPART_REVISION = "b1993e4e68b082835b80ffc65acc03325ea2e501"


def require_frozen_rampart_baseline(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise RuntimeError(
            "replacement training requires --baseline-manifest from a frozen "
            "Rampart-backed screen/OCR evaluation"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("bootstrap_model", {}).get("revision") != RAMPART_REVISION:
        raise RuntimeError(
            "baseline manifest does not pin the required Rampart revision"
        )
    if document.get("baseline_frozen") is not True:
        raise RuntimeError("Rampart baseline is contract-verified but not frozen")
    if document.get("evaluation", {}).get("all_required_gates_passed") is not True:
        raise RuntimeError(
            "Rampart baseline evaluation did not pass its required gates"
        )
    return document
