from __future__ import annotations

import io
import sys

from plva_coreml.worker import _library_output


def test_library_output_is_kept_off_worker_protocol(
    monkeypatch,
) -> None:
    protocol = io.StringIO()
    diagnostics = io.StringIO()
    monkeypatch.setattr(sys, "stdout", protocol)
    monkeypatch.setattr(sys, "stderr", diagnostics)

    with _library_output():
        print("third-party model banner")

    assert protocol.getvalue() == ""
    assert diagnostics.getvalue() == "third-party model banner\n"
