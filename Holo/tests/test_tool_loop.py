"""Step 6.5: relay tool-loop integration and grammar capture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plva_proxy.proxy import HookError, grammar_capture_hook


def test_grammar_capture_writes_schema_only(tmp_path: Path) -> None:
    out = tmp_path / "grammar.json"
    hook = grammar_capture_hook(out)
    document = {
        "model": "Hcompany/Holo3-35B-A3B",
        "messages": [{"role": "user", "content": "SECRET TEXT"}],
        "structured_outputs": {"json_schema": {"properties": {"tool_name": {"enum": ["click"]}}}},
        "chat_template_kwargs": {"reasoning": True},
    }
    returned, headers = hook(document, {"x-h": "1"})
    assert returned == document
    assert headers == {"x-h": "1"}
    snapshot = json.loads(out.read_text())
    assert snapshot["model"] == "Hcompany/Holo3-35B-A3B"
    assert snapshot["structured_outputs"]["json_schema"]["properties"]["tool_name"]["enum"] == [
        "click"
    ]
    assert "messages" not in snapshot
    assert "SECRET TEXT" not in out.read_text()
    assert sorted(snapshot["request_keys"]) == snapshot["request_keys"]


def test_grammar_capture_only_first_request(tmp_path: Path) -> None:
    out = tmp_path / "grammar.json"
    hook = grammar_capture_hook(out)
    hook({"model": "first", "structured_outputs": {}}, {})
    hook({"model": "second", "structured_outputs": {}}, {})
    assert json.loads(out.read_text())["model"] == "first"


def test_grammar_capture_write_failure_raises_hook_error(tmp_path: Path) -> None:
    hook = grammar_capture_hook(tmp_path)  # a directory: write_text raises OSError
    with pytest.raises(HookError):
        hook({"model": "m", "structured_outputs": {}}, {})
