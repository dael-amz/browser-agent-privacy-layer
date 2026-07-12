"""Shared test guards."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_history_dir(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    """Point run-history persistence at a throwaway directory.

    A test that constructs a DemoController without an explicit history root
    must never write into the repository's real ``history/`` audit store.
    """

    monkeypatch.setenv("PLVA_HISTORY_DIR", str(tmp_path_factory.mktemp("plva-history")))
