from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import pytest

import plva_proxy.egress_preflight as preflight


def test_preflight_reports_missing_role_and_uninspectable_pf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    anchor = tmp_path / "pf.anchor"
    anchor.write_text("pass out\n")

    def missing_user(_: str) -> None:
        raise KeyError

    monkeypatch.setattr(preflight.pwd, "getpwnam", missing_user)
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 1, "", "pfctl: Permission denied"),
    )

    status = preflight.packet_filter_status(anchor)

    assert status["ready"] is False
    assert status["role_user"] == {"name": "_plvaproxy", "exists": False}
    assert status["packet_filter"]["inspectable_without_elevation"] is False
    assert status["anchor"]["status"] == "parse-deferred-role-user-missing"
    assert status["anchor"]["loaded"] is None
    assert status["actions"]


def test_preflight_reports_ready_when_user_pf_and_anchor_are_confirmed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    anchor = tmp_path / "pf.anchor"
    anchor.write_text("pass out\n")
    monkeypatch.setattr(preflight.pwd, "getpwnam", lambda _: object())

    def successful_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        if command[-2:] == ["-s", "info"]:
            output = "Status: Enabled"
        elif command[-2:] == ["plva", "-sr"]:
            output = (
                "block return out proto tcp to <inference_providers>\n"
                "pass out proto tcp to <inference_providers>\n"
            )
        else:
            output = ""
        return CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(preflight.subprocess, "run", successful_run)

    status = preflight.packet_filter_status(anchor)

    assert status["ready"] is True
    assert status["anchor"]["status"] == "valid"
    assert status["anchor"]["loaded"] is True
    assert status["actions"] == []
