from __future__ import annotations

import json
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from plva_proxy import benchmark


def test_controlled_suite_completes_and_plva_prevents_frame_egress() -> None:
    off, on = benchmark.run_controlled_suite()

    assert off.mode == "plva-off"
    assert off.task_completed
    assert off.private_frame_exposures == 1
    assert off.placeholder_actions_resolved == 0
    assert on.mode == "plva-on"
    assert on.task_completed
    assert on.private_frame_exposures == 0
    assert on.private_text_exposures == 0
    assert on.actions_executed == on.actions_attempted == 3
    assert on.placeholder_actions_resolved == 1
    assert on.verification.assertions == {
        "submitted": True,
        "destination_matches_source": True,
        "destination_nonempty": True,
    }


def test_verifier_rejects_runtime_success_without_postcondition() -> None:
    verification = benchmark.verify_form(benchmark.FormState(submitted=True))

    assert not verification.passed
    assert not verification.assertions["destination_matches_source"]


def test_executor_rejects_malformed_and_unknown_actions() -> None:
    state = benchmark.FormState()

    assert not benchmark._execute({}, state)
    malformed = {"choices": [{"message": {"content": "not-json"}}]}
    assert not benchmark._execute(malformed, state)
    unknown = {
        "choices": [{"message": {"content": json.dumps({"tool_call": {"tool_name": "scroll"}})}}]
    }
    assert not benchmark._execute(unknown, state)


def test_loopback_fixture_records_matching_submission() -> None:
    receipt = benchmark._Receipt()
    server = benchmark.ThreadingHTTPServer(("127.0.0.1", 0), benchmark._handler(receipt))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/") as response:
            assert benchmark.SYNTHETIC_VALUE.encode() in response.read()
            assert response.headers["Cache-Control"] == "no-store"
        with urllib.request.urlopen(base + "/confirm") as response:
            confirmation = response.read()
            assert b"Confirm email" in confirmation
            assert benchmark.SYNTHETIC_VALUE.encode() not in confirmation
        request = urllib.request.Request(
            base + "/submit",
            data=urllib.parse.urlencode({"confirm_email": benchmark.SYNTHETIC_VALUE}).encode(),
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            assert b"Accepted" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert receipt.submitted.is_set()
    assert receipt.visited_source.is_set()
    assert receipt.visited_confirm
    assert receipt.matches


def test_loopback_fixture_rejects_unknown_paths_and_wrong_submission() -> None:
    receipt = benchmark._Receipt()
    handler = benchmark._handler(receipt)
    assert handler is not benchmark._handler(receipt)
    server = benchmark.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(base + "/missing")
        assert missing.value.code == 404
        request = urllib.request.Request(
            base + "/submit", data=b"confirm_email=wrong", method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as invalid:
            urllib.request.urlopen(request)
        assert invalid.value.code == 422
        request = urllib.request.Request(base + "/other", data=b"", method="POST")
        with pytest.raises(urllib.error.HTTPError) as missing_post:
            urllib.request.urlopen(request)
        assert missing_post.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert receipt.submitted.is_set()
    assert not receipt.matches


def test_run_live_uses_postcondition_not_process_exit(monkeypatch: Any, tmp_path: Path) -> None:
    runner = tmp_path / "runner.sh"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        if command[0] == "/usr/bin/open":
            with urllib.request.urlopen(command[1]) as response:
                response.read()
            return type("Completed", (), {"returncode": 0})()
        assert command[0] == str(runner)
        assert "already open" in command[1]
        assert kwargs["env"]["PLVA_PRIVACY"] == "1"
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    result = benchmark.run_live(privacy_on=True, runner=runner, timeout_s=1)

    assert result.runtime_reported_success
    assert not result.task_completed
    assert not result.verification.assertions["submitted"]
    assert result.live


def test_run_live_captures_launcher_error(monkeypatch: Any, tmp_path: Path) -> None:
    runner = tmp_path / "runner.sh"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError("synthetic")

    monkeypatch.setattr(benchmark.subprocess, "run", fail)
    result = benchmark.run_live(privacy_on=False, runner=runner, timeout_s=1)

    assert result.error == "OSError"
    assert not result.runtime_reported_success


def test_main_writes_machine_readable_result(tmp_path: Path) -> None:
    output = tmp_path / "result.json"

    assert benchmark.main(["--mode", "on", "--output", str(output)]) == 0
    document = json.loads(output.read_text())
    assert document["schema_version"] == 1
    assert document["results"][0]["mode"] == "plva-on"


@pytest.mark.parametrize(
    "args,message",
    [
        (["--live"], "requires --allow-provider-spend"),
        (["--timeout-s", "0"], "must be positive"),
        (
            ["--live", "--allow-provider-spend", "--runner", "/not/a/runner"],
            "runner not found",
        ),
    ],
)
def test_main_fail_closed_gates(args: list[str], message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        benchmark.main(args)


def test_main_live_mode_can_be_explicitly_selected(monkeypatch: Any, tmp_path: Path) -> None:
    runner = tmp_path / "runner.sh"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    result = benchmark.run_controlled(privacy_on=True)
    calls: list[dict[str, Any]] = []

    def fake_live(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return result

    monkeypatch.setattr(benchmark, "run_live", fake_live)

    assert benchmark.main(["--live", "--allow-provider-spend", "--runner", str(runner)]) == 0
    assert len(calls) == 1
    assert calls[0]["privacy_on"] is True
