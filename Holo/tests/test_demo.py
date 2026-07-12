from __future__ import annotations

import io
import json
import signal
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from plva_proxy import demo


class DormantThread:
    def __init__(self, *, target: Any, args: tuple[Any, ...], daemon: bool) -> None:
        self.target = target
        self.args = args
        self.daemon = daemon

    def start(self) -> None:
        return


class FakeProcess:
    def __init__(self, lines: tuple[str, ...] = (), *, return_code: int | None = None) -> None:
        self.pid = 4242
        self.stdout = io.StringIO("".join(lines))
        self.return_code = return_code

    def poll(self) -> int | None:
        return self.return_code

    def wait(self) -> int:
        return self.return_code or 0


def test_controller_validates_and_updates_policy_and_settings() -> None:
    controller = demo.DemoController()

    policy = controller.set_policy({"EMAIL": "blocked", "PASSWORD": "hide_use"})
    settings = controller.set_settings(
        {
            "plva_enabled": False,
            "provider": "overshoot",
            "vision_mode": "fast",
            "lifecycle": "cold",
            "features": {name: False for name in demo.FEATURE_ENV},
        }
    )

    assert policy["EMAIL"] == "blocked"
    assert policy["PASSWORD"] == "hide_use"
    assert settings["plva_enabled"] is False
    assert settings["features"]["manifest"] is False
    with pytest.raises(ValueError, match="policy"):
        controller.set_policy([])
    with pytest.raises(ValueError, match="provider"):
        controller.set_settings(
            {
                **settings,
                "provider": "unknown",
            }
        )


def test_demo_loads_editable_policy_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_file.write_text('{"EMAIL":"blocked"}', encoding="utf-8")
    monkeypatch.setenv("PLVA_POLICY_FILE", str(policy_file))

    controller = demo.DemoController()

    assert controller.snapshot()["policy"]["EMAIL"] == "blocked"
    policy_file.write_text('{"EMAIL":"invalid"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid safety level"):
        demo.DemoController()


def test_controller_starts_with_memory_only_environment_and_can_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    process = FakeProcess()

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        captured["command"] = command
        captured.update(kwargs)
        return process

    killed: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(demo.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(demo.threading, "Thread", DormantThread)
    monkeypatch.setattr(demo.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    controller = demo.DemoController(history_root=tmp_path / "history")
    controller.set_policy({"EMAIL": "approval"})

    controller.start("Use a synthetic placeholder")

    assert captured["command"] == [
        str(demo.ROOT / "run_step1.sh"),
        "Use a synthetic placeholder",
    ]
    environment = captured["env"]
    assert environment["PLVA_REDACT"] == "1"
    assert environment["PLVA_PRIVACY"] == "1"
    assert json.loads(environment["PLVA_POLICY_JSON"])["EMAIL"] == "approval"
    assert controller.snapshot()["running"] is True
    with pytest.raises(RuntimeError, match="active task"):
        controller.set_policy({"EMAIL": "blocked"})
    controller.stop()
    assert killed == [(4242, signal.SIGTERM)]
    assert controller.snapshot()["status"] == "stopping"


@pytest.mark.parametrize(
    ("line", "title"),
    [
        ("--- redaction ON (vision)", "Privacy engine ready"),
        ("--- redaction OFF", "PLVA bypassed"),
        ("--- preflight: provider=hcompany", "Checking provider"),
        ("holo3 advertised: True", "Provider connected"),
        ("--- runs dir shredded", "Private artifacts cleared"),
        ("--- holo exit: 0", "Agent finished"),
        ("ERROR provider failed", "Runner reported an error"),
    ],
)
def test_runner_events_are_allowlisted(line: str, title: str) -> None:
    assert demo._safe_runner_event(line)[0] == title  # type: ignore[index]
    assert demo._safe_runner_event("user: private prompt") is None


def test_process_reader_records_only_safe_events() -> None:
    controller = demo.DemoController()
    process = FakeProcess(
        (
            "user: do not retain this prompt\n",
            "--- redaction ON (vision)\n",
            "holo3 advertised: True\n",
        ),
        return_code=0,
    )
    controller._process = process  # type: ignore[assignment]

    controller._read_process(process)  # type: ignore[arg-type]
    snapshot = controller.snapshot()

    serialized = json.dumps(snapshot)
    assert "do not retain" not in serialized
    assert snapshot["status"] == "completed"
    assert "Privacy engine ready" in serialized


def test_proxy_monitor_keeps_only_latest_memory_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = demo.DemoController()
    polls = iter((None, 0))

    class PollingProcess(FakeProcess):
        def poll(self) -> int | None:
            return next(polls)

    monkeypatch.setattr(
        demo, "_fetch_bytes", lambda path: b"png" if path.endswith("frame") else None
    )
    reports = {
        "/viewer/stats": {"frames_seen": 1},
        "/viewer/findings": {"findings": []},
        "/viewer/vault": {"entries": [{"placeholder": "EMAIL_1_test"}]},
        "/viewer/approvals": {"approvals": [{"token": "EMAIL_1_test"}]},
        "/viewer/filter": {"status": "passed", "texts_scanned": 2},
    }
    monkeypatch.setattr(demo, "_fetch_json", lambda path: reports.get(path))
    monkeypatch.setattr(demo.time, "sleep", lambda _: None)

    controller._monitor_proxy(PollingProcess())  # type: ignore[arg-type]

    assert controller.frame() == b"png"
    assert controller.vault()["entries"][0]["placeholder"] == "EMAIL_1_test"
    assert controller.approvals()[0]["token"] == "EMAIL_1_test"
    assert controller.filter_diagnostics()["status"] == "passed"


async def test_demo_api_serves_ui_controls_and_memory_viewers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = demo.DemoController()
    controller._frame = b"synthetic-png"
    controller._vault = {
        "entries": [
            {
                "placeholder": "EMAIL_1_test",
                "safety_level": "approval",
            }
        ],
        "policy": {},
    }
    controller._approvals = [{"token": "EMAIL_1_test", "remaining_uses": 1}]
    controller._findings = {"findings": [{"labels": ["EMAIL"]}]}
    controller._filter = {"status": "passed"}
    started: list[str] = []
    approved: list[str] = []
    monkeypatch.setattr(controller, "start", lambda prompt: started.append(prompt))
    monkeypatch.setattr(
        controller,
        "approve_once",
        lambda token: approved.append(token) or {"token": token, "remaining_uses": 1},
    )
    app = demo.create_demo_app(controller)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://demo.test"
    ) as client:
        landing = await client.get("/")
        page = await client.get("/app")
        state = await client.get("/api/state")
        frame = await client.get("/api/frame")
        vault = await client.get("/api/vault")
        approvals = await client.get("/api/approvals")
        approval = await client.post("/api/approve", json={"token": "EMAIL_1_test"})
        findings = await client.get("/api/findings")
        filter_report = await client.get("/api/filter")
        run = await client.post("/api/run", json={"prompt": "synthetic task"})
        invalid = await client.put("/api/policy", json={"EMAIL": "invalid"})

    assert landing.status_code == 200 and "Launch" in landing.text
    assert page.status_code == 200 and "What can I do for you?" in page.text
    assert state.json()["settings"]["plva_enabled"] is True
    assert frame.content == b"synthetic-png"
    assert vault.json()["entries"][0]["placeholder"] == "EMAIL_1_test"
    assert approvals.json()["approvals"][0]["token"] == "EMAIL_1_test"
    assert approval.status_code == 201 and approved == ["EMAIL_1_test"]
    assert findings.json()["findings"][0]["labels"] == ["EMAIL"]
    assert filter_report.json()["status"] == "passed"
    assert run.status_code == 202 and started == ["synthetic task"]
    assert invalid.status_code == 409


def test_controller_grants_exact_one_write_without_exposing_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = demo.DemoController()
    controller._vault = {
        "entries": [
            {
                "placeholder": "API_KEY_1_test",
                "class": "API_KEY",
                "safety_level": "approval",
                "value": "local-secret",
            }
        ],
        "policy": {},
    }
    requests: list[dict[str, Any]] = []

    def grant(path: str, *, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        requests.append({"path": path, "method": method, "payload": payload})
        return {"token": payload["token"], "remaining_uses": 1}

    monkeypatch.setattr(demo, "_proxy_json_request", grant)

    result = controller.approve_once("API_KEY_1_test")

    assert result == {"token": "API_KEY_1_test", "remaining_uses": 1}
    assert requests[0]["payload"] == {
        "token": "API_KEY_1_test",
        "tool_name": "write_desktop",
        "argument_path": "content",
        "ttl_seconds": 60,
        "use_count": 1,
    }
    assert "local-secret" not in json.dumps(requests)


def test_fetch_helpers_fail_closed_on_bad_data(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: Any) -> None:
            return

        def read(self) -> bytes:
            return b"not-json"

    monkeypatch.setattr(demo.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    assert demo._fetch_bytes("/frame") == b"not-json"
    assert demo._fetch_json("/state") is None


def test_demo_main_validates_port_and_starts_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(sys, "argv", ["plva-demo", "--port", "18100"])
    monkeypatch.setattr(demo.uvicorn, "run", lambda app, **kwargs: captured.update(kwargs))

    demo.main()

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 18100


def test_demo_ui_file_is_packaged() -> None:
    assert demo.UI_PATH.is_file()
    assert "PLVA protection" in demo.UI_PATH.read_text("utf-8")
    assert demo.LANDING_PATH.is_file()
    assert 'href="/app"' in demo.LANDING_PATH.read_text("utf-8")
    assert Path(demo.ROOT / "run_demo.sh").is_file()
