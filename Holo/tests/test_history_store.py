"""Local run-history store: persistence, retrieval, and demo endpoints."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx

import plva_proxy.demo as demo

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def call_record(call_id: int) -> dict[str, Any]:
    return {
        "id": call_id,
        "at": 1783275000 + call_id,
        "duration_ms": 900,
        "status": 200,
        "state": "sent",
        "model": "Hcompany/Holo3-35B-A3B",
        "messages": 2,
        "images": ["image/png"],
        "preview": "click EMAIL_1_ab12",
        "request": {"messages": [{"role": "user", "content": "click EMAIL_1_ab12"}]},
        "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    }


def seeded_store(root: Path) -> tuple[demo.HistoryStore, str]:
    store = demo.HistoryStore(root)
    run_id = store.create_run("synthetic task", {"provider": "hcompany"}, {"EMAIL": "hide_use"})
    assert run_id is not None
    store.save_call(run_id, call_record(1), [("image/png", PNG_BYTES)])
    store.save_call(run_id, call_record(2), [])
    store.update_run(run_id, status="completed", finished_at=1783275100.0)
    return store, run_id


def test_history_store_persists_and_reads_back_runs(tmp_path: Path) -> None:
    store, run_id = seeded_store(tmp_path)

    (summary,) = store.runs()
    assert summary["id"] == run_id and summary["prompt"] == "synthetic task"
    assert summary["status"] == "completed" and summary["calls"] == 2

    detail = store.run(run_id)
    assert detail is not None
    assert detail["run"]["policy"] == {"EMAIL": "hide_use"}
    assert [c["id"] for c in detail["calls"]] == [1, 2]
    assert all("request" not in c and "response" not in c for c in detail["calls"])

    full = store.call(run_id, 1)
    assert full is not None and full["response"]["choices"][0]["message"]["content"] == "ok"
    assert store.image(run_id, 1, 0) == ("image/png", PNG_BYTES)
    assert store.image(run_id, 2, 0) is None

    # A fresh store over the same directory sees the same history (survives restarts).
    reread = demo.HistoryStore(tmp_path)
    assert [r["id"] for r in reread.runs()] == [run_id]


def test_history_store_rejects_invalid_ids_and_missing_data(tmp_path: Path) -> None:
    store = demo.HistoryStore(tmp_path)
    assert store.runs() == []
    assert store.run("../../etc") is None
    assert store.run("run-20260712-113405-ab12") is None
    assert store.call("not-a-run-id", 1) is None
    assert store.image("run-20260712-113405-ab12", 0, 0) is None
    # save/update against an unknown or None run must be a no-op, never a crash
    store.update_run(None, status="completed")
    store.save_call(None, call_record(1), [])
    store.save_call("run-20260712-113405-ab12", {"id": "bad"}, [])


async def test_demo_serves_history_endpoints(tmp_path: Path) -> None:
    controller = demo.DemoController(history_root=tmp_path)
    _, run_id = seeded_store(tmp_path)

    app = demo.create_demo_app(controller)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://demo.test"
    ) as client:
        runs = await client.get("/api/history/runs")
        run = await client.get(f"/api/history/runs/{run_id}")
        call = await client.get(f"/api/history/runs/{run_id}/calls/1")
        image = await client.get(f"/api/history/runs/{run_id}/calls/1/image/0")
        bad_run = await client.get("/api/history/runs/run-20990101-000000-ffff")
        bad_call = await client.get(f"/api/history/runs/{run_id}/calls/9")

    assert [r["id"] for r in runs.json()["runs"]] == [run_id]
    assert run.json()["run"]["prompt"] == "synthetic task"
    assert call.json()["preview"] == "click EMAIL_1_ab12"
    assert image.status_code == 200 and image.content == PNG_BYTES
    assert bad_run.status_code == 404 and bad_call.status_code == 404


def test_controller_records_run_lifecycle_into_history(tmp_path: Path) -> None:
    controller = demo.DemoController(history_root=tmp_path)
    store = demo.HistoryStore(tmp_path)
    run_id = store.create_run("seed", {}, {})
    controller._run_id = run_id  # noqa: SLF001 - lifecycle seam
    controller._event("Provider connected", "The selected Holo model is available.")  # noqa: SLF001

    detail = store.run(run_id)
    assert detail is not None
    assert detail["run"]["events"][-1]["title"] == "Provider connected"
