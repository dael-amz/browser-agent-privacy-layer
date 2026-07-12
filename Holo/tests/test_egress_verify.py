from __future__ import annotations

import json
import stat
from pathlib import Path
from subprocess import CompletedProcess

import pytest

import plva_proxy.egress_verify as egress
from plva_proxy.egress_verify import (
    Connection,
    VerificationError,
    _write_json,
    parse_lsof_fields,
    validate_connections,
)


def test_parse_lsof_fields_extracts_only_connected_socket_metadata() -> None:
    output = "\n".join(
        (
            "p123",
            "chai-agent-runt",
            "n127.0.0.1:54111->127.0.0.1:18081",
            "n127.0.0.1:54112",
            "p124",
            "cholo",
            "n[::1]:54113->[::1]:18081",
        )
    )

    assert parse_lsof_fields(output) == (
        Connection(123, "hai-agent-runt", "127.0.0.1:54111", "127.0.0.1:18081"),
        Connection(124, "holo", "[::1]:54113", "[::1]:18081"),
    )


def test_validate_connections_allows_local_control_ports_and_rejects_external_remote() -> None:
    connections = (
        Connection(1, "holo", "127.0.0.1:5000", "127.0.0.1:18081"),
        Connection(1, "holo", "127.0.0.1:5001", "127.0.0.1:9000"),
        Connection(1, "holo", "192.0.2.10:5002", "203.0.113.8:443"),
    )

    assert validate_connections(connections, allowed_port=18081) == connections[2:]


def test_parse_lsof_fields_fails_closed_on_hostnames() -> None:
    with pytest.raises(VerificationError, match="non-numeric"):
        parse_lsof_fields("p1\ncholo\nnhost.example:5000->provider.example:443\n")


def test_write_json_is_atomic_private_and_contains_only_payload(tmp_path: Path) -> None:
    destination = tmp_path / "status.json"

    _write_json(destination, {"verdict": "passed", "sample_count": 3})

    assert json.loads(destination.read_text()) == {"sample_count": 3, "verdict": "passed"}
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_detached_runtime_discovery_uses_exact_executable_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = "\n".join(
        (
            "  41 /Applications/Holo/hai-agent-runtime",
            "  42 /tmp/hai-agent-runtime --not-the-runtime-shape",
            "  43 python monitor.py",
        )
    )
    monkeypatch.setattr(
        egress.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, output, ""),
    )

    assert egress._detached_runtime_pids() == (41,)
