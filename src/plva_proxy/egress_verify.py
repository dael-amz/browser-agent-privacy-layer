"""Fail-closed, non-privileged egress monitor for the closed CUA runtime.

The launcher starts the runtime stopped, starts this monitor for the runtime's
process group, and only resumes the runtime after ``lsof`` visibility has been
proved with a local canary connection.  Evidence contains process and socket
metadata only; command lines, task text, request bodies, and frame bytes are
never recorded.
"""

from __future__ import annotations

import argparse
import contextlib
import ipaddress
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

LOOPBACK_HOST: Final = "127.0.0.1"


class VerificationError(RuntimeError):
    """Raised when loopback-only egress cannot be established."""


@dataclass(frozen=True, slots=True)
class Connection:
    """Privacy-safe metadata for one connected TCP socket."""

    pid: int
    process: str
    local: str
    remote: str


def _endpoint_host_port(endpoint: str) -> tuple[str, int]:
    """Split the numeric endpoint format emitted by ``lsof -nP``."""

    value = endpoint.strip()
    if value.startswith("["):
        host, separator, port_text = value[1:].partition("]:")
    else:
        host, separator, port_text = value.rpartition(":")
    if not separator:
        raise VerificationError("lsof emitted an endpoint without a numeric port")
    try:
        port = int(port_text)
        ipaddress.ip_address(host.removesuffix("%lo0"))
    except ValueError as exc:
        raise VerificationError("lsof emitted a non-numeric network endpoint") from exc
    return host, port


def parse_lsof_fields(output: str) -> tuple[Connection, ...]:
    """Parse ``lsof -Fpcn`` output without retaining command-line text."""

    pid: int | None = None
    process = ""
    connections: list[Connection] = []
    for line in output.splitlines():
        if not line:
            continue
        field, value = line[0], line[1:]
        if field == "p":
            try:
                pid = int(value)
            except ValueError as exc:
                raise VerificationError("lsof emitted an invalid pid") from exc
            process = ""
        elif field == "c":
            # lsof's command name is bounded and cannot contain the task argument.
            process = value
        elif field == "n" and "->" in value:
            if pid is None:
                raise VerificationError("lsof emitted a socket without a pid")
            local, remote = value.split("->", 1)
            _endpoint_host_port(local)
            _endpoint_host_port(remote)
            connections.append(Connection(pid, process, local, remote))
    return tuple(connections)


def validate_connections(
    connections: tuple[Connection, ...], *, allowed_port: int
) -> tuple[Connection, ...]:
    """Return connections that leave loopback.

    The Holo CLI legitimately talks to its local Agent API on a dynamic/fixed
    loopback port while the runtime talks to the selected PLVA proxy. Blocking
    every loopback port except the proxy kills that control channel and can
    orphan the detached runtime. The privacy boundary is therefore no external
    network egress from the complete CUA process group; ``allowed_port`` is
    retained in evidence as the configured model endpoint.
    """

    violations: list[Connection] = []
    for connection in connections:
        host, _ = _endpoint_host_port(connection.remote)
        try:
            is_loopback = ipaddress.ip_address(host.removesuffix("%lo0")).is_loopback
        except ValueError as exc:  # pragma: no cover - parse_lsof_fields validates first
            raise VerificationError("lsof emitted a non-numeric remote address") from exc
        if not is_loopback:
            violations.append(connection)
    return tuple(violations)


def _group_pids(pgid: int) -> tuple[int, ...]:
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,pgid="],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise VerificationError("ps could not enumerate the runtime process group")
    members: list[int] = []
    try:
        for line in result.stdout.splitlines():
            pid_text, group_text = line.split()
            if int(group_text) == pgid:
                members.append(int(pid_text))
    except (ValueError, IndexError) as exc:
        raise VerificationError("ps emitted an unexpected process table format") from exc
    return tuple(members)


def _detached_runtime_pids() -> tuple[int, ...]:
    """Find Holo runtimes that daemonized out of the launcher's process group."""

    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,command="],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise VerificationError("ps could not enumerate detached Holo runtimes")
    matches: list[int] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        pid_text, separator, command = stripped.partition(" ")
        if not separator:
            continue
        if command.rstrip().endswith("/hai-agent-runtime"):
            try:
                matches.append(int(pid_text))
            except ValueError as exc:
                raise VerificationError("ps emitted an invalid runtime pid") from exc
    return tuple(matches)


def _terminate(pgid: int, runtime_pids: tuple[int, ...]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    for pid in runtime_pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)


def _lsof_connections(pids: tuple[int, ...]) -> tuple[Connection, ...]:
    if not pids:
        return ()
    result = subprocess.run(
        [
            "/usr/sbin/lsof",
            "-nP",
            "-a",
            "-p",
            ",".join(str(pid) for pid in pids),
            "-iTCP",
            "-Fpcn",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    # lsof returns 1 when the selected processes have no matching sockets.
    if result.returncode not in {0, 1} or result.stderr.strip():
        raise VerificationError("lsof could not inspect the runtime process group")
    return parse_lsof_fields(result.stdout)


def prove_lsof_visibility() -> None:
    """Prove that this user can observe an established socket before monitoring."""

    with socket.socket() as listener:
        listener.bind((LOOPBACK_HOST, 0))
        listener.listen(1)
        with socket.create_connection(listener.getsockname()) as client:
            accepted, _ = listener.accept()
            with accepted:
                connections = _lsof_connections((os.getpid(),))
                client_port = client.getsockname()[1]
                if not any(
                    _endpoint_host_port(connection.local)[1] == client_port
                    for connection in connections
                ):
                    raise VerificationError("lsof did not expose the local canary connection")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        Path(temporary).unlink(missing_ok=True)
        raise


def monitor(
    *, pgid: int, allowed_port: int, ready_file: Path, status_file: Path, interval_ms: int
) -> int:
    """Monitor a process group until it exits or violates the egress invariant."""

    started = time.time()
    sample_count = 0
    observed: set[Connection] = set()
    violation: tuple[Connection, ...] = ()
    error: str | None = None
    verdict = "error"
    exit_code = 2
    try:
        prove_lsof_visibility()
        _write_json(
            ready_file,
            {"ready": True, "monitor_pid": os.getpid(), "pgid": pgid, "version": 1},
        )
        while True:
            group_pids = _group_pids(pgid)
            runtime_pids = _detached_runtime_pids()
            pids = tuple(dict.fromkeys((*group_pids, *runtime_pids)))
            if not pids:
                verdict = "passed"
                exit_code = 0
                break
            connections = _lsof_connections(pids)
            sample_count += 1
            observed.update(connections)
            violation = validate_connections(connections, allowed_port=allowed_port)
            if violation:
                verdict = "failed"
                exit_code = 3
                _terminate(pgid, runtime_pids)
                break
            time.sleep(interval_ms / 1000)
    except (OSError, VerificationError) as exc:
        error = str(exc)
        remaining_runtime_pids: tuple[int, ...] = ()
        with contextlib.suppress(OSError, VerificationError):
            remaining_runtime_pids = _detached_runtime_pids()
        _terminate(pgid, remaining_runtime_pids)
    payload: dict[str, object] = {
        "version": 1,
        "verdict": verdict,
        "pgid": pgid,
        "allowed_remote": "loopback-only",
        "configured_proxy": f"127.0.0.1:{allowed_port}",
        "sample_count": sample_count,
        "observed_connections": [asdict(item) for item in sorted(observed, key=repr)],
        "violations": [asdict(item) for item in violation],
        "error": error,
        "duration_ms": round((time.time() - started) * 1000),
    }
    try:
        _write_json(status_file, payload)
    except OSError:
        return 2
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor one CUA process group for local-only TCP")
    parser.add_argument("--pgid", type=int, required=True)
    parser.add_argument("--allowed-port", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=100)
    args = parser.parse_args()
    if args.pgid < 1 or not 1 <= args.allowed_port <= 65535 or args.interval_ms < 10:
        parser.error("pgid, port, or interval is out of range")
    raise SystemExit(
        monitor(
            pgid=args.pgid,
            allowed_port=args.allowed_port,
            ready_file=args.ready_file,
            status_file=args.status_file,
            interval_ms=args.interval_ms,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
