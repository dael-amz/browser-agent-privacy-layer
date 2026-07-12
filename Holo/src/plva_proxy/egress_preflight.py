"""Non-privileged status report for the optional macOS packet-filter boundary."""

from __future__ import annotations

import argparse
import json
import pwd
import subprocess
from pathlib import Path
from typing import Any, Final

ROLE_USER: Final = "_plvaproxy"


def packet_filter_status(anchor: Path) -> dict[str, Any]:
    """Return actionable status without requesting elevation or changing host state."""

    try:
        pwd.getpwnam(ROLE_USER)
        role_exists = True
    except KeyError:
        role_exists = False

    inspection = subprocess.run(
        ["/sbin/pfctl", "-s", "info"],
        check=False,
        capture_output=True,
        text=True,
    )
    inspection_text = f"{inspection.stdout}\n{inspection.stderr}".lower()
    inspectable = inspection.returncode == 0
    enabled = inspectable and "status: enabled" in inspection_text

    loaded: bool | None = None
    if not anchor.is_file():
        anchor_status = "missing"
    elif not role_exists:
        anchor_status = "parse-deferred-role-user-missing"
    else:
        syntax = subprocess.run(
            ["/sbin/pfctl", "-nf", str(anchor)],
            check=False,
            capture_output=True,
            text=True,
        )
        anchor_status = "valid" if syntax.returncode == 0 else "invalid-or-not-inspectable"
        if inspectable and anchor_status == "valid":
            loaded_rules = subprocess.run(
                ["/sbin/pfctl", "-a", "plva", "-sr"],
                check=False,
                capture_output=True,
                text=True,
            )
            loaded = loaded_rules.returncode == 0 and all(
                marker in loaded_rules.stdout
                for marker in ("block", "pass", "<inference_providers>")
            )

    ready = role_exists and inspectable and enabled and anchor_status == "valid" and loaded is True
    actions: list[str] = []
    if not role_exists:
        actions.append("create the _plvaproxy role user using docs/egress/bootstrap-pf.md")
    if not inspectable:
        actions.append("inspect and load the plva anchor from an administrator shell")
    if anchor_status != "valid":
        actions.append("validate docs/egress/pf-plva.anchor after the role user exists")
    elif loaded is not True:
        actions.append("load and inspect the plva anchor from an administrator shell")
    return {
        "version": 1,
        "ready": ready,
        "role_user": {"name": ROLE_USER, "exists": role_exists},
        "packet_filter": {
            "inspectable_without_elevation": inspectable,
            "enabled": enabled if inspectable else None,
        },
        "anchor": {"path": str(anchor), "status": anchor_status, "loaded": loaded},
        "actions": actions,
        "note": "No privileged command was executed by this preflight.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report PLVA packet-filter readiness")
    parser.add_argument("--anchor", type=Path, default=Path("docs/egress/pf-plva.anchor"))
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    status = packet_filter_status(args.anchor)
    print(json.dumps(status, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0 if status["ready"] or not args.require_ready else 2)


if __name__ == "__main__":  # pragma: no cover
    main()
