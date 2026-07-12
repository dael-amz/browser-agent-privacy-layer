from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def ensure_locked_training_asset(
    asset_name: str,
    destination: Path,
    lock_path: Path = Path("/root/models.lock.json"),
) -> dict[str, Any]:
    """Fetch one hash-pinned training asset without mutable auto-downloads."""

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    asset = lock["assets"][asset_name]

    def valid() -> bool:
        if not destination.is_file() or destination.stat().st_size != asset["bytes"]:
            return False
        return hashlib.sha256(destination.read_bytes()).hexdigest() == asset["sha256"]

    if valid():
        return asset | {"path": str(destination), "status": "verified-existing"}

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    received = 0
    try:
        request = Request(asset["url"], headers={"User-Agent": "plva-modal-fetch/1"})
        with urlopen(request, timeout=180) as response, temporary.open("xb") as output:
            final_host = urlparse(response.geturl()).hostname
            if final_host not in set(asset["allowed_redirect_hosts"]):
                raise RuntimeError(f"unexpected training asset host: {final_host}")
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > asset["bytes"]:
                    raise RuntimeError("training checkpoint exceeded its locked size")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if received != asset["bytes"]:
            raise RuntimeError(
                f"training checkpoint bytes {received} != {asset['bytes']}"
            )
        if digest.hexdigest() != asset["sha256"]:
            raise RuntimeError("training checkpoint SHA-256 does not match the lock")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return asset | {"path": str(destination), "status": "downloaded"}
