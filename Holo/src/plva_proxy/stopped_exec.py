"""Stop before exec so a parent can install process-group safeguards first."""

from __future__ import annotations

import os
import signal
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m plva_proxy.stopped_exec COMMAND [ARG ...]")
    os.kill(os.getpid(), signal.SIGSTOP)
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ.copy())


if __name__ == "__main__":  # pragma: no cover
    main()
