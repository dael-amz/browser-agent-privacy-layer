#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PLVA_DEMO_PORT:-18080}"
if [[ ! -x .venv/bin/plva-demo ]]; then
  echo "ERROR: install the project environment first" >&2
  exit 1
fi

(
  for _ in $(seq 1 80); do
    if curl -sf "http://127.0.0.1:$PORT/api/state" >/dev/null 2>&1; then
      open "http://127.0.0.1:$PORT/"
      exit 0
    fi
    sleep 0.1
  done
) &

exec .venv/bin/plva-demo --port "$PORT"
