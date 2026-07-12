#!/usr/bin/env bash
# One-command Step 1 run: proxy + key preflight + holo task + egress observation + cleanup.
#
#   ./run_step1.sh                          # default terminal task
#   ./run_step1.sh "your own task prompt"   # custom task for the agent
#   Press Esc twice during the run to abort it.
#
# Prereq: Holo/.env containing   API_KEY=<your Overshoot key>
# Details and manual variant: verification/step-1-runbook.md
#
# NOTE: this streams UNREDACTED screenshots of the visible desktop to Overshoot
# (redaction arrives in Step 4). Close anything sensitive first. The runtime
# kill switch stays enabled; Esc Esc is an additional local abort.
set -euo pipefail
cd "$(dirname "$0")"

DEFAULT_TASK="Open the Terminal application and run the command: echo plva-step1-ok"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "usage: $0 [\"task prompt for the agent\"]"
  echo "  default task: $DEFAULT_TASK"
  echo "  during the run: press Esc twice to abort"
  exit 0
fi

PORT="${PLVA_PORT:-18081}"
UV="${UV:-$HOME/.local/bin/uv}"
TASK="${1:-$DEFAULT_TASK}"

if [[ ! -f .env ]] || ! grep -Eq '^API_KEY=..*' .env; then
  echo "ERROR: put your Overshoot key in Holo/.env as a single line: API_KEY=<key>" >&2
  exit 1
fi

# 1) Start the loopback proxy (the sole provider egress; reads ./.env).
#    PLVA_HOOK=test enables the Step 3 test hooks for the hook-mode verify.
#    PLVA_HOOK_IMAGE=/path/to.png replaces every outbound screenshot with that
#    static image (no real desktop pixels egress; fails closed if no frame).
#    PLVA_REDACT=1 redacts every outbound screenshot through plva-v2-baseline
#    and serves the obscured frames at http://127.0.0.1:$PORT/viewer.
PROXY_LOG=/tmp/plva-proxy-step1.log
HOOK_ARGS=(--hook "${PLVA_HOOK:-none}")
if [[ -n "${PLVA_HOOK_IMAGE:-}" ]]; then
  if [[ ! -f "$PLVA_HOOK_IMAGE" ]]; then
    echo "ERROR: PLVA_HOOK_IMAGE file not found: $PLVA_HOOK_IMAGE (relative paths resolve against $(pwd))" >&2
    exit 1
  fi
  HOOK_ARGS+=(--hook-image "$PLVA_HOOK_IMAGE")
fi
if [[ -n "${PLVA_REDACT:-}" ]]; then
  HOOK_ARGS+=(--redact plva-v2-baseline)
  echo "--- redaction ON; watch the obscured frames at http://127.0.0.1:$PORT/viewer"
fi
.venv/bin/plva-proxy --port "$PORT" "${HOOK_ARGS[@]}" >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!
OBS_FILE=$(mktemp /tmp/plva-step1-egress.XXXXXX)
RUNS_DIR=""
cleanup() {
  kill "$PROXY_PID" "${OBS_PID:-}" 2>/dev/null || true
  # Frame-bearing artifacts must never survive, even on an aborted run.
  [[ -n "$RUNS_DIR" ]] && rm -rf "$RUNS_DIR"
}
trap cleanup EXIT
PROXY_UP=""
for _ in $(seq 1 20); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    PROXY_UP=1
    break
  fi
  kill -0 "$PROXY_PID" 2>/dev/null || break
  sleep 0.25
done
if [[ -z "$PROXY_UP" ]]; then
  echo "ERROR: the proxy failed to start. Its log says:" >&2
  tail -3 "$PROXY_LOG" >&2
  exit 1
fi

# 2) Preflight: proves key + upstream reachability without sending any frame.
echo "--- preflight: listing models through the proxy"
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import json, sys
raw = sys.stdin.read()
try:
    ids = [m.get('id') for m in json.loads(raw).get('data', [])]
except ValueError:
    print('preflight got no valid JSON from the proxy')
    raise SystemExit(1)
ok = any('Holo3-35B-A3B' in str(i) for i in ids)
print('Holo3-35B-A3B advertised:', ok)
raise SystemExit(0 if ok else 1)
"; then
  echo "ERROR: preflight failed — wrong key (401) or provider unreachable. See $PROXY_LOG" >&2
  exit 1
fi

# 3) Observe egress while the run is live: the runtime must only talk to loopback.
(
  while :; do
    lsof -nP -iTCP -sTCP:ESTABLISHED 2>/dev/null | grep -iE 'hai[-_]?agent|holo' >>"$OBS_FILE" || true
    sleep 10
  done
) &
OBS_PID=$!

# 4) The Step 1 task. Frame-bearing run artifacts go to ephemeral /tmp only.
RUNS_DIR=$(mktemp -d /tmp/holo-step1-runs.XXXXXX)
echo "--- task: $TASK"
echo "--- press Esc twice to abort; runs dir (shredded afterward): $RUNS_DIR"
set -m  # own process group for the holo job so an abort kills the runtime too
"$UV" tool run --from holo-desktop-cli holo run "$TASK" \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model Hcompany/Holo3-35B-A3B \
  --max-steps 20 --max-time-s 300 \
  --runs-dir "$RUNS_DIR" &
HOLO_PID=$!
set +m

ABORTED=""
if [[ -r /dev/tty ]]; then
  last_esc=0
  while kill -0 "$HOLO_PID" 2>/dev/null; do
    key=""
    IFS= read -rsn1 -t 1 key </dev/tty 2>/dev/null || true
    if [[ "$key" == $'\e' ]]; then
      now=$(date +%s)
      if (( now - last_esc <= 2 )); then
        echo
        echo "--- Esc Esc: aborting the run"
        kill -- -"$HOLO_PID" 2>/dev/null || kill "$HOLO_PID" 2>/dev/null || true
        pkill -f 'hai-agent-runtime' 2>/dev/null || true
        ABORTED=1
        break
      fi
      last_esc=$now
    elif [[ -n "$key" ]]; then
      last_esc=0
    fi
  done
fi

HOLO_EXIT=0
wait "$HOLO_PID" 2>/dev/null || HOLO_EXIT=$?
[[ -n "$ABORTED" ]] && HOLO_EXIT=130
kill "$OBS_PID" 2>/dev/null || true

# 5) Shred frame-bearing artifacts immediately; report privacy-safe evidence.
rm -rf "$RUNS_DIR"
RUNS_DIR=""
echo "--- runs dir shredded"
echo "--- runtime egress observed during the run (expect only ->127.0.0.1:$PORT):"
sort -u "$OBS_FILE" 2>/dev/null || echo "(no runtime connections captured)"
rm -f "$OBS_FILE"
echo "--- proxy relay timings (privacy-safe; seeds Step 2 latency): $PROXY_LOG"
grep -o 'relay .*' "$PROXY_LOG" | tail -5 || true
if [[ -n "$ABORTED" ]]; then
  echo "--- run aborted by Esc Esc"
else
  echo "--- holo exit: $HOLO_EXIT (0 = Step 1 task completed end-to-end)"
fi
exit "$HOLO_EXIT"
