#!/usr/bin/env bash
# One-command Step 1 run: proxy + key preflight + holo task + egress observation + cleanup.
#
#   ./run_step1.sh                          # default terminal task
#   ./run_step1.sh "your own task prompt"   # custom task for the agent
#   Press Esc twice during the run to abort it.
#
# Prereq: Holo/.env containing API_KEY=<Overshoot key>, or HAI_API_KEY=<H Company key>
# Details and manual variant: verification/step-1-runbook.md
#
# NOTE: PLVA_REDACT=0 sends unredacted screenshots; PLVA_REDACT=1 fails closed unless every frame
# is redacted before provider egress. Close unrelated sensitive windows either way. The runtime
# kill switch stays enabled; Esc Esc is an additional local abort.
set -euo pipefail
cd "$(dirname "$0")"

DEFAULT_TASK="Open the Terminal application and run the command: echo plva-step1-ok"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "usage: $0 [\"task prompt for the agent\"]"
  echo "  default task: $DEFAULT_TASK"
  echo "  provider: PLVA_PROVIDER=overshoot (default) or hcompany"
  echo "  during the run: press Esc twice to abort"
  exit 0
fi

PORT="${PLVA_PORT:-18081}"
UV="${UV:-$HOME/.local/bin/uv}"
TASK="${1:-$DEFAULT_TASK}"
PROVIDER="${PLVA_PROVIDER:-overshoot}"
REDACT_ENGINE="${PLVA_REDACT_ENGINE:-accelerated}"

parse_on_off() {
  local destination="$1" value="$2" label="$3"
  case "$value" in
    1|true|TRUE|yes|YES|on|ON) printf -v "$destination" '%s' 1 ;;
    0|false|FALSE|no|NO|off|OFF|"") printf -v "$destination" '%s' 0 ;;
    *)
      echo "ERROR: $label must be an on/off value" >&2
      exit 1
      ;;
  esac
}

case "$PROVIDER" in
  overshoot)
    MODEL="${PLVA_MODEL:-Hcompany/Holo3-35B-A3B}"
    if [[ -z "${OVERSHOOT_API_KEY:-}${API_KEY:-}" ]] && { [[ ! -f .env ]] || ! grep -Eq '^(OVERSHOOT_API_KEY|API_KEY)=..*' .env; }; then
      echo "ERROR: put API_KEY=<Overshoot key> in Holo/.env" >&2
      exit 1
    fi
    ;;
  hcompany)
    MODEL="${PLVA_MODEL:-holo3-1-35b-a3b}"
    if [[ -z "${HAI_API_KEY:-}" ]] && { [[ ! -f .env ]] || ! grep -Eq '^HAI_API_KEY=..*' .env; }; then
      echo "ERROR: put HAI_API_KEY=<H Company key> in Holo/.env" >&2
      exit 1
    fi
    ;;
  *)
    echo "ERROR: PLVA_PROVIDER must be overshoot or hcompany" >&2
    exit 1
    ;;
esac

# 1) Start the loopback proxy (the sole provider egress; reads ./.env).
#    PLVA_HOOK=test enables the Step 3 test hooks for the hook-mode verify.
#    PLVA_HOOK_IMAGE=/path/to.png replaces every outbound screenshot with that
#    static image (no real desktop pixels egress; fails closed if no frame).
#    PLVA_REDACT=1 redacts every outbound screenshot through plva-v2-baseline
#    and serves the obscured frames at http://127.0.0.1:$PORT/viewer.
PROXY_LOG=/tmp/plva-proxy-step1.log
HOOK_ARGS=(--provider "$PROVIDER" --hook "${PLVA_HOOK:-none}")
if [[ -n "${PLVA_UPSTREAM:-}" ]]; then
  HOOK_ARGS+=(--upstream "$PLVA_UPSTREAM")
fi
if [[ -n "${PLVA_HOOK_IMAGE:-}" ]]; then
  if [[ ! -f "$PLVA_HOOK_IMAGE" ]]; then
    echo "ERROR: PLVA_HOOK_IMAGE file not found: $PLVA_HOOK_IMAGE (relative paths resolve against $(pwd))" >&2
    exit 1
  fi
  HOOK_ARGS+=(--hook-image "$PLVA_HOOK_IMAGE")
fi
case "${PLVA_REDACT:-0}" in
  1|true|TRUE|yes|YES|on|ON) REDACTION_ENABLED=1 ;;
  0|false|FALSE|no|NO|off|OFF|"") REDACTION_ENABLED=0 ;;
  *)
    echo "ERROR: PLVA_REDACT must be an on/off value (for example 1 or 0)" >&2
    exit 1
    ;;
esac
PRIVACY_ENABLED=0
if [[ "$REDACTION_ENABLED" == 1 ]]; then
  PRIVACY_DEFAULT=0
  [[ "$REDACT_ENGINE" == "vision" ]] && PRIVACY_DEFAULT=1
  case "${PLVA_PRIVACY:-$PRIVACY_DEFAULT}" in
    1|true|TRUE|yes|YES|on|ON) PRIVACY_ENABLED=1 ;;
    0|false|FALSE|no|NO|off|OFF|"") PRIVACY_ENABLED=0 ;;
    *)
      echo "ERROR: PLVA_PRIVACY must be an on/off value" >&2
      exit 1
      ;;
  esac
  parse_on_off PRIVACY_HISTORY_SCRUB "${PLVA_PRIVACY_HISTORY_SCRUB:-1}" PLVA_PRIVACY_HISTORY_SCRUB
  parse_on_off PRIVACY_CHIPS "${PLVA_PRIVACY_CHIPS:-1}" PLVA_PRIVACY_CHIPS
  parse_on_off PRIVACY_SCHEME "${PLVA_PRIVACY_SCHEME:-1}" PLVA_PRIVACY_SCHEME
  parse_on_off PRIVACY_DUPLICATE_WARNING "${PLVA_PRIVACY_DUPLICATE_WARNING:-1}" PLVA_PRIVACY_DUPLICATE_WARNING
  parse_on_off PRIVACY_MANIFEST "${PLVA_PRIVACY_MANIFEST:-1}" PLVA_PRIVACY_MANIFEST
  parse_on_off PRIVACY_RESOLUTION "${PLVA_PRIVACY_RESOLUTION:-1}" PLVA_PRIVACY_RESOLUTION
  parse_on_off PRIVACY_POLICY_TEACHING "${PLVA_PRIVACY_POLICY_TEACHING:-1}" PLVA_PRIVACY_POLICY_TEACHING
  HOOK_ARGS+=(
    --redact plva-v2-baseline
    --redact-engine "$REDACT_ENGINE"
    --redact-backend "${PLVA_REDACT_BACKEND:-auto}"
    --vision-worker "${PLVA_VISION_WORKER:-coreml-redactor}"
    --vision-mode "${PLVA_VISION_MODE:-cascade}"
    --redact-lifecycle "${PLVA_REDACT_LIFECYCLE:-adaptive}"
    --redact-idle-seconds "${PLVA_REDACT_IDLE_SECONDS:-60}"
  )
  if [[ "$REDACT_ENGINE" == "vision" ]]; then
    VISUAL_MODEL="${PLVA_VISUAL_MODEL:-plvas-v3/harness/plva-v2-baseline/runtime/training/artifacts/plva-visual-agpl-test-v2/visual/detector.onnx}"
    if [[ ! -f "$VISUAL_MODEL" ]]; then
      echo "ERROR: visual detector not found: $VISUAL_MODEL" >&2
      exit 1
    fi
    HOOK_ARGS+=(--visual-model "$VISUAL_MODEL")
  fi
  [[ "$PRIVACY_ENABLED" == 1 ]] && HOOK_ARGS+=(--privacy)
  if [[ "$PRIVACY_ENABLED" == 1 ]]; then
    POLICY_FILE="${PLVA_POLICY_FILE:-config/privacy-policy.json}"
    if [[ -z "${PLVA_POLICY_JSON:-}" && -f "$POLICY_FILE" ]]; then
      export PLVA_POLICY_JSON="$(<"$POLICY_FILE")"
    fi
    [[ "$PRIVACY_HISTORY_SCRUB" == 0 ]] && HOOK_ARGS+=(--no-privacy-history-scrub)
    [[ "$PRIVACY_CHIPS" == 0 ]] && HOOK_ARGS+=(--no-privacy-chips)
    [[ "$PRIVACY_SCHEME" == 0 ]] && HOOK_ARGS+=(--no-privacy-scheme)
    [[ "$PRIVACY_DUPLICATE_WARNING" == 0 ]] && HOOK_ARGS+=(--no-privacy-duplicate-warning)
    [[ "$PRIVACY_MANIFEST" == 0 ]] && HOOK_ARGS+=(--no-privacy-manifest)
    [[ "$PRIVACY_RESOLUTION" == 0 ]] && HOOK_ARGS+=(--no-privacy-resolution)
    [[ "$PRIVACY_POLICY_TEACHING" == 0 ]] && HOOK_ARGS+=(--no-privacy-policy-teaching)
  fi
  echo "--- redaction ON ($REDACT_ENGINE, ${PLVA_REDACT_LIFECYCLE:-adaptive}); privacy=$PRIVACY_ENABLED; watch frames at http://127.0.0.1:$PORT/viewer and OCR at /viewer/findings"
  if [[ "$PRIVACY_ENABLED" == 1 ]]; then
    echo "--- privacy features: chips=$PRIVACY_CHIPS history_scrub=$PRIVACY_HISTORY_SCRUB scheme=$PRIVACY_SCHEME duplicate_warning=$PRIVACY_DUPLICATE_WARNING manifest=$PRIVACY_MANIFEST resolution=$PRIVACY_RESOLUTION policy_teaching=$PRIVACY_POLICY_TEACHING"
    if [[ "$PRIVACY_HISTORY_SCRUB" == 0 || "$PRIVACY_RESOLUTION" == 0 ]]; then
      echo "--- WARNING: diagnostic privacy feature disablement; use synthetic data only" >&2
    fi
  fi
else
  echo "--- redaction OFF"
fi
parse_on_off PRIVACY_SKILL "${PLVA_PRIVACY_SKILL:-$PRIVACY_ENABLED}" PLVA_PRIVACY_SKILL
.venv/bin/plva-proxy --port "$PORT" "${HOOK_ARGS[@]}" >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!
OBS_FILE=$(mktemp /tmp/plva-step1-egress.XXXXXX)
RUNS_DIR=""
SKILL_DISABLED_FILE=""
cleanup() {
  kill "$PROXY_PID" "${OBS_PID:-}" 2>/dev/null || true
  # Frame-bearing artifacts must never survive, even on an aborted run.
  [[ -n "$RUNS_DIR" ]] && rm -rf "$RUNS_DIR"
  if [[ -n "$SKILL_DISABLED_FILE" && -f "$SKILL_DISABLED_FILE" ]]; then
    mv "$SKILL_DISABLED_FILE" "$HOME/.holo/skills/plva-placeholders/SKILL.md"
  fi
}
trap cleanup EXIT
if [[ "$PRIVACY_SKILL" == 0 && -f "$HOME/.holo/skills/plva-placeholders/SKILL.md" ]]; then
  SKILL_DISABLED_FILE="$HOME/.holo/skills/plva-placeholders/SKILL.md.disabled.$$"
  mv "$HOME/.holo/skills/plva-placeholders/SKILL.md" "$SKILL_DISABLED_FILE"
  echo "--- native placeholder skill disabled for this diagnostic run"
elif [[ "$PRIVACY_SKILL" == 1 ]]; then
  echo "--- native placeholder skill enabled"
fi
PROXY_UP=""
for _ in $(seq 1 240); do
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
echo "--- preflight: provider=$PROVIDER model=$MODEL"
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import json, sys
raw = sys.stdin.read()
try:
    ids = [m.get('id') for m in json.loads(raw).get('data', [])]
except ValueError:
    print('preflight got no valid JSON from the proxy')
    raise SystemExit(1)
expected = sys.argv[1]
ok = expected in ids
print(expected + ' advertised:', ok)
raise SystemExit(0 if ok else 1)
" "$MODEL"; then
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
  --model "$MODEL" \
  --max-steps 20 --max-time-s 300 \
  --runs-dir "$RUNS_DIR" &
HOLO_PID=$!
set +m

ABORTED=""
if [[ -t 0 && -r /dev/tty ]]; then
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
