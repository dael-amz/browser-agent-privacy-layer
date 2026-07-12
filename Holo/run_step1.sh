#!/usr/bin/env bash
# One-command Step 1 run: proxy + key preflight + holo task + enforced egress monitor + cleanup.
#
#   ./run_step1.sh                          # default terminal task
#   ./run_step1.sh "your own task prompt"   # custom task for the agent
#   PLVA_AUDIT=1 ./run_step1.sh "task"       # keep the memory-only viewer alive afterward
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
  echo "  PLVA_AUDIT=1 keeps the redacted sent-frame viewer alive until Ctrl-C"
  echo "  PLVA_EGRESS_STATUS_FILE=/safe/path.json retains privacy-safe verifier evidence"
  echo "  during the run: press Esc twice to abort"
  exit 0
fi

PORT="${PLVA_PORT:-18081}"
UV="${UV:-$HOME/.local/bin/uv}"
TASK="${1:-$DEFAULT_TASK}"
PROVIDER="${PLVA_PROVIDER:-overshoot}"
REDACT_ENGINE="${PLVA_REDACT_ENGINE:-accelerated}"
REDACT_PROFILE="${PLVA_REDACT_PROFILE:-balanced}"
INSTANCE_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
MAX_STEPS="${PLVA_MAX_STEPS:-20}"
MAX_TIME_S="${PLVA_MAX_TIME_S:-300}"
if [[ ! "$MAX_STEPS" =~ ^[1-9][0-9]*$ || ! "$MAX_TIME_S" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PLVA_MAX_STEPS and PLVA_MAX_TIME_S must be positive integers" >&2
  exit 1
fi
if [[ "$REDACT_PROFILE" != "balanced" && "$REDACT_PROFILE" != "high-recall" ]]; then
  echo "ERROR: PLVA_REDACT_PROFILE must be balanced or high-recall" >&2
  exit 1
fi
case "${PLVA_AUDIT:-0}" in
  1|true|TRUE|yes|YES|on|ON) AUDIT_MODE=1 ;;
  0|false|FALSE|no|NO|off|OFF|"") AUDIT_MODE=0 ;;
  *)
    echo "ERROR: PLVA_AUDIT must be an on/off value" >&2
    exit 1
    ;;
esac

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
    if [[ -z "${OVERSHOOT_API_KEY:-}${API_KEY:-}" ]] \
      && { [[ ! -f "$HOME/.holo/.env" ]] || ! grep -Eq '^(OVERSHOOT_API_KEY|API_KEY)=..+' "$HOME/.holo/.env"; } \
      && { [[ ! -f .env ]] || ! grep -Eq '^(OVERSHOOT_API_KEY|API_KEY)=..+' .env; }; then
      echo "ERROR: put API_KEY=<Overshoot key> in Holo/.env or ~/.holo/.env" >&2
      exit 1
    fi
    ;;
  hcompany)
    MODEL="${PLVA_MODEL:-holo3-1-35b-a3b}"
    if [[ -z "${HAI_API_KEY:-}" ]] \
      && { [[ ! -f "$HOME/.holo/.env" ]] || ! grep -Eq '^HAI_API_KEY=..+' "$HOME/.holo/.env"; } \
      && { [[ ! -f .env ]] || ! grep -Eq '^HAI_API_KEY=..+' .env; }; then
      echo "ERROR: put HAI_API_KEY=<H Company key> in Holo/.env, or run holo login (saves to ~/.holo/.env)" >&2
      exit 1
    fi
    ;;
  *)
    echo "ERROR: PLVA_PROVIDER must be overshoot or hcompany" >&2
    exit 1
    ;;
esac

# Report the optional host-level PF boundary without requesting elevation. The
# per-run process-group guard below is mandatory regardless of PF status.
PF_STATUS_JSON=$(.venv/bin/python -m plva_proxy.egress_preflight)
echo "PLVA_PF_STATUS_JSON=$PF_STATUS_JSON"
if pgrep -f '/hai-agent-runtime$' >/dev/null 2>&1; then
  echo "ERROR: a Holo runtime is already active; stop it before starting a verified PLVA run" >&2
  exit 1
fi

# 1) Start the loopback proxy (the sole provider egress; reads ./.env).
#    PLVA_HOOK=test enables the Step 3 test hooks for the hook-mode verify.
#    PLVA_HOOK_IMAGE=/path/to.png replaces every outbound screenshot with that
#    static image (no real desktop pixels egress; fails closed if no frame).
#    PLVA_REDACT=1 redacts every outbound screenshot through plva-v2-baseline
#    and serves the obscured frames at http://127.0.0.1:$PORT/viewer.
PROXY_LOG=$(mktemp /tmp/plva-proxy-step1.XXXXXX)
chmod 600 "$PROXY_LOG"
HOOK_ARGS=(--provider "$PROVIDER" --hook "${PLVA_HOOK:-none}" --instance-token "$INSTANCE_TOKEN")
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
    --redact-profile "$REDACT_PROFILE"
    --redact-backend "${PLVA_REDACT_BACKEND:-auto}"
    --vision-worker "${PLVA_VISION_WORKER:-coreml-redactor}"
    --vision-mode "${PLVA_VISION_MODE:-fast}"
    --redact-lifecycle "${PLVA_REDACT_LIFECYCLE:-adaptive}"
    --redact-idle-seconds "${PLVA_REDACT_IDLE_SECONDS:-60}"
    --audit-capacity "${PLVA_AUDIT_CAPACITY:-32}"
  )
  if [[ "$REDACT_ENGINE" == "vision" ]]; then
    OCR_ENGINE="${PLVA_OCR_ENGINE:-apple}"
    case "$OCR_ENGINE" in
      apple|rapidocr) ;;
      *)
        echo "ERROR: PLVA_OCR_ENGINE must be apple or rapidocr" >&2
        exit 1
        ;;
    esac
    HOOK_ARGS+=(--ocr-engine "$OCR_ENGINE")
    SEMANTIC_ENGINE="${PLVA_SEMANTIC_ENGINE:-rampart}"
    case "$SEMANTIC_ENGINE" in
      rampart) ;;
      gliner2|openai-pf|openai-pf-4bit)
        SEMANTIC_DIR="coreml-redactor/models/semantic/gliner2-privacy-filter-PII-multi"
        SEMANTIC_REPO="fastino/gliner2-privacy-filter-PII-multi"
        if [[ "$SEMANTIC_ENGINE" == "openai-pf" ]]; then
          SEMANTIC_DIR="coreml-redactor/models/semantic/openai-privacy-filter"
          SEMANTIC_REPO="openai/privacy-filter"
        elif [[ "$SEMANTIC_ENGINE" == "openai-pf-4bit" ]]; then
          SEMANTIC_DIR="coreml-redactor/models/semantic/openai-privacy-filter-4bit"
          SEMANTIC_REPO="mlx-community/openai-privacy-filter-4bit"
        fi
        if [[ ! -d "$SEMANTIC_DIR" ]]; then
          echo "ERROR: $SEMANTIC_ENGINE semantic model is not installed: $SEMANTIC_DIR" >&2
          echo "       Fetch a frozen copy first: huggingface-cli download $SEMANTIC_REPO --local-dir $SEMANTIC_DIR" >&2
          exit 1
        fi
        ;;
      *)
        echo "ERROR: PLVA_SEMANTIC_ENGINE must be rampart, gliner2, openai-pf, or openai-pf-4bit" >&2
        exit 1
        ;;
    esac
    HOOK_ARGS+=(--semantic-engine "$SEMANTIC_ENGINE")
    parse_on_off VISUAL_DETECTOR "${PLVA_VISUAL_DETECTOR:-0}" PLVA_VISUAL_DETECTOR
    if [[ "$VISUAL_DETECTOR" == 0 ]]; then
      HOOK_ARGS+=(--no-visual-detector)
    else
      DETECTOR_VERSION="${PLVA_DETECTOR_VERSION:-v2}"
      case "$DETECTOR_VERSION" in
        v2) DEFAULT_VISUAL_MODEL="plva-v2-baseline/runtime/training/artifacts/plva-visual-agpl-test-v2/visual/detector.onnx" ;;
        v3) DEFAULT_VISUAL_MODEL="plva-v3/dist/visual/detector.onnx" ;;
        *)
          echo "ERROR: PLVA_DETECTOR_VERSION must be v2 or v3" >&2
          exit 1
          ;;
      esac
      VISUAL_MODEL="${PLVA_VISUAL_MODEL:-$DEFAULT_VISUAL_MODEL}"
      if [[ ! -f "$VISUAL_MODEL" ]]; then
        echo "ERROR: visual detector not found: $VISUAL_MODEL" >&2
        if [[ "$DETECTOR_VERSION" == "v3" && -z "${PLVA_VISUAL_MODEL:-}" ]]; then
          echo "       Extract the frozen plva-v3 bundle to Holo/plva-v3 or select v2." >&2
        fi
        exit 1
      fi
      HOOK_ARGS+=(--visual-detector --visual-model "$VISUAL_MODEL")
    fi
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
    [[ "${PRIVACY_TOOLS:-0}" == 1 ]] && HOOK_ARGS+=(--privacy-tools)
    [[ "${PRIVACY_MEDIATOR:-0}" == 1 ]] && HOOK_ARGS+=(--privacy-mediator)
  fi
  echo "--- redaction ON ($REDACT_ENGINE, $REDACT_PROFILE, ${PLVA_REDACT_LIFECYCLE:-adaptive}); privacy=$PRIVACY_ENABLED; watch frames at http://127.0.0.1:$PORT/viewer and OCR at /viewer/findings"
  if [[ "$PRIVACY_ENABLED" == 1 ]]; then
    echo "--- privacy features: chips=$PRIVACY_CHIPS history_scrub=$PRIVACY_HISTORY_SCRUB scheme=$PRIVACY_SCHEME duplicate_warning=$PRIVACY_DUPLICATE_WARNING manifest=$PRIVACY_MANIFEST resolution=$PRIVACY_RESOLUTION policy_teaching=$PRIVACY_POLICY_TEACHING"
    if [[ "$PRIVACY_HISTORY_SCRUB" == 0 || "$PRIVACY_RESOLUTION" == 0 ]]; then
      echo "--- WARNING: diagnostic privacy feature disablement; use synthetic data only" >&2
    fi
  fi
else
  if [[ "$AUDIT_MODE" == 1 ]]; then
    echo "ERROR: PLVA_AUDIT=1 requires PLVA_REDACT=1" >&2
    exit 1
  fi
  echo "--- redaction OFF"
fi
parse_on_off PRIVACY_SKILL "${PLVA_PRIVACY_SKILL:-$PRIVACY_ENABLED}" PLVA_PRIVACY_SKILL
# Tools do not require redaction, so this toggle lives outside the PLVA_REDACT block above.
parse_on_off TOOLS_ENABLED "${PLVA_TOOLS:-0}" PLVA_TOOLS
parse_on_off TOOLS_SKILL "${PLVA_TOOLS_SKILL:-$TOOLS_ENABLED}" PLVA_TOOLS_SKILL
if [[ "$TOOLS_SKILL" == 1 && "$TOOLS_ENABLED" == 0 ]]; then
  echo "ERROR: PLVA_TOOLS_SKILL=1 requires PLVA_TOOLS=1 (the proxy must detect tool calls the skill teaches)" >&2
  exit 1
fi
[[ "$TOOLS_ENABLED" == 1 ]] && HOOK_ARGS+=(--tools)
if [[ -n "${PLVA_CAPTURE_GRAMMAR:-}" ]]; then
  HOOK_ARGS+=(--capture-grammar "$PLVA_CAPTURE_GRAMMAR")
fi
.venv/bin/plva-proxy --port "$PORT" "${HOOK_ARGS[@]}" >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!
RUNS_DIR=""
SKILL_DISABLED_FILE=""
TOOLS_SKILL_DISABLED_FILE=""
EGRESS_READY_FILE=$(mktemp /tmp/plva-step1-egress-ready.XXXXXX)
if [[ -n "${PLVA_EGRESS_STATUS_FILE:-}" ]]; then
  EGRESS_STATUS_FILE="$PLVA_EGRESS_STATUS_FILE"
  EGRESS_STATUS_EPHEMERAL=0
else
  EGRESS_STATUS_FILE=$(mktemp /tmp/plva-step1-egress-status.XXXXXX)
  EGRESS_STATUS_EPHEMERAL=1
fi
cleanup() {
  set +e
  [[ -n "${HOLO_PID:-}" ]] && kill -- -"$HOLO_PID" 2>/dev/null
  # The closed runtime daemonizes outside the CLI process group. The preflight
  # above guarantees any matching runtime now belongs to this run.
  RUNTIME_PIDS=$(pgrep -f '/hai-agent-runtime$' 2>/dev/null || true)
  for runtime_pid in $RUNTIME_PIDS; do
    kill "$runtime_pid" 2>/dev/null
  done
  for _ in $(seq 1 20); do
    [[ -z "$(pgrep -f '/hai-agent-runtime$' 2>/dev/null || true)" ]] && break
    sleep 0.05
  done
  for runtime_pid in $(pgrep -f '/hai-agent-runtime$' 2>/dev/null || true); do
    kill -9 "$runtime_pid" 2>/dev/null
  done
  kill "$PROXY_PID" "${EGRESS_PID:-}" 2>/dev/null
  wait "$PROXY_PID" "${EGRESS_PID:-}" 2>/dev/null
  # Frame-bearing artifacts must never survive, including on signals and failed preflights.
  [[ -n "$RUNS_DIR" ]] && rm -rf "$RUNS_DIR"
  rm -f "$EGRESS_READY_FILE" "$PROXY_LOG"
  [[ "$EGRESS_STATUS_EPHEMERAL" == 1 ]] && rm -f "$EGRESS_STATUS_FILE"
  if [[ -n "$SKILL_DISABLED_FILE" && -f "$SKILL_DISABLED_FILE" ]]; then
    mv "$SKILL_DISABLED_FILE" "$HOME/.holo/skills/plva-placeholders/SKILL.md"
  fi
  if [[ -n "${TOOLS_SKILL_DISABLED_FILE:-}" && -f "$TOOLS_SKILL_DISABLED_FILE" ]]; then
    mv "$TOOLS_SKILL_DISABLED_FILE" "$HOME/.holo/skills/plva-tools/SKILL.md"
  fi
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM
if [[ "$PRIVACY_SKILL" == 0 && -f "$HOME/.holo/skills/plva-placeholders/SKILL.md" ]]; then
  SKILL_DISABLED_FILE="$HOME/.holo/skills/plva-placeholders/SKILL.md.disabled.$$"
  mv "$HOME/.holo/skills/plva-placeholders/SKILL.md" "$SKILL_DISABLED_FILE"
  echo "--- native placeholder skill disabled for this diagnostic run"
elif [[ "$PRIVACY_SKILL" == 1 ]]; then
  mkdir -p "$HOME/.holo/skills/plva-placeholders"
  cp "holo-skills/plva-placeholders/SKILL.md" "$HOME/.holo/skills/plva-placeholders/SKILL.md"
  echo "--- native placeholder skill enabled"
fi
if [[ "$TOOLS_SKILL" == 0 && -f "$HOME/.holo/skills/plva-tools/SKILL.md" ]]; then
  TOOLS_SKILL_DISABLED_FILE="$HOME/.holo/skills/plva-tools/SKILL.md.disabled.$$"
  mv "$HOME/.holo/skills/plva-tools/SKILL.md" "$TOOLS_SKILL_DISABLED_FILE"
  echo "--- native tools skill disabled for this diagnostic run"
elif [[ "$TOOLS_SKILL" == 1 ]]; then
  mkdir -p "$HOME/.holo/skills/plva-tools"
  cp "holo-skills/plva-tools/SKILL.md" "$HOME/.holo/skills/plva-tools/SKILL.md"
  echo "--- native tools skill enabled"
fi
PROXY_UP=""
for _ in $(seq 1 240); do
  if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    break
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | python3 -c '
import json, sys
try:
    payload = json.load(sys.stdin)
except (ValueError, OSError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("instance") == sys.argv[1] else 1)
' "$INSTANCE_TOKEN"; then
    PROXY_UP=1
    break
  fi
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
# Advertised is not usable: promotion-listed models can still 402 at call time.
# A one-token text probe through the proxy surfaces tier/credit gating up front.
PROBE_RESPONSE=$(curl -s -m 30 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
  -w $'\n%{http_code}')
PROBE_CODE="${PROBE_RESPONSE##*$'\n'}"
if [[ "$PROBE_CODE" != 200 ]]; then
  echo "ERROR: model $MODEL is advertised but rejected a 1-token probe (HTTP $PROBE_CODE)" >&2
  printf '%s' "${PROBE_RESPONSE%$'\n'*}" | head -c 400 >&2
  echo >&2
  exit 1
fi
echo "--- preflight: $MODEL answered a 1-token probe"

# 3) The Step 1 task. Frame-bearing run artifacts go to a private ephemeral directory.
RUNS_DIR=$(mktemp -d /tmp/holo-step1-runs.XXXXXX)
chmod 700 "$RUNS_DIR"
echo "--- task received (${#TASK} characters; content omitted from logs)"
echo "--- press Esc twice to abort; ephemeral runs dir will be removed afterward"
set -m  # own process group for the holo job so an abort kills the runtime too
# Stop before exec so the verifier is ready before uv or the runtime can open a socket.
UV_OFFLINE=1 .venv/bin/python -m plva_proxy.stopped_exec \
  "$UV" tool run --from holo-desktop-cli holo run "$TASK" \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL" \
  --max-steps "$MAX_STEPS" --max-time-s "$MAX_TIME_S" \
  --runs-dir "$RUNS_DIR" &
HOLO_PID=$!
set +m

# 4) Prove non-privileged socket visibility, then prohibit external runtime egress.
# Holo's local CLI↔Agent-API control channel uses another loopback port, so all
# loopback endpoints are safe; any non-loopback remote kills the runtime group.
.venv/bin/python -m plva_proxy.egress_verify \
  --pgid "$HOLO_PID" --allowed-port "$PORT" \
  --ready-file "$EGRESS_READY_FILE" --status-file "$EGRESS_STATUS_FILE" &
EGRESS_PID=$!
EGRESS_READY=""
for _ in $(seq 1 100); do
  if [[ -s "$EGRESS_READY_FILE" ]]; then
    EGRESS_READY=1
    break
  fi
  if ! kill -0 "$EGRESS_PID" 2>/dev/null; then
    break
  fi
  sleep 0.05
done
if [[ -z "$EGRESS_READY" ]]; then
  kill -- -"$HOLO_PID" 2>/dev/null || true
  wait "$HOLO_PID" 2>/dev/null || true
  echo "ERROR: egress verifier could not establish socket visibility; runtime was not resumed" >&2
  [[ -s "$EGRESS_STATUS_FILE" ]] && sed 's/^/PLVA_EGRESS_STATUS_JSON=/' "$EGRESS_STATUS_FILE" >&2
  exit 1
fi
echo "--- egress guard ready: runtime may use local control ports; external egress is blocked"
kill -CONT -- -"$HOLO_PID"

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
EGRESS_EXIT=0
wait "$EGRESS_PID" 2>/dev/null || EGRESS_EXIT=$?

# 5) Remove frame-bearing artifacts immediately; report privacy-safe evidence.
rm -rf "$RUNS_DIR"
RUNS_DIR=""
echo "--- ephemeral runs dir removed"
if [[ ! -s "$EGRESS_STATUS_FILE" ]]; then
  echo "ERROR: egress verifier produced no status evidence" >&2
  exit 1
fi
EGRESS_VERDICT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("verdict", "error"))' "$EGRESS_STATUS_FILE")
sed 's/^/PLVA_EGRESS_STATUS_JSON=/' "$EGRESS_STATUS_FILE"
if [[ "$EGRESS_EXIT" != 0 || "$EGRESS_VERDICT" != passed ]]; then
  echo "ERROR: loopback-only runtime egress was not verified; see status JSON above" >&2
  exit 1
fi
if [[ "$EGRESS_STATUS_EPHEMERAL" == 0 ]]; then
  echo "--- privacy-safe egress evidence retained at: $EGRESS_STATUS_FILE"
fi
echo "--- proxy relay timings (privacy-safe):"
grep -o 'relay .*' "$PROXY_LOG" | tail -5 || true
if [[ -n "$ABORTED" ]]; then
  echo "--- run aborted by Esc Esc"
else
  echo "--- holo exit: $HOLO_EXIT (0 = Step 1 task completed end-to-end)"
fi
if [[ "$AUDIT_MODE" == 1 ]]; then
  echo "--- safe sent-frame audit remains at http://127.0.0.1:$PORT/viewer"
  echo "--- JSON metadata: http://127.0.0.1:$PORT/viewer/frames"
  echo "--- press Ctrl-C when finished auditing; all buffered frames will be discarded"
  while kill -0 "$PROXY_PID" 2>/dev/null; do
    sleep 1
  done
fi
exit "$HOLO_EXIT"
