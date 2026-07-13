#!/bin/bash
# PLVA mediator sandbox lifecycle (NemoClaw/OpenShell, §7).
#
#   ./run_mediator_sandbox.sh up       start everything, verify zero egress, forward 127.0.0.1:8555
#   ./run_mediator_sandbox.sh status   stack + egress status
#   ./run_mediator_sandbox.sh down     stop the model + forward (frees ~3 GB in the VM)
#   ./run_mediator_sandbox.sh down --full   also quit Docker Desktop and the gateway (frees everything)
#
# The sandboxed model may see vault cleartext, so `up` HARD-FAILS unless a live
# deny-test from inside the sandbox proves outbound egress is blocked. Memory
# budget on this 16 GB machine: Docker VM capped at 4 GB, container at 3.5 GB.
# Run `down` as soon as a mediation session ends.
set -euo pipefail

SANDBOX=plva-llm
LOCAL_PORT=8555
INNER_PORT=11500
HOLO_DIR="$(cd "$(dirname "$0")" && pwd)"
POLICY="$HOLO_DIR/config/openshell-mediator-policy.yaml"
GGUF="$HOLO_DIR/models/qwen3-4b-instruct-2507-q4_k_m.gguf"
LLAMA_RELEASE=b9977
LLAMA_TARBALL="/tmp/llama-${LLAMA_RELEASE}-bin-ubuntu-arm64.tar.gz"

say() { printf '\033[1m[mediator-sandbox]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[mediator-sandbox] FATAL:\033[0m %s\n' "$*" >&2; exit 1; }

docker_up() {
  docker info >/dev/null 2>&1 && return 0
  say "starting Docker Desktop…"
  open -a Docker
  for _ in $(seq 1 60); do docker info >/dev/null 2>&1 && return 0; sleep 3; done
  die "Docker daemon did not come up"
}

gateway_up() {
  openshell status 2>/dev/null | grep -q Connected && return 0
  say "starting OpenShell gateway…"
  brew services start openshell >/dev/null
  for _ in $(seq 1 20); do
    openshell status 2>/dev/null | grep -q Connected && return 0
    sleep 2
  done
  die "OpenShell gateway did not connect"
}

sandbox_ready() { openshell sandbox list 2>/dev/null | grep -q "$SANDBOX.*Ready"; }

provision() {
  say "provisioning sandbox (one-time: image + 2.4 GB model upload)…"
  [ -f "$GGUF" ] || die "model file missing: $GGUF"
  openshell sandbox create --name "$SANDBOX" --from ollama --memory 3584Mi --cpu 3
  if [ ! -f "$LLAMA_TARBALL" ]; then
    say "downloading llama.cpp $LLAMA_RELEASE (linux-arm64)…"
    curl -sL -o "$LLAMA_TARBALL" \
      "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_RELEASE}/llama-${LLAMA_RELEASE}-bin-ubuntu-arm64.tar.gz"
  fi
  openshell sandbox upload "$SANDBOX" "$GGUF" /sandbox/models/qwen3-4b.gguf
  openshell sandbox upload "$SANDBOX" "$LLAMA_TARBALL" /sandbox/llama.tar.gz
  openshell sandbox exec -n "$SANDBOX" -- bash -c 'cd /sandbox && tar xzf llama.tar.gz'
}

enforce_zero_egress() {
  say "applying zero-egress policy…"
  openshell policy set "$SANDBOX" --policy "$POLICY" >/dev/null
  sleep 5
  say "deny-testing from inside the sandbox…"
  local result
  result=$(openshell sandbox exec -n "$SANDBOX" --timeout 40 -- bash -c \
    'ok=1; curl -sS -m 6 -o /dev/null https://example.com 2>/dev/null && ok=0; curl -sS -m 6 -o /dev/null https://ollama.com 2>/dev/null && ok=0; curl -sS -m 6 -o /dev/null https://1.1.1.1 2>/dev/null && ok=0; timeout 5 getent hosts example.com >/dev/null 2>&1 && ok=0; [ "$ok" = 1 ] && echo EGRESS_BLOCKED || echo EGRESS_OPEN')
  echo "$result" | grep -q EGRESS_BLOCKED || die "egress is NOT blocked — refusing to serve cleartext"
  say "zero egress verified (HTTPS, allowlisted host, raw IP, DNS all blocked)"
}

model_up() {
  if ! openshell sandbox exec -n "$SANDBOX" --timeout 15 -- \
      curl -s --max-time 4 "http://127.0.0.1:${INNER_PORT}/health" 2>/dev/null | grep -q ok; then
    say "starting llama-server inside the sandbox…"
    openshell sandbox exec -n "$SANDBOX" --timeout 20 -- bash -c \
      "pkill -x ollama 2>/dev/null; pkill -x llama-server 2>/dev/null; LD_LIBRARY_PATH=/sandbox/llama-${LLAMA_RELEASE} nohup /sandbox/llama-${LLAMA_RELEASE}/llama-server -m /sandbox/models/qwen3-4b.gguf --host 127.0.0.1 --port ${INNER_PORT} --ctx-size 4096 --alias qwen3-4b --no-webui > /tmp/llama-server.log 2>&1 & true"
    for _ in $(seq 1 30); do
      openshell sandbox exec -n "$SANDBOX" --timeout 15 -- \
        curl -s --max-time 4 "http://127.0.0.1:${INNER_PORT}/health" 2>/dev/null | grep -q ok && break
      sleep 4
    done
  fi
  say "model healthy inside the sandbox"
}

forward_up() {
  if ! curl -s --max-time 3 "http://127.0.0.1:${LOCAL_PORT}/health" 2>/dev/null | grep -q ok; then
    say "forwarding 127.0.0.1:${LOCAL_PORT} -> sandbox:${INNER_PORT}…"
    nohup openshell forward service "$SANDBOX" --target-port "$INNER_PORT" \
      --local "127.0.0.1:${LOCAL_PORT}" > /tmp/plva-forward.log 2>&1 &
    sleep 3
  fi
  curl -s --max-time 5 "http://127.0.0.1:${LOCAL_PORT}/health" | grep -q ok \
    || die "forward did not come up"
  say "ready: OpenAI-compatible endpoint at http://127.0.0.1:${LOCAL_PORT}/v1 (model alias qwen3-4b)"
}

case "${1:-}" in
  up)
    docker_up
    gateway_up
    sandbox_ready || provision
    enforce_zero_egress
    model_up
    forward_up
    ;;
  status)
    docker info >/dev/null 2>&1 && echo "docker: up" || echo "docker: down"
    openshell status 2>/dev/null | grep -q Connected && echo "gateway: connected" || echo "gateway: down"
    sandbox_ready && echo "sandbox: ready" || echo "sandbox: absent/stopped"
    curl -s --max-time 3 "http://127.0.0.1:${LOCAL_PORT}/health" 2>/dev/null | grep -q ok \
      && echo "endpoint: healthy on 127.0.0.1:${LOCAL_PORT}" || echo "endpoint: down"
    ;;
  down)
    pkill -f "openshell forward service $SANDBOX" 2>/dev/null || true
    openshell sandbox exec -n "$SANDBOX" --timeout 15 -- bash -c \
      'pkill -x llama-server 2>/dev/null; pkill -x ollama 2>/dev/null; true' 2>/dev/null || true
    say "model + forward stopped (sandbox and Docker stay provisioned)"
    if [ "${2:-}" = "--full" ]; then
      brew services stop openshell >/dev/null 2>&1 || true
      osascript -e 'quit app "Docker"' 2>/dev/null || true
      say "gateway + Docker Desktop stopped — all memory released"
    fi
    ;;
  *)
    echo "usage: $0 {up|status|down [--full]}"
    exit 2
    ;;
esac
