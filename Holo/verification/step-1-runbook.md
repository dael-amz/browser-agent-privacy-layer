# Step 1 runbook — end-to-end task via the loopback proxy

Date: 2026-07-11

Everything below is built, tested, and ready. The remaining input is an inference-provider key.
One full pass of §3 + §4 closes Step 1's "agent finishes a task end-to-end" verify.

**Fastest path:** do §1, then run `./run_step1.sh` from `Holo/` — it performs §2–§4
(proxy, preflight, task, egress observation, artifact shredding) in one command. The sections
below are the manual equivalent.

> **Privacy note (read first):** this run streams **unredacted** screenshots of the visible
> desktop to Overshoot — redaction does not exist until Step 4. Close anything sensitive before
> starting. Keep the runtime kill switch enabled (do **not** pass `--no-kill-switch`).

## 1. Put the key in place (operator)

For Overshoot, create `Holo/.env` (git-ignored; never committed) with:

```
API_KEY=<your Overshoot key>
```

For H Company instead, use:

```text
HAI_API_KEY=<your H Company key>
```

and launch with `PLVA_PROVIDER=hcompany`. The presets select the matching endpoint and model.

`plva-proxy` reads the selected provider's key from the environment first, then from `./.env`.

## 2. Start the proxy and preflight the key (Terminal A)

```bash
cd ~/Hackathon/Holo
.venv/bin/plva-proxy            # default: Overshoot /v1
# or: .venv/bin/plva-proxy --provider hcompany
```

Preflight without sending any frame — lists model ids only, through the proxy:

```bash
curl -s http://127.0.0.1:18081/v1/models | python3 -c \
  "import json,sys; print([m.get('id') for m in json.load(sys.stdin)['data'] if 'Holo3' in str(m.get('id'))])"
```

Expect `['Hcompany/Holo3-35B-A3B']`. A `401` here means the key is wrong; `502` means Overshoot
is unreachable.

## 3. Run the Step 1 task (Terminal B)

```bash
RUNS_DIR=$(mktemp -d /tmp/holo-step1-runs.XXXXXX)
uv tool run --from holo-desktop-cli holo run \
  "Open the Terminal application and run the command: echo plva-step1-ok" \
  --base-url http://127.0.0.1:18081/v1 \
  --model Hcompany/Holo3-35B-A3B \
  --max-steps 20 --max-time-s 300 \
  --runs-dir "$RUNS_DIR"
```

- The closed runtime unconditionally writes frame-bearing `events.jsonl` under `--runs-dir`
  (no disable knob), so it points at ephemeral `/tmp`; **shred it immediately after the run**:

```bash
rm -rf "$RUNS_DIR"
```

- Screen Recording permission was granted for Step 0; the first *actuating* run may additionally
  prompt for **Accessibility** — grant it and re-run.

## 4. Verify egress while the run is active (Terminal C)

**Tier A — enforced per-run guard (no sudo):** `run_step1.sh` starts the CUA
process group stopped, proves that `lsof` can observe a local canary socket,
then samples the whole process group and terminates it if any remote endpoint
leaves loopback. Holo's CLI-to-Agent-API control channel and the PLVA proxy are
both local and therefore allowed. The runtime is not resumed if visibility
cannot be proved. Each run prints a `PLVA_EGRESS_STATUS_JSON=...` evidence line;
set `PLVA_EGRESS_STATUS_FILE=/protected/path.json` to retain it with mode 0600.

For manual diagnosis only:

```bash
# The runtime's only remote endpoint must be the loopback proxy:
lsof -nP -iTCP -sTCP:ESTABLISHED | grep -iE "hai[-_]?agent|holo"
# Expect every line to end in ->127.0.0.1:18081

# Only the proxy (a python process) may hold a connection to the provider:
for ip in $(dig +short api.overshoot.ai); do
  lsof -nP -iTCP -sTCP:ESTABLISHED | grep "$ip"
done
# Expect only the plva-proxy python process
```

**Tier B — host packet filter (optional, administrator bootstrap):** first run
`.venv/bin/python -m plva_proxy.egress_preflight`. It makes no privileged calls
and emits machine-readable readiness plus actions. If it reports `ready:false`,
follow `docs/egress/bootstrap-pf.md`; the project never invokes sudo itself.

## 5. Record the outcome

On success, append to `verification/step-1-status.md`: holo exit code, that the echo ran on
screen, the Tier A `lsof` observations (endpoints only — never frames or bodies), and the proxy's
`duration_ms` log lines (these seed Step 2's latency measurements). Confirm `$RUNS_DIR` was
shredded. That closes Step 1.
