# Step 1 status — HoloDesktop under OpenShell isolation

Date: 2026-07-11

Status: **READY TO RUN — every buildable blocker is cleared; the end-to-end verify waits only on
the operator's Overshoot key. See the 2026-07-11 resume section at the bottom and
`step-1-runbook.md`.** (Original assessment kept below for the record.)

Original status: **PARTIAL — the §7 decision is resolved and recorded; the end-to-end "runs under
OpenShell isolation" verification cannot pass on this machine and is blocked, not failed.**

## What Step 1 asks for

- Build: get HoloDesktop running on the easiest working inference endpoint; resolve the §7 open
  decision and stand up the chosen OpenShell topology; confirm a basic task completes.
- Verify: the agent finishes a task end-to-end; you can name exactly what runs inside OpenShell and
  its allowed egress; the decision is recorded in `docs/decisions/`.

## Done

- **§7 decision resolved and recorded** in `docs/decisions/0001-openshell-sec7-egress-topology.md`:
  egress-isolation (option a), with a host packet-filter as the egress boundary because OpenShell
  cannot govern a host process; the PLVA proxy is the sole provider egress; option (b)
  contained-desktop is deferred until a Linux runtime exists.
- **Transport premise proven** (Step 0 critical gate): the closed runtime does send its screenshot
  through the configurable base URL — see `verification/step-0-runtime-capture.md`.

## Blocked (with the specific reason)

1. **OpenShell cannot isolate the host runtime.** OpenShell only constrains processes it launches
   inside its Linux sandbox; it "does not wrap or attach to existing host processes." The Holo
   runtime must run on the macOS host for Screen Recording + Accessibility. So the literal
   "runtime's egress forced through OpenShell" cannot be built here.
2. **No Linux `hai-agent-runtime` exists**, so the stronger contained-desktop topology (running the
   whole CUA inside the sandbox) is impossible on this host.
3. **The enforcing egress boundary is not yet stood up.** It depends on the Step 3 interception
   proxy (not built) plus a macOS packet-filter rule set pinning the runtime to loopback and the
   proxy to the provider only.
4. **A real end-to-end task would stream many real host frames to Overshoot** (there is no
   redaction until Step 4). Only a single local-loopback capture was authorized; streaming live
   desktop frames to the provider is a separate, larger authorization and is intentionally not done.

## What is needed to close Step 1

- Either a Linux `hai-agent-runtime` from H Company (unlocks the contained-desktop sandbox and makes
  OpenShell the genuine boundary), **or** acceptance of the host packet-filter substitute plus:
  - build the Step 3 pass-through proxy,
  - author + verify the macOS pf rules (runtime → 127.0.0.1 only; proxy → provider only),
  - authorization to run a real task that streams live frames to Overshoot (pre-redaction), or defer
    the end-to-end run until Step 4 so frames are redacted before egress.

Per the blueprint's rule to stop and surface when a step cannot be satisfied without breaking a
constraint or exceeding authorization, execution stops here pending an operator decision.

## Resume — 2026-07-11 (operator instructed: "pick it up and finish step 1")

The operator's instruction resolves the pending decision: accept the host packet-filter
substitute and run the Step 1 task against Overshoot. State of the four blockers above:

1. **OpenShell cannot isolate the host runtime** — unchanged (platform fact). The accepted
   substitute is the ADR-0001 topology: proxy as sole egress + host packet filter.
2. **No Linux runtime** — unchanged; option (b) stays deferred.
3. **Egress boundary not stood up** — **cleared.** The pass-through proxy is built
   (`src/plva_proxy/proxy.py`, command `plva-proxy`): loopback-only bind, verbatim body relay
   (unknown keys preserved), credential injection from `.env`/environment, SSE streamed through,
   fail-closed on upstream failure (streams truncate rather than fabricate), logs carry byte
   counts/statuses/durations/exception class names only. Gates: 40 tests pass, ~93% coverage,
   ruff format/check, strict mypy, `uv lock --check`, `uv build` all green; live loopback smoke
   (health 200, key absent from logs) passed. The enforcing packet-filter rule set is authored at
   `docs/egress/pf-plva.anchor` (applying it needs operator sudo; an observational `lsof`
   check is the no-sudo Tier A alternative).
4. **Authorization to stream live frames** — granted by the operator's instruction to finish
   Step 1 (the runbook's privacy note restates what the run sends).

**Remaining to close Step 1:** the operator places the Overshoot key in `Codex_RUn/.env` and
executes `step-1-runbook.md` (§2 preflight → §3 task → §4 egress check → §5 record). No key was
present on this machine at build time, so the live run could not be executed yet.
