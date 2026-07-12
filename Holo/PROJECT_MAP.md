# PLVA project map

This is the master guide to the active project structure. Keep it in sync whenever a file is
added, moved, removed, or given a materially different responsibility. Generated environments,
caches, build output, and privacy-sensitive runtime artifacts are intentionally omitted from the
tree.

Last updated: 2026-07-12

## Directory tree

```text
Hackathon/
├── BLUEPRINT.md                         Build order, architecture, constraints, and acceptance gates
├── README.md                            Short description of the outer repository
└── Holo/                                Active run (renamed from Codex_RUn on 2026-07-11)
    ├── PROJECT_MAP.md                   This living directory and status guide
    ├── README.md                        Active project overview and safety warning
    ├── run_step1.sh                     One-command live run (proxy + preflight + task + cleanup)
    ├── run_mediator_sandbox.sh          NemoClaw/OpenShell mediator stack: up (with in-sandbox deny-test gate) / status / down
    ├── redactor-worker/                 Persistent parallel WebGPU/WASM redaction worker source
    │   ├── bin/redactor-worker.mjs      Private loopback browser/IPC supervisor
    │   ├── src/worker.js                Warm parallel visual + OCR pipeline and PNG renderer
    │   ├── vite.config.js               Derived build against the separate frozen baseline
    │   ├── package.json                 Browser inference/build dependencies
    │   └── package-lock.json            Reproducible Node dependency lock
    ├── coreml-redactor/                 Separate accelerated Core ML hybrid redaction backend
    │   ├── src/plva_coreml/coreml_session.py Core ML execution-provider boundary and ANE policy
    │   ├── src/plva_coreml/model_cache.py Content-addressed fixed-shape model derivatives
    │   ├── src/plva_coreml/visual_ane.py Static-shape visual-model session
    │   ├── src/plva_coreml/visual_redactor.py Preprocess, decode, NMS, and mask rendering
    │   ├── src/plva_coreml/ocr.py        Core ML RapidOCR detection/recognition + findings
    │   ├── src/plva_coreml/semantics.py Structured rules + Core ML Rampart classification
    │   ├── src/plva_coreml/hybrid.py     Parallel visual/OCR orchestration, fusion, and render
    │   ├── src/plva_coreml/vision.py     Persistent native Apple Vision OCR client
    │   ├── src/plva_coreml/vision_hybrid.py Fast/accurate Vision cascade + Core ML fusion
    │   ├── src/plva_coreml/worker.py     Proxy-compatible memory-pipe worker protocol
    │   ├── src/plva_coreml/native/vision_ocr_worker.swift Native Vision request loop
    │   ├── src/plva_coreml/live.py       Local screen/fixture loop and localhost viewer
    │   ├── src/plva_coreml/probe.py      Synthetic-fixture latency and parity probe
    │   ├── pyproject.toml                Isolated Core ML probe dependencies
    │   └── README.md                     Results, limitations, and native-worker seam
    ├── plva-v2-baseline/                Frozen v2 detector harness — NOT vaulted in git; pulled from a
    │                                    separate clone and fed via `--redact <path>` (gitignored, AGPL, dev-only)
    ├── pyproject.toml                   Package metadata, dependencies, commands, and quality gates
    ├── uv.lock                          Exact reproducible Python dependency lock
    ├── .python-version                  Required Python version for local tooling
    ├── .env.example                     Safe environment-variable template; contains no key value
    ├── .gitignore                       Secret, cache, build, and privacy-artifact exclusions
    ├── src/
    │   └── plva_proxy/
    │       ├── __init__.py              Python package marker
    │       ├── contract_probe.py        Privacy-safe Overshoot model/JSON/SSE contract probe
    │       ├── live.py                  Continuous local capture→redact→viewer loop (no upstream)
    │       ├── local_llm.py             Loopback-pinned fail-closed client for the sandboxed local LLM + no-egress audit
    │       ├── mediator.py              §7 local LLM mediator: approval verdicts, trace watchdog, and CLI demos
    │       ├── semantic_executor.py     Step 13(B) placeholder-preserving sort/select via the local LLM (token-only returns)
    │       ├── tool_channel.py          Step 13/7 bridge: ⟦PLVA_TOOL⟧ marker channel, deterministic sort (A), mediator auto-approval, watchdog halt
    │       ├── privacy.py               Session vault, chip painter, resolver, and history scrub
    │       ├── proxy.py                 Loopback interception proxy: relay + mutation hooks + memory-only sent-frame audit viewer (fail-closed, SSE-safe)
    │       ├── providers.py             Overshoot and H Company endpoint/model/key presets
    │       ├── redactor.py              Persistent accelerated worker client + frozen CLI fallback
    │       └── runtime_capture.py       Loopback-only Holo screenshot-transport capture stub (+ /health)
    ├── tests/
    │   ├── test_contract_probe.py       Provider probe, failure, CLI, and safe-output tests
    │   ├── test_accelerated_redactor.py Persistent protocol, cache, lifecycle, and failure tests
    │   ├── test_proxy.py                Relay fidelity, credential, SSE, fail-closed, and log-hygiene tests
    │   ├── test_proxy_hooks.py          Step 3 hook seam: mutation, SSE re-emit, and fail-closed tests
    │   ├── test_local_llm.py            Loopback pinning, JSON extraction, leak scan, and egress-audit tests
    │   ├── test_tool_channel.py         Marker channel, deterministic sort, auto-approval, and watchdog-halt tests
    │   ├── test_mediator.py             Criteria validation, fail-closed verdicts, watchdog trigger, and CLI tests
    │   ├── test_semantic_executor.py    Token-only contract, retry/fail-closed, and validation tests
    │   ├── test_privacy.py              Vault, chip, scrub, resolution, and detector-stub tests
    │   ├── test_privacy_integration.py  Full request/response privacy loop and SSE tests
    │   ├── test_redaction.py            Redaction hook, FrameStore/viewer, and redactor wrapper tests
    │   └── test_runtime_capture.py      Capture validation, JSON/SSE, health, privacy, and bind tests
    ├── config/
    │   ├── privacy-policy.json          Step 6 editable per-class safety levels
    │   ├── mediator-criteria.json       User-written rules for approval-gated classes + trace-review thresholds
    │   └── openshell-mediator-policy.yaml  Zero-egress OpenShell policy for the mediator sandbox (§7)
    ├── docs/
    │   ├── decisions/
    │   │   └── 0001-openshell-sec7-egress-topology.md   §7 topology decision (egress-isolation)
    │   ├── local-llm-runbook.md         How to serve, sandbox, and verify the local Nemotron mediator/executor
    │   └── egress/
    │       └── pf-plva.anchor           macOS pf rules: only the proxy role user reaches the provider
    └── verification/
        ├── step-0-stop.md               Historical stop caused by the original unavailable model
        ├── step-0-wrap-up.md            Earlier partial pass before this resume audit
        ├── step-0-resume-audit.md       Prior blockers, now annotated with their resolution
        ├── step-0-runtime-capture.md    PASS: real runtime screenshot traversed the base URL
        ├── step-1-status.md             §7 decision plus completed live acceptance status
        ├── step-1-runbook.md            One-pass instructions to close Step 1 once the key exists
        ├── step-3-status.md             Interception hooks built; local and live verification evidence
        ├── step-4-accelerated-redaction.md  Persistent parallel GPU worker evidence and benchmark
        ├── step-4-obscuring.md          Real obscuring via frozen v2 detector + /viewer
        ├── step-5-privacy-core.md       PASS: vault, chips, resolution, and history scrub
        ├── cua-placeholder-optimization.md  Multi-step token reuse, cache, readiness, and residual gates
        ├── cua-live-acceptance.md       PASS: real Holo two-step placeholder reuse and local submission
        └── local-llm-mediator-executor.md  Step 7 mediator automation + Step 13(B) executor live evidence
```

## What each area is for

| Area | Responsibility |
|---|---|
| `BLUEPRINT.md` | Source of truth. Its step ordering and hard constraints override convenience. |
| `src/plva_proxy/` | Installable production/probe code. New proxy modules belong here, not beside `pyproject.toml`. |
| `redactor-worker/` | Derived persistent browser worker source; generated `dist/` and `node_modules/` remain ignored. |
| `coreml-redactor/` | Isolated ANE feasibility backend; not connected to outbound redaction until positive-detection parity passes. |
| `tests/` | Automated acceptance and privacy-regression tests. The configured gate requires at least 80% coverage. |
| `docs/decisions/` | Architecture decision records (ADRs). One numbered file per resolved blueprint decision. |
| `verification/` | Human-readable evidence and decisions from each blueprint checkpoint. It must not contain captured frames, request bodies, transcripts, credentials, or vault values. |
| `.env.example` | Documents required variable names only. A real `.env` remains local and ignored. |
| `PROJECT_MAP.md` | Catch-up document for humans; update it in the same change as structural work. |

The old flattened `Holo/plva_proxy/` location now contains only ignored Python cache data.
The actual source files are under `Holo/src/plva_proxy/`; IDE tabs pointing at the flattened
path are stale.

## Current blueprint checkpoint

**Step 0 is COMPLETE** (including the critical transport gate). **Step 1 is COMPLETE**: the §7
decision is resolved, the pass-through proxy (the ADR's sole-egress component, functionally the
Step 3 pass-through core built early) is implemented and gated, and a real Holo/H Company run
completed the controlled two-step PLVA acceptance fixture with independent local verification.
The pf egress rule set is authored; installing that additional host-wide layer remains an optional
administrator hardening step. **Step 3 is BUILT and live verified** (2026-07-12): the proxy exposes
a request/response mutation hook seam (`--hook test` enables
the blueprint's test hooks; default remains pass-through), with SSE responses buffered,
reconstructed, mutated, and re-emitted under a response hook, and every hook/parse failure failing
closed. **Step 4 is PARTIAL** (2026-07-11): real *obscuring* works through
a persistent accelerated worker by default. Its visual model uses WebGPU on supported hardware,
OCR runs concurrently through a separate WASM runtime, sessions stay warm during active CUA
bursts and release after 60 idle seconds, and exact repeated frames use a bounded
redacted-output-only memory cache. An opt-in native Apple Vision/Core ML engine now runs through
the same fail-closed CUA hook, emits OCR text and exact PII spans at memory-only
`/viewer/findings`, and measured 113–125 ms warm on the synthetic ATS fixture. **Step 5 is
COMPLETE** (2026-07-12): those OCR findings now feed stable nonce-namespaced placeholder chips and
a session-only vault; executed fields resolve locally; request history is scrubbed by plain match
plus warm Core ML Rampart. Synthetic Holo cooperation and end-to-end resolution passed. See
`verification/step-5-privacy-core.md`.
The CUA/placeholder optimization pass additionally binds each launch to its own proxy instance,
supports issued-token reuse across later frames, hardens action grammar handling, and removes the
frame/history cache-thrash paths. Its local integration gate and controlled live PLVA-on acceptance
pass. The current detector is release eligible; the development-only
language in the frozen July 11 v2 baseline describes historical provenance, not current status.
See `verification/cua-placeholder-optimization.md`.
The controlled PLVA-on live acceptance now passes against the real Holo runtime and H Company
provider: it navigated a two-step loopback form, reused an issued email token after the value left
the screen, resolved locally, and submitted the exact value. Independent server verification—not
runtime exit—established success. See `verification/cua-live-acceptance.md`.
Step 2 (Overshoot latency measurement) has not started.

Completed evidence:

- Overshoot advertised `Hcompany/Holo3-35B-A3B` as ready.
- Synthetic JSON and SSE provider contracts were probed without printing response content.
- The relocated package was repaired and now installs/builds correctly.
- The local capture stub passed a synthetic live smoke test and does not forward or retain frames.
- **Step 0 critical gate PASSED**: one authorized single-step run proved the closed
  `hai-agent-runtime` sends its screenshot (1920×1243 JPEG) through the configurable base URL to the
  loopback stub. Actions are structured JSON in `message.content` (`structured_outputs`, no
  `tools`), envelope uses plural `tool_calls`. Frame stayed local and was shredded; nothing reached
  the repo. See `verification/step-0-runtime-capture.md`.
- **§7 decision recorded** in `docs/decisions/0001-openshell-sec7-egress-topology.md`.
- The main automated gate is 178 passing tests with at least 80% coverage; the Core ML package has
  17 passing tests. Formatting, Ruff, strict mypy,
  lock validation, sdist, and wheel checks pass.
- **Pass-through proxy built and gated** (2026-07-11 resume): loopback-only, verbatim body relay,
  credential injection, SSE streamed through, fail-closed, privacy-safe logs; live loopback smoke
  passed. Enforcing pf rules authored in `docs/egress/pf-plva.anchor` (operator sudo to apply).

Active hardening / open items:

- A broader repeated-task PLVA-off/on corpus is still useful for success-rate and p50/p95 trend
  tracking. The production acceptance path itself is no longer waiting on a provider key or first
  live run.
- The closed runtime writes frame-bearing `events.jsonl` with no disable knob; every real run must
  relocate `--runs-dir` to an ephemeral local path and shred it. `~/.holo/runs` is off-limits.
- Installing the optional macOS `pf` anchor requires an administrator. The tested runtime-level
  monitor already fails closed on non-loopback runtime connections and allows the required local
  proxy and Agent API traffic.

## Installed project commands

| Command | Purpose |
|---|---|
| `plva-live` | Continuous local capture through the persistent accelerated redactor → `http://127.0.0.1:18082/viewer`; no provider or key. |
| `plva-probe` | Run the live synthetic Overshoot contract probe when `API_KEY` is supplied. |
| `plva-proxy` | Loopback proxy; `--provider` selects Overshoot or H Company; `--redact-engine vision` selects native Vision/Core ML, `accelerated` selects WebGPU/WASM, and `baseline` retains the oracle; `/viewer` and `/viewer/findings` are memory-only. |
| `plva-runtime-capture` | Start the metadata-only capture stub on `127.0.0.1`; it never contacts a provider. |
| `plva-mediator` | Probe the loopback local LLM (reachability + no-egress audit) and run synthetic approval/trace/sort demos; see `docs/local-llm-runbook.md`. |

`plva-proxy` is the runtime's sole endpoint and the sole provider egress (Step 1/ADR-0001 role),
now with the Step 3 mutation seam (`Hooks`): request hooks rewrite body + upstream headers,
response hooks rewrite the completion (JSON and SSE). Step 4 plugs redaction and placeholder
resolution into that seam. Its loopback viewer keeps a bounded memory-only ledger of each
post-redaction image included in an upstream request, with value-free delivery metadata and
selectable frame history; `PLVA_AUDIT=1` keeps that buffer available after a CUA task exits.

## Local verification commands

Run from `Holo/`:

```bash
~/.local/bin/uv sync --frozen
.venv/bin/pytest -q
.venv/bin/ruff format --check src tests
.venv/bin/ruff check .
.venv/bin/mypy src tests
~/.local/bin/uv lock --check
~/.local/bin/uv build --no-sources
```

Ignored/generated paths such as `.venv/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`,
`__pycache__/`, `.coverage`, and `dist/` are rebuildable and are not part of the maintained source
map. Privacy-sensitive paths such as `captures/`, `screenshots/`, `transcripts/`, and `vault/` are
ignored and must never be committed.
