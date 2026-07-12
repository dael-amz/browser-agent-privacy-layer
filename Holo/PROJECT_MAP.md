# PLVA project map

This is the master guide to the active project structure. Keep it in sync whenever a file is
added, moved, removed, or given a materially different responsibility. Generated environments,
caches, build output, and privacy-sensitive runtime artifacts are intentionally omitted from the
tree.

Last updated: 2026-07-11

## Directory tree

```text
Hackathon/
├── BLUEPRINT.md                         Build order, architecture, constraints, and acceptance gates
├── README.md                            Short description of the outer repository
└── Holo/                                Active run (renamed from Codex_RUn on 2026-07-11)
    ├── PROJECT_MAP.md                   This living directory and status guide
    ├── README.md                        Active project overview and safety warning
    ├── run_step1.sh                     One-command live run (proxy + preflight + task + cleanup)
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
    │       ├── proxy.py                 Loopback interception proxy: relay + mutation hooks + viewer (fail-closed, SSE-safe)
    │       ├── redactor.py              Subprocess wrapper around the frozen plva-v2 CLI (fail-closed)
    │       └── runtime_capture.py       Loopback-only Holo screenshot-transport capture stub (+ /health)
    ├── tests/
    │   ├── test_contract_probe.py       Provider probe, failure, CLI, and safe-output tests
    │   ├── test_proxy.py                Relay fidelity, credential, SSE, fail-closed, and log-hygiene tests
    │   ├── test_proxy_hooks.py          Step 3 hook seam: mutation, SSE re-emit, and fail-closed tests
    │   ├── test_redaction.py            Redaction hook, FrameStore/viewer, and redactor wrapper tests
    │   └── test_runtime_capture.py      Capture validation, JSON/SSE, health, privacy, and bind tests
    ├── docs/
    │   ├── decisions/
    │   │   └── 0001-openshell-sec7-egress-topology.md   §7 topology decision (egress-isolation)
    │   └── egress/
    │       └── pf-plva.anchor           macOS pf rules: only the proxy role user reaches the provider
    └── verification/
        ├── step-0-stop.md               Historical stop caused by the original unavailable model
        ├── step-0-wrap-up.md            Earlier partial pass before this resume audit
        ├── step-0-resume-audit.md       Prior blockers, now annotated with their resolution
        ├── step-0-runtime-capture.md    PASS: real runtime screenshot traversed the base URL
        ├── step-1-status.md             §7 decision recorded; resume notes: ready to run
        ├── step-1-runbook.md            One-pass instructions to close Step 1 once the key exists
        ├── step-3-status.md             Interception hooks built; local verify PASS, live verify pending
        └── step-4-obscuring.md          Real obscuring via frozen v2 detector + /viewer; vault not built
```

## What each area is for

| Area | Responsibility |
|---|---|
| `BLUEPRINT.md` | Source of truth. Its step ordering and hard constraints override convenience. |
| `src/plva_proxy/` | Installable production/probe code. New proxy modules belong here, not beside `pyproject.toml`. |
| `tests/` | Automated acceptance and privacy-regression tests. The configured gate requires at least 80% coverage. |
| `docs/decisions/` | Architecture decision records (ADRs). One numbered file per resolved blueprint decision. |
| `verification/` | Human-readable evidence and decisions from each blueprint checkpoint. It must not contain captured frames, request bodies, transcripts, credentials, or vault values. |
| `.env.example` | Documents required variable names only. A real `.env` remains local and ignored. |
| `PROJECT_MAP.md` | Catch-up document for humans; update it in the same change as structural work. |

The old flattened `Holo/plva_proxy/` location now contains only ignored Python cache data.
The actual source files are under `Holo/src/plva_proxy/`; IDE tabs pointing at the flattened
path are stale.

## Current blueprint checkpoint

**Step 0 is COMPLETE** (including the critical transport gate). **Step 1 is READY TO RUN**: the §7
decision is resolved, the pass-through proxy (the ADR's sole-egress component, functionally the
Step 3 pass-through core built early) is implemented and gated, the pf egress rule set is authored,
and `verification/step-1-runbook.md` closes the step in one pass — it waits only on the operator's
Overshoot key in `.env`. **Step 3 is BUILT and locally verified** (2026-07-11, on operator
instruction): the proxy now exposes a request/response mutation hook seam (`--hook test` enables
the blueprint's test hooks; default remains pass-through), with SSE responses buffered,
reconstructed, mutated, and re-emitted under a response hook, and every hook/parse failure failing
closed. Its live verify (the Step 1/2 task running unchanged through pass-through and hook modes)
rides on the pending live run. **Step 4 is PARTIAL** (2026-07-11): real *obscuring* works — the
frozen `plva-v2-baseline/` detector (pulled from a separate clone, not vaulted in git) is wired into the request-hook seam
(`--redact`, fail-closed, threadpooled) with a memory-only loopback `/viewer` showing the
obscured frames the model sees — but the report is geometry-only, so the vault / placeholders /
resolution / history-scrub half of Step 4 is not built. See `verification/step-4-obscuring.md`.
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
- The automated gate is 40 passing tests with ~93% total coverage; formatting, Ruff, strict mypy,
  lock validation, sdist, and wheel checks pass.
- **Pass-through proxy built and gated** (2026-07-11 resume): loopback-only, verbatim body relay,
  credential injection, SSE streamed through, fail-closed, privacy-safe logs; live loopback smoke
  passed. Enforcing pf rules authored in `docs/egress/pf-plva.anchor` (operator sudo to apply).

Active blockers / open items:

- **Step 1 live run awaits the operator's Overshoot key** in `Holo/.env` (`API_KEY=...`),
  then one pass of `verification/step-1-runbook.md`. Live-frame streaming for this run was
  authorized by the operator on 2026-07-11 ("finish step 1").
- The closed runtime writes frame-bearing `events.jsonl` with no disable knob; every real run must
  relocate `--runs-dir` to an ephemeral local path and shred it. `~/.holo/runs` is off-limits.
- The outer Git repository still represents the move into `Holo/` as deleted tracked root
  files plus an untracked directory. That user-created repository reorganization has not been
  staged, committed, or reversed.

## Installed project commands

| Command | Purpose |
|---|---|
| `plva-live` | Continuous local screen redaction demo: capture → redact → `http://127.0.0.1:18082/viewer`; no provider, no key. |
| `plva-probe` | Run the live synthetic Overshoot contract probe when `API_KEY` is supplied. |
| `plva-proxy` | Loopback interception proxy to Overshoot; reads `API_KEY` from env or `./.env`; `--hook test` enables the Step 3 test hooks, `--hook-image <path>` swaps every outbound screenshot for a static image, `--redact plva-v2-baseline` obscures screenshots through the frozen v2 detector and serves them at `/viewer` (default: pass-through). |
| `plva-runtime-capture` | Start the metadata-only capture stub on `127.0.0.1`; it never contacts a provider. |

`plva-proxy` is the runtime's sole endpoint and the sole provider egress (Step 1/ADR-0001 role),
now with the Step 3 mutation seam (`Hooks`): request hooks rewrite body + upstream headers,
response hooks rewrite the completion (JSON and SSE). Step 4 plugs redaction and placeholder
resolution into that seam.

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
