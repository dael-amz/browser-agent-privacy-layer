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
└── Codex_RUn/                           Active midway run being evaluated and continued
    ├── PROJECT_MAP.md                   This living directory and status guide
    ├── README.md                        Active project overview and safety warning
    ├── run_step1.sh                     One-command Step 1 run (proxy + preflight + task + cleanup)
    ├── pyproject.toml                   Package metadata, dependencies, commands, and quality gates
    ├── uv.lock                          Exact reproducible Python dependency lock
    ├── .python-version                  Required Python version for local tooling
    ├── .env.example                     Safe environment-variable template; contains no key value
    ├── .gitignore                       Secret, cache, build, and privacy-artifact exclusions
    ├── src/
    │   └── plva_proxy/
    │       ├── __init__.py              Python package marker
    │       ├── contract_probe.py        Privacy-safe Overshoot model/JSON/SSE contract probe
    │       ├── proxy.py                 Loopback pass-through relay to Overshoot (fail-closed, SSE-safe)
    │       └── runtime_capture.py       Loopback-only Holo screenshot-transport capture stub (+ /health)
    ├── tests/
    │   ├── test_contract_probe.py       Provider probe, failure, CLI, and safe-output tests
    │   ├── test_proxy.py                Relay fidelity, credential, SSE, fail-closed, and log-hygiene tests
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
        └── step-1-runbook.md            One-pass instructions to close Step 1 once the key exists
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

The old flattened `Codex_RUn/plva_proxy/` location now contains only ignored Python cache data.
The actual source files are under `Codex_RUn/src/plva_proxy/`; IDE tabs pointing at the flattened
path are stale.

## Current blueprint checkpoint

**Step 0 is COMPLETE** (including the critical transport gate). **Step 1 is READY TO RUN**: the §7
decision is resolved, the pass-through proxy (the ADR's sole-egress component, functionally the
Step 3 pass-through core built early) is implemented and gated, the pf egress rule set is authored,
and `verification/step-1-runbook.md` closes the step in one pass — it waits only on the operator's
Overshoot key in `.env`. Steps 2–4 have not started (Step 3's hook/mutation surface is still open).

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

- **Step 1 live run awaits the operator's Overshoot key** in `Codex_RUn/.env` (`API_KEY=...`),
  then one pass of `verification/step-1-runbook.md`. Live-frame streaming for this run was
  authorized by the operator on 2026-07-11 ("finish step 1").
- The closed runtime writes frame-bearing `events.jsonl` with no disable knob; every real run must
  relocate `--runs-dir` to an ephemeral local path and shred it. `~/.holo/runs` is off-limits.
- The outer Git repository still represents the move into `Codex_RUn/` as deleted tracked root
  files plus an untracked directory. That user-created repository reorganization has not been
  staged, committed, or reversed.

## Installed project commands

| Command | Purpose |
|---|---|
| `plva-probe` | Run the live synthetic Overshoot contract probe when `API_KEY` is supplied. |
| `plva-proxy` | Loopback pass-through relay to Overshoot; reads `API_KEY` from env or `./.env`. |
| `plva-runtime-capture` | Start the metadata-only capture stub on `127.0.0.1`; it never contacts a provider. |

`plva-proxy` is currently pass-through only: it is the runtime's sole endpoint and the sole
provider egress (Step 1/ADR-0001 role). Step 3 adds its mutation/test hooks; Step 4 adds
redaction and placeholder resolution.

## Local verification commands

Run from `Codex_RUn/`:

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
