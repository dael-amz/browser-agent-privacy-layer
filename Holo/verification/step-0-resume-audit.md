# Step 0 resume audit

Date: 2026-07-11

Status: **RESOLVED — the real host capture was authorized and run; the critical gate passed. See
`step-0-runtime-capture.md`. The blockers below are retained as history.**

> Resolution (2026-07-11): The operator authorized one controlled loopback capture and accepted
> that a run artifact may contain the frame **as long as it stays local** (never egressed, never
> committed). The single-step capture confirmed the closed runtime transmits its screenshot through
> the configurable base URL (`image_count: 1`, 1920×1243 JPEG). The frame-bearing `events.jsonl`
> was written to an ephemeral `/tmp` runs-dir and shredded immediately; no frame reached the repo.
> The §8.5 conflict is handled operationally: no disable knob exists, so every real run relocates
> artifacts to an ephemeral local path and shreds them. The §7 decision is recorded in
> `docs/decisions/0001-openshell-sec7-egress-topology.md`; Step 1 execution status is in
> `step-1-status.md`.

---

## Historical blockers (superseded by the resolution above)

Status at the time: **BLOCKED — the local capture harness is ready, but the real host capture has
not been authorized or run. A closed-runtime screenshot-log conflict must also be resolved.**

## Repaired baseline

The saved run had been flattened while moving it into `Holo`: the package no longer matched
the declared `src/` layout and the declared README was absent. The package layout and README are
now restored. The premature `plva-proxy` console script was removed because Step 3 has not been
built; the existing provider probe and the Step 0 runtime-capture probe are the only installed
commands.

The earlier wrap-up's `7 passed` result did not satisfy the configured coverage gate. The provider
probe now has mocked JSON, SSE, authentication-boundary, CLI-success, and privacy-safe failure
tests. Empty or truncated SSE responses fail closed.

Current local verification:

```text
pytest: 23 passed; 91% total coverage (80% required)
ruff check: clean
ruff format --check: clean
mypy --strict: clean
sdist and wheel: built; wheel contains the plva_proxy package and both probe entry points
```

## Capture harness

`plva-runtime-capture` binds only to `127.0.0.1`. It:

- accepts only the selected `Hcompany/Holo3-35B-A3B` model;
- requires at least one decodable inline PNG/JPEG/WebP screenshot;
- retains only request keys, roles, model, streaming flag, image media type, byte length, and
  dimensions in memory;
- never retains, prints, persists, or forwards request text or image bytes;
- supports JSON and SSE responses;
- returns only a terminal, non-executable `answer` action; and
- disables HTTP access logs.

A live synthetic loopback smoke test captured one 68-byte 1×1 PNG and returned the expected answer.
No provider or other network endpoint was contacted.

## Holo state

The official `holo-desktop-cli` v0.0.2 is installed. Its managed Darwin ARM64 runtime v0.1.8 is
present, and the CLI exposes the required `--model`, `--base-url`, `--max-steps`, `--max-time-s`,
`--runs-dir`, `--fake`, and `--no-kill-switch` controls. No real runtime session was launched by
this resume pass.

## Hard-constraint conflict discovered

Holo's official paths documentation states that every run writes `events.jsonl` and observation
events can contain base64 screenshots. The closed runtime exposes a run-directory override but no
documented disable-artifacts option. This conflicts with the blueprint's rule that logs never
contain frames, independently of whether the local capture stub or future proxy handles egress
safely.

Do not run the real screenshot probe until both conditions are met:

1. the user explicitly authorizes one controlled host capture with unrelated windows closed; and
2. a fake-runtime experiment or vendor-supported setting proves that run artifacts can be disabled
   without breaking the runtime.

If run artifacts cannot be disabled, stop: deleting a screenshot log after the run would not
satisfy the blueprint's “never logged” constraint.

## Repository-state caveat

The outer Git repository still sees its formerly tracked root project files as deleted and
`Holo/` as untracked. This resume pass intentionally did not discard, relocate, stage, or
commit that user-created move.
