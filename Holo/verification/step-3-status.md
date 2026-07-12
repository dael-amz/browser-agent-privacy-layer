# Step 3 status — interception proxy with mutation hooks

Date: 2026-07-11

Status: **BUILT; local verify PASS. The live-task portion of the blueprint's Verify rides on the
pending Step 1/2 live run.** Built on operator instruction ("implement step 3"); the OpenShell
sandboxing of the proxy discussed the same day was explicitly deferred by the operator.

## What Step 3 asks for

A local OpenAI-compatible proxy between the runtime and Overshoot that can apply arbitrary
modifications to both directions, transparently: pass-through mode, a test hook that can mutate
request and response, and streamed-response support (§8.7).

## Built

The Step 1 pass-through proxy (`src/plva_proxy/proxy.py`) gained the interception seam:

- **`Hooks` dataclass** — `on_request(body, upstream_headers)` and `on_response(completion)`;
  both `None` by default, which is byte-identical pass-through (Step 1 behavior unchanged).
- **Test hooks** (`--hook test` on `plva-proxy`, or `TEST_HOOKS` in code), per the blueprint:
  the request hook tags the upstream request with `x-plva-hook: request`; the response hook
  no-op-rewrites every action (decode `message.content` JSON → re-encode), exercising the exact
  parse → mutate → re-serialize path Step 4's placeholder resolution will use. Hooked responses
  carry `x-plva-hook: response` back to the caller so mutation is observable.
- **Streaming under a response hook (§8.7):** the SSE stream is fully buffered, reconstructed
  into one completion (`_assemble_sse_completion`), mutated, and re-emitted as a minimal SSE
  stream ending in `[DONE]` (`_sse_bytes`). A truncated stream (no `[DONE]`), an unparseable
  event, or a native `tool_calls` delta raises and nothing is forwarded. Without a response hook,
  SSE passes through incrementally exactly as before.
- **Fail-closed everywhere (§8.1):** any hook or parse failure on either leg → 502, nothing
  forwarded, log carries the exception class name only. Hooks apply only to `/chat/completions`
  with upstream status 200; provider errors relay verbatim.

## Addendum (same day): static image replacement hook

`--hook-image <path>` (also `PLVA_HOOK_IMAGE=<path> ./run_step1.sh`) builds a request hook that
replaces **every** outbound screenshot with a static PNG/JPEG/WebP, validated once at startup.
Semantics are deliberately fail-closed: if a hooked request contains no screenshot to replace,
nothing is forwarded (502) — a request meant to be scrubbed can never leave with its original
frame. Composes with `--hook test`. This doubles as a §8.2 rehearsal and as a way to run the full
live loop against Overshoot **without a single real desktop pixel egressing**.

Smoke evidence: a synthetic request carrying a 1×1 PNG went through
`plva-proxy --hook-image <64×64 png> --upstream <capture-stub>`; the stub's recorded metadata
showed the received image was 64×64 (the static file), and an imageless request was refused 502.

## Local verify evidence (no provider contact, no frames)

- Real-socket end-to-end: `plva-proxy --hook test --upstream <capture-stub>` with a synthetic
  runtime-shaped request (1×1 PNG). JSON mode returned 200 + `x-plva-hook: response` with the
  action content re-serialized; SSE mode returned `text/event-stream`, the re-emitted events, and
  a terminal `[DONE]`.
- Automated: hook tests in `tests/test_proxy_hooks.py` (mutation both directions, SSE re-emit,
  all fail-closed paths, `--hook` wiring, log hygiene) alongside the existing relay suite; full
  gate green (pytest + coverage, ruff format/check, strict mypy, `uv lock --check`, `uv build`).

## Remaining for the blueprint's full Verify

With the operator's key in `.env` (see `step-1-runbook.md`):

1. Pass-through: `./run_step1.sh` — the task completes unchanged (also closes Step 1).
2. Hook mode: start `plva-proxy --hook test` and run the same task; it should still complete, and
   the proxy log shows `hooks=on` relay lines (the injected modification is observable via the
   `x-plva-hook` headers and re-serialized actions).
