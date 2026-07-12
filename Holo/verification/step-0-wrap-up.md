# Step 0 wrap-up — external contract probe

Date: 2026-07-11

Overall status: **PARTIAL PASS — provider contract verified; closed-runtime image
interception remains blocked and unverified.** Step 1 has not started.

## Verified

- User-selected model: `Hcompany/Holo3-35B-A3B`.
- A live unauthenticated `GET https://api.overshoot.ai/v1beta/models`
  reported that exact model with status `ready`.
- The local `.env` supplied `API_KEY` through the process environment. The key
  was not printed, persisted, passed in argv, or copied into the repository.
- A paid non-streamed request containing only a synthetic one-pixel PNG returned:
  - model: `Hcompany/Holo3-35B-A3B`
  - object shape: `choices`, `created`, `id`, `model`, `object`, `overshoot`, `usage`
  - assistant message keys: `content`, `function_call`, `role`, `tool_calls`
  - action/output mode: JSON text in `message.content`, not native tool calls
- A paid `stream: true` request returned `text/event-stream` with seven JSON data
  events, `role`/`content` deltas, and one terminal `[DONE]`. It did not use
  native tool-call deltas.
- The official HoloDesktop CLI executes correctly in an isolated `uv tool run`
  environment. Its current `run` command exposes `--model`, `--base-url`,
  `--max-steps`, `--max-time-s`, `--fake`, and `--no-kill-switch`.
- The committed probe emits schema metadata only; it never emits response
  content, request bodies, images, or authorization headers.

## Verification evidence

Focused tests:

```text
tests/test_contract_probe.py: 7 passed
mypy src/plva_proxy/contract_probe.py: success, no issues
ruff: clean after import-order formatting
```

The TDD checkpoints are:

- `7fd01b7` — RED: missing `plva_proxy.contract_probe`
- `b3484e6` — GREEN: JSON and SSE contract implementation

## Critical gate not verified

The blueprint requires an empirical capture proving that the closed
`hai-agent-runtime` sends its screenshot through the configured base URL. That
capture was intentionally not run against the host desktop.

The safer contained-desktop plan cannot currently execute because the public
HoloDesktop v0.0.2 runtime manifest publishes only Darwin ARM64 and Windows
x86_64 binaries. OpenShell on this Apple Silicon Mac uses a Linux ARM64 Docker
sandbox, and no public Linux ARM64 `hai-agent-runtime` exists. A Darwin binary
cannot run in that sandbox.

There is also an unresolved schema discrepancy that only the capture can settle:
the public Holo3 model guide documents singular `tool_call`, while the current
HoloDesktop runtime contract reports a plural `tool_calls` list. The proxy must
not implement action rewriting until the exact runtime body is captured.

## Safe continuation requirement

One of the following is required before the critical gate can pass:

1. H Company supplies a trusted Linux ARM64 runtime artifact, SHA-256 digest,
   license, and proof of X11/Xvfb support for a contained OpenShell desktop; or
2. a separately authorized, one-step host capture is performed against a
   local-only stub that returns the non-executable `answer` action and never
   forwards or logs the screenshot.

Until then, no claim is made that HoloDesktop traffic interception, OpenShell
isolation, or end-to-end desktop execution works.

## References

- [Overshoot models](https://docs.overshoot.ai/models)
- [HoloDesktop CLI](https://github.com/hcompai/holo-desktop-cli)
- [Holo3 agent-loop schema](https://hub.hcompany.ai/agent-loop)
- [Holo runtime installer](https://github.com/hcompai/holo-desktop-cli/blob/main/src/holo_desktop/agent_client/runtime_install.py)
- [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell)
