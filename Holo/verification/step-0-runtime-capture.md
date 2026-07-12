# Step 0 critical gate — real runtime screenshot transport

Date: 2026-07-11

Status: **PASS — the closed `hai-agent-runtime` transmits its screenshot through the
configurable base URL.** This closes the blueprint's critical `VERIFY` for Step 0. The
interception-proxy premise (§4) is therefore empirically valid: the frame is visible at the base
URL and can be rewritten in transit.

## What was proven

The blueprint's Step 0 `VERIFY (critical)` requires confirming that the closed runtime sends the
screenshot through its configurable base URL — otherwise the proxy could never see the frame. One
authorized, controlled, single-step run was pointed at the local loopback capture stub
(`plva-runtime-capture`, `127.0.0.1:18080`). No inference provider was contacted; the only network
the run touched was loopback.

The stub recorded **one** capture and returned a terminal, non-executable `answer`. The runtime
accepted that answer and terminated cleanly (`holo` exit `0`).

## Captured metadata (the only data retained)

The stub retains schema metadata only — never pixels, request bodies, transcripts, or keys:

| Field | Value |
|---|---|
| model | `Hcompany/Holo3-35B-A3B` |
| stream | `false` |
| message roles | `system`, `user` |
| image_count | `1` |
| image media type | `image/jpeg` |
| image byte length | ~654 KB |
| image dimensions | 1920 × 1243 |
| request keys | `chat_template_kwargs`, `logit_bias`, `max_tokens`, `messages`, `model`, `structured_outputs`, `temperature` |
| has_structured_outputs | `true` |
| has_tools | `false` |

A 1920-wide JPEG of ~654 KB is a genuine screen frame, not the 1×1 68-byte synthetic PNG used by
the provider probe — so this is a real host screenshot that traversed the base URL.

## Contract facts learned (previously assumptions)

- **Image encoding is JPEG**, not PNG. The stub already accepts `image/jpeg | image/png |
  image/webp`, so this is covered.
- **Actions are structured JSON in `message.content`**, not native OpenAI `tool_calls`: the request
  carries `structured_outputs` and **no** `tools` key (`has_tools: false`). This matches the paid
  synthetic provider probe (JSON in content) and resolves the earlier open question.
- The runtime's action envelope uses **plural `tool_calls`** (an array) inside the content object.
  The stub's canned answer was updated from singular `tool_call` to the plural array shape, and the
  real runtime parsed it and terminated — direct confirmation of the top-level response contract.
- The real request also includes `chat_template_kwargs`, `logit_bias`, `max_tokens`, and
  `temperature`. The future proxy must forward unknown request keys verbatim.

## Command used (safe to record)

```bash
# Terminal A — loopback stub (binds 127.0.0.1 only; contacts no provider)
.venv/bin/plva-runtime-capture --port 18080

# Terminal B — exactly one authorized single-step host capture
uv tool run --from holo-desktop-cli holo run \
  "Return an answer immediately without acting." \
  --base-url http://127.0.0.1:18080/v1 \
  --model Hcompany/Holo3-35B-A3B \
  --max-steps 1 --max-time-s 60 --no-kill-switch \
  --runs-dir /tmp/holo-step0-runs

# Confirm transport (metadata only)
curl -s http://127.0.0.1:18080/_probe/status   # captured:true, capture_count:1, image present
```

Notes established by source reading of the installed CLI/runtime:

- A custom `--base-url` needs **no** H Company login; the launcher even strips `HAI_API_KEY` from
  the runtime's environment when a base URL is set, so no key reaches the loopback stub.
- The runtime POSTs to `<base-url>/chat/completions` and may `GET <base-host>/health` (trailing
  `/v1` stripped). The stub now serves `/health` and `/v1/health` → `200` so the run does not block.
- `--max-steps 1` bounds the run to a single perceive → request cycle; the stub's answer is
  non-executable, so no clicks or keystrokes were dispatched to the desktop.

## Privacy handling (how §8.5 was honored)

Per the operator's explicit decision, a run artifact containing the frame may exist **locally** as
long as it is never egressed or committed. Handling:

- `--runs-dir` was pointed at an ephemeral `/tmp` path **outside** the git repo. The closed runtime
  wrote exactly one frame-bearing `events.jsonl` there (it exposes no switch to disable artifact
  writing — only to relocate it).
- That directory and all temp logs were **shredded immediately** after reading `/_probe/status`.
- Verified afterward: **no** `data:image;base64` blob and **no** stray large file exists anywhere in
  the repository. Nothing was read out of the frame log; only counts/sizes were inspected.
- The stub itself never persists, prints, or forwards pixels — only the metadata table above.

## Residual note for later steps

The closed runtime unconditionally writes `<runs-dir>/<agent_id>/events.jsonl`, and observation
events carry the base64 screenshot. There is no disable knob; only `--runs-dir` /
`HAI_AGENT_RUNTIME_RUNS_DIR` relocates it. For every future real run, point `--runs-dir` at an
ephemeral local path and shred it; treat `~/.holo/runs` as off-limits and never commit either.
