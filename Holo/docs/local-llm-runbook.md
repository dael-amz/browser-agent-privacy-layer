# Local LLM runbook — mediator (§7) and semantic executor (Step 13 B)

The mediator and the semantic executor consult a **small local model that may see vault
cleartext**, so §7's rule applies: it must run **fully local with zero network egress**. The
code side is `src/plva_proxy/local_llm.py` (loopback-pinned, fail-closed client),
`mediator.py` (approval verdicts + trace watchdog), and `semantic_executor.py`
(token-only sort/select). None of it is bridged into the proxy yet; the plug points are
documented at the bottom.

## 1. Serve the model (one-time download, then fully offline)

**Recommended: llama.cpp `llama-server` with a local ~4B GGUF.** Unlike a daemon with its
own pull/telemetry machinery, `llama-server` holds one local file and opens no outbound
sockets, which is exactly what the lsof audit verifies. The models below live under
`models/` (gitignored).

**Sizing rule for this 16 GB MacBook Air:** stay in the 4B class (~2.5 GB weights) and cap
the context — the mediator/executor prompts are short, and an uncapped context allocates a
needlessly large KV cache. The 8B-class model earlier caused system-wide memory pressure
alongside a normal workload; treat it as ≥24 GB-machine material. Stop the server when it
is not needed.

```bash
# Recommended — mediator-grade judgment in a laptop-friendly footprint:
llama-server -m models/qwen3-4b-instruct-2507-q4_k_m.gguf \
  --host 127.0.0.1 --port 8555 --ctx-size 4096 --no-webui

# NVIDIA alternative — fine for the semantic executor, too weak for approval judgment
# (live-tested: sorts/selects correctly under the value-enum grammar, but falsely denies
# legitimate approval requests):
llama-server -m models/nemotron-mini-4b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 --port 8555 --ctx-size 4096 --no-webui

# 8B-class models (e.g. Llama-3.1-Nemotron-Nano-8B) are ≥24 GB-RAM material; the GGUF
# was deleted from this machine after it caused a crash-level swap storm. Re-download
# only on a bigger machine.
```

Defaults expected by the client: `http://127.0.0.1:8555/v1`; the model-name field is
advisory for llama-server (it serves whatever is loaded). Override with
`PLVA_LOCAL_LLM_URL` / `PLVA_LOCAL_LLM_MODEL` or `plva-mediator --url/--model`.

**Accepted for development: Ollama** (`ollama pull nemotron-mini`, then
`PLVA_LOCAL_LLM_URL=http://127.0.0.1:11434/v1`). Caveat: the Ollama daemon itself can reach
the network (model pulls, updates), so the no-egress audit will flag it whenever it does —
do not use it as the trusted cleartext mediator in a real run.

## 2. Sandboxing (per ADR-0001)

- **Linux hosts:** launch the server inside OpenShell with deny-all egress and verify with
  `openshell policy prove`; reach it via `openshell forward service` on loopback. Only trust
  it if enforcement (not observation mode) is confirmed by a deny-test from inside.
- **This macOS host:** OpenShell cannot enforce, so the substitute is (a) the client refuses
  any non-loopback URL and never follows redirects, and (b) `plva-mediator probe` runs a
  point-in-time `lsof` audit that flags any non-loopback bind or remote connection of the
  serving process. That audit is best-effort evidence, not a proof — same status as the
  pf-anchor layer in ADR-0001.

## 3. Verify

```bash
uv run plva-mediator probe          # reachability + no-egress audit (exit 0 = clean)
uv run plva-mediator demo-approval  # synthetic API_KEY approval verdict
uv run plva-mediator demo-trace     # synthetic misbehavior trace -> continue/warn/halt
uv run plva-mediator demo-sort      # synthetic token sort + leak scan of the observation
```

All demos use fixed synthetic values; nothing sensitive is read, sent, or logged.

## 4. Fail-closed semantics (what callers may rely on)

| Surface | On model outage / malformed output |
|---|---|
| `Mediator.decide_approval` | verdict `deny` (never raises) |
| `Mediator.review_trace` | verdict `halt` (never raises) |
| `SemanticExecutor.execute` | raises `LocalLLMError`; no partial answer is returned |

Additional guarantees: approval classes without user criteria in
`config/mediator-criteria.json` are denied without consulting the model; verdict text that
echoes the cleartext it was shown is withheld and converted to a deny; semantic results are
tokens-only by construction (membership-validated against the input tokens, raw model text
discarded).

## 5. Plug points (the not-yet-built bridge)

- **Approval gate:** where the proxy today denies an `approval`-class resolution without a
  grant, build an `ApprovalRequest` from the action context and, on an approving verdict,
  call `vault.grant_approval(request.placeholder, **verdict.grant_kwargs(request))`.
- **Watchdog:** feed value-free `TraceEvent`s (denied resolutions, forged tokens) into
  `Mediator.should_review`; when it fires, `review_trace` — action `halt` means terminate
  the CUA run.
- **Semantic ops:** parse the Step 6.5 marker `⟦PLVA_TOOL:<verb>:<request_id>⟧` against the
  allowlist (`sort`, `select`), build a `SemanticOpRequest` with `resolver=vault.resolve`,
  and inject `result.observation_text()` into the next observation. The same call is the
  app/proxy-initiated fallback when the model never emits the marker.
- **Rampart backstop:** pass the history-leg scrub as `scrubber=` to `Mediator` so verdict
  text gets the same reclassify pass as scrubbed history.
