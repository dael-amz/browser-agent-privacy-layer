# Local LLM component — Step 7 mediator automation + Step 13(B) semantic executor

Date: 2026-07-12. Status: **component built and live-verified standalone; deliberately not yet
bridged into the proxy** (the bridge is a later, small step — plug points below and in
`docs/local-llm-runbook.md`).

## What was built

- `src/plva_proxy/local_llm.py` — loopback-pinned, fail-closed OpenAI-compatible client
  (rejects non-loopback URLs, no redirects, timeout/HTTP/shape failures raise), JSON-object
  extraction robust to small-model prose, optional `json_schema` response formats with a
  400-fallback chain (json_schema → json_object → none), a cleartext leak scanner
  (NFKC/casefold/whitespace-collapsed plus digit-stream matching), and a point-in-time `lsof`
  no-egress audit of the serving process (ADR-0001's macOS substitute for OpenShell).
- `src/plva_proxy/mediator.py` — §7 mediator: `decide_approval` (user criteria from
  `config/mediator-criteria.json`; deterministic deny when a class has no criteria; malformed
  output, outage, or a verdict echoing the shown cleartext → deny), `review_trace` watchdog
  (deterministic `should_review` trigger; outage/malformed → halt), `plva-mediator` CLI.
- `src/plva_proxy/semantic_executor.py` — Step 13(B): `sort`/`select` over issued tokens. The
  model sees a list of resolved values and answers with exact item texts, grammar-constrained
  to that enum on grammar-capable servers; the executor maps answers back to tokens and
  discards the completion. The return to the CUA is tokens-only by construction.

## Live evidence (llama-server + Nemotron GGUF on 127.0.0.1:8555, synthetic values only)

- `plva-mediator probe`: endpoint reachable; `lsof` audit of the serving pid found **zero
  non-loopback sockets** (`"flagged": [], "checked": true`).
- **The token-only filter caught a real leak attempt.** With Nemotron-Mini-4B and a plain
  token-labeled contract, the model answered a sort with the hidden **values** ("Alice", …)
  instead of token labels; `_validate_answer` rejected it and the executor **failed closed**
  — exactly the §8.12 primary filter doing its job. The contract was then hardened to the
  value-enum design above, after which:
- `demo-sort`: `⟦PLVA_TOOL_RESULT:sort:demo_1⟧` returned the true alphabetical order
  («NAME_2», «NAME_3», «NAME_1» for Charlie/Alice/Bob inputs); leak scan of the injected
  observation: **clean**.
- live `select`: "the personal (non-work) email addresses" over two corporate + one gmail
  address returned exactly the gmail token; observation leak-scan **clean** — fuzzy semantic
  judgment, which is (B)'s reason to exist beyond the deterministic library.
- `demo-trace`: deterministic trigger fired on 4 suspicious events in the window; model
  verdict `warn` with an accurate value-free reason. Fail-closed paths (server down → deny /
  halt) covered by tests.
- `demo-approval` on Nemotron-Mini-4B: **denied** a legitimately-in-criteria use with a
  misapplied reason — conservative in the fail-safe direction, but imprecise. The 8B
  Nemotron Nano judged the same fixture correctly but caused system-wide memory pressure on
  the 16 GB MacBook Air and is banned from this host (see runbook sizing rule).
- **Recommended mediator model: Qwen3-4B-Instruct-2507 Q4_K_M** (2.3 GB file, **2.91 GB
  server RSS** with `--ctx-size 4096` — measured, comfortable on the Air). Live results:
  the approval fixture **approved** with a reason that walks each rule condition and a
  single-use scope pinned to the request's destination; the adversarial variant (same key
  into a support-chat message box) was **denied** citing the exact violated clause; trace
  demo `warn` with an accurate value-free reason; sort correct and leak-clean. Its initial
  deny of an underspecified fixture (generic field name `text`, no destination context)
  was correct-by-the-letter, which is why `ApprovalRequest.target` should carry on-screen
  destination context when the bridge is built. Denials never block the human path: the
  Step 7 manual grant UI still works.

## NemoClaw/OpenShell sandbox path (added later on 2026-07-12 — now the primary serving path)

The model now runs **inside an OpenShell sandbox** (gateway 0.0.80, Docker driver, Ubuntu
24.04 arm64 container capped at 3.5 GB inside a 4 GB Docker VM), managed by
`run_mediator_sandbox.sh`. Everything below was established empirically, per the operator's
instruction not to trust the blueprint/ADR:

- **Enforcement is real, not observation.** With the image's default policy, an allowlisted
  host (ollama.com) returned HTTP 200 from inside while a non-allowlisted host and a raw IP
  failed — deny-by-default confirmed. After applying
  `config/openshell-mediator-policy.yaml` (`network_policies: {}`), **every** channel is
  blocked from inside: HTTPS 403 (including previously-allowlisted hosts), plain HTTP 403,
  raw-IP 403, and DNS resolution fails. `run_mediator_sandbox.sh up` re-runs this deny-test
  every start and refuses to serve if any probe escapes.
- **Two pitfalls worth knowing:** the community `ollama` image ships a permissive built-in
  allowlist (ollama registry, GitHub, npm, nvidia, api.anthropic.com) — zero-egress must be
  applied explicitly; and `policy set` on a live sandbox takes a few seconds to reach the
  data plane (first re-test still showed the old allowlist).
- **`policy prove` is inconclusive here:** it PASSed our zero-egress policy, but a
  deliberately leaky negative control (curl + attacker endpoint) also PASSed with our
  hand-written credential descriptor, so its PASS carries no weight; the live deny-test is
  the load-bearing evidence.
- **Serving engine inside the sandbox:** llama.cpp `llama-server` b9977 (linux-arm64),
  uploaded with the GGUF via `openshell sandbox upload` (no sandbox egress involved),
  because Ollama's decode path lacks grammar enforcement and measurably degraded judgment
  (same model: approve→deny, warn→continue). Through the sandboxed llama-server, all
  results match the host run exactly: approval fixture **approved** with bounded scope, the
  chat-box misuse **denied** citing the violated clause, watchdog `warn` accurate, sort
  correct and leak-clean. Latency ≈27 s per verdict (CPU-only in the VM; no Metal) —
  acceptable for a sparingly-consulted mediator.
- The host reaches the model only through `openshell forward service` (host loopback 8555 →
  sandbox loopback 11500); the client's own loopback pinning still applies.

## Gates

Full suite after integration: **260 tests pass**, repo coverage 82% (gate ≥80%); ruff format,
ruff check, and strict mypy pass on all three new modules (new-module coverage: local_llm 90%,
mediator 78%, semantic_executor 98%).

## Known boundaries (per blueprint, not regressions)

- The `lsof` audit is point-in-time evidence, not an enforcement proof; OpenShell
  `policy prove` remains the enforcing path on hosts where it works (ADR-0001).
- Inferential leakage through free-text rationale is out of contract: verdict text is
  leak-scanned and scrub-hookable, and the executor returns no free text at all, but a
  rationale *describing* a hidden value is a known gap the blueprint assigns to gated
  aggregates (§ Step 13 boundary note).
- Small-model approval judgment is conservative (false denies possible); model choice is a
  config swap, interfaces unchanged.

## Rerun

```bash
llama-server -m models/<nemotron>.gguf --host 127.0.0.1 --port 8555 --no-webui &
uv run plva-mediator probe
uv run plva-mediator demo-approval && uv run plva-mediator demo-trace && uv run plva-mediator demo-sort
```
