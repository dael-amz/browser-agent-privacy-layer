# Local LLM runbook: mediator and semantic executor

The mediator and the semantic executor consult a small local model that may see vault
cleartext, so it must run fully local with zero network egress. The code side is
`src/plva_proxy/local_llm.py` (loopback-pinned, fail-closed client), `mediator.py`
(approval verdicts plus trace watchdog), and `semantic_executor.py` (token-only
sort/select). The bridge into the proxy is described at the bottom.

## 1. Primary path: NemoClaw/OpenShell sandbox

The model runs inside an OpenShell sandbox (Docker-driver Linux container) under a
zero-egress policy (`config/openshell-mediator-policy.yaml`: `network_policies: {}` on
OpenShell's deny-by-default L7 proxy). The proxy reaches it only through
`openshell forward service` on host loopback. One command manages the whole stack:

```bash
./run_mediator_sandbox.sh up      # Docker + gateway + sandbox, verify zero egress, forward :8555
./run_mediator_sandbox.sh status
./run_mediator_sandbox.sh down          # stop model + forward (frees ~3 GB); run after every session
./run_mediator_sandbox.sh down --full   # also quit Docker Desktop + gateway
```

`up` hard-fails unless a live deny-test from inside the sandbox proves egress is
blocked (HTTPS to a public host, the image's formerly-allowlisted hosts, a raw IP, and
DNS must all fail). Never skip that gate: the community `ollama` image ships a
permissive built-in allowlist (ollama.com, GitHub, npm, even api.anthropic.com) until
the zero-egress policy is applied, and `policy set` takes a few seconds to propagate to
the data plane.

Facts established empirically: OpenShell enforcement is real for sandboxes it launches
via the Docker driver (denied connects fail with L7 403s, DNS resolution is blocked);
`policy set` on a live sandbox works after a propagation delay; `openshell policy prove`
passes our policy but also passed a deliberately leaky negative control with our
hand-written credential descriptor, so treat prove as inconclusive and rely on the
deny-test.

Inside the sandbox: llama.cpp `llama-server` (linux-arm64, uploaded via
`openshell sandbox upload` together with the GGUF, so no sandbox egress is needed)
serving Qwen3-4B-Instruct-2507 Q4_K_M on the sandbox's own loopback with
`--ctx-size 4096`. We run llama-server rather than the image's Ollama because its
`json_schema` grammar enforcement is what makes verdicts reliable on a 4B model;
through Ollama the same model lost judgment quality (approve became deny, warn became
continue). CPU-only inference (no Metal in the VM) takes roughly 27 seconds per
approval verdict, acceptable for a sparingly consulted mediator.

Memory on a 16 GB machine: cap the Docker VM at 4 GB (Docker Desktop settings) and the
container at 3.5 GB, which costs about 5.5 GB host-side while up. Tolerable briefly, not
for background use; run `down` after every session. Do not serve 8B-class models on a
machine this size.

## 2. Fallback path: host llama-server (no sandbox; dev and latency-sensitive demos)

```bash
# Mediator-grade judgment, ~2.9 GB RSS, Metal-fast:
llama-server -m models/qwen3-4b-instruct-2507-q4_k_m.gguf \
  --host 127.0.0.1 --port 8555 --ctx-size 4096 --no-webui

# NVIDIA alternative, executor-grade only (falsely denies legitimate approvals):
llama-server -m models/nemotron-mini-4b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 --port 8555 --ctx-size 4096 --no-webui
```

On this path the zero-egress evidence is weaker: the client refuses non-loopback URLs and
`plva-mediator probe` runs a point-in-time `lsof` audit (flags any non-loopback bind or
remote connection of the serving process). Fine for development; the sandbox path is the
demo and trust story. Avoid Ollama for the cleartext mediator on either path: its daemon
has its own network reach, and its decode path lacks the grammar enforcement the 4B needs.

Defaults expected by the client: `http://127.0.0.1:8555/v1`; the model-name field is
advisory for llama-server (it serves whatever is loaded; the sandbox and host launches
both use alias `qwen3-4b`). Override with `PLVA_LOCAL_LLM_URL` / `PLVA_LOCAL_LLM_MODEL`
or `plva-mediator --url/--model`.

## 3. Verify

```bash
./run_mediator_sandbox.sh up        # sandbox path: includes the in-sandbox deny-test gate
uv run plva-mediator probe          # reachability plus lsof audit (host-path evidence)
uv run plva-mediator demo-approval  # synthetic API_KEY approval verdict
uv run plva-mediator demo-trace     # synthetic misbehavior trace -> continue/warn/halt
uv run plva-mediator demo-sort      # synthetic token sort + leak scan of the observation
./run_mediator_sandbox.sh down      # after the session
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

## 5. The bridge (`src/plva_proxy/tool_channel.py` plus proxy flags)

Run the proxy with `--privacy --privacy-tools --privacy-mediator` (endpoint from
`PLVA_LOCAL_LLM_URL`, default `http://127.0.0.1:8555/v1`, which is the sandbox forward):

- **Tool channel:** the request hook injects `[PLVA_TOOLS]` teaching; the CUA
  emits `⟦PLVA_TOOL:<verb>:<tokens>:<id>[:<instruction>]⟧` in thought text; the response
  hook validates it against the verb allowlist (`sort`, `select`) and executes it. A
  `sort` without an instruction runs the deterministic in-proxy library;
  instruction-bearing operations run the sandboxed model with the tokens-only contract.
  The value-free `⟦PLVA_TOOL_RESULT:…⟧` line is injected into the next observation beside
  the manifest. `POST /viewer/tools` `{verb, tokens[], instruction?}` is the mandatory
  proxy- or app-initiated fallback, since marker compliance varies by model. The
  standalone spike loop in `tools.py` (`--tools`, synthetic echo/add/sort, no vault
  access) is untouched and separate.
- **Approval gate:** an `approval`-class resolution without a grant consults the
  mediator with the action context (tool, argument path, destination, scrubbed task text);
  an approving verdict mints the exact vault grant via `verdict.grant_kwargs()` and the
  resolution is retried once; anything else keeps the original fail-closed denial.
- **Watchdog:** denials and tool events accumulate as value-free `TraceEvent`s; the
  deterministic trigger fires at most one `review_trace` per window; `halt` blocks every
  further forwarded request (fail closed), `warn` injects a `[PLVA_MEDIATOR]` notice.
- **Rampart backstop:** verdict text passes through the history-leg scrub before leaving
  the mediator; scrub failure withholds the text.

Mediator and executor calls block the proxy request briefly (a few seconds on the sandbox
CPU path), which is acceptable because both are consulted sparingly by design.
