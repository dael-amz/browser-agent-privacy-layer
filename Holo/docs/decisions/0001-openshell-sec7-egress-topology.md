# 0001 — §7 OpenShell topology: egress-isolation, with a host-level egress boundary

Date: 2026-07-11

Status: **Accepted (architecture) / execution blocked on this platform.** Resolves the BLUEPRINT
§7 open decision required by Step 1.

## Context

BLUEPRINT §7 leaves one decision open, to be resolved in Step 1:

- **(a) Egress-isolation** — the runtime drives the *host* desktop; the PLVA proxy and all outbound
  model/web traffic are forced through OpenShell's restricted network so the proxy is the only
  egress. Sandbox = the network boundary. (Recommended starting point.)
- **(b) Contained-desktop** — the CUA operates a desktop *inside* a sandbox; screenshots and actions
  stay inside it; egress is controlled at the sandbox edge. Stronger, but fights HoloDesktop's
  "drives your real computer" design.

The goal of either is to make "only the redacted frame leaves" an **infrastructure** guarantee, not
merely a code invariant.

## Findings (empirical, this host: macOS / Apple Silicon, Docker present)

1. **No Linux `hai-agent-runtime` binary exists.** The pinned runtime manifest (v0.1.8) publishes
   exactly `darwin-arm64` and `windows-x86_64`; `darwin-x86_64` and `linux-x86_64` are explicitly
   "not published yet", and there is **no `linux-arm64` entry at all**.
   → **Option (b) is impossible here:** the Holo runtime cannot run inside OpenShell's Linux
   sandbox, because no Linux build of it exists.

2. **OpenShell only isolates processes it launches inside its own sandbox.** Its egress control is a
   per-sandbox Linux network namespace (veth pair) behind an OPA-backed L7 proxy that
   allows / routes-for-inference / denies each connection (deny-by-default). The docs state it "does
   not wrap or attach to existing host processes." A macOS host process is in no such namespace, so
   OpenShell has no interception point for its traffic.
   → **Option (a) as literally worded is not achievable:** OpenShell cannot force a host-resident
   Holo runtime's egress through its proxy. The Holo runtime *must* run on the host to obtain macOS
   Screen Recording + Accessibility.

3. OpenShell *is* installable on this host (Homebrew or `uv tool install openshell`; Docker
   satisfies the driver requirement). It is alpha / "single-player". When it cannot create the
   namespace it degrades to pass-through/observation (logs verdicts, does **not** enforce egress) —
   so it must never be trusted as an enforcing boundary unless namespace creation is confirmed.

## Decision

Adopt **(a) egress-isolation** as the architecture, but **the egress boundary is not OpenShell on
this platform.** Concretely, the intended topology is:

```
 macOS host
   hai-agent-runtime (host process; Screen Recording + Accessibility)
        │  base_url = 127.0.0.1  (its ONLY model endpoint)
        ▼
   PLVA proxy  ── holds the vault, redacts frames, resolves actions ──
        │  the ONLY process permitted to reach the provider
        ▼
   Overshoot  (Hcompany/Holo3-35B-A3B)
```

The "only the redacted frame leaves" guarantee is enforced at two layers:

- **Code (fail-closed, built in Steps 3–4):** the proxy is the runtime's sole configured endpoint
  and the only component that talks to Overshoot; on any stage failure it forwards nothing.
- **Infrastructure (host-level, this platform's substitute for OpenShell's netns):** a macOS
  packet-filter / application-firewall rule set that (i) permits the Holo runtime to reach **only**
  `127.0.0.1` (the proxy), and (ii) permits **only** the proxy process to reach the provider host.
  This reproduces "the proxy is the only egress" that OpenShell would otherwise provide.

**Where OpenShell still earns its place:** run the **PLVA proxy itself** inside an OpenShell sandbox
whose policy allows egress only to the provider (and routes inference). OpenShell is good at exactly
this — sandboxing a process it launches — even though it cannot sandbox the host runtime. This is a
hybrid: host firewall pins the runtime to loopback; OpenShell constrains the proxy's upstream.

**Option (b) is deferred**, not rejected: revisit if/when H Company ships a Linux
`hai-agent-runtime` (then the whole CUA + desktop can run inside an OpenShell/Xvfb sandbox and this
host-firewall substitute is no longer needed).

## Consequences

- Step 1's "runs under OpenShell isolation" cannot be demonstrated end-to-end on this machine yet:
  the host runtime can't be sandboxed by OpenShell, and the enforcing egress boundary depends on the
  Step 3 proxy plus a host packet-filter rule set that has not been stood up. See
  `verification/step-1-status.md`.
- No secret is placed in OpenShell policy files; the provider credential stays in the untracked
  `.env` and is injected only by the proxy.
- If OpenShell is ever used, its namespace creation must be verified as *enforcing* (not degraded to
  observation) before any real frame is allowed to flow.
