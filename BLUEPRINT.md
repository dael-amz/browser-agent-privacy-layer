# PLVA Blueprint — Holo CUA Privacy Environment

This file specifies the **CUA-management component** of the PLVA project. It is written to be
built by an AI coding agent in small, independently testable steps.

> **Revision 2026-07-12.** Reworked from the original after building Steps 0–4 and clarifying the
> threat model with the operator. Material changes since the first draft: the threat model is now
> explicitly *"protect against the model inference provider and anything upstream of it"* (not the
> local desktop); per-class PII **safety levels** replace the fixed secret-tier list; an optional
> **LLM mediator** (approvals/steering) is added; the Gradium audio work is split into an
> **inbound** (context) and a privacy-free **outbound** (voice-read) direction; a thin **app** is
> added; and several statements are corrected (skills are native to HoloDesktop, not a NemoClaw
> feature; resolution scope is about leak-back, not destination). Build-step status and priority
> now reflect what is actually built. Search for **CHANGED** / **NEW** markers.

---

## 0. How to build from this document (read first)

- Build **one step at a time, in priority order** (§9). After each step, stop and run its
  **Verify** check. Do **not** start the next step until Verify passes. Steps marked ✅ are done.
- Prefer the smallest change that satisfies the step. Each step must be independently runnable.
- Treat everything under **Hard constraints** (§8) as non-negotiable. If a step seems to require
  breaking one, stop and surface it instead of proceeding.
- **Secrets:** never commit API keys, screenshots, transcripts, or vault contents. Keys live in
  environment variables and an untracked `.env` only. Keep a safe `.env.example`.
- **Stubs:** when a step depends on a component out of scope here (the PII detector), build against
  the **stub interface** in §5 so this component is testable before the real one exists.
- Any fact marked **VERIFY** is an assumption to confirm empirically. If it turns out false, stop
  and report — do not build on it.
- **NEW — local-model acceleration is in flight (separate agent).** The redaction detector and the
  local models it depends on are being sped up with GPU / WebGPU and Apple **CoreML** execution
  providers, behind a persistent warm worker (see the "Cross-cutting" note at the end of §9). Treat
  every local-model dependency (redactor, and later local TTS and a local mediator model) as
  *pluggable behind a client interface* whose backend may be WASM, WebGPU, or CoreML. Never hardcode
  a backend; keep the frozen baseline as the correctness oracle; keep the interface fail-closed.

---

## 1. Goal and scope

**PLVA** is a privacy layer for computer-use agents (CUAs). Full pipeline:

> screenshot → detect PII + bounding boxes → paint placeholders over the PII values
> (e.g. `PASSWORD_1`) → store the real values in a key-value **vault** keyed by placeholder →
> tell the model how to reason about placeholders → when the model returns actions, resolve each
> placeholder back to its real value **before** the action executes.

**Threat model (CHANGED — read carefully; it drives the rest of the doc).** Privacy is defined
**against the model inference provider (Overshoot) and anything upstream of the model.** The goal
is that the model **never *sees*** a value it should not need to know. It is *fine* for the model
to **cause a real value to be used** — typed into a form, placed in a URL, sent somewhere — because
that is the action the user asked for and the layer resolves the value locally, in transit, out of
the model's view. So the discipline is not "never let a real value reach any destination"; it is
"never let a real value reach the model or its provider." That reframing relaxes some earlier rules
(see §8.4) and sharpens others (history leak-back, §6/§8.9, is now the load-bearing concern).

**This component covers:**
- running the CUA, and **intercepting and modifying both directions of the model traffic** (request
  to the provider and returned actions) — the PLVA proxy;
- the **vault** and a **configurable per-class PII safety policy** (NEW, §6);
- an optional **LLM mediator** for approval-gated classes and privacy steering (NEW, §7);
- **audio** via Gradium, in two directions (NEW, §9 Steps 9, 10, 12): inbound STT → redacted context,
  and a privacy-free outbound **voice-read** of real values through a **local** TTS model;
- a thin **app** that takes a typed or (lower priority) spoken prompt and runs the task (NEW, §9
  Step 11);
- **placeholder-preserving semantic operations** so content-dependent tasks (sort / filter / reason
  over hidden values) work even though the model never sees the real values (NEW, §9 Step 13).

**Out of scope (separate component):** the actual detection/redaction of PII pixels, and its
GPU/CoreML acceleration. Here the detector is a **pluggable dependency** with a stub (see §5). The
detector must, however, emit recognized **values** to the vault locally — see the §5 contract note.

---

## 2. Glossary

- **CUA** — computer-use agent: takes a screenshot + task, emits GUI actions (click/type/scroll).
- **Holo3** — H Company's vision computer-use model. This project uses Overshoot's
  **`Hcompany/Holo3-35B-A3B`** only. Consumes images + text; has **no native audio input** (audio
  reaches it only as injected text, §9 Step 12).
- **HoloDesktop CLI** — the harness (`github.com/hcompai/holo-desktop-cli`, Apache-2.0, Python/uv).
  Open CLI wrapper; downloads a **closed `hai-agent-runtime`** binary that owns the perceive→act
  loop (screen capture, model call, action execution). **CHANGED — skills:** HoloDesktop has its
  **own** native skill mechanism (`~/.holo/skills/*/SKILL.md`, loaded into the model request by the
  runtime). Skills are **not** a NemoClaw feature. Teaching the model about placeholders can use a
  native skill *or* proxy injection; we prefer proxy injection for guaranteed presence, optionally
  reinforced by a skill (§9 Steps 5 and 5a).
- **Overshoot** — the inference provider. OpenAI-compatible HTTP API.
- **NemoClaw / OpenShell** — NVIDIA's agent stack. **OpenShell** is its sandbox (network + file
  isolation). In this project it is used to **sandbox processes we launch** (primarily the mediator,
  optionally the proxy / audio client) with deny-by-default egress — **not** as an orchestrator
  brain and **not** to wrap the host runtime (it cannot). See §7.
- **PLVA proxy** — the local OpenAI-compatible HTTP proxy we build; the single point where model
  traffic is intercepted and rewritten. Built (Steps 1, 3, 4).
- **Vault** — private, in-memory, session-scoped map from placeholder → real value, plus per-class
  policy state.
- **Placeholder** — a stable, session-nonce-namespaced semantic token painted over a PII value,
  e.g. `EMAIL_1_a3f9`. The namespace prevents on-screen/adversarial forgery of a valid token.
- **Safety level** (NEW) — the per-class policy that decides how a PII class is handled: hide-and-
  use, approval-gated, blocked, etc. (§6).
- **Mediator** (NEW) — an LLM the proxy *consults* on deterministic triggers to approve/deny
  gated resolutions (against user-defined criteria) and to steer. A *mediator*, never an
  orchestrator: the proxy stays in control.
- **Gradium** — voice-AI provider: cloud **STT** (real-time, WebSocket, semantic VAD) and a
  **local TTS** model. Cloud STT sees pre-redaction audio (trusted-boundary assumption); local TTS
  runs on-device with no egress and is the key to the privacy-free outbound voice channel.
- **Audio-in / audio-out** (NEW) — *in*: mic/loopback → STT → scrub → model context (must be
  redacted before the model). *out*: vault → local TTS → speakers (never reaches the model, so it
  may carry real values).
- **Placeholder-preserving operation** (NEW) — a content-dependent computation (sort, filter,
  dedupe, compare, or free reasoning) run by a **local** executor that *may* see cleartext but
  returns its result expressed **only in placeholders** (an ordering, a selected subset, or a
  carefully-gated aggregate). Lets the CUA obtain the *answer* to a task that needs semantic content
  without any real value entering the model's view (§9 Step 13).

---

## 3. Technical choices (fixed)

| Choice | Value | Notes |
|---|---|---|
| Model | **`Hcompany/Holo3-35B-A3B`** | Exact Overshoot model id. Do **not** use Nemotron, OpenClaw, or another model/harness. |
| Harness | **HoloDesktop CLI** | Closed `hai-agent-runtime` owns the loop — see §4 for why this forces a proxy. |
| Inference provider | **Overshoot** (OpenAI-compatible) | The proxy targets Overshoot by default. |
| NemoClaw integration | **OpenShell sandbox** for the **mediator** (and optionally proxy/audio) | Isolation + controlled egress for processes we launch; not an orchestrator; cannot wrap the host runtime. |
| Audio | **Gradium** STT (cloud) + **local** TTS | STT is trusted-boundary; local TTS enables the privacy-free voice-read (§9 Step 9). |
| Local-model acceleration | GPU / **WebGPU** / **CoreML** behind a warm worker | In flight (separate agent). Keep every local model behind a pluggable, fail-closed client. |

---

## 4. Architecture and why a proxy is required

```
 Host machine
 ┌───────────────────────────────────────────────────────────────────────┐
 │  App (NEW): typed prompt, or spoken prompt via Gradium STT (low prio)   │
 │      │ hands the task to HoloDesktop                                    │
 │      ▼                                                                  │
 │  HoloDesktop CLI → hai-agent-runtime (CLOSED)                           │
 │    • captures the real screen • builds the request (raw screenshot)     │
 │    • executes returned actions • base URL points at the PLVA proxy      │
 └──────┼──────────────────────────────────────────────────────────────────┘
        ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │  PLVA PROXY (local, OpenAI-compatible) ── holds VAULT + safety policy ──│
 │   REQUEST leg (runtime → model):                                        │
 │     1. redact screenshot (detector → new frame) + store spans+values    │
 │     2. history scrub: plain vault match, then Rampart reclassify (§6)   │
 │     3. inject placeholder instructions + redacted audio context (§9)    │
 │   RESPONSE leg (model → runtime):                                       │
 │     4. parse action; resolve placeholders in executed fields (§8.4);    │
 │        gated classes → consult MEDIATOR; route speak-actions to TTS     │
 └───┬───────────────────────────────┬───────────────────────────┬─────────┘
     ▼                               ▼ (deterministic trigger)     ▼ (audio-out)
 Overshoot                    MEDIATOR (OpenShell sandbox,     Local Gradium TTS
 (Hcompany/Holo3-35B-A3B)     local model, zero egress)        → user's speakers
```

**Why a proxy and not "just change the request":** the closed `hai-agent-runtime` builds and sends
the request; we cannot edit that code, so the request never exists as our own object to modify. The
**only** knob is **where** it sends (its base URL). Pointing that at a local proxy is the only way
to see and rewrite the traffic. OpenAI-compatibility makes the proxy easy to write; it does not
remove the need for it. Actions flow back into the closed binary to be executed, so placeholder
resolution must also happen in the proxy, in transit. The proxy is likewise the only per-step seam
for injecting live audio context and for routing speak-actions to local TTS.

---

## 5. The PII detector interface (stub for this component)

The real detector is a separate component (and is being GPU/CoreML-accelerated elsewhere). Build
against a **stub** with a stable contract so the pipeline is testable now.

**`redact(frame)` → `{ redactedFrame, spans }`**
- `frame`: the screenshot pulled from the request.
- `redactedFrame`: a **new** image with placeholder chips painted over PII values; pixels outside
  the covered regions are byte-identical to the input.
- `spans`: list of `{ class, value, boundingBox, placeholder? }` for each detected PII value.

**CHANGED — the detector must emit recognized `value`s to the vault, locally.** Placeholder
stability, resolution, *and* history scrub are all impossible if the detector returns geometry
only. (The current frozen `plva-v2-baseline` is geometry-only — good enough for Step 4 obscuring,
**not** for Steps 5+.) That development-only snapshot is historical provenance, not the current
detector's status: the current detector is release eligible and supplies the recognized values
required by the privacy core. The vault — not the detector — **owns placeholder assignment** for stability
(§6), so the detector emits `class/value/boundingBox` and the painter renders the vault's
placeholder. Values never leave the device and never appear in logs.

**Stub behavior for testing:** deterministic and configurable — detect a hardcoded string / fixed
region, paint a chip, return one span with its value. Provide a "detect nothing" mode and a "detect
fixture X" mode so tests can exercise the full store→resolve→scrub loop without a real detector.

The proxy depends only on this interface, never on the detector's internals or its acceleration
backend.

---

## 6. The vault and per-class safety policy (CHANGED)

Private, in-memory, session-scoped. Holds placeholder↔value maps plus per-class policy.

**Store (request leg):** for each detected span, canonicalize the value (normalize whitespace,
Unicode NFKC; keep credentials exact/case-sensitive), then look it up. Same value → **same**
placeholder; new value → next per-class counter → new placeholder, namespaced with a per-session
nonce (`EMAIL_1_a3f9`) so screen/adversarial text cannot forge a valid token.
> **CHANGED — duplicates tolerated for now.** Ideal is one stable placeholder per value across all
> frames. For this build it is acceptable for the same value to receive different placeholders
> across frames if stable assignment is hard; the only cost is possible multi-step incoherence
> ("I typed `EMAIL_1`, now it's `EMAIL_2`"). Revisit for stability later.

**Resolve (response leg):** given a placeholder, return its value **only if** it was issued this
session and its class's safety level permits, possibly after mediator approval (below).

**Per-class safety levels (NEW — replaces the fixed secret-tier list).** Each PII class maps to a
**configurable** level (defaults shipped, user-overridable via the app, §9 Step 6):
- **`hide_use`** — painted as a placeholder and freely resolvable. "Can use but hide." (Default for
  low-sensitivity classes: `EMAIL`, `PHONE`, `NAME`, `ADDRESS`.)
- **`approval`** — painted as a placeholder, but resolution is **denied until the mediator approves**
  against **user-defined criteria** (§7). "Use only with LLM approval." (Default for `API_KEY`,
  `AUTH_TOKEN`.)
- **`blocked`** — masked, **never stored, never resolvable**; the model gets an opaque mask, not a
  usable placeholder. "Cannot use." (Default for `PASSWORD`, `CVC`, `CARD_NUMBER`, `PRIVATE_KEY`.)
- **Additional levels to consider:** `voice_approval` (user approves by voice, §9 Step 9);
  `destination_scoped` (resolvable only toward an allowed destination); `out_of_band` (host fills a
  secret directly from the vault; the model never even receives a placeholder — the safe path for
  passwords, and the natural home for the voice secret-entry channel).

**Invariants:**
- Different values must **never** alias to the same placeholder (duplicates go the *other* way —
  one value may get several placeholders for now; never one placeholder for two values).
- **No persistence** anywhere (disk, IndexedDB, localStorage, logs, telemetry). Cleared on
  `dispose()`.
- A denied/blocked resolution **fails closed**: the action is not executed with a real value; the
  proxy may inform the CUA it was denied so it can adapt.

---

## 7. NemoClaw / OpenShell and the mediator (CHANGED)

**Egress topology (resolved — see `docs/decisions/0001`).** OpenShell **cannot wrap the host
runtime** (it only isolates processes it launches; no Linux `hai-agent-runtime` exists). So the
"only redacted data leaves" guarantee is enforced by (a) the fail-closed proxy code and (b) a
host-level packet-filter substitute on macOS, with OpenShell reserved for processes we *do* launch.
Skills do **not** require NemoClaw (corrected — see §2).

**Primary NemoClaw role now = the mediator sandbox (NEW).** The mediator is an LLM the proxy
**consults** when a deterministic flag trips (a resolution hits an `approval` class; a configurable
risk condition fires). It returns `{decision: approve|deny|modify, steering?, scope?}`; the proxy
enforces it. Control stays with the proxy — mediator, not orchestrator.

> **CRITICAL constraint.** A mediator that sees **cleartext** (real values / raw frame, needed for
> real judgment) **must run a fully local model with zero network egress** — otherwise its own LLM
> provider sees the secret and the leak reappears one layer up. Two allowed shapes:
> - **local-model mediator** → may see cleartext; **no egress** (run in OpenShell, egress denied;
>   or a local model on the host with a verified no-egress boundary);
> - **remote-model mediator** → may see **only** redacted/placeholder context, never raw values.
>
> Build toward the local, zero-egress mediator so "approve a secret unlock" is actually safe.

**Why OpenShell here (its genuine sweet spot):** it sandboxes a process *it launches*. Run the
mediator in a sandbox with **deny-by-default egress** (nothing, for a local model); reach it from
the proxy over `openshell forward service` (loopback-only gRPC); and use `openshell policy prove`
to *verify* the "no egress" invariant rather than assert it. Optionally sandbox the proxy and the
audio/TTS client the same way. **Caveat:** OpenShell is alpha and, if it cannot create its network
namespace, **degrades to observation (does not enforce)** — verify enforcement (a deny-test from
inside) before trusting it with cleartext.

Fail-closed: if the mediator is unreachable or times out, **deny**.

---

## 8. Hard constraints (do not violate)

1. **Fail closed.** If any stage fails, no request/response is forwarded and no raw frame or vault
   value ever reaches the model/provider. There is no raw fallback.
2. **Redacted-only egress *to the model*.** Overshoot must never receive an un-redacted screenshot,
   a vault value, or un-scrubbed history. (Real values reaching a *local* executor or the user's
   own destinations is expected — see the §1 threat model.)
3. **New output buffer.** Redaction renders into a new image; input pixels are never mutated.
4. **CHANGED — resolution scope is about leak-back, not destination.** Resolve placeholders in the
   **executed action fields** (type text, URLs, click targets — wherever the user's action needs
   the value). Do **not** resolve placeholders in the model's free-form **reasoning/thought text**:
   that is not executed and would only re-expose the value into history on the next turn.
5. **Privacy-safe logs.** Logs may contain class, placeholder, bounding box, backend, timings —
   **never** recognized text, real values, transcripts, or frames.
6. **No persistence of session state.** Vault and any cache are memory-only and dropped on dispose.
7. **Streaming-safe.** Handle **SSE / streamed** responses: buffer/parse enough to resolve the
   action before forwarding. Never forward an unresolved placeholder to the executor. (Built §3.)
8. **Model id is not hardcoded blindly.** Confirm the exact provider model-id via `/models` (§0).
9. **NEW — history scrub is mandatory and load-bearing.** Because the runtime remembers the
   *resolved* real value it executed, every outbound request's text history must be scrubbed:
   **plain vault match first, then a Rampart semantic reclassify** as a backstop for reformatted or
   never-vaulted PII (map hits back to the vault for a stable placeholder; fail-closed on unknown
   PII). This is best-effort, not a proof (Rampart has finite recall) — treat it as the primary
   residual risk and test it adversarially.
10. **NEW — the outbound voice channel must not feed back.** Real values spoken by local TTS must
    not be captured by audio-in (exclude our own output device / duck capture during playback) and
    must not be rendered on-screen as captions (a caption re-enters the model via the next frame).
11. **NEW — local acceleration stays fail-closed and offline.** A GPU/CoreML/WebGPU backend for any
    local model must not change fail-closed behavior and must not add network egress; the frozen
    baseline remains the correctness oracle.
12. **NEW — placeholder-preserving ops return tokens, not values.** A content-dependent operation
    (§9 Step 13) may resolve real values locally to compute its answer, but what it returns **to the
    model** must be placeholders/orderings/selections only (or a deliberately gated aggregate),
    never a real value — the return enters model context and would otherwise leak upstream exactly
    like an un-scrubbed history (cf. §8.4, §8.9). Its executor may see cleartext **only** if it is
    local with zero egress (same rule as the mediator, §7).

---

## 9. Build steps (priority order; status marked)

Each step lists **Goal → Build → Verify**. Do not proceed past a failing Verify. After each step,
pause for operator verification and describe how to test it.

### ✅ Step 0 — Setup and external-contract probe — **COMPLETE**
Clean repo, secrets handling, confirmed contracts. `/models` confirmed `Hcompany/Holo3-35B-A3B`
ready; response shape recorded (structured JSON action in `message.content`, `tool_calls` array,
`structured_outputs` present, no native `tools`); streaming characterized. **Critical VERIFY
passed:** the closed runtime sends its screenshot through the configurable base URL (proven against
a loopback stub). Evidence in `Holo/verification/step-0-*`.

### ✅ Step 1 — HoloDesktop under isolation — **LIVE ACCEPTANCE PASS**
Pass-through proxy is the runtime's sole endpoint and sole provider egress. §7 egress decision
resolved (ADR-0001). A real Holo/H Company run completed a controlled two-step form flow through
PLVA with provider egress monitoring enabled; the loopback fixture independently verified the exact
private value was submitted after local placeholder resolution.

### ✅/⚙️ Step 2 — Overshoot inference + latency — **inference done; latency pending**
The proxy targets Overshoot by default. Remaining: record per-step and end-to-end latency
(p50/p95) for a fixed small task set from the proxy's privacy-safe timing logs.

### ✅ Step 3 — Interception proxy (no redaction) — **BUILT & verified**
Loopback OpenAI-compatible proxy with a request/response mutation seam, SSE buffered→mutated→
re-emitted, fail-closed. Verified with pass-through, the `test` hook, and a `banana` hook that
rewrites every text the CUA types (a live proof of the response-leg action-rewrite seam that Step 5
resolution reuses).

### ✅ Step 4 — Outbound obscuring (redaction) — **BUILT; CURRENT DETECTOR RELEASE ELIGIBLE**
The proxy redacts every outbound screenshot through the detector and serves the obscured frames at
a loopback `/viewer`; a persistent WebGPU/WASM worker keeps it warm (see Cross-cutting note).
The frozen July 11 v2 fixture remains geometry-only for historical Step 4 reproduction. It does not
describe the current release-eligible detector, whose recognized values feed the completed Step 5
vault and placeholder path.

---

### ✅ Step 5 — Vault + placeholder resolution + history scrub (the privacy core) — **COMPLETE**
- **Goal:** the model works with placeholders it can reason about but never sees the real values;
  actions resolve to real values in transit; history never re-exposes a resolved value.
- **Build:**
  - Detector emits `class/value/boundingBox` (§5); vault stores and assigns stable, nonce-namespaced
    placeholders (§6); painter renders the vault's placeholder chips.
  - **Teach the model** via proxy request-injection (preferred) or a native HoloDesktop skill:
    "regions shown as «CLASS_N…» are real values you cannot see; to use one, emit its token
    verbatim in an action; never guess the underlying value."
  - **Response leg:** reuse the Step 3 seam to resolve placeholders in executed action fields only
    (§8.4), conservatively matched (fail-closed on no confident match).
  - **History scrub (§8.9):** plain vault match, then Rampart reclassify; fail-closed on unknown PII.
- **Verify (fixtures):** forwarded request carries the redacted image and **no** real value; a
  `type` action with `EMAIL_1` resolves to the real value before execution while the same token in
  reasoning text is untouched; after a value is "typed," the next request's history no longer
  contains it; every injected stage failure forwards nothing; no log carries a real value.
- **VERIFY (empirical, do first):** confirm Holo3 actually honors placeholder instructions rather
  than hallucinating the hidden value. If not, rethink before building the rest.

**Completed 2026-07-12:** the Vision OCR/Rampart finding supplies the value and box to one
session-only vault assignment; the proxy paints the vault token on that same redacted box, injects
placeholder instructions, resolves only executed action fields, and scrubs outbound history by
plain vault match followed by the warm Core ML Rampart backstop. Deterministic fixtures, injected
failures, JSON/SSE proxy gates, and a synthetic Holo provider pass succeeded. Holo copied the exact
visible token into `write`; the proxy resolved it locally while leaving reasoning untouched. See
`Holo/verification/step-5-privacy-core.md`.

### ✅ Step 5a — Richer placeholder teaching (scheme + on-screen manifest + duplicate warning + skill)
- **Goal:** strengthen how the CUA is taught about placeholders. Step 5 already ships a *basic*
  injection and confirmed Holo3 copies a visible token verbatim into a `write`; 5a makes the
  teaching explicit and robust so the model reasons well across many tokens and edge cases. It tells
  the model three things and optionally reinforces them with a skill.
- **Build — the proxy request hook injects three parts** (input only, so `structured_outputs` is
  unaffected; every part carries tokens + class labels, **never** real values, so it is safe to sit
  in scrubbed history). Use the real nonce-namespaced tokens (§6); examples elide the suffix:
  1. **General scheme (static — same every request).** Draft:
     > "Some sensitive values on screen are hidden behind placeholder tokens written «CLASS_N»
     > (e.g. «EMAIL_1», «PHONE_2»); the real token also carries a short session suffix. Each token
     > stands for a real value you cannot see. Treat a token exactly as the real value of that
     > class — you may click it, type it, or otherwise act on it. To use a value, emit its token
     > **verbatim** (exact spelling, including the suffix) in the action field. Never invent, guess,
     > describe, or alter the underlying value. If a value you need is not shown as a token, do not
     > fabricate it."
  2. **On-screen + active-session manifest (dynamic — rebuilt each frame).** The proxy reads the
     vault's manifest for the *current* redacted frame and separately lists issued tokens that
     remain active for the private session, with their class:
     > "Placeholders visible now: «EMAIL_1» (email), «PHONE_1» (phone), «ADDRESS_1» (postal
     > address)."
     Current tokens preserve visual grounding. The value-free active-session list lets a normal
     multi-step task discover a value, navigate or focus another field, then reuse that exact
     issued token after it leaves the latest screenshot. Tokens not issued by the local session
     fail closed.
  3. **Stable-token warning (static).** Assignment is stable within a session. When several tokens
     share a class, the model must preserve the exact token associated with the earlier value/field.
- **Placement / cadence:** append the **scheme** and **duplicate warning** to the runtime's existing
  system-role message without overwriting its content. H Company's Holo chat template rejects two
  consecutive system messages, so never add a second one; if no system message exists, attach the
  static teaching to the current user observation. Attach the **manifest** to the current step's
  user/observation message so it sits next to the frame it describes. Re-inject every step (the
  manifest is per-frame; repeating the static parts is cheap and survives context truncation).
- **Reinforcing skill (recommended):** also ship a native HoloDesktop skill
  (`~/.holo/skills/plva-placeholders/SKILL.md`) carrying the **general scheme only** — skills are
  static, so they cannot hold the per-frame manifest or duplicate specifics, which stay in the proxy
  injection. The runtime loads the skill into context as a first-class capability doc, reinforcing
  the injection. Keep the proxy injection as the source of truth (guaranteed present, carries the
  dynamic parts); treat the skill as belt-and-suspenders.
- **Verify:** the manifest distinguishes exactly-current tokens from issued active-session tokens;
  forged tokens fail closed; a token can leave the current frame and still resolve in a later
  executed action; injection is input-only and logs carry tokens + class only, never values.

**Completed 2026-07-12:** every privacy-enabled request now receives a scheme and duplicate warning
merged into Holo's existing system message plus a value-free manifest immediately beside the
current screenshot. The manifest is produced atomically by the same vault-paint operation, lists
only token/class pairs, explicitly says `none` when the current frame has no tokens, removes older
injected manifests, and lists exact issued tokens that remain active for multi-step reuse. Forged
manifest tokens are rejected against the vault. Decorative guillemets are tolerated and removed
before local execution. A static `plva-placeholders` HoloDesktop skill reinforces the scheme and
is refreshed for each run; the proxy remains authoritative. Unit/integration tests cover current
and active membership, later-step reuse, empty manifests, malformed-token rejection, and absence
of cleartext. See
`Holo/verification/step-5a-placeholder-teaching.md`.

### ✅ Step 6 — Configurable per-class PII safety policy
- **Goal:** the user chooses, per class, `hide_use` / `approval` / `blocked` (and later variants).
- **Build:** a policy config (file + app UI, §9 Step 11) with sensible defaults (§6); the vault's
  resolve gate enforces the level; `blocked` never stores; `approval` defers to Step 7.
- **Verify:** flipping a class between levels changes behavior as specified; `blocked` values are
  never resolvable and never stored; defaults protect secrets out of the box.

**Completed 2026-07-12:** `SafetyPolicy` validates editable JSON defaults and unknown classes fail
closed. The same policy instance drives token assignment, vault resolution, the current-frame
manifest, and a `[PLVA_SECURITY_POLICY]` model instruction. `blocked` findings keep the detector's
opaque mask but receive no token or vault entry; `approval` tokens are stored but resolution is
denied until Step 7 supplies a local approver. The consumer app exposes all 15 classes and three
levels. See `Holo/verification/step-6-policy.md`.

### ✅ Step 6.5 — Tool-call channel spike: can Holo3 make tool calls via skills + prompting?
- **Why this is next (not a refinement of Step 6 — a standalone prerequisite).** Several remaining
  steps assume the CUA can **invoke a tool and consume its result**: the local reasoning tool and
  deterministic ops (Step 13), SPEAK / voice-read (Steps 8–9), spoken-prompt routing (Step 10), and
  any future tool. But Step 0 found Holo3 exposes `structured_outputs` **and no native `tools`**, so
  *whether* — and *how* — we can get a parseable, round-trippable tool call out of it is an **open,
  load-bearing question**. Settle it **once, here, before** building anything that depends on it.
  This is the general form of the SPEAK-specific probe in Step 8, which now specializes it.
- **Goal:** an empirical, recorded answer to *"can Holo3 emit a tool call the proxy can parse, and
  then consume the tool's result to continue the task?"* — plus a recommended mechanism the
  dependent steps all adopt.
- **Build — a Step-0-style, non-sensitive capture** (record the action/grammar schema and
  compliance only; **never** real frames, values, or transcripts). Exercise a trivial round-trip
  tool (e.g. `echo` / `add`, or `sort` from Step 13) with prompts that should trigger it, across the
  **candidate invocation channels**, ranked:
  1. **Native skill-declared tool.** Declare a callable tool in a HoloDesktop `SKILL.md` (reusing
     the `plva-placeholders` skill plumbing from Step 5a) and see whether the runtime/model emits an
     invocation for it.
  2. **Structured-action tool call.** Via proxy prompt-injection, get the model to emit the call
     inside its `structured_outputs` action schema (a designated action type / field). Tests whether
     the grammar admits a novel action and the model complies.
  3. **Free-text marker convention.** The model emits a parseable marker (e.g. `⟦TOOL sort …⟧`) in a
     free-text field it already produces; the proxy scans and executes. Loosest coupling — the
     fallback if the grammar resists.
  For **each** channel record four yes/nos: (a) grammar/template **permits** it; (b) model
  **actually emits** it on cue; (c) proxy can **parse** it deterministically; (d) **round-trip** —
  the proxy injects the tool *result* back and the model **consumes** it and continues. A tool call
  is only useful if **(d)** holds.
- **Verify:** a recorded matrix (channels × a–d), the captured schema, and a **recommendation** of
  the single channel Steps 7/8/9/10/13 will use — plus the **fallback** if none round-trip
  model-side (proxy/app-mediated pseudo-tools that need no model cooperation: the proxy runs the op
  and folds the result into the next injected observation, or the point-and-flag shape from Step 8).
  Nothing sensitive recorded.

**Completed 2026-07-12.** An authentic one-step Holo request was captured on loopback and all
provider-bound user text and pixels were replaced with a fixed synthetic `add(3, 4)` fixture. The
runtime schema has `structured_outputs` and an unconstrained string `tool_name`, but no native
`tools`; the temporary skill declaration was not loaded into the request. Native invocation
therefore failed, and the model's attempted structured action did not match a parseable argument
shape. The free-text marker completed the full emit → parse → local execute → result-consumption
round trip. Adopt the marker channel for Steps 7/8/9/10/13, always backed by the deterministic
proxy/app pseudo-tool path because marker compliance varied between otherwise identical probe
runs. The value-free matrix and schema fingerprint are in
`Holo/verification/step-6.5-tool-channel.md`; rerun with `uv run plva-tool-probe`.

### ✅/⚙️ Step 7 — Local approval capabilities built; LLM mediator automation pending
- **Goal:** `approval`-class resolutions and configurable risk flags are decided by a mediator
  against user-defined criteria, safely.
- **Build:** proxy consults the mediator on a tripped flag (synchronous gate for approvals; async
  advisor for steering); verdict `{approve|deny|modify, steering?, scope?}`; fail-closed on
  unreachable/timeout. Mediator runs in an OpenShell sandbox — **local model, zero egress** if it
  sees cleartext (§7) — reached via `forward service`; verify enforcement with `policy prove`.
- **Verify:** an `approval` value resolves only after approval and only within granted scope; a
  denied request blocks with no real value used; mediator sandbox has provably no egress; with the
  mediator down, everything gated fails closed.

**Completed locally 2026-07-12:** exact short-lived approval capabilities are now implemented in
the session vault and enforced atomically during action resolution. Grants bind issued token, tool,
argument path, optional target, TTL, and bounded uses; mismatch, expiry, exhaustion, and revocation
fail closed. The loopback proxy exposes value-free grant/list/revoke endpoints and the app Vault
view can authorize one exact private write. A local LLM mediator remains an automation/steering
extension, not a prerequisite for manual approval-gated use.

**Mediator component built 2026-07-12 (not yet bridged):** `plva_proxy.mediator` consults a local
Nemotron model over a loopback-pinned fail-closed client (`plva_proxy.local_llm`) for
`{approve|deny|modify, steering?, scope?}` verdicts against user criteria
(`config/mediator-criteria.json`), plus a trace watchdog (`should_review` deterministic trigger →
`review_trace` → continue/warn/halt) for killing a misbehaving CUA. Outage/malformed output fails
closed (deny/halt); verdicts echoing shown cleartext are withheld; an approving verdict's
`grant_kwargs()` maps 1:1 onto `SessionVault.grant_approval`. On this host the zero-egress
invariant is evidenced by an `lsof` audit (`plva-mediator probe`) per ADR-0001, not OpenShell.
Evidence: `Holo/verification/local-llm-mediator-executor.md`; runbook:
`Holo/docs/local-llm-runbook.md`.

### 🔲 Step 8 — SPEAK-mechanism spike (prerequisite for voice-read)
- **Goal:** determine empirically *how*, if at all, Holo3 can signal "read this aloud" — before
  building voice-read (Step 9). One instrumented run also captures Holo3's `structured_outputs`
  grammar and settles the Step 5 placeholder-cooperation question, so do it once and reuse.
- **Build:** a Step-0-style capture (non-sensitive: record the action/grammar schema, never frames
  or values), then with prompts like *"read me the most upvoted comment on this thread"* test the
  **top 3 candidate mechanisms**, ranked, recording for each whether the grammar permits it and
  whether the model actually emits it. **Specializes Step 6.5:** reuse the tool-call channel chosen
  there; this step only settles the SPEAK-specific payload/reference question layered on top of it:
  1. **Structured SPEAK (text payload).** Teach the model to emit a SPEAK carrying the text to
     speak (placeholders allowed for embedded PII). Tests whether `structured_outputs` permits a
     novel action/field at all, and whether the model complies and reproduces the text. Weakness:
     long-text transcription fidelity.
  2. **Point-and-flag (SPEAK-by-reference).** Teach the model to indicate a screen location
     (coordinates / region / DOM element) plus a "read here" flag — reusing an existing
     coordinate-bearing action shape where possible — and have local DOM (`browser_exec get_text`)
     or original-frame OCR do the actual reading. Tests whether the model can reliably point at the
     intended element ("most upvoted comment") and signal read-intent in a proxy-parseable form.
     Strength: exact text, no length limit, model emits no content.
  3. **Free-text marker in native output.** The model embeds a recognizable marker
     (e.g. `⟦SPEAK⟧…⟦/SPEAK⟧`) in a free-text field it already emits (thought/answer), which the
     proxy scans. The loosest-coupling fallback: tests whether *any* smuggled convention survives
     the grammar and is actually used.
- **Verify:** a recorded yes/no per option on (a) grammar-permitted and (b) model-complies, the
  captured schema, and a recommendation of which mechanism Step 9 builds — plus the fallback
  (app/user-triggered read: the model or user selects a region, the app extracts+speaks locally) if
  none work model-side. Nothing sensitive is recorded.

### 🔲 Step 9 — Voice-read out: local Gradium TTS + placeholder resolution
- **Goal:** the CUA can read parts of the screen aloud to the user, **including complex,
  reasoning-selected content and PII**, without any real value reaching the model or Gradium's cloud.
- **Build:** implement the mechanism chosen in Step 8 behind one **SPEAK contract** that carries
  *either* inline `text` (verbatim, placeholders allowed — resolved via the vault) *or* a `ref`
  (region / DOM element) that local `browser_exec get_text` or original-frame OCR reads exactly.
  The response hook resolves/extracts and hands the real text to a **local** Gradium TTS client
  (fail-closed, offline-verified), returning a benign no-op to the runtime (it has no speech
  executor). For the `ref` path, intercept the tool *result* so extracted text goes to TTS, never
  back into the model's context. Enforce the anti-feedback rule (§8.10). Ship the inline-`text`
  shape first (stub TTS → real model), then add `ref` for fidelity.
- **Verify:** "read the most upvoted comment" voices the correct text locally; a `text` SPEAK with
  `EMAIL_1` voices the real email; the real value / extracted text appears in **no** request to
  Overshoot and **no** Gradium cloud call; TTS output is not recaptured by audio-in.

### 🔲 Step 10 — Spoken prompt input (Gradium STT) — **LOWER PRIORITY**
- **Goal:** the user can *speak* the initial/continuing task prompt instead of typing it.
- **Build:** mic capture → Gradium STT → scrub (any PII in the spoken prompt goes through the vault
  before reaching the model) → hand to the app/runtime as the task. A local intent classifier
  routes utterances: app-control (local only), CUA-steering (scrubbed→injected), mediator-approval
  (→ Step 7).
- **Verify:** a spoken prompt runs the task identically to the typed prompt; spoken PII is
  placeholdered before the model sees it; "approve"-type utterances reach the mediator, not Overshoot.

### 🟨 Step 11 — The app (fast-track UI complete; later-step controls deferred)
- **Goal for this pass:** one polished app for the capabilities already built: typed tasks,
  per-class safety levels (Step 6), PLVA on/off, testing knobs, and live inspection views.
- **Build:** loopback-only UI/controller over the proxy + runtime; safety-policy editor; run/stop
  controls; redacted model-frame, local vault, OCR, and stream-filter views; collapsed Advanced lab.
  No prompt, real value, transcript, or frame is persisted or logged by the app.
- **Deferred extension points:** spoken prompts (Step 10), voice-read (Step 9), mediator approvals
  (Step 7), and audio context (Step 12) will be added after those steps exist.
- **Verify:** a non-technical typed run works end-to-end; safety changes take effect on the next
  run; the model viewer shows the proxy's redacted frame; browser storage is unused.

**Fast-track completed 2026-07-12:** `Holo/run_demo.sh` launches the consumer GUI. A live H Company
run completed through it while the app displayed the final redacted frame, three protected
regions, memory-only OCR/vault data, and stream-guard status. Vault and OCR cleartext are blurred
until local reveal. See `Holo/verification/step-11-demo.md`. Step 11 remains partially open only
for the explicitly deferred Step 7/9/10/12 controls.

### 🔲 Step 12 — Audio-in context: Gradium STT (loopback) → scrub → inject — **LOWER PRIORITY**
- **Goal:** live computer audio becomes redacted text context the CUA can reason with.
- **Build:** capture system/loopback audio (OS-specific — the fiddly part; macOS needs a virtual
  device); stream to Gradium STT (WebSocket); take **finalized** segments; run each through the
  **same** vault + Rampart scrub (shared placeholders with the screen); inject via the proxy
  request seam with a de-dupe cursor. Optionally sandbox the Gradium client (egress only to
  Gradium). Note the trusted-boundary assumption: Gradium's cloud hears pre-redaction audio.
- **Verify:** a spoken value appears to the model only as a placeholder and resolves like an
  on-screen value; with audio disabled, behavior is unchanged.

### 🔲 Step 13 — Placeholder-preserving semantic operations (content-dependent reasoning without cleartext) — **NEW**
- **Problem this solves.** Some tasks need the *semantic content* of a redacted value, not just its
  identity: *"sort this list of names," "pick the most recent date," "which of these is a work
  email," "dedupe this list," "group these by topic."* With only opaque placeholders (`NAME_1`,
  `NAME_2`, …) the CUA has nothing to reason over — it cannot order letters it cannot see. Step 13
  gives the CUA a way to **delegate a content-dependent computation to a local executor that may see
  cleartext but hands back only placeholders**, so the model gains the *answer* (an ordering /
  selection / aggregate) while no real value ever reaches it or Overshoot.
- **Goal:** content-dependent operations over hidden values work end-to-end, with the executor's
  return to the model provably value-free (§8.12).
- **Two executor shapes — one shared privacy contract; choose per operation:**
  - **(A) Deterministic local tools.** A small fixed library of pure functions over *token lists* —
    `sort`, `filter`, `dedupe`, `group`, `min`/`max`, `count_matching`, `compare`. The proxy
    resolves the tokens → real values in the vault, runs the op **locally**, and returns the result
    **as reordered/selected tokens** (or a scalar for genuine aggregates). Output is value-free *by
    construction* (the contract emits tokens, never values), so leak-safety is trivially provable.
    Fast, deterministic, no extra model.
  - **(B) Local reasoning model (NemoTron in the OpenShell sandbox).** Reuse the §7 mediator sandbox
    pattern: a **fully local, zero-egress** model the CUA delegates a *fuzzy* sub-task to ("order
    these by relevance," "group by sentiment," "which reads like a personal vs. work address"). It
    resolves tokens to cleartext, reasons, and must return **only placeholders** (a permutation /
    subset) plus at most a value-free rationale. Handles arbitrary ops the deterministic library
    can't enumerate. *(This local **helper** does not conflict with §3's fixed CUA-model choice —
    NemoTron is not the CUA or its harness, just an on-device reasoning tool consulted by the proxy,
    exactly like the mediator.)*
- **Shared privacy contract (load-bearing — see §8.12):**
  - The executor may see cleartext **only** when it is local with **zero egress** (same rule as the
    mediator, §7); (A) runs in-proxy, (B) runs in the OpenShell sandbox.
  - Its **return to the CUA carries no real value** — only tokens the model already holds
    (orderings/selections add ~zero information beyond the ordering itself). **Scalar / derived**
    returns (counts, "3 of these are `.edu`", an extracted substring) *can* leak and are gated or
    aggregated with care, never emitted raw.
  - **Invocation depends on the Step 6.5 tool-call spike.** Holo3 exposes `structured_outputs` but
    has **no native `tools`** (§Step 0), so *how* the model emits the call (native skill-declared
    tool vs. structured action field vs. free-text marker) is settled once by **Step 6.5** — adopt
    its recommended channel rather than re-spiking it here.
- **Build:** ship **(A) first** — the deterministic tool set behind the existing response-leg seam
  (Step 3), resolving in-vault and returning tokens; then add **(B)** as the general fallback in the
  Step 7 sandbox, with a **two-layer output filter:**
  1. **Primary — structured token-only return.** Constrain (B) to return its answer as
     tokens/orderings/selections; reject (and re-ask / fail closed on) any structured return
     carrying a non-token value.
  2. **Backstop — Rampart reparse.** Run any free-text the model emits (e.g. a rationale) through
     the **same §8.9 scrub the history leg already uses** — plain vault match, then Rampart
     reclassify — re-tokenizing recognized values and **failing closed on PII it cannot map to a
     vault token.** This reuses existing, trusted machinery, so (B) can return a richer rationale.
  > **Boundary — do not over-trust reparse.** It catches a *literal* value the LLM echoes, but
  > Rampart has **finite recall** (§8.9's stated primary residual risk) and it does **not** catch
  > *inferential* leakage — a rationale that describes a property of the hidden value ("the one
  > starting with 'A'", "the `.edu` address") leaks without emitting the value. So reparse is
  > **defense-in-depth**, not the guarantee: keep the token-only structured return as the primary
  > contract, prefer minimal/no free rationale, and treat any rationale as the same controlled
  > semantic-leak surface as a scalar aggregate (§8.12).
- **Verify:** "sort these names" returns the tokens in the true alphabetical order of their hidden
  values while **no** name text appears in any request to Overshoot or any log; a `filter` returns
  exactly the correct token subset; **(B)**'s structured filter drops any response with a non-token
  value, and a rationale that echoes a real value is re-tokenized (or fails closed) by the Rampart
  reparse — while a crafted *inferential* leak is caught as a known gap, not a silent pass; with the
  sandbox down, **(B)** fails closed while **(A)** still works; no log carries a real value.

> **Recommendation for the demo — lead with (A), the deterministic tools.** For a live demo it is
> the stronger choice on every axis that matters on stage: it is **deterministic** (no flaky model
> output mid-demo), **fast** (no extra inference in the loop), and its privacy story is **provable
> in front of the audience** — the tool contract can *only* emit tokens, so you can show a list of
> real names get sorted correctly while the model/provider transcript contains **zero** real names.
> "Sort this list" is itself a deterministic op, so (A) showcases exactly the headline capability
> with none of (B)'s live risk (extra latency, plus a general LLM that might answer in cleartext and
> silently break the guarantee). Keep **(B)** as the *"and it generalizes to arbitrary reasoning"*
> talking point / stretch goal — it is little extra plumbing because it reuses the Step 7 sandbox,
> but it should not be on the critical path for the demo.

**Executor (B) component built 2026-07-12 (not yet bridged):** `plva_proxy.semantic_executor`
runs `sort`/`select` over issued tokens through the same local model. The model answers with
exact item texts (grammar-constrained to that enum on grammar-capable servers); the executor maps
answers back to tokens and discards the completion, so the return to the CUA is tokens-only by
construction. Live: correct alphabetical sort and a correct "personal vs work email" select with
leak-clean injected observations; the primary filter caught and failed closed on a real
cleartext-echo attempt by the 4B model during development. Kind names and `request_id` align with
the Step 6.5 marker channel (`⟦PLVA_TOOL:sort:request_42⟧`), and `observation_text()` renders the
value-free result line for injection. Deterministic library (A) remains unbuilt and separate.
Evidence: `Holo/verification/local-llm-mediator-executor.md`.

---

**Cross-cutting — local-model acceleration (separate agent, keep in mind).** The redaction detector
is being sped up with a **persistent warm worker**: the visual model runs on **WebGPU** where
available, OCR runs concurrently on a separate WASM runtime, sessions stay warm during a CUA burst
and release after idle, and exact repeated frames hit a bounded redacted-output-only memory cache.
Apple **CoreML** execution is part of the same effort. **Implications for anyone continuing this
work:** (1) local models (detector now; local TTS in Step 9; a local mediator model in Step 7) must
sit behind a **pluggable client** with `auto|webgpu|wasm|coreml` backends — never hardcode one;
(2) the frozen baseline stays the **correctness oracle**; (3) acceleration must not weaken
fail-closed behavior or add egress (§8.11); (4) the accelerated worker source lives in
`Holo/redactor-worker/` and its Python client in `Holo/src/plva_proxy/redactor.py`
(`AcceleratedRedactor`).

---

## 10. Definition of done (this component)

- HoloDesktop runs `Hcompany/Holo3-35B-A3B` via Overshoot, with the proxy as sole provider egress.
- The proxy intercepts both directions, redacts outbound frames, **resolves inbound placeholders in
  executed fields**, scrubs history (plain + Rampart), and handles streaming.
- Per-class **safety levels** are configurable; `blocked` classes are never resolvable; `approval`
  classes resolve only via the **mediator**, which runs local-and-zero-egress when it sees cleartext.
- **Voice-read** speaks real values locally with none reaching the model or Gradium's cloud; audio-in
  context is placeholdered before the model; the app runs a task from a typed (or spoken) prompt.
- Every §8 constraint has a direct test; the history-scrub weak point is tested adversarially.
- No key, real value, transcript, or frame is ever committed or logged.
- The detector remains a clean plug-in seam (real, accelerated detector replaces the stub with no
  proxy changes), and all local models stay behind pluggable, fail-closed, offline clients.
