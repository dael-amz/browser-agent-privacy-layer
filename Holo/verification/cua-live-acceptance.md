# Live PLVA-on CUA acceptance

Date: 2026-07-12

Status: **PASS**

## Controlled task

A loopback-only two-step form displayed one synthetic email on step 1. Step 2 removed the source
value, required Holo to reuse the issued session token, and accepted the form only when the local
POST exactly matched the original value. The task prompt contained no private value.

The real Holo runtime used the H Company provider through the PLVA proxy with Vision/Core ML
redaction, history scrub, placeholder teaching, response resolution, and the loopback-only egress
guard enabled.

## Result

```text
task_completed=true
runtime_reported_success=true
visited_private_free_step=true
submitted=true
destination_matches_source=true
wall_time_ms=79423.551
```

The success decision came from the local fixture server, not the runtime exit code. No Holo,
proxy, benchmark, or frame-bearing runtime process remained afterward. Temporary runtime and audit
files were removed.

## Defects found and closed during the run

1. Holo's local CLI-to-Agent-API connection used a second loopback port. The first egress policy
   incorrectly treated it as a violation. The enforced invariant is now no external CUA egress;
   local control and proxy ports remain allowed.
2. Blocked semantic spans in provider-bound text history were incorrectly passed to the vault,
   which must reject blocked values. They now become opaque non-resolvable history masks.
3. Holo's runtime daemonizes outside the CLI process group. The monitor and cleanup now discover
   and track that detached process explicitly, and refuse to start over an existing runtime.
4. After local typing, a horizontally scrolled input exposed a long suffix of the vaulted email.
   The semantic classifier did not label the clipped suffix. A deterministic vault-backed residual
   screen guard now re-masks OCR regions matching any issued value or meaningful prefix/suffix.
5. The live fixture is pre-opened and visually separates the private source from the Continue
   control, removing navigation and overlay ambiguity from the acceptance task.

## Remaining optional infrastructure step

The nonprivileged per-run egress guard is active and fail-closed. The additional macOS PF boundary
still requires administrator bootstrap described in `docs/egress/bootstrap-pf.md`.
