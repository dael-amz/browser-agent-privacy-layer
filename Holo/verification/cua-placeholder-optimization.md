# CUA + placeholder pipeline optimization

Date: 2026-07-12

Status: **LOCAL IMPLEMENTATION PASS; LIVE CUA COMPARISON STILL REQUIRED**

## Implemented

- Bound launcher readiness to a random per-process instance token. A stale process on the fixed
  loopback port can no longer be mistaken for the proxy started for the current private session.
- Kept exact locally issued tokens active across steps. The current-frame manifest remains
  distinct, while a value-free active-session list allows discover → navigate/focus → type flows.
- Validated every manifest token, class, and policy level against the local vault before egress.
- Accepted singular/plural Holo calls, direct action envelopes, nested actions, and function-call
  arguments. Malformed tool-call envelopes fail closed. Decorative guillemets are stripped before
  local execution, while notes/reasoning remain placeholdered.
- Cached semantic history classification per unchanged safe text. Growing message histories now
  classify only new or changed text rather than invalidating one whole-tuple cache.
- Increased the redacted-frame hot cache from 4 to 32 entries and added a session-aware cache of
  final painted frames + manifests. Exact historical frames avoid both detector work and repeated
  PNG chip painting.
- Replaced JSON serialize/parse cloning inside hot hooks with deep copies, which share immutable
  base64 strings instead of repeatedly encoding them.
- Refresh the native Holo placeholder skill on every privacy-enabled run and omit the raw task
  prompt from launcher logs.

## Verification

```text
146 passed in 2.67s
coverage: 83.53% (required: 80%)
targeted Ruff: passed
privacy.py strict mypy: passed
run_step1.sh syntax: passed
1920x1243 synthetic painted-frame cache: 29.216 ms first pass; 0.023 ms median hit;
  0.025 ms p95 hit (100 exact repeats)
```

The integration suite now proves that a value can appear in frame 1, disappear from the current
frame in frame 2, remain absent from all provider-bound JSON, and still resolve from its issued
token into the frame-2 local action.

## Residual release blockers

- The repository still delegates click execution, UI settling, retry policy, and success
  verification to the closed `hai-agent-runtime`; this proxy cannot independently prove that exit
  code 0 means the requested UI postcondition was reached.
- The current detector is release eligible. Earlier `plva-v2-baseline` snapshot metadata describes
  a superseded development artifact and must not be used to characterize the current detector.
- `approval` tokens remain denied until the local mediator exists, and `blocked` values remain
  intentionally unusable. Authentication tasks require an explicit policy override today.
- A live PLVA-off versus PLVA-on task-success benchmark was not run because no controlled desktop
  fixture/task and provider-spend authorization were supplied for this optimization pass.

## Next acceptance gate

Run the same controlled form-fill corpus with PLVA off and on. Require zero cleartext/provider
leaks, 100% issued-token resolution, PLVA-on success within five percentage points of PLVA-off,
warm p50 below 200 ms, p95 below 350 ms, and no material per-step growth from unchanged history.
This is a CUA integration/performance gate, not a detector release-eligibility blocker.
