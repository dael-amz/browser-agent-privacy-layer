# Step 5a verification — richer placeholder teaching

Verified 2026-07-12 with synthetic values only.

## Implemented contract

- The scheme is appended inside Holo's existing system message without replacing its content or
  creating a second system message; user-observation fallback handles requests without one.
- The static prompt requires an exact issued token in executed action fields and forbids guessing
  or mutation. Decorative guillemets are tolerated and stripped locally before execution.
- A separate `[PLVA_SECURITY_POLICY]` block lists the active per-class levels; manifests label each
  token with its level, and the native skill tells Holo that the live policy is authoritative.
- The same atomic vault-and-paint operation emits a value-free `{token, class}` manifest.
- Only the latest observation's frame manifest is attached beside its image.
- Prior PLVA system messages and frame manifests are removed before fresh injection.
- A current frame without tokens receives an explicit `none` current-frame manifest plus a
  value-free list of issued tokens still active for the private session.
- Manifest metadata is removed before provider forwarding and malformed/forged entries fail closed.
- `holo-skills/plva-placeholders/SKILL.md` is installed as the native `plva-placeholders` skill.

## Checks

```text
.venv/bin/ruff check src tests                           passed
.venv/bin/mypy src                                      passed
.venv/bin/pytest -q                                     107 passed
jq empty ~/.holo/settings.json                          passed
workspace/installed SKILL.md byte comparison            passed
```

Tests cover current/active membership, forged-token rejection, multi-step later-frame reuse, the
empty-frame case, token/class validation, single-system-message merging, observation placement,
cleartext exclusion, action-only resolution, JSON/SSE proxy behavior, and privacy-safe logs.

An H Company schema capture showed that Holo already supplies a 27,165-character system message.
The original PLVA insertion created a second consecutive system message and received HTTP 400.
Merging the identical PLVA content into the existing message returned HTTP 200; the final all-on
pipeline (including scheme, duplicate warning, manifest, skill, chips, scrub, and resolution) also
returned HTTP 200 and exited normally. The successful request was larger than the rejected one,
ruling out request size.

## Provider compatibility isolation

All switches default to `1`; use fake on-screen data when disabling them. To isolate a provider
400 while retaining the vault/chips, start with all model-prompt sources disabled:

```bash
PLVA_PRIVACY_SKILL=0 \
PLVA_PRIVACY_SCHEME=0 \
PLVA_PRIVACY_DUPLICATE_WARNING=0 \
PLVA_PRIVACY_MANIFEST=0 \
./run_step1.sh "synthetic diagnostic task"
```

Then enable `PLVA_PRIVACY_SCHEME`, `PLVA_PRIVACY_DUPLICATE_WARNING`,
`PLVA_PRIVACY_MANIFEST`, `PLVA_PRIVACY_POLICY_TEACHING`, and `PLVA_PRIVACY_SKILL` one at a time. The independent
`PLVA_PRIVACY_CHIPS`, `PLVA_PRIVACY_HISTORY_SCRUB`, and `PLVA_PRIVACY_RESOLUTION` switches isolate
the remaining Step 5 transformations. `PLVA_PRIVACY=0` remains the all-off control.
