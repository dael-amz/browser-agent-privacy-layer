# Step 5a verification — richer placeholder teaching

Verified 2026-07-12 with synthetic values only.

## Implemented contract

- A new system message teaches the scheme without replacing Holo's system prompt.
- The static prompt requires the exact inner token in executed action fields and forbids guessing,
  mutation, decorative guillemets, and stale-token reuse.
- The same atomic vault-and-paint operation emits a value-free `{token, class}` manifest.
- Only the latest observation's frame manifest is attached beside its image.
- Prior PLVA system messages and frame manifests are removed before fresh injection.
- A current frame without tokens receives an explicit `none` manifest.
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

Tests cover current-frame-only membership, stale-token removal, the empty-frame case, token/class
validation, system-message replacement, observation placement, cleartext exclusion, action-only
resolution, JSON/SSE proxy behavior, and privacy-safe logs.
