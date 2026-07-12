# Step 6 verification — per-class security policy

Verified 2026-07-12 using synthetic values only.

## Implemented contract

- `config/privacy-policy.json` supplies editable defaults for 15 normalized PII classes.
- `PLVA_POLICY_FILE` selects a different local file; `PLVA_POLICY_JSON` is the memory-only run
  override used by the GUI.
- Unknown classes and malformed configurations fail closed.
- `hide_use` values receive a session token and may resolve only in executed action fields.
- `approval` values receive a token and require an exact, short-lived local capability. The Vault
  view can grant one private write; mismatched, expired, exhausted, or revoked grants fail closed.
- `blocked` values receive no placeholder, are never stored, and retain the detector's opaque mask.
- The current-frame manifest includes each token's class and level without including its value.
- Holo receives the active value-free policy in its existing system message.

## Checks

Unit tests cover policy validation, default and override behavior, unknown classes, all three
resolution gates, blocked storage/manifest exclusion, prompt contents, and policy-safe viewer
output. The complete proxy suite, Ruff, mypy, and shell syntax checks pass; see the current test
output in the project handoff.
