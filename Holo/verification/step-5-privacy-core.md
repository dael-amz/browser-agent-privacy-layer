# Step 5 — privacy core verification

Date: 2026-07-12

Status: **PASS**

## Built

- Session-only nonce-namespaced vault with stable canonical assignment and explicit disposal.
- Placeholder chips painted on the same bounding boxes and from the same Vision OCR/Rampart values
  used by the vault; no tagging OCR rerun.
- Proxy-injected placeholder instructions.
- Executed-action-field resolution for singular and plural Holo tool-call shapes; reasoning and
  notes remain placeholdered; forged or unknown tokens fail closed.
- Request-history scrub: plain vault match first, then batched Core ML Rampart classification in
  the persistent Vision worker. Valid placeholder syntax is masked from Rampart classification to
  prevent recursive reclassification.
- Deterministic detector stub with detect-nothing and configurable fixture-span modes.

## Evidence

- Deterministic store → paint → resolve → next-request scrub loop passed.
- All injected request-classifier failures forwarded nothing; unknown response placeholders were
  not returned to the executor.
- Privacy-log assertions found no fixture value.
- Real Vision fixture: one fused mask, one stable vault placeholder, one visibly painted chip;
  repeated history scrubs were stable and removed the value.
- Synthetic Holo provider check: Holo emitted the exact visible token in `write`; the proxy resolved
  the executed field to a current-session vault value, while the token in reasoning was unchanged.
  No real desktop frame or real PII was used.
- Core ML Rampart history pass measured about 46 ms uncached and under 0.1 ms on its bounded
  memory-only cache.
- Main gate: 103 tests pass with at least 80% coverage; strict mypy and Ruff pass.
- Core ML package: 17 tests pass; strict mypy and Ruff pass.

## Operator test

```bash
PLVA_PROVIDER=hcompany \
PLVA_REDACT=1 \
PLVA_REDACT_ENGINE=vision \
PLVA_VISION_MODE=cascade \
./run_step1.sh "Use the visible email placeholder in the focused field"
```

Watch `http://127.0.0.1:18081/viewer`. `/viewer/findings` is loopback-only and contains sensitive
cleartext for vault development; it is `no-store` and never logged or persisted.
