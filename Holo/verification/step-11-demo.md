# Step 11 fast-track verification — consumer demo

Verified 2026-07-12. Future Step 7/9/10/12 features are intentionally deferred.

## Implemented surface

- Typed task, run/stop controls, and a PLVA master toggle.
- Per-class `hide_use`, `approval`, and `blocked` controls that apply on the next run.
- Redacted model-frame, memory-only agent trace, model-call history, vault, OCR, and stream-guard
  views.
- Trace output, vault values, and OCR text blurred until an explicit local reveal.
- Advanced Lab tab for provider, OCR mode, worker lifecycle, and every Step 5/5a switch.
- Loopback-only server, `Cache-Control: no-store`, no browser storage, and safe allowlisted runner
  events rather than raw process logs.

## Live check

A synthetic H Company task launched through the GUI controller and completed successfully. The
captured demo state showed one model frame, three protected regions, a populated local vault/OCR
snapshot, and stream-guard diagnostics. The proxy shredded its temporary run directory. Browser
visual QA covered the initial and completed states; screenshots were not retained in the repo.

Run it with:

```bash
./run_demo.sh
```
