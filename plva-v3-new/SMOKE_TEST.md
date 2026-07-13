# PLVA v3 smoke test

Run:

```sh
node bin/smoke-test.mjs
```

The test runs the frozen high-recall pipeline on the bundled fake ATS screenshot
and requires:

- nonzero final masks and masked pixels;
- nonzero visual detections from the isolated visual worker with no fallback;
- unchanged dimensions;
- exact opaque redaction color inside every mask;
- zero changed pixels outside all masks;
- full Rampart mode with no degraded inference path;
- the v10 detector, checkpoint, and threshold hashes;
- the frozen one-frame visual runtime-policy hash and single-thread WASM assets;
- local-only resource loading with external networking blocked;
- a report containing geometry and timings but no OCR text, email-like value,
  filename, or local filesystem path.

This one synthetic fixture validates packaging and execution, not general model
quality. Selection metrics and broader guard evidence are recorded in the model
and training manifests.

Run `node bin/tall-regression.mjs` separately to require at least 98% coverage
for every one of the five small header truths in the bundled 1280x3060 WebPII
cart fixture. That regression exercises the contextual OCR identity path without
enabling uncalibrated visual tiling.
