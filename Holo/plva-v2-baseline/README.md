# PLVA v2 baseline harness

This folder is a frozen, standalone development harness for the **old PLVA v2**
visual + RapidOCR + Rampart pipeline. It accepts one screenshot, permanently
burns detected regions into a PNG, and writes a geometry-only JSON report.

The frozen detector is **development-only and not release eligible**. Its
source checkpoint is AGPL-3.0-only; a closed-source product needs an
Ultralytics Enterprise license. Its minimum secret-class recall was zero and
its WebPII quick-100 diagnostic missed 6 of 7 secret boxes. Do not treat this
bundle as a production privacy boundary.

## Run it

Requirements: Node 20+ and a locally installed Chrome, Chromium, or Edge.
The built CLI does not need Vite, a dev server, Python, or a network download.

```sh
node bin/plva-v2.mjs screenshot.png \
  --output screenshot.redacted.png \
  --report screenshot.redacted.json \
  --profile high-recall
```

Use `--profile balanced` for fewer uncertainty masks. Use `--force` to replace
existing outputs. `node bin/plva-v2.mjs --help` lists all options.

The CLI starts a random-port server bound only to `127.0.0.1`, starts a fresh
headless Chrome profile, loads all inference assets locally, receives the PNG
and sanitized report, and shuts both down. External browser traffic is routed
to a closed loopback proxy and wildcard DNS is mapped to `0.0.0.0`; only
`127.0.0.1` is excluded. The runner also rejects any non-local page resource.
No recognized OCR text or filesystem path is retained in the report.

## Verify and smoke-test

```sh
node bin/verify-snapshot.mjs
node bin/smoke-test.mjs
```

The integrity check hashes the source/model snapshot, runnable harness build,
and original app build. It rejects symbolic links and any path escaping this
folder. The smoke test uses the single bundled synthetic ATS fixture; the full
ATS holdout is intentionally excluded. It requires nonzero masks and verifies:

- decoded PNG dimensions equal the input dimensions;
- every masked pixel has the exact redaction RGB value;
- no pixel outside a final mask changed after PNG encode/decode;
- full Rampart mode loaded, runtime resources stayed local, and the JSON has no
  OCR text, email-like value, or filesystem path.

## Folder map

- `dist/`: standalone CLI runner build.
- `baseline-app-dist/`: byte-for-byte copy of the authoritative 15-file v2 app
  build, retained for integrator comparison.
- `runtime/src/`: exact source closure used for detection, OCR, semantic policy,
  fusion, and rendering.
- `runtime/models/`: frozen OCR and Rampart assets.
- `runtime/training/artifacts/.../visual/`: old detector, model card, manifests,
  goldens, conversion evidence, and cross-runtime report.
- `snapshot.json`: release status, known diagnostic, and source/model hashes.
- `SHA256SUMS` and `*-integrity.json`: machine-readable integrity evidence.
- `PROVENANCE.md` and `LICENSES.md`: origin and license notes.

## Rebuild

The checked-in `dist/` is already runnable. To reproduce it from the bundled
source closure:

```sh
npm ci
npm run build
```

Do not run `npm run freeze` merely to make a modified build pass verification;
that command creates new integrity evidence and therefore a new snapshot. Keep
this folder immutable when using it as the v2 comparison baseline.

## Known implementation limits

- Browser RapidOCR postprocessing is the v2 axis-aligned approximation, not
  full polygon/unclip/perspective/vertical-rotation parity.
- Cue-to-value association across separate OCR boxes is not implemented.
- COOP is enabled, but COEP is intentionally not: enabling `crossOriginIsolated`
  activates ONNX Runtime's threaded worker route, while this exact frozen Vite
  bundle is not a worker-safe entry. It therefore stays on its proven
  single-thread WASM path.
- The v2 WebPII result is diagnostic, not a frozen release holdout.
