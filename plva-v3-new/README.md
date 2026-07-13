# PLVA v3 screenshot-redaction harness

This is a standalone, local screenshot-to-redacted-PNG harness for the PLVA v3
development pipeline. It combines the safety-selected v10 visual detector,
RapidOCR, Rampart semantic classification, deterministic secret rules, and
constrained cross-box form cue association. Recognized OCR text stays in browser
memory and is never written to the report.

The bundle is **not a production privacy boundary**. The detector is an
AGPL-3.0-only development artifact, the training schedule was stopped after 35
durable epochs, and the safety selector retained the first v10 epoch because no
later checkpoint improved its privacy objective. Closed-source use requires an
Ultralytics Enterprise license plus review of every model and dataset obligation.

## Run

Requirements: Node 20+ and Chrome, Chromium, or Edge.

```sh
node bin/plva-v3.mjs screenshot.png \
  --output screenshot.redacted.png \
  --report screenshot.redacted.json \
  --profile high-recall
```

The default high-recall profile is the checkpoint-bound calibrated profile.
`balanced` is retained for v2 CLI compatibility but is not calibrated for the
v10 detector.

The CLI verifies the frozen source, runtime, model, and executable hashes before
every run. It starts an ephemeral loopback-only server and a fresh headless
browser, blocks external networking, burns opaque masks into a new PNG, decodes
that PNG again, and proves that every masked pixel has the redaction color while
every pixel outside the masks is unchanged.

## What changed from v2

- Replaced the old detector with safety-best v10 FP32 ONNX (`2ed3f25c…b928`).
- Loads and hashes the exact checkpoint-bound threshold artifact at runtime.
- Starts visual, OCR, and Rampart model loading together. The visual detector
  runs in its own single-thread worker while OCR runs on the page; a worker
  failure retries once through the serialized main-thread runtime.
- Keeps the stable single-thread WASM configuration and measured six-crop OCR
  recognition batch. The standalone build removes a redundant 26.8 MB detector
  WASM copy and binds both visual paths to one hash-checked runtime asset.
- Uses class-indexed NMS with identical output semantics and less comparison work.
- Applies screenshot-specific default-deny policy, including city, state, and
  postal code that Rampart's chat policy intentionally keeps.
- Associates structured form labels such as “Email” or “Date of birth” with a
  nearby plausible value, while refusing speculative password/secret pairing.
- Projects tightly matched commerce identity headers such as “Hello, Name”,
  “Deliver to Name”, and “Name pay for this order” back into pixel regions.
- Rejects an input path reused as either output or report.
- Verifies pixels in bounded stripes and rejects images over 16,000,000 pixels
  or 16,384 pixels on either axis before allocating full pipeline canvases.
- Reports the actual visual engine, fallback status, tile count, and a frozen
  runtime-policy hash. V3 deliberately keeps the calibrated one-frame detector
  policy; an uncalibrated tiling experiment increased false masks and was rejected.
- Integrity manifests now reject unlisted files inside sealed runtime/build trees.

On the bundled cold-run benchmark, median browser-pipeline latency improved from
3354 ms in v2 to 3103 ms in v3 (7.5%) while the smoke fixture went from zero to
three visual detections. See `BENCHMARK.md` for the exact runs and limitations.

## Known limitation

Very tall full-page screenshots can shrink small text below visual-detector
resolution. Experimental visual tiling increased false masks and was rejected.
The bundled contextual OCR rules recover all 5/5 truths in the included
1280x3060 WebPII cart regression, but that one fixture does not prove arbitrary
full-page coverage. Use ordinary viewport screenshots when possible and do not
treat this development harness as a production privacy boundary until broader
runtime guard evaluation passes.

## Verify

```sh
node bin/verify-snapshot.mjs
node bin/smoke-test.mjs
node bin/tall-regression.mjs
```

The built `dist/` runs without `node_modules`, Python, Vite, or network access.
To rebuild from the bundled source closure:

```sh
npm ci
npm run build
```

Rebuilding creates new executable bytes. Do not run `npm run freeze` unless you
intend to create and review a new integrity snapshot.

## Folder map

- `dist/`: standalone browser runner and all inference assets.
- `runtime/src/`: exact v3 OCR, semantic, detector, fusion, and renderer source.
- `runtime/models/`: pinned RapidOCR and Rampart assets.
- `runtime/training/artifacts/`: detector, thresholds, model/training manifests,
  goldens, provenance, and Node/WASM parity evidence.
- `snapshot.json`, `*-integrity.json`, `SHA256SUMS`: immutable bundle evidence.
- `BENCHMARK.md`: reproducible v2/v3 cold-run measurements.
