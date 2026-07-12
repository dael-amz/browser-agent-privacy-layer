# PLVA accelerated Core ML hybrid redactor

This package contains two parallel Apple-accelerated backends and does not replace the stable
`redactor-worker/` path. The original experimental backend uses Core ML RapidOCR. The low-latency
backend uses native Apple Vision OCR plus the Core ML visual detector and Rampart, retaining
structured PII rules, high-recall uncertain masking, region fusion, opaque rendering, and
vault-ready findings.

Visual and OCR branches run concurrently. Ordinary OCR crops use a six-item 320-pixel recognizer
batch; only uncertain wide text is retried through a 1536-pixel single-crop model. On the bundled
960×960 ATS fixture, all 17 OCR regions are recognized, semantic and fused-region counts match the
frozen pipeline, output pixels are identical, and the latest warm side-by-side run measured roughly
665 ms instead of about 3.2 seconds for the frozen browser pipeline. Core ML may retain CPU fallback
for unsupported graph nodes.

The native Vision `cascade` path runs fast full-frame recognition, classifies it, and retries only
sensitive or uncertain regions using accurate recognition. On the same fixture it retains the same
one sensitive/fused region and label while measuring about 113–125 ms warm, versus roughly 665 ms
for the Core ML RapidOCR path in the latest side-by-side run.

Run from this directory:

```bash
$HOME/.local/bin/uv run plva-ane-probe --baseline ../plva-v2-baseline
```

Use `--visual-model ../plvas-v3/harness/plva-v2-baseline/runtime/training/artifacts/plva-visual-agpl-test-v2/visual/detector.onnx`
to select the v3 repository's detector without changing the OCR or Rampart inputs.

Run the live screen viewer:

```bash
$HOME/.local/bin/uv run plva-ane-live --baseline ../plva-v2-baseline
open http://127.0.0.1:18083/
```

Run the native Vision pipeline through the shared proxy viewer without contacting a provider:

```bash
cd ..
$HOME/.local/bin/uv run plva-live --redact-engine vision --vision-mode cascade
open http://127.0.0.1:18082/viewer
```

macOS may require Screen Recording permission. For a privacy-safe static smoke test:

```bash
$HOME/.local/bin/uv run plva-ane-live \
  --baseline ../plva-v2-baseline \
  --fixture ../plva-v2-baseline/fixtures/ats-smoke.png
```

The generated fixed models and compiled cache stay under `.cache/` and contain no frames. The probe
uses only the synthetic baseline fixture and emits timing/numerical metadata.

OCR findings are retained in memory for the latest frame and emitted at:

```text
http://127.0.0.1:18083/findings
```

Each finding includes recognized text, bounds, OCR confidence, semantic labels, sources, and exact
heuristic/NER value spans suitable for a future local vault. Findings are never logged or persisted.
The endpoint is loopback-only and `no-store`, but its contents are sensitive.

The Vision worker is selectable in the outbound proxy with `--redact-engine vision` and remains
opt-in while a broader positive-PII fixture suite validates recall. `HybridVisionRedactor.process()`
and `HybridANERedactor.process()` remain separate so the RapidOCR implementation stays available.

The same persistent Vision worker also exposes batched history classification through its already
warm Core ML Rampart session. A measured uncached call took about 46 ms and its bounded in-memory
result cache returned the identical classification in under 0.1 ms. No history or values are
written to disk.
