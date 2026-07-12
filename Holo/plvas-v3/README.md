# PLVA hybrid local playground

A local browser demo for screenshot and text PII redaction. The screenshot path
fuses the development PLVA visual detector with local RapidOCR, Rampart semantic
classification, and deterministic secret-field rules; the text path uses
[`nationaldesignstudio/rampart`](https://huggingface.co/nationaldesignstudio/rampart).

## Browser playground

```bash
npm install
npm run dev
```

Open the URL printed by Vite (normally <http://127.0.0.1:5173>). The first full
visual redaction loads the bundled 10.4 MB ONNX detector, 12 MB of RapidOCR ONNX
models, and the checked-in q4 Rampart model. Choose, drag, or paste a screenshot,
click **Redact screenshot**, inspect the opaque masks and their evidence labels,
and download the result as a PNG with the original pixel dimensions. Screenshot
pixels and recognized text are processed in the browser and are not uploaded or
printed into the UI.

The pipeline is:

1. The visual detector proposes sensitive regions directly from pixels.
2. RapidOCR detects text lines and recognizes them locally.
3. Rampart plus deterministic card/CVC/credential/document rules decides which
   OCR lines are sensitive. The MVP masks the whole line rather than exposing or
   attempting fragile character-level geometry.
4. Overlapping visual and OCR-semantic boxes are fused, then permanently painted
   black into a new canvas.

The visual detector is a development candidate trained and tuned on limited
synthetic/WebPII-derived data. OCR box extraction is an axis-aligned browser
approximation of RapidOCR's DBNet polygon postprocessor. Either path can miss PII,
so use fake data and inspect every output. The high-recall setting intentionally
uses a low CVC confidence threshold and may over-redact.

Rampart assets are served from `models/semantic/rampart`; remote model loading is
disabled. After the first local load, the browser cache is reused.

WASM is selected by default because it is the most compatible. Choose WebGPU in
a browser that supports it if you want to compare performance.

## Terminal

Pass a message directly:

```bash
npm run redact -- "My name is Alex Rivera and my SSN is 472-81-0094."
```

Or start the interactive prompt:

```bash
npm run redact
```

For a fast structured-PII-only check that skips the neural model:

```bash
npm run redact -- --heuristics-only "Email me at alex@example.com"
```

## Frozen v2 screenshot baseline

The immutable pre-retraining pipeline is packaged under
[`harness/plva-v2-baseline`](harness/plva-v2-baseline). It includes the old
visual detector, RapidOCR, Rampart, fusion and rendering code, model provenance,
and a standalone screenshot-to-redacted-PNG command that requires only Node and
a local Chrome/Chromium installation:

```bash
cd harness/plva-v2-baseline
node bin/plva-v2.mjs screenshot.png \
  --output screenshot.redacted.png \
  --report screenshot.redacted.json \
  --profile high-recall
```

This is an AGPL development baseline, not a release-eligible privacy boundary.

## Verify

```bash
npm test
npm run build
```

For the dev-only full-pipeline validation lab, open
`http://127.0.0.1:5173/evaluation.html`. It can run both sensitivity profiles on
the local synthetic smoke set and the preserved WebPII quick-100 diagnostic.
The evaluator stores boxes, labels, sources, timings, hashes, and integrity
results, never recognized OCR text.

Build the app plus the evaluation lab and its frozen diagnostic assets with:

```bash
npm run build:evaluation
```

Rampart is harm reduction rather than a perfect security boundary. Its current
model supports English, Spanish, French, German, Italian, Portuguese, and Dutch;
non-Latin scripts are outside the documented scope. Use synthetic test data in
this playground.

## Model engineering

The repository now includes the hybrid model track: pinned RapidOCR, Rampart,
and WebRedact contracts; safe asset fetching; OCR and visual reference harnesses;
screen-native and WebPII dataset preparation; optional semantic and detector
training; ONNX export and quantization; cross-runtime golden vectors; Modal L4
jobs; release validation; model cards; and license controls.

Start with [training/README.md](training/README.md). The current bootstrap and
known blockers are recorded in
[ADR 0001](docs/decisions/0001-model-and-runtime-contracts.md). No custom model
is marked release eligible yet.

```bash
npm run models:fetch
npm run models:verify
npm run models:inspect
```
