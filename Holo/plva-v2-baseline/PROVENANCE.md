# Provenance

Snapshot date: 2026-07-11 (America/Los_Angeles)

Origin workspace: `/Users/pyro/Documents/plvq` at the point immediately before
the replacement detector training run. Absolute origin paths are recorded here
for maintainers only; the runtime JSON report never emits them.

## Executable closure

`baseline-app-dist/` is a byte-for-byte copy of the 15-file production build
present in the origin workspace. The copied tree is about 83.75 MiB. Its visual
detector SHA-256 is:

`450ee07452d618eb3159d565f1787b17a8cabbfee3e72686e932460be576cc2e`

`dist/` is a dedicated harness entry built from the exact copied runtime
modules. It adds only screenshot input/output transport, sanitized reporting,
offline enforcement, and output-integrity checking. It does not alter detector,
OCR, Rampart, policy, fusion, threshold, or renderer logic.

## Frozen model origins

- Visual: `plva-visual-agpl-test-v2`, 20 epochs, Ultralytics/YOLO export,
  AGPL-3.0-only source checkpoint, trained-unpublished, release_eligible=false.
- OCR detector/recognizer: RapidOCR PP-OCRv4 mobile ONNX assets, Apache-2.0.
- Semantic classifier: nationaldesignstudio/rampart revision
  `b1993e4e68b082835b80ffc65acc03325ea2e501`, CC-BY-4.0 model assets.
- Browser inference: ONNX Runtime Web 1.27.0 and Transformers.js 3.7.5.

See `snapshot.json` for exact bytes and hashes, the detector model card and
manifest under `runtime/training/artifacts/`, and `LICENSES.md` for the fuller
license inventory.

## Diagnostic status

The v2 full-browser quick-100 WebPII diagnostic measured 63.56% fused recall at
98% truth-area coverage, 28.70% compatible precision, and 6 misses among 7
secret boxes. That sample is not a release holdout. The artifact must remain
clearly labeled as a development baseline.

The bundled `fixtures/ats-smoke.png` is one synthetic ATS test fixture with
SHA-256 `6013193effdafeddc606ff7a275d5575dcfe48adeca60fbff52c5035417d8690`.
No other ATS holdout image is included.
