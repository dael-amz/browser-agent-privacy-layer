# ADR 0001: Model and runtime contracts

**Status:** Accepted with recorded release blockers  
**Date:** 2026-07-11

## Decision

PLVA v0.1 uses independent visual and OCR discovery, then applies local semantic
classification and deterministic rules.

1. RapidOCR PP-OCRv4 detector and English recognizer provide the pinned OCR
   reference.
2. Rampart provides the bootstrap semantic model and its published runtime owns
   normalization, deterministic recognizers, span repair, and long-line merge
   behavior.
3. WebRedact nano 640 provides a coarse OpenVINO proposal reference in supported
   local environments.
4. A custom tagger or fine-grained detector is a replacement only. It is never a
   prerequisite for the bootstrap demo and cannot ship until it beats the
   frozen bootstrap without a secret-class regression.

The complete machine-readable contract is `models.lock.json` schema version 2.

## Pinned bootstrap

| Component | Revision | Runtime artifact | Verified signature |
|---|---|---|---|
| RapidOCR | source `7fe716f8e38bb9a43f2680159f38deb14d8b1930`, release 3.9.1 | detector and recognizer ONNX plus 95-line dictionary | detector opset 12, recognizer opset 14 with `[N,T,97]` probabilities |
| Rampart | `b1993e4e68b082835b80ffc65acc03325ea2e501` | Q4 ONNX and tokenizer bundle | three dynamic int64 inputs, `[N,T,35]` logits, opset 18 with `MatMulNBits` |
| WebRedact | `de27fa3ae82bfab34fa28281fc9dc9786fd01600` | nano 640 OpenVINO XML/BIN | FP32 `[1,3,640,640]` to `[1,6,8400]`, classes `text` and `image` |
| ONNX Runtime JS | 1.27.0 | Node and Web packages | exact versions and npm integrity values pinned |

Every default asset has an exact byte count and SHA-256. Fetching uses a bounded
temporary file, validates HTTPS redirect hosts, verifies bytes and hash, then
renames atomically. Inspection checks graph signatures before a model is
accepted.

## WebRedact conversion decision

The published WebRedact model release contains OpenVINO IR only. It has no ONNX
or source checkpoint, and OpenVINO's supported conversion flow imports source
models into IR rather than reconstructing a source-equivalent ONNX graph from
IR. The model repository also has no license file.

Therefore:

- the pinned OpenVINO model remains a local reference and possible Node-side
  proposal implementation;
- full browser hybrid mode is blocked until upstream supplies a licensed ONNX
  or source checkpoint, or the PLVA replacement detector passes parity and
  safety gates;
- PLVA must not fabricate a conversion, redistribute the public XML/BIN, or
  claim browser hybrid support;
- OCR plus Rampart remains the explicit degraded browser path.

## Semantic replacement gate

The replacement base candidate is Google BERT L4-H256. Training requires a
frozen Rampart-backed screen/OCR baseline. A contract-only Rampart smoke does not
satisfy this gate.

Release also requires:

- no secret-class recall regression;
- all clean, corrupted, screen, region, and real-OCR-output gates passing;
- INT8 model plus tokenizer/config at most 20 MB;
- PyTorch and ONNX Python parity;
- exact golden-vector predictions under Node, browser WASM, and browser WebGPU;
- an explicit compatible artifact license.

Development overrides are recorded as non-release and publishing rejects them.

## Visual replacement gate

WebPII train and test remain separate. Checkpoint selection uses a separate,
template-disjoint PLVA synthetic validation split. GUIGuard stays
evaluation-first. The selection tuple is minimum secret-class recall, then mean
non-secret recall, then mean precision.

AGPL development is authorized using official YOLO11n from Ultralytics assets
v8.4.0, pinned at 5,613,764 bytes and SHA-256
`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
Modal fetches this asset through the lock instead of using mutable Ultralytics
auto-download. Any closed-source or commercial release still requires an
Ultralytics Enterprise license. WebRedact remains the fallback reference until
the replacement improves fine-grained proposal precision without reducing
secret recall and passes ONNX cross-runtime, size, latency, and license review.

## Verified implementation evidence

Local contract verification on 2026-07-11 established:

- 11 default files, 40,021,722 total bytes, all hashes and tensor signatures
  valid;
- 27 OCR regions across three deterministic fake UI fixtures;
- six Rampart model-only golden vectors with original-offset preservation;
- WebRedact OpenVINO inference on the same fixtures, with threshold explicitly
  marked smoke-only;
- a compact development tagger export of 4,632,327 runtime bytes;
- PyTorch to FP32 ONNX agreement of 1.0 on the development smoke;
- Node and WASM golden-vector parity for that development artifact;
- WebGPU parity, full baseline quality, replacement quality, and release
  licensing remain unproven.

The authorized Ultralytics path was also exercised on Modal with a two-epoch
development smoke. The pinned YOLO11n checkpoint produced a 10,428,657-byte
opset-17 detector at SHA-256
`09a54eae1ba2c7a2585aff3aa03000f268da5fa9197b83b27c594e244ce112cc`.
Two clean exports were byte-identical after deterministic single-thread fusion
and metadata normalization. Python ONNX Runtime and Node matched the detector
goldens exactly; browser WASM passed 104 samples within the declared tolerance.
This proves the data/train/export/runtime path, not detector quality. WebGPU,
full training, published-test evaluation, and the commercial license gate are
still open.

WebPII mapping version 2 was fail-closed on a 600-row train/test audit: zero
unmapped visible source labels and 155 duplicate annotations merged. The full
published dataset must pass the same fail-closed audit before a full run.

These smoke measurements validate code paths and contracts. They are not release
quality metrics.

## Consequences

The bootstrap can progress without waiting for new model training. Model
replacement work remains reproducible and Modal-ready, but intentionally cannot
consume GPU spend or publish an artifact until its prerequisites are supplied.
Runtime teams receive exact assets, adapter contracts, fixtures, and explicit
degraded-mode boundaries instead of mutable model assumptions.
