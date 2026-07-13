# PLVA v3 provenance

Snapshot date: 2026-07-12 (America/Los_Angeles)

The visual artifact comes from Modal run `plva-visual-agpl-ats-v3`, attempt
`safe-head-v10`. Training was manually stopped after 35 complete, durably
committed epochs. The privacy high-water selector retained zero-based epoch 0;
all later completed epochs were rejected rather than promoted.

## Selected visual artifacts

- PyTorch safety-best checkpoint SHA-256:
  `5fcd871b76a6fe456d004ef465ad5cf97616b9f475080711ba0d059a5807a9ad`
- Deterministic FP32 ONNX SHA-256:
  `2ed3f25c9bee375dc1683cf0ffa2044374b6bc35dd89a465a7dce6451ce8b928`
- Runtime threshold artifact SHA-256:
  `c5c9892e0b87d5356645ac70090e22c21f95efd41d2391ad854c45ec87ecea60`
- Threshold profile SHA-256:
  `919da6455fc93ffb8ef929eeb5ded012b102c97a083b421cd0b5226b8be8b90e`
- Single-frame visual runtime-policy SHA-256:
  `c92fe90f228d7b47e7202fd09efbf77b75e8db5bc0837a6fc145a9774d0c102a`
- Source license: `AGPL-3.0-only`

The ONNX export used Ultralytics 8.4.92, Torch 2.13.0, one deterministic CPU
thread, opset 17, no simplifier, normalized metadata, and deterministic protobuf
serialization. ONNX Runtime 1.27.0 golden vectors pass both Node CPU and browser
WASM execution.

The browser harness keeps the selected single-frame preprocessing contract.
An adaptive visual-tiling experiment was rejected after it increased false masks
without recovering the audited tall-page truths. Constrained OCR context rules
cover all five truths in the bundled tall cart regression; this is a regression
result, not a broad release-quality claim.

## Other models

- RapidOCR PP-OCRv4 mobile detector/English recognizer: Apache-2.0.
- `nationaldesignstudio/rampart` revision
  `b1993e4e68b082835b80ffc65acc03325ea2e501`: CC-BY-4.0.
- Browser execution: ONNX Runtime Web 1.27.0 and Transformers.js 3.7.5.

Exact artifact metadata and selection evidence live under
`runtime/training/artifacts/plva-visual-agpl-ats-v3/visual-safe-head-v10/`.
