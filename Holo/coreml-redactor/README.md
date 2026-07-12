# PLVA Core ML / Neural Engine probe

This is a separate, non-production backend. It does not modify or replace the stable
`redactor-worker/` path. It now includes a local-only live viewer so ANE preprocessing, inference,
detection decoding, and burned visual masks can be inspected end to end.

It derives a fixed `1×3×640×640` visual ONNX model, then asks ONNX Runtime's Core ML provider for
`CPUAndNeuralEngine` compute with GPU excluded. The static `NeuralNetwork` format measured roughly
8.5–9.4 ms warm visual inference on the development M4, versus 68–148 ms with ONNX CPU. Core ML
accepted 312 of 318 graph nodes; unsupported nodes may fall back to CPU.

Run from this directory:

```bash
$HOME/.local/bin/uv run plva-ane-probe --baseline ../plva-v2-baseline
```

Run the live screen viewer:

```bash
$HOME/.local/bin/uv run plva-ane-live --baseline ../plva-v2-baseline
open http://127.0.0.1:18083/
```

macOS may require Screen Recording permission. For a privacy-safe static smoke test:

```bash
$HOME/.local/bin/uv run plva-ane-live \
  --baseline ../plva-v2-baseline \
  --fixture ../plva-v2-baseline/fixtures/ats-smoke.png
```

The generated fixed model and compiled cache stay under `.cache/` and contain no frames. The probe
uses only the synthetic baseline fixture and emits timing/numerical metadata.

This is intentionally not wired into outbound redaction yet. The viewer is **visual-detector only**;
it does not run RapidOCR or Rampart and must not be used to send frames upstream. Core ML output is
numerically different from the CPU/WebGPU output, and the current fixture has no visual detections,
so positive-detection geometry parity must be tested before this backend can fail closed in
production. The
`VisualANESession.infer()` tensor boundary is also the intended seam for a later parallel native
worker.
