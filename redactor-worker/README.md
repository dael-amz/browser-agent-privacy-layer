# PLVA accelerated redactor worker

A persistent loopback redaction worker. It keeps the visual detector and OCR models warm
across an agent's screenshot burst instead of paying model startup on every frame, and
runs them on WebGPU when available, with a WASM fallback.

The worker is supervised by `bin/redactor-worker.mjs`, which serves the built pipeline on
host loopback behind a per-session token. The pipeline itself lives in `src/worker.js`:
concurrent visual and OCR detection, region fusion, and opaque PNG mask rendering.

## Build

Requires Node 20+ and the frozen detector baseline extracted at `../plva-v2-baseline`:

```bash
npm install
PLVA_BASELINE_ROOT=../plva-v2-baseline \
PLVA_VISUAL_MODEL=../plvas-v3/harness/plva-v2-baseline/runtime/training/artifacts/plva-visual-agpl-test-v2/visual/detector.onnx \
npm run build
```

The proxy starts and stops the worker automatically; see the lifecycle switches
(`PLVA_REDACT_LIFECYCLE`, `PLVA_REDACT_BACKEND`, `PLVA_REDACT_IDLE_SECONDS`) in
[docs/usage.md](../docs/usage.md).
