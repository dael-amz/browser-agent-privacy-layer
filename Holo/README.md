# PLVA proxy

Fail-closed privacy proxy workbench for the HoloDesktop computer-use agent.

The project is being built one verified phase at a time from
[`../BLUEPRINT.md`](../BLUEPRINT.md). Step 0 probes external contracts using only synthetic
data; no API keys, screenshots, transcripts, or vault contents belong in this repository.

For a human-readable directory tour, file responsibilities, and the current resume point, see
[`PROJECT_MAP.md`](PROJECT_MAP.md).

## Accelerated redaction setup

The lowest-latency macOS path uses native Apple Vision OCR, the Core ML visual detector, and Core
ML Rampart. Install its isolated environment once:

```bash
cd coreml-redactor
$HOME/.local/bin/uv sync --group dev
cd ..
```

Run the complete CUA through that pipeline:

```bash
PLVA_REDACT=1 \
PLVA_REDACT_ENGINE=vision \
PLVA_REDACT_LIFECYCLE=eager \
./run_step1.sh "your task"
```

When `plvas-v3/` is present, this path uses its bundled visual detector while retaining the
existing accelerated OCR and Rampart assets. Override it with
`PLVA_VISUAL_MODEL=/path/to/detector.onnx`; the content-addressed Core ML cache recompiles only
when the selected weights change.

Choose the inference provider independently of the redaction engine. Overshoot remains the default.
To use H Company's managed Holo API, place `HAI_API_KEY=<key>` in `.env`, then run:

```bash
PLVA_PROVIDER=hcompany \
PLVA_REDACT=1 \
PLVA_REDACT_ENGINE=vision \
./run_step1.sh "your task"
```

The H Company preset uses `https://api.hcompany.ai/v1` and `holo3-1-35b-a3b`. Override the model
with `PLVA_MODEL` or either preset's URL with `PLVA_UPSTREAM` when testing another compatible
deployment.

With the Vision engine, `run_step1.sh` enables the Step 5 privacy core by default. The same OCR
finding supplies the recognized value and bounding box to the in-memory vault and to the visible
`«CLASS_N_nonce»` chip. Executed action fields resolve locally; reasoning remains placeholdered;
outbound history is scrubbed first by exact vault match and then by the warm Core ML Rampart
backstop. Each model request also receives the placeholder scheme plus an exact token/class
manifest beside the current frame; stale manifests are removed. The matching native Holo skill is
kept at `holo-skills/plva-placeholders/SKILL.md` and installed under `~/.holo/skills/`. Set
`PLVA_PRIVACY=0` only for comparison testing.

Watch exactly what the model receives at `http://127.0.0.1:18081/viewer`. The latest memory-only
OCR/vault candidates are at `http://127.0.0.1:18081/viewer/findings`; that endpoint contains
sensitive cleartext and is never persisted or logged. `PLVA_VISION_MODE=cascade` is the default:
it runs fast OCR over the frame and accurate OCR only over sensitive or uncertain regions.

The default redaction engine is an adaptive worker that runs visual and OCR detection concurrently
and uses WebGPU for the visual detector when available. Build its generated local assets after
placing the frozen detector at `plva-v2-baseline/`:

```bash
cd redactor-worker
npm install
PLVA_BASELINE_ROOT=../plva-v2-baseline \
PLVA_VISUAL_MODEL=../plvas-v3/harness/plva-v2-baseline/runtime/training/artifacts/plva-visual-agpl-test-v2/visual/detector.onnx \
npm run build
cd ..
```

`PLVA_REDACT=1 ./run_step1.sh` enables redaction; `PLVA_REDACT=0` (the default) explicitly disables
it. Adaptive mode starts the models on the first CUA screenshot, reuses them across the active
burst, and releases the roughly 1.6 GiB worker after 60 idle seconds. Set
`PLVA_REDACT_LIFECYCLE=eager` for frequent CUA calls, `cold` for minimum idle footprint, or adjust
`PLVA_REDACT_IDLE_SECONDS`. Use `PLVA_REDACT_BACKEND=wasm` when WebGPU is unavailable;
`PLVA_REDACT_ENGINE=baseline` retains the slow one-process-per-frame comparison path. Redacted
frames remain available in the memory-only viewer at `http://127.0.0.1:18081/viewer`.
