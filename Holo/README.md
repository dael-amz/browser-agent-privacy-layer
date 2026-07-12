# PLVA proxy

Fail-closed privacy proxy workbench for the HoloDesktop computer-use agent.

The project is being built one verified phase at a time from
[`../BLUEPRINT.md`](../BLUEPRINT.md). Step 0 probes external contracts using only synthetic
data; no API keys, screenshots, transcripts, or vault contents belong in this repository.

For a human-readable directory tour, file responsibilities, and the current resume point, see
[`PROJECT_MAP.md`](PROJECT_MAP.md).

## Accelerated redaction setup

The default redaction engine is an adaptive worker that runs visual and OCR detection concurrently
and uses WebGPU for the visual detector when available. Build its generated local assets after
placing the frozen detector at `plva-v2-baseline/`:

```bash
cd redactor-worker
npm install
PLVA_BASELINE_ROOT=../plva-v2-baseline npm run build
cd ..
```

`PLVA_REDACT=1 ./run_step1.sh` enables redaction; `PLVA_REDACT=0` (the default) explicitly disables
it. Adaptive mode starts the models on the first CUA screenshot, reuses them across the active
burst, and releases the roughly 1.6 GiB worker after 60 idle seconds. Set
`PLVA_REDACT_LIFECYCLE=eager` for frequent CUA calls, `cold` for minimum idle footprint, or adjust
`PLVA_REDACT_IDLE_SECONDS`. Use `PLVA_REDACT_BACKEND=wasm` when WebGPU is unavailable;
`PLVA_REDACT_ENGINE=baseline` retains the slow one-process-per-frame comparison path. Redacted
frames remain available in the memory-only viewer at `http://127.0.0.1:18081/viewer`.
