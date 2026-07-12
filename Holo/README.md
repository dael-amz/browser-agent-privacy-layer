# PLVA proxy

Fail-closed privacy proxy workbench for the HoloDesktop computer-use agent.

The project is being built one verified phase at a time from
[`../BLUEPRINT.md`](../BLUEPRINT.md). Step 0 probes external contracts using only synthetic
data; no API keys, screenshots, transcripts, or vault contents belong in this repository.

For a human-readable directory tour, file responsibilities, and the current resume point, see
[`PROJECT_MAP.md`](PROJECT_MAP.md).

## Consumer demo

Launch the fast-tracked Steps 1–6 app with:

```bash
./run_demo.sh
```

It opens `http://127.0.0.1:18080` with a task box, PLVA master toggle, per-class security editor,
redacted model view, memory-only Holo reasoning/action trace, vault, OCR findings, stream-guard
counters, a model-call History tab, and an Advanced Lab tab containing the testing switches.
History shows the redacted requests that crossed the privacy boundary and the model's raw replies;
the Agent trace shows runtime reasoning, actions, and errors. Trace, vault, and OCR values are
blurred until explicitly revealed. The original task is excluded from trace retention; all viewer
state is memory-only. The app binds only to loopback and sends `Cache-Control: no-store`; it uses no
browser storage. Spoken prompts, voice-read, and the mediator UI remain extension points for later
steps.

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

Live runs use the `balanced` detector profile by default so uncertain OCR and low-confidence
visual detections do not obscure ordinary controls. Set `PLVA_REDACT_PROFILE=high-recall` when
testing maximum recall on synthetic screens.

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

## Audited CUA run

Use audit mode when you want to prompt the real desktop agent and inspect every redacted image
included in its model requests:

```bash
PLVA_PROVIDER=hcompany \
PLVA_REDACT=1 \
PLVA_REDACT_ENGINE=vision \
PLVA_REDACT_LIFECYCLE=eager \
PLVA_AUDIT=1 \
./run_step1.sh "Open Terminal, run: echo plva-audit-ok, then report the output"
```

Write prompts as a concrete visible task with a success condition. Good prompts name the app,
the exact action, and what evidence should be reported at the end. Avoid putting passwords,
API keys, personal data, or other secrets in the prompt itself; the prompt is model input.

Open `http://127.0.0.1:18081/viewer` during or after the task. It is a memory-only ledger of the
redacted pixels sent in each upstream request, including frame number, SHA-256 prefix, mask count,
redaction latency, delivery state, and upstream HTTP status. Select any buffered frame to inspect
it. `http://127.0.0.1:18081/viewer/frames` exposes the same value-free metadata as JSON. Audit mode
keeps the viewer alive after Holo exits; press Ctrl-C in the launch terminal to discard the buffer.
Increase or reduce its in-memory ring with `PLVA_AUDIT_CAPACITY` (default `32`).

The launch terminal is the privacy-safe event log: it reports task status, loopback-only runtime
connections, request/response byte counts, HTTP statuses, and timings without bodies or pixels.
Holo's frame-bearing run directory is created under `/tmp` and shredded before audit mode waits.
Do not use `/viewer/findings` for a no-PII audit: that separate local diagnostic intentionally
contains the latest cleartext OCR/vault candidates and is never loaded by the safe viewer.

With the Vision engine, `run_step1.sh` enables the Step 5 privacy core by default. The same OCR
finding supplies the recognized value and bounding box to the in-memory vault and to the visible
`«CLASS_N_nonce»` chip. Executed action fields resolve locally; reasoning remains placeholdered;
outbound history is scrubbed first by exact vault match and then by the warm Core ML Rampart
backstop. Each model request also receives the placeholder scheme plus an exact token/class
manifest beside the current frame. Old injected manifests are removed, while the new manifest
separately lists value-free tokens still active for the private session so multi-step form flows
can reuse an issued token after it leaves the screen. Forged or non-session tokens fail closed. A
per-class policy additionally teaches Holo which tokens are usable, approval-gated, or blocked.
The matching native Holo skill is kept at `holo-skills/plva-placeholders/SKILL.md` and refreshed
under `~/.holo/skills/` for each privacy-enabled run. Set
`PLVA_PRIVACY=0` only for comparison testing.

Edit `config/privacy-policy.json` to change defaults, set `PLVA_POLICY_FILE` to select another
file, or pass an in-memory override through `PLVA_POLICY_JSON`. The GUI updates the policy for its
next run without writing it to disk. `blocked` findings receive no token and are never vaulted.
`approval` tokens require a short-lived exact local capability bound to token, tool, argument path,
optional target, TTL, and use count. The Vault view can grant one `write_desktop:content` use;
`GET/POST/DELETE /viewer/approvals` provides the loopback API. The later mediator can automate
policy decisions, but approval-gated local use no longer depends on it.

For synthetic diagnosis, individual Step 5/5a features can be disabled without changing the
secure all-on default:

| Environment variable | Feature | Default with privacy on |
|---|---|---|
| `PLVA_PRIVACY_CHIPS` | Vault current-frame OCR findings and paint placeholder chips | `1` |
| `PLVA_PRIVACY_HISTORY_SCRUB` | Vault + Rampart outbound-history scrub | `1` |
| `PLVA_PRIVACY_SCHEME` | Static placeholder system instruction | `1` |
| `PLVA_PRIVACY_DUPLICATE_WARNING` | Static duplicate-token warning | `1` |
| `PLVA_PRIVACY_MANIFEST` | Current-frame token/class observation | `1` |
| `PLVA_PRIVACY_RESOLUTION` | Resolve tokens in executed response fields | `1` |
| `PLVA_PRIVACY_POLICY_TEACHING` | Explain active per-class security levels to Holo | `1` |
| `PLVA_PRIVACY_SKILL` | Installed native Holo placeholder skill | follows `PLVA_PRIVACY` |

`PLVA_PRIVACY=0` disables the vault/chip wrapper and, unless explicitly overridden, the native
skill. Disabling history scrub or resolution is unsafe with real data; use these switches only on
synthetic screens. A manifest requires current-frame chips, so `PLVA_PRIVACY_CHIPS=0` also makes
the manifest absent even when its own switch remains enabled.

Watch exactly what the model receives at `http://127.0.0.1:18081/viewer`. The buffered memory-only
sent-frame history and delivery metadata are available there. The latest memory-only OCR
candidates are at `/viewer/findings`, the local vault is at `/viewer/vault`, and privacy-safe
history-filter counters are at `/viewer/filter`. The findings and vault endpoints contain
sensitive cleartext and are never persisted or logged. `PLVA_VISION_MODE=cascade` is the default:
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
burst, and releases the roughly 1.6 GiB worker after 60 idle seconds. The hot frame cache retains
32 redacted frames, and the privacy wrapper caches final painted frames so growing histories do
not thrash the detector or repeatedly repaint identical screenshots. History classification is
cached per unchanged safe text instead of per whole growing message tuple. Set
`PLVA_REDACT_LIFECYCLE=eager` for frequent CUA calls, `cold` for minimum idle footprint, or adjust
`PLVA_REDACT_IDLE_SECONDS`. Use `PLVA_REDACT_BACKEND=wasm` when WebGPU is unavailable;
`PLVA_REDACT_ENGINE=baseline` retains the slow one-process-per-frame comparison path. Redacted
frames remain available in the memory-only viewer at `http://127.0.0.1:18081/viewer`.

Longer tasks can raise the closed runtime's default budget with `PLVA_MAX_STEPS` and
`PLVA_MAX_TIME_S` (defaults: `20` and `300`). Both must be positive integers. A larger budget gives
the CUA room to recover, but exit code `0` still reflects runtime termination rather than an
independent UI-postcondition proof.

## CUA privacy benchmark and completion gate

Run the deterministic, provider-free PLVA-off/on comparison before every CUA change:

```bash
$HOME/.local/bin/uv run plva-benchmark
```

The benchmark drives a synthetic account form through the real request redaction, placeholder
manifest, response resolution, and local action-execution seams. It reports task completion,
private-frame/text exposures, action execution, placeholder resolutions, and interception latency
as JSON. The two-step fixture removes the private source value before the destination field is
shown, so PLVA-on must reuse the prior active-session token. Completion comes from assertions over
the final local form state; simulated runtime
success is recorded separately and cannot make the benchmark pass.

The live acceptance mode serves the same kind of form only on loopback, asks the real CUA to fill
and submit it, and considers the task complete only when the local server receives the exact
expected submission. It can call a paid provider, so both flags are mandatory:

```bash
$HOME/.local/bin/uv run plva-benchmark \
  --live --allow-provider-spend --mode on --runner ./run_step1.sh \
  --output verification/cua-live-plva-on.json
```

Use `--mode both` only when an explicit PLVA-off exposure comparison is intended. Live results use
`-1` for metrics that the closed runtime does not expose and report total wall time separately;
they never infer action success or frame
privacy from process exit status. The fixture contains synthetic data only, prompts never contain
the fixture value, and the loopback server does not log request bodies.

## Holo tool-channel probe

Recheck the Step 6.5 model-to-local-tool contract with synthetic data only:

```bash
$HOME/.local/bin/uv run plva-tool-probe \
  --provider hcompany \
  --output /tmp/plva-tool-probe.json
```

The probe captures one authentic Holo request on loopback, replaces every user/history message
and desktop image before provider egress, and tests native tools, structured actions, and a strict
free-text marker. It records only schema metadata and boolean outcomes. The current Holo3 contract
round-trips the free-text marker; native tools are absent and the attempted custom structured
action is not parseable. See `verification/step-6.5-tool-channel.md` for the matrix and fallback.
