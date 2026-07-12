# PLVA model engineering

This directory implements the hybrid plan's model track. The first working
system uses pinned OCR, Rampart semantics, deterministic rules, and WebRedact
visual proposals. A PLVA-specific tagger and fine-grained detector are optional
replacement tracks. Neither replacement may silently become a release model.

## Current state

| Track | State | Release consequence |
|---|---|---|
| RapidOCR detector, recognizer, and dictionary | Contract verified | Ready for runtime integration and parity testing |
| Rampart quantized tagger and tokenizer | Contract verified | Bootstrap semantic model; full pipeline baseline is not frozen yet |
| WebRedact nano 640 OpenVINO IR | Reference inference verified | Node proposal reference only; redistribution awaits upstream license clarification |
| WebRedact browser ONNX | Blocked | Public release has no source checkpoint or supported IR-to-ONNX path |
| PLVA compact tagger replacement | Development pipeline verified | Training and release require a frozen Rampart baseline |
| Fine-grained visual replacement | Pinned YOLO11n AGPL development base plus data, training, evaluation, and Modal pipeline | Closed-source release requires an Ultralytics commercial license and measured replacement gates |

No custom model in this repository is currently release eligible. The tooling
records development artifacts, but publication fails closed.

## Pinned asset bootstrap

The lock contains immutable revisions, byte counts, SHA-256 values, licenses,
redirect policy, tensor names, dtypes, shapes, and opsets.

```bash
node scripts/fetch-models.mjs
node scripts/verify-models.mjs
node scripts/inspect-models.mjs --output training/tmp/model-inspection.json
```

The default 11 assets total 40,021,722 bytes. Generated model files live under
`models/` and are ignored by git.

## Local reference environment

Use Python 3.12. The core training, OCR reference, and visual reference packages
are pinned separately so runtime consumers do not inherit training dependencies.

```bash
python3.12 -m venv training/.venv
training/.venv/bin/pip install -r training/requirements.lock.txt
training/.venv/bin/pip install -r training/ocr-requirements.lock.txt
training/.venv/bin/pip install -r training/visual-requirements.lock.txt
```

Generate fake UI fixtures and run all three bootstrap references:

```bash
training/.venv/bin/python -m training.ocr.generate_fixtures \
  --output-dir training/tmp/hybrid-fixtures

training/.venv/bin/python -m training.ocr.reference \
  --fixtures training/tmp/hybrid-fixtures/manifest.json \
  --output training/tmp/hybrid-fixtures/ocr_goldens.json

training/.venv/bin/python -m training.semantic.rampart_reference \
  --output training/tmp/hybrid-fixtures/rampart_goldens.json

training/.venv/bin/python -m training.visual.reference_webredact \
  --fixtures training/tmp/hybrid-fixtures/manifest.json \
  --output training/tmp/hybrid-fixtures/webredact_goldens.json
```

The OCR decoder directly consumes the recognizer's `[N,T,97]` probabilities.
The WebRedact threshold in this smoke is not a frozen release threshold.

## Never-trained OCR-output holdout

The replacement tagger is evaluated on text produced by the exact OCR stack,
not only on simulated character corruption.

```bash
training/.venv/bin/python -m training.ocr.build_semantic_holdout \
  --fixtures training/tmp/hybrid-fixtures/manifest.json \
  --ocr-goldens training/tmp/hybrid-fixtures/ocr_goldens.json \
  --output-dir training/tmp/ocr-holdout

training/.venv/bin/python -m training.prepare_data \
  --output-dir training/data/plva-v1 \
  --ocr-holdout training/tmp/ocr-holdout/ocr_holdout.jsonl \
  --ocr-holdout-manifest training/tmp/ocr-holdout/ocr_holdout_manifest.json
```

Preparation rejects OCR records not marked `never_train` and rejects canonical
PII values that overlap training data. Evaluation reports clean, corrupted,
region, frozen synthetic holdout, and OCR-stack holdout slices separately.

## Rampart baseline gate

Model-only goldens verify the pinned ONNX contract. They do not freeze the
baseline because the released runtime also owns normalization, deterministic
recognizers, span repair, and long-line merging.

```bash
training/.venv/bin/python -m training.freeze_rampart_baseline \
  --output-dir training/tmp/rampart-contract
```

To permit replacement training, supply a full screen/OCR evaluation containing
all required gates and explicitly freeze it:

```bash
training/.venv/bin/python -m training.freeze_rampart_baseline \
  --evaluation /path/to/rampart-baseline-evaluation.json \
  --output-dir training/tmp/rampart-frozen \
  --freeze
```

`training.train` refuses to run without the frozen manifest. The local
`--development-allow-unfrozen-baseline` flag exists only for code-path smoke
tests and marks the result as non-release.

## Compact tagger replacement

The optional tagger pipeline provides leakage-safe data preparation, tokenizer
construction, overlap-window training, per-class calibration, safety-weighted
checkpoint selection, confusion matrices, ONNX export, dynamic INT8
quantization, and artifact provenance. Use each command's `--help` for sizing
options.

```bash
training/.venv/bin/python -m training.build_tokenizer --help
training/.venv/bin/python -m training.train --help
training/.venv/bin/python -m training.calibrate --help
training/.venv/bin/python -m training.evaluate --help
training/.venv/bin/python -m training.export_onnx --help
training/.venv/bin/python -m training.validate_artifacts --help
```

Release gates require all of the following:

- frozen Rampart full-pipeline baseline;
- all semantic and OCR-stack evaluation gates passing;
- at most 20 MB for INT8 model plus runtime tokenizer/config files;
- PyTorch, ONNX Python, Node, WASM, and WebGPU parity;
- quantization recall within the fixed policy on both holdouts;
- explicit maintainer-approved artifact license.

Development export requires `--allow-development-artifact`. Publication always
runs release validation and cannot use that override.

## Cross-runtime tagger parity

The golden-vector harness uses exact model inputs, so it does not hide tokenizer
differences inside a backend comparison.

```bash
npm ci --prefix training/runtime-parity

node training/runtime-parity/verify.mjs \
  --backend node \
  --artifact-dir training/artifacts/plva-tagger-v1 \
  --output training/tmp/cross-runtime-report.json

node training/runtime-parity/verify.mjs \
  --backend wasm \
  --artifact-dir training/artifacts/plva-tagger-v1 \
  --output training/tmp/cross-runtime-report.json
```

Run the same command with `--backend webgpu` in a WebGPU-enabled host. Export
accepts the report only when its model SHA-256 matches and all three backends
pass. WebGPU evidence must also include a verified provider trace and measured
fallback-node count. Node and web packages are pinned at 1.27.0.

## Fine-grained visual replacement

WebPII's published test split is never used for checkpoint selection. The
separate PLVA synthetic validation split is template-family and seed disjoint.
GUIGuard remains evaluation-first and its raw screenshots/text are not persisted
unless explicitly requested.

The authorized development path pins official YOLO11n from Ultralytics assets
release v8.4.0 at 5,613,764 bytes and SHA-256
`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
It is AGPL-3.0-only. Development and evaluation may proceed under AGPL, but a
closed-source or commercial PLVA release remains blocked until an Ultralytics
Enterprise license is recorded.

The checked pipeline has completed a two-epoch Modal smoke only. Its generated
ONNX/model card/manifest are deliberately marked `development-smoke`; do not
publish those metrics as detector quality or use the artifact in release mode.

```bash
node scripts/fetch-models.mjs --asset visual.ultralytics.yolo11n

training/.venv/bin/python -m training.visual.prepare_webpii \
  --output-dir training/data/webpii \
  --fail-on-unmapped

training/.venv/bin/python -m training.visual.prepare_synthetic \
  --output-dir training/data/visual-synthetic

# User-provided WebPII-format ATS data is validated and converted into an
# immutable, value-free supplemental dataset. Its test identities are retained
# for diagnostics but never enter training or checkpoint selection.
training/.venv/bin/python -m training.visual.prepare_supplemental \
  --source-root /absolute/path/to/webpii-synth/out-v2 \
  --output-dir training/data/visual-supplemental-ats

training/.venv/bin/python -m training.visual.compose_dataset \
  --webpii-root training/data/webpii \
  --synthetic-root training/data/visual-synthetic \
  --supplemental-root training/data/visual-supplemental-ats \
  --output-dir training/data/visual-composed

training/.venv/bin/python -m training.visual.train_detector \
  --dataset-yaml training/data/visual-composed/dataset.yaml \
  --source-manifest training/data/visual-composed/manifest.json \
  --base-checkpoint models/visual-training/yolo11n-v8.4.0.pt \
  --base-license AGPL-3.0-only \
  --base-source ultralytics-assets-v8.4.0 \
  --output-dir training/runs/visual-v1
```

The supplemental adapter requires the generator's schema-v2 `manifest.json`
and independently verifies its zero-failure audit, tile-capture policy,
identity-disjoint split, fail-closed label policy, metadata/image hashes, and
`MISC_COMPANY` hard-negative disposition. Legacy full-page output is rejected.
Prepared image and label aggregates use
`sha256-sorted-path-nul-file-sha256-lf-v1`: sort relative POSIX paths by their
UTF-8 bytes, then hash each UTF-8 path, one NUL byte, its lowercase hexadecimal
file SHA-256, and one LF byte.

Before a full screenshot download, run the metadata-only source-label audit:

```bash
training/.venv/bin/python -m training.visual.audit_webpii_labels \
  --output training/tmp/webpii-label-audit.json \
  --fail-on-unmapped
```

Checkpoint selection orders minimum secret-class recall first, then non-secret
recall, then precision. A base checkpoint with missing license metadata is
rejected, and AGPL output is marked ineligible for closed-source release. The
public WebRedact IR conversion command emits a structured blocked report rather
than fabricating an ONNX file.

Detector training uses an explicit screen-native augmentation policy instead
of Ultralytics' natural-image defaults. Horizontal/vertical flips, rotation,
shear, perspective, RGB/BGR swapping, mosaic, MixUp, CutMix, copy-paste, and
multi-scale training are disabled. Only modest saturation/value jitter (0.1),
translation (0.02), and scale gain (0.05) remain; hue jitter is disabled. The
complete override dictionary is recorded in each `training_manifest.json`.

Create detector goldens and check the exported graph under Node and browser
WASM with the pinned ONNX Runtime packages:

```bash
training/.venv/bin/python -m training.visual.create_onnx_goldens \
  --model training/artifacts/<run-id>/visual/detector.onnx \
  --output training/artifacts/<run-id>/visual/detector_goldens.json

cd training/runtime-parity
node verify_visual.mjs --backend node \
  --artifact-dir ../artifacts/<run-id>/visual \
  --output ../artifacts/<run-id>/visual/visual_cross_runtime_report.json
node verify_visual.mjs --backend wasm \
  --artifact-dir ../artifacts/<run-id>/visual \
  --output ../artifacts/<run-id>/visual/visual_cross_runtime_report.json
```

### Detector INT8 policy

The FP32 detector is authoritative. Detector quantization is not the tagger's
dynamic-MatMul recipe: YOLO11n is Conv-heavy, and dynamic Conv quantization
introduces `ConvInteger`/`DynamicQuantizeLinear` graphs with unacceptable
numerical and browser-runtime risk. QOperator (`QLinearConv`) graphs are also
rejected. The only candidate generated by the checked path is static QDQ with
UINT8 activations, per-channel INT8 Conv weights, and MinMax calibration.

That candidate is never written to the runtime artifact name unless every gate
passes: at least 50% byte reduction; at least 256 calibration screenshots; an
all-class, content-disjoint evaluation set with at least 400 screenshots and 20
annotations per class; 99% class-argmax and FP32-proposal agreement; bounded
overall/per-class recall and precision changes; zero secret-class recall drop;
and successful Node plus browser-WASM parity under ONNX Runtime 1.27.0. The
published WebPII test split must not be used to choose a quantization candidate.

Install the pinned parity packages, then run the gate against a new run and a
frozen all-class acceptance split. Do not point this command at the v2 bundle:

```bash
npm ci --prefix training/runtime-parity

python -m training.visual.quantize_detector_onnx \
  --source training/artifacts/<run-id>/visual/detector.onnx \
  --calibration-images /path/to/training-only/images \
  --evaluation-images /path/to/frozen-quantization-acceptance/images \
  --evaluation-labels /path/to/frozen-quantization-acceptance/labels \
  --output-dir training/artifacts/<run-id>/visual/quantized \
  --require-output
```

The command always writes `detector_quantization_report.json`. A rejected run
leaves no candidate ONNX behind and preserves any previously approved output.
Only a passing run atomically creates `detector.int8.onnx`. Normal FP32 export
records quantization as `gated-not-produced`; it never silently substitutes a
quantized graph.

## Modal

The application defines CPU preparation/reference jobs, L4 semantic training,
H200 detector training with a 64-image batch, export jobs, durable run storage, and a separate model
cache. The local environment currently needs a Modal token before these jobs can
run.

```bash
uvx --from modal==1.5.2 modal token new
```

First verify or freeze the baseline:

```bash
uvx --from modal==1.5.2 modal run training/modal_app.py \
  --stage baseline \
  --run-id plva-tagger-v1

uvx --from modal==1.5.2 modal run training/modal_app.py \
  --stage baseline \
  --run-id plva-tagger-v1 \
  --baseline-evaluation /path/to/rampart-baseline-evaluation.json \
  --force
```

Upload the never-trained OCR holdout before semantic preparation:

```bash
uvx --from modal==1.5.2 modal volume put plva-model-engineering \
  training/tmp/ocr-holdout/ocr_holdout.jsonl \
  runs/plva-tagger-v1/inputs/ocr_holdout.jsonl

uvx --from modal==1.5.2 modal volume put plva-model-engineering \
  training/tmp/ocr-holdout/ocr_holdout_manifest.json \
  runs/plva-tagger-v1/inputs/ocr_holdout_manifest.json
```

Then run semantic stages explicitly:

```bash
uvx --from modal==1.5.2 modal run training/modal_app.py --stage prepare --run-id plva-tagger-v1
uvx --from modal==1.5.2 modal run training/modal_app.py --stage train --run-id plva-tagger-v1
uvx --from modal==1.5.2 modal run training/modal_app.py --stage export --quick --run-id plva-tagger-v1
uvx --from modal==1.5.2 modal run training/modal_app.py --stage download --run-id plva-tagger-v1
```

The quick export is a development artifact for cross-runtime testing. Rerun
export without `--quick`, with `--cross-runtime-report` and
`--artifact-license`, only after every gate passes.

For the optional detector, use stages `visual-prepare`, `visual-train`,
`visual-export`, and `visual-download`. The default `visual-train` stage securely
fetches the hash-pinned AGPL YOLO11n checkpoint, so no checkpoint upload is
needed. To use a purchased commercial checkpoint instead, upload
`detector-base.pt` and `detector-base-license.txt` under the run's `inputs/` and
pass `--visual-commercial-license-approved`.

For a bounded test candidate, combine `--quick` with an explicit epoch count,
for example `--visual-epochs 20`. The full profile defaults to 100 epochs.

For a run that must include a staged ATS supplement, fail closed instead of
silently falling back to WebPII plus built-in synthetic data:

```bash
uvx --from modal==1.5.2 modal run training/modal_app.py \
  --stage visual-prepare \
  --run-id <run-id> \
  --visual-require-supplemental

uvx --from modal==1.5.2 modal run --detach training/modal_app.py \
  --stage visual-train \
  --run-id <run-id>
```

The visual trainer selects checkpoints only on the screen-native synthetic
validation split. A frozen seed is evaluated on that split before epoch 1,
persisted as the epoch `-1` incumbent, and copied to `safety-best.pt`; a worse
first epoch therefore cannot displace it. Selection uses a constrained
high-water rule rather than a raw lexicographic tuple: meaningful secret-recall
gains must stay within per-class/aggregate recall and precision regression
bounds; sub-0.01 secret gains require a real other-sensitive recall gain; and
precision can break a tie only with no recall regression. Missing or non-finite
all-class metrics abort the attempt while the previous committed epoch remains
safe. The full per-class state is replayed and verified on schema-3 resume.

The trainer evaluates both the frozen input checkpoint and the final
safety-selected checkpoint on the untouched published WebPII test split. When
the verified ATS supplement is present, it also evaluates both checkpoints on
the identity-disjoint 200-image ATS test split. Those test metrics are written
to separate manifest blocks and never participate in checkpoint selection.

`visual-train` runs on an H200. Before Ultralytics starts, it validates the
composed source contracts and copies WebPII, screen-native synthetic data, and
the optional ATS supplement from the Modal Volume to container-local ephemeral
disk with 64 bounded I/O workers on the 16-CPU H200 container. Every source and
destination file is still hashed and the complete source/target tree inventory
is verified; the configured worker count is recorded in the staging manifest.
Training and evaluation then use only those verified local paths. The immutable
Volume provenance is persisted inside the selected attempt root as
`staged-dataset.yaml`, `staged-source-manifest.json`, and
`dataset-staging-manifest.json`.

New training attempts disable `optimizer=auto` and record an immutable
`stable-adamw-v1` policy: AdamW, `lr0=0.0005`, `lrf=0.1`, beta1/momentum `0.9`,
weight decay `0.0005`, one warmup epoch, warmup momentum `0.8`, zero bias warmup
LR, cosine decay, and `nbs=64`. Existing loss, conservative screenshot
augmentation, AMP, and gradient clipping remain unchanged. The manifest also
records a manual monitoring policy: review/stop after two consecutive epochs
with an absolute mAP50 drop over 0.10 or validation classification loss over 2x
the seed; automatic stopping is intentionally disabled.

After every model-save callback, training atomically writes schema-3
`safety-selection-progress.json` with full per-class selection state and the
last, aggregate-best, and PLVA safety-best checkpoint hashes, then commits the
Volume. Ultralytics is invoked with `save=True` and `save_period=1`, so completed
epochs are durable if the container is preempted.

Start the stable v3 retry as a new named, non-resume attempt. The snapshot path
is confined to `/vol/snapshots`, its caller-pinned SHA-256 is checked before and
after copying, and path/hash/license/inherited provenance are persisted under
`visual/training-stable/`. The existing `visual/training/` attempt is never
deleted or overwritten:

```bash
uvx --from modal==1.5.2 modal run --detach training/modal_app.py \
  --stage visual-train \
  --run-id plva-visual-agpl-ats-v3 \
  --visual-attempt stable \
  --visual-seed-checkpoint snapshots/2026-07-11-plva-visual-agpl-ats-v3-epoch2-safety/aggregate-best-epoch2.pt \
  --visual-seed-sha256 123e6d1d2129a8ba72eec9d736410b1d1b8f910319f987395dc4b12c4132d7bb \
  --visual-epochs 40
```

After an interrupted stable attempt has stopped, resume the same attempt with
the original total epoch count:

```bash
uvx --from modal==1.5.2 modal run --detach training/modal_app.py \
  --stage visual-train \
  --run-id plva-visual-agpl-ats-v3 \
  --visual-attempt stable \
  --visual-epochs 40 \
  --visual-resume
```

Resume loads only that attempt's hash-validated persisted `last.pt` and passes
`resume=True` to Ultralytics, restoring optimizer/scaler/EMA state and deriving
the scheduler position from the saved epoch. It also replays PLVA's seed
incumbent and complete constrained safety-selection history. The run ID,
attempt ID/root, frozen seed hash, attempt provenance, staged dataset/source
manifests, class contract, original epoch total, optimizer and augmentation
policies, and all progress checkpoint hashes must match. Named attempts cannot
be force-deleted, and resume cannot replace the frozen seed. The local
entrypoint launches long visual training with `.spawn().get()`.

This resume code is baked into a newly launched Modal image. It does not modify
or hot-patch an already-built running container, so do not issue the resume
command while the current job is still active; use it only for a future
relaunch after that job has exited or been preempted.

The earlier `visual/training/` v3 attempt used auto-selected MuSGD and is
intentionally stopped. Its old schema-1/2 progress is incompatible with the
stable optimizer/selection contract and must not be resumed. Export and
download the stable attempt explicitly after it completes:

```bash
uvx --from modal==1.5.2 modal run training/modal_app.py \
  --stage visual-export \
  --run-id plva-visual-agpl-ats-v3 \
  --visual-attempt stable

uvx --from modal==1.5.2 modal run training/modal_app.py \
  --stage visual-download \
  --run-id plva-visual-agpl-ats-v3 \
  --visual-attempt stable
```

## Tests

```bash
training/.venv/bin/python -m pytest -q training/tests
uvx --from ruff ruff check training scripts
npx --yes prettier@3.6.2 --check \
  scripts/fetch-models.mjs scripts/inspect-models.mjs \
  scripts/model-lock.mjs scripts/model-lock.test.mjs \
  scripts/verify-models.mjs training/runtime-parity/verify.mjs
```

Generated datasets, fixtures, checkpoints, model caches, and artifacts are
ignored by git.
