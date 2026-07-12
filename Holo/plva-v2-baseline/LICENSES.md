# PLVA model and data attribution

This file records model-engineering inputs and their current use. A dataset
license does not automatically license a model checkpoint from a different
repository.

## RapidOCR and PaddleOCR

PLVA pins the PP-OCRv4 mobile detector, English recognizer, and matching
dictionary distributed by RapidOCR v3.9.1.

- License: Apache-2.0
- RapidOCR source: https://github.com/RapidAI/RapidOCR
- PaddleOCR source: https://github.com/PaddlePaddle/PaddleOCR

## Rampart

Rampart is PLVA's active bootstrap semantic model and required replacement
baseline, not merely design inspiration. PLVA pins its quantized ONNX model,
tokenizer, class order, normalization contract, runtime package, and model-only
goldens.

- Model: `nationaldesignstudio/rampart`
- Model revision: `b1993e4e68b082835b80ffc65acc03325ea2e501`
- Runtime package: `@nationaldesignstudio/rampart@0.1.3`
- License: CC-BY-4.0
- Source: https://github.com/nationaldesignstudio/rampart

Attribution must accompany any permitted distribution or public use.

## WebRedact proposal model

PLVA pins WebRedact nano 640 OpenVINO IR at source revision
`de27fa3ae82bfab34fa28281fc9dc9786fd01600` for reference inference.

The public model repository does not currently contain a license file. Its
artifact license is recorded as `NOASSERTION`. Do not redistribute the XML/BIN
files or a derivative conversion until the maintainer confirms permission. The
Apache-2.0 license on the WebPII dataset does not resolve this model-license gap.

- Model repository: https://github.com/WebPII/models
- Dataset/project: https://webpii.github.io/

## WebPII dataset

- Dataset: `WebPII/webpii`
- Revision: `6d3317721b72bde719a361c564ceaf1fbded3a8e`
- License: Apache-2.0
- Use: visual detector train data and untouched published test split

## GUIGuard-Bench

- Dataset: `ShaofantuoshuzhengzhiSha/GUIGuard-Bench`
- Revision: `c2b5ca415cd854452585260b9eaa74699042d920`
- License: CC-BY-NC-4.0
- Use: evaluation-first

GUIGuard includes real sensitive screenshots. Raw images and OCR text are not
persisted by default. Its non-commercial license requires separate review for
any commercial adaptation.

## AI4Privacy OpenPII 1.5M

- Dataset: `ai4privacy/pii-masking-openpii-1.5m`
- Revision: `a785eb528e28be2693c3718a27e066970de5dadb`
- License: CC-BY-4.0
- Attribution: AI4Privacy / Ai Suisse SA
- Source: https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m

PLVA preserves the published train/validation split and filters English
variants. The dataset card describes the PII as synthetic.

## Google compact BERT candidate

- Model: `google/bert_uncased_L-4_H-256_A-4`
- Revision: `387825ce42dbb39b87911cdf8e383ee3b25184f8`
- License: Apache-2.0
- Use: optional semantic replacement base

## ONNX Runtime

The parity harness pins `onnxruntime-node@1.27.0` and
`onnxruntime-web@1.27.0`.

- License: MIT
- Source: https://github.com/microsoft/onnxruntime

## Optional Ultralytics training tool

The detector training environment pins Ultralytics 8.4.92, distributed under
AGPL-3.0 unless a separate commercial license applies. The authorized
development base is official YOLO11n from `ultralytics/assets` release v8.4.0:

- Bytes: 5,613,764
- SHA-256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`
- License: AGPL-3.0-only
- Source: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt

AGPL training and evaluation are authorized for this project. PLVA does not
infer that the resulting checkpoint may ship in a closed-source product. That
requires an Ultralytics Enterprise license plus review of all base-checkpoint,
data, and artifact obligations.

## PLVA-generated artifacts

Screen-native fixtures and generated PII are fake. Generated data, model caches,
checkpoints, and development artifacts are ignored by git.

The final semantic or visual replacement license is intentionally
`pending-maintainer-review` until confirmed. Release validation and publishing
refuse pending licenses.
