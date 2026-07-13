---
language:
  - en
library_name: transformers
pipeline_tag: token-classification
base_model: google/bert_uncased_L-4_H-256_A-4
license: other
---

# PLVA compact PII tagger candidate

Status: pipeline implemented; no custom replacement is approved for release.

The pinned Rampart model remains PLVA's semantic bootstrap. This card is a
template for a future screen-native replacement. Export generates a measured
card next to each artifact and marks it release eligible only after all safety,
cross-runtime, quantization, provenance, and license gates pass.

## Intended use

The candidate classifies English OCR text spans inside PLVA's on-device
screenshot-redaction pipeline. It is not a standalone detector, vault, policy
engine, compliance product, or general document-redaction guarantee.

## Taxonomy

`NAME`, `ADDRESS`, `DOB`, `GOV_ID`, `EMAIL`, `PHONE`, `BANK_ACCOUNT`,
`CARD_NUMBER`, `CVC`, `PASSWORD`, `API_KEY`, `AUTH_TOKEN`, and `PRIVATE_KEY`.

Secret classes require perfect frozen-holdout recall and zero missed entities.
Contextual and aggregate gates are recorded in the generated evaluation report.

## Training and evaluation data

- AI4Privacy OpenPII 1.5M at the pinned revision, English subset, CC BY 4.0.
- Deterministic screen-native fake text with template, seed, and canonical-value
  separation across train, validation, and holdout.
- Span-preserving synthetic OCR corruptions.
- A separate never-trained holdout made from the pinned RapidOCR stack's actual
  recognized text.

The published OpenPII train and validation splits remain separate. Generic
dates map to `DOB` only with birth context. Screen-only secret classes use fake
generated values.

## Model and export

- Candidate base: Google compact BERT L4-H256 at the pinned revision.
- Selection metric: safety-weighted recall.
- Calibration: per-class thresholds on validation only.
- Export: dynamic batch and sequence axes, explicit names, FP32 reference, then
  dynamic per-channel INT8 quantization.
- Runtime ceiling: 20 MB for INT8 model plus tokenizer and runtime config.
- Required parity: PyTorch, ONNX Python, Node, browser WASM, and browser WebGPU.

## Current metrics

No release metrics are reported here because no release candidate exists.
Generated cards copy measured values from committed evaluation artifacts. A
development smoke result must not be presented as model quality evidence.

## Limitations and risks

The model depends on OCR coverage and alignment. It does not detect faces,
signatures, handwriting, physical documents, or other non-text PII. Unsupported
languages and scripts are out of scope for v0.1. Both visual and OCR discovery
can miss content, so the full pipeline retains fail-closed deterministic and
uncertainty rules.

Synthetic data can underrepresent real UI diversity. AI4Privacy text is not a
substitute for screenshot holdouts. GUIGuard contains real sensitive
screenshots, is licensed CC BY-NC 4.0, and remains evaluation-first.

## Attribution and license

The candidate base is Apache-2.0. AI4Privacy data is CC BY 4.0. RapidOCR and
PaddleOCR assets are Apache-2.0. Rampart is the active bootstrap model and
replacement baseline under CC BY 4.0.

The final trained artifact license is pending maintainer review. Export records
that status and publication refuses any pending or incompatible license.

The optional visual detector development track uses a hash-pinned YOLO11n base
under AGPL-3.0-only. Its output cannot be treated as closed-source or
commercially distributable unless an Ultralytics Enterprise license is obtained
and recorded.
