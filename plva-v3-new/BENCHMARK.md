# PLVA v3 latency benchmark

Measured on 2026-07-12 on the same local Mac, Chrome build, fake 960x960 ATS
fixture, high-recall profile, and single-thread WASM runtime. Each measurement
starts a fresh CLI process and fresh browser profile, so it includes cold model
loading. Values are milliseconds from the browser pipeline timer.

| Bundle | Cold runs | Median | Visual detections | Final masks |
| --- | --- | ---: | ---: | ---: |
| v2 baseline | 3398, 3305, 3354 | 3354 | 0 | 1 |
| v3 | 3184, 3103, 3052 | 3103 | 3 | 3 |

V3 was 251 ms, or 7.5%, faster at the median while exercising the new visual
model. The main latency change is safe overlap: the detector runs in an isolated
single-thread worker while RapidOCR runs on the page. Worker failure retries the
same input on the serialized main runtime. Rampart loading overlaps both paths.

This is a packaging regression benchmark, not a general throughput or accuracy
claim. Host, browser, screenshot complexity, OCR box count, and warm-session
reuse materially change latency.
