# Step 4 (partial) — real obscuring via the frozen plva-v2-baseline detector

Date: 2026-07-11

Status: **Obscuring path BUILT and locally verified.** The operator supplied
`plva-v2-baseline/` and asked to hook it up for working obscuring plus a live view of the
obscured frames. This delivers Step 4's *redaction* half through the §5 plug-in seam; the vault /
placeholder / history-scrub half is NOT built (see "What this is not" below).

> The default performance path is now the persistent parallel WebGPU/WASM worker documented in
> `step-4-accelerated-redaction.md`. The frozen CLI below remains available through
> `--redact-engine baseline` as the correctness oracle.

## What plva-v2-baseline is

A frozen, standalone harness of the old PLVA v2 detection pipeline: a visual detector +
RapidOCR (PaddleOCR ONNX models) + the Rampart semantic policy, executed inside headless Chrome
via ONNX-Runtime WASM, fully offline (loopback-only server, wildcard DNS nulled). CLI:
`node bin/plva-v2.mjs <png> --output <png> --report <json>`. It burns masks into a new PNG and
emits a geometry-only report (no recognized text, no paths).

Bundle's own caveats (from its README/snapshot): **development-only, not release-eligible** —
the detector checkpoint is AGPL-3.0-only, and its measured secret-class recall was ~zero
(missed 6 of 7 secret boxes in its WebPII quick-100 diagnostic). Good enough to demonstrate the
pipeline; not a production privacy boundary.

## Integration (this repo)

- `src/plva_proxy/redactor.py` — subprocess wrapper: frame → private temp dir (deleted before
  returning) → v2 CLI → redacted PNG + region count. Any failure raises; log lines carry region
  counts and exit codes only.
- `frame_redaction_hook` in `proxy.py` — a request hook on the Step 3 seam: every outbound
  screenshot is decoded, converted to PNG when needed, redacted, and swapped (new buffer, §8.3).
  Requests with no screenshot pass through; **any redaction failure → 502, nothing forwarded**
  (§8.1) — there is no raw-frame fallback (§8.2). Runs in a threadpool so the proxy stays
  responsive during the ~4–6 s/frame pipeline.
- **Viewer** — `plva-proxy --redact …` serves `http://127.0.0.1:<port>/viewer`: a loopback-only
  page that live-polls the redacted frames the model sees. Frames live in a memory-only ring
  buffer (`FrameStore`, 8 frames), never persisted (§8.6), and only post-redaction pixels are
  ever exposed.
- CLI: `plva-proxy --redact plva-v2-baseline [--redact-profile high-recall|balanced]`;
  runbook: `PLVA_REDACT=1 ./run_step1.sh`.

## Local verify evidence

- The bundle's own `verify-snapshot` passed (frozen hashes intact) and a direct fixture run
  masked 1 sensitive region in ~5.6 s (Chrome startup included).
- Real-socket smoke: capture stub upstream, `--redact` on; the synthetic request carried the
  73,255-byte fixture PNG and the stub received a 105,264-byte PNG at identical 960×960
  dimensions — byte-identical (sha256) to the known-good redacted output. `/viewer/frame` served
  the same bytes; `/viewer/stats` counted it.
- Fail-closed and pass-through paths covered in `tests/test_redaction.py`.

## What this is not (yet)

- No spans/values → **no vault, no placeholders, no resolution, no history scrub**: the v2
  report is geometry-only by design, so the model sees masks, not `EMAIL_1`-style chips, and
  nothing is typed back. Full Step 4 needs the real §5 `redact(frame) → {redactedFrame, spans}`
  detector.
- Latency: ~4–6 s per step (fresh Chrome per frame). Acceptable for demo; a persistent worker
  would cut most of it.

## Live use

```bash
PLVA_REDACT=1 ./run_step1.sh "your task"
# then open http://127.0.0.1:18081/viewer to watch what the model sees
```

## Addendum — "stuck frame" investigation (same day)

A live run appeared frozen on the first redacted frame. The proxy log disproved a proxy bug:
18 requests flowed, each freshly redacted, but `request_bytes` grew by **exactly 1216 bytes per
step** (text history only) and every frame masked exactly 17 regions — i.e. **the closed runtime
re-sent the byte-identical first screenshot every step**. A controlled two-frame test through the
same proxy produced two distinct redacted frames and viewer updates. Conclusion: the runtime only
captures a new screenshot after it *executes an action*; answer-only tasks ("describe what you
see") never trigger recapture. Not fixable in transit.

Follow-ups shipped:

- **Per-frame instrumentation**: the proxy logs privacy-safe byte counts and `/viewer` shows the
  frame number, redacted-output sha, and timestamp — identical outputs are visible at a glance.
- **`plva-live`** — true continuous redaction independent of the runtime: captures the screen in
  a loop (`screencapture`), redacts each frame, and serves it at
  `http://127.0.0.1:18082/viewer`. No upstream, no key, memory-only frames. `--scale 0.5`
  (default) runs ~4–5 s/frame. Smoke-verified: 3 varying frames in 14 s.
- For agent runs, use tasks that act (click/scroll/open) — each executed action yields a fresh
  capture through the redaction path.
