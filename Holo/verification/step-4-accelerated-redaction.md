# Step 4 accelerated redaction worker

Date: 2026-07-11

Status: **BUILT and locally verified for the obscuring half of Step 4.** Vault, placeholder,
resolution, and history-scrub work remains open because the frozen detector emits geometry only.

## Architecture

- `redactor-worker/` imports the separate frozen baseline's detector source without modifying the
  baseline or its integrity manifests.
- One private headless-Chrome process starts on demand, warms the visual, RapidOCR, and Rampart
  sessions once per active burst, and releases after 60 idle seconds by default.
- Visual inference uses WebGPU when available. RapidOCR uses a separate WASM runtime and overlaps
  visual GPU work; semantic classification follows OCR.
- Both detector branches are mandatory. Any detector, semantic, render, browser, IPC, or protocol
  failure rejects the request without forwarding a frame.
- A bounded memory-only LRU maps an exact raw-frame digest to redacted PNG bytes. It stores no raw
  frame and is erased on close.
- Frame endpoints use random-port loopback plus an unguessable per-process token. Chrome external
  networking is forced to a closed proxy.
- The memory-only `/viewer` remains and refreshes every 250 ms.

## Performance evidence

Measured with the bundled 960×960 ATS fixture on the development M4 host:

| Path | Wall time |
|---|---:|
| Frozen CLI, fresh Node + Chrome per frame | 6.1–9.5 s |
| Accelerated WebGPU worker startup/warm-up | 5.2 s |
| First cold frame (startup plus inference) | about 8.6 s |
| Accelerated WebGPU, distinct warm frames | 3.3–3.4 s |
| Accelerated WASM, distinct warm frames | about 3.5 s |
| Accelerated exact-frame memory hit | below timer resolution |

The accelerated output was byte-for-byte identical to the frozen CLI's 105,264-byte known-good
redacted PNG. An integration smoke confirmed that one request reached the mock upstream, its frame
was not raw, `/viewer/frame` matched the outbound redacted bytes, and the viewer counted it.

The warm Chrome/ONNX process family measured about 1.6 GiB RSS. Because the containing app is
always on but CUA use may be sparse, retaining it for the full app lifetime is not a good default.
Adaptive mode pays startup once per CUA burst, keeps the sessions across closely spaced steps, and
then returns that memory after 60 idle seconds. `eager` remains available when CUA calls are frequent
enough to amortize the memory, while `cold` minimizes residency at the cost of paying startup on
every distinct frame.

Two more aggressive options were rejected after measurement. Cross-origin isolation for threaded
WASM caused ONNX Runtime warm-up to stall in headless Chrome on this host, including at one thread.
Moving OCR onto WebGPU shortened warm-up but rejected the first real frame. The known-good worker
therefore retains isolated one-thread WASM OCR alongside WebGPU visual inference rather than
accepting an unreliable startup or inference path.

## Verification

- Frozen baseline integrity verification passes unchanged.
- Python gate: 80 tests pass; Ruff and strict mypy pass; coverage is about 82%.
- Node dependency audit reports zero known vulnerabilities.
- Generated worker `dist/` and `node_modules/` are ignored and contain no session data.
