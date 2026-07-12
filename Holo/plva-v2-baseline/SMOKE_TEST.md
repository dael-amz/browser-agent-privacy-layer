# Frozen harness smoke evidence

Executed 2026-07-11 on macOS with Node 25.6.1 and local Google Chrome using:

```sh
node bin/smoke-test.mjs
```

Result:

```text
PASS: 1 mask(s), 8004 masked pixels, dimensions 960x960, outside changes 0, local-only runtime.
```

Assertions passed:

- fixture SHA-256 matched the bundled synthetic ATS fixture;
- the full visual, RapidOCR, Rampart, fusion, and PNG renderer path completed;
- semantic mode was `rampart`, not the heuristics-only fallback;
- one or more final masks and masked pixels were present;
- input and decoded output were both 960x960;
- zero pixels outside final masks changed;
- every pixel inside final masks decoded to the exact redaction color;
- browser page resources were loopback-only and external Chrome destinations
  were blocked by launch policy;
- the report contained no raw OCR text field, email-like value, or filesystem
  path;
- detector SHA-256 was the pinned v2 hash
  `450ee07452d618eb3159d565f1787b17a8cabbfee3e72686e932460be576cc2e`.

On failure, `bin/smoke-test.mjs` retains the complete captured CLI stdout and
stderr in its assertion message so model failures and timeouts are distinct.
