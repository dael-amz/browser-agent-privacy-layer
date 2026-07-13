---
name: PLVA Tool Probe
description: Synthetic experiment for invoking the local plva_add tool without desktop actions.
publisher: PLVA
version: "0.1.0"
license: MIT
tools:
  - name: plva_add
    description: Add two synthetic integers locally.
    parameters:
      type: object
      properties:
        a: {type: integer}
        b: {type: integer}
        request_id: {type: string}
      required: [a, b, request_id]
---

[PLVA_TOOL_PROBE_SKILL_BEGIN]

`plva_add(a, b, request_id)` is a synthetic, side-effect-free local tool used
only to test Holo's tool-call channel. When explicitly asked to use it, invoke
it through the runtime's native skill tool mechanism. Do not calculate the
answer yourself and do not click, type, scroll, or otherwise act on the desktop.

[PLVA_TOOL_PROBE_SKILL_END]
