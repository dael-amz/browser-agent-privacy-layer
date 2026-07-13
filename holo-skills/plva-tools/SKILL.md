---
name: plva-tools
description: Call fast local computation tools (echo, add, sort) through this private session when a task needs a computed result instead of a desktop action.
---

# Local computation tools

This session provides local tools that run on this computer, outside your view,
and return instantly. Available tools:

- `echo(text)` — repeats `text` back.
- `add(a, b)` — returns the sum of the numbers `a` and `b`.
- `sort(items)` — returns the strings in `items` in ascending order.

To call a tool, emit exactly one action of the form
`{"tool_calls": [{"tool_name": "plva_tool", "name": "<tool>", "args": {...}}]}`
and nothing else in that step. If your output format rejects that action, instead
write the single line `⟦TOOL⟧{"name": "<tool>", "args": {...}}⟦/TOOL⟧` inside your
thought or answer text.

After a call, the next user message begins with `[PLVA_TOOL_RESULT]` and carries
the result. Continue the task using that result and do not repeat an identical
call. The live session instructions are authoritative if they differ from this
document.
