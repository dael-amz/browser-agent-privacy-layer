# Step 6.5 verification — Holo tool-call channel

Verified 2026-07-12 against H Company `holo3-1-35b-a3b` using synthetic values only.

## Result

| Candidate channel | Grammar permits | Model emits | Proxy parses | Result round-trips |
|---|---:|---:|---:|---:|
| Native skill-declared tool | No | No | No | No |
| Structured action | Yes | Yes | No | No |
| Free-text marker | Yes | Yes | Yes | Yes |

The recommended model-mediated channel is the free-text marker. The proxy must parse a strict,
bounded marker grammar, execute only a registered local operation, and inject a value-free result
into the next observation. The mandatory fallback is the same operation initiated directly by the
proxy/app, without requiring a model-emitted call. Marker compliance varied across two otherwise
identical runs (one complete round trip, one missed emission), so dependent steps must never rely
on model compliance for correctness or safety.

## Captured schema summary

- Authentic Holo runtime request captured: yes; runtime exit: `0`.
- Request fields: `chat_template_kwargs`, `logit_bias`, `max_tokens`, `messages`, `model`,
  `structured_outputs`, `temperature`.
- `structured_outputs`: present; native `tools`: absent.
- Output string fields: `answer.content`, `answer.tool_name`, `thought`.
- `answer.tool_name`: unconstrained string; no action enum was present.
- Temporary skill instructions observed in the runtime system message: no.
- Value-free structured schema fingerprint: `499bddf3280b`.

The structured channel was grammatically able to name `plva_add`, and the model mentioned that
action, but it did not emit the exact `tool_calls` argument object required by the deterministic
parser. It therefore failed before local execution. The native channel cannot work on this
runtime contract because neither the skill nor an OpenAI-compatible `tools` declaration reaches
the model request.

## Privacy and reproduction

The probe retains only schema metadata, HTTP statuses, booleans, and stable error codes. It never
records request/response text, desktop pixels, credentials, or arguments beyond fixed synthetic
`add(3, 4)` values. It captures the real Holo request only in memory, replaces all user/history
messages and the screen image before provider egress, uses an ephemeral runtime directory, and
restores/removes its temporary skill after the run.

```bash
$HOME/.local/bin/uv run plva-tool-probe \
  --provider hcompany \
  --output /tmp/plva-tool-probe.json
```

The JSON report in `/tmp` is safe to inspect and contains the matrix and schema summary above.
