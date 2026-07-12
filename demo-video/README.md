# Holo / PLVA: 2-minute hackathon demo video

Built with [Remotion](https://www.remotion.dev/). 1920×1080 @ 30fps, exactly 120s.

```bash
npm install        # once
npm run dev        # opens Remotion Studio: scrub, tweak, preview live
npm run render     # renders out/holo-plva-demo.mp4
```

## Replacing the placeholders with real footage

Every dashed slot in the video is a **shot**. The full list lives in
[`src/shots.ts`](src/shots.ts): each entry describes exactly what to record.

1. Record/export the capture (1920×1080).
2. Drop the file into `public/shots/` (e.g. `public/shots/SHOT-03.mp4`).
3. In `src/shots.ts`, set that shot's `src` (e.g. `src: "shots/SHOT-03.mp4"`).
4. `npm run render`. No scene edits needed.

For videos, `startFrom: <seconds>` skips into your recording if the good part
isn't at the start.

### The shot list

| Shot | Kind | Appears | What to capture |
|---|---|---|---|
| SHOT-01 | image | 0:05–0:14 (problem beat) | A realistic desktop screenshot with **fake** PII visible: email inbox with an address, login form with a password, payment page with a card number. Same scenario as the live run. |
| SHOT-02A | image | 1:00–1:14 (split, left) | The **raw** frame from a real PLVA run: same frame as 02B, pixel-aligned. Export the exact upstream frame so it is provably what left the machine. |
| SHOT-02B | image | 1:00–1:14 (split, right) | The **redacted** frame the proxy actually sent upstream, from the audit viewer (`127.0.0.1:18081/viewer`), painted `«CLASS_n_nonce»` chips visible. |
| SHOT-03 | video ~13s | 1:14–1:28 (live proof) | Left: terminal running `PLVA_REDACT=1 PLVA_REDACT_ENGINE=vision ./run_step1.sh`. Right: audit viewer live-updating with redacted frames **during the same run**. Must include a chipped frame appearing at the same instant the agent acts, and the task finishing. |
| SHOT-04 | video ~6s | 1:41–1:52 (adoption) | Browser at `127.0.0.1:18080`: flip the PLVA master toggle ON, scroll the per-class editor (PASSWORD must show **blocked**). Clean cursor movement. |

## Music

Drop a 120s track at `public/music.mp3` and uncomment the `<Audio>` block at
the bottom of `src/Main.tsx`. Direction: precise minimal electronic ~122 BPM :
ticking pulse for 0–24s, confident cut at the title reveal (0:24), arps
layering through the pipeline (0:33–1:00), half-time pads under the split
(1:00), near-silence for the invariants (1:28), **hard cut to silence one
second before the end**.

## Timeline (for tweaks)

| Time | Scene | File |
|---|---|---|
| 0:00–0:05 | Cold open: "your agent screenshotted your password" | `scenes/S01ColdOpen.tsx` |
| 0:05–0:14 | The problem: raw frames to the provider, every step | `scenes/S02Problem.tsx` |
| 0:14–0:24 | The constraint → the hack: "we became the base URL" | `scenes/S03Constraint.tsx` |
| 0:24–0:33 | Title: Holo + architecture lockup (H Company Holo3) | `scenes/S04Title.tsx` |
| 0:33–0:47 | Centerpiece I: request leg (detect → paint → vault → scrub → ship) | `scenes/S05RequestLeg.tsx` |
| 0:47–1:00 | Centerpiece II: response leg (token resolves in transit) | `scenes/S06ResponseLeg.tsx` |
| 1:00–1:14 | With vs Without: same task, only one leaked | `scenes/S07Compare.tsx` |
| 1:14–1:28 | Live proof: "Don't trust us. Audit it." | `scenes/S08LiveProof.tsx` |
| 1:28–1:41 | Fail-closed invariants + per-class policy | `scenes/S09FailClosed.tsx` |
| 1:41–1:52 | Adoption: one config line + control panel | `scenes/S10OneLine.tsx` |
| 1:52–2:00 | Endcard: the koan + the chip flip | `scenes/S11Endcard.tsx` |

All timings live in the `BEATS` table in `src/Main.tsx`; scene-internal timing
is relative to each scene's start. Design tokens (colors/type) are in
`src/theme.ts` and match the Holo app.

An optional voiceover script (the video works without it) is in
[`VOICEOVER.md`](VOICEOVER.md).
