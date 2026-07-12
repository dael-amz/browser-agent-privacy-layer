/**
 * THE SHOT LIST — every real capture that replaces a placeholder.
 *
 * To replace a placeholder:
 *   1. Record/export the capture described below (1920x1080).
 *   2. Drop the file into demo-video/public/shots/  (e.g. public/shots/SHOT-03.mp4)
 *   3. Set `src` below (e.g. src: "shots/SHOT-03.mp4"). Done — the video updates.
 *
 * Videos are trimmed to fit their slot automatically; pick `startFrom` (seconds
 * into your recording) if the good part isn't at the start.
 */
export type ShotDef = {
  id: string;
  kind: "video" | "image";
  title: string;
  description: string;
  seconds: number;
  src: string | null;
  /** seconds to skip into the source video before playing */
  startFrom?: number;
};

export const SHOTS = {
  "SHOT-01": {
    id: "SHOT-01",
    kind: "image",
    title: "Raw desktop the agent sees",
    description:
      "Screenshot of a realistic desktop with FAKE PII visible: email inbox showing an address, a login form with a password, a payment page with a card number. Same scenario as the live run.",
    seconds: 9,
    src: null,
  },
  "SHOT-02A": {
    id: "SHOT-02A",
    kind: "image",
    title: "WITHOUT PLVA — raw frame sent upstream",
    description:
      "The original screenshot with fake PII visible (same frame as SHOT-02B, pixel-aligned). Export the exact upstream frame so it is provably what left the machine.",
    seconds: 14,
    src: null,
  },
  "SHOT-02B": {
    id: "SHOT-02B",
    kind: "image",
    title: "WITH PLVA — redacted frame sent upstream",
    description:
      "The actual redacted frame from the audit viewer (127.0.0.1:18081/viewer) with painted «CLASS_n_nonce» chips over the same coordinates as SHOT-02A.",
    seconds: 14,
    src: null,
  },
  "SHOT-03": {
    id: "SHOT-03",
    kind: "video",
    title: "Live run — terminal + audit viewer",
    description:
      "~13s screen recording: left = terminal running PLVA_REDACT=1 PLVA_REDACT_ENGINE=vision ./run_step1.sh with step logs; right = browser at 127.0.0.1:18081/viewer live-updating with redacted frames DURING the same run. Must include a chipped frame appearing in the viewer at the same instant the agent acts on the real screen, and the task finishing.",
    seconds: 13,
    src: null,
  },
  "SHOT-04": {
    id: "SHOT-04",
    kind: "video",
    title: "Holo control panel — policy editor",
    description:
      "~6s screen recording: browser at 127.0.0.1:18080 — flip the PLVA master toggle ON, then scroll the per-class security editor (PASSWORD must show 'blocked'). Clean cursor movement.",
    seconds: 6,
    src: null,
  },
} as const satisfies Record<string, ShotDef>;

export type ShotId = keyof typeof SHOTS;
