import { interpolate, spring } from "remotion";

export const EASE_OUT = (t: number) => 1 - Math.pow(1 - t, 4);
export const EASE_IN_OUT = (t: number) =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

/** Snappy default spring. */
export const pop = (frame: number, fps: number, delay = 0) =>
  spring({ frame: frame - delay, fps, config: { damping: 200, stiffness: 160, mass: 0.8 } });

/** Springier, with slight overshoot — for chips and badges. */
export const bounce = (frame: number, fps: number, delay = 0) =>
  spring({ frame: frame - delay, fps, config: { damping: 14, stiffness: 180, mass: 0.6 } });

/** 0→1 over [from, from+dur] with ease-out quart. */
export const ramp = (frame: number, from: number, dur: number) =>
  interpolate(frame, [from, from + dur], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: EASE_OUT,
  });

/** 1→0 over [from, from+dur]. */
export const rampOut = (frame: number, from: number, dur: number) =>
  interpolate(frame, [from, from + dur], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: EASE_IN_OUT,
  });

/** Fade in at start, fade out before `total`. */
export const inOut = (frame: number, total: number, inDur = 10, outDur = 10) =>
  Math.min(ramp(frame, 0, inDur), rampOut(frame, total - outDur, outDur));

/** Standard slide-up + fade entrance. Returns style. */
export const rise = (frame: number, fps: number, delay = 0, dist = 34) => {
  const p = pop(frame, fps, delay);
  return { opacity: p, transform: `translateY(${(1 - p) * dist}px)` };
};

export const sec = (s: number, fps = 30) => Math.round(s * fps);
