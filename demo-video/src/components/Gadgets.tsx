import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { C, FONT, MONO } from "../theme";
import { pop, ramp } from "../anim";

/** Rolling counter, odometer style. */
export const Odometer: React.FC<{
  value: number;
  label?: string;
  color?: string;
  size?: number;
}> = ({ value, label, color = C.red, size = 30 }) => (
  <span style={{ fontFamily: MONO, fontSize: size, fontWeight: 700, color }}>
    {label}
    <span
      style={{
        display: "inline-block",
        minWidth: size * 1.4,
        textAlign: "right",
        background: `${color}14`,
        borderRadius: 8,
        padding: `2px ${size * 0.3}px`,
        marginLeft: 10,
      }}
    >
      {value}
    </span>
  </span>
);

/** Continuous horizontal marquee. */
export const Ticker: React.FC<{
  text: string;
  color: string;
  background?: string;
  speed?: number;
  fontSize?: number;
}> = ({ text, color, background = "transparent", speed = 3, fontSize = 24 }) => {
  const frame = useCurrentFrame();
  const shift = (frame * speed) % 1400;
  return (
    <div style={{ overflow: "hidden", whiteSpace: "nowrap", background, padding: "10px 0" }}>
      <div style={{ transform: `translateX(${-shift}px)`, display: "inline-block" }}>
        {[0, 1, 2, 3].map((i) => (
          <span key={i} style={{ fontFamily: MONO, fontSize, fontWeight: 600, color, paddingRight: 60 }}>
            {text}
          </span>
        ))}
      </div>
    </div>
  );
};

/** Rubber-stamp entrance: rotated slam with a hard settle. */
export const Stamp: React.FC<{
  children: React.ReactNode;
  delay?: number;
  rotate?: number;
  style?: React.CSSProperties;
}> = ({ children, delay = 0, rotate = -4, style }) => {
  const frame = useCurrentFrame();
  const t = frame - delay;
  const s = t < 0 ? 0 : t < 3 ? 1.35 - 0.35 * (t / 3) : 1;
  const o = t < 0 ? 0 : Math.min(1, t / 2);
  return (
    <div
      style={{
        display: "inline-block",
        transform: `scale(${Math.max(s, 0.001)}) rotate(${rotate}deg)`,
        opacity: o,
        ...style,
      }}
    >
      {children}
    </div>
  );
};

const GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@._-";

/** Split-flap / decode morph from one string to another, char by char. */
export const DecodeText: React.FC<{
  from: string;
  to: string;
  progress: number; // 0..1
  style?: React.CSSProperties;
}> = ({ from, to, progress, style }) => {
  const len = Math.max(from.length, to.length);
  const out: string[] = [];
  for (let i = 0; i < len; i++) {
    // each char settles when overall progress passes its slot
    const settle = (i + 1) / (len + 2);
    if (progress >= settle) {
      if (i < to.length) out.push(to[i]);
    } else if (progress >= settle - 0.18) {
      // scrambling window — deterministic pseudo-random
      const k = (i * 31 + Math.floor(progress * 90) * 17) % GLYPHS.length;
      out.push(GLYPHS[k]);
    } else if (i < from.length) {
      out.push(from[i]);
    }
  }
  return <span style={{ fontFamily: MONO, ...style }}>{out.join("")}</span>;
};

/** Task progress bar with a checkmark that lands on completion. */
export const TaskBar: React.FC<{
  progress: number; // 0..1
  color: string;
  label: string;
  done: boolean;
}> = ({ progress, color, label, done }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
      <span style={{ fontFamily: FONT, fontSize: 22, fontWeight: 600, color: C.ink, width: 170 }}>{label}</span>
      <div style={{ flex: 1, height: 14, background: "rgba(12,12,12,.08)", borderRadius: 999, overflow: "hidden" }}>
        <div
          style={{
            width: `${Math.min(1, Math.max(0, progress)) * 100}%`,
            height: "100%",
            background: color,
            borderRadius: 999,
          }}
        />
      </div>
      <div
        style={{
          width: 40,
          height: 40,
          borderRadius: 999,
          background: done ? color : "rgba(12,12,12,.08)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          transform: `scale(${done ? pop(frame, fps, 0) * 0 + 1 : 1})`,
        }}
      >
        <svg width="22" height="22" viewBox="0 0 24 24">
          <path
            d="M4 12.5 L9.5 18 L20 6.5"
            stroke={done ? C.white : "rgba(12,12,12,.25)"}
            strokeWidth="3.4"
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
    </div>
  );
};

/** Green callout: arrow line + pill label, punches in. */
export const Callout: React.FC<{
  x: number;
  y: number;
  dx?: number;
  dy?: number;
  text: string;
  delay?: number;
  color?: string;
}> = ({ x, y, dx = 120, dy = -80, text, delay = 0, color = C.green }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = pop(frame, fps, delay);
  const draw = ramp(frame, delay, 12);
  return (
    <div style={{ position: "absolute", left: x, top: y, pointerEvents: "none" }}>
      <svg width={Math.abs(dx) + 20} height={Math.abs(dy) + 20} style={{ position: "absolute", left: dx < 0 ? dx : 0, top: dy < 0 ? dy : 0, overflow: "visible" }}>
        <line
          x1={dx < 0 ? Math.abs(dx) : 0}
          y1={dy < 0 ? Math.abs(dy) : 0}
          x2={dx < 0 ? Math.abs(dx) - dx * draw : dx * draw}
          y2={dy < 0 ? Math.abs(dy) - dy * draw : dy * draw}
          stroke={color}
          strokeWidth={4}
          strokeLinecap="round"
        />
        <circle cx={dx < 0 ? Math.abs(dx) : 0} cy={dy < 0 ? Math.abs(dy) : 0} r={7 * p} fill={color} />
      </svg>
      <div
        style={{
          position: "absolute",
          left: dx + (dx < 0 ? -14 : 14),
          top: dy + (dy < 0 ? -54 : 8),
          transform: `translateX(${dx < 0 ? "-100%" : "0"}) scale(${p})`,
          transformOrigin: dx < 0 ? "right center" : "left center",
          fontFamily: FONT,
          fontSize: 24,
          fontWeight: 650,
          color: C.white,
          background: color,
          borderRadius: 999,
          padding: "10px 24px",
          whiteSpace: "nowrap",
          boxShadow: "0 8px 30px rgba(12,12,12,.18)",
        }}
      >
        {text}
      </div>
    </div>
  );
};

/** Magnifier ring that can glide and park. */
export const MagnifierRing: React.FC<{
  x: number;
  y: number;
  r?: number;
  color?: string;
  opacity?: number;
}> = ({ x, y, r = 90, color = C.ink, opacity = 1 }) => (
  <div
    style={{
      position: "absolute",
      left: x - r,
      top: y - r,
      width: r * 2,
      height: r * 2,
      borderRadius: 999,
      border: `6px solid ${color}`,
      boxShadow: "0 18px 50px rgba(12,12,12,.25), inset 0 0 0 2px rgba(255,255,255,.6)",
      opacity,
    }}
  >
    <div
      style={{
        position: "absolute",
        right: -r * 0.52,
        bottom: -r * 0.28,
        width: r * 0.72,
        height: 12,
        background: color,
        borderRadius: 999,
        transform: "rotate(40deg)",
      }}
    />
  </div>
);

/** Blinking REC-style badge. */
export const RecBadge: React.FC<{ text?: string; style?: React.CSSProperties }> = ({ text = "REC", style }) => {
  const frame = useCurrentFrame();
  const on = Math.floor(frame / 15) % 2 === 0;
  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 12,
        fontFamily: MONO,
        fontSize: 28,
        fontWeight: 700,
        color: C.red,
        letterSpacing: "0.08em",
        ...style,
      }}
    >
      <span
        style={{
          width: 18,
          height: 18,
          borderRadius: 999,
          background: C.red,
          opacity: on ? 1 : 0.25,
          boxShadow: on ? `0 0 18px 2px ${C.red}88` : "none",
        }}
      />
      {text}
    </div>
  );
};

/** Four corner brackets that snap around a region (the "you are the screenshot" gesture). */
export const CornerBrackets: React.FC<{
  inset?: number;
  size?: number;
  thickness?: number;
  color?: string;
  delay?: number;
}> = ({ inset = 40, size = 90, thickness = 8, color = C.red, delay = 0 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = pop(frame, fps, delay);
  const off = (1 - p) * 60;
  const corners: Array<[React.CSSProperties, React.CSSProperties]> = [
    [{ top: inset - off, left: inset - off }, { borderTop: `${thickness}px solid ${color}`, borderLeft: `${thickness}px solid ${color}` }],
    [{ top: inset - off, right: inset - off }, { borderTop: `${thickness}px solid ${color}`, borderRight: `${thickness}px solid ${color}` }],
    [{ bottom: inset - off, left: inset - off }, { borderBottom: `${thickness}px solid ${color}`, borderLeft: `${thickness}px solid ${color}` }],
    [{ bottom: inset - off, right: inset - off }, { borderBottom: `${thickness}px solid ${color}`, borderRight: `${thickness}px solid ${color}` }],
  ];
  return (
    <>
      {corners.map(([posn, borders], i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            width: size,
            height: size,
            opacity: p,
            ...posn,
            ...borders,
          }}
        />
      ))}
    </>
  );
};

/** White camera-shutter flash covering the frame; fires at `at`. */
export const ShutterFlash: React.FC<{ at: number }> = ({ at }) => {
  const frame = useCurrentFrame();
  const o = interpolate(frame, [at, at + 2, at + 8], [0, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        background: C.white,
        opacity: o,
        pointerEvents: "none",
        zIndex: 50,
      }}
    />
  );
};
