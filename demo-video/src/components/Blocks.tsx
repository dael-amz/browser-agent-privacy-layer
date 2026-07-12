import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { C, FONT, T } from "../theme";
import { pop } from "../anim";

/** Full-frame background: light | dark, with an optional near-invisible dot grid. */
export const SceneBg: React.FC<{
  dark?: boolean;
  grid?: boolean;
  children?: React.ReactNode;
}> = ({ dark = false, grid = true, children }) => (
  <AbsoluteFill
    style={{
      background: dark
        ? `radial-gradient(1200px 800px at 50% 30%, #232323, ${C.inverseDeep})`
        : C.paper,
    }}
  >
    {grid && (
      <AbsoluteFill
        style={{
          backgroundImage: `radial-gradient(${dark ? "rgba(255,255,255,.06)" : "rgba(12,12,12,.05)"} 1.5px, transparent 1.5px)`,
          backgroundSize: "44px 44px",
        }}
      />
    )}
    {children}
  </AbsoluteFill>
);

/** The Holo wordmark + beta chip, matching the app header. */
export const Wordmark: React.FC<{ size?: number; dark?: boolean; sub?: string }> = ({
  size = 64,
  dark = false,
  sub,
}) => (
  <div style={{ display: "flex", alignItems: "center", gap: size * 0.28 }}>
    <span
      style={{
        fontFamily: FONT,
        fontSize: size,
        fontWeight: 700,
        letterSpacing: "-0.03em",
        color: dark ? C.white : C.ink,
      }}
    >
      Holo
    </span>
    <span
      style={{
        fontFamily: FONT,
        fontSize: size * 0.30,
        fontWeight: 650,
        letterSpacing: "0.1em",
        textTransform: "uppercase",
        color: dark ? C.inverseDeep : C.white,
        background: dark ? C.white : C.ink,
        padding: `${size * 0.10}px ${size * 0.22}px`,
        borderRadius: 999,
      }}
    >
      {sub ?? "PLVA"}
    </span>
  </div>
);

/** Word-staggered kinetic headline. */
export const KineticText: React.FC<{
  text: string;
  style?: React.CSSProperties;
  delay?: number;
  per?: number;
  as?: React.CSSProperties;
}> = ({ text, style, delay = 0, per = 3 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const words = text.split(" ");
  return (
    <div style={{ ...T.h1, color: C.ink, ...style }}>
      {words.map((w, i) => {
        const p = pop(frame, fps, delay + i * per);
        return (
          <span
            key={i}
            style={{
              display: "inline-block",
              opacity: p,
              transform: `translateY(${(1 - p) * 42}px)`,
              marginRight: "0.26em",
            }}
          >
            {w}
          </span>
        );
      })}
    </div>
  );
};

/** Small uppercase section label with a leading tick. */
export const Eyebrow: React.FC<{ text: string; color?: string; delay?: number; dark?: boolean }> = ({
  text,
  color,
  delay = 0,
  dark = false,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = pop(frame, fps, delay);
  const c = color ?? (dark ? C.whiteDim : C.gray);
  return (
    <div
      style={{
        ...T.label,
        color: c,
        display: "flex",
        alignItems: "center",
        gap: 16,
        opacity: p,
        transform: `translateY(${(1 - p) * 20}px)`,
      }}
    >
      <span style={{ width: 34, height: 3, background: c, display: "inline-block" }} />
      {text}
    </div>
  );
};

/** macOS-ish window chrome to host mock screens or placeholder slots. */
export const WindowFrame: React.FC<{
  title?: string;
  url?: string;
  width: number;
  height: number;
  children: React.ReactNode;
  style?: React.CSSProperties;
  dark?: boolean;
}> = ({ title, url, width, height, children, style, dark = false }) => (
  <div
    style={{
      width,
      height,
      borderRadius: 18,
      background: dark ? "#242424" : C.white,
      boxShadow: "0 40px 90px rgba(12,12,12,.22), 0 4px 18px rgba(12,12,12,.10)",
      border: `1px solid ${dark ? "rgba(255,255,255,.12)" : C.borderSoft}`,
      overflow: "hidden",
      display: "flex",
      flexDirection: "column",
      ...style,
    }}
  >
    <div
      style={{
        height: 52,
        flexShrink: 0,
        display: "flex",
        alignItems: "center",
        padding: "0 22px",
        gap: 10,
        borderBottom: `1px solid ${dark ? "rgba(255,255,255,.1)" : C.borderFade}`,
        background: dark ? "#2c2c2c" : C.paperAlt,
      }}
    >
      {["#ff5f57", "#febc2e", "#28c840"].map((c) => (
        <span key={c} style={{ width: 14, height: 14, borderRadius: 999, background: c }} />
      ))}
      {url ? (
        <span
          style={{
            marginLeft: 16,
            fontFamily: FONT,
            fontSize: 19,
            fontWeight: 500,
            color: dark ? C.whiteDim : C.gray,
            background: dark ? "rgba(255,255,255,.08)" : C.white,
            border: `1px solid ${dark ? "transparent" : C.borderFade}`,
            borderRadius: 999,
            padding: "5px 18px",
          }}
        >
          {url}
        </span>
      ) : (
        <span style={{ marginLeft: 12, fontFamily: FONT, fontSize: 19, fontWeight: 550, color: dark ? C.whiteDim : C.gray }}>
          {title}
        </span>
      )}
    </div>
    <div style={{ flex: 1, position: "relative", minHeight: 0 }}>{children}</div>
  </div>
);

/** Animated scanning line + expanding red boxes, used for "detection" moments. */
export const ScanLine: React.FC<{ progress: number; color?: string }> = ({ progress, color = C.red }) => (
  <div
    style={{
      position: "absolute",
      left: 0,
      right: 0,
      top: `${progress * 100}%`,
      height: 3,
      background: color,
      boxShadow: `0 0 24px 4px ${color}66`,
      opacity: progress <= 0 || progress >= 1 ? 0 : 1,
    }}
  />
);

/** Bottom-corner watermark shown through most scenes. */
export const CornerBadge: React.FC<{ dark?: boolean; text?: string }> = ({ dark = false, text = "Holo · PLVA — The Computer Use Hackathon" }) => {
  const frame = useCurrentFrame();
  const o = interpolate(frame, [0, 20], [0, 1], { extrapolateRight: "clamp" });
  return (
    <div
      style={{
        position: "absolute",
        bottom: 36,
        right: 48,
        fontFamily: FONT,
        fontSize: 20,
        fontWeight: 550,
        color: dark ? C.whiteFaint : C.grayLight,
        opacity: o,
      }}
    >
      {text}
    </div>
  );
};
