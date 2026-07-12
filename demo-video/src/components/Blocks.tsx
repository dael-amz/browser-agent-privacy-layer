import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { C, FONT, SHADOW, T } from "../theme";
import { pop } from "../anim";

/** Full-frame background. The site is plain white (light scheme, no texture). */
export const SceneBg: React.FC<{
  dark?: boolean;
  grid?: boolean;
  children?: React.ReactNode;
}> = ({ dark = false, children }) => (
  <AbsoluteFill style={{ background: dark ? C.inverse : C.paper }}>{children}</AbsoluteFill>
);

/** The Holo brand lockup, copied from the site's .brand: wordmark 500 + outline beta chip. */
export const Wordmark: React.FC<{ size?: number; dark?: boolean; sub?: string }> = ({
  size = 64,
  dark = false,
  sub,
}) => (
  <div style={{ display: "flex", alignItems: "center", gap: size * 0.24 }}>
    <span
      style={{
        fontFamily: FONT,
        fontSize: size,
        fontWeight: 500,
        letterSpacing: "-0.011em",
        color: dark ? C.white : C.ink,
      }}
    >
      Holo
    </span>
    <span
      style={{
        fontFamily: FONT,
        fontSize: size * 0.26,
        fontWeight: 400,
        color: dark ? C.whiteDim : C.gray,
        border: `1.5px solid ${dark ? "rgba(255,255,255,.25)" : C.borderSoft}`,
        padding: `${size * 0.05}px ${size * 0.18}px`,
        borderRadius: 999,
      }}
    >
      {sub ?? "PLVA"}
    </span>
  </div>
);

/** Word-staggered headline with the site's stagger-line reveal (rise + blur clear). */
export const KineticText: React.FC<{
  text: string;
  style?: React.CSSProperties;
  delay?: number;
  per?: number;
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
              transform: `translateY(${(1 - p) * 24}px)`,
              filter: `blur(${(1 - p) * 3}px)`,
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

/** Section label copied from the landing page's .eyebrow: outline pill + 7px status dot. */
export const Eyebrow: React.FC<{ text: string; color?: string; delay?: number; dark?: boolean }> = ({
  text,
  color = C.green,
  delay = 0,
  dark = false,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = pop(frame, fps, delay);
  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 13,
        padding: "11px 26px",
        border: `1.5px solid ${dark ? "rgba(255,255,255,.25)" : C.borderSoft}`,
        borderRadius: 999,
        fontFamily: FONT,
        fontSize: 24,
        fontWeight: 400,
        color: dark ? C.whiteDim : C.gray,
        opacity: p,
        transform: `translateY(${(1 - p) * 16}px)`,
        filter: `blur(${(1 - p) * 3}px)`,
      }}
    >
      <span style={{ width: 13, height: 13, borderRadius: "50%", background: color, display: "inline-block" }} />
      {text}
    </div>
  );
};

/** macOS-ish window chrome, surfaced like the site's cards (.card / .composer). */
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
      borderRadius: 16,
      background: dark ? "#242424" : C.white,
      boxShadow: SHADOW.float,
      border: `1px solid ${dark ? "rgba(255,255,255,.12)" : C.borderFade}`,
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
            fontWeight: 400,
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
        <span style={{ marginLeft: 12, fontFamily: FONT, fontSize: 19, fontWeight: 500, color: dark ? C.whiteDim : C.gray }}>
          {title}
        </span>
      )}
    </div>
    <div style={{ flex: 1, position: "relative", minHeight: 0 }}>{children}</div>
  </div>
);

/** Animated scanning line, used for "detection" moments. */
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
export const CornerBadge: React.FC<{ dark?: boolean; text?: string }> = ({ dark = false, text = "Holo · PLVA · The Computer Use Hackathon" }) => {
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
        fontWeight: 400,
        color: dark ? C.whiteFaint : C.grayLight,
        opacity: o,
      }}
    >
      {text}
    </div>
  );
};
