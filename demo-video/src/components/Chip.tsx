import React from "react";
import { useCurrentFrame, useVideoConfig } from "remotion";
import { C, MONO } from "../theme";
import { bounce } from "../anim";

/**
 * A painted placeholder chip, e.g. «EMAIL_1_a3f9»: the opaque mask PLVA paints
 * over real pixels. Dark #181818 surface (the site's back-inverse), mono label,
 * and the site's 7px status-dot language for the safety level.
 */
export const Chip: React.FC<{
  token: string;
  level?: "hide" | "approval" | "blocked";
  size?: number;
  delay?: number;
  animated?: boolean;
  style?: React.CSSProperties;
}> = ({ token, level = "hide", size = 26, delay = 0, animated = true, style }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = animated ? bounce(frame, fps, delay) : 1;
  const dot = level === "blocked" ? C.red : level === "approval" ? C.amber : C.green;
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: size * 0.32,
        fontFamily: MONO,
        fontSize: size,
        fontWeight: 400,
        color: C.white,
        background: C.inverse,
        borderRadius: Math.round(size * 0.24),
        padding: `${Math.round(size * 0.2)}px ${Math.round(size * 0.45)}px`,
        whiteSpace: "nowrap",
        opacity: Math.min(1, p * 1.2),
        transform: `scale(${p})`,
        ...style,
      }}
    >
      <span
        style={{
          width: size * 0.28,
          height: size * 0.28,
          borderRadius: "50%",
          background: dot,
          display: "inline-block",
          flexShrink: 0,
        }}
      />
      {"«"}
      {token}
      {"»"}
    </span>
  );
};
