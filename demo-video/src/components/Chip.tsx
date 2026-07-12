import React from "react";
import { useCurrentFrame, useVideoConfig } from "remotion";
import { C, MONO } from "../theme";
import { bounce } from "../anim";

/** A painted placeholder chip, e.g. «EMAIL_1_a3f9» — mirrors the chips PLVA paints on frames. */
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
  const edge =
    level === "blocked" ? C.red : level === "approval" ? C.amber : C.green;
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        fontFamily: MONO,
        fontSize: size,
        fontWeight: 600,
        color: C.white,
        background: C.inverse,
        borderRadius: Math.round(size * 0.32),
        padding: `${Math.round(size * 0.22)}px ${Math.round(size * 0.5)}px`,
        boxShadow: `inset 0 0 0 2px ${edge}`,
        whiteSpace: "nowrap",
        opacity: Math.min(1, p * 1.2),
        transform: `scale(${p})`,
        ...style,
      }}
    >
      {"«"}
      {token}
      {"»"}
    </span>
  );
};
