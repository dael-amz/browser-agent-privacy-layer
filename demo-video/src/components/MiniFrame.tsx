import React from "react";
import { useCurrentFrame, useVideoConfig } from "remotion";
import { C, FONT, MONO } from "../theme";
import { bounce, pop, ramp } from "../anim";
import { PiiMode } from "./MockScreen";

const FIELDS = [
  { label: "Email", value: "alex.rivera@example.com", token: "EMAIL_1_a3f9", level: "hide" as const, mask: "" },
  { label: "Password", value: "hunter2!x", token: "PASSWORD_1", level: "blocked" as const, mask: "▮▮▮▮▮▮▮▮▮" },
  { label: "Card", value: "4929 1188 3407 2216", token: "CARD_1", level: "blocked" as const, mask: "▮▮▮▮ ▮▮▮▮ ▮▮▮▮ ▮▮▮▮" },
];

/**
 * A simplified screenshot card, readable at pipeline scale (~620px wide).
 * Three PII fields switch raw → detected → redacted.
 */
export const MiniFrame: React.FC<{
  mode: PiiMode;
  width?: number;
  staggerFrom?: number;
  scanProgress?: number; // 0..1 shows a green scan beam
  style?: React.CSSProperties;
}> = ({ mode, width = 620, staggerFrom = 0, scanProgress, style }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const h = width * 0.62;
  return (
    <div
      style={{
        width,
        height: h,
        background: C.white,
        borderRadius: 16,
        border: `1px solid ${C.borderFade}`,
        boxShadow: "0 16px 48px rgba(12,12,12,.06)",
        overflow: "hidden",
        position: "relative",
        ...style,
      }}
    >
      <div style={{ height: h * 0.115, background: C.paperAlt, borderBottom: `1px solid ${C.borderFade}`, display: "flex", alignItems: "center", gap: 7, padding: "0 16px" }}>
        {["#ff5f57", "#febc2e", "#28c840"].map((c) => (
          <span key={c} style={{ width: 10, height: 10, borderRadius: 999, background: c }} />
        ))}
        <span style={{ fontFamily: FONT, fontSize: width * 0.026, color: C.gray, marginLeft: 8, fontWeight: 500 }}>
          checkout · aurora utilities
        </span>
      </div>
      <div style={{ padding: `${width * 0.04}px ${width * 0.055}px`, display: "flex", flexDirection: "column", gap: width * 0.028 }}>
        {FIELDS.map((f, i) => {
          const delay = staggerFrom + i * 5;
          const fs = width * 0.037;
          let valueEl: React.ReactNode;
          if (mode === "raw") {
            valueEl = <span style={{ color: C.ink, fontWeight: 500 }}>{f.value}</span>;
          } else if (mode === "detected") {
            const p = pop(frame, fps, delay);
            valueEl = (
              <span
                style={{
                  color: C.ink,
                  fontWeight: 500,
                  boxShadow: `0 0 0 ${3 * p}px ${C.amber}`,
                  background: `rgba(185,126,15,${0.12 * p})`,
                  borderRadius: 4,
                }}
              >
                {f.value}
              </span>
            );
          } else {
            const p = bounce(frame, fps, delay);
            const dot = f.level === "blocked" ? C.red : C.green;
            valueEl = (
              <span
                style={{
                  fontFamily: MONO,
                  fontSize: fs * 0.9,
                  fontWeight: 400,
                  color: C.white,
                  background: C.inverse,
                  borderRadius: 6,
                  padding: `${fs * 0.14}px ${fs * 0.4}px`,
                  transform: `scale(${Math.max(p, 0.001)})`,
                  display: "inline-flex",
                  alignItems: "center",
                  gap: fs * 0.3,
                  whiteSpace: "nowrap",
                }}
              >
                <span style={{ width: fs * 0.26, height: fs * 0.26, borderRadius: "50%", background: dot, display: "inline-block" }} />
                {f.level === "blocked" ? f.mask : `«${f.token}»`}
              </span>
            );
          }
          return (
            <div key={f.label} style={{ display: "flex", alignItems: "center", gap: 12, fontSize: fs, fontFamily: FONT, minHeight: fs * 1.9 }}>
              <span style={{ color: C.gray, width: width * 0.19, flexShrink: 0, fontWeight: 500 }}>{f.label}</span>
              <span
                style={{
                  flex: 1,
                  background: C.paperAlt,
                  borderRadius: 8,
                  padding: `${fs * 0.28}px ${fs * 0.5}px`,
                  border: `1px solid ${C.borderFade}`,
                  display: "flex",
                  alignItems: "center",
                  minHeight: fs * 1.5,
                }}
              >
                {valueEl}
              </span>
            </div>
          );
        })}
      </div>
      {scanProgress !== undefined && scanProgress > 0 && scanProgress < 1 && (
        <div
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            top: `${scanProgress * 100}%`,
            height: 4,
            background: C.green,
            boxShadow: `0 0 26px 5px ${C.green}66`,
          }}
        />
      )}
      {/* subtle green tint behind the beam already swept */}
      {scanProgress !== undefined && scanProgress > 0 && (
        <div
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            top: 0,
            height: `${Math.min(scanProgress, 1) * 100}%`,
            background: `linear-gradient(180deg, transparent 70%, ${C.green}0d)`,
            pointerEvents: "none",
          }}
        />
      )}
    </div>
  );
};

export const miniFrameRamp = ramp;
