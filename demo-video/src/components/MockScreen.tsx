import React from "react";
import { useCurrentFrame, useVideoConfig } from "remotion";
import { C, FONT, MONO } from "../theme";
import { bounce, pop } from "../anim";

export type PiiMode = "raw" | "detected" | "redacted";

/** One PII value that can be shown raw, boxed in red, or painted over with a chip. */
const Pii: React.FC<{
  value: string;
  token: string;
  mode: PiiMode;
  level?: "hide" | "approval" | "blocked";
  delay?: number;
  size?: number;
}> = ({ value, token, mode, level = "hide", delay = 0, size = 23 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const edge = level === "blocked" ? C.red : level === "approval" ? C.amber : C.green;

  if (mode === "raw") {
    return <span style={{ color: C.ink, fontWeight: 500 }}>{value}</span>;
  }
  if (mode === "detected") {
    const p = pop(frame, fps, delay);
    return (
      <span
        style={{
          position: "relative",
          color: C.ink,
          fontWeight: 500,
          boxShadow: `0 0 0 ${3 * p}px ${C.red}`,
          background: `rgba(209,52,56,${0.10 * p})`,
          borderRadius: 4,
        }}
      >
        {value}
      </span>
    );
  }
  // redacted: chip painted over
  const p = bounce(frame, fps, delay);
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: size * 0.3,
        fontFamily: MONO,
        fontSize: size * 0.92,
        fontWeight: 400,
        color: C.white,
        background: C.inverse,
        borderRadius: 6,
        padding: "3px 12px",
        transform: `scale(${Math.max(p, 0.001)})`,
        whiteSpace: "nowrap",
      }}
    >
      <span style={{ width: size * 0.27, height: size * 0.27, borderRadius: "50%", background: edge, display: "inline-block" }} />
      {level === "blocked" ? "▮▮▮▮▮▮▮▮" : `«${token}»`}
    </span>
  );
};

const Row: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 18, minHeight: 58 }}>
    <span style={{ fontFamily: FONT, fontSize: 22, fontWeight: 500, color: C.gray, width: 190, flexShrink: 0 }}>
      {label}
    </span>
    <span
      style={{
        fontFamily: FONT,
        fontSize: 23,
        flex: 1,
        background: C.paperAlt,
        border: `1px solid ${C.borderFade}`,
        borderRadius: 10,
        padding: "12px 18px",
        display: "flex",
        alignItems: "center",
      }}
    >
      {children}
    </span>
  </div>
);

/**
 * A fake "pay an invoice" page dense with personal data.
 * mode drives what the model would see: raw pixels → detected boxes → painted chips.
 */
export const MockScreen: React.FC<{ mode: PiiMode; staggerFrom?: number }> = ({
  mode,
  staggerFrom = 0,
}) => {
  const d = (i: number) => staggerFrom + i * 4;
  return (
    <div style={{ padding: "34px 44px", display: "flex", flexDirection: "column", gap: 10, background: C.white, height: "100%" }}>
      <div style={{ fontFamily: FONT, fontSize: 30, fontWeight: 500, letterSpacing: "-0.02em", color: C.ink, marginBottom: 6 }}>
        Checkout · Aurora Utilities
      </div>
      <Row label="Full name">
        <Pii mode={mode} value="Camille Fontaine" token="NAME_1_a3f9" delay={d(0)} />
      </Row>
      <Row label="Email">
        <Pii mode={mode} value="camille.fontaine@proton.me" token="EMAIL_1_a3f9" delay={d(1)} />
      </Row>
      <Row label="Phone">
        <Pii mode={mode} value="+33 6 48 21 07 55" token="PHONE_1_a3f9" delay={d(2)} />
      </Row>
      <Row label="Password">
        <Pii mode={mode} value="hunter2!x" token="PASSWORD_1" level="blocked" delay={d(3)} />
      </Row>
      <Row label="Card number">
        <Pii mode={mode} value="4929 1188 3407 2216" token="CARD_1" level="blocked" delay={d(4)} />
      </Row>
      <Row label="API key">
        <Pii mode={mode} value="hk-live-9f3ab27c41d8" token="API_KEY_1_a3f9" level="approval" delay={d(5)} />
      </Row>
    </div>
  );
};
