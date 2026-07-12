import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { C, T, FONT, MONO } from "../theme";
import { pop, ramp } from "../anim";
import { SceneBg } from "../components/Blocks";

const Node: React.FC<{
  title: string;
  sub: string;
  accent?: string;
  p: number;
  fromX: number;
  highlight?: boolean;
}> = ({ title, sub, accent = C.ink, p, fromX, highlight = false }) => (
  <div
    style={{
      opacity: p,
      transform: `translateX(${(1 - p) * fromX}px)`,
      background: highlight ? C.inverse : C.white,
      border: `2px solid ${highlight ? C.inverse : C.borderSoft}`,
      borderRadius: 18,
      padding: "30px 44px",
      textAlign: "center",
      boxShadow: highlight ? "0 24px 70px rgba(12,12,12,.3)" : "0 14px 40px rgba(12,12,12,.08)",
      minWidth: 330,
    }}
  >
    <div style={{ fontFamily: FONT, fontSize: 32, fontWeight: 700, color: highlight ? C.white : C.ink, letterSpacing: "-0.02em" }}>
      {title}
    </div>
    <div style={{ fontFamily: MONO, fontSize: 20, color: highlight ? C.whiteDim : accent, marginTop: 8, fontWeight: 600 }}>
      {sub}
    </div>
  </div>
);

/**
 * [24–33s] Title reveal: Holo wordmark + the three-node architecture lockup.
 * The runtime and Holo3 slide in from their story sides and handshake at PLVA.
 */
export const S04Title: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const wm = pop(frame, fps, 4);
  const cursorOn = Math.floor(frame / 12) % 2 === 0;
  const sub = pop(frame, fps, 22);
  const nodeL = pop(frame, fps, 70);
  const nodeC = pop(frame, fps, 92);
  const nodeR = pop(frame, fps, 70);
  const wires = ramp(frame, 108, 22);
  const legs = pop(frame, fps, 140);

  return (
    <SceneBg>
      <div style={{ position: "absolute", top: 200, left: 0, right: 0, textAlign: "center" }}>
        <div style={{ display: "inline-flex", alignItems: "baseline", gap: 20, opacity: wm, transform: `translateY(${(1 - wm) * 40}px)` }}>
          <span style={{ fontFamily: FONT, fontSize: 150, fontWeight: 700, letterSpacing: "-0.04em", color: C.ink }}>
            Holo
          </span>
          <span style={{ width: 26, height: 100, background: C.green, opacity: cursorOn ? 1 : 0.15, borderRadius: 4, display: "inline-block" }} />
        </div>
        <div style={{ ...T.h3, color: C.ink, marginTop: 4, opacity: sub, transform: `translateY(${(1 - sub) * 24}px)` }}>
          Private computer use
        </div>
        <div style={{ ...T.body, color: C.gray, marginTop: 14, opacity: pop(frame, fps, 40) }}>
          PLVA — a local, fail-closed privacy proxy between the agent runtime and the model
        </div>
      </div>

      {/* architecture lockup */}
      <div
        style={{
          position: "absolute",
          top: 640,
          left: 0,
          right: 0,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          gap: 100,
        }}
      >
        <Node title="AGENT RUNTIME" sub="closed binary" p={nodeL} fromX={-420} />
        <Node title="PLVA" sub="127.0.0.1:18081" p={nodeC} fromX={0} highlight accent={C.green} />
        <Node title="Holo3-35B" sub="H COMPANY" p={nodeR} fromX={420} />
      </div>

      {/* wires + leg labels */}
      <svg width="1920" height="1080" style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
        {/* left wire */}
        <line x1={960 - 245} y1={710} x2={960 - 245 - 130 * wires} y2={710} stroke={C.ink} strokeWidth="3.5" />
        <line x1={960 + 245} y1={710} x2={960 + 245 + 130 * wires} y2={710} stroke={C.ink} strokeWidth="3.5" />
      </svg>
      <div style={{ position: "absolute", top: 594, left: 0, right: 0, display: "flex", justifyContent: "center", gap: 320, opacity: legs }}>
        <span style={{ fontFamily: MONO, fontSize: 23, fontWeight: 700, color: C.green }}>→ redact</span>
        <span style={{ fontFamily: MONO, fontSize: 23, fontWeight: 700, color: C.green }}>← resolve</span>
      </div>

      <div
        style={{
          position: "absolute",
          bottom: 90,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: FONT,
          fontSize: 26,
          fontWeight: 550,
          color: C.gray,
          opacity: pop(frame, fps, 160),
        }}
      >
        intercepts <span style={{ color: C.ink, fontWeight: 700 }}>both directions</span> of model traffic
      </div>
    </SceneBg>
  );
};
