import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { C, T, FONT, MONO } from "../theme";
import { pop, ramp } from "../anim";
import { SceneBg, Wordmark } from "../components/Blocks";

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
      border: `1px solid ${highlight ? C.inverse : C.borderSoft}`,
      borderRadius: 16,
      padding: "30px 44px",
      textAlign: "center",
      boxShadow: highlight ? "0 12px 32px rgba(12,12,12,.1)" : "0 1px 2px rgba(12,12,12,.03)",
      minWidth: 330,
    }}
  >
    <div style={{ fontFamily: FONT, fontSize: 32, fontWeight: 500, color: highlight ? C.white : C.ink, letterSpacing: "-0.02em" }}>
      {title}
    </div>
    <div style={{ fontFamily: MONO, fontSize: 20, color: highlight ? C.whiteDim : accent, marginTop: 8, fontWeight: 500 }}>
      {sub}
    </div>
  </div>
);

/** Connector that lives IN the flex row, so its endpoints always touch its neighbors. */
const Wire: React.FC<{ label: string; draw: number; labelIn: number }> = ({ label, draw, labelIn }) => (
  <div style={{ width: 150, position: "relative", flexShrink: 0 }}>
    <div style={{ borderTop: `2px solid ${C.ink}`, transform: `scaleX(${draw})` }} />
    <span
      style={{
        position: "absolute",
        left: "50%",
        top: -46,
        transform: "translateX(-50%)",
        fontFamily: MONO,
        fontSize: 23,
        fontWeight: 500,
        color: C.green,
        whiteSpace: "nowrap",
        opacity: labelIn,
      }}
    >
      {label}
    </span>
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
  const sub = pop(frame, fps, 22);
  const nodeL = pop(frame, fps, 70);
  const nodeC = pop(frame, fps, 92);
  const nodeR = pop(frame, fps, 70);
  const wires = ramp(frame, 108, 22);
  const legs = pop(frame, fps, 140);

  return (
    <SceneBg>
      <div style={{ position: "absolute", top: 200, left: 0, right: 0, display: "flex", flexDirection: "column", alignItems: "center" }}>
        <div style={{ opacity: wm, transform: `translateY(${(1 - wm) * 24}px)`, filter: `blur(${(1 - wm) * 3}px)` }}>
          <Wordmark size={140} sub="Beta" />
        </div>
        <div style={{ ...T.h3, color: C.ink, marginTop: 20, opacity: sub, transform: `translateY(${(1 - sub) * 16}px)`, filter: `blur(${(1 - sub) * 3}px)` }}>
          Private computer use
        </div>
        <div style={{ ...T.body, color: C.gray, marginTop: 14, opacity: pop(frame, fps, 40), filter: `blur(${(1 - pop(frame, fps, 40)) * 3}px)` }}>
          PLVA: a local, fail-closed privacy proxy between the agent runtime and the model
        </div>
      </div>

      {/* architecture lockup — wires are flex children so they always meet the cards */}
      <div
        style={{
          position: "absolute",
          top: 640,
          left: 0,
          right: 0,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <Node title="AGENT RUNTIME" sub="closed binary" p={nodeL} fromX={-420} />
        <Wire label="→ redact" draw={wires} labelIn={legs} />
        <Node title="PLVA" sub="127.0.0.1:18081" p={nodeC} fromX={0} highlight accent={C.green} />
        <Wire label="← resolve" draw={wires} labelIn={legs} />
        <Node title="Holo3-35B" sub="H COMPANY" p={nodeR} fromX={420} />
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
          fontWeight: 500,
          color: C.gray,
          opacity: pop(frame, fps, 160),
        }}
      >
        intercepts <span style={{ color: C.ink, fontWeight: 500 }}>both directions</span> of model traffic
      </div>
    </SceneBg>
  );
};
