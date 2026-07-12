import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { C, T, FONT, MONO } from "../theme";
import { pop, ramp } from "../anim";
import { SceneBg, Eyebrow } from "../components/Blocks";
import { Shot } from "../components/Shot";
import { Odometer, Stamp } from "../components/Gadgets";

const CLASSES = ["PASSWORD", "EMAIL", "CARD_NUMBER", "API_KEY"];

/** Cloud provider node. */
const CloudNode: React.FC<{ pulse: number }> = ({ pulse }) => (
  <div style={{ textAlign: "center" }}>
    <svg width="300" height="190" viewBox="0 0 300 190">
      <path
        d="M75 150 a45 45 0 0 1 -1 -90 a62 62 0 0 1 118 -22 a52 52 0 0 1 68 50 a42 42 0 0 1 -14 62 z"
        fill={C.inverse}
        stroke="none"
        transform={`scale(${1 + pulse * 0.03})`}
        style={{ transformOrigin: "150px 95px" }}
      />
      <text x="150" y="105" textAnchor="middle" fill={C.white} fontFamily={FONT} fontSize="24" fontWeight="650">
        MODEL PROVIDER
      </text>
    </svg>
    <div style={{ fontFamily: FONT, fontSize: 24, color: C.gray, marginTop: 2 }}>sees everything you see</div>
  </div>
);

/**
 * [5–14s] The problem, precisely: the perceive→act loop ships raw frames
 * to the provider every step. Flying thumbnails + odometer + class chips.
 */
export const S02Problem: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const wireDraw = ramp(frame, 10, 34);
  const headIn = pop(frame, fps, 2);

  // a thumbnail departs every 24 frames after f40
  const PERIOD = 24;
  const sent = Math.max(0, Math.floor((frame - 40) / PERIOD));
  const flights = [0, 1, 2].map((k) => {
    const born = 40 + (sent - k) * PERIOD;
    const t = (frame - born) / PERIOD;
    return t >= 0 && t < 1 ? t : null;
  });

  const pulse = Math.max(0, Math.sin((frame / PERIOD) * Math.PI * 2));

  // wire path geometry (screen coords)
  const x0 = 800;
  const x1 = 1400;
  const yTop = 560;

  return (
    <SceneBg>
      <div style={{ position: "absolute", top: 108, left: 160, right: 160, zIndex: 5 }}>
        <Eyebrow text="The problem" color={C.red} />
        <div style={{ ...T.h1, fontSize: 72, marginTop: 26, opacity: headIn, transform: `translateY(${(1 - headIn) * 30}px)` }}>
          Agents see by screenshotting your <span style={{ color: C.red }}>entire screen.</span>
        </div>
        <div style={{ ...T.h3, color: C.gray, marginTop: 16, opacity: pop(frame, fps, 26) }}>
          Every step. Raw pixels. Straight to the model provider.
        </div>
      </div>

      {/* desktop node (real capture goes here) */}
      <div style={{ position: "absolute", left: 160, top: 480, width: 640 }}>
        <Shot id="SHOT-01" style={{ width: 640, height: 400 }} delay={8} />
        <div style={{ fontFamily: FONT, fontSize: 24, fontWeight: 500, color: C.ink, marginTop: 16, textAlign: "center" }}>
          your desktop
        </div>
      </div>

      {/* wires */}
      <svg width="1920" height="1080" style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
        {/* request wire (top) — endpoints anchored to the card edge and cloud */}
        <path
          d={`M ${x0} ${yTop} C ${x0 + 240} ${yTop - 70}, ${x1 - 240} ${yTop - 70}, ${x1 + 58} ${yTop}`}
          stroke={C.ink}
          strokeWidth="3.5"
          fill="none"
          strokeDasharray="780"
          strokeDashoffset={780 * (1 - wireDraw)}
        />
        <circle cx={x0} cy={yTop} r="7" fill={C.ink} opacity={wireDraw} />
        <circle cx={x1 + 58} cy={yTop} r="7" fill={C.ink} opacity={wireDraw >= 1 ? 1 : 0} />
        {/* action wire (bottom, returning) */}
        <path
          d={`M ${x1 + 58} ${yTop + 160} C ${x1 - 240} ${yTop + 230}, ${x0 + 240} ${yTop + 230}, ${x0} ${yTop + 160}`}
          stroke={C.grayLight}
          strokeWidth="3"
          fill="none"
          strokeDasharray="6 12"
          strokeDashoffset={-frame * 1.2}
        />
        <text x={(x0 + x1) / 2} y={yTop + 260} textAnchor="middle" fontFamily={FONT} fontSize="22" fill={C.gray}>
          ← actions (click, type, scroll)
        </text>
        {/* flying raw-frame thumbnails */}
        {flights.map((t, i) =>
          t === null ? null : (
            <g key={i} transform={`translate(${x0 + (x1 + 58 - x0) * t}, ${yTop - 52 * Math.sin(t * Math.PI)})`}>
              <rect x="-34" y="-24" width="68" height="48" rx="6" fill={C.white} stroke={C.red} strokeWidth="3" />
              <rect x="-24" y="-14" width="48" height="6" rx="2" fill={C.redSoft} stroke="none" />
              <rect x="-24" y="-3" width="34" height="6" rx="2" fill={C.red} stroke="none" />
              <rect x="-24" y="8" width="42" height="6" rx="2" fill={C.redSoft} stroke="none" />
            </g>
          )
        )}
      </svg>

      {/* provider */}
      <div style={{ position: "absolute", left: x1 - 10, top: yTop - 120, opacity: ramp(frame, 24, 14) }}>
        <CloudNode pulse={pulse} />
      </div>

      {/* counter + class chips */}
      <div style={{ position: "absolute", left: 160, bottom: 84, display: "flex", alignItems: "center", gap: 60 }}>
        <Odometer label="frames sent:" value={sent} />
        <div style={{ display: "flex", gap: 18 }}>
          {CLASSES.map((c, i) => (
            <Stamp key={c} delay={120 + i * 12} rotate={i % 2 === 0 ? -2 : 1.5}>
              <span
                style={{
                  fontFamily: FONT,
                  fontSize: 24,
                  fontWeight: 500,
                  color: C.red,
                  border: `1.5px solid currentColor`,
                  borderRadius: 999,
                  padding: "8px 22px",
                  display: "inline-block",
                }}
              >
                {c}
              </span>
            </Stamp>
          ))}
        </div>
      </div>
    </SceneBg>
  );
};
