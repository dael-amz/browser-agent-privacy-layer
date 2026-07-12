import React from "react";
import { useCurrentFrame, useVideoConfig } from "remotion";
import { C, T, FONT, MONO } from "../theme";
import { pop, ramp } from "../anim";
import { SceneBg, Eyebrow } from "../components/Blocks";
import { Shot } from "../components/Shot";
import { Stamp } from "../components/Gadgets";

const CHECKS = ["No SDK", "No agent patch", "No provider cooperation"];

/**
 * [101–112s] Adoption: one config line. The beat-3 config card returns,
 * beside a real capture of the control panel.
 */
export const S10OneLine: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const head = pop(frame, fps, 2);
  const cardIn = pop(frame, fps, 20);
  const panelIn = pop(frame, fps, 40);

  return (
    <SceneBg>
      <div style={{ position: "absolute", top: 100, left: 160, right: 160 }}>
        <Eyebrow text="Usefulness" color={C.green} />
        <div style={{ ...T.h1, fontSize: 74, marginTop: 22, opacity: head, transform: `translateY(${(1 - head) * 30}px)` }}>
          Adopted with <span style={{ color: C.green }}>one config line.</span>
        </div>
      </div>

      {/* the config card, back from scene 3 */}
      <div
        style={{
          position: "absolute",
          left: 160,
          top: 360,
          width: 700,
          opacity: cardIn,
          transform: `translateX(${(1 - cardIn) * -80}px)`,
        }}
      >
        <div style={{ background: C.inverse, borderRadius: 16, padding: "30px 36px", boxShadow: "0 12px 32px rgba(12,12,12,.1)" }}>
          <div style={{ fontFamily: MONO, fontSize: 21, color: C.whiteDim, marginBottom: 18 }}>agent.config · 🔒 read-only, still</div>
          <div style={{ fontFamily: MONO, fontSize: 26, lineHeight: 2 }}>
            <div style={{ color: C.whiteFaint, textDecoration: "line-through" }}>runtime: hai-agent-runtime</div>
            <div style={{ color: C.whiteFaint, textDecoration: "line-through" }}>model: Hcompany/Holo3-35B-A3B</div>
            <div
              style={{
                color: C.white,
                background: "rgba(10,138,79,.28)",
                borderRadius: 8,
                padding: "4px 12px",
                margin: "6px -12px 0",
                boxShadow: `inset 0 0 0 2px ${C.green}`,
              }}
            >
              <span style={{ color: "#7ecfa7" }}>base_url</span>: "http://127.0.0.1:18081/v1"
            </div>
          </div>
        </div>

        {/* the checklist */}
        <div style={{ display: "flex", gap: 22, marginTop: 44, flexWrap: "wrap" }}>
          {CHECKS.map((c, i) => (
            <Stamp key={c} delay={80 + i * 16} rotate={i % 2 ? 2 : -2}>
              <span
                style={{
                  fontFamily: FONT,
                  fontSize: 26,
                  fontWeight: 500,
                  color: C.green,
                  border: `1.5px solid currentColor`,
                  borderRadius: 999,
                  padding: "12px 28px",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 12,
                }}
              >
                <span style={{ width: 12, height: 12, borderRadius: "50%", background: C.green, display: "inline-block" }} />
                {c}
              </span>
            </Stamp>
          ))}
        </div>
      </div>

      {/* the control panel capture */}
      <div
        style={{
          position: "absolute",
          right: 140,
          top: 330,
          width: 860,
          opacity: panelIn,
          transform: `translateX(${(1 - panelIn) * 80}px)`,
        }}
      >
        <Shot id="SHOT-04" style={{ width: 860, height: 540 }} />
        <div style={{ fontFamily: FONT, fontSize: 24, color: C.gray, textAlign: "center", marginTop: 20, opacity: ramp(frame, 70, 14) }}>
          local control panel · <span style={{ fontFamily: MONO, color: C.ink }}>127.0.0.1:18080</span> · flip privacy per class, live
        </div>
      </div>
    </SceneBg>
  );
};
