import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate, Easing } from "remotion";
import { C, T, FONT, MONO } from "../theme";
import { pop, ramp, rampOut } from "../anim";
import { SceneBg, Eyebrow } from "../components/Blocks";
import { DecodeText } from "../components/Gadgets";
import { Chip } from "../components/Chip";

const TOKEN = "EMAIL_1_a3f9";
const REAL = "alex.rivera@example.com";

/**
 * [47–60s] Centerpiece II — the response leg. Holo3 answers with a token;
 * PLVA resolves it locally, in transit; the REAL value gets typed.
 * Ends on the thesis: "The model works with what it can't see."
 */
export const S06ResponseLeg: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Phase A (0–80): response card travels right → left into the proxy
  const travel = interpolate(frame, [6, 72], [1650, 660], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic),
  });
  const cardVisible = rampOut(frame, 88, 12);

  // Phase B (80–190): magnified resolve — decode morph
  const zoomIn = pop(frame, fps, 84);
  const decode = ramp(frame, 108, 52);
  const settled = decode >= 0.999;
  const zoomOut = rampOut(frame, 196, 12);

  // Phase C (196–280): the real value is typed into the field
  const typedChars = Math.max(0, Math.min(REAL.length, Math.floor((frame - 210) / 2)));
  const fieldIn = pop(frame, fps, 200);
  const caretOn = Math.floor(frame / 8) % 2 === 0;

  // Phase D (285–390): what-saw vs what-typed + thesis
  const splitIn = pop(frame, fps, 288);
  const thesisWords = ["The", "model", "works", "with", "what", "it", "can't", "see."];

  return (
    <SceneBg>
      <div style={{ position: "absolute", top: 96, left: 160, right: 160, zIndex: 10 }}>
        <Eyebrow text="Response leg — the twist" color={C.green} />
      </div>

      {/* endpoint nodes */}
      <div style={{ position: "absolute", top: 250, left: 150, opacity: 0.9 * cardVisible }}>
        <div style={{ background: C.white, border: `2px solid ${C.borderSoft}`, borderRadius: 16, padding: "22px 34px", fontFamily: FONT, fontSize: 26, fontWeight: 700, color: C.ink }}>
          AGENT RUNTIME
          <div style={{ fontFamily: FONT, fontSize: 18, fontWeight: 500, color: C.gray, marginTop: 4 }}>executes the action</div>
        </div>
      </div>
      <div style={{ position: "absolute", top: 250, right: 150, opacity: 0.9 * cardVisible }}>
        <div style={{ background: C.inverse, borderRadius: 16, padding: "22px 34px", fontFamily: FONT, fontSize: 26, fontWeight: 700, color: C.white, textAlign: "center" }}>
          Holo3-35B
          <div style={{ fontFamily: MONO, fontSize: 17, fontWeight: 500, color: C.whiteDim, marginTop: 4 }}>answers with a token</div>
        </div>
      </div>

      {/* wire */}
      <div style={{ position: "absolute", top: 306, left: 420, right: 420, borderTop: `3px solid ${C.borderSoft}`, opacity: cardVisible }} />

      {/* PLVA node (center) */}
      <div
        style={{
          position: "absolute",
          top: 262,
          left: 960,
          transform: "translateX(-50%)",
          background: C.green,
          borderRadius: 16,
          padding: "24px 40px",
          fontFamily: FONT,
          fontSize: 28,
          fontWeight: 700,
          color: C.white,
          boxShadow: `0 16px 50px ${C.green}44`,
          opacity: cardVisible,
          zIndex: 3,
        }}
      >
        PLVA
      </div>

      {/* the travelling response card */}
      <div
        style={{
          position: "absolute",
          top: 380,
          left: travel,
          transform: "translateX(-50%)",
          opacity: ramp(frame, 6, 8) * cardVisible,
          zIndex: 4,
        }}
      >
        {/* speed streaks */}
        <svg width="220" height="20" style={{ position: "absolute", right: -230, top: 34, opacity: frame < 70 ? 0.7 : 0 }}>
          {[0, 1, 2].map((i) => (
            <line key={i} x1={i * 40} y1={4 + i * 6} x2={i * 40 + 130} y2={4 + i * 6} stroke={C.grayLight} strokeWidth="3" strokeLinecap="round" />
          ))}
        </svg>
        <div
          style={{
            background: C.white,
            border: `2px solid ${C.borderSoft}`,
            borderRadius: 14,
            padding: "18px 28px",
            fontFamily: MONO,
            fontSize: 27,
            color: C.ink,
            boxShadow: "0 20px 50px rgba(12,12,12,.16)",
            whiteSpace: "nowrap",
          }}
        >
          <span style={{ color: C.gray }}>action:</span> write(text=
          <span style={{ color: C.green, fontWeight: 700 }}>"«{TOKEN}»"</span>)
        </div>
      </div>

      {/* Phase B: magnified resolution */}
      {frame >= 80 && frame < 212 && (
        <div
          style={{
            position: "absolute",
            top: 420,
            left: 0,
            right: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            opacity: Math.min(zoomIn, zoomOut),
            transform: `scale(${0.86 + 0.14 * zoomIn})`,
          }}
        >
          <div
            style={{
              borderRadius: 30,
              border: `5px solid ${settled ? C.green : C.ink}`,
              background: C.white,
              padding: "58px 90px",
              boxShadow: "0 40px 110px rgba(12,12,12,.22)",
              textAlign: "center",
            }}
          >
            <div style={{ ...T.label, color: C.gray, marginBottom: 26 }}>inside the proxy · resolving against the vault</div>
            <div style={{ fontSize: 58, fontWeight: 700 }}>
              <DecodeText from={`«${TOKEN}»`} to={REAL} progress={decode} style={{ color: settled ? C.green : C.ink }} />
            </div>
            <div style={{ display: "flex", gap: 40, justifyContent: "center", marginTop: 34, fontFamily: MONO, fontSize: 21, color: C.gray }}>
              <span>
                vault: <span style={{ color: C.ink }}>{TOKEN}</span> → <span style={{ color: C.green }}>████ · match</span>
              </span>
              <span style={{ color: settled ? C.green : C.grayLight, fontWeight: 700 }}>
                {settled ? "✓ resolved locally · in transit" : "…"}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Phase C: typed into the real field */}
      {frame >= 196 && frame < 292 && (
        <div
          style={{
            position: "absolute",
            top: 470,
            left: 0,
            right: 0,
            display: "flex",
            justifyContent: "center",
            opacity: Math.min(fieldIn, rampOut(frame, 282, 10)),
            transform: `translateY(${(1 - fieldIn) * 30}px)`,
          }}
        >
          <div style={{ width: 900 }}>
            <div style={{ fontFamily: FONT, fontSize: 24, fontWeight: 600, color: C.gray, marginBottom: 12 }}>Email</div>
            <div
              style={{
                background: C.white,
                border: `3px solid ${C.green}`,
                borderRadius: 14,
                padding: "26px 32px",
                fontFamily: MONO,
                fontSize: 36,
                color: C.ink,
                boxShadow: `0 24px 70px ${C.green}22`,
              }}
            >
              {REAL.slice(0, typedChars)}
              {caretOn && <span style={{ borderLeft: `4px solid ${C.ink}`, marginLeft: 3 }} />}
            </div>
            <div style={{ fontFamily: FONT, fontSize: 24, color: C.gray, marginTop: 18, textAlign: "center" }}>
              typed by <span style={{ color: C.ink, fontWeight: 650 }}>your machine</span> — not by the model
            </div>
          </div>
        </div>
      )}

      {/* Phase D: the two truths + thesis */}
      {frame >= 285 && (
        <>
          <div
            style={{
              position: "absolute",
              top: 400,
              left: 0,
              right: 0,
              display: "flex",
              justifyContent: "center",
              gap: 4,
              opacity: splitIn,
            }}
          >
            {[
              { label: "WHAT HOLO3 SAW", content: <Chip token={TOKEN} size={34} animated={false} />, edge: C.ink },
              {
                label: "WHAT GOT TYPED",
                content: <span style={{ fontFamily: MONO, fontSize: 34, fontWeight: 600, color: C.green }}>{REAL}</span>,
                edge: C.green,
              },
            ].map((side, i) => (
              <div
                key={i}
                style={{
                  width: 620,
                  textAlign: "center",
                  padding: "44px 30px",
                  borderTop: `6px solid ${side.edge}`,
                  background: i === 0 ? C.paperAlt : C.greenSoft,
                  borderRadius: i === 0 ? "18px 0 0 18px" : "0 18px 18px 0",
                }}
              >
                <div style={{ ...T.label, color: C.gray, marginBottom: 24 }}>{side.label}</div>
                {side.content}
              </div>
            ))}
          </div>

          <div style={{ position: "absolute", top: 700, left: 0, right: 0, textAlign: "center" }}>
            <div style={{ ...T.hero, fontSize: 92 }}>
              {thesisWords.map((w, i) => {
                const p = pop(frame, fps, 310 + i * 6);
                const green = i >= 6;
                return (
                  <span
                    key={i}
                    style={{
                      display: "inline-block",
                      marginRight: "0.24em",
                      opacity: p,
                      transform: `translateY(${(1 - p) * 40}px)`,
                      color: green ? C.green : C.ink,
                    }}
                  >
                    {w}
                  </span>
                );
              })}
            </div>
          </div>
        </>
      )}
    </SceneBg>
  );
};
