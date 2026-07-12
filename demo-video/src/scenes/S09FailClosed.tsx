import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate, Easing } from "remotion";
import { C, T, FONT, MONO } from "../theme";
import { pop, ramp, rampOut } from "../anim";
import { SceneBg } from "../components/Blocks";
import { Stamp } from "../components/Gadgets";
import { Chip } from "../components/Chip";

const POLICY: Array<{ level: string; color: string; classes: string[]; note: string }> = [
  { level: "hide_use", color: C.green, classes: ["EMAIL", "PHONE", "NAME", "ADDRESS"], note: "used freely, never seen" },
  { level: "approval", color: C.amber, classes: ["API_KEY", "AUTH_TOKEN"], note: "resolved only with approval" },
  { level: "blocked", color: C.red, classes: ["PASSWORD", "CARD_NUMBER", "CVC", "PRIVATE_KEY"], note: "never stored, never resolvable" },
];

/**
 * [88–101s] The invariants, stamped dry and precise. Nonce-forgery micro-drama,
 * then the per-class policy grid; PASSWORD → blocked takes the frame.
 */
export const S09FailClosed: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const gridPhase = frame >= 168;
  const zoomPhase = frame >= 300;
  const gridOut = rampOut(frame, 296, 10);
  const finale = pop(frame, fps, 308);

  return (
    <SceneBg grid={false}>
      {/* Phase 1: invariant stamps */}
      {!gridPhase && (
        <>
          <div style={{ position: "absolute", top: 160, left: 0, right: 0, textAlign: "center" }}>
            <Stamp delay={4} rotate={0}>
              <span style={{ ...T.h1, fontSize: 88, color: C.ink }}>
                FAIL-CLOSED. <span style={{ color: C.green }}>EVERYWHERE.</span>
              </span>
            </Stamp>
          </div>

          <div style={{ position: "absolute", top: 380, left: 0, right: 0, display: "flex", flexDirection: "column", alignItems: "center", gap: 40 }}>
            {/* invariant 1 */}
            <Stamp delay={40} rotate={-1.5}>
              <div style={{ display: "flex", alignItems: "center", gap: 24, background: C.white, border: `1px solid ${C.borderSoft}`, borderRadius: 16, padding: "24px 44px" }}>
                <svg width="42" height="42" viewBox="0 0 42 42">
                  <circle cx="21" cy="21" r="18" stroke={C.red} strokeWidth="3.5" fill="none" />
                  <line x1="8.3" y1="8.3" x2="33.7" y2="33.7" stroke={C.red} strokeWidth="3.5" strokeLinecap="round" />
                </svg>
                <span style={{ ...T.h3, fontSize: 38 }}>
                  Any stage fails → <span style={{ color: C.red, fontWeight: 500 }}>nothing is forwarded.</span> No raw fallback.
                </span>
              </div>
            </Stamp>

            {/* invariant 2: nonce forgery drama */}
            <Stamp delay={82} rotate={1}>
              <div style={{ display: "flex", alignItems: "center", gap: 28, background: C.white, border: `1px solid ${C.borderSoft}`, borderRadius: 16, padding: "24px 44px", position: "relative" }}>
                <span style={{ ...T.h3, fontSize: 38 }}>On-screen text tries to forge a token:</span>
                <Chip token="EMAIL_1_beef" size={30} delay={94} level="blocked" />
                {frame >= 116 && (
                  <Stamp delay={116} rotate={-8}>
                    <span
                      style={{
                        fontFamily: FONT,
                        fontSize: 32,
                        fontWeight: 500,
                        color: C.red,
                        border: `2px solid currentColor`,
                        borderRadius: 999,
                        padding: "8px 26px",
                        letterSpacing: "0.04em",
                        background: C.white,
                      }}
                    >
                      REJECTED
                    </span>
                  </Stamp>
                )}
                <span style={{ fontFamily: MONO, fontSize: 22, color: C.gray }}>session-nonce mismatch</span>
              </div>
            </Stamp>

            {/* invariant 3 */}
            <Stamp delay={134} rotate={-1}>
              <div style={{ display: "flex", gap: 24, fontFamily: FONT, fontSize: 26, fontWeight: 500, color: C.ink }}>
                {["SSE streaming-safe", "memory-only vault", "zero value logs"].map((t) => (
                  <span
                    key={t}
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 12,
                      border: `1.5px solid ${C.borderSoft}`,
                      borderRadius: 999,
                      padding: "12px 26px",
                    }}
                  >
                    <span style={{ width: 12, height: 12, borderRadius: "50%", background: C.green, display: "inline-block" }} />
                    {t}
                  </span>
                ))}
              </div>
            </Stamp>
          </div>
        </>
      )}

      {/* Phase 2: policy grid */}
      {gridPhase && !zoomPhase && (
        <div style={{ position: "absolute", inset: 0, opacity: gridOut }}>
          <div style={{ position: "absolute", top: 120, left: 0, right: 0, textAlign: "center" }}>
            <div style={{ ...T.h2, opacity: pop(frame, fps, 170) }}>
              You set the policy. <span style={{ color: C.gray, fontWeight: 400 }}>Per class.</span>
            </div>
          </div>
          <div style={{ position: "absolute", top: 300, left: 0, right: 0, display: "flex", justifyContent: "center", gap: 44 }}>
            {POLICY.map((col, ci) => {
              const p = pop(frame, fps, 184 + ci * 10);
              return (
                <div
                  key={col.level}
                  style={{
                    width: 440,
                    borderRadius: 16,
                    border: `1px solid ${C.borderFade}`,
                    background: C.white,
                    padding: "30px 34px",
                    boxShadow: "0 1px 2px rgba(12,12,12,.03)",
                    opacity: p,
                    transform: `translateY(${(1 - p) * 44}px)`,
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
                    <span style={{ width: 13, height: 13, borderRadius: "50%", background: col.color, display: "inline-block" }} />
                    <span style={{ fontFamily: MONO, fontSize: 30, fontWeight: 500, color: C.ink }}>{col.level}</span>
                  </div>
                  <div style={{ fontFamily: FONT, fontSize: 21, color: C.gray, marginBottom: 18 }}>{col.note}</div>
                  {col.classes.map((cls, i) => {
                    const rp = pop(frame, fps, 196 + ci * 10 + i * 4);
                    return (
                      <div
                        key={cls}
                        style={{
                          fontFamily: FONT,
                          fontSize: 24,
                          fontWeight: 400,
                          color: C.ink,
                          borderTop: i === 0 ? "none" : `1px solid ${C.borderFade}`,
                          padding: "14px 4px",
                          opacity: rp,
                          transform: `translateX(${(1 - rp) * -24}px)`,
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                        }}
                      >
                        {cls}
                        <span
                          style={{
                            fontFamily: FONT,
                            fontSize: 17,
                            fontWeight: 400,
                            color: col.color,
                            border: `1.5px solid currentColor`,
                            borderRadius: 999,
                            padding: "3px 14px",
                          }}
                        >
                          {col.level}
                        </span>
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Phase 3: PASSWORD → blocked, full frame */}
      {zoomPhase && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 40,
            opacity: finale,
          }}
        >
          <div
            style={{
              fontFamily: MONO,
              fontSize: 76,
              fontWeight: 500,
              color: C.ink,
              transform: `scale(${1 + ramp(frame, 308, 80) * 0.12})`,
            }}
          >
            PASSWORD → <span style={{ color: C.red }}>blocked</span>
          </div>
          <div style={{ ...T.h3, color: C.gray, opacity: pop(frame, fps, 330), textAlign: "center", lineHeight: 1.4 }}>
            Never resolvable. Not even by the model that asks nicely.
          </div>
        </div>
      )}
    </SceneBg>
  );
};
