import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate, Easing } from "remotion";
import { C, T, FONT, MONO } from "../theme";
import { pop } from "../anim";
import { SceneBg } from "../components/Blocks";

const TOKEN = "«EMAIL_1_a3f9»";
const REAL = "alex.rivera@example.com";

/**
 * [112–120s] Endcard. The koan with a pause; the chip flips once to show the
 * real value on its back face, flips back, locks. Then stillness.
 */
export const S11Endcard: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const wm = pop(frame, fps, 4);
  const cursorOn = Math.floor(frame / 12) % 2 === 0 || frame > 200;
  const koan1 = pop(frame, fps, 46);
  const koan2 = pop(frame, fps, 92); // the pause between lines is the point

  // chip flip: 0 → 180 (reveal back) → 0, then lock
  const flip = interpolate(frame, [140, 152, 168, 180], [0, 180, 180, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic),
  });
  const showBack = flip > 90;
  const lockP = pop(frame, fps, 184);

  const foot = pop(frame, fps, 110);

  return (
    <SceneBg grid={false}>
      <div style={{ position: "absolute", top: 300, left: 0, right: 0, textAlign: "center" }}>
        <div style={{ display: "inline-flex", alignItems: "baseline", gap: 18, opacity: wm, transform: `translateY(${(1 - wm) * 30}px)` }}>
          <span style={{ fontFamily: FONT, fontSize: 130, fontWeight: 700, letterSpacing: "-0.04em", color: C.ink }}>Holo</span>
          <span style={{ width: 22, height: 88, background: C.green, opacity: cursorOn ? 1 : 0.15, borderRadius: 4 }} />
        </div>
        <div style={{ ...T.h3, color: C.gray, marginTop: 0, opacity: wm }}>Private computer use</div>
      </div>

      {/* the koan, two lines, a breath apart */}
      <div style={{ position: "absolute", top: 580, left: 0, right: 0, textAlign: "center" }}>
        <span style={{ ...T.h2, fontSize: 54, opacity: koan1, display: "inline-block", transform: `translateY(${(1 - koan1) * 24}px)` }}>
          Redact for the <span style={{ color: C.green }}>model.</span>
        </span>
        <span
          style={{
            ...T.h2,
            fontSize: 54,
            marginLeft: 28,
            opacity: koan2,
            display: "inline-block",
            transform: `translateY(${(1 - koan2) * 24}px)`,
          }}
        >
          Not for the user.
        </span>
      </div>

      {/* the whole product in one gesture: the chip flips */}
      <div style={{ position: "absolute", top: 710, left: 0, right: 0, display: "flex", justifyContent: "center", alignItems: "center", gap: 24 }}>
        <div style={{ perspective: 900, opacity: pop(frame, fps, 120) }}>
          <div
            style={{
              transform: `rotateY(${flip}deg) scale(${1 + (lockP > 0 && lockP < 1 ? (1 - lockP) * 0.06 : 0)})`,
              fontFamily: MONO,
              fontSize: 34,
              fontWeight: 600,
              color: C.white,
              background: showBack ? C.green : C.inverse,
              borderRadius: 12,
              padding: "14px 30px",
              boxShadow: showBack ? `0 14px 44px ${C.green}55` : `inset 0 0 0 2px ${C.green}`,
            }}
          >
            <span style={{ display: "inline-block", transform: showBack ? "scaleX(-1)" : "none" }}>
              {showBack ? REAL : TOKEN}
            </span>
          </div>
        </div>
        <svg width="30" height="36" viewBox="0 0 30 36" style={{ opacity: lockP }}>
          <rect x="3" y="15" width="24" height="18" rx="4" fill={C.ink} />
          <path d="M8 15 v-4 a7 7 0 0 1 14 0 v4" stroke={C.ink} strokeWidth="4" fill="none" />
        </svg>
      </div>

      <div
        style={{
          position: "absolute",
          bottom: 90,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: FONT,
          fontSize: 25,
          fontWeight: 550,
          color: C.gray,
          opacity: foot,
        }}
      >
        Built at The Computer Use Hackathon · Powered by <span style={{ color: C.ink, fontWeight: 700 }}>H Company Holo3-35B</span>
      </div>
    </SceneBg>
  );
};
