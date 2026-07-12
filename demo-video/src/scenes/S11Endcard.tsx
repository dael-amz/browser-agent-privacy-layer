import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate, Easing } from "remotion";
import { C, T, FONT, MONO } from "../theme";
import { pop } from "../anim";
import { SceneBg, Wordmark } from "../components/Blocks";

const TOKEN = "«EMAIL_1_a3f9»";
const REAL = "alex.rivera@example.com";

/**
 * [112–120s] Endcard, structured like the site's landing hero: eyebrow pill
 * with the green dot, stacked two-line headline, then the chip-flip gesture.
 */
export const S11Endcard: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // landing-page stagger: each line rises 12px and clears a 3px blur
  const line = (delay: number) => {
    const p = pop(frame, fps, delay);
    return {
      opacity: p,
      transform: `translateY(${(1 - p) * 12}px)`,
      filter: `blur(${(1 - p) * 3}px)`,
    };
  };

  const eyebrow = line(6);
  const wm = line(20);
  const koan1 = line(48);
  const koan2 = line(94); // the pause between lines is the point

  // chip flip: 0 → 180 (reveal back) → 0, then lock
  const flip = interpolate(frame, [140, 152, 168, 180], [0, 180, 180, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic),
  });
  const showBack = flip > 90;
  const lockP = pop(frame, fps, 184);
  const chipIn = line(118);
  const foot = line(112);

  return (
    <SceneBg>
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 34,
          paddingBottom: 40,
        }}
      >
        {/* .eyebrow from the landing page */}
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 13,
            padding: "11px 26px",
            border: `1.5px solid ${C.borderSoft}`,
            borderRadius: 999,
            fontFamily: FONT,
            fontSize: 24,
            color: C.gray,
            ...eyebrow,
          }}
        >
          <span style={{ width: 13, height: 13, borderRadius: "50%", background: C.green, display: "inline-block" }} />
          Private computer use
        </div>

        <div style={wm}>
          <Wordmark size={110} sub="Beta" />
        </div>

        {/* the koan, two stacked lines like the landing h1, a breath apart */}
        <div style={{ textAlign: "center", lineHeight: 1.12 }}>
          <div style={{ ...T.h1, fontSize: 72, ...koan1 }}>
            Redact for the <span style={{ color: C.green }}>model.</span>
          </div>
          <div style={{ ...T.h1, fontSize: 72, ...koan2 }}>Not for the user.</div>
        </div>

        {/* the whole product in one gesture: the chip flips */}
        <div style={{ display: "flex", alignItems: "center", gap: 24, marginTop: 10, ...chipIn }}>
          <div style={{ perspective: 900 }}>
            <div
              style={{
                transform: `rotateY(${flip}deg) scale(${1 + (lockP > 0 && lockP < 1 ? (1 - lockP) * 0.06 : 0)})`,
                fontFamily: MONO,
                fontSize: 32,
                fontWeight: 400,
                color: showBack ? C.ink : C.white,
                background: showBack ? C.paperAlt : C.inverse,
                border: showBack ? `1.5px solid ${C.green}` : "1.5px solid transparent",
                borderRadius: 10,
                padding: "14px 30px",
                display: "inline-flex",
                alignItems: "center",
                gap: 12,
              }}
            >
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: "50%",
                  background: C.green,
                  display: "inline-block",
                }}
              />
              <span style={{ display: "inline-block", transform: showBack ? "scaleX(-1)" : "none" }}>
                {showBack ? REAL : TOKEN}
              </span>
            </div>
          </div>
          <svg width="28" height="34" viewBox="0 0 30 36" style={{ opacity: lockP }}>
            <rect x="3" y="15" width="24" height="18" rx="4" fill={C.ink} />
            <path d="M8 15 v-4 a7 7 0 0 1 14 0 v4" stroke={C.ink} strokeWidth="4" fill="none" />
          </svg>
        </div>

        {/* .localnote footer */}
        <div
          style={{
            position: "absolute",
            bottom: 84,
            left: 0,
            right: 0,
            textAlign: "center",
            fontFamily: FONT,
            fontSize: 24,
            fontWeight: 400,
            color: C.gray,
            ...foot,
          }}
        >
          Built at The Computer Use Hackathon · Powered by <span style={{ color: C.ink, fontWeight: 500 }}>H Company Holo3-35B</span>
        </div>
      </div>
    </SceneBg>
  );
};
