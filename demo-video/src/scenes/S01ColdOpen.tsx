import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { C, T, MONO } from "../theme";
import { pop, ramp, EASE_IN_OUT } from "../anim";
import { SceneBg } from "../components/Blocks";
import { MockScreen } from "../components/MockScreen";
import { WindowFrame } from "../components/Blocks";
import { CornerBrackets, RecBadge, ShutterFlash } from "../components/Gadgets";

/**
 * [0–5s] Cold open. Shutter flash reveals a personal desktop; corner brackets
 * snap around the WHOLE canvas (the video itself is the screenshot); red boxes
 * stamp the secrets; the frame is dragged off toward the provider.
 */
export const S01ColdOpen: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const detected = frame >= 26;
  // the whole window gets yanked off-right at the end
  const yank = interpolate(frame, [116, 144], [0, 1400], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: (t) => t * t * (3 - 2 * t),
  });
  const line1 = pop(frame, fps, 34);
  const line2 = pop(frame, fps, 72);
  const arrowIn = pop(frame, fps, 112);

  return (
    <SceneBg grid={false}>
      {/* headline */}
      <div style={{ position: "absolute", top: 96, left: 0, right: 0, textAlign: "center", zIndex: 10 }}>
        <div style={{ ...T.h1, fontSize: 76, color: C.ink, opacity: line1, transform: `translateY(${(1 - line1) * 36}px)` }}>
          Your agent just screenshotted your{" "}
          <span style={{ color: detected ? C.red : C.ink }}>password.</span>
        </div>
        <div
          style={{
            ...T.h2,
            color: C.red,
            marginTop: 18,
            opacity: line2,
            transform: `translateY(${(1 - line2) * 30}px)`,
          }}
        >
          And sent it to the cloud.
        </div>
      </div>

      {/* the desktop being screenshotted */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 330,
          display: "flex",
          justifyContent: "center",
          transform: `translateX(${yank}px) rotate(${yank * 0.004}deg)`,
          filter: yank > 5 ? "blur(2px)" : "none",
        }}
      >
        <div style={{ opacity: ramp(frame, 4, 5) }}>
          <WindowFrame url="mybank.example.com/checkout" width={1180} height={620}>
            <MockScreen mode={detected ? "detected" : "raw"} staggerFrom={26} />
          </WindowFrame>
        </div>
      </div>

      {/* POST arrow chasing the frame off-screen */}
      <div
        style={{
          position: "absolute",
          top: 620,
          right: 130 - yank * 0.2,
          opacity: arrowIn * (yank < 1300 ? 1 : 0),
          transform: `scale(${arrowIn})`,
          zIndex: 12,
          display: "flex",
          alignItems: "center",
          gap: 18,
        }}
      >
        <span style={{ fontFamily: MONO, fontSize: 30, fontWeight: 700, color: C.white, background: C.red, padding: "12px 26px", borderRadius: 12 }}>
          POST → api.provider.com
        </span>
        <svg width="120" height="48" viewBox="0 0 120 48">
          <path d="M0 24 H92 M92 24 L68 6 M92 24 L68 42" stroke={C.red} strokeWidth="7" fill="none" strokeLinecap="round" />
        </svg>
      </div>

      {/* the video itself becomes the screenshot */}
      <CornerBrackets delay={8} inset={34} />
      <RecBadge text="CAPTURING" style={{ position: "absolute", top: 52, right: 150 }} />
      <ShutterFlash at={2} />
    </SceneBg>
  );
};
