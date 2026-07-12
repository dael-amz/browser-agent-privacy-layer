import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig, interpolate, Easing } from "remotion";
import { C, T, FONT } from "../theme";
import { pop, ramp } from "../anim";
import { Shot } from "../components/Shot";
import { Ticker, TaskBar } from "../components/Gadgets";

/**
 * [60–74s] With vs Without — full-frame split. Both sides complete the SAME
 * task in sync; only one leaked. Green claims the frame at the end.
 */
export const S07Compare: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const flash = interpolate(frame, [0, 2, 5], [1, 1, 0], { extrapolateRight: "clamp" });
  const settle = interpolate(frame, [3, 18], [1.04, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });

  // twin task bars fill in perfect sync
  const progress = ramp(frame, 210, 80);
  const done = progress >= 0.999;

  // the divider slides right: green claims ~80% of the frame
  const claim = interpolate(frame, [330, 356], [0.5, 0.8], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic),
  });
  const leftW = (1 - claim) * 100;
  const verdict = pop(frame, fps, 352);

  const Side: React.FC<{ which: "without" | "with" }> = ({ which }) => {
    const red = which === "without";
    return (
      <div style={{ height: "100%", display: "flex", flexDirection: "column", overflow: "hidden", position: "relative" }}>
        <div
          style={{
            background: red ? C.red : C.green,
            color: C.white,
            fontFamily: FONT,
            fontSize: 34,
            fontWeight: 750,
            letterSpacing: "0.04em",
            textAlign: "center",
            padding: "20px 0",
            flexShrink: 0,
          }}
        >
          {red ? "WITHOUT PLVA" : "WITH PLVA"}
          <span style={{ fontWeight: 500, fontSize: 24, opacity: 0.85, marginLeft: 20 }}>the provider receives this</span>
        </div>
        <div style={{ flex: 1, padding: 34, display: "flex", flexDirection: "column", gap: 24, minHeight: 0 }}>
          <Shot id={red ? "SHOT-02A" : "SHOT-02B"} style={{ flex: 1, minHeight: 0 }} delay={10} />
          <div style={{ opacity: ramp(frame, 200, 12) }}>
            <TaskBar
              progress={progress}
              color={red ? C.red : C.green}
              label="task progress"
              done={done}
            />
          </div>
        </div>
        <div style={{ flexShrink: 0, background: red ? C.redSoft : C.greenSoft }}>
          <Ticker
            fontSize={22}
            color={red ? C.red : C.green}
            speed={red ? 3.2 : 2.6}
            text={
              red
                ? "upstream received: alex.rivera@example.com · 4929 1188 3407 2216 · hunter2!x · hk-live-9f3ab27c41d8 ·"
                : "upstream received: «EMAIL_1_a3f9» · ▮▮▮▮▮ (card blocked) · ▮▮▮▮▮ (password blocked) · «API_KEY_1_a3f9» ·"
            }
          />
        </div>
      </div>
    );
  };

  return (
    <AbsoluteFill style={{ background: C.white }}>
      <div style={{ position: "absolute", inset: 0, display: "flex", transform: `scale(${settle})` }}>
        <div style={{ width: `${leftW}%`, borderRight: `3px solid ${C.ink}`, transition: "none" }}>
          <Side which="without" />
        </div>
        <div style={{ width: `${100 - leftW}%` }}>
          <Side which="with" />
        </div>
      </div>

      {/* synced double-check flare */}
      {done && frame < 330 && (
        <div
          style={{
            position: "absolute",
            top: "42%",
            left: 0,
            right: 0,
            textAlign: "center",
            fontFamily: FONT,
            fontSize: 46,
            fontWeight: 750,
            color: C.ink,
            opacity: pop(frame, fps, 296),
            textShadow: "0 2px 30px rgba(255,255,255,.9)",
          }}
        >
          <span style={{ background: "rgba(255,255,255,.92)", borderRadius: 18, padding: "18px 44px", border: `2px solid ${C.borderSoft}` }}>
            ✓ both tasks completed — simultaneously
          </span>
        </div>
      )}

      {/* verdict over the green claim */}
      {frame >= 348 && (
        <div
          style={{
            position: "absolute",
            top: "38%",
            left: `${leftW}%`,
            right: 0,
            textAlign: "center",
            opacity: verdict,
            transform: `translateY(${(1 - verdict) * 30}px)`,
          }}
        >
          <div style={{ ...T.h1, fontSize: 64, color: C.ink }}>
            <span style={{ background: "rgba(255,255,255,.95)", borderRadius: 20, padding: "22px 40px", boxShadow: "0 20px 60px rgba(12,12,12,.12)", display: "inline-block", lineHeight: 1.25 }}>
              Same task. Same result.
              <br />
              Only one of them <span style={{ color: C.red }}>leaked.</span>
            </span>
          </div>
        </div>
      )}

      <AbsoluteFill style={{ background: "#000", opacity: flash, pointerEvents: "none" }} />
    </AbsoluteFill>
  );
};
