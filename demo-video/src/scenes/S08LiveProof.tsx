import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate, Easing } from "remotion";
import { C, T, FONT, MONO } from "../theme";
import { pop, ramp, rampOut } from "../anim";
import { SceneBg } from "../components/Blocks";
import { Shot } from "../components/Shot";
import { Callout, MagnifierRing, Stamp } from "../components/Gadgets";

/**
 * [74–88s] Live proof. "Don't trust us. Audit it." then the real run:
 * terminal + audit viewer, with authored callouts riding on top.
 */
export const S08LiveProof: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const head1 = pop(frame, fps, 2);
  const head2 = pop(frame, fps, 16);
  const headOut = rampOut(frame, 52, 12);
  const shotIn = pop(frame, fps, 58);

  // magnifier glides from mid-recording to a "chip" position and parks
  const glide = interpolate(frame, [200, 250], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic),
  });
  const magX = interpolate(glide, [0, 1], [700, 1330]);
  const magY = interpolate(glide, [0, 1], [560, 470]);
  const magIn = pop(frame, fps, 196);
  const parkLabel = pop(frame, fps, 254);

  const badge = pop(frame, fps, 368);

  return (
    <SceneBg>
      {/* headline (clears out before the recording) */}
      <div style={{ position: "absolute", top: 300, left: 0, right: 0, textAlign: "center", opacity: headOut, zIndex: 5 }}>
        <div style={{ ...T.hero, fontSize: 110, opacity: head1, transform: `translateY(${(1 - head1) * 40}px)` }}>
          Don't trust us.
        </div>
        <div style={{ ...T.hero, fontSize: 110, color: C.green, opacity: head2, transform: `translateY(${(1 - head2) * 40}px)` }}>
          Audit it.
        </div>
      </div>

      {/* the live recording */}
      {frame >= 56 && (
        <div style={{ position: "absolute", inset: 0, opacity: shotIn }}>
          <div style={{ position: "absolute", top: 80, left: 130, right: 130 }}>
            {/* command bar */}
            <div
              style={{
                fontFamily: MONO,
                fontSize: 27,
                color: C.white,
                background: C.inverse,
                borderRadius: "14px 14px 0 0",
                padding: "18px 30px",
                display: "flex",
                alignItems: "center",
                gap: 18,
              }}
            >
              <span style={{ color: "#5fdc9a" }}>$</span>
              PLVA_REDACT=1 PLVA_REDACT_ENGINE=vision ./run_step1.sh
              <span style={{ marginLeft: "auto", fontSize: 20, color: C.whiteDim, fontFamily: FONT, fontWeight: 600 }}>
                not a mockup — a live run
              </span>
            </div>
            <Shot id="SHOT-03" style={{ width: "100%", height: 700, borderRadius: "0 0 16px 16px" }} />
          </div>

          {/* callout 1: the env var */}
          {frame >= 120 && frame < 200 && (
            <Callout x={420} y={106} dx={150} dy={90} text="privacy: one env var" delay={120} />
          )}

          {/* callout 2: magnifier parks on a chip in the viewer */}
          {frame >= 196 && (
            <>
              <MagnifierRing x={magX} y={magY} r={95} color={C.green} opacity={magIn} />
              <div
                style={{
                  position: "absolute",
                  left: magX - 240,
                  top: magY + 120,
                  width: 480,
                  textAlign: "center",
                  fontFamily: FONT,
                  fontSize: 26,
                  fontWeight: 650,
                  color: C.white,
                  background: C.green,
                  borderRadius: 999,
                  padding: "12px 10px",
                  opacity: parkLabel,
                  transform: `translateY(${(1 - parkLabel) * 16}px)`,
                  boxShadow: "0 14px 40px rgba(12,12,12,.22)",
                }}
              >
                the only version that ever left
              </div>
            </>
          )}

          {/* lower third */}
          <div
            style={{
              position: "absolute",
              bottom: 64,
              left: 130,
              display: "flex",
              alignItems: "center",
              gap: 20,
              opacity: ramp(frame, 80, 14),
            }}
          >
            <span style={{ width: 16, height: 16, borderRadius: 999, background: C.red, boxShadow: `0 0 14px 2px ${C.red}77` }} />
            <span style={{ fontFamily: FONT, fontSize: 28, fontWeight: 750, color: C.ink }}>LIVE RUN</span>
            <span style={{ fontFamily: MONO, fontSize: 24, color: C.gray }}>Holo3-35B-A3B · H Company · end-to-end</span>
          </div>

          {/* completion badge */}
          {frame >= 366 && (
            <div style={{ position: "absolute", bottom: 54, right: 130 }}>
              <Stamp delay={368} rotate={-3}>
                <span
                  style={{
                    fontFamily: FONT,
                    fontSize: 30,
                    fontWeight: 750,
                    color: C.white,
                    background: C.green,
                    borderRadius: 14,
                    padding: "16px 32px",
                    display: "inline-block",
                    boxShadow: `0 16px 50px ${C.green}55`,
                    opacity: badge,
                  }}
                >
                  ✓ task completed — 0 values upstream
                </span>
              </Stamp>
            </div>
          )}
        </div>
      )}
    </SceneBg>
  );
};
