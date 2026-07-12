import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { C, FONT, MONO } from "../theme";
import { pop } from "../anim";

export type ShotKind = "video" | "image";

/**
 * A clearly-labeled slot for real footage to be dropped in later.
 *
 * To replace: swap the inner content for
 *   <OffthreadVideo src={staticFile("shots/SHOT-01.mp4")} style={{width:"100%",height:"100%",objectFit:"cover"}} />
 * or <Img src={staticFile("shots/SHOT-01.png")} ... />
 * keeping the same outer wrapper. See shots.ts for the full shot list.
 */
export const MediaPlaceholder: React.FC<{
  id: string;
  kind: ShotKind;
  title: string;
  description: string;
  seconds?: number;
  style?: React.CSSProperties;
  delay?: number;
  dark?: boolean;
}> = ({ id, kind, title, description, seconds, style, delay = 0, dark = false }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = pop(frame, fps, delay);
  const shimmer = interpolate(frame % 90, [0, 90], [-30, 130]);

  const ink = dark ? C.white : C.ink;
  const sub = dark ? C.whiteDim : C.gray;
  const border = dark ? "rgba(255,255,255,.35)" : "rgba(12,12,12,.28)";
  const stripe = dark ? "rgba(255,255,255,.05)" : "rgba(12,12,12,.035)";
  const bg = dark ? "rgba(255,255,255,.04)" : "rgba(12,12,12,.02)";

  return (
    <div
      style={{
        position: "relative",
        borderRadius: 16,
        border: `3px dashed ${border}`,
        background: `repeating-linear-gradient(-45deg, ${stripe} 0 18px, transparent 18px 36px), ${bg}`,
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        opacity: p,
        transform: `scale(${0.97 + 0.03 * p})`,
        ...style,
      }}
    >
      {/* shimmer sweep so the slot reads as "live", not broken */}
      <div
        style={{
          position: "absolute",
          top: 0,
          bottom: 0,
          left: `${shimmer}%`,
          width: "18%",
          background: `linear-gradient(90deg, transparent, ${dark ? "rgba(255,255,255,.06)" : "rgba(12,12,12,.04)"}, transparent)`,
        }}
      />
      <div style={{ textAlign: "center", padding: "0 60px", maxWidth: "92%" }}>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 14,
            marginBottom: 18,
          }}
        >
          <span
            style={{
              fontFamily: MONO,
              fontSize: 30,
              fontWeight: 700,
              color: dark ? C.inverseDeep : C.white,
              background: dark ? C.white : C.ink,
              borderRadius: 10,
              padding: "6px 18px",
            }}
          >
            {id}
          </span>
          <span
            style={{
              fontFamily: FONT,
              fontSize: 20,
              fontWeight: 650,
              letterSpacing: "0.12em",
              color: sub,
              border: `2px solid ${border}`,
              borderRadius: 999,
              padding: "6px 16px",
            }}
          >
            {kind === "video" ? "SCREEN RECORDING" : "SCREENSHOT"}
            {seconds ? ` · ~${seconds}s` : ""}
          </span>
        </div>
        <div style={{ fontFamily: FONT, fontSize: 34, fontWeight: 650, color: ink, letterSpacing: "-0.02em", marginBottom: 10 }}>
          {title}
        </div>
        <div style={{ fontFamily: FONT, fontSize: 22, fontWeight: 450, color: sub, lineHeight: 1.35 }}>{description}</div>
      </div>
    </div>
  );
};
