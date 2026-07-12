import React from "react";
import { Img, OffthreadVideo, staticFile } from "remotion";
import { SHOTS, ShotId } from "../shots";
import { MediaPlaceholder } from "./MediaPlaceholder";

/**
 * Renders the real capture if `src` is set in shots.ts, otherwise a labeled
 * placeholder slot. Swapping in footage requires zero scene edits.
 */
export const Shot: React.FC<{
  id: ShotId;
  style?: React.CSSProperties;
  delay?: number;
  dark?: boolean;
  dimAfter?: number;
}> = ({ id, style, delay, dark, dimAfter }) => {
  const def: import("../shots").ShotDef = SHOTS[id];
  if (def.src) {
    const media =
      def.kind === "video" ? (
        <OffthreadVideo
          src={staticFile(def.src)}
          startFrom={Math.round((def.startFrom ?? 0) * 30)}
          muted
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
      ) : (
        <Img
          src={staticFile(def.src)}
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
      );
    return (
      <div style={{ borderRadius: 16, overflow: "hidden", position: "relative", ...style }}>
        {media}
      </div>
    );
  }
  return (
    <MediaPlaceholder
      id={def.id}
      kind={def.kind}
      title={def.title}
      description={def.description}
      seconds={def.seconds}
      style={style}
      delay={delay}
      dark={dark}
      dimAfter={dimAfter}
    />
  );
};
