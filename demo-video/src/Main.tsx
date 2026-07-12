import React from "react";
import { AbsoluteFill, Sequence, useCurrentFrame } from "remotion";
import { C, FONT } from "./theme";
import { sec } from "./anim";
import { S01ColdOpen } from "./scenes/S01ColdOpen";
import { S02Problem } from "./scenes/S02Problem";
import { S03Constraint } from "./scenes/S03Constraint";
import { S04Title } from "./scenes/S04Title";
import { S05RequestLeg } from "./scenes/S05RequestLeg";
import { S06ResponseLeg } from "./scenes/S06ResponseLeg";
import { S07Compare } from "./scenes/S07Compare";
import { S08LiveProof } from "./scenes/S08LiveProof";
import { S09FailClosed } from "./scenes/S09FailClosed";
import { S10OneLine } from "./scenes/S10OneLine";
import { S11Endcard } from "./scenes/S11Endcard";

/**
 * THE BEAT SHEET — synthesized from the judged storyboards ("engineer's cut"
 * + grafts). Times in seconds; hard cuts between scenes by design.
 */
const BEATS: Array<{ from: number; to: number; C: React.FC }> = [
  { from: 0, to: 5, C: S01ColdOpen },
  { from: 5, to: 14, C: S02Problem },
  { from: 14, to: 24, C: S03Constraint },
  { from: 24, to: 33, C: S04Title },
  { from: 33, to: 47, C: S05RequestLeg },
  { from: 47, to: 60, C: S06ResponseLeg },
  { from: 60, to: 74, C: S07Compare },
  { from: 74, to: 88, C: S08LiveProof },
  { from: 88, to: 101, C: S09FailClosed },
  { from: 101, to: 112, C: S10OneLine },
  { from: 112, to: 120, C: S11Endcard },
];

/** Persistent sponsor tag — visible in every frame a judge scrubs to. */
const SponsorTag: React.FC = () => (
  <div
    style={{
      position: "absolute",
      bottom: 30,
      left: 44,
      fontFamily: FONT,
      fontSize: 20,
      fontWeight: 600,
      color: C.gray,
      opacity: 0.6,
      zIndex: 100,
    }}
  >
    powered by H Company · Holo3
  </div>
);

export const Main: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ background: C.white }}>
      {BEATS.map(({ from, to, C: Scene }, i) => (
        <Sequence key={i} from={sec(from)} durationInFrames={sec(to - from)} name={`S${String(i + 1).padStart(2, "0")}`}>
          <Scene />
        </Sequence>
      ))}
      {/* sponsor tag from the title reveal onward (except the full-bleed split and the endcard, which credits H Company itself) */}
      {frame >= sec(24) && !(frame >= sec(60) && frame < sec(74)) && frame < sec(112) && <SponsorTag />}
      {/*
        MUSIC: drop a 120s track at public/music.mp3 and uncomment:
        <Audio src={staticFile("music.mp3")} volume={0.8} />
        Direction (from the storyboard): precise minimal electronic ~122 BPM;
        ticking pulse for 0–24s, confident cut at the title, arps layering
        through the pipeline, half-time pads for the split, near-silence for
        the invariants, hard cut to silence 1s before the end.
      */}
    </AbsoluteFill>
  );
};
