import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { C, T, FONT, MONO } from "../theme";
import { pop, ramp } from "../anim";
import { SceneBg, Eyebrow } from "../components/Blocks";

const OLD_URL = `https://api.provider.com/v1`;
const NEW_URL = `http://127.0.0.1:18081/v1`;

/** Typewriter delete-then-retype of the base_url value. */
const useRetypedUrl = (frame: number, startAt: number) => {
  const del = Math.max(0, Math.min(OLD_URL.length, Math.floor((frame - startAt) / 1.6)));
  const afterDelete = OLD_URL.slice(0, OLD_URL.length - del);
  if (del < OLD_URL.length) return { text: afterDelete, done: false };
  const typeStart = startAt + OLD_URL.length * 1.6 + 8;
  const typed = Math.max(0, Math.min(NEW_URL.length, Math.floor((frame - typeStart) / 1.6)));
  return { text: NEW_URL.slice(0, typed), done: typed >= NEW_URL.length };
};

const DEAD_LINES = [
  "runtime:        hai-agent-runtime   # closed binary",
  "perceive_loop:  built-in            # not configurable",
  "action_exec:    built-in            # not configurable",
  "model:          Hcompany/Holo3-35B-A3B",
];

/**
 * [14–24s] The constraint → the hack. A read-only config; one live line.
 * base_url is retyped to localhost; a PLVA node docks into the wire.
 */
export const S03Constraint: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const { text: url, done } = useRetypedUrl(frame, 110);
  const commitPulse = done ? ramp(frame, 218, 20) : 0;
  const caretOn = Math.floor(frame / 8) % 2 === 0;

  const headline = (i: number) => pop(frame, fps, 6 + i * 22);
  const became = pop(frame, fps, 232);
  const underline = ramp(frame, 244, 16);

  // proxy node docking into a mini wire beneath the editor
  const dock = pop(frame, fps, 226);

  return (
    <SceneBg>
      {/* left: the argument */}
      <div style={{ position: "absolute", left: 150, top: 170, width: 700 }}>
        <Eyebrow text="The catch" />
        {["The agent runtime is a closed binary.", "You cannot edit its code.", "It exposes exactly one knob:"].map(
          (line, i) => (
            <div
              key={i}
              style={{
                ...T.h2,
                fontSize: 52,
                marginTop: i === 0 ? 30 : 22,
                opacity: headline(i),
                transform: `translateY(${(1 - headline(i)) * 26}px)`,
              }}
            >
              {line}
            </div>
          )
        )}
        <div
          style={{
            fontFamily: MONO,
            fontSize: 58,
            fontWeight: 700,
            marginTop: 30,
            color: C.ink,
            opacity: pop(frame, fps, 80),
            transform: `scale(${0.9 + 0.1 * pop(frame, fps, 80)})`,
            transformOrigin: "left center",
          }}
        >
          base_url
        </div>

        <div style={{ marginTop: 56, opacity: became, transform: `translateY(${(1 - became) * 30}px)` }}>
          <div style={{ ...T.h1, fontSize: 66 }}>
            So we <span style={{ position: "relative" }}>became<span style={{ position: "absolute", left: 0, bottom: -6, height: 8, width: `${underline * 100}%`, background: C.green, borderRadius: 4 }} /></span> the base URL.
          </div>
        </div>
      </div>

      {/* right: the config editor */}
      <div style={{ position: "absolute", right: 140, top: 190, width: 830 }}>
        <div
          style={{
            background: C.inverse,
            borderRadius: 18,
            padding: "34px 40px",
            boxShadow: `0 40px 90px rgba(12,12,12,.28)${commitPulse > 0 ? `, 0 0 0 ${6 * (1 - commitPulse)}px ${C.green}55` : ""}`,
            opacity: ramp(frame, 14, 12),
            transform: `translateY(${(1 - ramp(frame, 14, 12)) * 40}px)`,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", marginBottom: 26 }}>
            <span style={{ fontFamily: MONO, fontSize: 22, color: C.whiteDim }}>agent.config</span>
            <span
              style={{
                marginLeft: "auto",
                fontFamily: FONT,
                fontSize: 18,
                fontWeight: 650,
                letterSpacing: "0.1em",
                color: C.whiteDim,
                border: `2px solid rgba(255,255,255,.25)`,
                borderRadius: 999,
                padding: "4px 14px",
              }}
            >
              🔒 READ-ONLY
            </span>
          </div>
          {DEAD_LINES.map((l, i) => (
            <div
              key={i}
              style={{
                fontFamily: MONO,
                fontSize: 25,
                lineHeight: 1.85,
                color: C.whiteFaint,
                textDecoration: "line-through",
                textDecorationColor: "rgba(255,255,255,.2)",
                opacity: ramp(frame, 24 + i * 6, 8),
              }}
            >
              {l}
            </div>
          ))}
          <div style={{ fontFamily: MONO, fontSize: 27, lineHeight: 2.1, color: C.white, marginTop: 8 }}>
            <span style={{ color: "#7ecfa7" }}>base_url</span>
            <span style={{ color: C.whiteDim }}>: "</span>
            <span style={{ color: done ? "#5fdc9a" : C.white }}>{url}</span>
            {caretOn && <span style={{ borderLeft: `3px solid ${C.white}`, marginLeft: 2 }} />}
            <span style={{ color: C.whiteDim }}>"</span>
            <span
              style={{
                marginLeft: 18,
                fontSize: 20,
                color: "#5fdc9a",
                opacity: done ? ramp(frame, 220, 10) : 0,
              }}
            >
              ✓ the only line you control
            </span>
          </div>
        </div>

        {/* mini wire: runtime → [PLVA] → cloud */}
        <div style={{ marginTop: 44, position: "relative", height: 120, opacity: ramp(frame, 200, 16) }}>
          <svg width="830" height="120" style={{ position: "absolute", inset: 0 }}>
            <line x1="90" y1="60" x2="740" y2="60" stroke={C.grayLight} strokeWidth="3" />
            {done && (
              <circle cx={interpolate((frame - 226) % 40, [0, 40], [420, 740])} cy="60" r="7" fill={C.green} opacity={dock} />
            )}
          </svg>
          {["runtime", "cloud"].map((label, i) => (
            <div
              key={label}
              style={{
                position: "absolute",
                left: i === 0 ? 0 : 740,
                top: 28,
                fontFamily: FONT,
                fontSize: 21,
                fontWeight: 600,
                color: C.gray,
                border: `2px solid ${C.borderSoft}`,
                background: C.white,
                borderRadius: 12,
                padding: "14px 20px",
              }}
            >
              {label}
            </div>
          ))}
          <div
            style={{
              position: "absolute",
              left: 340,
              top: 16,
              transform: `translateY(${(1 - dock) * -70}px) scale(${0.6 + 0.4 * dock})`,
              opacity: dock,
              fontFamily: FONT,
              fontSize: 24,
              fontWeight: 700,
              color: C.white,
              background: C.green,
              borderRadius: 14,
              padding: "20px 28px",
              boxShadow: `0 14px 40px ${C.green}55`,
            }}
          >
            PLVA PROXY
          </div>
        </div>
      </div>
    </SceneBg>
  );
};
