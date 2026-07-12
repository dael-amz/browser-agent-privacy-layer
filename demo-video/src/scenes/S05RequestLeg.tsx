import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate, Easing } from "remotion";
import { C, T, FONT, MONO } from "../theme";
import { pop, ramp } from "../anim";
import { SceneBg, Eyebrow } from "../components/Blocks";
import { MiniFrame } from "../components/MiniFrame";

const PITCH = 1500;
const FIRST_X = 960; // strip-local center of station 0
const STATION_Y = 350;

// arrival frame of each station (dwell ≈ 50, travel ≈ 20)
const ARRIVE = [0, 70, 140, 210, 280, 350];

const STATIONS = [
  { n: "1", title: "FRAME", sub: "the screenshot, intercepted" },
  { n: "2", title: "DETECT", sub: "Apple Vision OCR + Core ML — on-device" },
  { n: "3", title: "PAINT", sub: "chips cover the real pixels" },
  { n: "4", title: "VAULT", sub: "memory-only · no disk · no logs" },
  { n: "5", title: "SCRUB", sub: "history: vault match + Rampart backstop" },
  { n: "6", title: "TEACH + SHIP", sub: "token manifest attached" },
];

const VaultCard: React.FC<{ active: boolean; at: number }> = ({ active, at }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const rows = [
    { k: "EMAIL_1_a3f9", v: "████████████", edge: C.green },
    { k: "PHONE_1_a3f9", v: "████████", edge: C.green },
    { k: "API_KEY_1_a3f9", v: "████████", edge: C.amber },
    { k: "PASSWORD", v: "never stored", edge: C.red },
  ];
  return (
    <div
      style={{
        width: 620,
        background: C.inverse,
        borderRadius: 16,
        padding: "26px 30px",
        boxShadow: "0 24px 60px rgba(12,12,12,.25)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 18 }}>
        <svg width="30" height="34" viewBox="0 0 30 34">
          <rect x="3" y="14" width="24" height="17" rx="4" fill={C.green} />
          <path d="M8 14 v-4 a7 7 0 0 1 14 0 v4" stroke={C.green} strokeWidth="4" fill="none" />
        </svg>
        <span style={{ fontFamily: FONT, fontSize: 26, fontWeight: 700, color: C.white }}>VAULT</span>
        <span style={{ fontFamily: FONT, fontSize: 19, color: C.whiteDim, marginLeft: "auto" }}>session-scoped RAM</span>
      </div>
      {rows.map((r, i) => {
        const p = active ? pop(frame, fps, at + 6 + i * 5) : 0;
        return (
          <div
            key={r.k}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              fontFamily: MONO,
              fontSize: 22,
              lineHeight: 1.9,
              opacity: p,
              transform: `translateY(${(1 - p) * -16}px)`,
            }}
          >
            <span style={{ color: C.white }}>
              <span style={{ color: r.edge }}>●</span> {r.k}
            </span>
            <span style={{ color: r.v === "never stored" ? C.red : C.whiteDim }}>{r.v}</span>
          </div>
        );
      })}
    </div>
  );
};

const ScrubCard: React.FC<{ active: boolean; at: number }> = ({ active, at }) => {
  const frame = useCurrentFrame();
  const local = active ? frame - at : 0;
  const lines = [
    { pre: "I typed ", secret: "alex.rivera@example.com", post: " into the field", token: "«EMAIL_1_a3f9»" },
    { pre: "then called ", secret: "hk-live-9f3ab27c41d8", post: "", token: "«API_KEY_1_a3f9»" },
  ];
  return (
    <div style={{ width: 640 }}>
      {lines.map((l, i) => {
        const t = ramp(local, 10 + i * 22, 16); // 0→1 as the secret is scrubbed
        return (
          <div
            key={i}
            style={{
              background: C.white,
              border: `1.5px solid ${C.borderSoft}`,
              borderRadius: 12,
              padding: "16px 22px",
              marginBottom: 16,
              fontFamily: MONO,
              fontSize: 21,
              color: C.ink,
              boxShadow: "0 10px 30px rgba(12,12,12,.08)",
            }}
          >
            <span style={{ color: C.gray, fontSize: 17, display: "block", marginBottom: 4 }}>
              history · step {7 - i}
            </span>
            {l.pre}
            {t < 0.5 ? (
              <span
                style={{
                  color: C.red,
                  background: C.redSoft,
                  borderRadius: 4,
                  padding: "1px 6px",
                  opacity: 1 - t * 2,
                  filter: `blur(${t * 6}px)`,
                }}
              >
                {l.secret}
              </span>
            ) : (
              <span
                style={{
                  color: C.white,
                  background: C.inverse,
                  borderRadius: 5,
                  padding: "1px 8px",
                  boxShadow: `inset 0 0 0 2px ${C.green}`,
                  opacity: (t - 0.5) * 2,
                }}
              >
                {l.token}
              </span>
            )}
            {l.post}
          </div>
        );
      })}
      <div style={{ fontFamily: FONT, fontSize: 20, color: C.gray, textAlign: "center" }}>
        the runtime remembers what it typed — the scrub makes sure the model doesn't
      </div>
    </div>
  );
};

const ShipCard: React.FC<{ active: boolean; at: number }> = ({ active, at }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = active ? frame - at : 0;
  const depart = interpolate(local, [34, 62], [0, 330], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.in(Easing.cubic),
  });
  const p = active ? pop(frame, fps, at) : 0;
  return (
    <div style={{ position: "relative", width: 900, height: 560 }}>
      {/* dotted trust boundary */}
      <div
        style={{
          position: "absolute",
          left: 560,
          top: -40,
          bottom: -20,
          borderLeft: `4px dashed ${C.green}`,
          opacity: p,
        }}
      />
      <div style={{ position: "absolute", left: 380, top: -34, fontFamily: FONT, fontSize: 20, fontWeight: 700, color: C.green, opacity: p }}>
        STAYS LOCAL
      </div>
      <div style={{ position: "absolute", left: 590, top: -34, fontFamily: FONT, fontSize: 20, fontWeight: 700, color: C.gray, opacity: p }}>
        → UPSTREAM
      </div>

      {/* departing bundle */}
      <div style={{ position: "absolute", left: depart * 1.1, top: 40, transform: `scale(${0.82})`, transformOrigin: "top left", opacity: p }}>
        <MiniFrame mode="redacted" width={480} staggerFrom={0} />
        <div
          style={{
            marginTop: -22,
            marginLeft: 320,
            width: 300,
            background: C.white,
            border: `2px solid ${C.green}`,
            borderRadius: 12,
            padding: "14px 18px",
            fontFamily: MONO,
            fontSize: 17,
            color: C.ink,
            boxShadow: "0 14px 36px rgba(12,12,12,.14)",
            transform: "rotate(3deg)",
          }}
        >
          <div style={{ fontFamily: FONT, fontWeight: 700, fontSize: 16, color: C.green, marginBottom: 6 }}>TOKEN MANIFEST</div>
          «EMAIL_1_a3f9» email
          <br />
          «PHONE_1_a3f9» phone
          <br />
          «API_KEY_1_a3f9» key
        </div>
      </div>

      {/* Holo3 destination */}
      <div
        style={{
          position: "absolute",
          right: 0,
          top: 130,
          background: C.inverse,
          borderRadius: 18,
          padding: "30px 38px",
          textAlign: "center",
          opacity: p,
        }}
      >
        <div style={{ fontFamily: FONT, fontSize: 30, fontWeight: 700, color: C.white }}>Holo3-35B</div>
        <div style={{ fontFamily: MONO, fontSize: 18, color: C.whiteDim, marginTop: 6 }}>H COMPANY</div>
      </div>
    </div>
  );
};

/**
 * [33–47s] Centerpiece I — the request leg. One continuous dolly across six
 * stations: FRAME → DETECT → PAINT → VAULT → SCRUB → TEACH+SHIP.
 */
export const S05RequestLeg: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // camera keyframes: dwell at each station, glide between
  const inputRange: number[] = [];
  const outputRange: number[] = [];
  ARRIVE.forEach((a, i) => {
    inputRange.push(a, a + 50);
    const x = -(FIRST_X + i * PITCH - 960);
    outputRange.push(x, x);
  });
  const camX = interpolate(frame, inputRange, outputRange, {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic),
  });

  const activeStation = ARRIVE.reduce((acc, a, i) => (frame >= a ? i : acc), 0);
  const scan = ramp(frame, ARRIVE[1] + 4, 34); // beam at DETECT

  return (
    <SceneBg>
      {/* fixed header */}
      <div style={{ position: "absolute", top: 96, left: 160, right: 160, zIndex: 10 }}>
        <Eyebrow text="Request leg — before anything leaves this machine" color={C.green} />
        <div style={{ ...T.h2, marginTop: 20 }}>
          {STATIONS[activeStation].n} · {STATIONS[activeStation].title}
          <span style={{ color: C.gray, fontWeight: 450, fontSize: 38, marginLeft: 26 }}>{STATIONS[activeStation].sub}</span>
        </div>
      </div>

      {/* station progress dots */}
      <div style={{ position: "absolute", bottom: 84, left: 0, right: 0, display: "flex", justifyContent: "center", gap: 22, zIndex: 10 }}>
        {STATIONS.map((s, i) => (
          <div
            key={i}
            style={{
              width: i === activeStation ? 46 : 16,
              height: 16,
              borderRadius: 999,
              background: i <= activeStation ? C.green : "rgba(12,12,12,.12)",
              transition: "none",
            }}
          />
        ))}
      </div>

      {/* the dolly strip */}
      <div style={{ position: "absolute", top: STATION_Y, left: 0, width: FIRST_X + PITCH * 6, height: 620, transform: `translateX(${camX}px)` }}>
        {/* conveyor wire */}
        <div style={{ position: "absolute", top: 300, left: 300, width: PITCH * 5 + 900, borderTop: `3px solid ${C.borderSoft}` }} />

        {/* Station 1 — FRAME */}
        <div style={{ position: "absolute", left: FIRST_X - 310, top: 80 }}>
          <MiniFrame mode="raw" />
        </div>

        {/* Station 2 — DETECT */}
        <div style={{ position: "absolute", left: FIRST_X + PITCH - 310, top: 80 }}>
          <MiniFrame mode="detected" staggerFrom={ARRIVE[1] + 22} scanProgress={scan} />
        </div>

        {/* Station 3 — PAINT */}
        <div style={{ position: "absolute", left: FIRST_X + PITCH * 2 - 310, top: 80 }}>
          <MiniFrame mode="redacted" staggerFrom={ARRIVE[2] + 8} />
        </div>

        {/* Station 4 — VAULT */}
        <div style={{ position: "absolute", left: FIRST_X + PITCH * 3 - 310, top: 110 }}>
          <VaultCard active={frame >= ARRIVE[3]} at={ARRIVE[3]} />
        </div>

        {/* Station 5 — SCRUB */}
        <div style={{ position: "absolute", left: FIRST_X + PITCH * 4 - 320, top: 70 }}>
          <ScrubCard active={frame >= ARRIVE[4]} at={ARRIVE[4]} />
        </div>

        {/* Station 6 — TEACH + SHIP */}
        <div style={{ position: "absolute", left: FIRST_X + PITCH * 5 - 450, top: 90 }}>
          <ShipCard active={frame >= ARRIVE[5]} at={ARRIVE[5]} />
        </div>
      </div>
    </SceneBg>
  );
};
