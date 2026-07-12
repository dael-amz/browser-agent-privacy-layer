import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";

const inter = loadInter();
const mono = loadMono();

export const FONT = inter.fontFamily;
export const MONO = mono.fontFamily;

// Matches the Holo app's design tokens (Holo/src/plva_proxy/demo_ui.html)
export const C = {
  paper: "#ffffff",
  paperAlt: "#f3f3f3",
  inverse: "#181818",
  inverseDeep: "#0c0c0c",
  ink: "#0c0c0c",
  gray: "#868686",
  grayLight: "#b4b4b4",
  borderSoft: "rgba(12,12,12,.1)",
  borderFade: "rgba(12,12,12,.05)",
  green: "#0a8a4f",
  greenSoft: "rgba(10,138,79,.12)",
  red: "#d13438",
  redSoft: "rgba(209,52,56,.12)",
  amber: "#b97e0f",
  white: "#ffffff",
  whiteDim: "rgba(255,255,255,.64)",
  whiteFaint: "rgba(255,255,255,.28)",
} as const;

export const VIDEO = {
  width: 1920,
  height: 1080,
  fps: 30,
  durationInFrames: 3600, // 120s
} as const;

export const MARGIN = 120;

// Type scale
export const T = {
  hero: { fontFamily: FONT, fontSize: 118, fontWeight: 700, letterSpacing: "-0.035em", lineHeight: 1.04 },
  h1: { fontFamily: FONT, fontSize: 84, fontWeight: 700, letterSpacing: "-0.03em", lineHeight: 1.06 },
  h2: { fontFamily: FONT, fontSize: 56, fontWeight: 650, letterSpacing: "-0.025em", lineHeight: 1.1 },
  h3: { fontFamily: FONT, fontSize: 40, fontWeight: 600, letterSpacing: "-0.02em", lineHeight: 1.15 },
  body: { fontFamily: FONT, fontSize: 32, fontWeight: 450, letterSpacing: "-0.01em", lineHeight: 1.35 },
  small: { fontFamily: FONT, fontSize: 24, fontWeight: 500, letterSpacing: "0em", lineHeight: 1.3 },
  label: { fontFamily: FONT, fontSize: 21, fontWeight: 600, letterSpacing: "0.14em", textTransform: "uppercase" as const },
  mono: { fontFamily: MONO, fontSize: 26, fontWeight: 500, letterSpacing: "-0.01em" },
} as const;
