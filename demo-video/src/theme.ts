import { loadFont as loadInter } from "@remotion/google-fonts/Inter";

const inter = loadInter();

// Exact font stacks from the Holo frontend (Holo/src/plva_proxy/demo_ui.html).
// ABC Diatype is used when installed locally; Inter (loaded) is the site's fallback.
export const FONT = `"ABC Diatype","ABC Diatype Variable",Diatype,${inter.fontFamily},-apple-system,"SF Pro Text","Helvetica Neue","Segoe UI",Arial,sans-serif`;
export const MONO = `ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace`;

// Design tokens copied 1:1 from the site's :root
export const C = {
  paper: "#ffffff", // --back-primary
  paperAlt: "#f3f3f3", // --back-secondary
  inverse: "#181818", // --back-inverse
  inverseDeep: "#181818",
  ink: "#0c0c0c", // --fore-primary
  gray: "#868686", // --fore-secondary
  grayLight: "#b4b4b4",
  borderSoft: "rgba(12,12,12,.1)", // --border-soft
  borderFade: "rgba(12,12,12,.05)", // --border-fade
  green: "#0a8a4f", // --ok
  greenSoft: "rgba(10,138,79,.1)",
  red: "#d13438", // --bad
  redSoft: "rgba(209,52,56,.1)",
  amber: "#b97e0f", // --warn
  white: "#ffffff",
  whiteDim: "rgba(255,255,255,.6)", // .runcard.active small
  whiteFaint: "rgba(255,255,255,.28)",
} as const;

// Shadow tiers copied from the site
export const SHADOW = {
  card: "0 1px 2px rgba(12,12,12,.03)", // .card
  float: "0 16px 48px rgba(12,12,12,.06)", // .composer
  menu: "0 12px 32px rgba(12,12,12,.1)", // .modelmenu
} as const;

export const RADIUS = { card: 16, full: 999, well: 12, token: 6 } as const;

export const VIDEO = {
  width: 1920,
  height: 1080,
  fps: 30,
  durationInFrames: 3600, // 120s
} as const;

export const MARGIN = 120;

// Type scale: the site sets display text at weight 400 with tight negative
// tracking (h1: -.03em) and reserves 500 for emphasis/labels. Nothing heavier.
export const T = {
  hero: { fontFamily: FONT, fontSize: 112, fontWeight: 400, letterSpacing: "-0.03em", lineHeight: 1.06 },
  h1: { fontFamily: FONT, fontSize: 82, fontWeight: 400, letterSpacing: "-0.03em", lineHeight: 1.06 },
  h2: { fontFamily: FONT, fontSize: 56, fontWeight: 400, letterSpacing: "-0.0275em", lineHeight: 1.1 },
  h3: { fontFamily: FONT, fontSize: 40, fontWeight: 400, letterSpacing: "-0.02em", lineHeight: 1.2 },
  body: { fontFamily: FONT, fontSize: 32, fontWeight: 400, letterSpacing: "-0.006em", lineHeight: 1.5 },
  small: { fontFamily: FONT, fontSize: 24, fontWeight: 400, letterSpacing: "0em", lineHeight: 1.4 },
  label: { fontFamily: FONT, fontSize: 22, fontWeight: 500, letterSpacing: "0.08em", textTransform: "uppercase" as const },
  mono: { fontFamily: MONO, fontSize: 26, fontWeight: 400, letterSpacing: "0em" },
} as const;
