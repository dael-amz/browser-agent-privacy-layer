import React from "react";
import { Composition } from "remotion";
import { Main } from "./Main";
import { VIDEO } from "./theme";

export const Root: React.FC = () => (
  <Composition
    id="Demo"
    component={Main}
    durationInFrames={VIDEO.durationInFrames}
    fps={VIDEO.fps}
    width={VIDEO.width}
    height={VIDEO.height}
  />
);
