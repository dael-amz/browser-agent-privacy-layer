import * as ort from "onnxruntime-web";

import {
  THRESHOLD_PROFILES,
  computeLetterboxTransform,
  decodeDetections,
} from "./postprocess.js";

const MODEL_SIZE = 640;
export const VISUAL_MODEL_URL = new URL(
  "../../training/artifacts/plva-visual-agpl-test-v2/visual/detector.onnx",
  import.meta.url,
).href;

let sessionPromise;

export async function detectSensitiveRegions(
  sourceCanvas,
  { profile = "high-recall", onStage = () => {} } = {},
) {
  const thresholds = THRESHOLD_PROFILES[profile];
  if (!thresholds) throw new Error(`unknown detector sensitivity profile: ${profile}`);

  onStage("Loading the 10.4 MB visual model locally…");
  const session = await getSession();
  onStage("Preparing screenshot pixels…");
  const { tensor, transform } = createInputTensor(sourceCanvas);
  let output;

  try {
    onStage("Finding sensitive regions…");
    const result = await session.run({ [session.inputNames[0]]: tensor });
    output = result[session.outputNames[0]];
    if (!output) throw new Error("the detector returned no output tensor");
    return decodeDetections(output.data, output.dims, transform, { thresholds });
  } finally {
    tensor.dispose?.();
    output?.dispose?.();
  }
}

async function getSession() {
  if (!sessionPromise) {
    ort.env.wasm.numThreads = globalThis.crossOriginIsolated
      ? Math.max(1, Math.min(4, globalThis.navigator?.hardwareConcurrency ?? 1))
      : 1;
    ort.env.wasm.proxy = false;
    sessionPromise = fetch(VISUAL_MODEL_URL)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`could not load visual model (${response.status})`);
        }
        return response.arrayBuffer();
      })
      .then((model) =>
        ort.InferenceSession.create(model, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        }),
      )
      .catch((error) => {
        sessionPromise = undefined;
        throw error;
      });
  }
  return sessionPromise;
}

function createInputTensor(sourceCanvas) {
  const transform = computeLetterboxTransform(
    sourceCanvas.width,
    sourceCanvas.height,
    MODEL_SIZE,
  );
  const canvas = document.createElement("canvas");
  canvas.width = MODEL_SIZE;
  canvas.height = MODEL_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("this browser cannot create a 2D canvas");

  context.fillStyle = "rgb(114, 114, 114)";
  context.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  context.imageSmoothingEnabled = true;
  // Ultralytics letterboxing uses OpenCV INTER_LINEAR. Browser "high" quality
  // resampling is visibly softer and materially changes small-text confidence.
  context.imageSmoothingQuality = "low";
  context.drawImage(
    sourceCanvas,
    0,
    0,
    sourceCanvas.width,
    sourceCanvas.height,
    transform.padLeft,
    transform.padTop,
    transform.scaledWidth,
    transform.scaledHeight,
  );

  const rgba = context.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const plane = MODEL_SIZE * MODEL_SIZE;
  const nchw = new Float32Array(plane * 3);
  for (let pixel = 0, offset = 0; pixel < plane; pixel += 1, offset += 4) {
    nchw[pixel] = rgba[offset] / 255;
    nchw[plane + pixel] = rgba[offset + 1] / 255;
    nchw[plane * 2 + pixel] = rgba[offset + 2] / 255;
  }

  return {
    tensor: new ort.Tensor("float32", nchw, [1, 3, MODEL_SIZE, MODEL_SIZE]),
    transform,
  };
}
