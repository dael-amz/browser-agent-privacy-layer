import * as ort from "onnxruntime-web";

import { runOrtSerialized, selectWasmThreadCount } from "../runtime.js";
import { VISUAL_ARTIFACT_CONTRACT } from "./model-contract.js";
import {
  THRESHOLD_PROFILES,
  computeLetterboxTransform,
  decodeDetections,
} from "./postprocess.js";

const MODEL_SIZE = 640;
export const VISUAL_MODEL_URL = new URL(
  "../../training/artifacts/plva-visual-agpl-ats-v3/visual-safe-head-v10/detector.onnx",
  import.meta.url,
).href;
export const VISUAL_THRESHOLDS_URL = new URL(
  "../../training/artifacts/plva-visual-agpl-ats-v3/visual-safe-head-v10/safety-best-thresholds.json",
  import.meta.url,
).href;

let sessionPromise;
let thresholdPromise;
let enginePromise;
let detectorWorker;
let workerDisabled = false;
let workerRequestId = 0;
const workerRequests = new Map();

export async function warmVisualDetector(profile = "high-recall") {
  const [engine] = await Promise.all([getDetectorEngine(), getThresholds(profile)]);
  return engine.kind;
}

export async function detectSensitiveRegions(
  sourceCanvas,
  { profile = "high-recall", onStage = () => {} } = {},
) {
  if (!THRESHOLD_PROFILES[profile]) {
    throw new Error(`unknown detector sensitivity profile: ${profile}`);
  }

  onStage("Loading the 10.4 MB visual model locally…");
  const [engine, thresholds] = await Promise.all([
    getDetectorEngine(),
    getThresholds(profile),
  ]);
  onStage("Preparing screenshot pixels…");
  const { data, transform } = createInputData(sourceCanvas);
  onStage("Finding sensitive regions…");

  if (engine.kind === "worker") {
    try {
      const workerInput = data.buffer.slice(0);
      const result = await requestDetectorWorker(
        {
          type: "run",
          input: workerInput,
          modelUrl: VISUAL_MODEL_URL,
          wasmUrl: configuredOrtWasmUrl(),
        },
        [workerInput],
      );
      return {
        regions: decodeDetections(
          new Float32Array(result.output),
          result.dimensions,
          transform,
          { thresholds },
        ),
        engine: "worker",
        tileCount: 1,
        workerFallback: false,
      };
    } catch (error) {
      disableDetectorWorker(error);
      const mainEngine = { kind: "main", session: await getSession() };
      enginePromise = Promise.resolve(mainEngine);
      return {
        regions: await runMainDetector(mainEngine.session, data, transform, thresholds),
        engine: "main",
        tileCount: 1,
        workerFallback: true,
      };
    }
  }

  return {
    regions: await runMainDetector(engine.session, data, transform, thresholds),
    engine: "main",
    tileCount: 1,
    workerFallback: false,
  };
}

function getDetectorEngine() {
  if (!enginePromise) {
    enginePromise = (async () => {
      if (typeof Worker === "function" && !workerDisabled) {
        try {
          await requestDetectorWorker({
            type: "warm",
            modelUrl: VISUAL_MODEL_URL,
            wasmUrl: configuredOrtWasmUrl(),
          });
          return { kind: "worker" };
        } catch {
          disableDetectorWorker();
        }
      }
      return { kind: "main", session: await getSession() };
    })().catch((error) => {
      enginePromise = undefined;
      throw error;
    });
  }
  return enginePromise;
}

function requestDetectorWorker(message, transfer = []) {
  const worker = getDetectorWorker();
  if (!worker) return Promise.reject(new Error("detector worker is unavailable"));
  const id = ++workerRequestId;
  return new Promise((resolve, reject) => {
    workerRequests.set(id, { resolve, reject });
    try {
      worker.postMessage({ ...message, id }, transfer);
    } catch (error) {
      workerRequests.delete(id);
      reject(error);
    }
  });
}

function getDetectorWorker() {
  if (workerDisabled || typeof Worker !== "function") return null;
  if (!detectorWorker) {
    detectorWorker = new Worker(new URL("./detector-worker.js", import.meta.url), {
      type: "module",
      name: "plva-visual-detector",
    });
    detectorWorker.addEventListener("message", (event) => {
      const message = event.data;
      const pending = workerRequests.get(message?.id);
      if (!pending) return;
      workerRequests.delete(message.id);
      if (message.ok === true) pending.resolve(message);
      else pending.reject(new Error(String(message.error ?? "detector worker failed")));
    });
    detectorWorker.addEventListener("error", (event) => {
      event.preventDefault?.();
      disableDetectorWorker(new Error(event.message || "detector worker crashed"));
    });
  }
  return detectorWorker;
}

function disableDetectorWorker(error = new Error("detector worker disabled")) {
  workerDisabled = true;
  enginePromise = undefined;
  detectorWorker?.terminate();
  detectorWorker = undefined;
  for (const { reject } of workerRequests.values()) reject(error);
  workerRequests.clear();
}

async function getSession() {
  if (!sessionPromise) {
    ort.env.wasm.numThreads = selectWasmThreadCount();
    ort.env.wasm.proxy = false;
    const wasmUrl = configuredOrtWasmUrl();
    if (wasmUrl) ort.env.wasm.wasmPaths = { wasm: wasmUrl };
    sessionPromise = fetch(VISUAL_MODEL_URL)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`could not load visual model (${response.status})`);
        }
        return response.arrayBuffer();
      })
      .then(async (model) => {
        await assertSha256(model, VISUAL_ARTIFACT_CONTRACT.modelSha256, "visual model");
        return runOrtSerialized(() =>
          ort.InferenceSession.create(model, {
            executionProviders: ["wasm"],
            graphOptimizationLevel: "all",
          }),
        );
      })
      .catch((error) => {
        sessionPromise = undefined;
        throw error;
      });
  }
  return sessionPromise;
}

async function getThresholds(profile) {
  if (profile !== "high-recall") return THRESHOLD_PROFILES[profile];
  if (!thresholdPromise) {
    thresholdPromise = fetch(VISUAL_THRESHOLDS_URL)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`could not load visual thresholds (${response.status})`);
        }
        const bytes = await response.arrayBuffer();
        await assertSha256(
          bytes,
          VISUAL_ARTIFACT_CONTRACT.thresholdArtifactSha256,
          "visual threshold artifact",
        );
        return JSON.parse(new TextDecoder().decode(bytes));
      })
      .then((artifact) => {
        if (
          artifact?.schema_version !== 1 ||
          artifact.checkpoint_sha256 !== VISUAL_ARTIFACT_CONTRACT.checkpointSha256 ||
          artifact.threshold_profile_sha256 !==
            VISUAL_ARTIFACT_CONTRACT.thresholdProfileSha256
        ) {
          throw new Error("visual threshold artifact does not match the detector contract");
        }
        for (const [label, expected] of Object.entries(
          THRESHOLD_PROFILES["high-recall"],
        )) {
          if (artifact.threshold_profile?.[label] !== expected) {
            throw new Error(`visual threshold mismatch for ${label}`);
          }
        }
        return Object.freeze({ ...artifact.threshold_profile });
      })
      .catch((error) => {
        thresholdPromise = undefined;
        throw error;
      });
  }
  return thresholdPromise;
}

async function assertSha256(bytes, expected, name) {
  if (!globalThis.crypto?.subtle) {
    throw new Error(`cannot verify ${name}: Web Crypto is unavailable`);
  }
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  const actual = [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
  if (actual !== expected) throw new Error(`${name} SHA-256 mismatch`);
}

function createInputData(sourceCanvas) {
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

  const plane = MODEL_SIZE * MODEL_SIZE;
  const nchw = new Float32Array(plane * 3);
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
  for (let pixel = 0, offset = 0; pixel < plane; pixel += 1, offset += 4) {
    nchw[pixel] = rgba[offset] / 255;
    nchw[plane + pixel] = rgba[offset + 1] / 255;
    nchw[plane * 2 + pixel] = rgba[offset + 2] / 255;
  }

  return {
    data: nchw,
    transform,
  };
}

async function runMainDetector(session, data, transform, thresholds) {
  const tensor = new ort.Tensor("float32", data, [1, 3, MODEL_SIZE, MODEL_SIZE]);
  let output;
  try {
    const result = await runOrtSerialized(() =>
      session.run({ [session.inputNames[0]]: tensor }),
    );
    output = result[session.outputNames[0]];
    if (!output) throw new Error("the detector returned no output tensor");
    return decodeDetections(output.data, output.dims, transform, { thresholds });
  } finally {
    tensor.dispose?.();
    output?.dispose?.();
  }
}

function configuredOrtWasmUrl() {
  const value = globalThis.__PLVA_ORT_WASM_URL__;
  if (typeof value !== "string" || !value) return undefined;
  const url = new URL(value, globalThis.location?.href ?? import.meta.url);
  if (globalThis.location && url.origin !== globalThis.location.origin) {
    throw new Error("ONNX Runtime WASM must be served from the application origin");
  }
  return url.href;
}
