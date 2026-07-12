import * as ort from "onnxruntime-web";

import { runOrtSerialized, selectWasmThreadCount } from "../runtime.js";
import {
  computeOcrDetectorSize,
  decodeCtc,
  extractOcrBoxes,
  parseOcrDictionary,
} from "./postprocess.js";

const DETECTOR_URL = "/ocr/ch_PP-OCRv4_det_mobile.onnx";
const RECOGNIZER_URL = "/ocr/en_PP-OCRv4_rec_mobile.onnx";
const DICTIONARY_URL = "/ocr/en_dict.txt";
export const OCR_ASSET_URLS = Object.freeze({
  detector: DETECTOR_URL,
  recognizer: RECOGNIZER_URL,
  dictionary: DICTIONARY_URL,
});
const RECOGNIZER_HEIGHT = 48;
const RECOGNIZER_BASE_WIDTH = 320;
const RECOGNIZER_MAX_WIDTH = 2048;
const RECOGNIZER_BATCH = 6;
const MIN_TEXT_CONFIDENCE = 0.5;

let detectorPromise;
let recognizerPromise;
let dictionaryPromise;

export async function warmRapidOcr() {
  await Promise.all([getDetector(), getRecognizer(), getDictionary()]);
}

export async function recognizeScreenshotText(
  sourceCanvas,
  { onStage = () => {} } = {},
) {
  onStage("OCR: loading the 12 MB local RapidOCR models…");
  const [detector, recognizer, dictionary] = await Promise.all([
    getDetector(),
    getRecognizer(),
    getDictionary(),
  ]);

  onStage("OCR: locating text on the screenshot…");
  const boxes = await detectTextBoxes(sourceCanvas, detector);
  if (boxes.length === 0) {
    return { regions: [], uncertainRegions: [], detectedCount: 0 };
  }

  onStage(`OCR: reading ${boxes.length} text region${boxes.length === 1 ? "" : "s"}…`);
  const recognition = await recognizeBoxes(
    sourceCanvas,
    boxes,
    recognizer,
    dictionary,
  );
  return { ...recognition, detectedCount: boxes.length };
}

async function detectTextBoxes(sourceCanvas, session) {
  const size = computeOcrDetectorSize(sourceCanvas.width, sourceCanvas.height);
  const tensor = createDetectorTensor(sourceCanvas, size.width, size.height);
  let output;
  try {
    const result = await runOrtSerialized(() =>
      session.run({ [session.inputNames[0]]: tensor }),
    );
    output = result[session.outputNames[0]];
    if (!output || output.dims.length !== 4) {
      throw new Error("the OCR detector returned an unexpected output");
    }
    const [, channels, height, width] = output.dims;
    if (channels !== 1) throw new Error("the OCR detector output is not a text map");
    return extractOcrBoxes(
      output.data,
      width,
      height,
      sourceCanvas.width,
      sourceCanvas.height,
    );
  } finally {
    tensor.dispose?.();
    output?.dispose?.();
  }
}

async function recognizeBoxes(sourceCanvas, boxes, session, dictionary) {
  const ordered = boxes
    .map((box, index) => ({
      box,
      index,
      ratio: Math.max(1, (box.x2 - box.x1) / (box.y2 - box.y1)),
    }))
    .sort((left, right) => left.ratio - right.ratio);
  const results = new Array(boxes.length);
  const uncertain = new Array(boxes.length);

  for (let offset = 0; offset < ordered.length; offset += RECOGNIZER_BATCH) {
    const batch = ordered.slice(offset, offset + RECOGNIZER_BATCH);
    const widestRatio = Math.max(
      RECOGNIZER_BASE_WIDTH / RECOGNIZER_HEIGHT,
      ...batch.map((entry) => entry.ratio),
    );
    const targetWidth = Math.min(
      RECOGNIZER_MAX_WIDTH,
      Math.max(RECOGNIZER_BASE_WIDTH, Math.floor(RECOGNIZER_HEIGHT * widestRatio)),
    );
    const tensor = createRecognizerTensor(sourceCanvas, batch, targetWidth);
    let output;
    try {
      const inference = await runOrtSerialized(() =>
        session.run({ [session.inputNames[0]]: tensor }),
      );
      output = inference[session.outputNames[0]];
      if (!output || output.dims.length !== 3) {
        throw new Error("the OCR recognizer returned an unexpected output");
      }
      for (let sample = 0; sample < batch.length; sample += 1) {
        const decoded = decodeCtc(output.data, output.dims, dictionary, sample);
        if (decoded.confidence >= MIN_TEXT_CONFIDENCE && decoded.text.trim()) {
          const entry = batch[sample];
          results[entry.index] = {
            ...entry.box,
            text: decoded.text,
            ocrConfidence: decoded.confidence,
          };
        } else {
          const entry = batch[sample];
          uncertain[entry.index] = {
            ...entry.box,
            label: "UNREADABLE",
            labels: ["UNREADABLE"],
            sources: ["OCR+UNCERTAIN"],
            score: Math.max(entry.box.detectorScore ?? 0, decoded.confidence),
            ocrConfidence: decoded.confidence,
          };
        }
      }
    } finally {
      tensor.dispose?.();
      output?.dispose?.();
    }
  }

  return {
    regions: results.filter(Boolean),
    uncertainRegions: uncertain.filter(Boolean),
  };
}

function createDetectorTensor(sourceCanvas, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("this browser cannot create a 2D OCR canvas");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "low";
  context.drawImage(sourceCanvas, 0, 0, width, height);
  const rgba = context.getImageData(0, 0, width, height).data;
  const plane = width * height;
  const data = new Float32Array(plane * 3);
  for (let pixel = 0, offset = 0; pixel < plane; pixel += 1, offset += 4) {
    data[pixel] = rgba[offset + 2] / 127.5 - 1;
    data[plane + pixel] = rgba[offset + 1] / 127.5 - 1;
    data[plane * 2 + pixel] = rgba[offset] / 127.5 - 1;
  }
  return new ort.Tensor("float32", data, [1, 3, height, width]);
}

function createRecognizerTensor(sourceCanvas, batch, targetWidth) {
  const plane = RECOGNIZER_HEIGHT * targetWidth;
  const data = new Float32Array(batch.length * plane * 3);
  const canvas = document.createElement("canvas");
  canvas.height = RECOGNIZER_HEIGHT;

  for (let sample = 0; sample < batch.length; sample += 1) {
    const { box, ratio } = batch[sample];
    const resizedWidth = Math.min(
      targetWidth,
      Math.max(1, Math.ceil(RECOGNIZER_HEIGHT * ratio)),
    );
    canvas.width = resizedWidth;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("this browser cannot create an OCR crop canvas");
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "low";
    context.drawImage(
      sourceCanvas,
      box.x1,
      box.y1,
      box.x2 - box.x1,
      box.y2 - box.y1,
      0,
      0,
      resizedWidth,
      RECOGNIZER_HEIGHT,
    );
    const rgba = context.getImageData(0, 0, resizedWidth, RECOGNIZER_HEIGHT).data;
    const sampleOffset = sample * plane * 3;
    for (let y = 0; y < RECOGNIZER_HEIGHT; y += 1) {
      for (let x = 0; x < resizedWidth; x += 1) {
        const rgbaOffset = (y * resizedWidth + x) * 4;
        const target = y * targetWidth + x;
        data[sampleOffset + target] = rgba[rgbaOffset + 2] / 127.5 - 1;
        data[sampleOffset + plane + target] = rgba[rgbaOffset + 1] / 127.5 - 1;
        data[sampleOffset + plane * 2 + target] = rgba[rgbaOffset] / 127.5 - 1;
      }
    }
  }

  return new ort.Tensor("float32", data, [
    batch.length,
    3,
    RECOGNIZER_HEIGHT,
    targetWidth,
  ]);
}

function getDetector() {
  detectorPromise ??= createSession(DETECTOR_URL, "OCR detector").catch((error) => {
    detectorPromise = undefined;
    throw error;
  });
  return detectorPromise;
}

function getRecognizer() {
  recognizerPromise ??= createSession(RECOGNIZER_URL, "OCR recognizer").catch((error) => {
    recognizerPromise = undefined;
    throw error;
  });
  return recognizerPromise;
}

function getDictionary() {
  dictionaryPromise ??= fetch(DICTIONARY_URL)
    .then((response) => {
      if (!response.ok) throw new Error(`could not load OCR dictionary (${response.status})`);
      return response.text();
    })
    .then(parseOcrDictionary)
    .catch((error) => {
      dictionaryPromise = undefined;
      throw error;
    });
  return dictionaryPromise;
}

function createSession(url, name) {
  configureOrt();
  return fetch(url)
    .then((response) => {
      if (!response.ok) throw new Error(`could not load ${name} (${response.status})`);
      return response.arrayBuffer();
    })
    .then((model) =>
      runOrtSerialized(() =>
        ort.InferenceSession.create(model, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        }),
      ),
    );
}

function configureOrt() {
  ort.env.wasm.numThreads = selectWasmThreadCount();
  ort.env.wasm.proxy = false;
  const value = globalThis.__PLVA_ORT_WASM_URL__;
  if (typeof value === "string" && value) {
    const url = new URL(value, globalThis.location?.href ?? import.meta.url);
    if (globalThis.location && url.origin !== globalThis.location.origin) {
      throw new Error("ONNX Runtime WASM must be served from the application origin");
    }
    ort.env.wasm.wasmPaths = { wasm: url.href };
  }
}
