import { env } from "@huggingface/transformers";
import transformersWasmUrl from "@plva-transformers-wasm";
import {
  createGuard,
  detectNer,
  loadNerClassifier,
} from "@nationaldesignstudio/rampart";

const LOCAL_MODEL = "semantic/rampart";
const MIN_SCORE = 0.4;

const classifierPromises = new Map();

// `/semantic/rampart` is served from the checked-in `models` directory by
// Vite. Remote loading is disabled so a typo or missing asset cannot silently
// turn a local-only redaction into a network request.
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = "/";
env.useBrowserCache = true;
env.backends.onnx.wasm.wasmPaths = { wasm: transformersWasmUrl };

export function getRampartClassifier(device = "wasm") {
  if (!classifierPromises.has(device)) {
    const promise = loadNerClassifier({
      model: LOCAL_MODEL,
      device,
      minScore: MIN_SCORE,
    }).catch((error) => {
      classifierPromises.delete(device);
      throw error;
    });
    classifierPromises.set(device, promise);
  }
  return classifierPromises.get(device);
}

export async function getRampartGuard(device = "wasm") {
  const classifier = await getRampartClassifier(device);
  // The model/session is shared, but every call receives a fresh placeholder
  // table so "new private session" truly forgets previous mappings.
  return createGuard({
    ner: (text) => detectNer(text, classifier, MIN_SCORE),
  });
}

export const RAMPART_MIN_SCORE = MIN_SCORE;
