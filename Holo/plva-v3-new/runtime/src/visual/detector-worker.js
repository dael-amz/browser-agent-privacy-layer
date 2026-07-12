import * as ort from "onnxruntime-web";

import { VISUAL_ARTIFACT_CONTRACT } from "./model-contract.js";

const MODEL_SIZE = 640;
let sessionPromise;
let activeModelUrl;
let activeWasmUrl;
let runQueue = Promise.resolve();

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

self.addEventListener("message", (event) => {
  const message = event.data;
  if (!message || !Number.isInteger(message.id)) return;
  runQueue = runQueue
    .then(() => handle(message))
    .catch((error) => {
      self.postMessage({
        id: message.id,
        ok: false,
        error: String(error?.message ?? error).slice(0, 500),
      });
    });
});

async function handle(message) {
  const modelUrl = validatedModelUrl(message.modelUrl);
  const wasmUrl = validatedOptionalUrl(message.wasmUrl, "ONNX Runtime WASM");
  const session = await getSession(modelUrl, wasmUrl);
  if (message.type === "warm") {
    self.postMessage({ id: message.id, ok: true });
    return;
  }
  if (message.type !== "run" || !(message.input instanceof ArrayBuffer)) {
    throw new Error("invalid detector worker request");
  }

  const values = new Float32Array(message.input);
  if (values.length !== 3 * MODEL_SIZE * MODEL_SIZE) {
    throw new Error("invalid detector input length");
  }
  const tensor = new ort.Tensor("float32", values, [1, 3, MODEL_SIZE, MODEL_SIZE]);
  let output;
  try {
    const result = await session.run({ [session.inputNames[0]]: tensor });
    output = result[session.outputNames[0]];
    if (!output) throw new Error("the detector returned no output tensor");
    const copied = Float32Array.from(output.data);
    self.postMessage(
      {
        id: message.id,
        ok: true,
        output: copied.buffer,
        dimensions: [...output.dims],
      },
      [copied.buffer],
    );
  } finally {
    tensor.dispose?.();
    output?.dispose?.();
  }
}

function getSession(modelUrl, wasmUrl) {
  if (activeModelUrl && activeModelUrl !== modelUrl) {
    throw new Error("detector worker model URL changed after initialization");
  }
  if (activeWasmUrl && activeWasmUrl !== wasmUrl) {
    throw new Error("detector worker WASM URL changed after initialization");
  }
  if (!sessionPromise) {
    activeModelUrl = modelUrl;
    activeWasmUrl = wasmUrl;
    if (wasmUrl) ort.env.wasm.wasmPaths = { wasm: wasmUrl };
    sessionPromise = fetch(modelUrl)
      .then((response) => {
        if (!response.ok) throw new Error(`could not load visual model (${response.status})`);
        return response.arrayBuffer();
      })
      .then(async (model) => {
        await assertSha256(model, VISUAL_ARTIFACT_CONTRACT.modelSha256);
        return ort.InferenceSession.create(model, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
      })
      .catch((error) => {
        sessionPromise = undefined;
        activeModelUrl = undefined;
        activeWasmUrl = undefined;
        throw error;
      });
  }
  return sessionPromise;
}

function validatedOptionalUrl(value, name) {
  if (value == null || value === "") return undefined;
  if (typeof value !== "string") throw new Error(`invalid ${name} URL`);
  const url = new URL(value, self.location.href);
  if (url.origin !== self.location.origin) {
    throw new Error(`${name} must be served from the harness origin`);
  }
  return url.href;
}

function validatedModelUrl(value) {
  if (typeof value !== "string" || !value) throw new Error("missing detector model URL");
  const url = new URL(value, self.location.href);
  if (url.origin !== self.location.origin) {
    throw new Error("detector model must be served from the harness origin");
  }
  return url.href;
}

async function assertSha256(bytes, expected) {
  const digest = await self.crypto.subtle.digest("SHA-256", bytes);
  const actual = [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
  if (actual !== expected) throw new Error("visual model SHA-256 mismatch");
}
