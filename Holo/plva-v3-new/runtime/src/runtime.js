export function selectWasmThreadCount({
  isolated = globalThis.crossOriginIsolated === true,
  requested = globalThis.__PLVA_WASM_THREADS__,
  hardwareConcurrency = globalThis.navigator?.hardwareConcurrency ?? 1,
} = {}) {
  if (!isolated) return 1;
  const hardware = finiteInteger(hardwareConcurrency, 1);
  const desired = finiteInteger(requested, Math.min(4, hardware));
  return Math.max(1, Math.min(8, hardware, desired));
}

let ortRunQueue = Promise.resolve();

// onnxruntime-web's WASM backend shares native state across sessions in a page.
// Concurrent session creation can initialize the allocator twice, while
// concurrent session.run calls can fail with "Session mismatch". Fetches still
// overlap, but session creation and inference enter the shared runtime in order.
export function runOrtSerialized(operation) {
  const result = ortRunQueue.then(operation, operation);
  ortRunQueue = result.then(
    () => undefined,
    () => undefined,
  );
  return result;
}

function finiteInteger(value, fallback) {
  const number = Number(value);
  return Number.isInteger(number) && number > 0 ? number : fallback;
}
