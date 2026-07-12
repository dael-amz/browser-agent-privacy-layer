import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const RUNTIME_VERSION = "1.27.0";

function parseArgs(argv) {
  const result = { backend: "node", artifactDir: null, output: null };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    const take = () => {
      const value = argv[index + 1];
      if (!value || value.startsWith("--"))
        throw new Error(`${argument} requires a value`);
      index += 1;
      return value;
    };
    if (argument === "--backend") result.backend = take();
    else if (argument === "--artifact-dir") result.artifactDir = take();
    else if (argument === "--output") result.output = take();
    else throw new Error(`unknown argument: ${argument}`);
  }
  if (!result.artifactDir) throw new Error("--artifact-dir is required");
  if (!result.output) throw new Error("--output is required");
  if (!new Set(["node", "wasm", "webgpu"]).has(result.backend)) {
    throw new Error("--backend must be node, wasm, or webgpu");
  }
  return result;
}

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function int64Tensor(ort, values) {
  return new ort.Tensor(
    "int64",
    BigInt64Array.from(values, (value) => BigInt(value)),
    [1, values.length],
  );
}

function argmaxRows(tensor, tokenCount) {
  const labelCount = tensor.dims.at(-1);
  const predicted = [];
  for (let token = 0; token < tokenCount; token += 1) {
    let bestClass = 0;
    let bestScore = Number.NEGATIVE_INFINITY;
    for (let label = 0; label < labelCount; label += 1) {
      const score = Number(tensor.data[token * labelCount + label]);
      if (score > bestScore) {
        bestScore = score;
        bestClass = label;
      }
    }
    predicted.push(bestClass);
  }
  return predicted;
}

async function runtime(backend) {
  if (backend === "node") {
    return {
      ort: await import("onnxruntime-node"),
      executionProviders: ["cpu"],
      packageName: "onnxruntime-node",
    };
  }
  if (backend === "webgpu") {
    if (!globalThis.navigator?.gpu) {
      throw new Error(
        "WebGPU is unavailable in this process; run in a WebGPU-enabled host",
      );
    }
    return {
      ort: await import("onnxruntime-web/webgpu"),
      executionProviders: ["webgpu"],
      packageName: "onnxruntime-web",
    };
  }
  const ort = await import("onnxruntime-web");
  ort.env.wasm.numThreads = 1;
  return {
    ort,
    executionProviders: ["wasm"],
    packageName: "onnxruntime-web",
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const artifactDir = path.resolve(args.artifactDir);
  const outputPath = path.resolve(args.output);
  const modelBytes = await readFile(path.join(artifactDir, "model.int8.onnx"));
  const modelSha256 = digest(modelBytes);
  const goldens = JSON.parse(
    await readFile(path.join(artifactDir, "golden_vectors.json"), "utf8"),
  );
  if (goldens.model_sha256 !== modelSha256)
    throw new Error("golden model hash mismatch");

  let existing = {};
  try {
    existing = JSON.parse(await readFile(outputPath, "utf8"));
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  if (existing.model_sha256 && existing.model_sha256 !== modelSha256) {
    throw new Error("output report belongs to a different model hash");
  }

  let backendResult;
  try {
    const selected = await runtime(args.backend);
    const session = await selected.ort.InferenceSession.create(modelBytes, {
      executionProviders: selected.executionProviders,
    });
    const inputNames = new Set(session.inputNames);
    const failures = [];
    for (const vector of goldens.vectors) {
      const feeds = {
        input_ids: int64Tensor(selected.ort, vector.input_ids),
        attention_mask: int64Tensor(selected.ort, vector.attention_mask),
      };
      if (inputNames.has("token_type_ids")) {
        feeds.token_type_ids = int64Tensor(selected.ort, vector.token_type_ids);
      }
      const output = await session.run(feeds);
      const logits = output[session.outputNames[0]];
      const predicted = argmaxRows(logits, vector.predicted_label_ids.length);
      if (
        predicted.some(
          (value, index) => value !== vector.predicted_label_ids[index],
        )
      ) {
        failures.push(vector.id);
      }
    }
    backendResult = {
      passed: failures.length === 0,
      package: selected.packageName,
      package_version: RUNTIME_VERSION,
      execution_providers_requested: selected.executionProviders,
      vectors_checked: goldens.vectors.length,
      failed_vector_ids: failures,
      ...(args.backend === "webgpu"
        ? {
            provider_trace: {
              verified: false,
              fallback_nodes: -1,
              method: "Attach an ONNX Runtime provider trace before release",
            },
          }
        : {}),
    };
  } catch (error) {
    backendResult = {
      passed: false,
      package: args.backend === "node" ? "onnxruntime-node" : "onnxruntime-web",
      package_version: RUNTIME_VERSION,
      vectors_checked: 0,
      error: String(error?.message ?? error),
    };
  }

  const report = {
    schema_version: 1,
    model_sha256: modelSha256,
    golden_vectors_sha256: digest(
      await readFile(path.join(artifactDir, "golden_vectors.json")),
    ),
    backends: { ...(existing.backends ?? {}), [args.backend]: backendResult },
  };
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  if (!backendResult.passed) process.exitCode = 1;
}

await main();
