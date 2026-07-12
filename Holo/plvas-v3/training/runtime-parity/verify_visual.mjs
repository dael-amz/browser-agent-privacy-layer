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

function deterministicImage(ort, shape) {
  const size = shape.reduce((product, value) => product * value, 1);
  const values = new Float32Array(size);
  for (let index = 0; index < size; index += 1) {
    values[index] = (index % 256) / 255;
  }
  return new ort.Tensor("float32", values, shape);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const artifactDir = path.resolve(args.artifactDir);
  const outputPath = path.resolve(args.output);
  const modelBytes = await readFile(path.join(artifactDir, "detector.onnx"));
  const modelSha256 = digest(modelBytes);
  const goldenBytes = await readFile(
    path.join(artifactDir, "detector_goldens.json"),
  );
  const goldens = JSON.parse(goldenBytes.toString("utf8"));
  if (goldens.model_sha256 !== modelSha256) {
    throw new Error("golden model hash mismatch");
  }

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
    const result = await session.run({
      [goldens.input.name]: deterministicImage(selected.ort, goldens.input.shape),
    });
    const tensor = result[goldens.output.name];
    const expectedShape = goldens.output.shape.join(",");
    if (tensor.dims.join(",") !== expectedShape) {
      throw new Error(
        `output shape ${tensor.dims.join(",")} differs from ${expectedShape}`,
      );
    }
    let maximumAbsoluteError = 0;
    let maximumRelativeError = 0;
    const failures = [];
    for (const sample of goldens.output.samples) {
      const actual = Number(tensor.data[sample.flat_index]);
      const expected = Number(sample.value);
      const absoluteError = Math.abs(actual - expected);
      const relativeError = absoluteError / Math.max(Math.abs(expected), 1e-12);
      maximumAbsoluteError = Math.max(maximumAbsoluteError, absoluteError);
      maximumRelativeError = Math.max(maximumRelativeError, relativeError);
      const allowed =
        goldens.tolerance.absolute +
        goldens.tolerance.relative * Math.abs(expected);
      if (!Number.isFinite(actual) || absoluteError > allowed) {
        failures.push({
          flat_index: sample.flat_index,
          expected,
          actual,
          absolute_error: absoluteError,
        });
      }
    }
    backendResult = {
      passed: failures.length === 0,
      package: selected.packageName,
      package_version: RUNTIME_VERSION,
      execution_providers_requested: selected.executionProviders,
      vectors_checked: goldens.output.samples.length,
      maximum_absolute_error: maximumAbsoluteError,
      maximum_relative_error: maximumRelativeError,
      failed_samples: failures.slice(0, 10),
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
    golden_vectors_sha256: digest(goldenBytes),
    backends: { ...(existing.backends ?? {}), [args.backend]: backendResult },
  };
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  if (!backendResult.passed) process.exitCode = 1;
}

await main();
