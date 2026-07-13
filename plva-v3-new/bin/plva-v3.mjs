#!/usr/bin/env node

import { randomBytes, createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import {
  access,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { createServer } from "node:http";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

import { verifyIntegrityManifest } from "./integrity.mjs";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const DIST = path.join(ROOT, "dist");
const VALID_PROFILES = new Set(["balanced", "high-recall"]);
const PNG_SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const BROWSER_SCHEMA_VERSION = 2;
const SNAPSHOT_NAME = "plva-visual-agpl-ats-v3-safe-head-v10";
const MODEL_HASHES = Object.freeze({
  visualDetector: "2ed3f25c9bee375dc1683cf0ffa2044374b6bc35dd89a465a7dce6451ce8b928",
  visualCheckpoint: "5fcd871b76a6fe456d004ef465ad5cf97616b9f475080711ba0d059a5807a9ad",
  visualThresholds: "c5c9892e0b87d5356645ac70090e22c21f95efd41d2391ad854c45ec87ecea60",
  visualRuntimePolicy: "c92fe90f228d7b47e7202fd09efbf77b75e8db5bc0837a6fc145a9774d0c102a",
  ocrDetector: "d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9",
  ocrRecognizer: "e8770c967605983d1570cdf5352041dfb68fa0c21664f49f47b155abd3e0e318",
  ocrDictionary: "5662df9d2d03f0e8ca0d3b0649d6acbab904b6a14b3d3521463c71c37c668ce3",
  rampart: "9f27d24949b0581701071ea5ef522d77ccd3f50c525cc91eac4d265b0fc2afe5",
});

try {
  await main();
} catch (error) {
  process.stderr.write(`PLVA v3 harness failed: ${error.message}\n`);
  process.exitCode = 1;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.help) {
    process.stdout.write(helpText());
    return;
  }

  const inputPath = path.resolve(options.input);
  const input = await readFile(inputPath);
  if (input.length === 0) throw new Error("input screenshot is empty");
  const inputDetails = await stat(inputPath);
  if (!inputDetails.isFile()) throw new Error("input screenshot is not a regular file");

  const outputPath = path.resolve(
    options.output ?? defaultOutputPath(inputPath),
  );
  const reportPath = path.resolve(options.report ?? `${outputPath}.json`);
  if (inputPath === outputPath) throw new Error("output must not overwrite the input screenshot");
  if (inputPath === reportPath) throw new Error("report must not overwrite the input screenshot");
  if (outputPath === reportPath) throw new Error("PNG output and JSON report must use different paths");
  if (path.extname(outputPath).toLowerCase() !== ".png") {
    throw new Error("output path must end in .png");
  }
  if (!options.force) {
    await ensureMissing(outputPath, "output PNG");
    await ensureMissing(reportPath, "JSON report");
  }

  process.stderr.write("Verifying the frozen PLVA v3 runtime…\n");
  await verifyIntegrityManifest(ROOT, "snapshot.json");
  await verifyIntegrityManifest(ROOT, "source-integrity.json");
  await verifyIntegrityManifest(ROOT, "dist-integrity.json");

  const chrome = await findChrome(options.chrome);
  const token = randomBytes(16).toString("hex");
  const exchange = createExchange();
  const server = createHarnessServer({ token, input, inputPath, exchange });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  const url = `http://127.0.0.1:${address.port}/?token=${token}&profile=${options.profile}&threads=${options.threads}`;
  const userData = await mkdtemp(path.join(os.tmpdir(), "plva-v3-chrome-"));
  let browser;

  try {
    process.stderr.write(`Running frozen visual + OCR + Rampart pipeline (${options.profile})…\n`);
    browser = launchChrome(chrome, url, userData);
    const outcome = await Promise.race([
      exchange.promise,
      browserExit(browser),
      timeoutAfter(options.timeoutMs),
    ]);

    if (!outcome.png.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) {
      throw new Error("browser returned a non-PNG output");
    }
    const browserReport = normalizeBrowserReport(outcome.report, options.profile);
    if (!browserReport.integrity.passed) {
      throw new Error("browser reported a redaction pixel-integrity failure");
    }
    if (!browserReport.network.localOnly) {
      throw new Error("browser reported a non-local inference resource");
    }

    const report = {
      schemaVersion: 2,
      pipeline: "plva-v3",
      snapshot: SNAPSHOT_NAME,
      releaseEligible: false,
      intendedUse: "development-only local screenshot redaction",
      profile: options.profile,
      input: {
        bytes: input.length,
        sha256: sha256(input),
        width: browserReport.dimensions.width,
        height: browserReport.dimensions.height,
      },
      output: {
        bytes: outcome.png.length,
        sha256: sha256(outcome.png),
        format: "image/png",
        width: browserReport.dimensions.width,
        height: browserReport.dimensions.height,
      },
      models: MODEL_HASHES,
      counts: browserReport.counts,
      semanticMode: browserReport.semanticMode,
      warnings: [
        ...browserReport.warnings,
        ...(options.profile === "balanced"
          ? ["balanced thresholds are retained for compatibility but are not calibrated for the v10 detector"]
          : []),
      ],
      degradations: browserReport.degradations,
      timings: browserReport.timings,
      regions: browserReport.regions,
      diagnostics: browserReport.diagnostics,
      integrity: browserReport.integrity,
      network: {
        ...browserReport.network,
        externalNetworkBlocked: true,
      },
      runtime: browserReport.runtime,
    };
    assertNoSensitiveReportFields(report);

    await mkdir(path.dirname(outputPath), { recursive: true });
    await mkdir(path.dirname(reportPath), { recursive: true });
    await writeFile(outputPath, outcome.png, { flag: options.force ? "w" : "wx" });
    await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, {
      flag: options.force ? "w" : "wx",
    });

    process.stdout.write(
      `${JSON.stringify({
        output: outputPath,
        report: reportPath,
        profile: options.profile,
        regions: report.counts.fused,
        semanticMode: report.semanticMode,
        totalMs: report.timings.totalMs,
        wasmThreads: report.runtime.effectiveWasmThreads,
        visualEngine: report.runtime.visualEngine,
        integrityPassed: report.integrity.passed,
        localOnly: report.network.localOnly,
      })}\n`,
    );
  } finally {
    exchange.reject(new Error("harness stopped before receiving a result"));
    await stopBrowser(browser);
    await closeServer(server);
    await rm(userData, {
      recursive: true,
      force: true,
      maxRetries: 8,
      retryDelay: 100,
    });
  }
}

async function stopBrowser(browser) {
  if (!browser || browser.exitCode !== null || browser.signalCode !== null) return;

  let resolveExit;
  const exited = new Promise((resolve) => {
    resolveExit = resolve;
    browser.once("exit", resolve);
  });

  browser.kill("SIGTERM");
  await Promise.race([exited, delay(2_000)]);
  if (browser.exitCode === null && browser.signalCode === null) {
    browser.kill("SIGKILL");
    await Promise.race([exited, delay(2_000)]);
  }
  browser.removeListener("exit", resolveExit);
}

function delay(milliseconds) {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, milliseconds);
    timer.unref?.();
  });
}

function createHarnessServer({ token, input, inputPath, exchange }) {
  const mime = inputMime(inputPath);
  const state = { png: null, report: null };
  return createServer(async (request, response) => {
    setSecurityHeaders(response);
    try {
      const url = new URL(request.url ?? "/", "http://127.0.0.1");
      if (process.env.PLVA_DEBUG === "1") {
        process.stderr.write(`[harness] ${request.method} ${url.pathname}\n`);
      }
      if (request.method === "GET" && url.pathname === `/__input/${token}`) {
        response.writeHead(200, {
          "Content-Type": mime,
          "Content-Length": input.length,
          "Cache-Control": "no-store",
        });
        response.end(input);
        return;
      }
      if (request.method === "POST" && url.pathname === `/__output/${token}`) {
        state.png = await readRequest(request, 300 * 1024 * 1024);
        response.writeHead(204).end();
        settleIfComplete(state, exchange);
        return;
      }
      if (request.method === "POST" && url.pathname === `/__report/${token}`) {
        const body = await readRequest(request, 8 * 1024 * 1024);
        state.report = JSON.parse(body.toString("utf8"));
        response.writeHead(204).end();
        settleIfComplete(state, exchange);
        return;
      }
      if (request.method === "POST" && url.pathname === `/__error/${token}`) {
        const body = await readRequest(request, 64 * 1024);
        const failure = JSON.parse(body.toString("utf8"));
        response.writeHead(204).end();
        exchange.reject(
          new Error(`browser pipeline error: ${String(failure.message ?? "unknown error")}`),
        );
        return;
      }
      if (request.method === "GET" || request.method === "HEAD") {
        await serveStatic(url.pathname, request.method, response);
        return;
      }
      response.writeHead(404).end("Not found");
    } catch (error) {
      if (!response.headersSent) response.writeHead(500);
      response.end("Harness request failed");
      exchange.reject(error);
    }
  });
}

async function serveStatic(pathname, method, response) {
  let relative;
  try {
    relative = pathname === "/" ? "index.html" : decodeURIComponent(pathname.slice(1));
  } catch {
    response.writeHead(400).end("Bad path");
    return;
  }
  const candidate = path.resolve(DIST, relative);
  const inside = path.relative(DIST, candidate);
  if (!inside || inside.startsWith("..") || path.isAbsolute(inside)) {
    response.writeHead(404).end("Not found");
    return;
  }
  let details;
  try {
    details = await stat(candidate);
  } catch {
    response.writeHead(404).end("Not found");
    return;
  }
  if (!details.isFile()) {
    response.writeHead(404).end("Not found");
    return;
  }
  response.writeHead(200, {
    "Content-Type": contentType(candidate),
    "Content-Length": details.size,
    "Cache-Control": "no-store",
  });
  if (method === "HEAD") response.end();
  else createReadStream(candidate).pipe(response);
}

function normalizeBrowserReport(value, expectedProfile) {
  if (!value || typeof value !== "object") throw new Error("browser report is missing");
  if (value.schemaVersion !== BROWSER_SCHEMA_VERSION) {
    throw new Error("browser report schema mismatch");
  }
  if (value.pipeline !== "plva-v3") throw new Error("browser report pipeline mismatch");
  if (value.snapshot !== SNAPSHOT_NAME) throw new Error("browser report snapshot mismatch");
  if (value.profile !== expectedProfile) throw new Error("browser report profile mismatch");
  const dimensions = {
    width: positiveInteger(value.dimensions?.width, "image width"),
    height: positiveInteger(value.dimensions?.height, "image height"),
  };
  const regionList = (regions) => {
    if (!Array.isArray(regions)) throw new Error("browser report has invalid regions");
    return regions.map(normalizeRegion);
  };
  const numberMap = (source, name) => {
    if (!source || typeof source !== "object") throw new Error(`browser report has invalid ${name}`);
    return Object.fromEntries(
      Object.entries(source).map(([key, number]) => [safeKey(key), finiteNumber(number, name)]),
    );
  };
  const stringList = (source, name) => {
    if (!Array.isArray(source)) throw new Error(`browser report has invalid ${name}`);
    return source.map((entry) => String(entry).slice(0, 500));
  };
  const integrity = {
    passed: value.integrity?.passed === true,
    dimensionsMatch: value.integrity?.dimensionsMatch === true,
    maskedPixels: finiteNumber(value.integrity?.maskedPixels, "masked pixels"),
    outsideChangedPixels: finiteNumber(
      value.integrity?.outsideChangedPixels,
      "outside changed pixels",
    ),
    insideWrongColorPixels: finiteNumber(
      value.integrity?.insideWrongColorPixels,
      "inside wrong-color pixels",
    ),
  };
  const counts = numberMap(value.counts, "counts");
  if (Object.hasOwn(counts, "ocrText")) {
    counts.ocrRegions = counts.ocrText;
    delete counts.ocrText;
  }
  const visualEngine = normalizeVisualEngine(value.runtime?.visualEngine);
  const visualTileCount = nonNegativeInteger(
    value.runtime?.visualTileCount,
    "visual tile count",
  );
  if (
    (visualEngine === "unavailable" && visualTileCount !== 0) ||
    (visualEngine !== "unavailable" && visualTileCount < 1)
  ) {
    throw new Error("browser report visual engine/tile count mismatch");
  }
  return {
    dimensions,
    counts,
    semanticMode: String(value.semanticMode ?? "unknown").slice(0, 80),
    warnings: stringList(value.warnings, "warnings"),
    degradations: stringList(value.degradations, "degradations"),
    timings: numberMap(value.timings, "timings"),
    regions: regionList(value.regions),
    diagnostics: {
      visual: regionList(value.diagnostics?.visual),
      ocrSemantic: regionList(value.diagnostics?.ocrSemantic),
      ocrUncertain: regionList(value.diagnostics?.ocrUncertain),
    },
    integrity,
    network: {
      localOnly: value.network?.localOnly === true,
      resourceCount: finiteNumber(value.network?.resourceCount, "resource count"),
    },
    runtime: {
      crossOriginIsolated: value.runtime?.crossOriginIsolated === true,
      requestedWasmThreads: positiveInteger(
        value.runtime?.requestedWasmThreads,
        "requested WASM threads",
      ),
      effectiveWasmThreads: positiveInteger(
        value.runtime?.effectiveWasmThreads,
        "effective WASM threads",
      ),
      visualEngine,
      visualTileCount,
      visualWorkerFallback: value.runtime?.visualWorkerFallback === true,
      visualRuntimePolicySha256: exactString(
        value.runtime?.visualRuntimePolicySha256,
        MODEL_HASHES.visualRuntimePolicy,
        "visual runtime policy",
      ),
    },
  };
}

function normalizeVisualEngine(value) {
  if (value !== "worker" && value !== "main" && value !== "unavailable") {
    throw new Error("browser report has invalid visual engine");
  }
  return value;
}

function exactString(value, expected, name) {
  if (value !== expected) throw new Error(`${name} mismatch`);
  return value;
}

function normalizeRegion(region) {
  if (!region || typeof region !== "object") throw new Error("invalid region in browser report");
  return {
    x1: finiteNumber(region.x1, "region x1"),
    y1: finiteNumber(region.y1, "region y1"),
    x2: finiteNumber(region.x2, "region x2"),
    y2: finiteNumber(region.y2, "region y2"),
    label: String(region.label ?? "UNKNOWN").slice(0, 120),
    labels: safeStringArray(region.labels, "region labels"),
    sources: safeStringArray(region.sources, "region sources"),
    score: finiteNumber(region.score, "region score"),
  };
}

function assertNoSensitiveReportFields(value, trail = []) {
  if (Array.isArray(value)) {
    value.forEach((entry, index) => assertNoSensitiveReportFields(entry, [...trail, index]));
    return;
  }
  if (!value || typeof value !== "object") return;
  const forbidden = /^(text|recognizedText|ocrText|value|inputPath|outputPath|fileName|filename)$/i;
  for (const [key, entry] of Object.entries(value)) {
    if (forbidden.test(key)) {
      throw new Error(`sensitive field is forbidden in report: ${[...trail, key].join(".")}`);
    }
    assertNoSensitiveReportFields(entry, [...trail, key]);
  }
}

function createExchange() {
  let resolve;
  let reject;
  let settled = false;
  const promise = new Promise((accept, decline) => {
    resolve = (value) => {
      if (!settled) {
        settled = true;
        accept(value);
      }
    };
    reject = (error) => {
      if (!settled) {
        settled = true;
        decline(error);
      }
    };
  });
  return { promise, resolve, reject };
}

function settleIfComplete(state, exchange) {
  if (state.png && state.report) exchange.resolve(state);
}

function launchChrome(executable, url, userData) {
  const debugArguments =
    process.env.PLVA_CHROME_DEBUG === "1" ? ["--enable-logging=stderr", "--v=1"] : [];
  return spawn(
    executable,
    [
      "--headless=new",
      `--user-data-dir=${userData}`,
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-background-networking",
      "--disable-component-update",
      "--disable-domain-reliability",
      "--disable-default-apps",
      "--disable-extensions",
      "--disable-features=OptimizationHints,MediaRouter,Translate",
      "--disable-client-side-phishing-detection",
      "--safebrowsing-disable-auto-update",
      "--disable-sync",
      "--metrics-recording-only",
      "--password-store=basic",
      "--use-mock-keychain",
      "--proxy-server=http://127.0.0.1:9",
      "--proxy-bypass-list=127.0.0.1",
      "--host-resolver-rules=MAP * 0.0.0.0, EXCLUDE 127.0.0.1",
      ...debugArguments,
      url,
    ],
    { stdio: ["ignore", "ignore", "pipe"] },
  );
}

function browserExit(browser) {
  let stderr = "";
  browser.stderr.on("data", (chunk) => {
    stderr = `${stderr}${chunk}`.slice(-8000);
    if (process.env.PLVA_CHROME_DEBUG === "1") process.stderr.write(chunk);
  });
  return new Promise((_, reject) => {
    browser.once("error", reject);
    browser.once("exit", (code, signal) => {
      reject(
        new Error(
          `headless Chrome exited before completion (${signal ?? code})${stderr ? `: ${stderr.trim()}` : ""}`,
        ),
      );
    });
  });
}

function timeoutAfter(milliseconds) {
  return new Promise((_, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`pipeline timed out after ${milliseconds} ms`)),
      milliseconds,
    );
    timer.unref?.();
  });
}

async function findChrome(explicit) {
  const candidates = [
    explicit,
    process.env.PLVA_CHROME,
    process.env.CHROME_PATH,
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ].filter(Boolean);
  for (const candidate of candidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Try the next known local browser.
    }
  }
  throw new Error("no local Chrome/Chromium found; pass --chrome /path/to/browser");
}

async function readRequest(request, maximum) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maximum) throw new Error("browser response exceeded the size limit");
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

function setSecurityHeaders(response) {
  response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  response.setHeader("Cross-Origin-Resource-Policy", "same-origin");
  response.setHeader(
    "Content-Security-Policy",
    "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'; worker-src 'self' blob:; img-src 'self' blob:; style-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
  );
  response.setHeader("X-Content-Type-Options", "nosniff");
}

function contentType(file) {
  if (file.endsWith(".html")) return "text/html; charset=utf-8";
  if (file.endsWith(".js")) return "text/javascript; charset=utf-8";
  if (file.endsWith(".json")) return "application/json; charset=utf-8";
  if (file.endsWith(".css")) return "text/css; charset=utf-8";
  if (file.endsWith(".wasm")) return "application/wasm";
  if (file.endsWith(".onnx")) return "application/octet-stream";
  if (file.endsWith(".txt")) return "text/plain; charset=utf-8";
  return "application/octet-stream";
}

function inputMime(file) {
  const extension = path.extname(file).toLowerCase();
  if (extension === ".png") return "image/png";
  if (extension === ".jpg" || extension === ".jpeg") return "image/jpeg";
  if (extension === ".webp") return "image/webp";
  return "application/octet-stream";
}

function parseArguments(arguments_) {
  const options = {
    profile: "high-recall",
    threads: 1,
    timeoutMs: 300_000,
    force: false,
    help: false,
  };
  const positional = [];
  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index];
    if (argument === "--help" || argument === "-h") options.help = true;
    else if (argument === "--force" || argument === "-f") options.force = true;
    else if (argument === "--output" || argument === "-o") {
      options.output = requiredValue(arguments_, ++index, argument);
    } else if (argument === "--report" || argument === "-r") {
      options.report = requiredValue(arguments_, ++index, argument);
    } else if (argument === "--profile" || argument === "-p") {
      options.profile = requiredValue(arguments_, ++index, argument);
    } else if (argument === "--chrome") {
      options.chrome = requiredValue(arguments_, ++index, argument);
    } else if (argument === "--timeout-ms") {
      options.timeoutMs = Number(requiredValue(arguments_, ++index, argument));
    } else if (argument.startsWith("-")) {
      throw new Error(`unknown option: ${argument}`);
    } else positional.push(argument);
  }
  if (options.help) return options;
  if (positional.length !== 1) throw new Error("provide exactly one input screenshot path");
  if (!VALID_PROFILES.has(options.profile)) {
    throw new Error("--profile must be balanced or high-recall");
  }
  if (!Number.isInteger(options.timeoutMs) || options.timeoutMs < 1_000) {
    throw new Error("--timeout-ms must be an integer of at least 1000");
  }
  options.input = positional[0];
  return options;
}

function requiredValue(arguments_, index, option) {
  const value = arguments_[index];
  if (!value || value.startsWith("-")) throw new Error(`${option} requires a value`);
  return value;
}

function defaultOutputPath(input) {
  const parsed = path.parse(input);
  return path.join(parsed.dir, `${parsed.name}.redacted.png`);
}

async function ensureMissing(file, description) {
  try {
    await access(file);
    throw new Error(`${description} already exists; pass --force to replace it: ${file}`);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
}

function positiveInteger(value, name) {
  const number = finiteNumber(value, name);
  if (!Number.isInteger(number) || number <= 0) throw new Error(`${name} must be positive`);
  return number;
}

function nonNegativeInteger(value, name) {
  const number = finiteNumber(value, name);
  if (!Number.isInteger(number)) throw new Error(`${name} must be an integer`);
  return number;
}

function finiteNumber(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) throw new Error(`${name} must be non-negative`);
  return number;
}

function safeStringArray(value, name) {
  if (!Array.isArray(value)) throw new Error(`${name} must be an array`);
  return value.map((entry) => String(entry).slice(0, 120));
}

function safeKey(value) {
  const key = String(value);
  if (!/^[A-Za-z][A-Za-z0-9]*$/.test(key)) throw new Error("unsafe key in browser report");
  return key;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function closeServer(server) {
  return new Promise((resolve) => server.close(resolve));
}

function helpText() {
  return `PLVA v3 screenshot harness

Usage:
  node bin/plva-v3.mjs <screenshot> [options]

Options:
  -o, --output <file>       Redacted PNG path (default: <input>.redacted.png)
  -r, --report <file>       Sanitized JSON report path (default: <output>.json)
  -p, --profile <profile>   balanced or high-recall (default: high-recall)
      --chrome <file>       Chrome/Chromium executable
      --timeout-ms <ms>     Pipeline timeout (default: 300000)
  -f, --force               Replace output/report if they already exist
  -h, --help                Show this help

The CLI uses only frozen local model assets, an ephemeral 127.0.0.1 server,
and headless Chrome. Recognized OCR text and filesystem paths are not written
to the JSON report.
`;
}
