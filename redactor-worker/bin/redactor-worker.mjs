#!/usr/bin/env node

import { randomBytes } from "node:crypto";
import { createReadStream } from "node:fs";
import { access, mkdtemp, rm, stat } from "node:fs/promises";
import { createServer } from "node:http";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const DIST = path.join(ROOT, "dist");
const MAX_INPUT = 100 * 1024 * 1024;
const MAX_OUTPUT = 300 * 1024 * 1024;
const backend = parseBackend(process.argv.slice(2));
const baseline = path.resolve(process.env.PLVA_BASELINE_ROOT ?? "../plva-v2-baseline");
const token = randomBytes(16).toString("hex");
const jobs = [];
const active = new Map();
const waiters = [];
let browser;
let userData;
let server;
let stopping = false;

try {
  await access(path.join(DIST, "index.html"));
  await access(path.join(baseline, "snapshot.json"));
  const chrome = await findChrome();
  server = createWorkerServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  userData = await mkdtemp(path.join(os.tmpdir(), "plva-accelerated-chrome-"));
  const address = server.address();
  const debug = process.env.PLVA_WORKER_DEBUG === "1" ? "&debug=1" : "";
  const url = `http://127.0.0.1:${address.port}/?backend=${backend}&token=${token}${debug}`;
  browser = launchChrome(chrome, url, userData);
  browser.once("exit", () => {
    if (!stopping) fatal("BrowserExit");
  });
  readProtocolInput();
} catch (error) {
  fatal(error?.name ?? "StartupError");
}

function createWorkerServer() {
  return createServer(async (request, response) => {
    setSecurityHeaders(response);
    try {
      const url = new URL(request.url ?? "/", "http://127.0.0.1");
      if (url.pathname.startsWith("/__") && url.searchParams.get("token") !== token) {
        response.writeHead(403).end();
        return;
      }
      if (request.method === "POST" && url.pathname === "/__ready") {
        const details = JSON.parse((await readRequest(request, 64 * 1024)).toString("utf8"));
        writeProtocol({
          ready: true,
          backend: String(details.backend ?? "unknown"),
          threaded: details.crossOriginIsolated === true,
        });
        response.writeHead(204).end();
        return;
      }
      if (request.method === "POST" && url.pathname === "/__fatal") {
        response.writeHead(204).end();
        fatal("BrowserWorkerError");
        return;
      }
      if (request.method === "GET" && url.pathname === "/__job") {
        const job = jobs.shift();
        if (job) {
          active.set(job.id, job);
          sendJson(response, 200, { id: job.id, profile: job.profile });
        } else {
          const timer = setTimeout(() => {
            const index = waiters.indexOf(deliver);
            if (index >= 0) waiters.splice(index, 1);
            response.writeHead(204).end();
          }, 1000);
          const deliver = (next) => {
            clearTimeout(timer);
            active.set(next.id, next);
            sendJson(response, 200, { id: next.id, profile: next.profile });
          };
          waiters.push(deliver);
        }
        return;
      }
      const inputId = routeId(url.pathname, "/__input/");
      if (request.method === "GET" && inputId) {
        const job = requireJob(inputId);
        response.writeHead(200, {
          "Content-Type": "image/png",
          "Content-Length": job.input.length,
          "Cache-Control": "no-store",
        });
        response.end(job.input);
        return;
      }
      const outputId = routeId(url.pathname, "/__output/");
      if (request.method === "POST" && outputId) {
        const job = requireJob(outputId);
        job.output = await readRequest(request, MAX_OUTPUT);
        response.writeHead(204).end();
        settle(job);
        return;
      }
      const reportId = routeId(url.pathname, "/__report/");
      if (request.method === "POST" && reportId) {
        const job = requireJob(reportId);
        job.report = JSON.parse((await readRequest(request, 2 * 1024 * 1024)).toString("utf8"));
        response.writeHead(204).end();
        settle(job);
        return;
      }
      const errorId = routeId(url.pathname, "/__error/");
      if (request.method === "POST" && errorId) {
        const job = requireJob(errorId);
        const failure = JSON.parse((await readRequest(request, 64 * 1024)).toString("utf8"));
        response.writeHead(204).end();
        failJob(job, String(failure.error ?? "WorkerFrameError"));
        return;
      }
      if (request.method === "GET" || request.method === "HEAD") {
        await serveStatic(url.pathname, request.method, response);
        return;
      }
      response.writeHead(404).end();
    } catch (error) {
      if (!response.headersSent) response.writeHead(500);
      response.end();
    }
  });
}

function readProtocolInput() {
  let buffered = "";
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", (chunk) => {
    buffered += chunk;
    let newline;
    while ((newline = buffered.indexOf("\n")) >= 0) {
      const line = buffered.slice(0, newline);
      buffered = buffered.slice(newline + 1);
      if (!line) continue;
      try {
        const request = JSON.parse(line);
        const input = Buffer.from(String(request.image ?? ""), "base64");
        if (!request.id || input.length === 0 || input.length > MAX_INPUT) {
          throw new Error("invalid protocol request");
        }
        const job = {
          id: String(request.id),
          profile: String(request.profile),
          input,
          output: null,
          report: null,
        };
        const waiter = waiters.shift();
        if (waiter) waiter(job);
        else jobs.push(job);
      } catch (error) {
        writeProtocol({ id: "", ok: false, error: "ProtocolError" });
      }
    }
  });
  process.stdin.once("end", shutdown);
}

function settle(job) {
  if (!job.output || !job.report) return;
  if (job.output.length < 8 || !job.output.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10]))) {
    failJob(job, "InvalidWorkerOutput");
    return;
  }
  active.delete(job.id);
  writeProtocol({
    id: job.id,
    ok: true,
    image: job.output.toString("base64"),
    backend: String(job.report.backend ?? "unknown"),
    counts: job.report.counts ?? {},
    timings: job.report.timings ?? {},
  });
  job.input = null;
  job.output = null;
  job.report = null;
}

function failJob(job, error) {
  active.delete(job.id);
  writeProtocol({ id: job.id, ok: false, error: String(error).slice(0, 80) });
  job.input = null;
  job.output = null;
  job.report = null;
}

function requireJob(id) {
  const job = active.get(id);
  if (!job) throw new Error("unknown job");
  return job;
}

function routeId(pathname, prefix) {
  if (!pathname.startsWith(prefix)) return null;
  return decodeURIComponent(pathname.slice(prefix.length));
}

async function serveStatic(pathname, method, response) {
  let relative;
  try {
    relative = pathname === "/" ? "index.html" : decodeURIComponent(pathname.slice(1));
  } catch {
    response.writeHead(400).end();
    return;
  }
  const candidate = path.resolve(DIST, relative);
  const inside = path.relative(DIST, candidate);
  if (!inside || inside.startsWith("..") || path.isAbsolute(inside)) {
    response.writeHead(404).end();
    return;
  }
  let details;
  try {
    details = await stat(candidate);
  } catch {
    response.writeHead(404).end();
    return;
  }
  if (!details.isFile()) {
    response.writeHead(404).end();
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

function launchChrome(executable, url, profile) {
  const debugArguments = process.env.PLVA_WORKER_DEBUG === "1"
    ? ["--enable-logging=stderr", "--v=1"]
    : [];
  return spawn(executable, [
    "--headless=new",
    `--user-data-dir=${profile}`,
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
  ], {
    stdio: [
      "ignore",
      "ignore",
      process.env.PLVA_WORKER_DEBUG === "1" ? "inherit" : "ignore",
    ],
  });
}

async function findChrome() {
  const candidates = process.platform === "darwin"
    ? [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
      ]
    : ["/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"];
  for (const candidate of candidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {}
  }
  throw new Error("Chrome not found");
}

async function readRequest(request, maximum) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maximum) throw new Error("request too large");
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

function sendJson(response, status, value) {
  const body = Buffer.from(JSON.stringify(value));
  response.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": body.length,
    "Cache-Control": "no-store",
  });
  response.end(body);
}

function writeProtocol(value) {
  process.stdout.write(`${JSON.stringify(value)}\n`);
}

function contentType(file) {
  if (file.endsWith(".html")) return "text/html; charset=utf-8";
  if (file.endsWith(".js")) return "text/javascript; charset=utf-8";
  if (file.endsWith(".wasm")) return "application/wasm";
  if (file.endsWith(".onnx")) return "application/octet-stream";
  if (file.endsWith(".json")) return "application/json";
  if (file.endsWith(".txt")) return "text/plain; charset=utf-8";
  return "application/octet-stream";
}

function parseBackend(arguments_) {
  const index = arguments_.indexOf("--backend");
  const value = index >= 0 ? arguments_[index + 1] : "auto";
  if (!["auto", "webgpu", "wasm"].includes(value)) throw new Error("invalid backend");
  return value;
}

function fatal(error) {
  writeProtocol({ ready: false, ok: false, error: String(error).slice(0, 80) });
  shutdown().finally(() => process.exit(1));
}

async function shutdown() {
  if (stopping) return;
  stopping = true;
  if (browser && browser.exitCode === null) browser.kill("SIGTERM");
  if (server) await new Promise((resolve) => server.close(resolve));
  if (userData) await rm(userData, { recursive: true, force: true });
}

process.once("SIGTERM", () => shutdown().finally(() => process.exit(0)));
process.once("SIGINT", () => shutdown().finally(() => process.exit(0)));
