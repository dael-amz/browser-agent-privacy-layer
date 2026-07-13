#!/usr/bin/env node

import { createHash, randomBytes } from "node:crypto";
import { mkdir, open, rename, unlink } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  loadLock,
  parseCli,
  redirectHostAllowed,
  resolveAssetPath,
  resolveAssetRoot,
  selectAssets,
  verifyAssetFile,
} from "./model-lock.mjs";

const REPO_ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

function usage() {
  return `Usage: node scripts/fetch-models.mjs [options]

Options:
  --lock PATH            model lock (default: models.lock.json)
  --output DIR           asset root (default: lock asset_root)
  --asset NAME           fetch one named asset; repeatable
  --group NAME           fetch one asset group; repeatable
  --include-optional     include default_fetch=false assets
  --force                replace an already valid local file
`;
}

async function requestWithCheckedRedirects(source, asset) {
  let current = new URL(source);
  for (let redirects = 0; redirects <= 5; redirects += 1) {
    if (current.protocol !== "https:" || !redirectHostAllowed(current, asset)) {
      throw new Error(
        `redirect host is not permitted by the lock: ${current.hostname}`,
      );
    }
    const response = await fetch(current, {
      redirect: "manual",
      headers: { "user-agent": "plva-model-fetch/1" },
    });
    if (response.status >= 300 && response.status < 400) {
      const location = response.headers.get("location");
      if (!location)
        throw new Error(`redirect ${response.status} did not include Location`);
      current = new URL(location, current);
      continue;
    }
    if (!response.ok || !response.body) {
      throw new Error(`download failed with HTTP ${response.status}`);
    }
    const declaredLength = Number(response.headers.get("content-length"));
    if (Number.isFinite(declaredLength) && declaredLength > asset.bytes) {
      throw new Error(
        `server declared ${declaredLength} bytes, over locked ${asset.bytes}`,
      );
    }
    return { response, finalUrl: current };
  }
  throw new Error("too many redirects");
}

async function writeChunk(handle, chunk) {
  let offset = 0;
  while (offset < chunk.byteLength) {
    const { bytesWritten } = await handle.write(
      chunk,
      offset,
      chunk.byteLength - offset,
    );
    if (bytesWritten <= 0)
      throw new Error("short write while downloading asset");
    offset += bytesWritten;
  }
}

async function fetchAsset(name, asset, root, force) {
  const destination = resolveAssetPath(root, asset);
  const existing = await verifyAssetFile(name, asset, root);
  if (existing.ok && !force)
    return { ...existing, status: "verified-existing" };

  await mkdir(path.dirname(destination), { recursive: true });
  const temporary = `${destination}.partial-${process.pid}-${randomBytes(6).toString("hex")}`;
  const digest = createHash("sha256");
  let handle;
  try {
    const { response, finalUrl } = await requestWithCheckedRedirects(
      asset.url,
      asset,
    );
    handle = await open(temporary, "wx", 0o600);
    let bytes = 0;
    for await (const value of response.body) {
      const chunk = Buffer.from(value);
      bytes += chunk.byteLength;
      if (bytes > asset.bytes)
        throw new Error(`download exceeded locked size ${asset.bytes}`);
      digest.update(chunk);
      await writeChunk(handle, chunk);
    }
    await handle.sync();
    await handle.close();
    handle = undefined;
    const sha256 = digest.digest("hex");
    if (bytes !== asset.bytes)
      throw new Error(`expected ${asset.bytes} bytes, received ${bytes}`);
    if (sha256 !== asset.sha256) {
      throw new Error(`expected SHA-256 ${asset.sha256}, received ${sha256}`);
    }
    if (force)
      await unlink(destination).catch(
        (error) => error?.code === "ENOENT" || Promise.reject(error),
      );
    await rename(temporary, destination);
    return {
      name,
      path: destination,
      status: "downloaded",
      bytes,
      sha256,
      final_host: finalUrl.hostname,
    };
  } catch (error) {
    if (handle) await handle.close().catch(() => {});
    await unlink(temporary).catch(() => {});
    throw new Error(`${name}: ${error.message}`, { cause: error });
  }
}

async function main() {
  const options = parseCli(process.argv.slice(2));
  if (options.help) {
    process.stdout.write(usage());
    return;
  }
  const lock = await loadLock(
    options.lock ?? path.join(REPO_ROOT, "models.lock.json"),
  );
  const root = resolveAssetRoot(lock, options.output);
  const selected = selectAssets(lock.data, options);
  const results = [];
  for (const [name, asset] of selected) {
    results.push(await fetchAsset(name, asset, root, options.force));
  }
  process.stdout.write(
    `${JSON.stringify({ asset_root: root, assets: results }, null, 2)}\n`,
  );
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error.message}\n`);
  process.exitCode = 1;
});
