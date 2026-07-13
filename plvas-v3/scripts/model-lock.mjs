import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";

export function parseCli(argv) {
  const values = {
    assets: [],
    groups: [],
    force: false,
    includeOptional: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const take = () => {
      const value = argv[index + 1];
      if (!value || value.startsWith("--")) {
        throw new Error(`${arg} requires a value`);
      }
      index += 1;
      return value;
    };
    if (arg === "--lock") values.lock = take();
    else if (arg === "--output") values.output = take();
    else if (arg === "--asset") values.assets.push(take());
    else if (arg === "--group") values.groups.push(take());
    else if (arg === "--force") values.force = true;
    else if (arg === "--include-optional") values.includeOptional = true;
    else if (arg === "--help" || arg === "-h") values.help = true;
    else throw new Error(`unknown argument: ${arg}`);
  }
  return values;
}

export async function loadLock(lockPath) {
  const absolute = path.resolve(lockPath);
  const parsed = JSON.parse(await readFile(absolute, "utf8"));
  if (parsed.schema_version !== 2 || typeof parsed.assets !== "object") {
    throw new Error(
      "models.lock.json must use schema_version 2 with an assets object",
    );
  }
  for (const [name, asset] of Object.entries(parsed.assets)) {
    if (
      !asset.path ||
      !asset.url ||
      !asset.sha256 ||
      !Number.isSafeInteger(asset.bytes)
    ) {
      throw new Error(`asset ${name} is missing path, URL, bytes, or SHA-256`);
    }
    if (!/^[a-f0-9]{64}$/.test(asset.sha256)) {
      throw new Error(`asset ${name} has an invalid SHA-256`);
    }
    const url = new URL(asset.url);
    if (url.protocol !== "https:")
      throw new Error(`asset ${name} must use HTTPS`);
    const allowed = new Set(asset.allowed_redirect_hosts ?? []);
    if (!allowed.has(url.hostname)) {
      throw new Error(
        `asset ${name} must permit its origin host ${url.hostname}`,
      );
    }
  }
  return { path: absolute, data: parsed };
}

export function selectAssets(lock, options = {}) {
  const requestedAssets = new Set(options.assets ?? []);
  const requestedGroups = new Set(options.groups ?? []);
  for (const name of requestedAssets) {
    if (!(name in lock.assets)) throw new Error(`unknown asset: ${name}`);
  }
  const explicit = requestedAssets.size > 0 || requestedGroups.size > 0;
  const selected = Object.entries(lock.assets).filter(([name, asset]) => {
    if (requestedAssets.has(name)) return true;
    if (requestedGroups.has(asset.group)) return true;
    return (
      !explicit && (options.includeOptional || asset.default_fetch !== false)
    );
  });
  if (selected.length === 0)
    throw new Error("no assets matched the requested filters");
  return selected;
}

export function resolveAssetRoot(lockRecord, output) {
  return path.resolve(
    output ??
      path.join(
        path.dirname(lockRecord.path),
        lockRecord.data.asset_root ?? "models",
      ),
  );
}

export function resolveAssetPath(root, asset) {
  const destination = path.resolve(root, asset.path);
  const relative = path.relative(root, destination);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`unsafe asset path: ${asset.path}`);
  }
  return destination;
}

export function redirectHostAllowed(url, asset) {
  const exact = new Set(asset.allowed_redirect_hosts ?? []);
  if (exact.has(url.hostname)) return true;
  return (asset.allowed_redirect_host_suffixes ?? []).some(
    (suffix) => suffix.startsWith(".") && url.hostname.endsWith(suffix),
  );
}

export async function sha256File(file) {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(file)) digest.update(chunk);
  return digest.digest("hex");
}

export async function verifyAssetFile(name, asset, root) {
  const file = resolveAssetPath(root, asset);
  let details;
  try {
    details = await stat(file);
  } catch (error) {
    if (error?.code === "ENOENT")
      return { name, path: file, ok: false, error: "missing" };
    throw error;
  }
  if (!details.isFile())
    return { name, path: file, ok: false, error: "not-a-file" };
  if (details.size !== asset.bytes) {
    return {
      name,
      path: file,
      ok: false,
      error: "byte-count-mismatch",
      expected: asset.bytes,
      actual: details.size,
    };
  }
  const digest = await sha256File(file);
  if (digest !== asset.sha256) {
    return {
      name,
      path: file,
      ok: false,
      error: "sha256-mismatch",
      expected: asset.sha256,
      actual: digest,
    };
  }
  return { name, path: file, ok: true, bytes: details.size, sha256: digest };
}
