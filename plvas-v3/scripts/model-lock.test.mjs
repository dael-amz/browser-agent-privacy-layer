import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  loadLock,
  redirectHostAllowed,
  resolveAssetPath,
  verifyAssetFile,
} from "./model-lock.mjs";

async function temporaryRoot(t) {
  const root = await mkdtemp(path.join(tmpdir(), "plva-model-lock-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

test("verification rejects missing, truncated, and corrupt assets", async (t) => {
  const root = await temporaryRoot(t);
  const content = Buffer.from("locked-model");
  const asset = {
    path: "semantic/model.onnx",
    bytes: content.length,
    sha256: createHash("sha256").update(content).digest("hex"),
  };
  assert.equal((await verifyAssetFile("model", asset, root)).error, "missing");

  const target = resolveAssetPath(root, asset);
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, content.subarray(0, 3));
  assert.equal(
    (await verifyAssetFile("model", asset, root)).error,
    "byte-count-mismatch",
  );

  await writeFile(target, Buffer.alloc(content.length, 0));
  assert.equal(
    (await verifyAssetFile("model", asset, root)).error,
    "sha256-mismatch",
  );
  await writeFile(target, content);
  assert.equal((await verifyAssetFile("model", asset, root)).ok, true);
});

test("asset paths and redirect hosts fail closed", async (t) => {
  const root = await temporaryRoot(t);
  assert.throws(
    () => resolveAssetPath(root, { path: "../escape.bin" }),
    /unsafe asset path/,
  );
  const asset = {
    allowed_redirect_hosts: ["huggingface.co"],
    allowed_redirect_host_suffixes: [".hf.co"],
  };
  assert.equal(
    redirectHostAllowed(new URL("https://huggingface.co/a"), asset),
    true,
  );
  assert.equal(
    redirectHostAllowed(new URL("https://cdn-lfs.hf.co/a"), asset),
    true,
  );
  assert.equal(
    redirectHostAllowed(new URL("https://evil.example/a"), asset),
    false,
  );
});

test("lock loading rejects invalid hashes", async (t) => {
  const root = await temporaryRoot(t);
  const lock = root + "/models.lock.json";
  await writeFile(
    lock,
    JSON.stringify({
      schema_version: 2,
      assets: {
        bad: {
          path: "bad.bin",
          url: "https://example.com/bad.bin",
          bytes: 1,
          sha256: "not-a-sha",
          allowed_redirect_hosts: ["example.com"],
        },
      },
    }),
  );
  await assert.rejects(loadLock(lock), /invalid SHA-256/);
});
