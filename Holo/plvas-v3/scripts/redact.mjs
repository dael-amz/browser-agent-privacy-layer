#!/usr/bin/env node

import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { createGuard } from "@nationaldesignstudio/rampart";

const args = process.argv.slice(2);
const heuristicsOnly = args.includes("--heuristics-only");
const message = args.filter((arg) => arg !== "--heuristics-only").join(" ").trim();

process.stderr.write(
  heuristicsOnly
    ? "Starting Rampart (structured PII only)…\n"
    : "Loading Rampart locally (the first run downloads about 15 MB)…\n",
);

const guard = await createGuard({
  device: "cpu",
  heuristicsOnly,
});

process.stderr.write("Ready.\n");

if (message) {
  await printProtected(message);
} else {
  const terminal = createInterface({ input, output });
  output.write("Paste text and press Enter. Submit an empty line to quit.\n\n");

  while (true) {
    const line = await terminal.question("> ");
    if (!line.trim()) break;
    await printProtected(line);
  }

  terminal.close();
}

async function printProtected(text) {
  const startedAt = performance.now();
  const safe = await guard.protect(text);
  const elapsed = Math.max(1, Math.round(performance.now() - startedAt));

  output.write(`\nProtected (${elapsed} ms):\n${safe.text}\n`);
  output.write(`Placeholders: ${safe.placeholders.join(", ") || "none"}\n`);
  output.write(`Local round-trip:\n${guard.reveal(safe.text)}\n\n`);
}
