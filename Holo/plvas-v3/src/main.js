import "./styles.css";
import { getRampartGuard } from "./rampart.js";
import { setupScreenshotRedaction } from "./visual/ui.js";

const EXAMPLE =
  "My name is Alex Rivera. Email me at alex.rivera@example.com or call (415) 555-0136. I live at 742 Evergreen Terrace, San Francisco, CA 94107. My test SSN is 472-81-0094 and my test card is 4111 1111 1111 1111.";

const sourceText = document.querySelector("#source-text");
const backendSelect = document.querySelector("#backend-select");
const redactButton = document.querySelector("#redact-button");
const exampleButton = document.querySelector("#example-button");
const resetButton = document.querySelector("#reset-button");
const copyButton = document.querySelector("#copy-button");
const status = document.querySelector("#status");
const statusText = document.querySelector("#status-text");
const resultBox = document.querySelector("#result-box");
const resultText = document.querySelector("#result-text");
const entities = document.querySelector("#entities");
const entityList = document.querySelector("#entity-list");
const roundtrip = document.querySelector("#roundtrip");
const revealedText = document.querySelector("#revealed-text");

let guard;
let loadedBackend;
let protectedText = "";

sourceText.value = EXAMPLE;
setupScreenshotRedaction();

if (!navigator.gpu) {
  backendSelect.querySelector('option[value="webgpu"]').disabled = true;
}

exampleButton.addEventListener("click", () => {
  sourceText.value = EXAMPLE;
  sourceText.focus();
  sourceText.setSelectionRange(sourceText.value.length, sourceText.value.length);
});

backendSelect.addEventListener("change", () => {
  if (loadedBackend && loadedBackend !== backendSelect.value) {
    clearSession();
    setStatus("idle", "Backend changed — model will reload on the next run");
  }
});

redactButton.addEventListener("click", redact);
sourceText.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    redact();
  }
});

copyButton.addEventListener("click", async () => {
  if (!protectedText) return;
  await navigator.clipboard.writeText(protectedText);
  const original = copyButton.textContent;
  copyButton.textContent = "Copied";
  window.setTimeout(() => {
    copyButton.textContent = original;
  }, 1400);
});

resetButton.addEventListener("click", () => {
  clearSession();
  sourceText.value = "";
  sourceText.focus();
  setStatus("idle", "New session — no placeholder mappings retained");
});

async function redact() {
  const input = sourceText.value.trim();
  if (!input) {
    setStatus("error", "Enter a message first");
    sourceText.focus();
    return;
  }

  setBusy(true);

  try {
    if (!guard || loadedBackend !== backendSelect.value) {
      const startedAt = performance.now();
      setStatus(
        "loading",
        "Loading the checked-in 14 MB local model…",
      );
      guard = await getRampartGuard(backendSelect.value);
      loadedBackend = backendSelect.value;
      const seconds = ((performance.now() - startedAt) / 1000).toFixed(1);
      setStatus("ready", `Model ready via ${labelForBackend()} in ${seconds}s`);
    }

    const startedAt = performance.now();
    const safe = await guard.protect(input);
    const elapsed = Math.max(1, Math.round(performance.now() - startedAt));

    protectedText = safe.text;
    renderProtectedText(safe.text);
    renderPlaceholders(safe.placeholders);
    revealedText.textContent = guard.reveal(safe.text);
    roundtrip.hidden = false;
    copyButton.disabled = false;
    setStatus(
      "ready",
      `${safe.placeholders.length} placeholder${safe.placeholders.length === 1 ? "" : "s"} · ${elapsed} ms · processed locally`,
    );
  } catch (error) {
    console.error(error);
    guard = undefined;
    loadedBackend = undefined;
    setStatus(
      "error",
      backendSelect.value === "webgpu"
        ? "WebGPU could not load this model. Switch to WASM and try again."
        : `Could not load Rampart: ${friendlyError(error)}`,
    );
  } finally {
    setBusy(false);
  }
}

function renderProtectedText(text) {
  resultText.replaceChildren();
  const pattern = /(\[[A-Z][A-Z_]*_\d+\])/g;
  let cursor = 0;

  for (const match of text.matchAll(pattern)) {
    if (match.index > cursor) {
      resultText.append(document.createTextNode(text.slice(cursor, match.index)));
    }

    const token = document.createElement("mark");
    token.textContent = match[0];
    resultText.append(token);
    cursor = match.index + match[0].length;
  }

  if (cursor < text.length) {
    resultText.append(document.createTextNode(text.slice(cursor)));
  }

  resultBox.classList.remove("is-empty");
}

function renderPlaceholders(placeholders) {
  entityList.replaceChildren();
  const unique = [...new Set(placeholders)];

  for (const placeholder of unique) {
    const chip = document.createElement("span");
    chip.textContent = placeholder;
    entityList.append(chip);
  }

  entities.hidden = unique.length === 0;
}

function setBusy(isBusy) {
  redactButton.disabled = isBusy;
  backendSelect.disabled = isBusy;
  redactButton.querySelector("span:first-child").textContent = isBusy
    ? "Working locally…"
    : "Redact locally";
}

function setStatus(kind, message) {
  status.dataset.state = kind;
  statusText.textContent = message;
}

function clearSession() {
  guard = undefined;
  loadedBackend = undefined;
  protectedText = "";
  resultText.textContent = "Your protected message will appear here.";
  resultBox.classList.add("is-empty");
  entityList.replaceChildren();
  entities.hidden = true;
  roundtrip.hidden = true;
  roundtrip.open = false;
  copyButton.disabled = true;
}

function labelForBackend() {
  return loadedBackend === "webgpu" ? "WebGPU" : "WASM";
}

function friendlyError(error) {
  const message = error instanceof Error ? error.message : String(error);
  return message.length > 110 ? `${message.slice(0, 107)}…` : message;
}
