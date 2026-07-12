import { detectHybridSensitiveRegions } from "./hybrid.js";
import { burnRegionsIntoCanvas, canvasToPngBlob } from "./render.js";

const MAX_IMAGE_PIXELS = 50_000_000;

export function setupScreenshotRedaction() {
  const fileInput = document.querySelector("#visual-file-input");
  const chooseButton = document.querySelector("#visual-choose-button");
  const dropZone = document.querySelector("#visual-drop-zone");
  const emptyInput = document.querySelector("#visual-input-empty");
  const sourceCanvas = document.querySelector("#visual-source-canvas");
  const sensitivity = document.querySelector("#visual-sensitivity");
  const redactButton = document.querySelector("#visual-redact-button");
  const resetButton = document.querySelector("#visual-reset-button");
  const status = document.querySelector("#visual-status");
  const statusText = document.querySelector("#visual-status-text");
  const outputEmpty = document.querySelector("#visual-output-empty");
  const outputCanvas = document.querySelector("#visual-output-canvas");
  const downloadButton = document.querySelector("#visual-download-button");
  const detectionSummary = document.querySelector("#visual-detections");
  const detectionList = document.querySelector("#visual-detection-list");

  let sourceName = "screenshot";
  let hasSource = false;
  let busy = false;

  chooseButton.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    const [file] = fileInput.files;
    if (file) void loadImage(file);
  });

  dropZone.addEventListener("click", (event) => {
    if (!busy && !hasSource && event.currentTarget === dropZone) fileInput.click();
  });
  dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  for (const name of ["dragenter", "dragover"]) {
    dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      if (!busy) dropZone.classList.add("is-dragging");
    });
  }
  for (const name of ["dragleave", "drop"]) {
    dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      dropZone.classList.remove("is-dragging");
    });
  }
  dropZone.addEventListener("drop", (event) => {
    if (busy) return;
    const file = [...event.dataTransfer.files].find((item) =>
      item.type.startsWith("image/"),
    );
    if (file) void loadImage(file);
    else setStatus("error", "Drop a PNG, JPEG, or WebP screenshot");
  });

  window.addEventListener("paste", (event) => {
    const file = [...(event.clipboardData?.files ?? [])].find((item) =>
      item.type.startsWith("image/"),
    );
    if (file && !busy) {
      event.preventDefault();
      void loadImage(file, "pasted-screenshot.png");
    }
  });

  sensitivity.addEventListener("change", () => {
    if (!hasSource) return;
    clearOutput();
    setStatus("idle", "Sensitivity changed — run redaction again");
  });
  redactButton.addEventListener("click", () => void redactScreenshot());
  resetButton.addEventListener("click", reset);
  downloadButton.addEventListener("click", downloadOutput);

  async function loadImage(file, fallbackName = "screenshot.png") {
    if (!file.type.startsWith("image/")) {
      setStatus("error", "Choose a PNG, JPEG, or WebP screenshot");
      return;
    }
    setBusy(true);
    setStatus("loading", "Decoding screenshot locally…");

    try {
      const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
      if (bitmap.width * bitmap.height > MAX_IMAGE_PIXELS) {
        bitmap.close();
        throw new Error("screenshot is too large; use an image below 50 megapixels");
      }
      sourceCanvas.width = bitmap.width;
      sourceCanvas.height = bitmap.height;
      const context = sourceCanvas.getContext("2d");
      if (!context) throw new Error("this browser cannot create a 2D canvas");
      context.drawImage(bitmap, 0, 0);
      bitmap.close();

      sourceName = file.name || fallbackName;
      hasSource = true;
      emptyInput.hidden = true;
      sourceCanvas.hidden = false;
      dropZone.classList.add("has-image");
      clearOutput();
      redactButton.disabled = false;
      resetButton.hidden = false;
      setStatus(
        "ready",
        `${sourceName} · ${sourceCanvas.width}×${sourceCanvas.height} · ready locally`,
      );
    } catch (error) {
      setStatus("error", friendlyError(error));
    } finally {
      fileInput.value = "";
      setBusy(false);
    }
  }

  async function redactScreenshot() {
    if (!hasSource || busy) return;
    setBusy(true);
    const startedAt = performance.now();

    try {
      const result = await detectHybridSensitiveRegions(sourceCanvas, {
        profile: sensitivity.value,
        onStage: (message) => setStatus("loading", message),
      });
      renderRedactedImage(result.regions);
      const elapsed = Math.max(1, Math.round(performance.now() - startedAt));
      const evidence = `${result.counts.visual} visual · ${result.counts.ocrSensitive} OCR-semantic`;
      if (result.regions.length === 0) {
        setStatus(
          "warning",
          `No regions detected in ${elapsed} ms (${evidence}) — this does not prove the screenshot is safe`,
        );
      } else if (result.warnings.length > 0) {
        setStatus(
          "warning",
          `${result.regions.length} masks · ${evidence} · degraded: ${result.warnings.join("; ")}`,
        );
      } else {
        setStatus(
          "ready",
          `${result.regions.length} mask${result.regions.length === 1 ? "" : "s"} · ${evidence} · ${elapsed} ms · no upload`,
        );
      }
    } catch (error) {
      console.error(error);
      clearOutput();
      setStatus("error", `Screenshot redaction failed: ${friendlyError(error)}`);
    } finally {
      setBusy(false);
    }
  }

  function renderRedactedImage(detections) {
    burnRegionsIntoCanvas(sourceCanvas, outputCanvas, detections);

    outputEmpty.hidden = true;
    outputCanvas.hidden = false;
    downloadButton.disabled = false;
    renderDetections(detections);
  }

  function renderDetections(detections) {
    detectionList.replaceChildren();
    for (const detection of detections) {
      const chip = document.createElement("span");
      const labels = detection.labels?.join(" + ") || detection.label;
      const sources = detection.sources?.join(" + ") || "VISUAL";
      chip.textContent = `${labels} · ${sources} · ${Math.round(detection.score * 100)}%`;
      detectionList.append(chip);
    }
    detectionSummary.hidden = detections.length === 0;
  }

  async function downloadOutput() {
    if (outputCanvas.hidden) return;
    try {
      const blob = await canvasToPngBlob(outputCanvas);
      const link = document.createElement("a");
      const base = sourceName.replace(/\.[^.]+$/, "") || "screenshot";
      const url = URL.createObjectURL(blob);
      link.href = url;
      link.download = `${base}-redacted.png`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      setStatus("error", friendlyError(error));
    }
  }

  function clearOutput() {
    outputCanvas.width = 0;
    outputCanvas.height = 0;
    outputCanvas.hidden = true;
    outputEmpty.hidden = false;
    downloadButton.disabled = true;
    detectionList.replaceChildren();
    detectionSummary.hidden = true;
  }

  function reset() {
    hasSource = false;
    sourceName = "screenshot";
    sourceCanvas.width = 0;
    sourceCanvas.height = 0;
    sourceCanvas.hidden = true;
    emptyInput.hidden = false;
    dropZone.classList.remove("has-image", "is-dragging");
    redactButton.disabled = true;
    resetButton.hidden = true;
    clearOutput();
    setStatus("idle", "Choose, drop, or paste a screenshot");
  }

  function setBusy(value) {
    busy = value;
    fileInput.disabled = value;
    chooseButton.disabled = value;
    sensitivity.disabled = value;
    redactButton.disabled = value || !hasSource;
    resetButton.disabled = value;
    redactButton.querySelector("span:first-child").textContent = value
      ? "Working locally…"
      : "Redact screenshot";
  }

  function setStatus(kind, message) {
    status.dataset.state = kind;
    statusText.textContent = message;
  }
}

function friendlyError(error) {
  const message = error instanceof Error ? error.message : String(error);
  return message.length > 130 ? `${message.slice(0, 127)}…` : message;
}
