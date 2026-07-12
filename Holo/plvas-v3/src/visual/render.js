export const REDACTION_RGB = Object.freeze([5, 8, 7]);

export function burnRegionsIntoCanvas(sourceCanvas, outputCanvas, regions) {
  outputCanvas.width = sourceCanvas.width;
  outputCanvas.height = sourceCanvas.height;
  const context = outputCanvas.getContext("2d");
  if (!context) throw new Error("this browser cannot create a 2D canvas");
  context.drawImage(sourceCanvas, 0, 0);
  context.fillStyle = `rgb(${REDACTION_RGB.join(", ")})`;
  for (const region of regions) {
    const rectangle = integerMask(region, outputCanvas.width, outputCanvas.height);
    context.fillRect(rectangle.x, rectangle.y, rectangle.width, rectangle.height);
  }
  return outputCanvas;
}

export function integerMask(region, canvasWidth, canvasHeight) {
  const x1 = clamp(Math.floor(region.x1), 0, canvasWidth);
  const y1 = clamp(Math.floor(region.y1), 0, canvasHeight);
  const x2 = clamp(Math.ceil(region.x2), 0, canvasWidth);
  const y2 = clamp(Math.ceil(region.y2), 0, canvasHeight);
  return {
    x: x1,
    y: y1,
    width: Math.max(0, x2 - x1),
    height: Math.max(0, y2 - y1),
  };
}

export function canvasToPngBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("could not encode the redacted PNG"));
    }, "image/png");
  });
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}
