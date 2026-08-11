const VALID_ROTATIONS = new Set([0, 90, 180, 270]);

function rotation(value) {
  const normalized = Number(value);
  return VALID_ROTATIONS.has(normalized) ? normalized : 0;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value)));
}

function rotatePoint(x, y, degrees) {
  if (degrees === 90) return [1 - y, x];
  if (degrees === 180) return [1 - x, 1 - y];
  if (degrees === 270) return [y, 1 - x];
  return [x, y];
}

export function transformNormalizedBox(
  bbox,
  inferenceRotationDeg = 0,
  displayRotationDeg = 0,
  mirrored = false,
) {
  if (!Array.isArray(bbox) || bbox.length !== 4 || bbox.some(v => !Number.isFinite(Number(v)))) {
    return null;
  }
  const [x1, y1, x2, y2] = bbox.map(clamp01);
  const delta = (rotation(displayRotationDeg) - rotation(inferenceRotationDeg) + 360) % 360;
  const corners = [
    rotatePoint(x1, y1, delta),
    rotatePoint(x2, y1, delta),
    rotatePoint(x2, y2, delta),
    rotatePoint(x1, y2, delta),
  ].map(([x, y]) => mirrored ? [1 - x, y] : [x, y]);
  const xs = corners.map(point => point[0]);
  const ys = corners.map(point => point[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

export function isPerceptionFresh(perception, elapsedSinceFetchMs, maxDisplayAgeMs = 3000) {
  if (!perception || !perception.available || perception.stale || !Array.isArray(perception.objects)) {
    return false;
  }
  const serverAgeMs = Number(perception.age_ms);
  const elapsedMs = Number(elapsedSinceFetchMs);
  if (!Number.isFinite(serverAgeMs) || !Number.isFinite(elapsedMs)) return false;
  return serverAgeMs + Math.max(0, elapsedMs) <= maxDisplayAgeMs;
}

export function resizeOverlayCanvas(canvas, cssWidth, cssHeight, pixelRatio = window.devicePixelRatio || 1) {
  const dpr = Math.max(1, Number(pixelRatio) || 1);
  const width = Math.max(1, Math.round(cssWidth * dpr));
  const height = Math.max(1, Math.round(cssHeight * dpr));
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  return dpr;
}

export function drawObjectDetections(ctx, objects, viewport, orientation = {}) {
  const width = Math.max(0, Number(viewport.width) || 0);
  const height = Math.max(0, Number(viewport.height) || 0);
  if (!width || !height || !Array.isArray(objects)) return;

  ctx.font = '600 13px system-ui, sans-serif';
  ctx.textBaseline = 'top';
  ctx.lineWidth = 2;
  for (const object of objects) {
    const box = transformNormalizedBox(
      object.bbox_norm,
      orientation.inferenceRotationDeg,
      orientation.displayRotationDeg,
      orientation.mirrored,
    );
    if (!box) continue;
    const x = box[0] * width;
    const y = box[1] * height;
    const boxWidth = (box[2] - box[0]) * width;
    const boxHeight = (box[3] - box[1]) * height;
    const label = `${object.label} ${Math.round(Number(object.confidence) * 100)}%`;
    const labelWidth = Math.ceil(ctx.measureText(label).width) + 10;
    const labelHeight = 21;
    const labelX = Math.min(Math.max(0, x), Math.max(0, width - labelWidth));
    const labelY = y >= labelHeight ? y - labelHeight : Math.min(height - labelHeight, y);

    ctx.strokeStyle = '#7ee787';
    ctx.strokeRect(x, y, boxWidth, boxHeight);
    ctx.fillStyle = 'rgba(11, 15, 20, 0.88)';
    ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
    ctx.fillStyle = '#e6edf3';
    ctx.fillText(label, labelX + 5, labelY + 3);
  }
}
