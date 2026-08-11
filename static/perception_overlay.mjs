function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value)));
}

export function transformNormalizedBox(bbox) {
  if (!Array.isArray(bbox) || bbox.length !== 4 || bbox.some(v => !Number.isFinite(Number(v)))) {
    return null;
  }
  return bbox.map(clamp01);
}

export function containRect(containerWidth, containerHeight, contentWidth, contentHeight) {
  const cw = Math.max(0, Number(containerWidth) || 0);
  const ch = Math.max(0, Number(containerHeight) || 0);
  const iw = Math.max(0, Number(contentWidth) || 0);
  const ih = Math.max(0, Number(contentHeight) || 0);
  if (!cw || !ch || !iw || !ih) return {left: 0, top: 0, width: 0, height: 0};
  const scale = Math.min(cw / iw, ch / ih);
  const width = iw * scale;
  const height = ih * scale;
  return {left: (cw - width) / 2, top: (ch - height) / 2, width, height};
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

export function drawObjectDetections(ctx, objects, viewport) {
  const width = Math.max(0, Number(viewport.width) || 0);
  const height = Math.max(0, Number(viewport.height) || 0);
  if (!width || !height || !Array.isArray(objects)) return;

  ctx.font = '600 13px system-ui, sans-serif';
  ctx.textBaseline = 'top';
  ctx.lineWidth = 2;
  for (const object of objects) {
    const box = transformNormalizedBox(object.bbox_norm);
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
