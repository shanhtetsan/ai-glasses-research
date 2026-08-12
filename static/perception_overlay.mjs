function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value)));
}

export function transformNormalizedBox(bbox) {
  if (!Array.isArray(bbox) || bbox.length !== 4 || bbox.some(v => !Number.isFinite(Number(v)))) {
    return null;
  }
  return bbox.map(clamp01);
}

export const HAND_CONNECTIONS = Object.freeze([
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [17, 0],
]);

export function transformNormalizedLandmark(landmark) {
  if (!landmark || !Number.isFinite(Number(landmark.x)) || !Number.isFinite(Number(landmark.y))) {
    return null;
  }
  // Canonical JPEG coordinates map directly to the canonical browser frame.
  return [clamp01(landmark.x), clamp01(landmark.y)];
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

export function isHandResultFresh(result, elapsedSinceFetchMs, maxDisplayAgeMs = 1500) {
  if (!result || !result.available || result.stale || !Array.isArray(result.hands)) {
    return false;
  }
  const serverAgeMs = Number(result.age_ms);
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

export function drawHandLandmarks(ctx, hands, viewport, {showBbox = true} = {}) {
  const width = Math.max(0, Number(viewport.width) || 0);
  const height = Math.max(0, Number(viewport.height) || 0);
  if (!width || !height || !Array.isArray(hands)) return;

  ctx.font = '600 12px system-ui, sans-serif';
  ctx.textBaseline = 'bottom';
  for (const hand of hands.slice(0, 2)) {
    if (!hand || !Array.isArray(hand.landmarks) || hand.landmarks.length !== 21) continue;
    const points = hand.landmarks.map(transformNormalizedLandmark);
    if (points.some(point => point === null)) continue;

    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (const [from, to] of HAND_CONNECTIONS) {
      ctx.moveTo(points[from][0] * width, points[from][1] * height);
      ctx.lineTo(points[to][0] * width, points[to][1] * height);
    }
    ctx.stroke();

    for (let landmarkId = 0; landmarkId < points.length; landmarkId++) {
      const [xNorm, yNorm] = points[landmarkId];
      ctx.beginPath();
      ctx.arc(
        xNorm * width,
        yNorm * height,
        landmarkId === 8 ? 6 : 3,
        0,
        Math.PI * 2,
      );
      ctx.fillStyle = landmarkId === 8 ? '#ff7b72' : '#79c0ff';
      ctx.fill();
    }

    const box = showBbox ? transformNormalizedBox(hand.bbox_norm) : null;
    if (box) {
      const x = box[0] * width;
      const y = box[1] * height;
      const boxWidth = (box[2] - box[0]) * width;
      const boxHeight = (box[3] - box[1]) * height;
      ctx.strokeStyle = 'rgba(88, 166, 255, 0.75)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, boxWidth, boxHeight);
      const score = Math.round(Number(hand.handedness_score) * 100);
      const label = `${hand.handedness || 'Hand'} ${Number.isFinite(score) ? score : 0}%`;
      ctx.fillStyle = '#e6edf3';
      ctx.fillText(label, Math.max(0, x), Math.max(12, y - 3));
    }
  }
}
