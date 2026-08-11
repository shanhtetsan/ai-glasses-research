import assert from 'node:assert/strict';
import test from 'node:test';

import {
  containRect,
  isPerceptionFresh,
  resizeOverlayCanvas,
  transformNormalizedBox,
} from '../static/perception_overlay.mjs';

test('canonical portrait boxes map directly at top-left, bottom-right, and full frame', () => {
  assert.deepEqual(
    transformNormalizedBox([0.1, 0.2, 0.4, 0.6]),
    [0.1, 0.2, 0.4, 0.6],
  );
  assert.deepEqual(transformNormalizedBox([0, 0, 0.2, 0.2]), [0, 0, 0.2, 0.2]);
  assert.deepEqual(transformNormalizedBox([0.8, 0.8, 1, 1]), [0.8, 0.8, 1, 1]);
  assert.deepEqual(transformNormalizedBox([0, 0, 1, 1]), [0, 0, 1, 1]);
});

test('portrait frames use aspect-preserving contain geometry across resize', () => {
  assert.deepEqual(containRect(800, 600, 240, 320), {
    left: 175, top: 0, width: 450, height: 600,
  });
  assert.deepEqual(containRect(360, 640, 240, 320), {
    left: 0, top: 80, width: 360, height: 480,
  });
});

test('portrait thermal geometry remains 3:4 and DPR scales only backing pixels', () => {
  assert.deepEqual(containRect(195, 260, 240, 320), {
    left: 0, top: 0, width: 195, height: 260,
  });
  const canvas = {width: 0, height: 0};
  assert.equal(resizeOverlayCanvas(canvas, 240, 320, 2), 2);
  assert.deepEqual(canvas, {width: 480, height: 640});
});

test('freshness includes server age and clears stale or unavailable results', () => {
  const result = {available: true, stale: false, age_ms: 2500, objects: []};
  assert.equal(isPerceptionFresh(result, 499, 3000), true);
  assert.equal(isPerceptionFresh(result, 501, 3000), false);
  assert.equal(isPerceptionFresh({...result, stale: true}, 0, 3000), false);
  assert.equal(isPerceptionFresh({...result, available: false}, 0, 3000), false);
});
