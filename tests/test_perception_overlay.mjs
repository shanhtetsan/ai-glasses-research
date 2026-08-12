import assert from 'node:assert/strict';
import test from 'node:test';

import {
  containRect,
  drawHandLandmarks,
  HAND_CONNECTIONS,
  isHandResultFresh,
  isPerceptionFresh,
  resizeOverlayCanvas,
  transformNormalizedBox,
  transformNormalizedLandmark,
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

function handResult() {
  const landmarks = Array.from({length: 21}, (_, id) => ({
    id,
    x: id / 20,
    y: 1 - id / 20,
    z: -id / 100,
  }));
  return {
    available: true,
    stale: false,
    age_ms: 100,
    hands: [{
      handedness: 'Left',
      handedness_score: 0.98,
      landmarks,
      bbox_norm: [0, 0, 1, 1],
    }],
  };
}

test('all 21 hand landmarks map directly in canonical coordinates', () => {
  const result = handResult();
  assert.equal(HAND_CONNECTIONS.length, 21);
  result.hands[0].landmarks.forEach((landmark) => {
    assert.deepEqual(transformNormalizedLandmark(landmark), [landmark.x, landmark.y]);
  });

  const arcs = [];
  const ctx = {
    beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, fill() {},
    strokeRect() {}, fillText() {},
    arc(x, y, radius) { arcs.push({x, y, radius}); },
  };
  drawHandLandmarks(ctx, result.hands, {width: 200, height: 100});
  assert.equal(arcs.length, 21);
  assert.deepEqual(arcs[8], {x: 80, y: 60, radius: 6});
  assert.equal(arcs.filter(({radius}) => radius === 6).length, 1);
});

test('stale hand state clears independently from object freshness', () => {
  const result = handResult();
  assert.equal(isHandResultFresh(result, 100, 1500), true);
  assert.equal(isHandResultFresh({...result, stale: true}, 0, 1500), false);
  assert.equal(isHandResultFresh({...result, available: false}, 0, 1500), false);
  assert.equal(isPerceptionFresh({
    available: true, stale: false, age_ms: 100, objects: [],
  }, 100, 3000), true);
});

test('Hands and Objects controls are independent and add no browser transform', async () => {
  const source = await import('node:fs/promises').then(fs => fs.readFile('static/main.js', 'utf8'));
  assert.match(source, /let objectsEnabled = true;/);
  assert.match(source, /let handsEnabled = true;/);
  assert.match(source, /objectsToggle\.onclick/);
  assert.match(source, /handsToggle\.onclick/);
  const overlaySource = await import('node:fs/promises').then(
    fs => fs.readFile('static/perception_overlay.mjs', 'utf8'),
  );
  assert.doesNotMatch(overlaySource, /rotate|scale\(-1|mirror/i);
});
