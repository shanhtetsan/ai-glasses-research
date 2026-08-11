import assert from 'node:assert/strict';
import test from 'node:test';

import {
  isPerceptionFresh,
  transformNormalizedBox,
} from '../static/perception_overlay.mjs';

test('box coordinates stay unchanged when inference and display orientations match', () => {
  assert.deepEqual(
    transformNormalizedBox([0.1, 0.2, 0.4, 0.6], 90, 90),
    [0.1, 0.2, 0.4, 0.6],
  );
});

test('box coordinates map from rotated inference space into raw display space', () => {
  assert.deepEqual(
    transformNormalizedBox([0.1, 0.2, 0.4, 0.6], 90, 0),
    [0.2, 0.6, 0.6, 0.9],
  );
});

test('freshness includes server age and clears stale or unavailable results', () => {
  const result = {available: true, stale: false, age_ms: 2500, objects: []};
  assert.equal(isPerceptionFresh(result, 499, 3000), true);
  assert.equal(isPerceptionFresh(result, 501, 3000), false);
  assert.equal(isPerceptionFresh({...result, stale: true}, 0, 3000), false);
  assert.equal(isPerceptionFresh({...result, available: false}, 0, 3000), false);
});
