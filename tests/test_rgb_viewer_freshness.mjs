import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';

import {
  RGB_VIEWER_STATE,
  RgbViewerFreshness,
} from '../static/rgb_viewer_freshness.mjs';

function harness() {
  let now = 0;
  const recoveries = [];
  const tracker = new RgbViewerFreshness({
    now: () => now,
    staleAfterMs: 2500,
    recoveryBackoffMs: [2000, 4000, 8000],
    onRecovery: event => recoveries.push(event),
  });
  return {tracker, recoveries, advance: ms => { now += ms; }};
}

test('successfully rendered RGB frames keep the viewer FRESH', () => {
  const h = harness();
  h.tracker.beginConnection();
  h.tracker.socketOpened();
  h.tracker.frameRendered();
  h.advance(2499);
  assert.equal(h.tracker.tick().state, RGB_VIEWER_STATE.FRESH);
  assert.equal(h.recoveries.length, 0);
});

test('browser-stale feed starts exactly one bounded recovery attempt', () => {
  const h = harness();
  h.tracker.updateBackend({socketConnected: true, canonicalFrameAgeMs: 0});
  h.tracker.beginConnection();
  h.tracker.socketOpened();
  h.tracker.frameRendered();
  h.advance(2600);
  h.tracker.updateBackend({socketConnected: true, canonicalFrameAgeMs: 0});
  assert.equal(h.tracker.snapshot().state, RGB_VIEWER_STATE.RECOVERING);
  assert.equal(h.recoveries.length, 1);
  h.tracker.tick();
  h.advance(1999);
  h.tracker.tick();
  assert.equal(h.recoveries.length, 1);
  h.advance(1);
  h.tracker.tick();
  assert.equal(h.recoveries.length, 2);
  assert.equal(h.recoveries[1].cooldownMs, 4000);
});

test('a rendered recovery frame returns to FRESH and resets backoff', () => {
  const h = harness();
  h.tracker.updateBackend({socketConnected: true, canonicalFrameAgeMs: 0});
  h.tracker.beginConnection();
  h.tracker.socketOpened();
  h.advance(2600);
  h.tracker.updateBackend({socketConnected: true, canonicalFrameAgeMs: 0});
  assert.equal(h.recoveries.length, 1);
  h.tracker.beginConnection();
  h.tracker.socketOpened();
  h.tracker.frameRendered();
  const state = h.tracker.snapshot();
  assert.equal(state.state, RGB_VIEWER_STATE.FRESH);
  assert.equal(state.recoveryAttempt, 0);
  assert.equal(state.nextRecoveryInMs, 0);
});

test('backend-stale classification suppresses browser reconnect storms', () => {
  const h = harness();
  h.tracker.beginConnection();
  h.tracker.socketOpened();
  h.advance(3000);
  h.tracker.updateBackend({socketConnected: true, canonicalFrameAgeMs: 3000});
  for (let i = 0; i < 20; i += 1) {
    h.advance(1000);
    assert.equal(h.tracker.tick().state, RGB_VIEWER_STATE.STALE);
    assert.equal(h.tracker.snapshot().cause, 'backend_frame');
  }
  assert.equal(h.recoveries.length, 0);
});

test('backend with no canonical frame is upstream-stale, not browser-stale', () => {
  const h = harness();
  h.tracker.beginConnection();
  h.tracker.socketOpened();
  h.advance(3000);
  h.tracker.updateBackend({socketConnected: true, canonicalFrameAgeMs: null});
  assert.equal(h.tracker.snapshot().state, RGB_VIEWER_STATE.STALE);
  assert.equal(h.tracker.snapshot().cause, 'backend_frame');
  assert.equal(h.recoveries.length, 0);
});

test('backend socket loss is DISCONNECTED and disposal disables recovery', () => {
  const h = harness();
  h.tracker.updateBackend({socketConnected: false, canonicalFrameAgeMs: null});
  assert.equal(h.tracker.snapshot().state, RGB_VIEWER_STATE.DISCONNECTED);
  h.tracker.dispose();
  h.advance(10000);
  h.tracker.tick();
  assert.equal(h.recoveries.length, 0);
});

test('a viewer connection that never opens is retried with backoff', () => {
  const h = harness();
  h.tracker.updateBackend({socketConnected: true, canonicalFrameAgeMs: 0});
  h.tracker.beginConnection();
  h.advance(2501);
  h.tracker.updateBackend({socketConnected: true, canonicalFrameAgeMs: 0});
  assert.equal(h.recoveries.length, 1);
  assert.equal(h.tracker.snapshot().state, RGB_VIEWER_STATE.RECOVERING);
});

test('main viewer integration preserves manual recovery and cleanup', () => {
  const source = readFileSync(new URL('../static/main.js', import.meta.url), 'utf8');
  assert.match(source, /connectCamera\(\{manual: true\}\)/);
  assert.match(source, /window\.addEventListener\('pagehide', cleanupViewer\)/);
  assert.match(source, /clearInterval\(freshnessTimer\)/);
  assert.match(source, /window\.removeEventListener\('resize', fitCanvas\)/);
  assert.match(source, /onRecovery: \(\)=>connectCamera\(\)/);
});
