import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';

import {audioBadgePresentation} from '../static/audio_freshness.mjs';

test('audio badge maps fresh to green', () => {
  assert.equal(audioBadgePresentation({state: 'fresh'}).className, 'chip ok');
});

test('audio badge maps intentional TTS suppression to neutral', () => {
  const badge = audioBadgePresentation({state: 'suppressed', audio_gate_reason: 'tts'});
  assert.equal(badge.className, 'chip neutral');
  assert.match(badge.text, /TTS suppressed/);
});

test('audio badge maps stale to yellow and disconnected to red', () => {
  assert.equal(audioBadgePresentation({state: 'stale'}).className, 'chip warn');
  assert.equal(audioBadgePresentation({state: 'disconnected'}).className, 'chip err');
});

test('browser polls compact freshness endpoint without health overwrite', () => {
  const main = readFileSync(new URL('../static/main.js', import.meta.url), 'utf8');
  const template = readFileSync(new URL('../templates/index.html', import.meta.url), 'utf8');
  assert.match(main, /fetch\('\/api\/audio-freshness'/);
  assert.match(main, /setInterval\(refreshAudioFreshness, 1000\)/);
  assert.match(main, /clearInterval\(audioFreshnessTimer\)/);
  assert.doesNotMatch(template, /setChip\(q\('asrStatus'\)/);
});
