import assert from 'node:assert/strict';
import test from 'node:test';

import {
  DEVICE_ID_PATTERN, MIC_CHUNK_BYTES, PcmChunker, cameraMessage, decodeServerEvent,
  compassHeading, floatToInt16, frameSize, headingMessage, makeDeviceId, newResampleState,
  normalizeDegrees, pcm16BytesToFloat, landmarkId, siteName, SITE_PATTERN, parseSteps,
  removeLandmark, graphPayload, autoTrialId,
  resampleToInt16, stampJpegSequence, trialFormError, trialRequestBody,
} from '../static/field_test_core.mjs';

// The backend's reader for the seq stamp (phone_capture.jpeg_comment_seq),
// mirrored here so the stamp is checked against the same parse.
function readSeq(bytes) {
  let i = 2;
  while (i + 4 <= bytes.length && bytes[i] === 0xff) {
    const marker = bytes[i + 1];
    if (marker === 0xda) return null;
    const length = (bytes[i + 2] << 8) | bytes[i + 3];
    if (marker === 0xfe) {
      const text = new TextDecoder().decode(bytes.subarray(i + 4, i + 2 + length));
      if (text.startsWith('seq=')) return Number(text.slice(4));
    }
    i += 2 + length;
  }
  return null;
}

test('device id passes the backend ?device= rule', () => {
  for (let i = 0; i < 50; i += 1) assert.match(makeDeviceId(), DEVICE_ID_PATTERN);
  assert.equal(makeDeviceId(() => 0), 'web-00000000');
});

test('resampling 48 kHz to 16 kHz keeps the sample count across blocks', () => {
  const state = newResampleState();
  let total = 0;
  for (let block = 0; block < 100; block += 1) {
    total += resampleToInt16(new Float32Array(4096).fill(0.5), 48000, 16000, state).length;
  }
  // 409600 input samples -> ~136533 output samples, within one sample.
  assert.ok(Math.abs(total - 409600 / 3) <= 1, `got ${total}`);
});

test('resampling 44.1 kHz keeps the rate within a sample', () => {
  const state = newResampleState();
  let total = 0;
  for (let block = 0; block < 50; block += 1) {
    total += resampleToInt16(new Float32Array(4096), 44100, 16000, state).length;
  }
  assert.ok(Math.abs(total - (50 * 4096 * 16000) / 44100) <= 1, `got ${total}`);
});

test('resampled ramp is continuous across a block boundary', () => {
  const state = newResampleState();
  const ramp = Float32Array.from({ length: 8 }, (_, i) => i / 100);
  const first = resampleToInt16(ramp, 2, 1, state);
  const next = Float32Array.from({ length: 8 }, (_, i) => (i + 8) / 100);
  const second = resampleToInt16(next, 2, 1, state);
  const joined = [...first, ...second].map((v) => Math.round((v / 0x7fff) * 100));
  assert.deepEqual(joined, [0, 2, 4, 6, 8, 10, 12, 14]);
});

test('float to int16 clamps and keeps sign', () => {
  assert.equal(floatToInt16(2), 32767);
  assert.equal(floatToInt16(-2), -32768);
  assert.equal(floatToInt16(0), 0);
});

test('chunker emits whole 640-byte little-endian chunks and keeps the rest', () => {
  const chunker = new PcmChunker();
  assert.equal(chunker.push(new Int16Array(300)).length, 0);
  const chunks = chunker.push(Int16Array.from({ length: 400 }, () => 0x0102));
  // 300 + 400 samples = 1400 bytes: two whole chunks, 120 bytes left over.
  assert.equal(chunks.length, 2);
  assert.ok(chunks.every((chunk) => chunk.length === MIC_CHUNK_BYTES));
  assert.equal(chunks[0][600], 0x02);   // little-endian low byte first
  assert.equal(chunks[0][601], 0x01);
  assert.equal(chunker.pending.length, 1400 - 2 * 640);
  chunker.reset();
  assert.equal(chunker.pending.length, 0);
});

test('8 kHz reply bytes decode and upsample to the output rate', () => {
  const bytes = new Uint8Array(8);
  const view = new DataView(bytes.buffer);
  [0, 16384, -16384, 32767].forEach((v, i) => view.setInt16(i * 2, v, true));
  const same = pcm16BytesToFloat(bytes, 8000, 8000);
  assert.deepEqual([...same].map((v) => Math.round(v * 4) / 4), [0, 0.5, -0.5, 1]);
  assert.equal(pcm16BytesToFloat(bytes, 8000, 48000).length, 24);
  assert.equal(pcm16BytesToFloat(new Uint8Array(0), 8000, 48000).length, 0);
});

test('seq stamp is read back by the backend parse and keeps the image', () => {
  const jpeg = Uint8Array.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x04, 0x4a, 0x46, 0xff, 0xda, 1, 2, 0xff, 0xd9]);
  const stamped = stampJpegSequence(jpeg, 1234);
  assert.equal(readSeq(stamped), 1234);
  assert.deepEqual([...stamped.subarray(0, 4)], [0xff, 0xd8, 0xff, 0xfe]);
  assert.deepEqual([...stamped.subarray(stamped.length - jpeg.length + 2)], [...jpeg.subarray(2)]);
  assert.throws(() => stampJpegSequence(Uint8Array.from([1, 2, 3]), 1));
});

test('camera message is type byte 0x01 then the JPEG', () => {
  const message = cameraMessage(Uint8Array.from([0xff, 0xd8, 9]));
  assert.deepEqual([...message], [0x01, 0xff, 0xd8, 9]);
});

test('frame size is 640 wide, keeps aspect, never upscales', () => {
  assert.deepEqual(frameSize(720, 1280), { width: 640, height: 1138 });
  assert.deepEqual(frameSize(1280, 720), { width: 640, height: 360 });
  assert.deepEqual(frameSize(480, 640), { width: 480, height: 640 });
  assert.equal(frameSize(0, 0), null);
});

test('server events decode like the app', () => {
  assert.deepEqual(decodeServerEvent('OK:STARTED'), { type: 'started' });
  assert.deepEqual(decodeServerEvent('OK:STOPPED'), { type: 'stopped' });
  assert.deepEqual(decodeServerEvent('RESET'), { type: 'reset' });
  assert.deepEqual(decodeServerEvent('TTS:START:7'), { type: 'ttsStart', turnId: 7 });
  assert.deepEqual(decodeServerEvent('TTS:END:7'), { type: 'ttsEnd', turnId: 7 });
  assert.deepEqual(decodeServerEvent('TTS:RESET:7'), { type: 'ttsReset', turnId: 7 });
  assert.equal(decodeServerEvent('TTS:START:x').type, 'unknown');
  assert.equal(decodeServerEvent('LATENCY:PONG:1:2').type, 'unknown');
});

test('trial body matches the /trials form and validates ids', () => {
  const body = trialRequestBody({
    trial: ' T03 ', pid: 'evis', site: 'BMCC', cond: 'no-graph', task: 'find door',
    success_criterion: '', room: 'hallway', present: 'door, stairs ,', absent: '',
  });
  assert.deepEqual(body, {
    trial: 'T03', pid: 'evis', site: 'BMCC', cond: 'no-graph', task: 'find door',
    success_criterion: '',
    ground_truth: { room: 'hallway', present: ['door', 'stairs'], absent: [] },
  });
  assert.equal(trialFormError(body), '');
  assert.equal(trialRequestBody({ trial: 'T1', pid: 'e' }).ground_truth, undefined);
  assert.match(trialFormError({ trial: '', pid: 'e' }), /required/);
  assert.match(trialFormError({ trial: 'T 1', pid: 'e' }), /letters/);
});

test('compass heading: iOS value as is, absolute alpha flipped, else null', () => {
  assert.equal(compassHeading({ webkitCompassHeading: 90 }), 90);
  assert.equal(compassHeading({ webkitCompassHeading: 360 }), 0);
  assert.equal(compassHeading({ webkitCompassHeading: -1, absolute: false, alpha: 10 }), null);
  assert.equal(compassHeading({ absolute: true, alpha: 90 }), 270);
  assert.equal(compassHeading({ absolute: true, alpha: 0 }), 0);
  assert.equal(compassHeading({ absolute: false, alpha: 90 }), null);
  assert.equal(compassHeading({}), null);
  assert.equal(normalizeDegrees(-10), 350);
  assert.equal(normalizeDegrees(725), 5);
});

test('heading packet is 0x05 + <Iff and never sends 360', () => {
  const message = headingMessage(7, 123.5, 15);
  const view = new DataView(message.buffer);
  assert.equal(message.length, 13);
  assert.equal(message[0], 0x05);
  assert.equal(view.getUint32(1, true), 7);
  assert.equal(view.getFloat32(5, true), 123.5);
  assert.equal(view.getFloat32(9, true), 15);
  assert.equal(new DataView(headingMessage(1, 10, null).buffer).getFloat32(9, true), -1);
  assert.equal(new DataView(headingMessage(1, 359.99999999, 1).buffer).getFloat32(5, true), 0);
});

test('landmark ids are slugs, unique, never "start"', () => {
  assert.equal(landmarkId('Front Door!'), 'front-door');
  assert.equal(landmarkId('front door', new Set(['front-door'])), 'front-door-2');
  assert.equal(landmarkId('Start'), 'start-point');
  assert.equal(landmarkId('!!!'), 'landmark');
  assert.ok(landmarkId('x'.repeat(80)).length <= 40);
});

test('site names, steps, removal cascade', () => {
  assert.equal(siteName(' My Home '), 'my-home');
  assert.ok(SITE_PATTERN.test(siteName('BMCC hall 3')));
  assert.equal(parseSteps(''), null);
  assert.equal(parseSteps('6'), 6);
  assert.equal(parseSteps('6.5'), undefined);
  assert.equal(parseSteps('-1'), undefined);
  const list = [
    { id: 'door', from: 'start' }, { id: 'hall', from: 'door' },
    { id: 'stairs', from: 'hall' }, { id: 'exit', from: 'start' },
  ];
  assert.deepEqual(removeLandmark(list, 'door').map((x) => x.id), ['exit']);
});

test('graph payload rounds bearings and never sends 360', () => {
  const body = graphPayload([
    { id: 'a', name: 'a', kind: 'door', from: 'start', bearing_deg: 359.97, steps: null, note: '' },
    { id: 'b', name: 'b', kind: 'exit', bearing_deg: 12.34, steps: 4 },
  ]);
  assert.deepEqual(body.landmarks.map((x) => [x.bearing_deg, x.from, x.steps, x.note]),
    [[0, 'start', null, ''], [12.3, 'start', 4, '']]);
});

test('trial body carries the experiment arms only when chosen', () => {
  assert.equal(trialRequestBody({ trial: 'T1', pid: 'e' }).config, undefined);
  assert.deepEqual(trialRequestBody({ trial: 'T1', pid: 'e', pin: 'off', graph: 'home' }).config,
    { pin_frame_every_turn: false, landmark_graph: 'home' });
  assert.deepEqual(trialRequestBody({ trial: 'T1', pid: 'e', pin: 'on' }).config,
    { pin_frame_every_turn: true });
});

test('auto trial id is valid, local time, to the second', () => {
  const id = autoTrialId(new Date(2026, 10, 8, 14, 5, 9));
  assert.equal(id, 'T1108-140509');
  assert.equal(trialFormError({ trial: id, pid: 'evis' }), '');
});
