// Pure helpers for the browser field-test page (/test). No DOM, no network,
// so tests/test_field_test_core.mjs can run them under node.
//
// The page stands in for the iOS app during internal testing, so every wire
// format here copies what the app sends (ai-glasses-link):
//   mic   -> /ws_audio           16 kHz mono PCM16, 640-byte (20 ms) chunks
//   reply <- /ws_audio           8 kHz mono PCM16
//   frame -> /ws/camera_thermal  0x01 + JPEG with a "seq=<n>" COM segment

export const MSG_TYPE_CAM = 0x01;
export const MSG_TYPE_HEADING = 0x05;
export const HEADING_INTERVAL_MS = 250;        // 4 Hz to the server; it records 2 Hz in a turn
export const HEADING_MAX_AGE_MS = 1000;        // an older reading is not sent
export const MIC_SAMPLE_RATE = 16000;
export const MIC_CHUNK_BYTES = 640;            // 20 ms at 16 kHz PCM16
export const TTS_SAMPLE_RATE = 8000;
export const FRAME_WIDTH = 640;                // the app scales to 640 px wide
export const FRAME_INTERVAL_MS = 250;          // the app sends at most 4 frames/s
export const MAX_JPEG_BYTES = 300 * 1024;      // stability_runtime.MAX_JPEG_PAYLOAD_BYTES

// Same rule the backend enforces on ?device= (_CAMERA_DEVICE_ID).
export const DEVICE_ID_PATTERN = /^[A-Za-z0-9._:-]{1,64}$/;

export function makeDeviceId(random = Math.random) {
  let suffix = "";
  for (let i = 0; i < 8; i += 1) suffix += Math.floor(random() * 36).toString(36);
  return `web-${suffix}`;
}

// Float samples at `inputRate` -> Int16 samples at `outputRate`, by linear
// interpolation. `state.pos` is the read position relative to this block
// (it can be in [-1, 0): index -1 is `state.last`, the previous block's final
// sample), so consecutive blocks join without a gap or a click.
export function resampleToInt16(input, inputRate, outputRate, state) {
  if (!input.length) return new Int16Array(0);
  const step = inputRate / outputRate;
  const out = [];
  let pos = state.pos;
  while (pos < input.length - 1) {
    const i = Math.floor(pos);
    const frac = pos - i;
    const a = i < 0 ? state.last : input[i];
    const b = input[i + 1];
    out.push(floatToInt16(a + (b - a) * frac));
    pos += step;
  }
  state.pos = pos - input.length;
  state.last = input[input.length - 1];
  return Int16Array.from(out);
}

export function newResampleState() {
  return { pos: 0, last: 0 };
}

export function floatToInt16(value) {
  const clamped = Math.max(-1, Math.min(1, value));
  return clamped < 0 ? Math.round(clamped * 0x8000) : Math.round(clamped * 0x7fff);
}

// Collects PCM16 bytes and hands back whole 640-byte chunks (PCMChunker in the app).
export class PcmChunker {
  constructor(chunkBytes = MIC_CHUNK_BYTES) {
    this.chunkBytes = chunkBytes;
    this.pending = new Uint8Array(0);
  }

  push(int16) {
    const bytes = int16ToLittleEndianBytes(int16);
    const merged = new Uint8Array(this.pending.length + bytes.length);
    merged.set(this.pending, 0);
    merged.set(bytes, this.pending.length);
    const chunks = [];
    let offset = 0;
    while (merged.length - offset >= this.chunkBytes) {
      chunks.push(merged.slice(offset, offset + this.chunkBytes));
      offset += this.chunkBytes;
    }
    this.pending = merged.slice(offset);
    return chunks;
  }

  reset() {
    this.pending = new Uint8Array(0);
  }
}

export function int16ToLittleEndianBytes(int16) {
  const bytes = new Uint8Array(int16.length * 2);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < int16.length; i += 1) view.setInt16(i * 2, int16[i], true);
  return bytes;
}

// 8 kHz little-endian PCM16 bytes -> Float32 at the output context's rate.
// Resampled here rather than relying on createBuffer(…, 8000), which older
// Safari refuses below 22.05 kHz.
export function pcm16BytesToFloat(bytes, inputRate, outputRate) {
  const count = Math.floor(bytes.byteLength / 2);
  const view = new DataView(bytes.buffer, bytes.byteOffset, count * 2);
  const src = new Float32Array(count);
  for (let i = 0; i < count; i += 1) src[i] = view.getInt16(i * 2, true) / 0x8000;
  if (inputRate === outputRate || count === 0) return src;
  const outLength = Math.max(1, Math.round((count * outputRate) / inputRate));
  const out = new Float32Array(outLength);
  const step = inputRate / outputRate;
  for (let i = 0; i < outLength; i += 1) {
    const pos = i * step;
    const j = Math.floor(pos);
    const frac = pos - j;
    const a = src[Math.min(j, count - 1)];
    const b = src[Math.min(j + 1, count - 1)];
    out[i] = a + (b - a) * frac;
  }
  return out;
}

// The app's JPEGSequenceStamp.stamp: a "seq=<n>" COM segment right after SOI.
// The backend reads it before re-encoding (frame_recv.phone_seq).
export function stampJpegSequence(jpeg, sequence) {
  if (jpeg.length < 2 || jpeg[0] !== 0xff || jpeg[1] !== 0xd8) {
    throw new Error("camera frame does not start with a JPEG SOI marker");
  }
  const text = new TextEncoder().encode(`seq=${sequence}`);
  const length = text.length + 2;
  const out = new Uint8Array(jpeg.length + 2 + length);
  out.set([0xff, 0xd8, 0xff, 0xfe, length >> 8, length & 0xff], 0);
  out.set(text, 6);
  out.set(jpeg.subarray(2), 6 + text.length);
  return out;
}

// The frame as /ws/camera_thermal expects it: one type byte, then the JPEG.
export function cameraMessage(jpeg) {
  const out = new Uint8Array(jpeg.length + 1);
  out[0] = MSG_TYPE_CAM;
  out.set(jpeg, 1);
  return out;
}

// Compass heading in degrees from magnetic north, clockwise, in [0, 360).
// iOS Safari gives it directly (webkitCompassHeading). Elsewhere an absolute
// deviceorientation alpha is counter-clockwise, so it is flipped. null when
// the event carries no absolute heading.
export function compassHeading(event) {
  if (typeof event.webkitCompassHeading === "number" && Number.isFinite(event.webkitCompassHeading)
      && event.webkitCompassHeading >= 0) {
    return normalizeDegrees(event.webkitCompassHeading);
  }
  if (event.absolute === true && typeof event.alpha === "number" && Number.isFinite(event.alpha)) {
    return normalizeDegrees(360 - event.alpha);
  }
  return null;
}

export function normalizeDegrees(value) {
  const wrapped = value % 360;
  const positive = wrapped < 0 ? wrapped + 360 : wrapped;
  return positive >= 360 ? 0 : positive;   // float edge: 359.9999… % 360 can round to 360
}

// /ws/camera_thermal heading packet: 0x05 + struct "<Iff" (seq, deg, accuracy).
// accuracy < 0 means unknown (stability_runtime.HEADING_STRUCT).
export function headingMessage(sequence, degrees, accuracy) {
  const out = new Uint8Array(13);
  const view = new DataView(out.buffer);
  out[0] = MSG_TYPE_HEADING;
  view.setUint32(1, sequence >>> 0, true);
  // float32 can round 359.99999 up to 360, which the server refuses.
  view.setFloat32(5, Math.fround(degrees) >= 360 ? 0 : degrees, true);
  view.setFloat32(9, Number.isFinite(accuracy) && accuracy >= 0 ? accuracy : -1, true);
  return out;
}

// Frame size for a 640-px-wide copy of a video track, keeping its aspect ratio.
export function frameSize(videoWidth, videoHeight, targetWidth = FRAME_WIDTH) {
  if (!videoWidth || !videoHeight) return null;
  const width = Math.min(targetWidth, videoWidth);
  return { width, height: Math.round((videoHeight * width) / videoWidth) };
}

// Server text on /ws_audio (ConversationServerEvent.decode in the app).
export function decodeServerEvent(text) {
  if (text === "OK:STARTED") return { type: "started" };
  if (text === "OK:STOPPED") return { type: "stopped" };
  if (text === "RESET") return { type: "reset" };
  const parts = text.split(":");
  if (parts.length === 3 && parts[0] === "TTS" && /^-?\d+$/.test(parts[2])) {
    const turnId = Number(parts[2]);
    if (parts[1] === "START") return { type: "ttsStart", turnId };
    if (parts[1] === "END") return { type: "ttsEnd", turnId };
    if (parts[1] === "RESET") return { type: "ttsReset", turnId };
  }
  return { type: "unknown", text };
}

// Trial form -> POST /internal/trial body. Same field names as the /trials form.
export function trialRequestBody(form) {
  const list = (value) =>
    String(value || "").split(",").map((item) => item.trim()).filter(Boolean);
  const body = {
    trial: String(form.trial || "").trim(),
    pid: String(form.pid || "").trim(),
    site: String(form.site || "").trim(),
    cond: String(form.cond || "").trim(),
    task: String(form.task || "").trim(),
    success_criterion: String(form.success_criterion || "").trim(),
  };
  const groundTruth = {
    room: String(form.room || "").trim(),
    present: list(form.present),
    absent: list(form.absent),
  };
  if (groundTruth.room || groundTruth.present.length || groundTruth.absent.length) {
    body.ground_truth = groundTruth;
  }
  return body;
}

export const TRIAL_ID_PATTERN = /^[A-Za-z0-9-]{1,32}$/;

export function trialFormError(body) {
  if (!body.trial || !body.pid) return "Trial and participant are required.";
  if (!TRIAL_ID_PATTERN.test(body.trial) || !TRIAL_ID_PATTERN.test(body.pid)) {
    return "Trial and participant: letters, digits and '-' only (max 32).";
  }
  return "";
}
