// Browser field-test page (/test): the phone's camera and mic, from Safari,
// for internal trials without installing the iOS app. It speaks the same
// protocol as the app (ai-glasses-link ConversationModel + PhoneSensorClient):
//   /ws_audio                       START on connect; push-to-talk sends
//                                   SPEECH_START:<n>, 640-byte PCM chunks,
//                                   SPEECH_END:<n>; replies arrive as
//                                   TTS:START/END/RESET + 8 kHz PCM16.
//   /ws/camera_thermal?source=phone 0x01 + seq-stamped JPEG, 4 frames/s.
// Trials are opened and closed through /internal/trial, exactly as the
// /trials page does (the trial_token cookie authorises both).
// The core module is loaded with this file's own ?v= so a deploy never pairs
// a new page script with a cached old core (the route versions both by mtime).
const {
  FRAME_INTERVAL_MS, MAX_JPEG_BYTES, MIC_SAMPLE_RATE, TTS_SAMPLE_RATE,
  PcmChunker, cameraMessage, decodeServerEvent, frameSize, makeDeviceId,
  newResampleState, pcm16BytesToFloat, resampleToInt16, stampJpegSequence,
  trialFormError, trialRequestBody,
} = await import(`./field_test_core.mjs${new URL(import.meta.url).search}`);

const $ = (selector) => document.querySelector(selector);
const STICKY_KEY = "field_test_sticky";
const DEVICE_KEY = "field_test_device";
const RECONNECT_MAX_MS = 5000;
const SENSOR_OPEN_TIMEOUT_MS = 8000;

const STATE_LABELS = {
  disconnected: "Connecting…",
  idle: "Ready — hold to ask",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking: "Speaking…",
};

const run = {
  active: false,
  trialId: null,
  stream: null,
  audioCtx: null,
  micSource: null,
  micProcessor: null,
  micSink: null,
  resample: newResampleState(),
  chunker: new PcmChunker(),
  audioWs: null,
  sensorWs: null,
  audioRetry: 0,
  sensorRetry: 0,
  audioTimer: null,
  sensorTimer: null,
  frameTimer: null,
  frameBusy: false,
  frameSeq: 0,
  framesSent: 0,
  framesSkipped: 0,
  state: "disconnected",
  micGate: "open",
  turnSeq: 0,
  activeTurn: null,
  turnsAsked: 0,
  playbackTurn: null,
  measuredLatency: false,
  speechEndTimes: new Map(),
  sources: new Set(),
  playEnd: 0,
  ttsEndPending: false,
  drainTimer: null,
  pressHeld: false,
  wakeLock: null,
  cameraNote: "",
};

// ---------- small helpers ----------

function readStorage(key) {
  try { return window.localStorage.getItem(key); } catch { return null; }
}

function writeStorage(key, value) {
  try { window.localStorage.setItem(key, value); } catch { /* private mode */ }
}

function deviceId() {
  let id = readStorage(DEVICE_KEY);
  if (!id || !/^[A-Za-z0-9._:-]{1,64}$/.test(id)) {
    id = makeDeviceId();
    writeStorage(DEVICE_KEY, id);
  }
  return id;
}

function wsUrl(path) {
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${window.location.host}${path}`;
}

function showError(message) {
  const box = $("#error");
  box.textContent = message || "";
  box.hidden = !message;
}

async function postJSON(path, body) {
  const response = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let payload = {};
  try { payload = await response.json(); } catch { /* empty body */ }
  if (!response.ok) {
    throw new Error(payload.error || `${path} failed (${response.status})`);
  }
  return payload;
}

// ---------- UI ----------

function setState(next) {
  if (run.state === next) return;
  run.state = next;
  render();
  if (navigator.vibrate) navigator.vibrate(next === "listening" ? 30 : 0);
}

function render() {
  $("#state").textContent = STATE_LABELS[run.state] || run.state;
  $("#state").dataset.state = run.state;
  const ask = $("#ask");
  ask.disabled = run.state === "disconnected";
  ask.classList.toggle("held", run.state === "listening");
  ask.textContent = run.state === "listening" ? "Release to send" : "Hold to ask";
  setDot("#micDot", run.audioWs && run.audioWs.readyState === WebSocket.OPEN);
  setDot("#camDot", run.sensorWs && run.sensorWs.readyState === WebSocket.OPEN);
  $("#frames").textContent = String(run.framesSent);
  $("#skipped").textContent = String(run.framesSkipped);
  $("#turns").textContent = String(run.turnsAsked);
  $("#camNote").textContent = run.cameraNote;
  $("#camNote").hidden = !run.cameraNote;
}

function setDot(selector, ok) {
  const dot = $(selector);
  dot.classList.toggle("ok", Boolean(ok));
  dot.classList.toggle("bad", !ok);
}

function showPanel(name) {
  for (const panel of ["setup", "live", "done"]) $(`#${panel}`).hidden = panel !== name;
}

function loadSticky() {
  let saved = {};
  try { saved = JSON.parse(readStorage(STICKY_KEY) || "{}"); } catch { saved = {}; }
  for (const key of ["pid", "site", "cond"]) {
    if (saved[key]) $(`#f_${key}`).value = saved[key];
  }
}

function formValues() {
  const values = {};
  for (const key of ["trial", "pid", "site", "cond", "task", "success_criterion", "room", "present", "absent"]) {
    values[key] = $(`#f_${key}`).value;
  }
  return values;
}

// ---------- start / end ----------

async function startTrial(event) {
  event.preventDefault();
  showError("");
  const body = trialRequestBody(formValues());
  const formError = trialFormError(body);
  if (formError) { showError(formError); return; }
  const button = $("#start");
  button.disabled = true;
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("This browser cannot open the camera here. Use Safari over https.");
    }
    // Both permissions in one prompt, from this tap (Safari requires a gesture).
    run.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
    const Ctx = window.AudioContext || window.webkitAudioContext;
    run.audioCtx = new Ctx();
    await run.audioCtx.resume();

    run.active = true;
    // The camera socket must be up before the trial opens: meta.json stamps
    // sensor_source once, at open, and it stays null if no device is connected.
    connectSensor();
    await waitForSensor(SENSOR_OPEN_TIMEOUT_MS);

    await postJSON("/internal/trial", body);
    writeStorage(STICKY_KEY, JSON.stringify({ pid: body.pid, site: body.site, cond: body.cond }));

    run.trialId = body.trial;
    $("#liveTrial").textContent = body.trial;
    showPanel("live");

    const video = $("#video");
    video.srcObject = run.stream;
    await video.play().catch(() => {});

    startMic();
    connectAudio();
    run.frameTimer = window.setInterval(sendFrame, FRAME_INTERVAL_MS);
    requestWakeLock();
    render();
  } catch (error) {
    run.active = false;
    window.clearTimeout(run.sensorTimer);
    if (run.sensorWs) {
      run.sensorWs.onclose = null;
      try { run.sensorWs.close(1000, "start_failed"); } catch { /* closed */ }
      run.sensorWs = null;
    }
    stopMedia();
    showError(error.message || String(error));
  } finally {
    button.disabled = false;
  }
}

async function endTrial() {
  if (!window.confirm(`End trial ${run.trialId}? Scoring starts on the server.`)) return;
  const button = $("#end");
  button.disabled = true;
  showError("");
  if (run.state === "listening" && run.activeTurn !== null) {
    sendAudioText(`SPEECH_END:${run.activeTurn}`);
  }
  sendAudioText("STOP");
  run.active = false;
  window.clearInterval(run.frameTimer);
  window.clearTimeout(run.audioTimer);
  window.clearTimeout(run.sensorTimer);
  window.clearTimeout(run.drainTimer);
  stopPlayback();
  for (const ws of [run.audioWs, run.sensorWs]) {
    if (ws) { ws.onclose = null; try { ws.close(1000, "trial_end"); } catch { /* closed */ } }
  }
  run.audioWs = null;
  run.sensorWs = null;
  stopMedia();
  releaseWakeLock();
  try {
    await postJSON("/internal/trial/end", {});
    $("#doneTrial").textContent = run.trialId;
    $("#doneLink").href = `/trials/${encodeURIComponent(run.trialId)}`;
    $("#doneStats").textContent =
      `${run.turnsAsked} questions, ${run.framesSent} frames sent, ${run.framesSkipped} skipped.`;
    showPanel("done");
  } catch (error) {
    showError(`The trial may still be open on the server: ${error.message}. End it from /trials.`);
  } finally {
    button.disabled = false;
    resetRun();
  }
}

function resetRun() {
  Object.assign(run, {
    state: "disconnected", micGate: "open", turnSeq: 0,
    activeTurn: null, turnsAsked: 0, playbackTurn: null, frameSeq: 0,
    framesSent: 0, framesSkipped: 0, audioRetry: 0, sensorRetry: 0,
    speechEndTimes: new Map(), ttsEndPending: false, pressHeld: false, cameraNote: "",
  });
}

function stopMedia() {
  if (run.micProcessor) { run.micProcessor.onaudioprocess = null; try { run.micProcessor.disconnect(); } catch { /* */ } }
  if (run.micSource) { try { run.micSource.disconnect(); } catch { /* */ } }
  if (run.micSink) { try { run.micSink.disconnect(); } catch { /* */ } }
  run.micProcessor = run.micSource = run.micSink = null;
  if (run.stream) for (const track of run.stream.getTracks()) track.stop();
  run.stream = null;
  if (run.audioCtx) { run.audioCtx.close().catch(() => {}); run.audioCtx = null; }
  const video = $("#video");
  if (video) video.srcObject = null;
}

// ---------- microphone ----------

function startMic() {
  const ctx = run.audioCtx;
  run.micSource = ctx.createMediaStreamSource(run.stream);
  // ScriptProcessor rather than an AudioWorklet: works in every Safari that
  // has getUserMedia, and needs no extra module file.
  run.micProcessor = ctx.createScriptProcessor(4096, 1, 1);
  run.micSink = ctx.createGain();
  run.micSink.gain.value = 0;   // Safari only runs the processor when it reaches a destination
  run.micProcessor.onaudioprocess = (event) => {
    if (run.state !== "listening" || run.micGate !== "open") return;
    const ws = run.audioWs;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const input = event.inputBuffer.getChannelData(0);
    const pcm = resampleToInt16(input, ctx.sampleRate, MIC_SAMPLE_RATE, run.resample);
    for (const chunk of run.chunker.push(pcm)) ws.send(chunk);
  };
  run.micSource.connect(run.micProcessor);
  run.micProcessor.connect(run.micSink);
  run.micSink.connect(ctx.destination);
}

function beginAsk(event) {
  if (event) event.preventDefault();
  // Same guard as the app: a press while "thinking" is allowed, so a reply
  // that never comes cannot lock the button.
  if (!run.active || run.state === "disconnected" || run.state === "listening") return;
  run.pressHeld = true;
  if (run.audioCtx && run.audioCtx.state !== "running") run.audioCtx.resume().catch(() => {});
  if (run.state === "speaking") {
    sendAudioText("BARGE_IN");
    stopPlayback();
    openMicGate();
  }
  run.turnSeq += 1;
  run.activeTurn = run.turnSeq;
  for (const key of run.speechEndTimes.keys()) {
    if (key < run.activeTurn - 2) run.speechEndTimes.delete(key);
  }
  run.resample = newResampleState();
  run.chunker.reset();
  sendAudioText(`SPEECH_START:${run.activeTurn}`);
  run.micGate = "open";
  run.turnsAsked += 1;
  setState("listening");
}

function endAsk(event) {
  if (event) event.preventDefault();
  run.pressHeld = false;
  if (run.state !== "listening" || run.activeTurn === null) return;
  sendAudioText(`SPEECH_END:${run.activeTurn}`);
  run.speechEndTimes.set(run.activeTurn, performance.now());
  setState("thinking");
}

function openMicGate() {
  if (run.micGate === "open") return;
  run.micGate = "open";
  sendAudioText("MIC_GATE:OPEN");
}

// ---------- /ws_audio ----------

function sendAudioText(text) {
  const ws = run.audioWs;
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(text);
}

function connectAudio() {
  if (!run.active) return;
  const ws = new WebSocket(wsUrl("/ws_audio"));
  ws.binaryType = "arraybuffer";
  run.audioWs = ws;
  ws.onopen = () => {
    run.audioRetry = 0;
    ws.send("START");      // the app sends START on every (re)connect
    render();
  };
  ws.onmessage = (message) => {
    if (typeof message.data === "string") handleServerEvent(decodeServerEvent(message.data));
    else handleReplyAudio(new Uint8Array(message.data));
  };
  ws.onerror = () => {};
  ws.onclose = () => {
    if (run.audioWs !== ws) return;
    run.audioWs = null;
    stopPlayback();
    run.micGate = "open";
    run.activeTurn = null;
    setState("disconnected");
    if (!run.active) return;
    const delay = Math.min(RECONNECT_MAX_MS, 500 * 2 ** run.audioRetry);
    run.audioRetry += 1;
    run.audioTimer = window.setTimeout(connectAudio, delay);
  };
}

function handleServerEvent(event) {
  switch (event.type) {
    case "started":
      if (run.state === "disconnected") setState("idle");
      break;
    case "stopped":
      break;
    case "ttsStart":
      run.playbackTurn = event.turnId;
      run.measuredLatency = false;
      run.ttsEndPending = false;
      run.micGate = "tts";
      sendAudioText("MIC_GATE:TTS");
      setState("speaking");
      break;
    case "ttsEnd":
      if (run.playbackTurn !== event.turnId) return;
      run.ttsEndPending = true;
      scheduleDrain();
      break;
    case "ttsReset":
      if (run.playbackTurn !== null && run.playbackTurn !== event.turnId) return;
      stopPlayback();
      run.playbackTurn = null;
      openMicGate();
      if (run.state === "speaking") setState("idle");
      break;
    case "reset":
      stopPlayback();
      run.playbackTurn = null;
      openMicGate();
      if (run.state !== "listening" && run.state !== "disconnected") setState("idle");
      break;
    default:
      break;
  }
}

function handleReplyAudio(bytes) {
  const ctx = run.audioCtx;
  if (!ctx || !bytes.byteLength) return;
  if (!run.measuredLatency && run.playbackTurn !== null && run.speechEndTimes.has(run.playbackTurn)) {
    const elapsed = performance.now() - run.speechEndTimes.get(run.playbackTurn);
    run.speechEndTimes.delete(run.playbackTurn);
    sendAudioText(`LATENCY:DEVICE:${run.playbackTurn}:${Math.max(0, Math.round(elapsed))}`);
    run.measuredLatency = true;
  }
  const samples = pcm16BytesToFloat(bytes, TTS_SAMPLE_RATE, ctx.sampleRate);
  const buffer = ctx.createBuffer(1, samples.length, ctx.sampleRate);
  buffer.getChannelData(0).set(samples);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  // A small lead on the first chunk absorbs network jitter between chunks.
  const startAt = Math.max(ctx.currentTime + 0.12, run.playEnd);
  source.start(startAt);
  run.playEnd = startAt + buffer.duration;
  run.sources.add(source);
  source.onended = () => run.sources.delete(source);
  if (run.ttsEndPending) scheduleDrain();
}

// After TTS:END, wait for the scheduled audio to finish, then reopen the mic
// gate (the app's PacedAudioPlayer.onDrained).
function scheduleDrain() {
  window.clearTimeout(run.drainTimer);
  const ctx = run.audioCtx;
  const remainingMs = ctx ? Math.max(0, (run.playEnd - ctx.currentTime) * 1000) : 0;
  run.drainTimer = window.setTimeout(() => {
    if (!run.ttsEndPending) return;
    run.ttsEndPending = false;
    openMicGate();
    if (run.state === "speaking") setState("idle");
  }, remainingMs + 50);
}

function stopPlayback() {
  window.clearTimeout(run.drainTimer);
  run.ttsEndPending = false;
  for (const source of run.sources) {
    source.onended = null;
    try { source.stop(); } catch { /* already stopped */ }
  }
  run.sources.clear();
  run.playEnd = 0;
}

// ---------- /ws/camera_thermal ----------

function connectSensor() {
  if (!run.active) return;
  const url = wsUrl(`/ws/camera_thermal?source=phone&device=${encodeURIComponent(deviceId())}`);
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  run.sensorWs = ws;
  ws.onopen = () => {
    run.sensorRetry = 0;
    run.cameraNote = "";
    render();
  };
  ws.onerror = () => {};
  ws.onclose = (event) => {
    if (run.sensorWs !== ws) return;
    run.sensorWs = null;
    if (event.code === 1013) {
      run.cameraNote = "Camera slot is taken (glasses or another phone connected). Disconnect it; retrying…";
    } else if (event.code === 1008) {
      run.cameraNote = `Server refused the camera: ${event.reason || "bad parameters"}.`;
      render();
      return;   // a parameter error will not fix itself
    } else {
      run.cameraNote = "Camera connection dropped; reconnecting…";
    }
    render();
    if (!run.active) return;
    const delay = Math.min(RECONNECT_MAX_MS, 500 * 2 ** run.sensorRetry);
    run.sensorRetry += 1;
    run.sensorTimer = window.setTimeout(connectSensor, delay);
  };
}

function waitForSensor(timeoutMs) {
  return new Promise((resolve, reject) => {
    const started = performance.now();
    const check = () => {
      if (run.sensorWs && run.sensorWs.readyState === WebSocket.OPEN) { resolve(); return; }
      if (!run.active) { reject(new Error("Start was cancelled.")); return; }
      if (performance.now() - started > timeoutMs) {
        reject(new Error(run.cameraNote || "Could not connect the camera to the server. Check the internet and try again."));
        return;
      }
      window.setTimeout(check, 100);
    };
    check();
  });
}

function encodeJpeg(canvas, quality) {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", quality);
  });
}

async function sendFrame() {
  const ws = run.sensorWs;
  const video = $("#video");
  if (!run.active || run.frameBusy || !ws || ws.readyState !== WebSocket.OPEN) return;
  if (video.readyState < 2) return;
  // Never queue behind a slow uplink: a stale frame is worse than none.
  if (ws.bufferedAmount > MAX_JPEG_BYTES) { run.framesSkipped += 1; render(); return; }
  const size = frameSize(video.videoWidth, video.videoHeight);
  if (!size) return;
  run.frameBusy = true;
  try {
    const canvas = $("#canvas");
    canvas.width = size.width;
    canvas.height = size.height;
    canvas.getContext("2d").drawImage(video, 0, 0, size.width, size.height);
    let blob = await encodeJpeg(canvas, 0.8);
    if (blob && blob.size > MAX_JPEG_BYTES - 64) blob = await encodeJpeg(canvas, 0.5);
    if (!blob || blob.size > MAX_JPEG_BYTES - 64) { run.framesSkipped += 1; return; }
    const jpeg = new Uint8Array(await blob.arrayBuffer());
    run.frameSeq += 1;
    const message = cameraMessage(stampJpegSequence(jpeg, run.frameSeq));
    if (run.sensorWs === ws && ws.readyState === WebSocket.OPEN) {
      ws.send(message);
      run.framesSent += 1;
    }
  } catch (error) {
    run.framesSkipped += 1;
    run.cameraNote = `Frame error: ${error.message || error}`;
  } finally {
    run.frameBusy = false;
    render();
  }
}

// ---------- screen ----------

async function requestWakeLock() {
  try {
    if ("wakeLock" in navigator) run.wakeLock = await navigator.wakeLock.request("screen");
  } catch { /* not supported or refused: the note on screen covers it */ }
}

function releaseWakeLock() {
  if (run.wakeLock) run.wakeLock.release().catch(() => {});
  run.wakeLock = null;
}

document.addEventListener("visibilitychange", () => {
  if (!run.active) return;
  if (document.visibilityState === "hidden") {
    // Safari stops the camera and mic in a background tab.
    if (run.state === "listening") endAsk();
    return;
  }
  $("#video").play().catch(() => {});
  if (run.audioCtx && run.audioCtx.state !== "running") run.audioCtx.resume().catch(() => {});
  requestWakeLock();
});

window.addEventListener("pagehide", () => {
  if (run.active) sendAudioText("STOP");
});

// ---------- wiring ----------

function wire() {
  loadSticky();
  $("#setupForm").addEventListener("submit", startTrial);
  $("#end").addEventListener("click", endTrial);
  $("#again").addEventListener("click", () => {
    $("#f_trial").value = "";
    showError("");
    showPanel("setup");
  });
  const ask = $("#ask");
  ask.addEventListener("pointerdown", (event) => {
    try { ask.setPointerCapture(event.pointerId); } catch { /* */ }
    beginAsk(event);
  });
  ask.addEventListener("pointerup", endAsk);
  ask.addEventListener("pointercancel", endAsk);
  ask.addEventListener("contextmenu", (event) => event.preventDefault());
  ask.addEventListener("keydown", (event) => {
    if ((event.key === " " || event.key === "Enter") && !event.repeat) beginAsk(event);
  });
  ask.addEventListener("keyup", (event) => {
    if (event.key === " " || event.key === "Enter") endAsk(event);
  });
  showPanel("setup");
  render();
}

wire();
