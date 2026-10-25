(() => {
  "use strict";

  const $ = selector => document.querySelector(selector);
  const node = (tag, className, text) => {
    const value = document.createElement(tag);
    if (className) value.className = className;
    if (text != null) value.textContent = text;
    return value;
  };
  const LABELS = {
    right: ["right", "v-right"],
    useful_refusal: ["useful refusal", "v-useful"],
    dead_refusal: ["dead refusal", "v-dead"],
    bad_guess: ["bad guess", "v-bad"],
    unscored: ["unscored", "v-unscored"]
  };

  async function getJSON(url) {
    const response = await fetch(url, {credentials: "same-origin", headers: {Accept: "application/json"}});
    if (!response.ok) {
      let message = `Request failed (${response.status})`;
      try { message = (await response.json()).error || message; } catch (_) {}
      throw new Error(message);
    }
    return response.json();
  }

  // The trial_token cookie the page already holds authorises these; nothing is
  // typed or pasted on the phone.
  async function postJSON(url, body) {
    const response = await fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: {Accept: "application/json", "Content-Type": "application/json"},
      body: JSON.stringify(body || {})
    });
    let payload = null;
    try { payload = await response.json(); } catch (_) {}
    if (!response.ok) throw new Error((payload && payload.error) || `Request failed (${response.status})`);
    return payload || {};
  }

  function percent(rate) {
    const value = Number(rate);
    return Number.isFinite(value) ? Math.round(value * 100) : 0;
  }

  function dateLabel(value) {
    if (!value) return "—";
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? value : new Intl.DateTimeFormat(undefined, {
      day: "numeric", month: "short", year: "numeric"
    }).format(parsed);
  }

  function renderIndex(rows) {
    const host = $("#idx");
    host.replaceChildren();
    for (const trial of rows) {
      const scored = trial.scored !== false && trial.bad_guess_rate != null;
      const pct = percent(trial.bad_guess_rate);
      const row = node("tr");
      row.tabIndex = 0;
      row.setAttribute("role", "link");
      row.setAttribute("aria-label", `Open trial ${trial.trial || trial.id}`);
      const trialCell = node("td");
      trialCell.append(node("span", "tid", trial.trial || trial.id));
      trialCell.append(node("div", "sub", `participant ${trial.participant || "—"}`));
      row.append(trialCell, node("td", null, trial.site || "—"), node("td", null, trial.condition || "—"));
      row.append(node("td", "num", trial.prompt_version || "—"));
      row.append(node("td", "r num", trial.turn_count == null ? "—" : String(trial.turn_count)));
      const rateCell = node("td", "r");
      if (scored) {
        const bar = node("span", "bar");
        bar.append(node("span", "num", `${pct}%`));
        const track = node("i");
        const fill = node("b");
        fill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
        track.append(fill); bar.append(track); rateCell.append(bar);
      } else {
        rateCell.append(node("span", "num sub", "not scored"));
      }
      row.append(rateCell);
      const status = trial.status || (trial.scored ? "scored" : "unscored");
      const statusCell = node("td");
      const badge = node("span", `st st-${status}`, status === "unscored" ? "not scored" : status);
      if (status === "failed" && trial.job && trial.job.last_error) badge.title = trial.job.last_error;
      statusCell.append(badge);
      row.append(statusCell);
      row.append(node("td", "num sub", dateLabel(trial.date)));
      const open = () => { window.location.href = `/trials/${encodeURIComponent(trial.id)}`; };
      row.addEventListener("click", open);
      row.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); open(); }
      });
      host.append(row);
    }
    const state = $("#indexState");
    state.textContent = rows.length ? "" : "No scored trials are available yet.";
    state.hidden = Boolean(rows.length);
  }

  function addGroundTruth(groundTruth) {
    const host = $("#gt");
    host.replaceChildren(node("b", null, "Ground truth — written before the run"));
    host.append(node("p", null, groundTruth.room || "Room not recorded"));
    host.append(node("p", null, `In the room: ${(groundTruth.present || []).join(", ") || "—"}`));
    const absent = node("p");
    absent.append("Not here: ", node("span", "absent", (groundTruth.absent || []).join(", ") || "—"));
    host.append(absent);
    const planted = node("p");
    planted.append("Planted as unseeable: ", node("span", "plant", (groundTruth.planted_unseeable || []).join(", ") || "—"));
    host.append(planted);
    host.append(node("p", null, `Visible from start: ${(groundTruth.visible_from_start || []).join(", ") || "—"}`));
  }

  function thermalLabel(thermal) {
    if (!thermal) return "not captured";
    const bits = [];
    if (thermal.ahead_c != null) bits.push(`${thermal.ahead_c} °C ahead`);
    if (thermal.ambient_c != null) bits.push(`${thermal.ambient_c} °C ambient`);
    if (thermal.hotspot_c != null) bits.push(`${thermal.hotspot_c} °C hotspot`);
    return bits.join(" · ") || "not captured";
  }

  function latencyLabel(latency) {
    if (!latency) return "not recorded";
    // Name the measure: when the model answers before speech_end, only the
    // first-mic measure exists and must not be passed off as end-of-speech latency.
    const measures = [
      ["speech_end_to_first_gemini_audio_ms", "from end of speech to first model audio"],
      ["first_mic_to_first_gemini_audio_ms", "from first mic audio to first model audio"],
      ["first_gemini_audio_received_ms", "from turn start to first model audio"]
    ];
    for (const [key, label] of measures) {
      if (latency[key] != null) return `${Math.round(latency[key])} ms ${label}`;
    }
    return latency.status || "not recorded";
  }

  function frameHeading(windowData, index) {
    // Frames carry server wall time; IMU samples carry device uptime. Align
    // both on the window start: elapsed time since the first frame maps onto
    // elapsed uptime since the speech_start IMU sample.
    const samples = windowData.heading || [];
    const frames = windowData.frames || [];
    if (!samples.length || !frames[index]) return null;
    const first = Date.parse(frames[0].ts);
    const here = Date.parse(frames[index].ts);
    const base = Number(samples[0].uptime_ms);
    if (!Number.isFinite(first) || !Number.isFinite(here) || !Number.isFinite(base)) {
      const fraction = frames.length > 1 ? index / (frames.length - 1) : 0;
      return samples[Math.round(fraction * (samples.length - 1))]?.heading_deg ?? null;
    }
    const target = base + (here - first);
    // IMU streams only while a server turn is open; past the last sample the
    // heading is unknown, not frozen at the last value.
    if (target > Number(samples[samples.length - 1].uptime_ms) + 1000) return null;
    let best = samples[0];
    for (const sample of samples) {
      if (Number(sample.uptime_ms) <= target) best = sample; else break;
    }
    return best.heading_deg ?? null;
  }

  function promptText(windowData) {
    const prompt = windowData.prompt || {};
    const sent = prompt.sent || {};
    const lines = [`system_instruction  ${prompt.version || "unversioned"}  sha256 ${(prompt.sha256 || "—").slice(0, 12)}`, "", "realtime_input"];
    lines.push(sent.frames != null
      ? `  video  image/jpeg × ${sent.frames}   seq ${sent.seq_first ?? "—"}–${sent.seq_last ?? "—"}`
      : "  video  not recorded");
    lines.push(sent.audio_ms != null ? `  audio  pcm;16000  ${Math.round(sent.audio_ms)} ms user speech` : "  audio  not recorded");
    if (sent.thermal_text) {
      let text = sent.thermal_text;
      const brace = text.indexOf("{");
      if (brace >= 0) {
        try { text = text.slice(0, brace) + JSON.stringify(JSON.parse(text.slice(brace)), null, 2); } catch (_) {}
      }
      lines.push("  text   " + text.replace(/\n/g, "\n         "));
    } else {
      lines.push("  text   no thermal text sent");
    }
    return lines.join("\n");
  }

  function paintFrame(canvas, windowData, index, status, headingValue) {
    const frames = windowData.frames || [];
    const frame = frames[index];
    if (!frame) {
      canvas.width = 680; canvas.height = 382;
      const context = canvas.getContext("2d");
      context.fillStyle = "#0c0d0f"; context.fillRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = "#95969c"; context.font = "14px IBM Plex Mono, monospace";
      context.fillText("NO FRAME CAPTURED IN THIS WINDOW", 24, 40);
      status.textContent = "no frames";
      headingValue.textContent = "not recorded";
      return;
    }
    const image = new Image();
    const requested = frame.url;
    canvas.dataset.requested = requested;
    image.onload = () => {
      if (canvas.dataset.requested !== requested) return;
      canvas.width = image.naturalWidth || 680;
      canvas.height = image.naturalHeight || 382;
      const context = canvas.getContext("2d");
      context.drawImage(image, 0, 0, canvas.width, canvas.height);

      const hotspot = windowData.thermal?.hotspot_rgb_xy;
      if (Array.isArray(hotspot) && hotspot.length === 2) {
        const hx = Math.max(0, Math.min(1, Number(hotspot[0]))) * canvas.width;
        const hy = Math.max(0, Math.min(1, Number(hotspot[1]))) * canvas.height;
        const radius = Math.max(18, Math.min(canvas.width, canvas.height) * 0.09);
        const glow = context.createRadialGradient(hx, hy, 2, hx, hy, radius);
        glow.addColorStop(0, "rgba(249,115,22,.58)"); glow.addColorStop(1, "rgba(249,115,22,0)");
        context.fillStyle = glow; context.beginPath(); context.arc(hx, hy, radius, 0, Math.PI * 2); context.fill();
        context.strokeStyle = "#f97316"; context.lineWidth = 2;
        context.beginPath(); context.arc(hx, hy, Math.max(8, radius * .28), 0, Math.PI * 2); context.stroke();
      }

      const heading = frameHeading(windowData, index);
      const stripY = canvas.height - 26;
      context.fillStyle = "rgba(0,0,0,.58)"; context.fillRect(0, stripY - 14, canvas.width, 40);
      context.strokeStyle = "rgba(255,255,255,.35)"; context.lineWidth = 1;
      context.beginPath(); context.moveTo(14, stripY); context.lineTo(canvas.width - 14, stripY); context.stroke();
      if (heading != null) {
        const clamped = Math.max(-180, Math.min(180, heading));
        const x = 14 + (canvas.width - 28) * ((clamped + 180) / 360);
        context.fillStyle = "#7d92ff"; context.beginPath();
        context.moveTo(x, stripY - 2); context.lineTo(x - 5, stripY - 11); context.lineTo(x + 5, stripY - 11); context.closePath(); context.fill();
        context.fillStyle = "#fff"; context.font = "500 11px IBM Plex Mono, monospace";
        context.fillText(`${Math.round(heading)}°`, Math.min(x + 8, canvas.width - 38), stripY - 5);
      }
      headingValue.textContent = heading == null
        ? ((windowData.heading || []).length ? "no IMU samples at this frame" : "not recorded")
        : `turned ${heading.toFixed(0)}° since the question began`;
      const age = frame.age_ms == null ? "age unknown" : `${Math.round(frame.age_ms)} ms old`;
      status.textContent = `frame ${index + 1}/${frames.length} · seq ${frame.seq ?? "—"} · ${age}`;
    };
    image.onerror = () => {
      if (canvas.dataset.requested === requested) status.textContent = `frame ${index + 1}/${frames.length} unavailable`;
    };
    image.src = requested;
  }

  function appendMeta(host, label, value) {
    host.append(node("dt", null, label), node("dd", null, value == null || value === "" ? "—" : String(value)));
  }

  function evidence(windowData) {
    const score = windowData.score || {};
    const panel = node("div", "ev"); panel.hidden = true;
    const visual = node("div");
    const canvas = node("canvas"); canvas.width = 680; canvas.height = 382;
    const scrub = node("div", "scrub");
    const range = node("input"); range.type = "range"; range.min = "0";
    range.max = String(Math.max(0, (windowData.frames || []).length - 1)); range.value = "0";
    range.disabled = !(windowData.frames || []).length;
    range.setAttribute("aria-label", `Frame within turn ${windowData.number}`);
    const frameStatus = node("span", null, "loading frame…");
    scrub.append(range, frameStatus); visual.append(canvas, scrub);

    const facts = node("div");
    const scored = Boolean(windowData.score);
    facts.append(node("p", "why", scored ? (score.why || "The judge gave no reason.") : "Not scored yet — no judge reason."));
    const detectors = node("div", "dets");
    if (scored) {
      detectors.append(node("span", `det${score.distance_units ? " hit" : ""}`,
        score.distance_units ? `distance: “${score.distance_units}”` : "distance units: none"));
      detectors.append(node("span", "det", `${score.words ?? "—"} words`));
      detectors.append(node("span", "det", `frame: ${score.spatial_frame || "none"}`));
    } else {
      detectors.append(node("span", "det", "detectors not run"));
    }
    if (windowData.answer_arrived_under) detectors.append(node("span", "det hit", `answer UID: ${windowData.answer_arrived_under}`));
    facts.append(detectors);

    const meta = node("dl", "meta2");
    const seqs = (windowData.frames || []).map(frame => frame.seq).filter(value => value != null);
    appendMeta(meta, "Target", windowData.target);
    appendMeta(meta, "Frames", `${(windowData.frames || []).length}${seqs.length ? ` · seq ${seqs[0]}–${seqs[seqs.length - 1]}` : ""}`);
    appendMeta(meta, "Thermal", thermalLabel(windowData.thermal));
    const headingValue = node("dd", null, "not recorded");
    meta.append(node("dt", null, "IMU"), headingValue);
    appendMeta(meta, "Latency", latencyLabel(windowData.latency));
    appendMeta(meta, "Session", `generation ${windowData.session_generation ?? "—"} · ${(windowData.latency || {}).status || "status unknown"}`);
    facts.append(meta);
    const prompt = node("pre", "pr"); prompt.textContent = promptText(windowData); facts.append(prompt);
    panel.append(visual, facts);

    let frameIndex = 0;
    const paint = () => paintFrame(canvas, windowData, frameIndex, frameStatus, headingValue);
    range.addEventListener("input", event => { frameIndex = Number(event.target.value); paint(); });
    return {panel, paint};
  }

  function renderTrial(data) {
    const summary = data.summary || {};
    const counts = summary.counts || {};
    const windows = data.windows || [];
    const scored = data.scored ?? Boolean(data.summary && Object.keys(data.summary).length);
    const total = Number(summary.turn_count ?? windows.length);
    const bad = Number(counts.bad_guess ?? 0);
    if (scored) {
      $("#dRate").textContent = `${percent(summary.bad_guess_rate)}%`;
      $("#dCap").textContent = `${bad} of ${total} answers stated something the glasses could not see.`;
    } else {
      $("#dRate").textContent = "—";
      $("#dRate").classList.add("unscored");
      $("#dCap").textContent = `Not scored yet — ${windows.length} question windows recorded.`;
    }
    const meta = data.meta || {};
    document.title = `${meta.trial || data.id} · Trial Archive`;
    $("#dWho").textContent = [meta.trial || data.id, `participant ${meta.pid || "—"}`, meta.site, meta.cond,
      `prompt ${meta.prompt_version || "—"}`, dateLabel(meta.created_ts)].filter(Boolean).join(" · ");
    const judged = value => (scored ? (value ?? 0) : "—");
    $("#tRight").textContent = judged(counts.right); $("#tUseful").textContent = judged(counts.useful_refusal);
    $("#tDead").textContent = judged(counts.dead_refusal); $("#tBad").textContent = judged(counts.bad_guess);
    $("#tDist").textContent = judged(summary.distance_flags);
    // Misattribution comes from the event log, not the judge, so it is known before scoring.
    $("#tMis").textContent = summary.misattributed_turns ?? windows.filter(item => item.answer_arrived_under).length;
    $("#tDrop").textContent = judged(summary.dropped_events);
    addGroundTruth(data.ground_truth || {});

    const host = $("#turns"); host.replaceChildren();
    let openPanel = null; let openButton = null;
    for (const windowData of data.windows || []) {
      const score = windowData.score || {};
      const label = LABELS[score.label] || LABELS.unscored;
      const wrapper = node("div", "turn");
      const button = node("button"); button.type = "button"; button.setAttribute("aria-expanded", "false");
      const index = node("div", "idx", String(windowData.number).padStart(2, "0"));
      const qa = node("div", "qa");
      qa.append(node("div", "q", `“${windowData.question || "Transcription unavailable"}”`));
      qa.append(node("div", "a", `“${windowData.answer || "No answer recorded"}”`));
      button.append(index, qa, node("div", `verd ${label[1]}`, label[0]));
      const detail = evidence(windowData); wrapper.append(button, detail.panel); host.append(wrapper);
      button.addEventListener("click", () => {
        const show = detail.panel.hidden;
        if (openPanel && openPanel !== detail.panel) {
          openPanel.hidden = true; openButton.setAttribute("aria-expanded", "false");
        }
        detail.panel.hidden = !show; button.setAttribute("aria-expanded", String(show));
        openPanel = show ? detail.panel : null; openButton = show ? button : null;
        if (show) detail.paint();
      });
    }
    const state = $("#trialState");
    state.textContent = data.windows?.length ? "" : "No question windows were recorded.";
    state.hidden = Boolean(data.windows?.length);
  }

  /* ---------------- start / end controls ---------------- */

  const STICKY_KEY = "trialControls.sticky";
  const POLL_MS = 3000;

  const readSticky = () => {
    try { return JSON.parse(localStorage.getItem(STICKY_KEY)) || {}; } catch (_) { return {}; }
  };
  const writeSticky = value => {
    try { localStorage.setItem(STICKY_KEY, JSON.stringify(value)); } catch (_) {}
  };
  // "a, b , ,c" -> ["a","b","c"]; the viewer reads these four keys as arrays.
  const toList = text => String(text || "").split(",").map(item => item.trim()).filter(Boolean);

  function controlError(message) {
    const host = $("#ctlErr");
    host.textContent = message || "";
    host.hidden = !message;
  }

  function showForm(show) {
    $("#ctlForm").hidden = !show;
    $("#ctlIdle").hidden = show;
    if (!show) return;
    const sticky = readSticky();
    for (const [id, key] of [["#fSite", "site"], ["#fCond", "cond"], ["#fPrompt", "prompt_version"]]) {
      if (!$(id).value && sticky[key]) $(id).value = sticky[key];
    }
    $("#fTrial").focus();
  }

  function renderControls(status) {
    $("#ctl").hidden = false;
    const open = Boolean(status.session_id);
    const job = status.job || status.last_job || null;
    const dot = $("#ctlDot");
    const state = $("#ctlState");
    dot.className = "ctl-dot";

    if (open) {
      dot.classList.add("live");
      state.textContent = "recording";
      $("#liveId").textContent = status.session_id;
      $("#liveTurns").textContent = status.turn_count == null ? "—" : String(status.turn_count);
      $("#liveQueue").textContent = status.queue_depth == null
        ? "—" : `${status.queue_depth}/${status.queue_maxsize ?? "—"}`;
      $("#ctlSub").textContent = status.untitled ? "untitled session — set a trial id to name it" : "";
    } else if (job && (job.state === "queued" || job.state === "scoring")) {
      dot.classList.add("work");
      state.textContent = job.state;
      $("#ctlSub").textContent = `${job.trial_id} · attempt ${job.attempts || 1}`;
    } else if (job && job.state === "failed") {
      dot.classList.add("work");
      state.textContent = "scoring failed";
      $("#ctlSub").textContent = `${job.trial_id} · ${job.last_error || "no reason recorded"}`;
    } else if (job && job.state === "scored") {
      dot.classList.add("done");
      state.textContent = "idle";
      const left = (job.unscored_windows || []).length;
      $("#ctlSub").textContent = `${job.trial_id} scored${left ? ` · ${left} window(s) unscored` : ""}`;
    } else {
      state.textContent = "idle";
      $("#ctlSub").textContent = status.enabled === false ? "recorder disabled (TRIAL_RECORDER=1)" : "";
    }

    $("#ctlOpen").hidden = !open;
    if (open) { $("#ctlForm").hidden = true; $("#ctlIdle").hidden = true; }
    else if ($("#ctlForm").hidden) $("#ctlIdle").hidden = false;
    $("#btnOpen").disabled = status.enabled === false;
  }

  async function refreshControls() {
    try {
      renderControls(await getJSON("/internal/trial"));
    } catch (error) {
      $("#ctl").hidden = false;
      $("#ctlState").textContent = "unavailable";
      $("#ctlSub").textContent = error.message;
    }
  }

  async function refreshIndex() {
    try {
      renderIndex((await getJSON("/api/trials")).trials || []);
    } catch (error) {
      const state = $("#indexState");
      state.hidden = false; state.classList.add("error"); state.textContent = error.message;
    }
  }

  async function submitStart(event) {
    event.preventDefault();
    controlError("");
    const button = $("#btnStart");
    const groundTruth = {room: $("#gRoom").value.trim()};
    for (const [id, key] of [["#gPresent", "present"], ["#gAbsent", "absent"],
                             ["#gPlanted", "planted_unseeable"], ["#gVisible", "visible_from_start"]]) {
      groundTruth[key] = toList($(id).value);
    }
    const body = {
      trial: $("#fTrial").value.trim(),
      pid: $("#fPid").value.trim(),
      site: $("#fSite").value.trim(),
      cond: $("#fCond").value.trim(),
      task: $("#fTask").value.trim(),
      success_criterion: $("#fCrit").value.trim(),
      prompt_version: $("#fPrompt").value.trim(),
      ground_truth: groundTruth
    };
    if (!body.trial || !body.pid) { controlError("Trial and participant are required."); return; }
    button.disabled = true;
    try {
      const status = await postJSON("/internal/trial", body);
      writeSticky({site: body.site, cond: body.cond, prompt_version: body.prompt_version});
      $("#ctlForm").reset();
      renderControls(status);
      refreshIndex();
    } catch (error) {
      controlError(error.message);
    } finally {
      button.disabled = false;
    }
  }

  async function endTrial() {
    controlError("");
    const button = $("#btnEnd");
    const open = $("#liveId").textContent;
    if (!window.confirm(`End ${open}? Scoring starts in the background.`)) return;
    button.disabled = true;
    try {
      const status = await postJSON("/internal/trial/end", {});
      renderControls(status);
      refreshIndex();
    } catch (error) {
      controlError(error.message);
    } finally {
      button.disabled = false;
    }
  }

  function wireControls() {
    $("#btnOpen").addEventListener("click", () => { controlError(""); showForm(true); });
    $("#btnCancel").addEventListener("click", () => { controlError(""); showForm(false); });
    $("#ctlForm").addEventListener("submit", submitStart);
    $("#btnEnd").addEventListener("click", endTrial);
    refreshControls();
    // Keep the open trial's turn count and the scoring job live, but stop
    // polling while the tab is hidden so a phone in a pocket is left alone.
    let timer = window.setInterval(refreshControls, POLL_MS);
    document.addEventListener("visibilitychange", () => {
      window.clearInterval(timer);
      if (!document.hidden) { refreshControls(); timer = window.setInterval(refreshControls, POLL_MS); }
    });
  }

  async function start() {
    const parts = window.location.pathname.split("/").filter(Boolean);
    const detail = parts.length === 2 && parts[0] === "trials";
    $(detail ? "#vTrial" : "#vIndex").hidden = false;
    if (!detail) wireControls();
    try {
      if (detail) renderTrial(await getJSON(`/api/trials/${encodeURIComponent(parts[1])}`));
      else renderIndex((await getJSON("/api/trials")).trials || []);
    } catch (error) {
      const state = $(detail ? "#trialState" : "#indexState");
      state.hidden = false; state.classList.add("error"); state.textContent = error.message;
    }
  }

  start();
})();
