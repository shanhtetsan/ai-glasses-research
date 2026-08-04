/*
 * AI Smart Glasses — real-time latency panel patch
 * Load AFTER /static/main.js.
 * This file touches only the existing latency panel.
 */
(() => {
  'use strict';

  const POLL_MS = 1000;
  const ENDPOINT = '/latency/metrics';

  const ids = {
    turn:    ['latencyTurn', 'lTurn'],
    status:  ['latencyStatus', 'lStatus'],
    rtt:     ['latencyRtt', 'lRtt'],
    gemini:  ['latencyGemini', 'lGemini'],
    tts:     ['latencyTts', 'lTts'],
    backend: ['latencyBackend', 'lBackend'],
    device:  ['latencyDevice', 'lDevice'],
    median:  ['latencyMedian'],
    p95:     ['latencyP95'],
    medp95:  ['lMedP95'],
    mResp:   ['mResponse'],
    mTurn:   ['mTurn'],
  };

  const nodes = {};
  for (const [name, candidates] of Object.entries(ids)) {
    nodes[name] = candidates.map(id => document.getElementById(id)).find(Boolean) || null;
  }

  function first(...values) {
    return values.find(v => v !== undefined && v !== null && v !== '');
  }

  function latestRecord(payload) {
    if (!payload || typeof payload !== 'object') return {};

    for (const key of [
      'latest', 'latest_turn', 'current', 'last_completed',
      'latest_completed', 'current_turn_data'
    ]) {
      const value = payload[key];
      if (value && typeof value === 'object' && !Array.isArray(value)) return value;
    }

    for (const key of ['turns', 'history', 'records', 'completed_turns', 'recent']) {
      const value = payload[key];
      if (Array.isArray(value) && value.length) return value[value.length - 1] || {};
      if (value && typeof value === 'object') {
        const records = Object.values(value);
        if (records.length) return records[records.length - 1] || {};
      }
    }

    return payload;
  }

  function numberOf(value) {
    if (value === undefined || value === null || value === '') return null;

    if (typeof value === 'object') {
      return numberOf(first(
        value.value,
        value.ms,
        value.latest,
        value.latest_ms,
        value.average_ms
      ));
    }

    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function metricText(value) {
    if (value && typeof value === 'object' && value.text !== undefined) {
      return String(value.text);
    }

    const n = numberOf(value);
    return n === null ? '--' : `${n.toFixed(1)} ms`;
  }

  function metricColor(value, thresholds) {
    if (value && typeof value === 'object' && value.color) {
      return value.color;
    }

    const n = numberOf(value);
    if (n === null) return 'neutral';
    if (n < thresholds[0]) return 'green';
    if (n < thresholds[1]) return 'yellow';
    return 'red';
  }

  function render(node, value, thresholds) {
    if (!node) return;
    node.textContent = metricText(value);
    node.dataset.color = metricColor(value, thresholds);

    node.classList.remove('ok', 'warn', 'err');
    const color = node.dataset.color;
    if (color === 'green') node.classList.add('ok');
    if (color === 'yellow') node.classList.add('warn');
    if (color === 'red') node.classList.add('err');
  }

  function normalize(payload) {
    const display =
      payload && payload.display && typeof payload.display === 'object'
        ? payload.display
        : {};

    const latest = latestRecord(payload);
    const stats = first(
      payload && payload.summary,
      payload && payload.statistics,
      payload && payload.stats,
      {}
    ) || {};

    return {
      turn: first(
        display.current_turn,
        latest.turn_id,
        payload && payload.current_turn,
        payload && payload.turn_id
      ),

      status: first(
        display.status,
        latest.status,
        payload && payload.status,
        'idle'
      ),

      rtt: first(
        display.network_rtt,
        latest.network_rtt,
        latest.network_rtt_ms,
        latest.latest_rtt_ms,
        payload && payload.network_rtt,
        payload && payload.network_rtt_ms,
        payload && payload.latest_rtt_ms,
        payload && payload.network && payload.network.latest_rtt_ms
      ),

      gemini: first(
        display.speech_end_to_gemini,
        latest.speech_end_to_first_gemini_audio_ms,
        latest.speech_end_to_gemini_ms,
        latest.speech_end_to_gemini,
        payload && payload.speech_end_to_first_gemini_audio_ms
      ),

      tts: first(
        display.speech_end_to_first_tts,
        latest.speech_end_to_first_tts_send_ms,
        latest.speech_end_to_first_tts_ms,
        latest.speech_end_to_first_tts,
        payload && payload.speech_end_to_first_tts_send_ms
      ),

      backend: first(
        display.backend_total,
        latest.backend_turn_total_ms,
        latest.backend_total_ms,
        latest.backend_total,
        payload && payload.backend_turn_total_ms
      ),

      device: first(
        display.device_end_to_end,
        latest.speech_end_to_first_i2s_ms,
        latest.device_end_to_end_ms,
        latest.device_metrics && latest.device_metrics.speech_end_to_first_i2s_ms,
        payload && payload.speech_end_to_first_i2s_ms
      ),

      median: first(
        display.median,
        stats.median,
        stats.median_ms,
        payload && payload.median,
        payload && payload.median_ms
      ),

      p95: first(
        display.p95,
        stats.p95,
        stats.p95_ms,
        payload && payload.p95,
        payload && payload.p95_ms
      ),
    };
  }

  let requestInFlight = false;

  async function refreshLatency() {
    if (requestInFlight) return;
    requestInFlight = true;

    try {
      const response = await fetch(`${ENDPOINT}?t=${Date.now()}`, {
        method: 'GET',
        cache: 'no-store',
        headers: { Accept: 'application/json' },
      });

      if (!response.ok) {
        if (nodes.status) nodes.status.textContent = `HTTP ${response.status}`;
        return;
      }

      const data = normalize(await response.json());

      if (nodes.turn) nodes.turn.textContent = data.turn ?? '--';
      if (nodes.status) nodes.status.textContent = data.status || 'idle';

      render(nodes.rtt, data.rtt, [100, 250]);
      render(nodes.gemini, data.gemini, [1000, 2500]);
      render(nodes.tts, data.tts, [1200, 3000]);
      render(nodes.backend, data.backend, [5000, 12000]);
      render(nodes.device, data.device, [1500, 3500]);
      render(nodes.median, data.median, [1500, 3500]);
      render(nodes.p95, data.p95, [2500, 6000]);

      if (nodes.medp95) {
        nodes.medp95.textContent =
          `${metricText(data.median)} / ${metricText(data.p95)}`;
      }
      if (nodes.mResp) nodes.mResp.textContent = metricText(data.device);
      if (nodes.mTurn) nodes.mTurn.textContent = data.turn ?? '--';

      window.dispatchEvent(new CustomEvent('latency:update', {
        detail: data
      }));
    } catch (error) {
      if (nodes.status) nodes.status.textContent = 'unavailable';
      console.warn('[LATENCY PANEL] refresh failed:', error);
    } finally {
      requestInFlight = false;
    }
  }

  refreshLatency();
  window.setInterval(refreshLatency, POLL_MS);
  window.refreshLatencyPanel = refreshLatency;
})();