export const AUDIO_FRESHNESS_STATE = Object.freeze({
  FRESH: 'fresh',
  SUPPRESSED: 'suppressed',
  STALE: 'stale',
  DISCONNECTED: 'disconnected',
  RECOVERING: 'recovering',
});

export function audioBadgePresentation(snapshot) {
  const state = snapshot && snapshot.state;
  if (state === AUDIO_FRESHNESS_STATE.FRESH) {
    return {state, className: 'chip ok', text: 'Audio: fresh'};
  }
  if (state === AUDIO_FRESHNESS_STATE.SUPPRESSED) {
    const reason = snapshot.audio_gate_reason === 'tts' ? 'TTS suppressed' : 'suppressed';
    return {state, className: 'chip neutral', text: `Audio: ${reason}`};
  }
  if (state === AUDIO_FRESHNESS_STATE.DISCONNECTED) {
    return {state, className: 'chip err', text: 'Audio: disconnected'};
  }
  return {
    state: state || AUDIO_FRESHNESS_STATE.RECOVERING,
    className: 'chip warn',
    text: state === AUDIO_FRESHNESS_STATE.STALE
      ? 'Audio: stale'
      : 'Audio: recovering…',
  };
}
