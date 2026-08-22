export const RGB_VIEWER_STATE = Object.freeze({
  FRESH: 'FRESH',
  STALE: 'STALE',
  RECOVERING: 'RECOVERING',
  DISCONNECTED: 'DISCONNECTED',
});

const DEFAULT_BACKOFF_MS = Object.freeze([2000, 4000, 8000, 15000]);

export class RgbViewerFreshness {
  constructor({
    now = () => performance.now(),
    staleAfterMs = 2500,
    recoveryBackoffMs = DEFAULT_BACKOFF_MS,
    onRecovery = () => {},
    onStateChange = () => {},
  } = {}) {
    this.now = now;
    this.staleAfterMs = staleAfterMs;
    this.recoveryBackoffMs = [...recoveryBackoffMs];
    this.onRecovery = onRecovery;
    this.onStateChange = onStateChange;
    this.viewerSocketState = 'closed';
    this.started = false;
    this.connectionStartedAt = null;
    this.lastRenderedAt = null;
    this.backendSocketConnected = null;
    this.backendCanonicalFrameAgeMs = null;
    this.backendHealthAt = null;
    this.backendHealthKnown = false;
    this.recoveryAttempt = 0;
    this.nextRecoveryAt = 0;
    this.disposed = false;
    this.current = {state: RGB_VIEWER_STATE.DISCONNECTED, cause: 'viewer'};
  }

  beginConnection({manual = false} = {}) {
    if (this.disposed) return;
    const now = this.now();
    this.started = true;
    if (manual) {
      this.recoveryAttempt = 0;
      this.nextRecoveryAt = 0;
    }
    this.viewerSocketState = 'connecting';
    this.connectionStartedAt = now;
    this.lastRenderedAt = null;
    this._setState(RGB_VIEWER_STATE.RECOVERING, manual ? 'manual' : 'viewer');
  }

  socketOpened() {
    if (this.disposed) return;
    this.viewerSocketState = 'open';
    this.connectionStartedAt = this.now();
    this._setState(RGB_VIEWER_STATE.RECOVERING, 'waiting_for_frame');
  }

  socketClosed() {
    if (this.disposed) return;
    this.viewerSocketState = 'closed';
    this.connectionStartedAt = null;
    this.lastRenderedAt = null;
    this.tick();
  }

  frameRendered() {
    if (this.disposed) return;
    this.lastRenderedAt = this.now();
    this.recoveryAttempt = 0;
    this.nextRecoveryAt = 0;
    this._setState(RGB_VIEWER_STATE.FRESH, 'rendered_frame');
  }

  updateBackend({socketConnected, canonicalFrameAgeMs, available = true}) {
    if (this.disposed) return;
    this.backendHealthKnown = available;
    if (!available) {
      this.backendSocketConnected = null;
      this.backendCanonicalFrameAgeMs = null;
      this.backendHealthAt = null;
      this.tick();
      return;
    }
    this.backendSocketConnected = typeof socketConnected === 'boolean'
      ? socketConnected
      : null;
    const age = Number(canonicalFrameAgeMs);
    const hasAge = canonicalFrameAgeMs !== null
      && canonicalFrameAgeMs !== undefined
      && Number.isFinite(age);
    this.backendCanonicalFrameAgeMs = hasAge ? Math.max(0, age) : null;
    this.backendHealthAt = this.now();
    this.tick();
  }

  backendFrameAgeMs(now = this.now()) {
    if (this.backendCanonicalFrameAgeMs === null || this.backendHealthAt === null) return null;
    return this.backendCanonicalFrameAgeMs + Math.max(0, now - this.backendHealthAt);
  }

  tick() {
    if (this.disposed) return this.snapshot();
    const now = this.now();
    const backendAgeMs = this.backendFrameAgeMs(now);

    if (this.backendSocketConnected === false) {
      this._setState(RGB_VIEWER_STATE.DISCONNECTED, 'backend_socket');
      return this.snapshot();
    }

    if (!this.started) {
      this._setState(RGB_VIEWER_STATE.DISCONNECTED, 'viewer');
      return this.snapshot();
    }

    if (this.viewerSocketState === 'connecting') {
      const connectingForMs = this.connectionStartedAt === null
        ? Infinity
        : Math.max(0, now - this.connectionStartedAt);
      if (connectingForMs <= this.staleAfterMs) {
        this._setState(RGB_VIEWER_STATE.RECOVERING, 'viewer_connecting');
        return this.snapshot();
      }
    }

    const renderedAgeMs = this.lastRenderedAt === null
      ? null
      : Math.max(0, now - this.lastRenderedAt);
    if (this.viewerSocketState === 'open' && renderedAgeMs !== null
        && renderedAgeMs <= this.staleAfterMs) {
      this._setState(RGB_VIEWER_STATE.FRESH, 'rendered_frame');
      return this.snapshot();
    }

    const waitingSince = this.connectionStartedAt === null
      ? Infinity
      : Math.max(0, now - this.connectionStartedAt);
    if (this.viewerSocketState === 'open' && this.lastRenderedAt === null
        && waitingSince <= this.staleAfterMs) {
      this._setState(RGB_VIEWER_STATE.RECOVERING, 'waiting_for_frame');
      return this.snapshot();
    }

    if (this.backendHealthKnown
        && (backendAgeMs === null || backendAgeMs > this.staleAfterMs)) {
      this._setState(RGB_VIEWER_STATE.STALE, 'backend_frame');
      return this.snapshot();
    }

    this._setState(RGB_VIEWER_STATE.STALE, 'browser_frame');
    if (now >= this.nextRecoveryAt) {
      const attempt = this.recoveryAttempt + 1;
      const index = Math.min(attempt - 1, this.recoveryBackoffMs.length - 1);
      const cooldownMs = this.recoveryBackoffMs[index];
      this.recoveryAttempt = attempt;
      this.nextRecoveryAt = now + cooldownMs;
      this._setState(RGB_VIEWER_STATE.RECOVERING, 'browser_frame');
      this.onRecovery({attempt, cooldownMs});
    }
    return this.snapshot();
  }

  snapshot() {
    const now = this.now();
    return {
      ...this.current,
      viewerSocketState: this.viewerSocketState,
      renderedFrameAgeMs: this.lastRenderedAt === null
        ? null
        : Math.max(0, now - this.lastRenderedAt),
      backendSocketConnected: this.backendSocketConnected,
      backendCanonicalFrameAgeMs: this.backendFrameAgeMs(now),
      recoveryAttempt: this.recoveryAttempt,
      nextRecoveryInMs: Math.max(0, this.nextRecoveryAt - now),
    };
  }

  dispose() {
    this.disposed = true;
  }

  _setState(state, cause) {
    if (this.current.state === state && this.current.cause === cause) return;
    this.current = {state, cause};
    this.onStateChange(this.snapshot());
  }
}
