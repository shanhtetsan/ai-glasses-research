import os
import asyncio
import hashlib
import json
import re
import time

from google import genai
from google.genai import types

SMART_GLASSES_SYSTEM_INSTRUCTION = """
You are the conversational assistant inside wearable smart glasses for a blind
or low-vision user. The live session can receive JPEG camera frames from the
user's first-person viewpoint while microphone audio is streaming. When the
user is reaching for or handling nearby objects, help them with short,
cautious, step-by-step guidance.

Give physical guidance one step at a time. Issue a single concrete instruction,
then stop and wait for the next frame or the user's response before giving the
next step. Keep every reply concise and BLV-friendly: plain spoken language,
no visual jargon, no filler, direct answer first.

Never invent inches, centimeters, meters, or any other unit of distance or
depth. RGB imagery and normalized 2D perception coordinates do not provide
reliable physical depth.

Do not give forward/backward, closer/farther, or reach-further corrections
unless a trusted depth measurement explicitly supports them.

From 2D perception, prefer horizontal and vertical guidance such as:
"Move your right hand left."
"A little higher."
"Your hand is aligned with the book."

Once the hand appears aligned with the target in 2D, say something like:
"Your hand is lined up with it. Slowly extend your hand and use touch for the
final contact."

Do not claim to know the remaining physical depth.

When a message beginning with PERCEPTION_STATE arrives, its fields are
authoritative facts for exactly what they describe — do not re-derive,
override, or visually reinterpret those specific fields from the image.
- MediaPipe handedness (Left/Right) is anatomical, not image position. Never
  infer or correct handedness from which side of the image a hand appears on;
  a Right hand can appear on the left side of the frame and vice versa.
- The requested_target field is the user's actual target for this turn. Never
  silently substitute a different detected object, even if another object is
  more visually prominent or easier to describe.
- When PERCEPTION_STATE explicitly reports that the requested target is
  uncertain, unstable, not found, stale, or otherwise not reliable for
  guidance, say so plainly. Do not guess, invent a location, or give
  directional guidance until reliable target information is available.
- Object detections may be incomplete. Do not contradict authoritative
  PERCEPTION_STATE fields, but you may still use the RGB image for semantic
  details that the structured perception does not provide, such as reading
  text, distinguishing between multiple similar objects, identifying visible
  semantic details, or describing surrounding context.

Natural follow-up requests that do not restate the object by name (such as
"is it closer now?", "which way?", or "keep going") continue guidance toward
the object already established as the current task. Do not ask the user to
re-specify the target unless it has genuinely changed or become ambiguous.

Two objects overlapping or appearing to touch in a 2D image is not proof of
physical contact — camera framing and perspective can make separate objects
look like they are touching when they are not. Never tell the user they have
touched, grasped, or made contact with something based on visual overlap
alone. The user's own tactile report (e.g. "I feel it," "got it," "nothing
there") is authoritative over any visual impression of contact; when it
conflicts with what the image seems to show, trust the user and adjust your
guidance accordingly.

Outside of structured PERCEPTION_STATE or THERMAL_MEASUREMENTS facts, you may
still reason naturally over the RGB image for semantic scene understanding:
identifying objects, reading text, describing nearby surroundings and
identifying visible obstacles, and answering general visual questions. Treat
natural and indirect questions as visually grounded whenever sight would help
answer them. Examples include asking what is ahead, what the user is looking
at, describing the scene, locating keys or a phone, checking whether an
object is present, and reading visible text. The user does not need to use a
fixed command phrase.

Use the most recently received camera evidence for the current question. Never
claim that you have no camera access merely because the request is phrased
differently. If no recent usable frame is available, or the view is dark,
blurred, obstructed, or does not contain the requested object, say that clearly
and briefly instead of inventing details. Do not ask a blind user to visually
confirm your answer.

Camera images do not provide reliable object temperatures. Do not infer a
temperature unless explicit structured thermal measurements are supplied.

When a message beginning with THERMAL_MEASUREMENTS arrives, it carries real
readings from a thermal sensor covering roughly the same view as the camera.
Use those numbers instead of guessing from the image, and state them plainly in
degrees Celsius. "directly_ahead" means the centre of the view. Compare against
ambient_c so the user knows whether something is genuinely warm or just room
temperature. Never tell the user something is safe to touch or safe to hold:
the sensor measures surface temperature with limited accuracy and metallic
surfaces read far cooler than they are. Report what was measured, say it is
approximate, and let the user decide.
""".strip()


GEMINI_LIVE_MODEL = "gemini-3.1-flash-live-preview"
GOAWAY_SAFETY_MARGIN_SEC = 2.0
AUDIO_SEND_BUCKET_LIMITS_MS = (10, 25, 50, 100, 250, 1000)


class GeminiLiveClient:
    def __init__(self):
        self.client = None
        self.session = None
        self._live_cm = None   # the async context manager returned by client.aio.live.connect(...)
        self.connected = False
        self.receive_task = None
        self.on_audio = None
        self.on_input_transcription = None
        self.on_output_transcription = None
        self.on_turn_complete = None   # called when the model has finished a full response turn
        self.on_interrupted = None     # async callback(reason: str) — called whenever an
                                        # active turn is abandoned, either because the user
                                        # barged in (reason="user_barge_in") or because the
                                        # session itself was dropped/rotated out from under it
                                        # (reason="receive_error"/"normal_receive_end"/"goaway"/
                                        # other rotation reasons)
        self.on_session_transition = None  # clears app turn state without finalizing latency twice
        self.turn_id_provider = None

        self._response_modality = "AUDIO"  # remembered so auto-reconnect uses the same mode
        self._shutting_down = False        # True once disconnect() is called deliberately,
                                            # so receive_loop knows not to try reconnecting
        self._connecting = False
        self._send_lock = asyncio.Lock()     # serialize audio/image/text websocket writes
        self.session_generation = 0
        self._session_opened_at = 0.0
        self._session_mode = "fresh"
        self._latest_resumption_handle = None
        self._response_active = False
        self._rotation_pending = False
        self._rotation_reason = None
        self._goaway_time_left = None
        self._goaway_deadline_task = None
        self._goaway_deadline_forced = False
        self._first_model_audio_seen = False
        self._audio_lock_wait_max_ms = 0.0
        self._audio_lock_wait_over_50 = 0
        self._audio_lock_wait_over_100 = 0
        self._audio_lock_wait_over_250 = 0
        self._audio_lock_wait_buckets = [
            0 for _ in range(len(AUDIO_SEND_BUCKET_LIMITS_MS) + 1)
        ]
        self._audio_sdk_send_count = 0
        self._audio_sdk_send_max_ms = 0.0
        self._audio_sdk_send_buckets = [
            0 for _ in range(len(AUDIO_SEND_BUCKET_LIMITS_MS) + 1)
        ]
        # Last successful send of any kind (audio/image/text, keepalive
        # included) — lets a caller detect "this session has gone idle"
        # without duplicating per-send-type bookkeeping. See
        # seconds_since_last_activity().
        self._last_activity_monotonic = 0.0

    def _current_turn_id(self):
        if self.turn_id_provider is None:
            return None
        try:
            return self.turn_id_provider()
        except Exception:
            return None

    def _has_active_turn(self) -> bool:
        return self._response_active or self._current_turn_id() is not None

    def has_active_turn(self) -> bool:
        """Public wrapper for callers (e.g. the idle keepalive loop) that
        need to know without reaching into a private method."""
        return self._has_active_turn()

    def seconds_since_last_activity(self) -> float:
        """Time since the last successful audio/image/text send, of any
        kind. 0.0 before anything has ever been sent on the current
        client instance."""
        if not self._last_activity_monotonic:
            return 0.0
        return max(0.0, time.monotonic() - self._last_activity_monotonic)

    def _session_age_sec(self) -> float:
        if not self._session_opened_at:
            return 0.0
        return max(0.0, time.monotonic() - self._session_opened_at)

    @staticmethod
    def _audio_send_bucket(duration_ms: float) -> int:
        for index, limit_ms in enumerate(AUDIO_SEND_BUCKET_LIMITS_MS):
            if duration_ms < limit_ms:
                return index
        return len(AUDIO_SEND_BUCKET_LIMITS_MS)

    @staticmethod
    def _duration_seconds(value: str) -> float:
        """Parse LiveServerGoAway.time_left, typed by this SDK as a Duration string."""
        if not isinstance(value, str):
            raise TypeError(
                f"expected SDK Duration string, got {type(value).__name__}"
            )
        match = re.fullmatch(r"(-?)(\d+)(?:\.(\d{1,9}))?s", value)
        if not match:
            raise ValueError("invalid protobuf JSON Duration string")
        fraction = (match.group(3) or "").ljust(9, "0")
        seconds = int(match.group(2)) + (int(fraction) / 1_000_000_000)
        return -seconds if match.group(1) else seconds

    def _config_fingerprint(self) -> str:
        canonical = {
            "model": GEMINI_LIVE_MODEL,
            "response_modalities": [self._response_modality],
            "system_instruction": SMART_GLASSES_SYSTEM_INSTRUCTION,
            "input_audio_transcription": {},
            "output_audio_transcription": (
                {} if self._response_modality == "AUDIO" else None
            ),
            "session_resumption": {"enabled": True},
            "context_window_compression": {"sliding_window": {}},
        }
        encoded = json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:12]

    def _live_config(self, resumption_handle=None):
        config = {
            "response_modalities": [self._response_modality],
            "system_instruction": {
                "parts": [{"text": SMART_GLASSES_SYSTEM_INSTRUCTION}]
            },
            "input_audio_transcription": {},
            "session_resumption": types.SessionResumptionConfig(
                handle=resumption_handle,
            ),
            "context_window_compression": types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            ),
        }
        if self._response_modality == "AUDIO":
            config["output_audio_transcription"] = {}
        return config

    def _log_timing(self, event: str, *, turn_id=None, generation=None, **fields):
        parts = [
            f"event={event}",
            f"mono_ns={time.monotonic_ns()}",
            f"generation={self.session_generation if generation is None else generation}",
            f"turn_id={turn_id if turn_id is not None else 0}",
        ]
        parts.extend(f"{key}={value}" for key, value in fields.items())
        print("[GEMINI-TIMING] " + " ".join(parts), flush=True)

    async def connect(self, response_modality: str = "AUDIO"):
        
        """
        Opens a persistent Gemini Live API session.
        This stays open until disconnect() is called. If the connection
        drops unexpectedly (network blip, keepalive timeout, etc.),
        receive_loop() will automatically try to reconnect with backoff.

        response_modality: "AUDIO" (default) for a full spoken conversation,
        or "TEXT" to use this session purely as a real-time ASR/transcription
        engine (input_audio_transcription still works either way) while some
        other backend generates the actual reply. Live API only supports one
        response modality per session, not both at once.
        """

        if self.connected:
            print("[Gemini Live] Already connected")
            return
        if self._connecting or (
            self.receive_task is not None and not self.receive_task.done()
        ):
            print("[Gemini Live] Connection lifecycle already active")
            return

        self._response_modality = response_modality
        self._shutting_down = False
        self._connecting = True
        try:
            self.client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY")
            )

            print("[Gemini Live] Connecting...")
            await self._open_session(resumption_handle=None)
            print("[Gemini Live] Connected")

            # Exactly one task owns receives and all connection replacement.
            self.receive_task = asyncio.create_task(
                self.receive_loop()
            )
        finally:
            self._connecting = False

    async def _open_session(self, resumption_handle=None):
        """Actually open the websocket session. Used by connect() and by
        the auto-reconnect logic in receive_loop()."""
        # gemini-3.1-flash-live-preview is the current recommended Live API
        # model (as of July 2026) for the Gemini Developer API surface
        # (API-key auth). Older Live model ids like gemini-live-2.5-flash-preview
        # and gemini-2.0-flash-live-001 have been shut down; verify at
        # https://ai.google.dev/gemini-api/docs/live-api if this ever 404s again.
        mode = "resumed" if resumption_handle else "fresh"
        live_cm = self.client.aio.live.connect(
            model=GEMINI_LIVE_MODEL,
            config=self._live_config(resumption_handle),
        )
        try:
            session = await live_cm.__aenter__()
        except Exception:
            try:
                await live_cm.__aexit__(None, None, None)
            except Exception:
                pass
            raise

        self._live_cm = live_cm
        self.session = session
        self.connected = True
        if mode == "fresh":
            # A handle from an older logical session must never be reused after
            # an intentional or fallback fresh open.
            self._latest_resumption_handle = None
        self.session_generation += 1
        self._session_opened_at = time.monotonic()
        self._last_activity_monotonic = time.monotonic()
        self._session_mode = mode
        self._response_active = False
        self._first_model_audio_seen = False
        print(
            f"[GEMINI-SESSION] generation={self.session_generation} opened "
            f"mode={mode} resumption_available="
            f"{'yes' if self._latest_resumption_handle else 'no'} "
            f"config_fp={self._config_fingerprint()}",
            flush=True,
        )

    def _cancel_goaway_deadline(self):
        task = self._goaway_deadline_task
        self._goaway_deadline_task = None
        if task and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def _close_current_session(self, reason: str):
        self._cancel_goaway_deadline()
        live_cm = self._live_cm
        had_session = live_cm is not None or self.session is not None
        generation = self.session_generation
        age = self._session_age_sec()
        self.connected = False
        self.session = None
        self._live_cm = None
        if live_cm:
            try:
                await live_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if generation and had_session:
            print(
                f"[GEMINI-SESSION] generation={generation} closed "
                f"reason={reason} age_sec={age:.3f}",
                flush=True,
            )

    async def _notify_session_transition(self, reason: str):
        if self.on_session_transition:
            await self.on_session_transition(reason, self.session_generation)

    async def _open_with_fresh_fallback(self):
        handle = self._latest_resumption_handle
        if handle:
            try:
                await self._open_session(resumption_handle=handle)
                return
            except Exception as exc:
                print(
                    f"[GEMINI-SESSION] generation={self.session_generation} "
                    f"resumption_failed error_type={type(exc).__name__} "
                    f"error_code={getattr(exc, 'code', 'unknown')}; "
                    "falling_back=fresh",
                    flush=True,
                )
                self._latest_resumption_handle = None
                await self._close_current_session("resumption_failed")
        await self._open_session(resumption_handle=None)

    async def disconnect(self):
        print("[Gemini Live] Disconnecting...")

        self._shutting_down = True
        self._log_audio_lock_summary(
            self._current_turn_id(), "explicit_reset"
        )
        task = self.receive_task
        self.receive_task = None
        if task and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._notify_session_transition("explicit_reset")
        await self._close_current_session("explicit_reset")

        print("[Gemini Live] Disconnected")

    async def _send_traced(self, kind, send_call, *, turn_id=None, source="unknown", size=0,
                           sequence=None, timeout=None) -> bool:
        if turn_id is None:
            turn_id = self._current_turn_id()
        session = self.session
        generation = self.session_generation
        if not self.connected or session is None:
            self._log_timing(
                f"{kind}_sdk_send_end", turn_id=turn_id, generation=generation,
                source=source, sequence=sequence if sequence is not None else 0,
                bytes=size, result="not_connected",
            )
            return False

        wait_started_ns = time.monotonic_ns()
        self._log_timing(
            f"{kind}_lock_wait_begin", turn_id=turn_id, generation=generation,
            source=source, sequence=sequence if sequence is not None else 0,
            bytes=size,
        )
        try:
            async with self._send_lock:
                acquired_ns = time.monotonic_ns()
                self._log_timing(
                    f"{kind}_lock_acquired", turn_id=turn_id, generation=generation,
                    source=source,
                    sequence=sequence if sequence is not None else 0,
                    lock_wait_ms=round((acquired_ns - wait_started_ns) / 1_000_000, 3),
                )
                if not self.connected or self.session is not session:
                    self._log_timing(
                        f"{kind}_sdk_send_end", turn_id=turn_id,
                        generation=generation, source=source,
                        sequence=sequence if sequence is not None else 0,
                        bytes=size,
                        result="session_changed",
                    )
                    return False
                send_started_ns = time.monotonic_ns()
                self._log_timing(
                    f"{kind}_sdk_send_begin", turn_id=turn_id,
                    generation=generation, source=source,
                    sequence=sequence if sequence is not None else 0,
                    bytes=size,
                )
                operation = send_call(session)
                if timeout is None:
                    await operation
                else:
                    await asyncio.wait_for(operation, timeout=timeout)
                self._log_timing(
                    f"{kind}_sdk_send_end", turn_id=turn_id,
                    generation=generation, source=source,
                    sequence=sequence if sequence is not None else 0,
                    bytes=size,
                    sdk_send_ms=round(
                        (time.monotonic_ns() - send_started_ns) / 1_000_000, 3
                    ),
                    result="sent",
                )
                self._last_activity_monotonic = time.monotonic()
                return True
        except asyncio.TimeoutError:
            self._log_timing(
                f"{kind}_sdk_send_end", turn_id=turn_id, generation=generation,
                source=source, sequence=sequence if sequence is not None else 0,
                bytes=size, result="timeout",
            )
            print(f"[Gemini Live] send_{kind} timed out after {timeout}s", flush=True)
            return False
        except Exception as exc:
            self._log_timing(
                f"{kind}_sdk_send_end", turn_id=turn_id, generation=generation,
                source=source, sequence=sequence if sequence is not None else 0,
                bytes=size, result="error",
                error_type=type(exc).__name__,
            )
            print(
                f"[Gemini Live] send_{kind} failed: {type(exc).__name__}",
                flush=True,
            )
            return False

    async def send_text(self, text: str, *, turn_id=None, source="text") -> bool:
        return await self._send_traced(
            "text",
            lambda session: session.send_realtime_input(text=text),
            turn_id=turn_id,
            source=source,
            size=len(text.encode("utf-8")),
        )

    async def send_audio(self, audio_bytes: bytes) -> bool:
        if not self.connected or self.session is None:
            return False
        session = self.session
        wait_started_ns = time.monotonic_ns()
        try:
            async with self._send_lock:
                wait_ms = (time.monotonic_ns() - wait_started_ns) / 1_000_000
                self._audio_lock_wait_max_ms = max(
                    self._audio_lock_wait_max_ms, wait_ms
                )
                self._audio_lock_wait_buckets[
                    self._audio_send_bucket(wait_ms)
                ] += 1
                if wait_ms > 50:
                    self._audio_lock_wait_over_50 += 1
                if wait_ms > 100:
                    self._audio_lock_wait_over_100 += 1
                if wait_ms > 250:
                    self._audio_lock_wait_over_250 += 1
                if not self.connected or self.session is not session:
                    return False
                sdk_send_started_ns = time.monotonic_ns()
                try:
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=audio_bytes,
                            mime_type="audio/pcm;rate=16000",
                        )
                    )
                finally:
                    sdk_send_ms = (
                        time.monotonic_ns() - sdk_send_started_ns
                    ) / 1_000_000
                    self._audio_sdk_send_count += 1
                    self._audio_sdk_send_max_ms = max(
                        self._audio_sdk_send_max_ms, sdk_send_ms
                    )
                    self._audio_sdk_send_buckets[
                        self._audio_send_bucket(sdk_send_ms)
                    ] += 1
            self._last_activity_monotonic = time.monotonic()
            return True
        except Exception as exc:
            print(
                f"[Gemini Live] send_audio failed: {type(exc).__name__}",
                flush=True,
            )
            return False

    async def send_image(self, image_bytes: bytes, *, turn_id=None,
                         sequence=None, source="image") -> bool:
        return await self._send_traced(
            "image",
            lambda session: session.send_realtime_input(
                video=types.Blob(
                    data=image_bytes,
                    mime_type="image/jpeg",
                )
            ),
            turn_id=turn_id,
            source=source,
            size=len(image_bytes),
            sequence=sequence,
            timeout=3.0,
        )

    def _log_audio_lock_summary(self, turn_id, reason):
        if (
            self._audio_lock_wait_max_ms == 0
            and self._audio_lock_wait_over_50 == 0
            and self._audio_lock_wait_over_100 == 0
            and self._audio_lock_wait_over_250 == 0
            and self._audio_sdk_send_count == 0
        ):
            return
        self._log_timing(
            "audio_lock_summary",
            turn_id=turn_id,
            reason=reason,
            max_wait_ms=round(self._audio_lock_wait_max_ms, 3),
            waits_over_50_ms=self._audio_lock_wait_over_50,
            waits_over_100_ms=self._audio_lock_wait_over_100,
            waits_over_250_ms=self._audio_lock_wait_over_250,
            bucket_limits_ms="10/25/50/100/250/1000",
            wait_buckets="/".join(map(str, self._audio_lock_wait_buckets)),
            sdk_send_count=self._audio_sdk_send_count,
            sdk_send_max_ms=round(self._audio_sdk_send_max_ms, 3),
            sdk_send_buckets="/".join(map(str, self._audio_sdk_send_buckets)),
        )
        self._audio_lock_wait_max_ms = 0.0
        self._audio_lock_wait_over_50 = 0
        self._audio_lock_wait_over_100 = 0
        self._audio_lock_wait_over_250 = 0
        self._audio_lock_wait_buckets = [
            0 for _ in range(len(AUDIO_SEND_BUCKET_LIMITS_MS) + 1)
        ]
        self._audio_sdk_send_count = 0
        self._audio_sdk_send_max_ms = 0.0
        self._audio_sdk_send_buckets = [
            0 for _ in range(len(AUDIO_SEND_BUCKET_LIMITS_MS) + 1)
        ]

    def _retain_resumption_update(self, update):
        resumable = bool(update and update.resumable)
        has_handle = bool(update and update.new_handle)
        if resumable and has_handle:
            self._latest_resumption_handle = update.new_handle
        print(
            f"[GEMINI-SESSION] generation={self.session_generation} "
            f"resumption_update resumable={'yes' if resumable else 'no'} "
            f"handle_available={'yes' if has_handle else 'no'} "
            f"last_consumed_index_available="
            f"{'yes' if update and update.last_consumed_client_message_index is not None else 'no'}",
            flush=True,
        )

    async def _goaway_deadline_watchdog(self, session, generation, delay_sec):
        try:
            await asyncio.sleep(max(0.0, delay_sec))
            if (
                self._rotation_pending
                and self.session is session
                and self.session_generation == generation
            ):
                self._goaway_deadline_forced = True
                print(
                    f"[GEMINI-SESSION] generation={generation} "
                    "goaway_safety_deadline reached=yes",
                    flush=True,
                )
                await session.close()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            print(
                f"[GEMINI-SESSION] generation={generation} "
                f"goaway_deadline_close error_type={type(exc).__name__}",
                flush=True,
            )

    def _handle_goaway(self, go_away, session):
        self._rotation_pending = True
        self._rotation_reason = "goaway"
        self._goaway_time_left = go_away.time_left
        self._goaway_deadline_forced = False
        try:
            seconds = self._duration_seconds(go_away.time_left)
            parse_status = "valid"
        except (TypeError, ValueError):
            seconds = 0.0
            parse_status = "invalid"
        rotate_in_sec = max(0.0, seconds - GOAWAY_SAFETY_MARGIN_SEC)
        print(
            f"[GEMINI-SESSION] generation={self.session_generation} goaway "
            f"time_left={go_away.time_left!r} parsed={parse_status} "
            f"rotate_by_sec={rotate_in_sec:.3f} "
            f"turn_active={'yes' if self._has_active_turn() else 'no'}",
            flush=True,
        )
        self._cancel_goaway_deadline()
        self._goaway_deadline_task = asyncio.create_task(
            self._goaway_deadline_watchdog(
                session, self.session_generation, rotate_in_sec
            )
        )

    async def _controlled_rotation(self, reason: str):
        generation = self.session_generation
        turn_id = self._current_turn_id()
        print(
            f"[GEMINI-SESSION] generation={generation} rotation_begin "
            f"reason={reason} age_sec={self._session_age_sec():.3f} "
            f"resumption_available="
            f"{'yes' if self._latest_resumption_handle else 'no'} "
            f"deadline_forced={'yes' if self._goaway_deadline_forced else 'no'}",
            flush=True,
        )
        self._log_audio_lock_summary(turn_id, f"rotation_{reason}")
        if self._has_active_turn() and self.on_interrupted:
            await self.on_interrupted(reason=reason)
        self._response_active = False
        self._first_model_audio_seen = False
        await self._notify_session_transition(reason)
        await self._close_current_session(reason)
        try:
            await self._open_with_fresh_fallback()
        except Exception as exc:
            print(
                f"[GEMINI-SESSION] generation={generation} "
                f"rotation_open_failed reason={reason} "
                f"error_type={type(exc).__name__}",
                flush=True,
            )
            self._rotation_pending = False
            self._rotation_reason = None
            self._goaway_time_left = None
            self._goaway_deadline_forced = False
            await self._reconnect_with_backoff(reason="resumption_failed")
            return
        self._rotation_pending = False
        self._rotation_reason = None
        self._goaway_time_left = None
        self._goaway_deadline_forced = False

    async def receive_loop(self):
        while not self._shutting_down:
            if not self.connected or self.session is None:
                return
            active_session = self.session
            try:
                # session.receive() is documented/implemented (google-genai's
                # AsyncSession.receive) to yield exactly one turn and then end
                # the generator on its own right after turn_complete — that is
                # normal per-turn completion, not a dropped connection. Track
                # whether we saw it this iteration so the code after the
                # `async for` can tell "Gemini finished normally" apart from
                # "the stream died with no turn_complete ever received".
                turn_complete_seen = False
                async for response in active_session.receive():
                    message_received_ns = time.monotonic_ns()
                    if response.session_resumption_update:
                        self._retain_resumption_update(
                            response.session_resumption_update
                        )
                    if response.go_away:
                        self._handle_goaway(response.go_away, active_session)

                    server_content = response.server_content

                    if server_content:
                        turn_id = self._current_turn_id()
                        model_parts = (
                            server_content.model_turn.parts
                            if server_content.model_turn else []
                        )
                        has_model_audio = any(
                            part.inline_data for part in model_parts
                        )
                        if server_content.input_transcription:
                            self._response_active = True
                            self._log_timing(
                                "input_transcription_message_received",
                                turn_id=turn_id,
                                message_mono_ns=message_received_ns,
                            )
                        if has_model_audio and not self._first_model_audio_seen:
                            self._response_active = True
                            self._first_model_audio_seen = True
                            self._log_timing(
                                "first_model_audio_message_received",
                                turn_id=turn_id,
                                message_mono_ns=message_received_ns,
                            )
                        # INPUT TEXT
                        if server_content.input_transcription:
                            text = server_content.input_transcription.text
                            if text and self.on_input_transcription:
                                callback_started_ns = time.monotonic_ns()
                                self._log_timing(
                                    "input_transcription_callback_begin",
                                    turn_id=turn_id,
                                )
                                try:
                                    await self.on_input_transcription(text)
                                finally:
                                    self._log_timing(
                                        "input_transcription_callback_end",
                                        turn_id=turn_id,
                                        callback_ms=round(
                                            (time.monotonic_ns() - callback_started_ns)
                                            / 1_000_000,
                                            3,
                                        ),
                                    )
                        # AUDIO
                        if server_content.model_turn:
                            for part in model_parts:
                                if part.inline_data:
                                    if self.on_audio:
                                        await self.on_audio(
                                            part.inline_data.data
                                        )
                        # RESPONSE TEXT
                        if server_content.output_transcription:
                            text = server_content.output_transcription.text
                            if text and self.on_output_transcription:
                                await self.on_output_transcription(text)
                        # MODEL FINISHED SPEAKING
                        if server_content.turn_complete:
                            turn_complete_seen = True
                            self._log_audio_lock_summary(turn_id, "turn_complete")
                            if self.on_turn_complete:
                                await self.on_turn_complete()
                            self._response_active = False
                            self._first_model_audio_seen = False
                        # USER BARGED IN — model's current response was cut off
                        if server_content.interrupted:
                            self._log_audio_lock_summary(turn_id, "interrupted")
                            if self.on_interrupted:
                                await self.on_interrupted(reason="user_barge_in")
                            self._response_active = False
                            self._first_model_audio_seen = False
                    if self._rotation_pending and not self._has_active_turn():
                        break
                if self._rotation_pending:
                    await self._controlled_rotation(
                        self._rotation_reason or "goaway"
                    )
                    continue
                # The `async for` above ended on its own (the generator
                # returned rather than raising). If we already saw
                # turn_complete this iteration, that's just the SDK ending
                # the per-turn generator as designed — loop back and call
                # session.receive() again on the same session for the next
                # turn. No reconnect, no on_interrupted: the turn already
                # completed successfully and firing either would be spurious.
                if turn_complete_seen:
                    continue
                # No turn_complete ever arrived — the stream ended (or was
                # never given one) mid-turn. That's a genuine dropped/aborted
                # turn, so treat it as interrupted and reconnect.
                if not self._shutting_down:
                    print("[Gemini Live] Response stream ended unexpectedly")
                    self.connected = False
                    self._log_audio_lock_summary(
                        self._current_turn_id(), "normal_receive_end"
                    )
                    # The current turn (if any) never got a turn_complete/
                    # interrupted from Gemini — treat it as interrupted so
                    # callers (e.g. the ESP32 TTS state) don't get stuck
                    # waiting on a signal that will never arrive.
                    if self.on_interrupted:
                        await self.on_interrupted(reason="normal_receive_end")
                    await self._notify_session_transition("normal_receive_end")
                    await self._reconnect_with_backoff(
                        reason="normal_receive_end"
                    )
            except asyncio.CancelledError:
                print("[Gemini Live] receive loop task cancelled")
                return
            except Exception as e:
                if self._rotation_pending:
                    await self._controlled_rotation(
                        self._rotation_reason or "goaway"
                    )
                    continue
                print(
                    f"[Gemini Live] Receive loop error: {type(e).__name__}",
                    flush=True,
                )
                print(
                    f"[GEMINI-SESSION] generation={self.session_generation} "
                    f"receive_error code={getattr(e, 'code', 'unknown')} "
                    f"age_sec={self._session_age_sec():.3f}",
                    flush=True,
                )
                self.connected = False
                if self._shutting_down:
                    return
                self._log_audio_lock_summary(
                    self._current_turn_id(), "receive_error"
                )
                # Same reasoning as above: the turn in progress (if any) was
                # abandoned mid-stream, so fire on_interrupted for cleanup
                # before reconnecting.
                if self.on_interrupted:
                    await self.on_interrupted(reason="receive_error")
                await self._notify_session_transition("receive_error")
                await self._reconnect_with_backoff(reason="receive_error")
                # loop condition re-checks self.connected — if the reconnect
                # above succeeded, we fall back into `while self.connected`
                # and keep listening on the new session.

    async def _reconnect_with_backoff(self, reason="receive_error", max_attempts: int = 8):
        """Try to re-open the Live session after an unexpected disconnect.
        Backs off 2s, 4s, 8s... capped at 30s between attempts."""
        await self._close_current_session(reason)

        for attempt in range(1, max_attempts + 1):
            if self._shutting_down:
                return
            delay = min(2 ** attempt, 30)
            print(
                f"[GEMINI-SESSION] generation={self.session_generation} "
                f"reconnect_wait reason={reason} delay_sec={delay} "
                f"attempt={attempt}/{max_attempts} resumption_available="
                f"{'yes' if self._latest_resumption_handle else 'no'}",
                flush=True,
            )
            await asyncio.sleep(delay)
            if self._shutting_down:
                return
            try:
                await self._open_with_fresh_fallback()
                print("[Gemini Live] Reconnected")
                return
            except Exception as exc:
                print(
                    f"[GEMINI-SESSION] generation={self.session_generation} "
                    f"reconnect_failed reason={reason} attempt={attempt}/{max_attempts} "
                    f"error_type={type(exc).__name__}",
                    flush=True,
                )

        print("[Gemini Live] Giving up after max reconnect attempts — "
              "send_audio/send_text/send_image will silently no-op until "
              "the app is restarted or connect() is called again.")
        self.connected = False
