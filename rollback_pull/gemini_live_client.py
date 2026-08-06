import os
import asyncio

from google import genai
from google.genai import types

SMART_GLASSES_SYSTEM_INSTRUCTION = """
You are the conversational assistant inside wearable smart glasses for a blind
or low-vision user. The live session can receive JPEG camera frames from the
user's first-person viewpoint while microphone audio is streaming.

Treat natural and indirect questions as visually grounded whenever sight would
help answer them. Examples include asking what is ahead, what the user is
looking at, describing the scene, locating keys or a phone, checking whether an
object is present, reading visible text, or asking whether a path appears clear.
The user does not need to use a fixed command phrase.

Use the most recently received camera evidence for the current question. Never
claim that you have no camera access merely because the request is phrased
differently. If no recent usable frame is available, or the view is dark,
blurred, obstructed, or does not contain the requested object, say that clearly
and briefly instead of inventing details. Do not ask a blind user to visually
confirm your answer. Give the direct answer first and keep spoken responses
concise unless the user requests more detail.

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
        self.on_interrupted = None     # called when the user barges in and cuts off the model

        self._response_modality = "AUDIO"  # remembered so auto-reconnect uses the same mode
        self._shutting_down = False        # True once disconnect() is called deliberately,
                                            # so receive_loop knows not to try reconnecting
        self._send_lock = asyncio.Lock()     # serialize audio/image/text websocket writes

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

        self._response_modality = response_modality
        self._shutting_down = False

        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        print("[Gemini Live] Connecting...")
        await self._open_session()
        print("[Gemini Live] Connected")

        # Start listening for Gemini responses
        self.receive_task = asyncio.create_task(
            self.receive_loop()
        )

    async def _open_session(self):
        """Actually open the websocket session. Used by connect() and by
        the auto-reconnect logic in receive_loop()."""
        # gemini-3.1-flash-live-preview is the current recommended Live API
        # model (as of July 2026) for the Gemini Developer API surface
        # (API-key auth). Older Live model ids like gemini-live-2.5-flash-preview
        # and gemini-2.0-flash-live-001 have been shut down; verify at
        # https://ai.google.dev/gemini-api/docs/live-api if this ever 404s again.
        self._live_cm = self.client.aio.live.connect(
            model="gemini-3.1-flash-live-preview",
            config={
                "response_modalities": [self._response_modality],
                "system_instruction": {
                    "parts": [{"text": SMART_GLASSES_SYSTEM_INSTRUCTION}]
                },
                "input_audio_transcription": {},
                **({"output_audio_transcription": {}} if self._response_modality == "AUDIO" else {}),
            }
        )
        self.session = await self._live_cm.__aenter__()
        self.connected = True
    
    async def disconnect(self):
        print("[Gemini Live] Disconnecting...")

        self._shutting_down = True
        self.connected = False

        if self.receive_task:
            self.receive_task.cancel()

        if self._live_cm:
            try:
                await self._live_cm.__aexit__(None, None, None)
            except Exception as e:
                print(f"[Gemini Live] Error closing session: {e}")

        self.session = None
        self._live_cm = None

        print("[Gemini Live] Disconnected")


    async def send_text(self, text: str) -> bool:
        if not self.connected or self.session is None:
            return False
        session = self.session
        try:
            async with self._send_lock:
                if not self.connected or self.session is not session:
                    return False
                await session.send_realtime_input(text=text)
            return True
        except Exception as e:
            print(f"[Gemini Live] send_text failed: {e}")
            return False

    async def send_audio(self, audio_bytes: bytes) -> bool:
        if not self.connected or self.session is None:
            return False
        session = self.session
        try:
            async with self._send_lock:
                if not self.connected or self.session is not session:
                    return False
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=audio_bytes,
                        mime_type="audio/pcm;rate=16000",
                    )
                )
            return True
        except Exception as e:
            print(f"[Gemini Live] send_audio failed: {e}")
            return False

    async def send_image(self, image_bytes: bytes) -> bool:
        if not self.connected or self.session is None:
            return False
        session = self.session
        try:
            async with self._send_lock:
                if not self.connected or self.session is not session:
                    return False
                await asyncio.wait_for(
                    session.send_realtime_input(
                        video=types.Blob(
                            data=image_bytes,
                            mime_type="image/jpeg",
                        )
                    ),
                    timeout=3.0,
                )
            return True
        except asyncio.TimeoutError:
            print("[Gemini Live] send_image timed out after 3s")
            return False
        except Exception as e:
            print(f"[Gemini Live] send_image failed: {e}")
            return False

    async def receive_loop(self):
        while self.connected:
            try:
                # session.receive() is documented/implemented (google-genai's
                # AsyncSession.receive) to yield exactly one turn and then end
                # the generator on its own right after turn_complete — that is
                # normal per-turn completion, not a dropped connection. Track
                # whether we saw it this iteration so the code after the
                # `async for` can tell "Gemini finished normally" apart from
                # "the stream died with no turn_complete ever received".
                turn_complete_seen = False
                async for response in self.session.receive():
                    server_content = response.server_content

                    if server_content:
                        # INPUT TEXT
                        if server_content.input_transcription:
                            text = server_content.input_transcription.text
                            if text and self.on_input_transcription:
                                await self.on_input_transcription(text)
                        # AUDIO
                        if server_content.model_turn:
                            for part in server_content.model_turn.parts:
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
                            if self.on_turn_complete:
                                await self.on_turn_complete()
                        # USER BARGED IN — model's current response was cut off
                        if server_content.interrupted:
                            if self.on_interrupted:
                                await self.on_interrupted()
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
                    # The current turn (if any) never got a turn_complete/
                    # interrupted from Gemini — treat it as interrupted so
                    # callers (e.g. the ESP32 TTS state) don't get stuck
                    # waiting on a signal that will never arrive.
                    if self.on_interrupted:
                        await self.on_interrupted()
                    await self._reconnect_with_backoff()
            except asyncio.CancelledError:
                print("[Gemini Live] receive loop task cancelled")
                return
            except Exception as e:
                print(f"[Gemini Live] Receive loop error: {e}")
                self.connected = False
                if self._shutting_down:
                    return
                # Same reasoning as above: the turn in progress (if any) was
                # abandoned mid-stream, so fire on_interrupted for cleanup
                # before reconnecting.
                if self.on_interrupted:
                    await self.on_interrupted()
                await self._reconnect_with_backoff()
                # loop condition re-checks self.connected — if the reconnect
                # above succeeded, we fall back into `while self.connected`
                # and keep listening on the new session.

    async def _reconnect_with_backoff(self, max_attempts: int = 8):
        """Try to re-open the Live session after an unexpected disconnect.
        Backs off 2s, 4s, 8s... capped at 30s between attempts."""
        if self._live_cm:
            try:
                await self._live_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._live_cm = None
        self.session = None

        for attempt in range(1, max_attempts + 1):
            if self._shutting_down:
                return
            delay = min(2 ** attempt, 30)
            print(f"[Gemini Live] Reconnecting in {delay}s (attempt {attempt}/{max_attempts})...")
            await asyncio.sleep(delay)
            if self._shutting_down:
                return
            try:
                await self._open_session()
                print("[Gemini Live] Reconnected")
                return
            except Exception as e:
                print(f"[Gemini Live] Reconnect attempt {attempt} failed: {e}")

        print("[Gemini Live] Giving up after max reconnect attempts — "
              "send_audio/send_text/send_image will silently no-op until "
              "the app is restarted or connect() is called again.")
        self.connected = False