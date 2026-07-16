import os
import asyncio

from google import genai
from google.genai import types

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


    async def send_text(self, text):
        if not self.connected:
            return
        
        await self.session.send_realtime_input(
            text=text
        )

    async def send_audio(self, audio_bytes):
        if not self.connected:
            return

        await self.session.send_realtime_input(
            audio=types.Blob(
                data=audio_bytes,
                mime_type="audio/pcm;rate=16000"
            )
        )

    async def send_image(self, frame):
        if not self.connected:
            return

        # Bounded even though callers now fire this via create_task: without
        # a timeout a stalled/degraded Gemini socket can leave this await
        # pending indefinitely, and a pile of never-finishing tasks would
        # defeat the caller's "single in-flight slot" backpressure guard.
        await asyncio.wait_for(
            self.session.send_realtime_input(
                video=types.Blob(
                    data=frame,
                    mime_type="image/jpeg"
                )
            ),
            timeout=3.0,
        )

    async def receive_loop(self):
        while self.connected:
            try:
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
                            if self.on_turn_complete:
                                await self.on_turn_complete()
                        # USER BARGED IN — model's current response was cut off
                        if server_content.interrupted:
                            if self.on_interrupted:
                                await self.on_interrupted()
                # The `async for` above ended on its own (server closed the
                # response stream without erroring) — if we're not shutting
                # down deliberately, treat this the same as a dropped
                # connection and try to reconnect rather than exiting silently.
                if not self._shutting_down:
                    print("[Gemini Live] Response stream ended unexpectedly")
                    self.connected = False
                    await self._reconnect_with_backoff()
            except asyncio.CancelledError:
                print("[Gemini Live] receive loop task cancelled")
                return
            except Exception as e:
                print(f"[Gemini Live] Receive loop error: {e}")
                self.connected = False
                if self._shutting_down:
                    return
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