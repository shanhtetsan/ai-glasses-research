import asyncio
import contextlib
import io
import unittest

from google.genai import types

from gemini_live_client import (
    GEMINI_LIVE_MODEL,
    SMART_GLASSES_SYSTEM_INSTRUCTION,
    GeminiLiveClient,
)


class _FakeSession:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class _ReceiveSession(_FakeSession):
    def __init__(self, responses=None, block_after=False):
        super().__init__()
        self.responses = list(responses or [])
        self.block_after = block_after
        self.blocker = asyncio.Event()

    async def receive(self):
        for response in self.responses:
            yield response
        if self.block_after:
            await self.blocker.wait()


class _FakeContextManager:
    def __init__(self, outcome):
        self.outcome = outcome
        self.exited = False

    async def __aenter__(self):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    async def __aexit__(self, *_args):
        self.exited = True


class _FakeLive:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.context_managers = []

    def connect(self, **kwargs):
        self.calls.append(kwargs)
        cm = _FakeContextManager(self.outcomes.pop(0))
        self.context_managers.append(cm)
        return cm


class _FakeAio:
    def __init__(self, live):
        self.live = live


class _FakeClient:
    def __init__(self, outcomes):
        self.live = _FakeLive(outcomes)
        self.aio = _FakeAio(self.live)


class GeminiLiveConfigurationTests(unittest.TestCase):
    def test_fresh_config_enables_resumption_compression_and_prompt(self):
        client = GeminiLiveClient()
        config = types.LiveConnectConfig.model_validate(client._live_config())

        self.assertEqual(config.response_modalities, ["AUDIO"])
        self.assertIsNotNone(config.session_resumption)
        self.assertIsNone(config.session_resumption.handle)
        self.assertIsNone(config.session_resumption.transparent)
        self.assertNotIn(
            "transparent", config.session_resumption.model_fields_set
        )
        self.assertIsNotNone(config.context_window_compression)
        self.assertIsNotNone(config.context_window_compression.sliding_window)
        self.assertEqual(
            config.system_instruction.parts[0].text,
            SMART_GLASSES_SYSTEM_INSTRUCTION,
        )
        self.assertIsNotNone(config.input_audio_transcription)
        self.assertIsNotNone(config.output_audio_transcription)

    def test_goaway_time_left_uses_sdk_duration_string(self):
        self.assertEqual(GeminiLiveClient._duration_seconds("5s"), 5.0)
        self.assertEqual(GeminiLiveClient._duration_seconds("5.250s"), 5.25)
        with self.assertRaises((TypeError, ValueError)):
            GeminiLiveClient._duration_seconds(None)
        with self.assertRaises(ValueError):
            GeminiLiveClient._duration_seconds("5")

    def test_fingerprint_is_stable_and_does_not_contain_prompt(self):
        first = GeminiLiveClient()._config_fingerprint()
        second = GeminiLiveClient()._config_fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertNotIn(SMART_GLASSES_SYSTEM_INSTRUCTION, first)


class GeminiLiveLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_connect_refuses_second_receive_lifecycle(self):
        client = GeminiLiveClient()
        client.receive_task = asyncio.create_task(asyncio.sleep(60))
        try:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                await client.connect()
            self.assertIsNone(client.client)
            self.assertIn("lifecycle already active", output.getvalue())
        finally:
            client.receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await client.receive_task

    async def test_failed_resumption_falls_back_to_fully_configured_fresh_session(self):
        secret_handle = "secret-resumption-handle"
        sdk_client = _FakeClient([RuntimeError("rejected"), _FakeSession()])
        client = GeminiLiveClient()
        client.client = sdk_client
        client._latest_resumption_handle = secret_handle

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            await client._open_with_fresh_fallback()

        self.assertEqual(len(sdk_client.live.calls), 2)
        resumed_call, fresh_call = sdk_client.live.calls
        self.assertEqual(resumed_call["model"], GEMINI_LIVE_MODEL)
        self.assertEqual(
            resumed_call["config"]["session_resumption"].handle,
            secret_handle,
        )
        self.assertIsNone(
            fresh_call["config"]["session_resumption"].handle
        )
        self.assertIsNotNone(fresh_call["config"]["context_window_compression"])
        self.assertEqual(
            fresh_call["config"]["system_instruction"]["parts"][0]["text"],
            SMART_GLASSES_SYSTEM_INSTRUCTION,
        )
        self.assertEqual(client.session_generation, 1)
        self.assertEqual(client._session_mode, "fresh")
        self.assertTrue(client.connected)
        self.assertNotIn(secret_handle, output.getvalue())

    async def test_controlled_rotation_closes_old_before_opening_resumed(self):
        old_session = _FakeSession()
        old_cm = _FakeContextManager(old_session)
        new_session = _FakeSession()
        sdk_client = _FakeClient([new_session])
        original_connect = sdk_client.live.connect

        def connect_after_old_close(**kwargs):
            self.assertTrue(old_cm.exited)
            return original_connect(**kwargs)

        sdk_client.live.connect = connect_after_old_close
        client = GeminiLiveClient()
        client.client = sdk_client
        client.session = old_session
        client._live_cm = old_cm
        client.connected = True
        client.session_generation = 1
        client._session_opened_at = 1.0
        client._latest_resumption_handle = "secret-resumption-handle"

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            await client._controlled_rotation("goaway")

        self.assertTrue(old_cm.exited)
        self.assertIs(client.session, new_session)
        self.assertEqual(client.session_generation, 2)
        self.assertEqual(client._session_mode, "resumed")
        self.assertEqual(len(sdk_client.live.calls), 1)
        self.assertNotIn("secret-resumption-handle", output.getvalue())

    async def test_receive_loop_consumes_handle_and_rotates_on_goaway(self):
        secret_handle = "secret-resumption-handle"
        old_session = _ReceiveSession(
            responses=[
                types.LiveServerMessage(
                    session_resumption_update=(
                        types.LiveServerSessionResumptionUpdate(
                            resumable=True,
                            new_handle=secret_handle,
                        )
                    ),
                    go_away=types.LiveServerGoAway(time_left="10s"),
                )
            ]
        )
        old_cm = _FakeContextManager(old_session)
        new_session = _ReceiveSession(block_after=True)
        sdk_client = _FakeClient([new_session])
        client = GeminiLiveClient()
        client.client = sdk_client
        client.session = old_session
        client._live_cm = old_cm
        client.connected = True
        client.session_generation = 1

        output = io.StringIO()
        task = asyncio.create_task(client.receive_loop())
        client.receive_task = task
        try:
            with contextlib.redirect_stdout(output):
                for _ in range(20):
                    if client.session_generation == 2:
                        break
                    await asyncio.sleep(0)
            self.assertEqual(client.session_generation, 2)
            self.assertTrue(old_cm.exited)
            self.assertIs(client.session, new_session)
            self.assertEqual(client._session_mode, "resumed")
            self.assertEqual(
                sdk_client.live.calls[0]["config"]["session_resumption"].handle,
                secret_handle,
            )
            self.assertNotIn(secret_handle, output.getvalue())
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def test_resumption_update_retains_only_valid_handle_without_logging_it(self):
        client = GeminiLiveClient()
        secret_handle = "secret-resumption-handle"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            client._retain_resumption_update(
                types.LiveServerSessionResumptionUpdate(
                    resumable=True,
                    new_handle=secret_handle,
                )
            )
            client._retain_resumption_update(
                types.LiveServerSessionResumptionUpdate(
                    resumable=False,
                    new_handle="ignored-handle",
                )
            )

        self.assertEqual(client._latest_resumption_handle, secret_handle)
        self.assertNotIn(secret_handle, output.getvalue())
        self.assertNotIn("ignored-handle", output.getvalue())

    async def test_goaway_watchdog_uses_safety_margin(self):
        client = GeminiLiveClient()
        session = _FakeSession()
        client.session = session
        client.connected = True
        client.session_generation = 7

        goaway = types.LiveServerGoAway(time_left="2.5s")
        client._handle_goaway(goaway, session)
        self.assertTrue(client._rotation_pending)
        self.assertEqual(client._rotation_reason, "goaway")
        self.assertIsNotNone(client._goaway_deadline_task)
        client._cancel_goaway_deadline()
        await asyncio.sleep(0)

    async def test_goaway_watchdog_forces_close_before_short_deadline(self):
        client = GeminiLiveClient()
        session = _FakeSession()
        client.session = session
        client.connected = True
        client.session_generation = 8

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            client._handle_goaway(
                types.LiveServerGoAway(time_left="1s"), session
            )
            await asyncio.sleep(0.01)

        self.assertTrue(session.closed)
        self.assertTrue(client._goaway_deadline_forced)


if __name__ == "__main__":
    unittest.main()
