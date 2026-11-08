import asyncio
import unittest

import app_main
import gemini_live_client


class StreamEndQueueTests(unittest.TestCase):
    def test_marker_goes_after_pending_audio(self):
        queue = asyncio.Queue(maxsize=4)
        queue.put_nowait(b"a")
        queue.put_nowait(b"b")
        app_main._queue_audio_stream_end(queue)
        self.assertEqual([queue.get_nowait() for _ in range(3)], [b"a", b"b", app_main._AUDIO_STREAM_END])

    def test_full_queue_drops_oldest_audio_not_the_marker(self):
        queue = asyncio.Queue(maxsize=2)
        queue.put_nowait(b"a")
        queue.put_nowait(b"b")
        app_main._queue_audio_stream_end(queue)
        self.assertEqual([queue.get_nowait() for _ in range(2)], [b"b", app_main._AUDIO_STREAM_END])


class StreamEndSendTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_audio_stream_end_on_the_live_session(self):
        calls = []

        class FakeSession:
            async def send_realtime_input(self, **kwargs):
                calls.append(kwargs)

        client = app_main.gemini_live
        old = (client.session, client.connected)
        client.session, client.connected = FakeSession(), True
        try:
            sent = await client.send_audio_stream_end(turn_id=3)
        finally:
            client.session, client.connected = old
        self.assertTrue(sent)
        self.assertEqual(calls, [{"audio_stream_end": True}])

    async def test_not_connected_is_false_not_an_error(self):
        client = app_main.gemini_live
        old = (client.session, client.connected)
        client.session, client.connected = None, False
        try:
            self.assertFalse(await client.send_audio_stream_end(turn_id=1))
        finally:
            client.session, client.connected = old


if __name__ == "__main__":
    unittest.main()
