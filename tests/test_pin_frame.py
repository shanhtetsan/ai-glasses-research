"""PIN_FRAME_EVERY_TURN: one fresh pre_response frame per turn, phrase list or not."""
import time
import unittest
from unittest import mock

import app_main
from stability_runtime import LatestFrameStore


class PinFrameEveryTurnTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = LatestFrameStore()
        self.sends = []
        self.texts = []

        async def send_image(data, *, turn_id=None, sequence=None, source="image"):
            self.sends.append({"turn_id": turn_id, "sequence": sequence, "source": source})
            return True

        async def send_text(text, *, turn_id=None, source="text"):
            self.texts.append(source)
            return True

        async def broadcast(_text):
            return None

        self.patches = [
            mock.patch.object(app_main, "latest_rgb", self.store),
            mock.patch.object(app_main.gemini_live, "send_image", send_image),
            mock.patch.object(app_main.gemini_live, "send_text", send_text),
            mock.patch.object(app_main, "ui_broadcast_partial", broadcast),
        ]
        for patch in self.patches:
            patch.start()
        app_main._clear_gemini_turn_state("test_setup")

    async def asyncTearDown(self):
        app_main._clear_gemini_turn_state("test_teardown")
        for patch in reversed(self.patches):
            patch.stop()

    def _pre_response(self):
        return [send for send in self.sends if send["source"] == "pre_response"]

    async def test_non_phrase_question_gets_exactly_one_fresh_frame(self):
        self.store.update(b"\xff\xd8old")
        frame = self.store.update(b"\xff\xd8new")
        with mock.patch.object(app_main, "PIN_FRAME_EVERY_TURN", True):
            # Streamed transcript deltas: none match _VISION_PHRASES.
            for delta in ("what side ", "is the ", "bottle on"):
                await app_main._on_input_transcription(delta)
        sends = self._pre_response()
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0]["sequence"], frame.sequence)
        self.assertNotEqual(sends[0]["sequence"], 0)
        self.assertFalse(app_main._vision_intent_for_turn)   # ablation signal kept
        self.assertEqual(app_main._vision_frame_sequence_for_turn, frame.sequence)

    async def test_flag_off_restores_phrase_gate(self):
        self.store.update(b"\xff\xd8frame")
        with mock.patch.object(app_main, "PIN_FRAME_EVERY_TURN", False):
            await app_main._on_input_transcription("what side is the bottle on")
        self.assertEqual(self._pre_response(), [])

    async def test_phrase_question_still_sends_once_and_sets_intent(self):
        self.store.update(b"\xff\xd8frame")
        with mock.patch.object(app_main, "PIN_FRAME_EVERY_TURN", True):
            await app_main._on_input_transcription("what do you see")
            await app_main._on_input_transcription(" in front of me")
        self.assertEqual(len(self._pre_response()), 1)
        self.assertTrue(app_main._vision_intent_for_turn)

    async def test_stale_frame_is_not_pinned_and_model_is_not_told_to_decline(self):
        self.store.update(b"\xff\xd8frame", time.monotonic() - app_main.VISION_FRAME_MAX_AGE_SEC - 5)
        with mock.patch.object(app_main, "PIN_FRAME_EVERY_TURN", True):
            await app_main._on_input_transcription("how big is it")
        self.assertEqual(self._pre_response(), [])
        # Only explicit vision requests get the "you cannot see" instruction.
        self.assertNotIn("vision_unavailable", self.texts)


class PumpSequenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_pump_send_carries_frame_sequence_and_turn(self):
        sends = []

        async def send_image(data, *, turn_id=None, sequence=None, source="image"):
            sends.append((turn_id, sequence, source))
            return True

        holder = {"data": b"\xff\xd8x", "turn": 7, "seq": 4242, "received_at": time.monotonic()}
        event = __import__("asyncio").Event()
        event.set()
        with mock.patch.object(app_main.gemini_live, "send_image", send_image), \
                mock.patch.object(app_main.trial_recorder, "enabled", False):
            task = __import__("asyncio").create_task(app_main._gemini_image_pump_loop(holder, event))
            for _ in range(10):
                await __import__("asyncio").sleep(0)
            task.cancel()
        self.assertEqual(sends, [(7, 4242, "image")])


if __name__ == "__main__":
    unittest.main()
