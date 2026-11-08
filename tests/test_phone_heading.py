import json
import math
import os
import struct
import tempfile
import time
import unittest

from stability_runtime import HEADING_STRUCT, MSG_TYPE_HEADING, parse_sensor_message
import trial_recorder
from trial_recorder import TrialRecorder


def _packet(seq, deg, accuracy):
    return bytes([MSG_TYPE_HEADING]) + HEADING_STRUCT.pack(seq, deg, accuracy)


def _events(root, session_id):
    with open(os.path.join(root, session_id, "session.jsonl"), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _rec(root):
    return TrialRecorder(enabled=True, root=root,
                         static_fields=lambda: {"model": "m", "prompt_sha256": "abc", "git_sha": "g",
                                                "prompt_version_default": "sha-abc"})


class HeadingParseTests(unittest.TestCase):
    def test_parses_the_page_packet(self):
        msg_type, payload = parse_sensor_message(_packet(7, 123.5, 15.0))
        self.assertEqual(msg_type, MSG_TYPE_HEADING)
        self.assertEqual(payload, {"sequence": 7, "heading_deg": 123.5, "accuracy_deg": 15.0})

    def test_negative_accuracy_means_unknown(self):
        _, payload = parse_sensor_message(_packet(1, 0.0, -1.0))
        self.assertIsNone(payload["accuracy_deg"])

    def test_rejects_bad_packets(self):
        for bad in (
            _packet(1, 360.0, 1.0),
            _packet(1, -0.5, 1.0),
            _packet(1, math.nan, 1.0),
            _packet(1, 10.0, math.inf),
            bytes([MSG_TYPE_HEADING]) + struct.pack("<If", 1, 10.0),
        ):
            with self.assertRaises(ValueError):
                parse_sensor_message(bad)

    def test_matches_the_page_format(self):
        with open("static/field_test_core.mjs", encoding="utf-8") as handle:
            core = handle.read()
        self.assertIn(f"MSG_TYPE_HEADING = 0x{MSG_TYPE_HEADING:02x}", core)
        self.assertEqual(HEADING_STRUCT.format, "<Iff")
        self.assertEqual(HEADING_STRUCT.size + 1, 13)   # headingMessage builds 13 bytes


class HeadingRecorderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = tempfile.mkdtemp()

    def _heading(self, deg, seq=1):
        return {"timestamp": time.monotonic(),
                "data": {"sequence": seq, "heading_deg": deg, "accuracy_deg": 10.0}}

    async def test_heading_at_speech_start_and_streamed_in_turn(self):
        rec = _rec(self.root)
        await rec.start()
        await rec.set_trial({"trial": "H01", "pid": "evis"}, None)
        rec.heading_sample(self._heading(5.0))          # no turn yet: not recorded
        rec.speech_start(1, 1, None, None, self._heading(90.0))
        rec.heading_sample(self._heading(100.0, 2))
        rec.heading_sample(self._heading(101.0, 3))      # inside 0.5 s: throttled
        rec._last_heading_stream -= trial_recorder.HEADING_STREAM_INTERVAL_SEC
        rec.heading_sample(self._heading(180.0, 4))
        rec.end_session("internal_end")
        await rec.stop()

        headings = [e for e in _events(self.root, "H01_evis") if e["t"] == "heading"]
        self.assertEqual([(e["at"], e["deg"]) for e in headings],
                         [("speech_start", 90.0), ("stream", 100.0), ("stream", 180.0)])
        self.assertTrue(all(e["turn"] == headings[0]["turn"] for e in headings))
        self.assertEqual(headings[0]["accuracy_deg"], 10.0)
        self.assertIsInstance(headings[0]["age_ms"], int)

    async def test_no_heading_event_without_a_phone_heading(self):
        rec = _rec(self.root)
        await rec.start()
        await rec.set_trial({"trial": "H02", "pid": "evis"}, None)
        rec.speech_start(1, 1, None, None)
        rec.speech_start(2, 2, None, None, {"timestamp": 0.0, "data": None})
        rec.end_session("internal_end")
        await rec.stop()
        types = [e["t"] for e in _events(self.root, "H02_evis")]
        self.assertNotIn("heading", types)
        self.assertEqual(types.count("imu"), 2)          # glasses-style IMU record unchanged


if __name__ == "__main__":
    unittest.main()
