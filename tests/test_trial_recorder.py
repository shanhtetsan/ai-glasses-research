import asyncio
import json
import os
import tempfile
import time
import unittest

import numpy as np

from trial_recorder import TrialRecorder

BASE_KEYS = ["t", "ts", "ns", "src", "turn"]


def _events(root, session_id):
    with open(os.path.join(root, session_id, "session.jsonl"), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _summarizer(matrix, age_sec, calibration):
    return {"directly_ahead_mean_c": 21.5, "ambient_c": 20.0,
            "hotspot": {"temperature_c": 40.0, "thermal_norm": [0.5, 0.25], "rgb_norm_coarse": [0.5, 0.25]}}


def _rec(root, **kw):
    return TrialRecorder(enabled=True, root=root, thermal_summarizer=_summarizer,
                         static_fields=lambda: {"model": "m", "prompt_sha256": "abc", "git_sha": "g",
                                                "prompt_version_default": "sha-abc"}, **kw)


class TrialRecorderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = tempfile.mkdtemp()

    async def test_disabled_is_inert(self):
        rec = TrialRecorder(enabled=False, root=self.root)
        await rec.start()
        rec.ensure_session("ws_start")
        rec.frame_sent(1, 1, b"x", time.monotonic(), "pump")
        rec.mic_streaming(True, "start", 1)
        await rec.stop()
        self.assertEqual(os.listdir(self.root), [])

    async def test_turn_count_tracks_questions_and_survives_a_restart(self):
        rec = _rec(self.root)
        await rec.start()
        await rec.set_trial({"trial": "T08", "pid": "PSIM"}, None)
        self.assertEqual(rec.status()["turn_count"], 0)
        for n in (1, 2, 3):
            rec.speech_start(n, n, None, None)
            rec.speech_end(n, n)
        self.assertEqual(rec.status()["turn_count"], 3)
        await rec.stop()

        # A restart resumes the open session: the count comes back from the log
        # rather than restarting at zero under the researcher's eyes.
        resumed = _rec(self.root)
        await resumed.start()
        self.assertEqual(resumed.status()["session_id"], "T08_PSIM")
        self.assertEqual(resumed.status()["turn_count"], 3)
        resumed.speech_start(4, 4, None, None)
        self.assertEqual(resumed.status()["turn_count"], 4)
        await resumed.stop()

    async def test_turn_count_and_meta_are_none_when_no_session_is_open(self):
        rec = _rec(self.root)
        await rec.start()
        self.assertIsNone(rec.status()["turn_count"])
        self.assertIsNone(rec.status()["meta"])
        await rec.set_trial({"trial": "T09", "pid": "PSIM", "site": "VISIONS"}, None)
        self.assertEqual(rec.status()["meta"]["site"], "VISIONS")
        rec.end_session(reason="internal_end")
        self.assertIsNone(rec.status()["turn_count"])
        self.assertIsNone(rec.status()["meta"])
        await rec.stop()

    async def test_full_turn_schema_and_layout(self):
        rec = _rec(self.root)
        await rec.start()
        await rec.set_trial({"trial": "T07", "pid": "P03", "site": "lab", "cond": "A", "task": "find cup",
                             "success_criterion": "names cup"},
                            {"room": "4B", "present": ["cup"]})
        rec.mic_streaming(True, "start", 1)
        rec.frame_sent(None, 1, b"\xff\xd8one", time.monotonic(), "pump")
        imu = {"timestamp": time.monotonic(), "sequence": 9,
               "data": {"sequence": 9, "uptime_ms": 1000, "accel": {"x": 0, "y": 0, "z": 9.8},
                        "gyro": {"x": 1, "y": 2, "z": 3}}}
        rec.speech_start(3, 3, (np.zeros((24, 32)), time.monotonic(), {}), imu)
        rec.note_audio(3200)
        rec.frame_sent(3, 2, b"\xff\xd8two", time.monotonic(), "pump")
        rec.prompt_sent(3)   # model audio before speech_end: tail of an older reply, ignored
        rec.speech_end(3, 99)  # server's active turn differs; tagged by n
        rec.prompt_sent(3)
        rec.turn_end(3, {"turn_id": 3, "status": "completed",
                         "timestamps_ns": {"turn_created": 1_000_000, "turn_complete": 3_000_000},
                         "device_metrics": {"speech_end_to_first_i2s_ms": 5.0},
                         "input_transcription": "where is the cup", "response_text": "On the table."}, 1)
        rec.session_rotate("goaway", 2, 600.0, True)
        rec.end_session("internal_end")
        await rec.stop()

        sid = "T07_P03"
        events = _events(self.root, sid)
        for event in events:
            self.assertEqual(list(event)[:5], BASE_KEYS)
            self.assertEqual(event["src"], "backend")
            for value in event.values():
                self.assertNotIsInstance(value, dict)   # one level only
        types = [e["t"] for e in events]
        self.assertEqual(types[0], "session_start")
        self.assertIn("thermal", types)
        by_type = {e["t"]: e for e in events}
        self.assertEqual(by_type["speech_start"]["turn"], "T07.P03.3")
        self.assertEqual(by_type["speech_end"]["turn"], "T07.P03.3")
        self.assertEqual(types.count("prompt_sent"), 1)
        self.assertIsNone(events[types.index("frame_sent")]["turn"])   # outside a turn: null
        self.assertEqual(by_type["thermal"]["ahead_c"], 21.5)
        self.assertEqual(by_type["imu"]["gyro"], [1, 2, 3])
        self.assertEqual(by_type["prompt_sent"]["frames"], 1)
        self.assertEqual(by_type["prompt_sent"]["audio_ms"], 100.0)
        self.assertEqual(by_type["turn_end"]["turn_complete_ms"], 2.0)
        self.assertEqual(by_type["turn_end"]["device_metrics_speech_end_to_first_i2s_ms"], 5.0)
        self.assertEqual(by_type["turn_end"]["reply"], "On the table.")
        self.assertTrue(os.path.exists(os.path.join(self.root, sid, "frames", "2.jpg")))
        with open(os.path.join(self.root, sid, "meta.json")) as fh:
            self.assertEqual(json.load(fh)["model"], "m")
        with open(os.path.join(self.root, sid, "ground_truth.json")) as fh:
            self.assertEqual(json.load(fh)["room"], "4B")
        self.assertFalse(os.path.exists(os.path.join(self.root, "current.json")))

    async def test_untitled_session(self):
        rec = _rec(self.root)
        await rec.start()
        rec.ensure_session("ws_start")
        rec.mic_streaming(True, "start", 1)
        await rec.stop()
        dirs = [d for d in os.listdir(self.root) if d.startswith("untitled_")]
        self.assertEqual(len(dirs), 1)

    async def test_restart_resumes_and_never_reuses_uid_or_frame(self):
        rec = _rec(self.root)
        await rec.start()
        await rec.set_trial({"trial": "T1", "pid": "P1"})
        rec.speech_start(1, 1, None, None)
        rec.frame_sent(1, 1, b"first", time.monotonic(), "pump")
        await rec.stop()   # simulated deploy: session stays open

        rec2 = _rec(self.root)
        await rec2.start()  # resumes from current.json
        self.assertEqual(rec2.status()["session_id"], "T1_P1")
        rec2.speech_start(1, 1, None, None)  # iOS relaunched: n restarts at 1
        rec2.frame_sent(1, 1, b"second", time.monotonic(), "pump")  # rgb seq restarted
        await rec2.stop()

        events = _events(self.root, "T1_P1")
        starts = [e["turn"] for e in events if e["t"] == "speech_start"]
        self.assertEqual(starts, ["T1.P1.1", "T1.P1.1.r2"])
        self.assertIn("session_resume", [e["t"] for e in events])
        frames = sorted(os.listdir(os.path.join(self.root, "T1_P1", "frames")))
        self.assertEqual(frames, ["1.jpg", "1_1.jpg"])
        with open(os.path.join(self.root, "T1_P1", "frames", "1.jpg"), "rb") as fh:
            self.assertEqual(fh.read(), b"first")

    async def test_repeat_start_same_trial_is_resume(self):
        rec = _rec(self.root)
        await rec.start()
        await rec.set_trial({"trial": "T2", "pid": "P2"})
        await rec.set_trial({"trial": "T2", "pid": "P2"})
        rec.ensure_session("ws_start")
        await rec.stop()
        types = [e["t"] for e in _events(self.root, "T2_P2")]
        self.assertEqual(types.count("session_start"), 1)
        self.assertEqual(types.count("session_resume"), 2)

    async def test_end_session_notifies_derived_scoring_after_close_is_queued(self):
        ended = []
        rec = TrialRecorder(
            enabled=True,
            root=self.root,
            on_session_ended=lambda session_id, directory: ended.append((session_id, directory)),
        )
        await rec.start()
        await rec.set_trial({"trial": "T9", "pid": "P2"}, {"present": []})
        self.assertEqual(rec.end_session("test"), "T9_P2")
        await rec.flush()
        await rec.stop()
        self.assertEqual(ended, [("T9_P2", os.path.join(self.root, "T9_P2"))])
        self.assertEqual(_events(self.root, "T9_P2")[-1]["t"], "session_end")

    async def test_server_created_turn_gets_s_uid(self):
        rec = _rec(self.root)
        await rec.start()
        await rec.set_trial({"trial": "T3", "pid": "P3"})
        rec.turn_end(17, {"turn_id": 17, "status": "completed", "timestamps_ns": {}}, 1)
        await rec.stop()
        end = [e for e in _events(self.root, "T3_P3") if e["t"] == "turn_end"][0]
        self.assertEqual(end["turn"], "T3.P3.s1")

    async def test_backpressure_drops_counts_and_recovers(self):
        rec = _rec(self.root, queue_maxsize=8)
        await rec.start()
        await rec.set_trial({"trial": "T4", "pid": "P4"})
        await asyncio.sleep(0.05)
        rec._task.cancel()          # stall the writer
        try:
            await rec._task
        except asyncio.CancelledError:
            pass
        started = time.perf_counter()
        for i in range(100):
            rec.mic_streaming(bool(i % 2), "flood", i)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.assertLess(elapsed_ms, 50)   # never blocks the caller
        dropped = rec.status()["events_dropped"]
        self.assertGreater(dropped, 80)
        rec._task = asyncio.create_task(rec._writer())   # writer recovers
        await rec._queue.join()
        rec.mic_streaming(True, "after", 999)
        await rec.stop()
        drops = [e for e in _events(self.root, "T4_P4") if e["t"] == "drop"]
        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0]["dropped_count"], dropped)

    async def test_rejects_unsafe_ids(self):
        rec = _rec(self.root)
        await rec.start()
        with self.assertRaises(ValueError):
            await rec.set_trial({"trial": "../etc", "pid": "P1"})
        await rec.stop()

    async def test_writer_failure_never_raises(self):
        rec = _rec(self.root)
        await rec.start()
        await rec.set_trial({"trial": "T5", "pid": "P5"})
        await rec._queue.join()
        rec._fh.close()   # force write errors
        rec._fh = open(os.devnull, "r")
        rec.mic_streaming(True, "start", 1)
        await rec._queue.join()
        self.assertGreaterEqual(rec.status()["write_errors"], 1)
        await rec.stop()


if __name__ == "__main__":
    unittest.main()
