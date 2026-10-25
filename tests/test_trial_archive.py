import json
import tempfile
import unittest
from pathlib import Path

from trial_archive import TrialArchive, integrate_heading


class TrialArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / "T1_P1"
        (self.folder / "frames").mkdir(parents=True)
        (self.folder / "meta.json").write_text(json.dumps({
            "trial": "T1", "pid": "P1", "site": "lab", "cond": "A",
            "prompt_version": "v1", "created_ts": "2026-01-01T00:00:00Z",
        }))
        (self.folder / "ground_truth.json").write_text('{"present":["cup"]}')
        (self.folder / "summary.json").write_text(json.dumps({
            "turn_count": 1, "bad_guess_rate": 0.0,
        }))
        events = [
            {"t": "speech_start", "turn": "T1.P1.1", "n": 1},
            {"t": "imu", "turn": "T1.P1.1", "at": "speech_start", "uptime_ms": 1000,
             "accel": [0, 0, 9.8], "gyro": [0, 0, 10]},
            {"t": "imu", "turn": "T1.P1.1", "at": "stream", "uptime_ms": 1100,
             "accel": [0, 0, 9.8], "gyro": [0, 0, 10]},
            {"t": "frame_sent", "turn": "T1.P1.1", "seq": 7, "file": "frames/7.jpg"},
            {"t": "response_text", "turn": "T1.P1.1", "kind": "input_transcription", "text": "Cup?"},
            {"t": "response_text", "turn": "T1.P1.1", "kind": "output_transcription", "text": "Yes."},
            {"t": "session_end", "turn": None},
        ]
        (self.folder / "session.jsonl").write_text("".join(json.dumps(item) + "\n" for item in events))
        (self.folder / "frames" / "7.jpg").write_bytes(b"jpeg")

    def tearDown(self):
        self.temp.cleanup()

    def test_local_list_detail_frame_and_zip(self):
        archive = TrialArchive(self.root, None)
        self.assertEqual(archive.list_trials()[0]["id"], "T1_P1")
        detail = archive.get_trial("T1_P1")
        self.assertEqual(detail["windows"][0]["question"], "Cup?")
        self.assertEqual(detail["windows"][0]["frames"][0]["url"], "/trials/T1_P1/frames/7.jpg")
        self.assertGreater(len(archive.zip_bytes("T1_P1")), 0)

    def test_heading_integrates_projected_gyro(self):
        heading = integrate_heading([
            {"uptime_ms": 1000, "accel": [0, 0, 9.8], "gyro": [0, 0, 10]},
            {"uptime_ms": 1100, "accel": [0, 0, 9.8], "gyro": [0, 0, 10]},
        ])
        self.assertEqual(heading[-1]["heading_deg"], 1.0)


if __name__ == "__main__":
    unittest.main()
