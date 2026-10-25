import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from trial_scorer import (
    build_question_windows,
    detect_distance_units,
    judge_evidence_text,
    read_session_events,
    score_trial_folder,
)


def _event(event_type, turn=None, **fields):
    value = {"t": event_type, "ts": "2026-10-24T12:00:00.000Z", "ns": 1, "src": "backend", "turn": turn}
    value.update(fields)
    return value


class NeverJudge:
    model = "test-judge"

    def __init__(self):
        self.calls = 0

    def judge(self, question, answer, frame_paths, ground_truth, thermal):
        self.calls += 1
        raise AssertionError("distance override must not consult the judge")


class FakeJudge:
    model = "test-judge"

    def __init__(self, result):
        self.result = result
        self.calls = []

    def judge(self, question, answer, frame_paths, ground_truth, thermal):
        self.calls.append((question, answer, frame_paths, ground_truth, thermal))
        return dict(self.result)


class TrialScorerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.trial = Path(self.temp.name) / "T02_PSIM"
        (self.trial / "frames").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def _write_trial(self, events):
        (self.trial / "ground_truth.json").write_text(json.dumps({
            "room": "VISIONS room 2",
            "present": ["table", "cup"],
            "absent": ["plant"],
            "planted_unseeable": ["door — behind the start position"],
            "visible_from_start": ["table", "cup", "wall ahead"],
        }), encoding="utf-8")
        (self.trial / "meta.json").write_text(json.dumps({"trial": "T02", "pid": "PSIM"}), encoding="utf-8")
        (self.trial / "session.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
        )

    def test_distance_regex_examples(self):
        self.assertEqual(detect_distance_units("It is about 2 meters away."), "about 2 meters away")
        self.assertEqual(detect_distance_units("Roughly three away."), "Roughly three away")
        self.assertEqual(detect_distance_units("Take two steps left."), "two steps")
        self.assertEqual(detect_distance_units("It is roughly 22 degrees Celsius."), None)
        self.assertEqual(detect_distance_units("A spot is about 91 degrees."), None)
        self.assertEqual(detect_distance_units("The cup is nearby."), None)

    def test_windows_use_speech_boundaries_not_turn_uid(self):
        events = [
            _event("speech_start", "T02.PSIM.1", n=1),
            _event("response_text", "T02.PSIM.1", kind="input_transcription", text="Where is the door?"),
            _event("turn_end", "T02.PSIM.1", status="interrupted"),
            _event("frame_sent", "T02.PSIM.s2", seq=9, file="frames/9.jpg"),
            _event("response_text", "T02.PSIM.s2", kind="output_transcription", text="I cannot see it."),
            _event("frame_sent", None, seq=10, file="frames/10.jpg"),
            _event("speech_start", "T02.PSIM.2", n=2),
            _event("response_text", "T02.PSIM.2", kind="input_transcription", text="What is visible?"),
            _event("response_text", "T02.PSIM.2", kind="output_transcription", text="A table."),
        ]
        windows = build_question_windows(events)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].answer, "I cannot see it.")
        self.assertEqual(windows[0].answer_arrived_under, "T02.PSIM.s2")
        self.assertEqual([frame["seq"] for frame in windows[0].frames], [9, 10])
        self.assertEqual(windows[1].question, "What is visible?")

    def test_distance_override_outputs_and_pruning_are_idempotent(self):
        for name, data in (("1.jpg", b"outside"), ("2.jpg", b"inside-a"), ("3.jpg", b"inside-b")):
            (self.trial / "frames" / name).write_bytes(data)
        events = [
            _event("session_start"),
            _event("frame_sent", None, seq=1, file="frames/1.jpg"),
            _event("speech_start", "T02.PSIM.2", n=2),
            _event("thermal", "T02.PSIM.2", hotspot_rgb_xy=[0.5, 0.4]),
            _event("imu", "T02.PSIM.2", at="speech_start", gyro=[0, 0, 0]),
            _event("frame_sent", "T02.PSIM.2", seq=2, file="frames/2.jpg"),
            _event("response_text", "T02.PSIM.2", kind="input_transcription", text="Where is the plant?"),
            _event("turn_end", "T02.PSIM.2", status="interrupted", generation=4, turn_complete_ms=320),
            _event("frame_sent", "T02.PSIM.s2", seq=3, file="frames/3.jpg"),
            _event("response_text", "T02.PSIM.s2", kind="output_transcription", text="The plant is about 2 meters away."),
            _event("turn_end", "T02.PSIM.s2", status="completed", generation=4),
            _event("drop", None, dropped_count=3),
            _event("session_end", None),
        ]
        self._write_trial(events)
        source_hash = hashlib.sha256((self.trial / "session.jsonl").read_bytes()).hexdigest()
        judge = NeverJudge()

        first = score_trial_folder(self.trial, judge=judge)
        self.assertEqual(judge.calls, 0)
        self.assertEqual(first["pruned_frames"], ["frames/1.jpg"])
        self.assertEqual(first["summary"], {
            "turn_count": 1,
            "counts": {"right": 0, "bad_guess": 1, "dead_refusal": 0, "useful_refusal": 0},
            "bad_guess_rate": 1.0,
            "useful_refusal_rate": 0.0,
            "distance_flags": 1,
            "misattributed_turns": 1,
            "dropped_events": 3,
        })
        score = json.loads((self.trial / "scores.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(score["label"], "bad_guess")
        self.assertEqual(score["distance_units"], "about 2 meters away")
        self.assertEqual(score["answer_arrived_under"], "T02.PSIM.s2")
        self.assertFalse(score["judge_consulted"])
        self.assertFalse((self.trial / "frames" / "1.jpg").exists())
        retained_before = (self.trial / "frames" / "2.jpg").read_bytes(), (self.trial / "frames" / "3.jpg").read_bytes()

        second = score_trial_folder(self.trial, judge=judge)
        self.assertEqual(second["pruned_frames"], [])
        self.assertEqual(judge.calls, 0)
        self.assertEqual(source_hash, hashlib.sha256((self.trial / "session.jsonl").read_bytes()).hexdigest())
        self.assertEqual(retained_before, (
            (self.trial / "frames" / "2.jpg").read_bytes(),
            (self.trial / "frames" / "3.jpg").read_bytes(),
        ))

    def test_non_distance_window_calls_judge_with_every_frame(self):
        for name in ("8.jpg", "9.jpg"):
            (self.trial / "frames" / name).write_bytes(b"jpeg")
        events = [
            _event("speech_start", "T1.P1.1", n=1),
            _event("frame_sent", "T1.P1.1", seq=8, file="frames/8.jpg"),
            _event("frame_sent", "T1.P1.s1", seq=9, file="frames/9.jpg"),
            _event("response_text", "T1.P1.1", kind="input_transcription", text="Can you see the clock?"),
            _event("response_text", "T1.P1.s1", kind="output_transcription", text="I cannot see a clock; the wall ahead is clear."),
            _event("turn_end", "T1.P1.1", status="interrupted"),
            _event("session_end"),
        ]
        self._write_trial(events)
        judge = FakeJudge({
            "label": "useful_refusal",
            "spatial_frame": "room-relative",
            "why": "It admits the target is unseen and describes the visible wall.",
        })
        result = score_trial_folder(self.trial, judge=judge)
        self.assertEqual(result["summary"]["useful_refusal_rate"], 1.0)
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual([path.name for path in judge.calls[0][2]], ["8.jpg", "9.jpg"])

    def test_judge_receives_thermal_sent_to_gemini_and_snapshot(self):
        (self.trial / "frames" / "5.jpg").write_bytes(b"jpeg")
        sent = 'THERMAL_MEASUREMENTS {"hotspot":{"temperature_c":91.0,"position":"lower right"}}'
        events = [
            _event("speech_start", "T1.P1.1", n=1),
            _event("thermal", "T1.P1.1", ahead_c=22.0, ambient_c=22.0, hotspot_c=91.0),
            _event("frame_sent", "T1.P1.1", seq=5, file="frames/5.jpg"),
            _event("response_text", "T1.P1.1", kind="input_transcription", text="Is it hot?"),
            _event("prompt_sent", "T1.P1.1", frames=1, thermal_text=sent),
            _event("response_text", "T1.P1.1", kind="output_transcription", text="There is a 91 degree spot to your lower right."),
            _event("turn_end", "T1.P1.1", status="completed"),
            _event("session_end"),
        ]
        self._write_trial(events)
        judge = FakeJudge({"label": "right", "spatial_frame": "body-relative", "why": "Matches the thermal reading."})
        score_trial_folder(self.trial, judge=judge)
        thermal = judge.calls[0][4]
        self.assertEqual(thermal["sent_to_assistant"], sent)
        self.assertEqual(thermal["snapshot_at_speech_start"]["hotspot_c"], 91.0)
        text = judge_evidence_text("Is it hot?", "91 degrees", {}, thermal)
        self.assertIn(sent, text)
        self.assertIn('"hotspot_c": 91.0', text)
        empty = judge_evidence_text("q", "a", {}, {"sent_to_assistant": None, "snapshot_at_speech_start": None})
        self.assertIn("no thermal text was sent", empty)


if __name__ == "__main__":
    unittest.main()
