import json
import os
import tempfile
import time
import unittest
from unittest import mock

import app_main


def _trial(root, name, events, summary=False, age_sec=0.0):
    directory = os.path.join(root, name)
    os.makedirs(directory)
    log = os.path.join(directory, "session.jsonl")
    with open(log, "w", encoding="utf-8") as handle:
        for kind in events:
            handle.write(json.dumps({"t": kind}) + "\n")
    if summary:
        with open(os.path.join(directory, "summary.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
    if age_sec:
        stamp = time.time() - age_sec
        os.utime(log, (stamp, stamp))
    return directory


class PendingScoreResumeTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def test_needs_resume_only_for_recent_closed_unscored_trials_with_questions(self):
        cases = {
            "cut_off": (["session_start", "speech_start", "session_end"], False, 0, True),
            "scored": (["session_start", "speech_start", "session_end"], True, 0, False),
            "still_open": (["session_start", "speech_start"], False, 0, False),
            "no_questions": (["session_start", "session_end"], False, 0, False),
            "old": (["session_start", "speech_start", "session_end"], False, 3 * 86400, False),
        }
        for name, (events, summary, age, expected) in cases.items():
            directory = _trial(self.root, name, events, summary, age)
            self.assertEqual(app_main._trial_needs_resumed_score(directory), expected, name)

    def test_resumes_once_and_skips_the_open_trial(self):
        _trial(self.root, "T01_evis", ["session_start", "speech_start", "session_end"])
        _trial(self.root, "T02_evis", ["session_start", "speech_start", "session_end"])
        scheduled = []
        recorder = mock.Mock(enabled=True, root=self.root)
        recorder.status.return_value = {"session_id": "T02_evis"}
        with mock.patch.object(app_main, "trial_recorder", recorder), \
                mock.patch.object(app_main, "_schedule_trial_score",
                                  lambda name, directory: scheduled.append(name)):
            app_main._schedule_pending_trial_scores()
            app_main._schedule_pending_trial_scores()   # a second cold start
        self.assertEqual(scheduled, ["T01_evis"])
        self.assertTrue(os.path.isfile(os.path.join(self.root, "T01_evis", app_main.PENDING_SCORE_MARKER)))

    def test_disabled_recorder_does_nothing(self):
        _trial(self.root, "T01_evis", ["session_start", "speech_start", "session_end"])
        recorder = mock.Mock(enabled=False, root=self.root)
        with mock.patch.object(app_main, "trial_recorder", recorder), \
                mock.patch.object(app_main, "_schedule_trial_score") as schedule:
            app_main._schedule_pending_trial_scores()
        schedule.assert_not_called()


class FlyConfigTests(unittest.TestCase):
    def test_machine_stops_when_idle(self):
        with open("fly.toml", encoding="utf-8") as handle:
            config = handle.read()
        self.assertIn("auto_stop_machines = 'stop'", config)
        self.assertIn("min_machines_running = 0", config)
        self.assertIn("auto_start_machines = true", config)


if __name__ == "__main__":
    unittest.main()
