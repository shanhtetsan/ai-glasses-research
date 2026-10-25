"""The experiment arm as a per-trial setting, and the record of what really ran.

Three things are under test, in this order: the resolver on its own, the
override's lifetime against a real recorder and the real routes, and the
mislabelling checks the viewer draws its warnings from.

The behavioural assertions deliberately go through _on_input_transcription
rather than reading a flag back: the point of the change is that a turn is
pinned or not pinned, not that a dict holds a boolean.
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from starlette.requests import Request

import app_main
from stability_runtime import LatestFrameStore
from trial_config import (
    RECORDED_CONFIG_KEYS,
    NullSessionConfig,
    SessionConfig,
    condition_mismatch,
    parse_bool,
    recorded_config,
    resume_drift,
)
from trial_recorder import TrialRecorder


def _post(path, body, token=None):
    headers = [] if token is None else [(b"x-trial-token", token.encode("utf-8"))]
    scope = {
        "type": "http", "http_version": "1.1", "method": "POST", "scheme": "https",
        "path": path, "raw_path": path.encode("utf-8"), "query_string": b"",
        "headers": headers, "client": ("test", 1), "server": ("test", 443),
    }
    payload = json.dumps(body).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    return Request(scope, receive)


def _json(response):
    return json.loads(bytes(response.body).decode("utf-8"))


def _events(root, session_id):
    with open(os.path.join(root, session_id, "session.jsonl"), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _meta(root, session_id):
    with open(os.path.join(root, session_id, "meta.json"), encoding="utf-8") as fh:
        return json.load(fh)


class SessionConfigTests(unittest.TestCase):
    """The resolver alone: what it accepts, what it refuses, what it reports."""

    def setUp(self):
        self.default = True
        self.config = SessionConfig()
        self.config.register("pin_frame_every_turn", lambda: self.default,
                             parse=parse_bool, settable=True)
        self.config.register("model", lambda: "model-x")

    def test_default_until_an_override_is_applied_then_cleared(self):
        self.assertTrue(self.config.get("pin_frame_every_turn"))
        self.config.apply(self.config.parse({"pin_frame_every_turn": False}))
        self.assertFalse(self.config.get("pin_frame_every_turn"))
        self.config.clear()
        self.assertTrue(self.config.get("pin_frame_every_turn"))

    def test_default_is_read_at_the_moment_it_is_asked_for(self):
        self.default = False
        self.assertFalse(self.config.get("pin_frame_every_turn"))

    def test_no_config_and_an_empty_config_both_mean_no_override(self):
        for value in (None, {}):
            self.assertEqual(self.config.parse(value), {})
        # An explicit null means "use the environment", not "set it to null".
        self.assertEqual(self.config.parse({"pin_frame_every_turn": None}), {})

    def test_the_env_spellings_are_accepted_and_nothing_else_is(self):
        for text, expected in (("1", True), ("true", True), ("on", True),
                               ("0", False), ("false", False), ("off", False)):
            self.assertEqual(self.config.parse({"pin_frame_every_turn": text}),
                             {"pin_frame_every_turn": expected})
        for bad in ("maybe", "", 2, [], {"a": 1}):
            with self.assertRaises(ValueError):
                self.config.parse({"pin_frame_every_turn": bad})

    def test_unknown_and_read_only_keys_are_refused_and_nothing_is_applied(self):
        for body in ({"pin_frame": True}, {"model": "other"}, {"stability_mode": False}):
            with self.assertRaises(ValueError):
                self.config.parse(body)
        with self.assertRaises(ValueError):
            self.config.parse([("pin_frame_every_turn", True)])
        # A refused body is refused whole: the valid key next to it is not kept.
        with self.assertRaises(ValueError):
            self.config.parse({"pin_frame_every_turn": False, "model": "other"})
        self.assertEqual(self.config.overrides(), {})

    def test_observed_reports_every_knob_with_the_override_laid_over_it(self):
        self.config.apply(self.config.parse({"pin_frame_every_turn": False}))
        self.assertEqual(self.config.observed(),
                         {"pin_frame_every_turn": False, "model": "model-x"})
        self.assertEqual(self.config.overrides(), {"pin_frame_every_turn": False})

    def test_conflict_only_fires_on_a_value_that_differs(self):
        self.config.apply(self.config.parse({"pin_frame_every_turn": False}))
        self.assertIsNone(self.config.conflict({"pin_frame_every_turn": False}))
        self.assertIsNone(self.config.conflict({}))
        self.assertIn("pin_frame_every_turn", self.config.conflict({"pin_frame_every_turn": True}))

    def test_the_registry_and_the_names_the_archive_reads_stay_in_step(self):
        # trial_archive reads a meta.json without importing app_main, so it has
        # its own copy of the knob names. Adding a knob means adding it there.
        self.assertEqual(sorted(app_main.session_config.names()),
                         sorted(RECORDED_CONFIG_KEYS))

    def test_null_config_is_inert(self):
        null = NullSessionConfig()
        self.assertEqual(null.parse({"anything": 1}), {})
        self.assertIsNone(null.conflict({"pin_frame_every_turn": False}))
        self.assertEqual(null.observed(), {})


class PerTrialOverrideTests(unittest.IsolatedAsyncioTestCase):
    """The override against a real recorder, the real routes, and a real turn."""

    async def asyncSetUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.old_token = os.environ.pop("TRIAL_TOKEN", None)
        if self.old_token is not None:
            self.addCleanup(os.environ.__setitem__, "TRIAL_TOKEN", self.old_token)

        self.store = LatestFrameStore()
        self.sends = []

        async def send_image(data, *, turn_id=None, sequence=None, source="image"):
            self.sends.append(source)
            return True

        async def send_text(text, *, turn_id=None, source="text"):
            return True

        async def broadcast(_text):
            return None

        # A fresh resolver per test, built exactly as a new process builds it.
        self.patches = [
            mock.patch.object(app_main, "session_config", app_main._build_session_config()),
            mock.patch.object(app_main, "PIN_FRAME_EVERY_TURN", True),
            mock.patch.object(app_main, "latest_rgb", self.store),
            mock.patch.object(app_main.gemini_live, "send_image", send_image),
            mock.patch.object(app_main.gemini_live, "send_text", send_text),
            mock.patch.object(app_main, "ui_broadcast_partial", broadcast),
        ]
        for patch in self.patches:
            patch.start()
        self.addCleanup(lambda: [patch.stop() for patch in reversed(self.patches)])
        self.recorder = await self._recorder()

    async def _recorder(self):
        """A recorder wired to whatever resolver app_main currently holds."""
        recorder = TrialRecorder(
            enabled=True, root=self.root,
            static_fields=app_main._trial_static_fields,
            session_config=app_main.session_config,
        )
        await recorder.start()
        self.addAsyncCleanup(recorder.stop)
        patch = mock.patch.object(app_main, "trial_recorder", recorder)
        patch.start()
        self.addCleanup(patch.stop)
        return recorder

    async def _turn_is_pinned(self):
        """Run one non-phrase question; True if a frame was pinned to the turn."""
        self.sends.clear()
        app_main._clear_gemini_turn_state("test")
        self.store.update(b"\xff\xd8frame")
        await app_main._on_input_transcription("how big is it")
        app_main._clear_gemini_turn_state("test")
        return "pre_response" in self.sends

    async def _start(self, **body):
        body.setdefault("trial", "T20")
        body.setdefault("pid", "P1")
        return await app_main.trial_set(_post("/internal/trial", body))

    # ---------------------------------------------------------------- lifetime

    async def test_no_config_leaves_todays_behaviour_untouched(self):
        response = await self._start(cond="pinned")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(_json(response)["config_overrides"], {})
        self.assertTrue(await self._turn_is_pinned())
        self.assertEqual(app_main.session_config.overrides(), {})

    async def test_an_override_applies_for_the_session_and_is_dropped_at_the_end(self):
        await self._start(cond="unpinned", config={"pin_frame_every_turn": False})
        self.assertFalse(await self._turn_is_pinned())
        self.assertEqual(app_main.session_config.overrides(), {"pin_frame_every_turn": False})

        ended = await app_main.trial_end(_post("/internal/trial/end", {}))
        self.assertEqual(_json(ended)["ended"], "T20_P1")
        self.assertEqual(app_main.session_config.overrides(), {})
        # Back on the deploy default, with nothing of the arm left behind.
        self.assertTrue(await self._turn_is_pinned())

    async def test_the_next_trial_starts_from_the_env_default_again(self):
        await self._start(trial="T21", config={"pin_frame_every_turn": False})
        self.assertFalse(await self._turn_is_pinned())
        # Starting a different trial closes the first one; no config carries over.
        await self._start(trial="T22")
        self.assertEqual(app_main.session_config.overrides(), {})
        self.assertTrue(await self._turn_is_pinned())

    async def test_an_untitled_session_after_an_arm_runs_on_the_env_default(self):
        await self._start(config={"pin_frame_every_turn": False})
        app_main.trial_recorder.end_session(reason="internal_end")
        app_main.trial_recorder.speech_start(1, 1, None, None)   # opens untitled
        self.assertTrue(app_main.trial_recorder.status()["untitled"])
        self.assertTrue(await self._turn_is_pinned())

    async def test_a_bad_config_is_a_400_and_opens_nothing(self):
        for config in ({"pin_frame_every_turn": "maybe"}, {"stability_mode": False},
                       {"nonsense": 1}, "pinned"):
            response = await self._start(config=config)
            self.assertEqual(response.status_code, 400, config)
        self.assertIsNone(app_main.trial_recorder.status()["session_id"])
        self.assertEqual(app_main.session_config.overrides(), {})

    async def test_the_arm_cannot_be_swapped_under_an_open_trial(self):
        await self._start(config={"pin_frame_every_turn": False})
        clash = await self._start(config={"pin_frame_every_turn": True})
        self.assertEqual(clash.status_code, 400)
        self.assertIn("end the trial", _json(clash)["error"])
        # Refused, so the run keeps the arm it opened with.
        self.assertFalse(await self._turn_is_pinned())
        # Reposting the same arm is an ordinary resume.
        again = await self._start(config={"pin_frame_every_turn": False})
        self.assertEqual(again.status_code, 200)

    # ------------------------------------------------------------- the record

    async def test_meta_json_carries_the_config_that_was_really_in_force(self):
        await self._start(cond="unpinned", config={"pin_frame_every_turn": False})
        await self.recorder.flush()
        meta = _meta(self.root, "T20_P1")
        self.assertEqual(meta["pin_frame_every_turn"], False)
        self.assertEqual(meta["gemini_video_interval_sec"], app_main.GEMINI_VIDEO_INTERVAL_SEC)
        self.assertEqual(meta["vision_frame_max_age_sec"], app_main.VISION_FRAME_MAX_AGE_SEC)
        self.assertEqual(meta["stability_mode"], app_main.STABILITY_MODE)
        self.assertEqual(meta["model"], app_main._gemini_live_model_id())
        self.assertEqual(meta["config_overridden"], "pin_frame_every_turn")
        self.assertEqual(sorted(recorded_config(meta)), sorted(RECORDED_CONFIG_KEYS))

    async def test_meta_json_records_the_env_default_when_no_config_is_sent(self):
        await self._start()
        await self.recorder.flush()
        meta = _meta(self.root, "T20_P1")
        self.assertEqual(meta["pin_frame_every_turn"], True)
        self.assertIsNone(meta["config_overridden"])

    async def test_the_session_start_event_says_the_same_as_meta_json(self):
        await self._start(cond="unpinned", config={"pin_frame_every_turn": False})
        await self.recorder.flush()
        meta = _meta(self.root, "T20_P1")
        start = [event for event in _events(self.root, "T20_P1")
                 if event["t"] == "session_start"][0]
        for key in RECORDED_CONFIG_KEYS:
            self.assertEqual(start[key], meta[key], key)

    async def test_the_live_status_reports_what_is_in_force(self):
        await self._start(config={"pin_frame_every_turn": False})
        status = _json(await app_main.trial_status())
        self.assertFalse(status["config"]["pin_frame_every_turn"])
        self.assertEqual(status["config_overrides"], {"pin_frame_every_turn": False})

    # ----------------------------------------------------- restart / staleness

    async def test_a_restart_drops_the_override_and_the_log_says_so(self):
        """The guarantee: an override lives in the resolver and nowhere else.

        A restart builds a new resolver, so the resumed session runs on the env
        default. meta.json keeps what the trial opened with, and session_resume
        records what the restarted process is actually doing, so the two can be
        compared instead of the drift going unnoticed.
        """
        await self._start(cond="unpinned", config={"pin_frame_every_turn": False})
        await self.recorder.flush()
        await self.recorder.stop()

        # Nothing on disk asks for the override to be reinstated.
        with open(os.path.join(self.root, "current.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh), {"session_id": "T20_P1"})

        # Restart: a new process, a new resolver, the same trial folder.
        for patch in (mock.patch.object(app_main, "session_config",
                                        app_main._build_session_config()),):
            patch.start()
            self.addCleanup(patch.stop)
        restarted = await self._recorder()
        self.assertEqual(restarted.status()["session_id"], "T20_P1")
        self.assertEqual(app_main.session_config.overrides(), {})
        self.assertTrue(await self._turn_is_pinned())

        await restarted.flush()
        meta = _meta(self.root, "T20_P1")
        self.assertFalse(meta["pin_frame_every_turn"])       # written once, at open
        resume = [event for event in _events(self.root, "T20_P1")
                  if event["t"] == "session_resume"][-1]
        self.assertTrue(resume["pin_frame_every_turn"])      # what is really running
        drift = resume_drift(meta, _events(self.root, "T20_P1"))
        self.assertIn("pin_frame_every_turn", drift)


class RecorderDisabledTests(unittest.IsolatedAsyncioTestCase):
    """TRIAL_RECORDER unset: nothing new happens, including nothing new to the flag."""

    async def test_a_config_on_a_disabled_recorder_changes_nothing(self):
        config = app_main._build_session_config()
        disabled = TrialRecorder(enabled=False, root="/nonexistent", session_config=config)
        token = os.environ.pop("TRIAL_TOKEN", None)
        try:
            with mock.patch.object(app_main, "session_config", config), \
                    mock.patch.object(app_main, "trial_recorder", disabled):
                response = await app_main.trial_set(_post("/internal/trial", {
                    "trial": "T30", "pid": "P1", "config": {"pin_frame_every_turn": False},
                }))
                self.assertEqual(response.status_code, 409)
                self.assertEqual(config.overrides(), {})
                self.assertEqual(config.get("pin_frame_every_turn"),
                                 app_main.PIN_FRAME_EVERY_TURN)
        finally:
            if token is not None:
                os.environ["TRIAL_TOKEN"] = token


class ArchiveSurfacesTheConfigTests(unittest.TestCase):
    """What /api/trials hands the viewer for a mislabelled trial, and for an old one."""

    def _trial(self, directory, **meta):
        os.makedirs(directory)
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"trial": "T40", "pid": "P1", "created_ts": "2026-10-25T10:00:00.000Z",
                       **meta}, fh)
        open(os.path.join(directory, "session.jsonl"), "w", encoding="utf-8").close()

    def setUp(self):
        from trial_archive import TrialArchive
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self._trial(os.path.join(self.root, "T40_P1"), cond="unpinned arm B",
                    pin_frame_every_turn=True, gemini_video_interval_sec=0.75,
                    vision_frame_max_age_sec=3.0, stability_mode=True, model="m",
                    config_overridden=None)
        self._trial(os.path.join(self.root, "T41_P1"), cond="unpinned")   # before the stamp
        self.archive = TrialArchive(self.root, None)

    def test_the_detail_payload_carries_the_config_and_the_warning(self):
        detail = self.archive.get_trial("T40_P1")
        self.assertEqual(detail["config"]["pin_frame_every_turn"], True)
        self.assertEqual(sorted(detail["config"]), sorted(RECORDED_CONFIG_KEYS))
        self.assertIn("unpinned arm B", detail["condition_mismatch"])
        self.assertIsNone(detail["config_drift"])

    def test_the_list_row_flags_the_same_trial(self):
        rows = {row["id"]: row for row in self.archive.list_trials()}
        self.assertIn("pin_frame_every_turn", rows["T40_P1"]["condition_mismatch"])

    def test_a_trial_from_before_the_stamp_shows_nothing_and_warns_nothing(self):
        detail = self.archive.get_trial("T41_P1")
        self.assertEqual(detail["config"], {})
        self.assertIsNone(detail["condition_mismatch"])
        self.assertIsNone(detail["config_drift"])


class MislabelledTrialTests(unittest.TestCase):
    """cond is typed by hand; pin_frame_every_turn is what ran. They can disagree."""

    def test_a_label_that_contradicts_the_flag_is_reported(self):
        self.assertIn("unpinned", condition_mismatch("unpinned", True))
        self.assertIn("pinned", condition_mismatch("pinned-thermal-on", False))
        self.assertIn("Un-Pinned", condition_mismatch("Un-Pinned", True))   # case kept as typed

    def test_an_agreeing_label_is_silent(self):
        self.assertIsNone(condition_mismatch("pinned", True))
        self.assertIsNone(condition_mismatch("unpinned", False))
        self.assertIsNone(condition_mismatch("UNPINNED arm B", False))

    def test_a_label_making_no_claim_is_not_a_mismatch(self):
        for cond in ("thermal-on", "", None, "arm B", 7):
            self.assertIsNone(condition_mismatch(cond, True), cond)

    def test_a_trial_recorded_before_the_flag_was_stamped_is_not_a_mismatch(self):
        self.assertIsNone(condition_mismatch("unpinned", None))
        self.assertEqual(recorded_config({"trial": "T01", "cond": "unpinned"}), {})

    def test_resume_drift_is_silent_when_the_resume_agrees(self):
        meta = {"pin_frame_every_turn": False, "model": "m"}
        events = [{"t": "session_resume", "pin_frame_every_turn": False, "model": "m"},
                  {"t": "speech_start"}]
        self.assertIsNone(resume_drift(meta, events))
        self.assertIsNone(resume_drift({}, events))


if __name__ == "__main__":
    unittest.main()
