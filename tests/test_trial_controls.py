"""Trial control plane: judge rate-limit handling, resumable scoring, and the
cookie-or-header auth the /trials control block relies on.

The scoring semantics (labels, question windows, the distance detector) are a
research instrument and are deliberately not re-tested here; these cover only
the new failure handling around them.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from starlette.requests import Request

import app_main
from trial_scorer import (
    JudgeRetryableError,
    RateLimiter,
    is_retryable_judge_error,
    load_existing_scores,
    retry_delay_seconds,
    score_trial_folder,
)


def _event(event_type, turn=None, **fields):
    value = {"t": event_type, "ts": "2026-10-25T12:00:00.000Z", "ns": 1, "src": "backend", "turn": turn}
    value.update(fields)
    return value


class FakeApiError(Exception):
    """Shaped like a google-genai API error: code, status and RetryInfo."""

    def __init__(self, code, status, retry_delay=None):
        self.code = code
        self.status = status
        self.details = {}
        if retry_delay is not None:
            self.details = {"error": {"code": code, "status": status, "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo",
                 "retryDelay": f"{retry_delay}s"},
            ]}}
        super().__init__(f"{code} {status}")


class ScriptedJudge:
    """Raises/returns per call from a script; records every call."""

    model = "test-judge"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def judge(self, question, answer, frame_paths, ground_truth, thermal):
        self.calls.append(question)
        outcome = self.script.pop(0) if self.script else self.script
        if isinstance(outcome, BaseException):
            raise outcome
        return dict(outcome)


class ByQuestionJudge:
    """Answers or fails depending on the question text."""

    model = "test-judge"

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def judge(self, question, answer, frame_paths, ground_truth, thermal):
        self.calls.append(question)
        outcome = self.outcomes[question]
        if isinstance(outcome, BaseException):
            raise outcome
        return dict(outcome)


GOOD = {"label": "useful_refusal", "spatial_frame": "room-relative",
        "why": "Admits the target is unseen and describes the visible wall."}
RIGHT = {"label": "right", "spatial_frame": "body-relative", "why": "The table is visible ahead."}


class JudgeErrorClassificationTests(unittest.TestCase):
    def test_429_and_503_are_retryable_and_other_errors_are_not(self):
        self.assertTrue(is_retryable_judge_error(FakeApiError(429, "RESOURCE_EXHAUSTED")))
        self.assertTrue(is_retryable_judge_error(FakeApiError(503, "UNAVAILABLE")))
        self.assertTrue(is_retryable_judge_error(Exception("429 RESOURCE_EXHAUSTED quota")))
        self.assertTrue(is_retryable_judge_error(Exception("503 UNAVAILABLE backend")))
        self.assertFalse(is_retryable_judge_error(FakeApiError(400, "INVALID_ARGUMENT")))
        self.assertFalse(is_retryable_judge_error(ValueError("judge returned a non-object")))

    def test_retry_delay_comes_from_retry_info_then_from_the_message(self):
        self.assertEqual(retry_delay_seconds(FakeApiError(429, "RESOURCE_EXHAUSTED", 7)), 7.0)
        self.assertEqual(
            retry_delay_seconds(Exception("429 RESOURCE_EXHAUSTED {'retryDelay': '12s'}")), 12.0)
        self.assertIsNone(retry_delay_seconds(FakeApiError(429, "RESOURCE_EXHAUSTED")))
        self.assertEqual(retry_delay_seconds(JudgeRetryableError("x", 3.5)), 3.5)


class RateLimiterTests(unittest.TestCase):
    def test_paces_calls_to_the_configured_ceiling(self):
        now = [0.0]
        slept = []

        def sleep(seconds):
            slept.append(seconds)
            now[0] += seconds

        limiter = RateLimiter(rpm=5, sleep=sleep, monotonic=lambda: now[0])
        for _ in range(3):
            limiter.acquire()
        # 5 rpm is one call every 12s: the first is free, then 12s each.
        self.assertEqual([round(value, 3) for value in slept], [12.0, 12.0])

    def test_zero_rpm_disables_pacing(self):
        slept = []
        limiter = RateLimiter(rpm=0, sleep=slept.append, monotonic=lambda: 0.0)
        limiter.acquire()
        limiter.acquire()
        self.assertEqual(slept, [])


class ScorerRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.trial = Path(self.temp.name) / "T08_PSIM"
        (self.trial / "frames").mkdir(parents=True)
        self.slept = []

    def tearDown(self):
        self.temp.cleanup()

    def _sleep(self, seconds):
        self.slept.append(seconds)

    def _limiter(self):
        return RateLimiter(rpm=0, sleep=self._sleep, monotonic=lambda: 0.0)

    def _write(self, events, frames=("5.jpg",)):
        for name in frames:
            (self.trial / "frames" / name).write_bytes(b"jpeg")
        (self.trial / "ground_truth.json").write_text(json.dumps({
            "room": "VISIONS room 2", "present": ["table"], "absent": ["plant"],
            "planted_unseeable": [], "visible_from_start": ["table"],
        }), encoding="utf-8")
        (self.trial / "meta.json").write_text(
            json.dumps({"trial": "T08", "pid": "PSIM"}), encoding="utf-8")
        (self.trial / "session.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    def _one_window(self):
        return [
            _event("speech_start", "T08.PSIM.1", n=1),
            _event("frame_sent", "T08.PSIM.1", seq=5, file="frames/5.jpg"),
            _event("response_text", "T08.PSIM.1", kind="input_transcription",
                   text="Can you see the clock?"),
            _event("response_text", "T08.PSIM.1", kind="output_transcription",
                   text="I cannot see a clock; the wall ahead is clear."),
            _event("turn_end", "T08.PSIM.1", status="completed"),
            _event("session_end"),
        ]

    def _two_windows(self):
        return [
            _event("speech_start", "T08.PSIM.1", n=1),
            _event("frame_sent", "T08.PSIM.1", seq=5, file="frames/5.jpg"),
            _event("response_text", "T08.PSIM.1", kind="input_transcription",
                   text="Can you see the clock?"),
            _event("response_text", "T08.PSIM.1", kind="output_transcription",
                   text="I cannot see a clock; the wall ahead is clear."),
            _event("turn_end", "T08.PSIM.1", status="completed"),
            _event("speech_start", "T08.PSIM.2", n=2),
            _event("frame_sent", "T08.PSIM.2", seq=6, file="frames/6.jpg"),
            _event("response_text", "T08.PSIM.2", kind="input_transcription",
                   text="What is in front of me?"),
            _event("response_text", "T08.PSIM.2", kind="output_transcription",
                   text="A table is directly ahead of you."),
            _event("turn_end", "T08.PSIM.2", status="completed"),
            _event("session_end"),
        ]

    def test_429_with_retry_delay_is_honoured_then_the_window_scores(self):
        self._write(self._one_window())
        judge = ScriptedJudge([FakeApiError(429, "RESOURCE_EXHAUSTED", 7), GOOD])
        result = score_trial_folder(
            self.trial, judge=judge, limiter=self._limiter(), sleep=self._sleep)
        self.assertEqual(len(judge.calls), 2)
        self.assertEqual(self.slept, [7.0])           # the server's own retryDelay, not a guess
        self.assertEqual(result["summary"]["counts"]["useful_refusal"], 1)
        self.assertEqual(result["unscored_windows"], [])

    def test_503_unavailable_is_retried(self):
        self._write(self._one_window())
        judge = ScriptedJudge([FakeApiError(503, "UNAVAILABLE"), GOOD])
        result = score_trial_folder(
            self.trial, judge=judge, limiter=self._limiter(), sleep=self._sleep)
        self.assertEqual(len(judge.calls), 2)
        self.assertEqual(len(self.slept), 1)          # no retryDelay given: backed off
        self.assertGreaterEqual(self.slept[0], 0.0)
        self.assertEqual(result["summary"]["counts"]["useful_refusal"], 1)

    def test_attempts_are_capped_and_the_window_is_recorded_as_unscored(self):
        self._write(self._one_window())
        judge = ScriptedJudge([FakeApiError(429, "RESOURCE_EXHAUSTED", 1)] * 10)
        result = score_trial_folder(
            self.trial, judge=judge, limiter=self._limiter(), max_attempts=3, sleep=self._sleep)
        self.assertEqual(len(judge.calls), 3)
        self.assertEqual(result["scored_windows"], 0)
        self.assertEqual(len(result["unscored_windows"]), 1)
        self.assertIn("RESOURCE_EXHAUSTED", result["unscored_windows"][0]["reason"])
        summary = json.loads((self.trial / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["unscored_count"], 1)
        self.assertEqual(summary["window_count"], 1)

    def test_one_permanent_failure_does_not_stop_the_other_windows(self):
        self._write(self._two_windows(), frames=("5.jpg", "6.jpg"))
        judge = ByQuestionJudge({
            "Can you see the clock?": FakeApiError(400, "INVALID_ARGUMENT"),
            "What is in front of me?": RIGHT,
        })
        result = score_trial_folder(
            self.trial, judge=judge, limiter=self._limiter(), max_attempts=2, sleep=self._sleep)
        self.assertEqual(result["scored_windows"], 1)
        self.assertEqual([item["window"] for item in result["unscored_windows"]], [1])
        self.assertIn("INVALID_ARGUMENT", result["unscored_windows"][0]["reason"])
        # A 400 is permanent: it is not retried, and it did not cost a sleep.
        self.assertEqual(judge.calls, ["Can you see the clock?", "What is in front of me?"])
        self.assertEqual(self.slept, [])
        self.assertIn("after 1 attempt(s)", result["unscored_windows"][0]["reason"])
        # The surviving window is fully written, not discarded with the failure.
        scores = [json.loads(line) for line
                  in (self.trial / "scores.jsonl").read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual([score["window"] for score in scores], [2])
        self.assertEqual(scores[0]["label"], "right")
        summary = json.loads((self.trial / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["counts"]["right"], 1)
        self.assertEqual(summary["window_count"], 2)
        self.assertEqual(summary["unscored"][0]["turn_uid"], "T08.PSIM.1")

    def test_rescoring_a_scored_trial_makes_no_judge_calls(self):
        self._write(self._two_windows(), frames=("5.jpg", "6.jpg"))
        outcomes = {"Can you see the clock?": GOOD, "What is in front of me?": RIGHT}
        first = score_trial_folder(
            self.trial, judge=ByQuestionJudge(dict(outcomes)),
            limiter=self._limiter(), sleep=self._sleep)
        self.assertEqual(first["scored_windows"], 2)
        self.assertEqual(first["reused_windows"], 0)

        resumed = ByQuestionJudge(dict(outcomes))
        second = score_trial_folder(
            self.trial, judge=resumed, limiter=self._limiter(), sleep=self._sleep)
        self.assertEqual(resumed.calls, [])           # the whole point: no quota burned
        self.assertEqual(second["reused_windows"], 2)
        self.assertEqual(second["summary"]["counts"], first["summary"]["counts"])

    def test_resume_rejudges_only_the_window_that_is_still_missing(self):
        self._write(self._two_windows(), frames=("5.jpg", "6.jpg"))
        exhausted = ByQuestionJudge({
            "Can you see the clock?": GOOD,
            "What is in front of me?": FakeApiError(429, "RESOURCE_EXHAUSTED", 1),
        })
        first = score_trial_folder(
            self.trial, judge=exhausted, limiter=self._limiter(), max_attempts=2, sleep=self._sleep)
        self.assertEqual(first["scored_windows"], 1)
        self.assertEqual(len(first["unscored_windows"]), 1)

        recovered = ByQuestionJudge({
            "Can you see the clock?": GOOD, "What is in front of me?": RIGHT})
        second = score_trial_folder(
            self.trial, judge=recovered, limiter=self._limiter(), sleep=self._sleep)
        self.assertEqual(recovered.calls, ["What is in front of me?"])
        self.assertEqual(second["scored_windows"], 2)
        self.assertEqual(second["reused_windows"], 1)
        self.assertEqual(second["unscored_windows"], [])
        self.assertNotIn("unscored", second["summary"])

    def test_a_changed_answer_is_rejudged_rather_than_reused(self):
        self._write(self._one_window())
        score_trial_folder(self.trial, judge=ScriptedJudge([GOOD]),
                           limiter=self._limiter(), sleep=self._sleep)
        self.assertEqual(len(load_existing_scores(self.trial)), 1)

        events = self._one_window()
        events[3]["text"] = "The clock is on the wall to your right."
        self._write(events)
        judge = ScriptedJudge([RIGHT])
        score_trial_folder(self.trial, judge=judge, limiter=self._limiter(), sleep=self._sleep)
        self.assertEqual(len(judge.calls), 1)

    def test_resume_false_rejudges_everything(self):
        self._write(self._one_window())
        score_trial_folder(self.trial, judge=ScriptedJudge([GOOD]),
                           limiter=self._limiter(), sleep=self._sleep)
        judge = ScriptedJudge([GOOD])
        score_trial_folder(self.trial, judge=judge, resume=False,
                           limiter=self._limiter(), sleep=self._sleep)
        self.assertEqual(len(judge.calls), 1)


def _request(path, method="POST", token=None, cookie=None, query="", body=b""):
    headers = []
    if token is not None:
        headers.append((b"x-trial-token", token.encode("utf-8")))
    if cookie is not None:
        headers.append((b"cookie", f"trial_token={cookie}".encode("utf-8")))
    scope = {
        "type": "http", "http_version": "1.1", "method": method, "scheme": "https",
        "path": path, "raw_path": path.encode("utf-8"), "query_string": query.encode("utf-8"),
        "headers": headers, "client": ("test", 1), "server": ("test", 443),
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


class ControlAuthTests(unittest.IsolatedAsyncioTestCase):
    """The control endpoints must take the page's cookie as well as the header,
    and must still refuse a request carrying neither."""

    async def asyncSetUp(self):
        self.old_token = os.environ.get("TRIAL_TOKEN")
        os.environ["TRIAL_TOKEN"] = "control-secret"

    async def asyncTearDown(self):
        if self.old_token is None:
            os.environ.pop("TRIAL_TOKEN", None)
        else:
            os.environ["TRIAL_TOKEN"] = self.old_token

    async def test_no_token_and_no_cookie_is_rejected(self):
        responses = [
            await app_main.trial_set(_request("/internal/trial", body=b"{}")),
            await app_main.trial_end(_request("/internal/trial/end")),
            await app_main.trial_score("T08_PSIM", _request("/internal/trial/T08_PSIM/score")),
            await app_main.trial_upload("T08_PSIM", _request("/internal/trial/T08_PSIM/upload")),
        ]
        self.assertEqual([response.status_code for response in responses], [401] * 4)

    async def test_a_wrong_cookie_is_rejected(self):
        response = await app_main.trial_end(_request("/internal/trial/end", cookie="guessed"))
        self.assertEqual(response.status_code, 401)

    async def test_an_empty_cookie_is_rejected(self):
        response = await app_main.trial_end(_request("/internal/trial/end", cookie=""))
        self.assertEqual(response.status_code, 401)

    async def test_header_and_cookie_both_authorise(self):
        # The recorder is off in tests, so passing auth lands on 409 rather than
        # 401 — which is exactly what distinguishes "authorised" from "refused".
        by_header = await app_main.trial_end(
            _request("/internal/trial/end", token="control-secret"))
        by_cookie = await app_main.trial_end(
            _request("/internal/trial/end", cookie="control-secret"))
        for response in (by_header, by_cookie):
            self.assertNotEqual(response.status_code, 401)
            self.assertEqual(response.status_code, 409)

    async def test_auth_helper_accepts_either_carrier(self):
        self.assertIsNone(app_main._trial_auth_failed(
            _request("/internal/trial", token="control-secret")))
        self.assertIsNone(app_main._trial_auth_failed(
            _request("/internal/trial", cookie="control-secret")))
        self.assertIsNotNone(app_main._trial_auth_failed(_request("/internal/trial")))


class JobRecordTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.saved = dict(app_main._trial_jobs)
        app_main._trial_jobs.clear()

    async def asyncTearDown(self):
        app_main._trial_jobs.clear()
        app_main._trial_jobs.update(self.saved)

    async def test_job_records_track_state_attempts_error_and_retry_time(self):
        app_main._trial_job_set("T08_PSIM", "queued")
        self.assertEqual(app_main._trial_job("T08_PSIM")["state"], "queued")
        app_main._trial_job_set("T08_PSIM", "scoring", bump_attempt=True)
        app_main._trial_job_set("T08_PSIM", "queued", error="429 RESOURCE_EXHAUSTED",
                                next_retry_in=60)
        job = app_main._trial_job("T08_PSIM")
        self.assertEqual(job["attempts"], 1)
        self.assertEqual(job["last_error"], "429 RESOURCE_EXHAUSTED")
        self.assertIsNotNone(job["next_retry_ts"])
        app_main._trial_job_set("T08_PSIM", "scored")
        self.assertEqual(app_main._trial_job("T08_PSIM")["state"], "scored")
        self.assertIsNone(app_main._trial_job("T08_PSIM")["next_retry_ts"])

    async def test_row_status_prefers_recording_then_job_then_scored(self):
        self.assertEqual(app_main._trial_row_status("T08_PSIM", False, "T08_PSIM"), "recording")
        app_main._trial_job_set("T08_PSIM", "scoring")
        self.assertEqual(app_main._trial_row_status("T08_PSIM", False, None), "scoring")
        app_main._trial_job_set("T08_PSIM", "failed", error="boom")
        self.assertEqual(app_main._trial_row_status("T08_PSIM", False, None), "failed")
        app_main._trial_job_set("T08_PSIM", "scored")
        self.assertEqual(app_main._trial_row_status("T08_PSIM", True, None), "scored")
        self.assertEqual(app_main._trial_row_status("T99_PXX", False, None), "unscored")


if __name__ == "__main__":
    unittest.main()
