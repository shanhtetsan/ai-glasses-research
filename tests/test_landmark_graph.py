import asyncio
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from starlette.requests import Request

import landmark_graph
from landmark_graph import GraphError


def _graph():
    return {"landmarks": [
        {"id": "front-door", "name": "front door", "kind": "door", "from": "start",
         "bearing_deg": 90.0, "steps": 6, "note": "push to open"},
        {"id": "stairs", "name": "stairs", "kind": "stairs", "from": "front-door",
         "bearing_deg": 350.0, "steps": None},
    ]}


class GraphFileTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def test_save_load_list_round_trip(self):
        stored = landmark_graph.save("home", _graph(), self.root)
        loaded = landmark_graph.load("home", self.root)
        self.assertEqual(loaded["landmarks"], stored["landmarks"])
        self.assertEqual(loaded["sha256"], stored["sha256"])
        self.assertEqual(landmark_graph.list_sites(self.root)[0]["site"], "home")
        self.assertEqual(landmark_graph.list_sites(self.root)[0]["landmarks"], 2)
        self.assertEqual([n for n in os.listdir(self.root)], ["home.json"])   # no temp left

    def test_sha_ignores_save_time_but_not_content(self):
        first = landmark_graph.save("home", _graph(), self.root)
        time.sleep(0.01)
        second = landmark_graph.save("home", _graph(), self.root)
        self.assertEqual(first["sha256"], second["sha256"])
        changed = _graph()
        changed["landmarks"][0]["bearing_deg"] = 91.0
        self.assertNotEqual(landmark_graph.save("home", changed, self.root)["sha256"], first["sha256"])

    def test_rejects_bad_graphs(self):
        bad = [
            ("Home", _graph()),
            ("home", {"landmarks": []}),
            ("home", {"landmarks": [{**_graph()["landmarks"][0], "bearing_deg": 360}]}),
            ("home", {"landmarks": [{**_graph()["landmarks"][0], "bearing_deg": True}]}),
            ("home", {"landmarks": [{**_graph()["landmarks"][0], "kind": "window"}]}),
            ("home", {"landmarks": [{**_graph()["landmarks"][0], "from": "stairs"}]}),
            ("home", {"landmarks": [_graph()["landmarks"][0], _graph()["landmarks"][0]]}),
            ("home", {"landmarks": [{**_graph()["landmarks"][0], "id": "start"}]}),
            ("home", {"landmarks": [{**_graph()["landmarks"][0], "steps": 1.5}]}),
            ("home", {"landmarks": [{**_graph()["landmarks"][0], "name": " "}]}),
        ]
        for site, graph in bad:
            with self.assertRaises(GraphError, msg=json.dumps(graph)[:80]):
                landmark_graph.save(site, graph, self.root)

    def test_missing_graph_is_a_clear_error(self):
        with self.assertRaisesRegex(GraphError, "no graph named"):
            landmark_graph.load("nowhere", self.root)

    def test_parse_site_for_the_trial_arm(self):
        landmark_graph.save("home", _graph(), self.root)
        with mock.patch.object(landmark_graph, "GRAPH_ROOT", self.root):
            self.assertEqual(landmark_graph.parse_site(" Home "), "home")
            self.assertIsNone(landmark_graph.parse_site(""))
            self.assertIsNone(landmark_graph.parse_site(None))
            with self.assertRaises(ValueError):
                landmark_graph.parse_site("nowhere")
            with self.assertRaises(ValueError):
                landmark_graph.parse_site(3)


class ContextTextTests(unittest.TestCase):
    def test_relative_direction(self):
        self.assertEqual(landmark_graph.relative_direction(90, 90), "straight ahead (12 o'clock)")
        self.assertEqual(landmark_graph.relative_direction(90, 0), "to the right (3 o'clock)")
        self.assertEqual(landmark_graph.relative_direction(0, 90), "to the left (9 o'clock)")
        self.assertEqual(landmark_graph.relative_direction(10, 350), "straight ahead (1 o'clock)")
        self.assertEqual(landmark_graph.relative_direction(180, 0), "behind (6 o'clock)")

    def test_text_rotates_to_heading_and_never_carries_step_counts(self):
        graph = landmark_graph.validate("home", _graph())
        text = landmark_graph.context_text(graph, 0.0)
        self.assertIn("facing compass 0°", text)
        self.assertIn("front door (door): from the start spot, compass 90°", text)
        self.assertIn("to the right (3 o'clock)", text)
        self.assertIn("from the front door, compass 350°", text)
        self.assertIn("Note: push to open", text)
        self.assertIn("trust the camera", text)
        self.assertNotIn(" 6 ", text)
        # Nothing in the map text itself trips the scorer's distance rule.
        from trial_scorer import detect_distance_units
        for line in text.splitlines():
            if not line.startswith("Never state"):
                self.assertIsNone(detect_distance_units(line), line)

    def test_unknown_heading_gives_no_left_right(self):
        graph = landmark_graph.validate("home", _graph())
        text = landmark_graph.context_text(graph, None)
        self.assertIn("facing direction is unknown", text)
        self.assertNotIn("o'clock", text)


def _request(method, path, token="viewer-secret", body=None):
    headers = [(b"x-trial-token", token.encode())] if token else []
    raw = json.dumps(body).encode() if body is not None else b""
    headers.append((b"content-type", b"application/json"))
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": raw, "more_body": False}

    return Request({
        "type": "http", "http_version": "1.1", "method": method, "scheme": "https",
        "path": path, "raw_path": path.encode(), "query_string": b"", "headers": headers,
        "client": ("test", 1), "server": ("test", 443),
    }, receive)


class GraphRouteAndArmTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import app_main
        self.app = app_main
        self.root = tempfile.mkdtemp()
        self.patch_root = mock.patch.object(landmark_graph, "GRAPH_ROOT", self.root)
        self.patch_root.start()
        self.old_token = os.environ.get("TRIAL_TOKEN")
        os.environ["TRIAL_TOKEN"] = "viewer-secret"

    async def asyncTearDown(self):
        self.patch_root.stop()
        self.app.session_config.clear()
        if self.old_token is None:
            os.environ.pop("TRIAL_TOKEN", None)
        else:
            os.environ["TRIAL_TOKEN"] = self.old_token

    async def test_routes_need_token_and_round_trip(self):
        app = self.app
        denied = await app.graph_put("home", _request("PUT", "/api/graphs/home", None, _graph()))
        self.assertEqual(denied.status_code, 401)
        saved = await app.graph_put("home", _request("PUT", "/api/graphs/home", body=_graph()))
        self.assertEqual(saved.status_code, 200)
        listed = json.loads((await app.graph_list(_request("GET", "/api/graphs"))).body)
        self.assertEqual([g["site"] for g in listed["graphs"]], ["home"])
        got = json.loads((await app.graph_get("home", _request("GET", "/api/graphs/home"))).body)
        self.assertEqual(len(got["landmarks"]), 2)
        missing = await app.graph_get("nowhere", _request("GET", "/api/graphs/nowhere"))
        self.assertEqual(missing.status_code, 404)
        bad = await app.graph_put("home", _request("PUT", "/api/graphs/home", body={"landmarks": []}))
        self.assertEqual(bad.status_code, 422)

    async def test_arm_is_a_settable_recorded_knob(self):
        from trial_config import RECORDED_CONFIG_KEYS
        app = self.app
        self.assertIn("landmark_graph", app.session_config.settable_names())
        self.assertIn("landmark_graph", RECORDED_CONFIG_KEYS)
        with self.assertRaisesRegex(ValueError, "no graph named"):
            app.session_config.parse({"landmark_graph": "home"})
        landmark_graph.save("home", _graph(), self.root)
        self.assertEqual(app.session_config.parse({"landmark_graph": "home"}), {"landmark_graph": "home"})

    async def test_send_landmark_graph_sends_rotated_text_and_records_it(self):
        app = self.app
        landmark_graph.save("home", _graph(), self.root)
        sent_texts = []

        async def fake_send_text(text, *, turn_id=None, source="text"):
            sent_texts.append((text, turn_id, source))
            return True

        recorded = []
        with mock.patch.object(app.gemini_live, "send_text", fake_send_text), \
                mock.patch.object(app.trial_recorder, "graph_context",
                                  lambda *args, **kw: recorded.append((args, kw))):
            app.latest_heading.update({"timestamp": time.monotonic(),
                                       "data": {"sequence": 1, "heading_deg": 0.0, "accuracy_deg": None}})
            await app._send_landmark_graph(7, "home")
            app.latest_heading.update({"timestamp": time.monotonic() - 10,
                                       "data": {"sequence": 2, "heading_deg": 0.0, "accuracy_deg": None}})
            await app._send_landmark_graph(8, "home")
            await app._send_landmark_graph(9, "nowhere")
        app.latest_heading.update({"timestamp": 0.0, "data": None})

        self.assertEqual([(t[1], t[2]) for t in sent_texts], [(7, "landmark_graph"), (8, "landmark_graph")])
        self.assertIn("3 o'clock", sent_texts[0][0])
        self.assertIn("facing direction is unknown", sent_texts[1][0])   # stale heading not used
        self.assertEqual(recorded[0][0][3], 0.0)
        self.assertIsNone(recorded[1][0][3])
        self.assertIsNone(recorded[2][0][2])                             # missing map: no sha
        self.assertIn("no graph named", recorded[2][1]["error"])

    async def test_map_goes_once_per_turn_from_the_transcription_path(self):
        app = self.app
        landmark_graph.save("home", _graph(), self.root)
        app.session_config.apply({"pin_frame_every_turn": False, "landmark_graph": "home"})
        calls = []

        async def fake_send(turn_id, site):
            calls.append((turn_id, site))

        app._clear_gemini_turn_state("test")
        with mock.patch.object(app, "_send_landmark_graph", fake_send):
            await app._on_input_transcription("where is")
            await app._on_input_transcription(" the door")      # same turn: not again
            app._clear_gemini_turn_state("test")
            await app._on_input_transcription("and the stairs")  # next turn: again
            app.session_config.apply({"pin_frame_every_turn": False})
            app._clear_gemini_turn_state("test")
            await app._on_input_transcription("no map arm now")
        app._clear_gemini_turn_state("test")
        self.assertEqual([site for _, site in calls], ["home", "home"])

    async def test_map_in_use_by_open_trial_cannot_be_overwritten(self):
        app = self.app
        landmark_graph.save("home", _graph(), self.root)
        app.session_config.apply({"landmark_graph": "home"})
        with mock.patch.object(app.trial_recorder, "status", lambda: {"session_id": "T1_evis"}):
            refused = await app.graph_put("home", _request("PUT", "/api/graphs/home", body=_graph()))
        self.assertEqual(refused.status_code, 409)


class RecorderGraphEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_context_event_is_written_under_the_turn(self):
        from trial_recorder import TrialRecorder
        root = tempfile.mkdtemp()
        rec = TrialRecorder(enabled=True, root=root,
                            static_fields=lambda: {"model": "m", "prompt_sha256": "a", "git_sha": "g",
                                                   "prompt_version_default": "sha-a"})
        await rec.start()
        await rec.set_trial({"trial": "G01", "pid": "evis"}, None)
        rec.speech_start(1, 1, None, None)
        rec.graph_context(1, "home", "abc", 12.34, "LANDMARK_MAP ...", True)
        rec.end_session("internal_end")
        await rec.stop()
        with open(os.path.join(root, "G01_evis", "session.jsonl"), encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle]
        graph = [e for e in events if e["t"] == "graph_context"]
        self.assertEqual(len(graph), 1)
        self.assertEqual(graph[0]["turn"], "G01.evis.1")
        self.assertEqual((graph[0]["site"], graph[0]["sha256"], graph[0]["heading_deg"], graph[0]["sent"]),
                         ("home", "abc", 12.3, True))


if __name__ == "__main__":
    unittest.main()
