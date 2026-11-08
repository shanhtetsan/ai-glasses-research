"""/ws/camera_thermal?source=phone|glasses: phone frames skip the rotation, glasses unchanged."""
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

import cv2
import numpy as np
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app_main
from perception_orientation import (
    RgbCanonicalizerTelemetry,
    canonicalize_phone_rgb_jpeg,
    canonicalize_rgb_jpeg,
)
from stability_runtime import MSG_TYPE_CAM
from trial_archive import sensor_source_changes
from trial_recorder import TrialRecorder


def _jpeg(width=64, height=32) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = (0, 0, 255)       # left half red, so rotation is visible
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _shape(jpeg: bytes):
    return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR).shape[:2]


class TransformTests(unittest.TestCase):
    def test_phone_frame_is_not_rotated(self):
        self.assertEqual(_shape(canonicalize_phone_rgb_jpeg(_jpeg())), (32, 64))

    def test_glasses_frame_is_still_rotated(self):
        self.assertEqual(_shape(canonicalize_rgb_jpeg(_jpeg())), (64, 32))

    def test_phone_rejects_non_jpeg(self):
        self.assertIsNone(canonicalize_phone_rgb_jpeg(b"\xff\xd8not a jpeg"))


class SocketTests(unittest.TestCase):
    """Drive the real handler; capture what reaches the fan-out."""

    def setUp(self):
        self.published = []
        self.got_frame = threading.Event()

        async def capture(data, frame_counter, *args, **kwargs):
            self.published.append(data)
            self.got_frame.set()
            return frame_counter + 1

        self.recorder = mock.Mock()
        self.patches = [
            mock.patch.object(app_main, "_handle_camera_frame", capture),
            mock.patch.object(app_main, "trial_recorder", self.recorder),
            mock.patch.object(app_main, "esp32_camera_ws", None),
            mock.patch.object(app_main, "camera_source", None),
        ]
        for patch in self.patches:
            patch.start()
        self.client = TestClient(app_main.app)

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()

    def _send_one_frame(self, path):
        with self.client.websocket_connect(path) as ws:
            ws.send_bytes(bytes([MSG_TYPE_CAM]) + _jpeg())
            self.assertTrue(self.got_frame.wait(5), "frame never reached fan-out")
        return self.published[0]

    def test_phone_source_skips_rotation(self):
        frame = self._send_one_frame("/ws/camera_thermal?source=phone")
        self.assertEqual(_shape(frame), (32, 64))
        self.assertEqual(app_main.camera_source, "phone")
        self.recorder.sensor_source.assert_called_once_with("phone")

    def test_no_param_is_glasses_and_byte_identical(self):
        frame = self._send_one_frame("/ws/camera_thermal")
        self.assertEqual(frame, canonicalize_rgb_jpeg(_jpeg()))
        self.assertEqual(app_main.camera_source, "glasses")
        self.recorder.sensor_source.assert_called_once_with("glasses")

    def test_unknown_source_is_refused(self):
        with self.assertRaises(WebSocketDisconnect) as caught:
            with self.client.websocket_connect("/ws/camera_thermal?source=phnoe") as ws:
                ws.receive_bytes()
        self.assertEqual(caught.exception.code, 1008)
        self.assertIsNone(app_main.camera_source)
        self.recorder.sensor_source.assert_not_called()


def _rec(root):
    # The real static-fields function, so the field name is the one app_main writes.
    return TrialRecorder(enabled=True, root=root, static_fields=app_main._trial_static_fields)


def _read(root, session_id):
    with open(os.path.join(root, session_id, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(os.path.join(root, session_id, "session.jsonl"), encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh]
    return meta, events


class RecordingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = tempfile.mkdtemp()

    async def test_source_lands_in_meta_and_session_start(self):
        with mock.patch.object(app_main, "camera_source", "phone"):
            rec = _rec(self.root)
            await rec.start()
            await rec.set_trial({"trial": "T10", "pid": "PSIM"}, None)
            await rec.stop()
        meta, events = _read(self.root, "T10_PSIM")
        self.assertEqual(meta["sensor_source"], "phone")
        start = next(e for e in events if e["t"] == "session_start")
        self.assertEqual(start["sensor_source"], "phone")

    async def test_null_before_any_device_and_first_connect_is_not_a_change(self):
        with mock.patch.object(app_main, "camera_source", None):
            rec = _rec(self.root)
            await rec.start()
            await rec.set_trial({"trial": "T11", "pid": "PSIM"}, None)
            rec.sensor_source("glasses")      # first device: not a switch
            rec.sensor_source("glasses")      # reconnect, same device: not a switch
            rec.sensor_source("phone")        # switch
            rec.sensor_source("glasses")      # switch back
            await rec.stop()
        meta, events = _read(self.root, "T11_PSIM")
        self.assertIsNone(meta["sensor_source"])
        changes = [(e["from"], e["to"]) for e in events if e["t"] == "sensor_source_change"]
        self.assertEqual(changes, [("glasses", "phone"), ("phone", "glasses")])

    async def test_switch_after_restart_is_still_a_change(self):
        with mock.patch.object(app_main, "camera_source", "glasses"):
            rec = _rec(self.root)
            await rec.start()
            await rec.set_trial({"trial": "T12", "pid": "PSIM"}, None)
            await rec.stop()
        # Restarted process: no device connected yet.
        with mock.patch.object(app_main, "camera_source", None):
            resumed = _rec(self.root)
            await resumed.start()
            self.assertEqual(resumed.status()["session_id"], "T12_PSIM")
            resumed.sensor_source("phone")
            await resumed.stop()
        _, events = _read(self.root, "T12_PSIM")
        changes = [(e["from"], e["to"]) for e in events if e["t"] == "sensor_source_change"]
        self.assertEqual(changes, [("glasses", "phone")])

    async def test_disabled_recorder_ignores_source(self):
        rec = TrialRecorder(enabled=False, root=self.root)
        rec.sensor_source("phone")
        self.assertEqual(os.listdir(self.root), [])


class ReconnectTests(unittest.TestCase):
    """A stale socket from a WiFi drop must not lock its own device out."""

    def setUp(self):
        self.published = 0
        self.lock = threading.Lock()
        self.frame = threading.Event()

        async def capture(data, frame_counter, *args, **kwargs):
            with self.lock:
                self.published += 1
            self.frame.set()
            return frame_counter + 1

        self.recorder = mock.Mock()
        self.telemetry = RgbCanonicalizerTelemetry()
        self.patches = [
            mock.patch.object(app_main, "_handle_camera_frame", capture),
            mock.patch.object(app_main, "trial_recorder", self.recorder),
            mock.patch.object(app_main, "rgb_canonicalizer_telemetry", self.telemetry),
            mock.patch.object(app_main, "esp32_camera_ws", None),
            mock.patch.object(app_main, "_camera_conn", None),
            mock.patch.object(app_main, "camera_source", None),
        ]
        for patch in self.patches:
            patch.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])
        self.client = TestClient(app_main.app)

    def _send_frame(self, ws):
        self.frame.clear()
        ws.send_bytes(bytes([MSG_TYPE_CAM]) + _jpeg())
        self.assertTrue(self.frame.wait(5), "frame never reached fan-out")

    def _replaces(self, first, second, old_device, new_device):
        with self.client.websocket_connect(first) as old:
            self._send_frame(old)
            with self.client.websocket_connect(second) as new:
                # The old socket is told why it was dropped.
                with self.assertRaises(WebSocketDisconnect) as closed:
                    old.receive_bytes()
                self.assertEqual(closed.exception.code, 1000)
                self.assertEqual(closed.exception.reason, "replaced_by_reconnect")
                old.__exit__(None, None, None)       # old handler runs its finally now
                # Its cleanup left the new socket's state alone ...
                held = app_main._camera_conn
                self.assertIsNotNone(held)
                self.assertIs(app_main.esp32_camera_ws, held["ws"])
                self.assertEqual(held["device"], new_device)
                self.assertFalse(held["replaced"])
                self.assertEqual(self.telemetry.health()["worker_state"], "running")
                # ... and frames keep flowing on the new one.
                before = self.published
                self._send_frame(new)
                self.assertEqual(self.published, before + 1)
        self.assertIsNone(app_main._camera_conn)
        self.assertIsNone(app_main.esp32_camera_ws)
        args = self.recorder.camera_replaced.call_args.args
        self.assertEqual(args[:3], ("phone", old_device, new_device))

    def test_same_device_reconnect_replaces_the_stale_socket(self):
        self._replaces("/ws/camera_thermal?source=phone&device=iphone-A",
                       "/ws/camera_thermal?source=phone&device=iphone-A", "iphone-A", "iphone-A")

    def test_missing_device_on_either_side_still_replaces(self):
        self._replaces("/ws/camera_thermal?source=phone",
                       "/ws/camera_thermal?source=phone&device=iphone-A", None, "iphone-A")
        self.recorder.reset_mock()
        self._replaces("/ws/camera_thermal?source=phone&device=iphone-A",
                       "/ws/camera_thermal?source=phone", "iphone-A", None)

    def _refused(self, first, second):
        with self.client.websocket_connect(first) as held:
            with self.assertRaises(WebSocketDisconnect) as caught:
                with self.client.websocket_connect(second) as ws:
                    ws.receive_bytes()
            self.assertEqual(caught.exception.code, 1013)
            self._send_frame(held)          # the holder is untouched
            self.assertFalse(app_main._camera_conn["replaced"])
        self.recorder.camera_replaced.assert_not_called()

    def test_different_phone_gets_1013(self):
        self._refused("/ws/camera_thermal?source=phone&device=iphone-A",
                      "/ws/camera_thermal?source=phone&device=iphone-B")

    def test_different_source_still_gets_1013(self):
        self._refused("/ws/camera_thermal?source=phone", "/ws/camera_thermal")

    def test_bad_device_param_is_refused(self):
        with self.assertRaises(WebSocketDisconnect) as caught:
            with self.client.websocket_connect("/ws/camera_thermal?source=phone&device=a/b") as ws:
                ws.receive_bytes()
        self.assertEqual(caught.exception.code, 1008)


class ReplacedRecordingTests(unittest.IsolatedAsyncioTestCase):
    async def test_camera_replaced_is_logged_and_is_not_a_source_change(self):
        root = tempfile.mkdtemp()
        with mock.patch.object(app_main, "camera_source", "phone"):
            rec = _rec(root)
            await rec.start()
            await rec.set_trial({"trial": "T13", "pid": "PSIM"}, None)
            rec.camera_replaced("phone", "iphone-A", "iphone-A", "1.2.3.4:1", "1.2.3.4:2")
            rec.sensor_source("phone")     # what the new connect calls next
            await rec.stop()
        _, events = _read(root, "T13_PSIM")
        replaced = [e for e in events if e["t"] == "camera_replaced"]
        self.assertEqual(len(replaced), 1)
        self.assertEqual((replaced[0]["source"], replaced[0]["old_device"], replaced[0]["device"],
                          replaced[0]["old_peer"], replaced[0]["peer"]),
                         ("phone", "iphone-A", "iphone-A", "1.2.3.4:1", "1.2.3.4:2"))
        self.assertFalse([e for e in events if e["t"] == "sensor_source_change"])


class ArchiveTests(unittest.TestCase):
    def test_viewer_gets_each_switch_in_order(self):
        events = [
            {"t": "session_start", "sensor_source": "glasses"},
            {"t": "sensor_source_change", "from": "glasses", "to": "phone", "ts": "a"},
            {"t": "speech_start"},
            {"t": "sensor_source_change", "from": "phone", "to": "glasses", "ts": "b"},
        ]
        self.assertEqual(sensor_source_changes(events), [
            {"from": "glasses", "to": "phone", "ts": "a"},
            {"from": "phone", "to": "glasses", "ts": "b"},
        ])
        self.assertEqual(sensor_source_changes([]), [])


if __name__ == "__main__":
    unittest.main()
