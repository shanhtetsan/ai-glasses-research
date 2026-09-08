import unittest
from pathlib import Path

from mobile_control import MobileControlState


class MobileControlStateTests(unittest.TestCase):
    def test_default_state(self):
        state = MobileControlState()

        self.assertEqual(state.volume, 90)
        self.assertAlmostEqual(state.gain, 0.9)
        self.assertFalse(state.recording)
        self.assertTrue(state.consume_start_requested())
        self.assertFalse(state.consume_start_requested())
        self.assertFalse(state.consume_stop_requested())

    def test_set_volume_clamps_and_reports_integer_percent(self):
        state = MobileControlState(default_volume=90)

        self.assertEqual(state.set_volume(-10), 0)
        self.assertEqual(state.volume, 0)
        self.assertAlmostEqual(state.gain, 0.0)

        self.assertEqual(state.set_volume(101), 100)
        self.assertEqual(state.volume, 100)
        self.assertAlmostEqual(state.gain, 1.0)

        self.assertEqual(state.set_volume(42.7), 43)
        self.assertEqual(state.volume, 43)
        self.assertAlmostEqual(state.gain, 0.43)

    def test_recording_requests_are_edge_triggered(self):
        state = MobileControlState()

        state.set_recording(True)
        self.assertTrue(state.recording)
        self.assertTrue(state.consume_start_requested())
        self.assertFalse(state.consume_start_requested())
        self.assertFalse(state.consume_stop_requested())

        state.set_recording(True)
        self.assertFalse(state.consume_start_requested())

        state.set_recording(False)
        self.assertFalse(state.recording)
        self.assertTrue(state.consume_stop_requested())
        self.assertFalse(state.consume_stop_requested())

        state.set_recording(False)
        self.assertFalse(state.consume_stop_requested())


class AppMainMobileControlWiringTests(unittest.TestCase):
    def test_app_exposes_mobile_control_endpoints(self):
        source = Path("app_main.py").read_text(encoding="utf-8")

        self.assertIn('@app.get("/api/mobile-control")', source)
        self.assertIn('@app.post("/api/mobile-control/volume")', source)
        self.assertIn('@app.post("/api/mobile-control/recording")', source)
        self.assertIn("MobileVolumePayload", source)
        self.assertIn("MobileRecordingPayload", source)

    def test_audio_paths_apply_mobile_volume_and_recording_gate(self):
        source = Path("app_main.py").read_text(encoding="utf-8")

        self.assertIn("mobile_control_state = MobileControlState", source)
        self.assertIn("_apply_mobile_volume", source)
        self.assertIn("mobile_control_state.consume_start_requested()", source)
        self.assertIn("mobile_control_state.consume_stop_requested()", source)
        self.assertIn("mobile_control_state.recording", source)

    def test_mobile_control_cors_is_env_scoped(self):
        source = Path("app_main.py").read_text(encoding="utf-8")

        self.assertIn("MOBILE_CONTROL_ALLOWED_ORIGINS", source)
        self.assertIn("CORSMiddleware", source)
        self.assertIn("allow_origins=mobile_control_allowed_origins", source)


if __name__ == "__main__":
    unittest.main()
