import unittest
from pathlib import Path


class ThermalModeTests(unittest.TestCase):
    """app_main.py has import-time hardware/network side effects, so these
    check the source text rather than importing the module (same convention
    as test_stability_runtime.py / test_perception_orientation.py)."""

    def setUp(self):
        self.source = Path("app_main.py").read_text(encoding="utf-8")

    def test_env_var_and_modes_declared(self):
        self.assertIn('os.getenv("THERMAL_MODE", "auto")', self.source)
        self.assertIn('_THERMAL_MODES = ("auto", "always", "never")', self.source)

    def test_live_toggle_endpoints_exist(self):
        self.assertIn('@app.get("/api/thermal-mode")', self.source)
        self.assertIn('@app.post("/api/thermal-mode")', self.source)

    def test_thermal_send_is_gated_by_thermal_gate_open(self):
        self.assertIn(
            "if thermal_gate_open and not _thermal_submitted_for_turn:", self.source
        )

    def test_vision_frame_trigger_stays_on_raw_phrase_not_mode(self):
        # Pins that THERMAL_MODE only gates the THERMAL_MEASUREMENTS send,
        # not the incidental vision-frame capture that piggybacks on the
        # same phrase detection — keeps the A/B test isolated to thermal text.
        self.assertIn(
            "if not _vision_submitted_for_turn and (wants_vision or wants_thermal):",
            self.source,
        )

    def test_turn_telemetry_records_mode_and_sent(self):
        self.assertIn('record["thermal_mode"] = thermal_mode_config["mode"]', self.source)
        self.assertIn('record["thermal_sent"] = _thermal_submitted_for_turn', self.source)

    def test_turn_telemetry_records_facts_available(self):
        # Distinguishes "thermal never attempted this turn" (None) from
        # "attempted but build_thermal_facts() was stale/empty" (False) —
        # without this, both looked identical to a null input_transcription.
        self.assertIn(
            'record["thermal_facts_available"] = _thermal_facts_available_for_turn',
            self.source,
        )
        self.assertIn("_thermal_facts_available_for_turn = facts is not None", self.source)

    def test_mode_change_triggers_session_reset(self):
        self.assertIn("async def update_thermal_mode(settings: ThermalModeSettings):", self.source)
        self.assertIn("await gemini_live.reset_session(", self.source)
        self.assertIn("if normalized != previous:", self.source)

    def test_manual_session_reset_endpoint_exists(self):
        self.assertIn('@app.post("/api/gemini-session/reset")', self.source)


if __name__ == "__main__":
    unittest.main()
