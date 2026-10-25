import ast
import unittest
from pathlib import Path

from vision_intent import is_hand_perception_request, is_target_directed_request


class HandPerceptionIntentTests(unittest.TestCase):
    def test_requested_hand_phrases_trigger_conservatively(self):
        phrases = (
            "Which hand am I holding up?",
            "What hand am I holding up?",
            "Is this my hand?",
            "Show my left hand",
            "Can you see my right hand?",
            "Is this a fist?",
            "Do I have an open palm?",
            "What palm is visible?",
            "Recognize this gesture",
            "Am I pointing?",
            "I am reaching for it",
            "Can I grab that?",
            "I am grabbing the cup",
        )
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertTrue(is_hand_perception_request(phrase))

    def test_unrelated_words_do_not_make_every_turn_visual(self):
        for phrase in (
            "What time is it?",
            "Tell me a joke",
            "The graph is useful",
            "This paragraph is long",
            "I reached a conclusion",
        ):
            with self.subTest(phrase=phrase):
                self.assertFalse(is_hand_perception_request(phrase))


class TargetDirectedIntentTests(unittest.TestCase):
    def test_any_named_thing_counts_not_only_detector_classes(self):
        for phrase in (
            "Where is the door?",
            "Find the exit sign",
            "Grab the medicine",
            "Where are my keys",
            "Pick up the cup",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(is_target_directed_request(phrase))

    def test_unrelated_speech_is_not_target_directed(self):
        for phrase in ("What time is it?", "Tell me a joke", ""):
            with self.subTest(phrase=phrase):
                self.assertFalse(is_target_directed_request(phrase))


class AppVisionWiringTests(unittest.TestCase):
    def test_app_explicit_vision_path_has_no_detector_gate(self):
        source = Path("app_main.py").read_text(encoding="utf-8")
        function = source.split("def is_explicit_vision_request", 1)[1].split(
            "# ---- Thermal facts", 1
        )[0]
        self.assertIn("is_hand_perception_request", function)
        self.assertIn("is_target_directed_request", function)
        # No detector-vocabulary gate: "where is the door?" must send a frame
        # even though "door" is not a COCO class.
        self.assertNotIn("extract_requested_target", function)
        self.assertNotIn("should_wait_for_explicit_target", function)
        callback = source.split("async def _on_input_transcription", 1)[1].split(
            "async def _on_output_transcription", 1
        )[0]
        self.assertIn("wants_vision = is_explicit_vision_request(combined)", callback)
        # Detector facts are gone; thermal facts still go out on their own send.
        self.assertNotIn("PERCEPTION_STATE", source)
        self.assertNotIn("perception_state_", source)
        self.assertIn('"THERMAL_MEASUREMENTS "', callback)
        for reason in ("no_frame", "frame_too_old", "already_sent_this_turn"):
            self.assertIn(f'"{reason}"', callback)
        self.assertIn('"not_a_vision_request"', source)
        for module in (
            "yolo_client", "hand_client", "yolo_seg_client", "yoloe_client",
            "perception_fusion",
        ):
            self.assertNotIn(module, source)

    def test_per_turn_flags_are_declared_global_wherever_they_are_reset(self):
        # A bare assignment without `global` only makes a local, so the flag
        # never resets — one declined turn then disables vision for good.
        tree = ast.parse(Path("app_main.py").read_text(encoding="utf-8"))
        checked = 0
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in ("_on_turn_complete", "_clear_gemini_turn_state"):
                continue
            declared = {
                name
                for stmt in ast.walk(node) if isinstance(stmt, ast.Global)
                for name in stmt.names
            }
            assigned = {
                target.id
                for stmt in ast.walk(node) if isinstance(stmt, ast.Assign)
                for target in stmt.targets
                if isinstance(target, ast.Name) and target.id.endswith("_for_turn")
            }
            self.assertIn("_vision_declined_for_turn", assigned)
            self.assertEqual(assigned - declared, set(), node.name)
            checked += 1
        self.assertEqual(checked, 2)

    def test_system_prompt_describes_image_only_vision(self):
        from gemini_live_client import SMART_GLASSES_SYSTEM_INSTRUCTION as prompt
        for gone in ("PERCEPTION_STATE", "MediaPipe", "requested_target", "authoritative facts"):
            self.assertNotIn(gone, prompt)
        self.assertIn("THERMAL_MEASUREMENTS", prompt)


if __name__ == "__main__":
    unittest.main()
