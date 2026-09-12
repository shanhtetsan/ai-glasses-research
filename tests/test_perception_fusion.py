import unittest
from pathlib import Path

from perception_fusion import (
    HandTargetGuidanceTracker,
    TargetStabilityTracker,
    build_hand_fusion_fact,
    classify_hand_gesture,
    compact_guidance_for_log,
    compact_hand_facts_for_log,
    extract_requested_target,
    guidance_anchor_for_utterance,
    guidance_anchor_reason,
    is_hand_perception_request,
    should_wait_for_explicit_target,
)


def _gesture_landmarks(pattern):
    points = {
        0: (0.50, 0.90),
        1: (0.43, 0.82), 2: (0.38, 0.75), 3: (0.33, 0.70), 4: (0.28, 0.66),
    }
    for (mcp, pip, dip, tip), x, state in zip(
        ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)),
        (0.38, 0.46, 0.54, 0.62),
        pattern,
    ):
        points[mcp] = (x, 0.66)
        points[pip] = (x, 0.50)
        if state == "extended":
            points[dip] = (x, 0.34)
            points[tip] = (x, 0.18)
        elif state == "folded":
            points[dip] = (x + 0.04, 0.55)
            points[tip] = (x + 0.07, 0.62)
        else:
            points[dip] = (x, 0.50)
            points[tip] = (x, 0.50)
    return [
        {"id": index, "x": points[index][0], "y": points[index][1], "z": 0.0}
        for index in range(21)
    ]


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

    def test_app_explicit_vision_path_includes_hand_intent(self):
        source = Path("app_main.py").read_text(encoding="utf-8")
        function = source.split("def is_explicit_vision_request", 1)[1].split(
            "# ---- Thermal facts", 1
        )[0]
        self.assertIn("is_hand_perception_request", function)
        callback = source.split("async def _on_input_transcription", 1)[1].split(
            "async def _on_output_transcription", 1
        )[0]
        self.assertIn("wants_vision = is_explicit_vision_request(combined)", callback)
        self.assertIn("not _perception_submitted_for_turn", callback)

    def test_streamed_target_command_waits_for_supported_noun(self):
        self.assertTrue(should_wait_for_explicit_target("grab"))
        self.assertTrue(should_wait_for_explicit_target("point"))
        self.assertTrue(should_wait_for_explicit_target("point at the"))
        self.assertFalse(should_wait_for_explicit_target("grab the cup"))
        self.assertFalse(should_wait_for_explicit_target("grab that"))


class RequestedTargetExtractionTests(unittest.TestCase):
    def test_exact_supported_targets_and_plurals(self):
        self.assertEqual(extract_requested_target("Grab the cup."), "cup")
        self.assertEqual(extract_requested_target("Point at the BOTTLES"), "bottle")
        self.assertEqual(extract_requested_target("Find my books"), "book")
        self.assertEqual(extract_requested_target("Reach for the mice"), "mouse")

    def test_declared_aliases_and_separator_normalization(self):
        self.assertEqual(extract_requested_target("Find the sofa"), "couch")
        self.assertEqual(extract_requested_target("Pick up my cell-phone"), "cell phone")
        self.assertEqual(extract_requested_target("Where is the fridge?"), "refrigerator")

    def test_unknown_or_multiple_targets_are_not_guessed(self):
        self.assertIsNone(extract_requested_target("Grab the medicine"))
        self.assertIsNone(extract_requested_target("Choose the cup or bottle"))

    def test_target_extraction_is_current_turn_only(self):
        self.assertEqual(extract_requested_target("Grab the cup"), "cup")
        self.assertIsNone(extract_requested_target("Tell me a joke"))
        source = Path("app_main.py").read_text(encoding="utf-8")
        self.assertIn("def build_perception_state(utterance: str =", source)
        self.assertIn("perception, perception_raw = build_perception_state(combined)", source)


class AuthoritativeHandednessTests(unittest.TestCase):
    @staticmethod
    def _hand(label, score, x):
        return {
            "hand_index": 0,
            "handedness": label,
            "handedness_score": score,
            "hand_center_norm": [x, 0.5],
            "index_tip_norm": [x, 0.3],
            "wrist_norm": [x, 0.8],
            "bbox_norm": [max(0, x - 0.1), 0.2, min(1, x + 0.1), 0.9],
        }

    def test_anatomical_labels_are_not_swapped_by_image_position(self):
        right = build_hand_fusion_fact(self._hand("Right", 0.94, 0.2))
        left = build_hand_fusion_fact(self._hand("Left", 0.99, 0.8))

        self.assertEqual(right["anatomical_handedness"], "Right")
        self.assertEqual(right["handedness"], "Right")
        self.assertEqual(right["image_position"], "image_left")
        self.assertTrue(right["handedness_authoritative"])

        self.assertEqual(left["anatomical_handedness"], "Left")
        self.assertEqual(left["handedness"], "Left")
        self.assertEqual(left["image_position"], "image_right")
        self.assertTrue(left["handedness_authoritative"])
        self.assertEqual(
            compact_hand_facts_for_log([right, left]),
            "Right:0.94@image_left,Left:0.99@image_right",
        )

    def test_low_confidence_label_is_preserved_but_not_authoritative(self):
        fact = build_hand_fusion_fact(self._hand("Right", 0.79, 0.8))
        self.assertEqual(fact["anatomical_handedness"], "Right")
        self.assertEqual(fact["image_position"], "image_right")
        self.assertFalse(fact["handedness_authoritative"])

    def test_structured_prompt_states_authority_and_gesture_limit(self):
        source = Path("app_main.py").read_text(encoding="utf-8")
        callback = source.split("PERCEPTION_STATE", 1)[1].split(
            "# Thermal goes as structured text", 1
        )[0]
        self.assertIn("ANATOMICAL handedness", callback)
        self.assertIn("MUST NOT override", callback)
        self.assertIn("Right hand can appear at image_left", callback)
        self.assertIn('"hand can appear at image_right', callback)
        self.assertIn("gesture_authoritative", callback)
        self.assertIn("MUST NOT override it from the", callback)


class GestureClassificationTests(unittest.TestCase):
    def test_obvious_open_palm(self):
        result = classify_hand_gesture(_gesture_landmarks(["extended"] * 4))
        self.assertEqual(result["gesture"], "open_palm")
        self.assertEqual(result["gesture_source"], "landmark_geometry")
        self.assertTrue(result["gesture_authoritative"])

    def test_obvious_closed_fist(self):
        result = classify_hand_gesture(_gesture_landmarks(["folded"] * 4))
        self.assertEqual(result["gesture"], "fist")
        self.assertTrue(result["gesture_authoritative"])

    def test_index_pointing(self):
        landmarks = _gesture_landmarks(
            ["extended", "folded", "folded", "folded"]
        )
        result = classify_hand_gesture(landmarks)
        self.assertEqual(result["gesture"], "pointing")
        self.assertTrue(result["gesture_authoritative"])
        hand = AuthoritativeHandednessTests._hand("Right", 0.97, 0.4)
        hand["landmarks"] = landmarks
        fused = build_hand_fusion_fact(hand)
        self.assertEqual(fused["anatomical_handedness"], "Right")
        self.assertEqual(fused["gesture"], "pointing")

    def test_ambiguous_pose_is_unknown(self):
        result = classify_hand_gesture(
            _gesture_landmarks(["extended", "ambiguous", "folded", "folded"])
        )
        self.assertEqual(result["gesture"], "unknown")
        self.assertFalse(result["gesture_authoritative"])
        self.assertEqual(result["gesture_confidence"], 0.0)


class TargetStabilityTests(unittest.TestCase):
    @staticmethod
    def _object(label="cup", confidence=0.83, bbox=None):
        return {
            "label": label,
            "confidence": confidence,
            "bbox_norm": bbox or [0.55, 0.35, 0.75, 0.65],
        }

    def test_low_confidence_never_becomes_guidance_target(self):
        tracker = TargetStabilityTracker(persistence_frames=2)
        for frame_id in range(1, 5):
            state = tracker.update([self._object(confidence=0.49)], frame_id)
        self.assertEqual(state["target_state"], "uncertain")
        self.assertNotIn("target", state)

    def test_label_and_overlap_must_persist(self):
        tracker = TargetStabilityTracker(persistence_frames=3, iou_threshold=0.3)
        first = tracker.update([self._object()], 10)
        second = tracker.update(
            [self._object(bbox=[0.56, 0.35, 0.76, 0.65])], 11
        )
        third = tracker.update(
            [self._object(bbox=[0.57, 0.35, 0.77, 0.65])], 12
        )
        self.assertEqual(first["target_state"], "uncertain")
        self.assertEqual(second["persistence_count"], 2)
        self.assertEqual(third["target_state"], "stable")
        self.assertEqual(third["target"], "cup")
        self.assertEqual(third["target_center"], [0.67, 0.5])

    def test_stable_target_is_sticky_and_switch_requires_persistence(self):
        tracker = TargetStabilityTracker(persistence_frames=2)
        tracker.update([self._object()], 1)
        stable_cup = tracker.update([self._object()], 2)
        still_cup = tracker.update([
            self._object(),
            self._object("bottle", 0.99, [0.1, 0.2, 0.3, 0.6]),
        ], 3)
        switching = tracker.update(
            [self._object("bottle", 0.99, [0.1, 0.2, 0.3, 0.6])], 4
        )
        qualifying_bottle = tracker.update(
            [self._object("bottle", 0.98, [0.11, 0.2, 0.31, 0.6])], 5
        )
        stable_bottle = tracker.update(
            [self._object("bottle", 0.97, [0.12, 0.2, 0.32, 0.6])], 6
        )
        self.assertEqual(stable_cup["target"], "cup")
        self.assertEqual(still_cup["target"], "cup")
        self.assertEqual(switching["target_state"], "stable")
        self.assertEqual(switching["target"], "cup")
        self.assertTrue(switching["held_through_miss"])
        self.assertEqual(qualifying_bottle["target_state"], "uncertain")
        self.assertEqual(stable_bottle["target_state"], "stable")
        self.assertEqual(stable_bottle["target"], "bottle")

    def test_lost_target_must_requalify_after_reappearing(self):
        tracker = TargetStabilityTracker(persistence_frames=2)
        tracker.update([self._object()], 1)
        self.assertEqual(tracker.update([self._object()], 2)["target_state"], "stable")
        held = tracker.update([], 3)
        expired = tracker.update([], 4)
        reacquired_once = tracker.update([self._object()], 5)
        reacquired_twice = tracker.update([self._object()], 6)
        self.assertEqual(held["target_state"], "stable")
        self.assertTrue(held["held_through_miss"])
        self.assertEqual(held["missed_frames"], 1)
        self.assertEqual(expired["target_state"], "uncertain")
        self.assertEqual(reacquired_once["target_state"], "uncertain")
        self.assertEqual(reacquired_twice["target_state"], "stable")

    def test_miss_grace_preserves_last_bbox_and_rejects_competitor(self):
        tracker = TargetStabilityTracker(persistence_frames=2, max_missed_frames=1)
        original = self._object(bbox=[0.5, 0.3, 0.7, 0.7])
        tracker.update([original], 1)
        stable = tracker.update([original], 2)
        held = tracker.update([
            self._object("keyboard", 0.99, [0.05, 0.1, 0.4, 0.4])
        ], 3)
        after_grace = tracker.update([
            self._object("keyboard", 0.99, [0.05, 0.1, 0.4, 0.4])
        ], 4)
        self.assertEqual(stable["target"], "cup")
        self.assertEqual(held["target"], "cup")
        self.assertEqual(held["target_bbox"], [0.5, 0.3, 0.7, 0.7])
        self.assertEqual(held["target_observation"], "held_miss")
        self.assertEqual(after_grace["target_state"], "uncertain")
        self.assertEqual(after_grace["target"], "keyboard")


class HandTargetGeometryTests(unittest.TestCase):
    @staticmethod
    def _target(center_x=0.60, center_y=0.50):
        return {
            "target_state": "stable",
            "target": "cup",
            "target_confidence": 0.83,
            "target_bbox": [center_x - 0.1, center_y - 0.1, center_x + 0.1, center_y + 0.1],
        }

    @staticmethod
    def _hand(index_tip=(0.74, 0.45), hand_center=(0.50, 0.60)):
        return {
            "hand_index": 0,
            "anatomical_handedness": "Right",
            "handedness_score": 0.97,
            "handedness_authoritative": True,
            "gesture": "pointing",
            "gesture_source": "landmark_geometry",
            "gesture_authoritative": True,
            "index_tip_norm": list(index_tip),
            "hand_center_norm": list(hand_center),
        }

    def test_geometry_uses_target_bbox_center_and_canonical_axes(self):
        tracker = HandTargetGuidanceTracker(deadband=0.06, hysteresis=0.015)
        guidance = tracker.build(
            self._target(), self._hand(), objects_age_ms=100, hands_age_ms=40,
            object_frame_id=9, hand_frame_id=10,
        )
        self.assertEqual(guidance["target_center"], [0.6, 0.5])
        self.assertEqual(guidance["dx_norm"], -0.14)
        self.assertEqual(guidance["dy_norm"], 0.05)
        self.assertEqual(guidance["horizontal_relation"], "target_left")
        self.assertEqual(guidance["vertical_relation"], "vertically_aligned")
        self.assertFalse(guidance["index_inside_target_bbox"])
        self.assertFalse(guidance["physical_distance_available"])
        self.assertEqual(guidance["object_frame_id"], 9)
        self.assertEqual(guidance["hand_frame_id"], 10)
        self.assertEqual(
            compact_guidance_for_log(guidance),
            "requested=none target=cup:0.83 stable=yes hand=Right:0.97 "
            "gesture=pointing anchor=index_tip dx=-0.14 dy=0.05 "
            "h=target_left v=aligned inside=no",
        )

    def test_pointing_uses_index_tip_and_grabbing_uses_hand_center(self):
        hand = self._hand(index_tip=(0.74, 0.45), hand_center=(0.50, 0.60))
        pointing = HandTargetGuidanceTracker().build(
            self._target(), hand,
            guidance_anchor=guidance_anchor_for_utterance("point at the cup"),
            anchor_reason=guidance_anchor_reason("point at the cup"),
            requested_target="cup",
        )
        grabbing = HandTargetGuidanceTracker().build(
            self._target(), hand,
            guidance_anchor=guidance_anchor_for_utterance("grab the cup"),
            anchor_reason=guidance_anchor_reason("grab the cup"),
            requested_target="cup",
        )
        self.assertEqual(pointing["guidance_anchor"], "index_tip")
        self.assertEqual(pointing["guidance_anchor_norm"], [0.74, 0.45])
        self.assertEqual(pointing["dx_norm"], -0.14)
        self.assertEqual(grabbing["guidance_anchor"], "hand_center")
        self.assertEqual(grabbing["guidance_anchor_norm"], [0.5, 0.6])
        self.assertEqual(grabbing["dx_norm"], 0.1)
        self.assertEqual(grabbing["dy_norm"], -0.1)
        self.assertEqual(pointing["anatomical_hand"], "Right")
        self.assertEqual(grabbing["anatomical_hand"], "Right")
        self.assertEqual(
            compact_guidance_for_log(grabbing),
            "requested=cup target=cup:0.83 stable=yes hand=Right:0.97 "
            "gesture=pointing anchor=hand_center dx=0.10 dy=-0.10 "
            "h=target_right v=target_above inside=yes",
        )

    def test_ambiguous_anchor_intent_defaults_to_index_tip(self):
        utterance = "point while reaching for the cup"
        self.assertEqual(guidance_anchor_for_utterance(utterance), "index_tip")
        self.assertEqual(guidance_anchor_reason(utterance), "ambiguous_default")

    def test_deadband_hysteresis_prevents_alignment_chatter(self):
        tracker = HandTargetGuidanceTracker(deadband=0.06, hysteresis=0.015)
        right = tracker.build(self._target(0.60), self._hand((0.50, 0.50)))
        held_right = tracker.build(self._target(0.55), self._hand((0.50, 0.50)))
        aligned = tracker.build(self._target(0.54), self._hand((0.50, 0.50)))
        held_aligned = tracker.build(self._target(0.57), self._hand((0.50, 0.50)))
        right_again = tracker.build(self._target(0.58), self._hand((0.50, 0.50)))
        self.assertEqual(right["horizontal_relation"], "target_right")
        self.assertEqual(held_right["horizontal_relation"], "target_right")
        self.assertEqual(aligned["horizontal_relation"], "horizontally_aligned")
        self.assertEqual(held_aligned["horizontal_relation"], "horizontally_aligned")
        self.assertEqual(right_again["horizontal_relation"], "target_right")

    def test_uncertain_target_never_produces_directional_facts(self):
        tracker = HandTargetGuidanceTracker()
        guidance = tracker.build(
            {"target_state": "uncertain", "target": "cup", "target_confidence": 0.45},
            self._hand(),
        )
        self.assertEqual(guidance["target_state"], "uncertain")
        self.assertNotIn("dx_norm", guidance)
        self.assertNotIn("horizontal_relation", guidance)


if __name__ == "__main__":
    unittest.main()
