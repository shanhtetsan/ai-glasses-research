"""Phrase-level checks for whether an utterance needs a camera frame.

Pure string matching with no detector vocabulary: a request about a specific
thing sends a frame whether or not any model has a class for that thing.
"""

from __future__ import annotations

import re


_HAND_PERCEPTION_PHRASES = (
    "which hand",
    "what hand",
    "hand am i",
    "holding up",
    "left hand",
    "right hand",
    "my hand",
    "open palm",
)
_HAND_PERCEPTION_WORDS = (
    "fist",
    "palm",
    "gesture",
    "pointing",
    "reach",
    "reaching",
    "grab",
    "grabbing",
    "grasp",
    "grasping",
)
_TARGET_DIRECTED_INTENT_RE = re.compile(
    r"\b(?:grab(?:bing|bed)?|grasp(?:ing|ed)?|pick(?:ing|ed)?\s+up|"
    r"reach(?:ing|ed)?\s+(?:for|toward|towards)|point(?:ing|ed)?\s+at|"
    r"find|locate|where\s+is|where\s+are)\b"
)


def is_target_directed_request(text: str) -> bool:
    """Recognize requests to find, locate, or reach for a specific thing."""
    normalized = " ".join(re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).split())
    return _TARGET_DIRECTED_INTENT_RE.search(normalized) is not None


def is_hand_perception_request(text: str) -> bool:
    """Recognize only explicit hand/gesture language, not every utterance."""
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return False
    if any(phrase in normalized for phrase in _HAND_PERCEPTION_PHRASES):
        return True
    return any(
        re.search(rf"\b{re.escape(word)}\b", normalized) is not None
        for word in _HAND_PERCEPTION_WORDS
    )
