"""Per-session experiment knobs: the one place that answers "what is in force?".

Switching an A/B arm used to be a deploy — `fly secrets set PIN_FRAME_EVERY_TURN=0`
plus a redeploy between arms, about two minutes each way, with the two arms
landing on different machine instances. A knob registered here can instead be set
per trial, in the POST /internal/trial body, and lasts exactly as long as that
recorder session.

Scope and teardown, deliberately:

  * An override lives in this object, in memory, and is never written anywhere.
    meta.json records the values observed when the session opened; nothing on
    disk records an override that could be replayed.
  * clear() runs inside TrialRecorder.end_session(), which is the single close
    path — /internal/trial/end, the implicit close when a new trial replaces an
    open one, and the idle expiry of an untitled session all go through it.
  * A crash, a redeploy or any other restart therefore comes back on the env
    default. A session resumed from current.json resumes its *recording*, not
    its override, and the session_resume event carries the config the restarted
    process is really running, so the drop is visible in the log.

Adding a knob is one register() call. The request body, its validation, the
observed block in meta.json and what the viewer shows all follow from the
registry, so no part of the API has to be reshaped for the second knob.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def parse_bool(value: Any) -> bool:
    """Strict: a real bool, or the spellings env_bool() accepts. Never truthiness."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
    raise ValueError(f"expected a boolean, got {value!r}")


def keep(value: Any) -> Any:
    """Parser for a read-only knob; nothing may be set, so nothing is parsed."""
    return value


@dataclass(frozen=True)
class Knob:
    name: str
    default: Callable[[], Any]      # read at the moment it is asked for
    parse: Callable[[Any], Any] = keep
    settable: bool = False          # False: observed and recorded, never set per trial


class SessionConfig:
    """Resolver for knob values, plus the per-session overrides laid over them."""

    def __init__(self) -> None:
        self._knobs: Dict[str, Knob] = {}
        self._overrides: Dict[str, Any] = {}

    # ------------------------------------------------------------------ registry

    def register(self, name: str, default: Callable[[], Any], *,
                 parse: Callable[[Any], Any] = keep, settable: bool = False) -> None:
        self._knobs[name] = Knob(name, default, parse, settable)

    def names(self) -> Iterable[str]:
        return tuple(self._knobs)

    def settable_names(self) -> tuple:
        return tuple(name for name, knob in self._knobs.items() if knob.settable)

    # ------------------------------------------------------------------ reading

    def get(self, name: str) -> Any:
        """The value actually in force: the session override, else the env default."""
        if name in self._overrides:
            return self._overrides[name]
        knob = self._knobs.get(name)
        if knob is None:
            raise KeyError(f"unknown config knob: {name}")
        return knob.default()

    def observed(self) -> Dict[str, Any]:
        """Every knob's effective value, read now. This is what meta.json stores."""
        return {name: self.get(name) for name in self._knobs}

    def overrides(self) -> Dict[str, Any]:
        return dict(self._overrides)

    # ------------------------------------------------------------------ writing

    def parse(self, config: Any) -> Dict[str, Any]:
        """Validate a request's `config` object. Raises ValueError; changes nothing.

        None or {} means "no override": the env default stands, which is exactly
        today's behaviour for a request that carries no config at all.
        """
        if config is None:
            return {}
        if not isinstance(config, dict):
            raise ValueError("config must be a JSON object")
        known = self.settable_names()
        parsed: Dict[str, Any] = {}
        for key, value in config.items():
            knob = self._knobs.get(key)
            if knob is None or not knob.settable:
                raise ValueError(
                    f"unknown config key {key!r}; settable: {', '.join(known) or 'none'}"
                )
            if value is None:
                continue        # explicit "use the env default"
            try:
                parsed[key] = knob.parse(value)
            except ValueError as exc:
                raise ValueError(f"config.{key}: {exc}") from exc
        return parsed

    def apply(self, parsed: Dict[str, Any]) -> None:
        """Install an already-parsed override set, replacing any previous one."""
        self._overrides = dict(parsed)

    def clear(self) -> None:
        self._overrides = {}

    def conflict(self, parsed: Dict[str, Any]) -> Optional[str]:
        """Why `parsed` cannot be applied to the session already running, if so.

        The arm of a trial cannot change halfway through it and leave the trial
        meaning anything, and meta.json is written once at open, so a resumed
        session keeps the config it opened with.
        """
        for key, value in parsed.items():
            current = self.get(key)
            if value != current:
                return (f"{key} is {_render(current)} for the open trial and cannot be "
                        f"changed to {_render(value)}; end the trial and start a new one")
        return None


def _render(value: Any) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


class NullSessionConfig:
    """What TrialRecorder uses when no config is wired in (tests, other callers)."""

    def parse(self, config: Any) -> Dict[str, Any]:
        return {}

    def apply(self, parsed: Dict[str, Any]) -> None:
        return None

    def clear(self) -> None:
        return None

    def conflict(self, parsed: Dict[str, Any]) -> Optional[str]:
        return None

    def observed(self) -> Dict[str, Any]:
        return {}

    def overrides(self) -> Dict[str, Any]:
        return {}


# Every knob app_main registers, in the order the trial page shows them. The
# defaults live in app_main (they are its env vars); the names live here so the
# viewer and the archive can read a meta.json without importing the app. A test
# asserts the two stay in step, so adding a knob means adding it here too.
RECORDED_CONFIG_KEYS = (
    "pin_frame_every_turn",
    "gemini_video_interval_sec",
    "vision_frame_max_age_sec",
    "stability_mode",
    "model",
)


def recorded_config(meta: Dict[str, Any]) -> Dict[str, Any]:
    """The observed config a meta.json carries. Empty for a pre-stamp trial."""
    return {key: meta[key] for key in RECORDED_CONFIG_KEYS if key in meta}


def resume_drift(meta: Dict[str, Any], events: Iterable[Dict[str, Any]]) -> Optional[str]:
    """A session_resume whose observed config differs from meta.json's.

    meta.json is written once, at the first open, and a per-trial override is
    held in memory only — so a restart part-way through a trial resumes the
    recording on the env default instead. session_resume carries what the
    restarted process is really running, and when the two disagree, part of the
    run happened under a different setting and the trial is not one arm.
    """
    recorded = recorded_config(meta)
    if not recorded:
        return None
    for event in events:
        if event.get("t") != "session_resume":
            continue
        for key, value in recorded.items():
            if key in event and event[key] != value:
                return (f"{key} was {_render(value)} when this trial opened but "
                        f"{_render(event[key])} when it resumed"
                        f"{' at ' + str(event['ts']) if event.get('ts') else ''}.")
    return None


# ---------------------------------------------------------------- label checking

# What a typed cond label claims about the pin_frame_every_turn arm. "unpinned"
# is tested first: it contains "pinned".
_COND_CLAIMS = (("unpinned", False), ("un-pinned", False), ("pinned", True))


def condition_mismatch(cond: Any, pin_frame_every_turn: Any) -> Optional[str]:
    """The typed label against the flag that actually ran. None when they agree.

    The researcher types `cond` by hand ("pinned" / "unpinned") while the flag is
    whatever the backend was running. They can disagree, and a mislabelled trial
    invalidates the comparison it belongs to, so the disagreement has to be
    visible on the trial itself rather than found weeks later. A label that makes
    no claim about pinning, and a trial recorded before the flag was stamped into
    meta.json, both return None: unknown is not a mismatch.
    """
    if not isinstance(cond, str) or pin_frame_every_turn is None:
        return None
    text = cond.strip().lower()
    for token, claimed in _COND_CLAIMS:
        if token in text:
            actual = bool(pin_frame_every_turn)
            if claimed != actual:
                return (f"Condition says “{cond.strip()}” but this trial ran with "
                        f"pin_frame_every_turn={_render(actual)}.")
            return None
    return None
