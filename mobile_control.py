from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass
class _Snapshot:
    volume: int
    recording: bool


class MobileControlState:
    """In-process state for the mobile remote control."""

    def __init__(self, default_volume: int = 90) -> None:
        self._lock = Lock()
        self._volume = self._coerce_volume(default_volume)
        self._recording = False
        self._start_requested = True
        self._stop_requested = False

    @staticmethod
    def _coerce_volume(value: Any) -> int:
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError):
            numeric = 90
        return max(0, min(100, numeric))

    @property
    def volume(self) -> int:
        with self._lock:
            return self._volume

    @property
    def gain(self) -> float:
        with self._lock:
            return self._volume / 100.0

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._recording

    def snapshot(self) -> dict:
        with self._lock:
            snap = _Snapshot(volume=self._volume, recording=self._recording)
        return {"volume": snap.volume, "recording": snap.recording}

    def set_volume(self, value: Any) -> int:
        volume = self._coerce_volume(value)
        with self._lock:
            self._volume = volume
        return volume

    def set_recording(self, active: bool) -> bool:
        active = bool(active)
        with self._lock:
            if active and not self._recording:
                self._start_requested = True
            elif not active and self._recording:
                self._stop_requested = True
            self._recording = active
            return self._recording

    def consume_start_requested(self) -> bool:
        with self._lock:
            requested = self._start_requested
            self._start_requested = False
            return requested

    def consume_stop_requested(self) -> bool:
        with self._lock:
            requested = self._stop_requested
            self._stop_requested = False
            return requested
