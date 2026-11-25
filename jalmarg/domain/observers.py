"""Observers of RoadSegment changes (Observer pattern)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Dict, List, Set

if TYPE_CHECKING:  # pragma: no cover
    from .segment import RoadSegment
    from .states import SegmentState


class SegmentObserver(ABC):
    @abstractmethod
    def on_state_change(self, segment: "RoadSegment", old: "SegmentState", new: "SegmentState") -> None: ...


@dataclass
class Alert:
    time: datetime
    segment_id: str
    segment_name: str
    old_state: str
    new_state: str
    depth_cm: float
    message: str


class AlertService(SegmentObserver):
    """Raises hotspot alerts for municipal operators and subscribed users."""

    def __init__(self, hotspot_depth_cm: float = 15.0):
        self.hotspot_depth_cm = hotspot_depth_cm
        self.alerts: List[Alert] = []
        self._subscriptions: Dict[str, Set[str]] = {}
        self._listeners: List[Callable[[str, Alert], None]] = []

    def subscribe(self, user_id: str, segment_ids) -> None:
        self._subscriptions.setdefault(user_id, set()).update(segment_ids)

    def unsubscribe(self, user_id: str) -> None:
        self._subscriptions.pop(user_id, None)

    def add_listener(self, fn: Callable[[str, Alert], None]) -> None:
        """fn(user_id, alert); user_id == '*' for municipal broadcast."""
        self._listeners.append(fn)

    def on_state_change(self, segment, old, new) -> None:
        if new.name == "Rising" and segment.depth_cm >= self.hotspot_depth_cm:
            msg = f"Hotspot: {segment.name} rising, ~{segment.depth_cm:.0f} cm"
        elif new.name == "Unknown":
            msg = f"{segment.name}: no recent data, treated as unsafe for low vehicles"
        else:
            msg = f"{segment.name}: {old.name} -> {new.name} (~{segment.depth_cm:.0f} cm)"
        alert = Alert(segment.now, segment.segment_id, segment.name, old.name, new.name, segment.depth_cm, msg)
        self.alerts.append(alert)
        for fn in self._listeners:
            fn("*", alert)
            for user, segs in self._subscriptions.items():
                if segment.segment_id in segs:
                    fn(user, alert)

    def for_user(self, user_id: str) -> List[Alert]:
        segs = self._subscriptions.get(user_id, set())
        return [a for a in self.alerts if a.segment_id in segs]


class RouteCacheInvalidator(SegmentObserver):
    """Drops cached routes/advice that use a segment whose state changed."""

    def __init__(self):
        self._cache: Dict[str, object] = {}
        self._segments_of: Dict[str, Set[str]] = {}
        self.invalidations = 0

    def put(self, key: str, value: object, segment_ids) -> None:
        self._cache[key] = value
        self._segments_of[key] = set(segment_ids)

    def get(self, key: str):
        return self._cache.get(key)

    def on_state_change(self, segment, old, new) -> None:
        stale = [k for k, segs in self._segments_of.items() if segment.segment_id in segs]
        for k in stale:
            self._cache.pop(k, None)
            self._segments_of.pop(k, None)
        self.invalidations += len(stale)

    def clear(self) -> None:
        self._cache.clear()
        self._segments_of.clear()
