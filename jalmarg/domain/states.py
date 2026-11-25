"""RoadSegment state machine (State pattern).

    Dry --rain > r0--> Wet --h > 5 cm--> Rising
    Wet --dry for 30 min--> Dry
    Rising --dh/dt < 0 sustained--> Receding
    Receding --rain resumes / rising again--> Rising
    Receding --h < 5 cm--> Wet
    any --no observation > 20 min--> Unknown --new observation--> Wet / Rising / Receding

"Impassable" is deliberately NOT a state: passability depends on the vehicle,
so it is a query, RoadSegment.time_to_passable(vehicle).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from .segment import RoadSegment
    from .vehicles import VehicleProfile


class StateConfig:
    rain_threshold_mm_h: float = 2.0   # r0
    wet_depth_cm: float = 5.0          # h threshold between Wet and Rising/Receding
    dry_after_min: float = 30.0
    stale_after_min: float = 20.0
    rate_eps_cm_min: float = 0.05
    sustained_count: int = 2


class SegmentState(ABC):
    name: str = "state"
    reforecast_every_min: Optional[float] = None
    color: str = "gray"

    # --- behaviour that differs per state -------------------------------
    def blocks(self, vehicle: "VehicleProfile") -> bool:
        """Fail-safe rule: does this state block the vehicle regardless of forecast?"""
        return False

    # --- transitions ------------------------------------------------------
    def on_observation(self, seg: "RoadSegment") -> "SegmentState":
        """Called after the tracker assimilated a new observation."""
        return self._next(seg)

    def on_tick(self, seg: "RoadSegment") -> "SegmentState":
        """Called on the clock (rain updates, staleness)."""
        if seg.is_stale():
            return UNKNOWN
        return self._next(seg)

    @abstractmethod
    def _next(self, seg: "RoadSegment") -> "SegmentState": ...

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return isinstance(other, SegmentState) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


class Dry(SegmentState):
    name, reforecast_every_min, color = "Dry", 30.0, "green"

    def _next(self, seg):
        if seg.depth_cm > StateConfig.wet_depth_cm:
            return RISING
        if seg.rain_mm_h > StateConfig.rain_threshold_mm_h or seg.depth_cm > 1.0:
            return WET
        return self


class Wet(SegmentState):
    name, reforecast_every_min, color = "Wet", 10.0, "blue"

    def _next(self, seg):
        if seg.depth_cm > StateConfig.wet_depth_cm:
            return RISING
        dry_since = seg.dry_since
        if (
            dry_since is not None
            and seg.depth_cm < 1.0
            and seg.now - dry_since >= timedelta(minutes=StateConfig.dry_after_min)
        ):
            return DRY
        return self


class Rising(SegmentState):
    name, reforecast_every_min, color = "Rising", 2.0, "red"

    def _next(self, seg):
        if seg.falling_streak >= StateConfig.sustained_count:
            return RECEDING
        return self


class Receding(SegmentState):
    name, reforecast_every_min, color = "Receding", 5.0, "orange"

    def _next(self, seg):
        if seg.rain_mm_h > StateConfig.rain_threshold_mm_h or seg.rising_streak >= StateConfig.sustained_count:
            return RISING
        if seg.depth_cm < StateConfig.wet_depth_cm:
            return WET
        return self


class Unknown(SegmentState):
    name, reforecast_every_min, color = "Unknown", None, "gray"

    def blocks(self, vehicle):
        return vehicle.is_low  # fail-safe for vehicles with safe depth < 30 cm

    def on_tick(self, seg):
        return self  # stays Unknown until a new observation arrives

    def on_observation(self, seg):
        if seg.depth_cm < StateConfig.wet_depth_cm:
            return WET
        if seg.rate_cm_min < -StateConfig.rate_eps_cm_min:
            return RECEDING
        return RISING

    def _next(self, seg):  # pragma: no cover - not used
        return self


DRY, WET, RISING, RECEDING, UNKNOWN = Dry(), Wet(), Rising(), Receding(), Unknown()
STATES = {s.name: s for s in (DRY, WET, RISING, RECEDING, UNKNOWN)}
