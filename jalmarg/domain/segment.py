"""RoadSegment: tracked depth, state machine, observers and passability queries."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..temporal.forecaster import HORIZON, PhysicsForecaster, Quantiles, RecessionForecaster
from ..temporal.kalman import KalmanDepthTracker
from .observation import Observation
from .observers import SegmentObserver
from .states import DRY, StateConfig, SegmentState, UNKNOWN
from .vehicles import VehicleProfile


@dataclass
class OperatorOverride:
    """Human in the loop: a municipal operator's decision beats the model."""
    mode: str                      # "open" | "closed"
    set_by: str = "operator"
    until: Optional[datetime] = None
    reason: str = ""

    def active(self, now: datetime) -> bool:
        return self.until is None or now < self.until


@dataclass
class TimeToPassable:
    status: str                    # "now" | "in" | "beyond_horizon" | "blocked"
    minutes: Optional[float]       # None unless status in ("now", "in")
    q90_now_cm: Optional[float]
    reason: str = ""

    @property
    def passable_now(self) -> bool:
        return self.status == "now"

    def describe(self, vehicle_label: str) -> str:
        if self.status == "now":
            return f"Likely passable now for {vehicle_label} (estimate)"
        if self.status == "in":
            return f"Likely passable for {vehicle_label} in ~{self.minutes:.0f} min (estimate)"
        if self.status == "blocked":
            return f"Avoid for {vehicle_label}: {self.reason}"
        return f"Not expected to be passable for {vehicle_label} within the forecast horizon"


class RoadSegment:
    def __init__(
        self,
        segment_id: str,
        name: str = "",
        forecaster: Optional[RecessionForecaster] = None,
        tracker: Optional[KalmanDepthTracker] = None,
        horizon_min: int = HORIZON,
    ):
        self.segment_id = segment_id
        self.name = name or segment_id
        self.forecaster = forecaster or PhysicsForecaster()
        self.tracker = tracker or KalmanDepthTracker()
        self.horizon_min = horizon_min

        self._state: SegmentState = DRY
        self._observers: List[SegmentObserver] = []
        self.override: Optional[OperatorOverride] = None

        self.now: datetime = datetime.min
        self.rain_mm_h: float = 0.0
        self.rain_log: List[Tuple[datetime, float]] = []
        self.last_obs_time: Optional[datetime] = None
        self.dry_since: Optional[datetime] = None
        self.falling_streak = 0
        self.rising_streak = 0
        self.n_observations = 0
        self._forecast_cache: Dict[tuple, Quantiles] = {}

    # ---------------------------------------------------------------- observers
    def attach(self, obs: SegmentObserver) -> None:
        if obs not in self._observers:
            self._observers.append(obs)

    def detach(self, obs: SegmentObserver) -> None:
        if obs in self._observers:
            self._observers.remove(obs)

    def _notify(self, old: SegmentState, new: SegmentState) -> None:
        for o in list(self._observers):
            o.on_state_change(self, old, new)

    # ---------------------------------------------------------------- state
    @property
    def state(self) -> SegmentState:
        return self._state

    def _set_state(self, new: SegmentState) -> None:
        if new != self._state:
            old, self._state = self._state, new
            self._forecast_cache.clear()
            self._notify(old, new)

    def _cascade(self, first: SegmentState) -> None:
        """Apply transitions until stable (e.g. Dry -> Wet -> Rising in one step)."""
        self._set_state(first)
        for _ in range(4):
            nxt = self._state._next(self) if self._state is not UNKNOWN else self._state
            if nxt == self._state:
                break
            self._set_state(nxt)

    # ---------------------------------------------------------------- tracked values
    @property
    def depth_cm(self) -> float:
        tp = self.tracker.predict_at(self.now) if self.tracker.initialized else None
        return tp.depth_cm if tp else 0.0

    @property
    def sigma_cm(self) -> float:
        tp = self.tracker.predict_at(self.now) if self.tracker.initialized else None
        return tp.sigma_cm if tp else 0.0

    @property
    def rate_cm_min(self) -> float:
        return self.tracker.x[1] if self.tracker.initialized else 0.0

    def is_stale(self) -> bool:
        return (
            self.last_obs_time is not None
            and self.now - self.last_obs_time > timedelta(minutes=StateConfig.stale_after_min)
        )

    # ---------------------------------------------------------------- inputs
    def _update_rain(self, rain_mm_h: Optional[float]) -> None:
        if rain_mm_h is None:
            return
        self.rain_mm_h = float(rain_mm_h)
        if not self.rain_log or self.rain_log[-1][0] < self.now:
            self.rain_log.append((self.now, self.rain_mm_h))
        if self.rain_mm_h > StateConfig.rain_threshold_mm_h:
            self.dry_since = None
        elif self.dry_since is None:
            self.dry_since = self.now

    def assimilate(self, obs: Observation, rain_mm_h: Optional[float] = None) -> None:
        self.now = max(self.now, obs.timestamp)
        self._update_rain(rain_mm_h)
        self.tracker.update(obs.timestamp, obs.depth_cm, obs.effective_sigma_cm)
        self.last_obs_time = max(self.last_obs_time or obs.timestamp, obs.timestamp)
        self.n_observations += 1
        eps = StateConfig.rate_eps_cm_min
        r = self.rate_cm_min
        self.falling_streak = self.falling_streak + 1 if r < -eps else 0
        self.rising_streak = self.rising_streak + 1 if r > eps else 0
        if self.depth_cm >= 1.0:
            self.dry_since = None
        elif self.dry_since is None and self.rain_mm_h <= StateConfig.rain_threshold_mm_h:
            self.dry_since = self.now
        self._forecast_cache.clear()
        self._cascade(self._state.on_observation(self))

    def tick(self, now: datetime, rain_mm_h: Optional[float] = None) -> None:
        self.now = max(self.now, now)
        self._update_rain(rain_mm_h)
        self._cascade(self._state.on_tick(self))

    def set_override(self, override: Optional[OperatorOverride]) -> None:
        self.override = override
        self._forecast_cache.clear()
        self._notify(self._state, self._state)

    # ---------------------------------------------------------------- forecasting
    def _history(self, now: datetime, minutes: int = 60):
        """Tracked depth and rain resampled to a 1-minute grid ending at `now`."""
        hist = [tp for tp in self.tracker.history if tp.time >= now - timedelta(minutes=minutes)]
        cur = self.tracker.predict_at(now)
        pts = [((tp.time - now).total_seconds() / 60.0, tp.depth_cm) for tp in hist]
        pts.append((0.0, cur.depth_cm))
        t = np.array([p[0] for p in pts])
        h = np.array([p[1] for p in pts])
        start = int(np.floor(t.min()))
        grid = np.arange(start, 1, dtype=float)
        h_grid = np.interp(grid, t, h)
        if self.rain_log:
            rt = np.array([(rt - now).total_seconds() / 60.0 for rt, _ in self.rain_log])
            rv = np.array([v for _, v in self.rain_log])
            idx = np.searchsorted(rt, grid, side="right") - 1
            r_grid = np.where(idx >= 0, rv[np.clip(idx, 0, None)], 0.0)
        else:
            r_grid = np.zeros_like(grid)
        return h_grid, r_grid, cur.sigma_cm

    def forecast(self, now: Optional[datetime] = None, rain_future: Optional[Sequence[float]] = None) -> Optional[Quantiles]:
        if not self.tracker.initialized:
            return None
        now = now or self.now
        key = (now.replace(second=0, microsecond=0), tuple(np.round(rain_future, 1)) if rain_future is not None else None)
        if key not in self._forecast_cache:
            if rain_future is None:
                rain_future = np.full(self.horizon_min, self.rain_mm_h)
            h, r, sigma = self._history(now)
            self._forecast_cache[key] = self.forecaster.forecast(h, r, rain_future, sigma, self.horizon_min)
        return self._forecast_cache[key]

    # ---------------------------------------------------------------- passability queries
    def passable_minutes(self, vehicle: VehicleProfile, now: Optional[datetime] = None,
                         rain_future=None) -> np.ndarray:
        """Boolean array for minutes 0..horizon: is the segment passable for this vehicle?"""
        now = now or self.now
        n = self.horizon_min + 1
        if self.override is not None and self.override.active(now):
            return np.full(n, self.override.mode == "open")
        if self._state.blocks(vehicle):
            return np.zeros(n, dtype=bool)
        q = self.forecast(now, rain_future)
        if q is None:
            return np.ones(n, dtype=bool)  # never observed water here
        return q.q90 < vehicle.threshold_cm

    def time_to_passable(self, vehicle: VehicleProfile, now: Optional[datetime] = None,
                         rain_future=None) -> TimeToPassable:
        """t*_v = min{ t >= now : q90(h(t)) < d_v - m }"""
        now = now or self.now
        if self.override is not None and self.override.active(now):
            if self.override.mode == "open":
                return TimeToPassable("now", 0.0, None, "operator override: open")
            return TimeToPassable("blocked", None, None, f"closed by {self.override.set_by}")
        if self._state.blocks(vehicle):
            return TimeToPassable("blocked", None, None, "no recent data (fail-safe for low vehicles)")
        q = self.forecast(now, rain_future)
        if q is None:
            return TimeToPassable("now", 0.0, 0.0)
        ok = np.where(q.q90 < vehicle.threshold_cm)[0]
        if len(ok) == 0:
            return TimeToPassable("beyond_horizon", None, float(q.q90[0]))
        if ok[0] == 0:
            return TimeToPassable("now", 0.0, float(q.q90[0]))
        return TimeToPassable("in", float(q.minutes[ok[0]]), float(q.q90[0]))

    def snapshot(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "name": self.name,
            "state": self._state.name,
            "depth_cm": round(self.depth_cm, 1),
            "sigma_cm": round(self.sigma_cm, 1),
            "rate_cm_min": round(float(self.rate_cm_min), 3),
            "rain_mm_h": self.rain_mm_h,
            "last_obs_time": self.last_obs_time.isoformat() if self.last_obs_time else None,
            "override": self.override.mode if self.override and self.override.active(self.now) else None,
        }

    def __repr__(self) -> str:
        return f"RoadSegment({self.segment_id!r}, state={self._state.name}, h={self.depth_cm:.1f}cm)"
