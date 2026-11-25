"""JalMargService: the single facade the API and UI talk to."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .adapters.camera_source import Frame
from .adapters.weather import StaticRainfall, WeatherService
from .domain.observation import CitizenReport, Observation
from .domain.observers import AlertService, RouteCacheInvalidator
from .domain.segment import OperatorOverride, RoadSegment
from .domain.vehicles import VehicleProfile, VehicleProfileFactory
from .perception.estimators import DepthEstimator
from .repositories.sqlite_repos import (
    InMemoryObservationRepository, InMemorySegmentRepository, ObservationRepository, SegmentRepository,
)
from .routing.graph import RoadGraph
from .routing.planner import Advice, RoutePlanner
from .temporal.forecaster import RecessionForecaster, default_forecaster
from .validation.report_chain import ValidationContext, ValidationResult, Verdict, build_default_chain


class JalMargService:
    def __init__(
        self,
        graph: RoadGraph,
        segment_repo: Optional[SegmentRepository] = None,
        observation_repo: Optional[ObservationRepository] = None,
        vehicle_factory: Optional[VehicleProfileFactory] = None,
        estimator: Optional[DepthEstimator] = None,
        weather: Optional[WeatherService] = None,
        forecaster: Optional[RecessionForecaster] = None,
    ):
        self.graph = graph
        self.segments = segment_repo or InMemorySegmentRepository()
        self.observations = observation_repo or InMemoryObservationRepository()
        self.vehicles = vehicle_factory or VehicleProfileFactory()
        self.estimator = estimator
        self.weather = weather or StaticRainfall()
        self.forecaster = forecaster or default_forecaster()
        self.alerts = AlertService()
        self.route_cache = RouteCacheInvalidator()
        self.report_chain, self.moderation = build_default_chain()
        self.cameras: Dict[str, str] = {}         # camera_id -> segment_id
        self.user_vehicles: Dict[str, str] = {}   # user_id -> vehicle key
        self.now: Optional[datetime] = None

    # ------------------------------------------------------------ setup / admin (UC-11)
    def add_segment(self, segment_id: str, name: str = "") -> RoadSegment:
        if not self.graph.edges_of_segment(segment_id):
            raise ValueError(f"No graph edges are tagged with segment '{segment_id}'")
        seg = RoadSegment(segment_id, name, forecaster=self.forecaster)
        seg.attach(self.alerts)
        seg.attach(self.route_cache)
        self.segments.add(seg)
        return seg

    def register_camera(self, camera_id: str, segment_id: str) -> None:
        if self.segments.get(segment_id) is None:
            raise ValueError(f"Unknown segment '{segment_id}'")
        self.cameras[camera_id] = segment_id

    def set_threshold(self, vehicle_key: str, safe_depth_cm: float) -> None:
        self.vehicles.overrides[vehicle_key] = safe_depth_cm
        self.route_cache.clear()

    # ------------------------------------------------------------ users (UC-01..03)
    def set_vehicle_profile(self, user_id: str, vehicle_key: str) -> VehicleProfile:
        v = self.vehicles.create(vehicle_key)  # validates the key
        self.user_vehicles[user_id] = vehicle_key
        return v

    def vehicle_for(self, user_or_vehicle: str) -> VehicleProfile:
        return self.vehicles.create(self.user_vehicles.get(user_or_vehicle, user_or_vehicle))

    def _segment_map(self) -> Dict[str, RoadSegment]:
        return {s.segment_id: s for s in self.segments.all()}

    def plan_route(self, user_or_vehicle: str, origin, dest, t0: Optional[datetime] = None,
                   user_id: Optional[str] = None) -> Advice:
        t0 = t0 or self.now or datetime.now()
        vehicle = self.vehicle_for(user_or_vehicle)
        n_obs = sum(s.n_observations for s in self.segments.all())
        key = f"{vehicle.key}|{vehicle.safe_depth_cm}|{origin}|{dest}|{t0:%Y%m%d%H%M}|{n_obs}"
        cached = self.route_cache.get(key)
        if cached is not None:
            return cached
        planner = RoutePlanner(self.graph, self._segment_map())
        advice = planner.wait_or_detour(origin, dest, vehicle, t0)
        self.route_cache.put(key, advice, advice.segments_considered)
        if user_id:
            self.alerts.subscribe(user_id, advice.segments_considered)  # UC-01 postcondition
        return advice

    def time_to_passable(self, segment_id: str, vehicle_key: str, now: Optional[datetime] = None):
        seg = self._require(segment_id)
        return seg.time_to_passable(self.vehicles.create(vehicle_key), now or self.now)

    # ------------------------------------------------------------ observations (UC-05/06/07)
    def _rain_at(self, segment_id: str, t: datetime) -> float:
        loc = self.graph.segment_location(segment_id) or (0.0, 0.0)
        try:
            return self.weather.now(loc[0], loc[1], t)
        except Exception:
            return 0.0

    def ingest_observation(self, obs: Observation) -> RoadSegment:
        seg = self._require(obs.segment_id)
        self.now = max(self.now or obs.timestamp, obs.timestamp)
        seg.assimilate(obs, self._rain_at(obs.segment_id, obs.timestamp))
        self.observations.add(obs)
        self.segments.save(seg)
        return seg

    def ingest_frame(self, frame: Frame):
        """UC-06: camera frame -> depth estimate -> tracker."""
        if self.estimator is None:
            raise RuntimeError("No depth estimator configured")
        segment_id = self.cameras.get(frame.camera_id, frame.segment_id)
        obs = self.estimator.estimate(frame.image, segment_id, frame.timestamp, frame.camera_id)
        if obs is not None:
            self.ingest_observation(obs)
        return obs

    def submit_report(self, report: CitizenReport) -> ValidationResult:
        """UC-05: validate through the chain; accepted reports update the tracker."""
        now = self.now or report.timestamp

        def tracked(sid):
            s = self.segments.get(sid)
            return (s.depth_cm, s.sigma_cm) if s is not None and s.tracker.initialized else None

        ctx = ValidationContext(now=now, known_segments={s.segment_id for s in self.segments.all()},
                                tracked_depth=tracked)
        result = self.report_chain.handle(report, ctx)
        if result.verdict == Verdict.ACCEPTED:
            report.image = None  # privacy: never keep pixels
            self.ingest_observation(report)
        return result

    def pending_reports(self) -> List[CitizenReport]:
        return list(self.moderation.queue.values())

    def moderate_report(self, obs_id: str, approve: bool) -> Optional[CitizenReport]:
        """UC-09: municipal operator verifies a report."""
        if not approve:
            return self.moderation.reject(obs_id)
        r = self.moderation.approve(obs_id)
        if r is not None:
            r.image = None
            self.ingest_observation(r)
        return r

    # ------------------------------------------------------------ operator (UC-12) and clock
    def override_segment(self, segment_id: str, mode: Optional[str], by: str = "operator",
                         minutes: Optional[float] = None, reason: str = "") -> None:
        seg = self._require(segment_id)
        if mode is None:
            seg.set_override(None)
            return
        if mode not in ("open", "closed"):
            raise ValueError("mode must be 'open', 'closed' or None")
        now = self.now or datetime.now()
        until = now + timedelta(minutes=minutes) if minutes else None
        seg.set_override(OperatorOverride(mode, by, until, reason))

    def tick(self, now: datetime) -> None:
        self.now = now
        for seg in self.segments.all():
            seg.tick(now, self._rain_at(seg.segment_id, now))
            self.segments.save(seg)

    # ------------------------------------------------------------ views (UC-04, UC-10)
    def segment_snapshots(self, vehicle_key: Optional[str] = None) -> List[dict]:
        out = []
        for seg in self.segments.all():
            snap = seg.snapshot()
            if vehicle_key:
                ttp = seg.time_to_passable(self.vehicles.create(vehicle_key), self.now or seg.now)
                snap["time_to_passable"] = {"status": ttp.status, "minutes": ttp.minutes,
                                            "text": ttp.describe(self.vehicles.create(vehicle_key).label)}
            out.append(snap)
        return out

    def _require(self, segment_id: str) -> RoadSegment:
        seg = self.segments.get(segment_id)
        if seg is None:
            raise KeyError(f"Unknown segment '{segment_id}'")
        return seg
