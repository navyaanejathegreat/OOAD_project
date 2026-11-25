"""RoutePlanner: safe routes and wait-or-detour advice for one vehicle."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import networkx as nx

from ..domain.segment import RoadSegment
from ..domain.vehicles import VehicleProfile
from .graph import RoadGraph
from .td_dijkstra import Availability, Route, earliest_arrival

DISCLAIMER = ("Advisory estimate only. Use your own judgement and follow police and municipal "
              "instructions. Fast currents, open manholes and road damage cannot be seen in water depth.")


@dataclass
class Advice:
    decision: str                     # "go_now" | "wait" | "no_safe_route"
    message: str
    vehicle: str
    t0: datetime
    recommended: Optional[Route]
    go_now: Optional[Route]
    wait_then_go: Optional[Route]
    naive: Optional[Route]
    naive_unsafe_segments: List[str] = field(default_factory=list)
    risk: str = "low"
    disclaimer: str = DISCLAIMER

    @property
    def segments_considered(self) -> List[str]:
        segs = set()
        for r in (self.go_now, self.wait_then_go, self.naive):
            if r:
                segs.update(r.segments)
        return sorted(segs)

    def to_dict(self, graph: Optional[RoadGraph] = None) -> dict:
        def route(r: Optional[Route]):
            if r is None:
                return None
            d = {
                "nodes": [str(n) for n in r.nodes],
                "eta": (self.t0 + timedelta(seconds=r.arrive_s)).isoformat(timespec="minutes"),
                "travel_min": round(r.travel_s / 60, 1),
                "wait_min": round(r.wait_s / 60, 1),
                "length_km": round(r.length_m / 1000, 2),
                "segments": r.segments,
            }
            if graph is not None:
                d["coords"] = [graph.coords(n) for n in r.nodes]
            return d

        return {
            "decision": self.decision, "message": self.message, "vehicle": self.vehicle, "risk": self.risk,
            "t0": self.t0.isoformat(timespec="minutes"),
            "recommended": route(self.recommended), "go_now": route(self.go_now),
            "wait_then_go": route(self.wait_then_go), "naive": route(self.naive),
            "naive_unsafe_segments": self.naive_unsafe_segments, "disclaimer": self.disclaimer,
        }


class RoutePlanner:
    def __init__(self, graph: RoadGraph, segments: Dict[str, RoadSegment], min_gain_s: float = 60.0):
        self.graph = graph
        self.segments = segments
        self.min_gain_s = min_gain_s

    def availability(self, vehicle: VehicleProfile, t0: datetime, rain_future=None) -> Dict[str, Availability]:
        return {sid: Availability(seg.passable_minutes(vehicle, t0, rain_future))
                for sid, seg in self.segments.items()}

    def plan(self, origin, dest, vehicle: VehicleProfile, t0: datetime, allow_wait: bool = True,
             availability=None) -> Optional[Route]:
        av = availability if availability is not None else self.availability(vehicle, t0)
        return earliest_arrival(self.graph.G, origin, dest, 0.0, av, allow_wait, vehicle.speed_factor)

    def naive_route(self, origin, dest, vehicle: VehicleProfile) -> Optional[Route]:
        """What a water-blind router would do."""
        return earliest_arrival(self.graph.G, origin, dest, 0.0, {}, True, vehicle.speed_factor)

    def _risk(self, route: Optional[Route], vehicle: VehicleProfile, t0: datetime) -> str:
        if route is None:
            return "n/a"
        worst = 0.0
        for leg in route.legs:
            seg = self.segments.get(leg.segment_id) if leg.segment_id else None
            if seg is None or seg.tracker.initialized is False:
                continue
            q = seg.forecast(t0)
            if q is None:
                continue
            worst = max(worst, q.at(int(leg.depart_s // 60))[2] / max(vehicle.safe_depth_cm, 1e-6))
        return "low" if worst < 0.5 else "medium" if worst < 0.85 else "high"

    def wait_or_detour(self, origin, dest, vehicle: VehicleProfile, t0: datetime) -> Advice:
        av = self.availability(vehicle, t0)
        go_now = self.plan(origin, dest, vehicle, t0, allow_wait=False, availability=av)
        wait = self.plan(origin, dest, vehicle, t0, allow_wait=True, availability=av)
        naive = self.naive_route(origin, dest, vehicle)
        unsafe = []
        if naive is not None:
            unsafe = [l.segment_id for l in naive.legs
                      if l.segment_id in av and not av[l.segment_id].is_open(l.depart_s, l.arrive_s - l.depart_s)]

        label = vehicle.label
        if go_now is None and wait is None:
            return Advice("no_safe_route",
                          f"No route is expected to be passable for {label} within the forecast horizon. "
                          f"Stay where you are or wait at a safe, dry place; check again later.",
                          vehicle.key, t0, None, None, None, naive, unsafe, "n/a")

        use_wait = (
            wait is not None and wait.wait_s > 0
            and (go_now is None or wait.arrive_s < go_now.arrive_s - self.min_gain_s)
        )
        if use_wait:
            first_wait = next(l for l in wait.legs if l.wait_s > 0)
            seg = self.segments.get(first_wait.segment_id)
            where = seg.name if seg else first_wait.name
            msg = (f"Wait ~{max(1, round(wait.wait_s / 60))} min near {self.graph.name(first_wait.u)}, then take {where}: "
                   f"likely passable for {label} by then (estimate). Arrive in ~{wait.travel_s / 60:.0f} min")
            if go_now is not None:
                msg += f", vs ~{go_now.travel_s / 60:.0f} min by detouring now."
            else:
                msg += "; no passable detour right now."
            return Advice("wait", msg, vehicle.key, t0, wait, go_now, wait, naive, unsafe,
                          self._risk(wait, vehicle, t0))

        extra = (go_now.travel_s - naive.travel_s) / 60 if naive else 0.0
        if unsafe and extra > 0.5:
            names = ", ".join(self.segments[s].name for s in dict.fromkeys(unsafe) if s in self.segments)
            msg = (f"Go now via the detour (+{extra:.0f} min, ~{go_now.travel_s / 60:.0f} min total). "
                   f"Avoid {names}: likely not passable for {label}.")
        else:
            msg = f"Go now: route expected to be passable for {label} (~{go_now.travel_s / 60:.0f} min, estimate)."
        if wait is not None and wait.wait_s > 0:
            msg += f" Waiting would save under {self.min_gain_s / 60:.0f} min."
        return Advice("go_now", msg, vehicle.key, t0, go_now, go_now, wait, naive, unsafe,
                      self._risk(go_now, vehicle, t0))
