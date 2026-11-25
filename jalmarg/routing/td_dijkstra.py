"""Time-dependent earliest-arrival Dijkstra with optional waiting.

Each water-affected edge has a per-minute availability array for the user's
vehicle (from the forecast). Arriving at a closed edge either waits until it
opens (wait mode) or is not allowed (go-now mode). Departure time
`next_open(t) >= t` is non-decreasing in t, so the FIFO property holds and
plain Dijkstra on arrival time stays correct.
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

INF = math.inf


class Availability:
    """Per-minute boolean availability, relative to t0. Beyond the horizon the last value persists."""

    def __init__(self, passable: np.ndarray):
        self.passable = np.asarray(passable, dtype=bool)

    def _ok(self, minute: int) -> bool:
        if minute < 0:
            minute = 0
        return bool(self.passable[minute]) if minute < len(self.passable) else bool(self.passable[-1])

    def is_open(self, t_s: float, travel_s: float = 0.0) -> bool:
        """Open at entry and at exit of the edge."""
        return self._ok(int(t_s // 60)) and self._ok(int((t_s + travel_s) // 60))

    def next_open(self, t_s: float, travel_s: float = 0.0) -> float:
        start = int(t_s // 60)
        if self.is_open(t_s, travel_s):
            return t_s
        for m in range(start + 1, len(self.passable) + 1):
            if self.is_open(m * 60.0, travel_s):
                return m * 60.0
        return INF


@dataclass
class Leg:
    u: object
    v: object
    depart_s: float
    arrive_s: float
    wait_s: float
    name: str
    segment_id: Optional[str]


@dataclass
class Route:
    nodes: List
    legs: List[Leg]
    depart_s: float
    arrive_s: float
    length_m: float = 0.0

    @property
    def travel_s(self) -> float:
        return self.arrive_s - self.depart_s

    @property
    def wait_s(self) -> float:
        return sum(l.wait_s for l in self.legs)

    @property
    def segments(self) -> List[str]:
        return [l.segment_id for l in self.legs if l.segment_id]



def earliest_arrival(
    G,
    source,
    target,
    t0_s: float = 0.0,
    availability: Optional[Dict[str, Availability]] = None,
    allow_wait: bool = True,
    time_factor: float = 1.0,
) -> Optional[Route]:
    availability = availability or {}
    best: Dict = {source: t0_s}
    pred: Dict = {}
    counter = itertools.count()
    pq: List[Tuple[float, int, object]] = [(t0_s, next(counter), source)]
    done = set()
    while pq:
        t, _, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        if u == target:
            break
        for v, d in G[u].items():
            if v in done:
                continue
            travel = d["time_s"] * time_factor
            sid = d.get("segment_id")
            dep = t
            if sid is not None and sid in availability:
                av = availability[sid]
                dep = av.next_open(t, travel) if allow_wait else (t if av.is_open(t, travel) else INF)
            if dep == INF:
                continue
            arr = dep + travel
            if arr < best.get(v, INF):
                best[v] = arr
                pred[v] = (u, t, dep, d)
                heapq.heappush(pq, (arr, next(counter), v))
    if target not in done:
        return None
    legs: List[Leg] = []
    nodes = [target]
    v = target
    length = 0.0
    while v != source:
        u, t_at_u, dep, d = pred[v]
        legs.append(Leg(u, v, dep, best[v], dep - t_at_u, d.get("name", ""), d.get("segment_id")))
        length += d.get("length_m", 0.0)
        nodes.append(u)
        v = u
    legs.reverse()
    nodes.reverse()
    return Route(nodes, legs, t0_s, best[target], length)
