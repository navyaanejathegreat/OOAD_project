"""Road graph: a small hand-built demo area in Dehradun, or a real one from OSMnx."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

DEFAULT_SPEED_KMH = 30.0


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class RoadGraph:
    """Directed road graph. Edge attributes: length_m, time_s (free flow), name,
    segment_id (None if the edge is not water-affected)."""

    def __init__(self, G: Optional[nx.DiGraph] = None):
        self.G = G if G is not None else nx.DiGraph()

    # ------------------------------------------------------------ building
    def add_node(self, node, lat: float, lon: float, name: str = "") -> None:
        self.G.add_node(node, lat=lat, lon=lon, name=name or str(node))

    def add_road(self, u, v, name: str = "", segment_id: Optional[str] = None,
                 speed_kmh: float = DEFAULT_SPEED_KMH, length_m: Optional[float] = None,
                 two_way: bool = True) -> None:
        if length_m is None:
            a, b = self.G.nodes[u], self.G.nodes[v]
            length_m = haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
        attrs = dict(length_m=length_m, time_s=length_m / (speed_kmh / 3.6), name=name, segment_id=segment_id)
        self.G.add_edge(u, v, **attrs)
        if two_way:
            self.G.add_edge(v, u, **attrs)

    def tag_segment(self, edges: Iterable[Tuple], segment_id: str) -> None:
        for u, v in edges:
            self.G.edges[u, v]["segment_id"] = segment_id

    # ------------------------------------------------------------ queries
    def coords(self, node) -> Tuple[float, float]:
        d = self.G.nodes[node]
        return d["lat"], d["lon"]

    def name(self, node) -> str:
        return self.G.nodes[node].get("name", str(node))

    def edges_of_segment(self, segment_id: str) -> List[Tuple]:
        return [(u, v) for u, v, d in self.G.edges(data=True) if d.get("segment_id") == segment_id]

    def segment_location(self, segment_id: str) -> Optional[Tuple[float, float]]:
        edges = self.edges_of_segment(segment_id)
        if not edges:
            return None
        (la1, lo1), (la2, lo2) = self.coords(edges[0][0]), self.coords(edges[0][1])
        return (la1 + la2) / 2, (lo1 + lo2) / 2

    def nodes(self) -> List:
        return list(self.G.nodes)


def build_demo_graph() -> RoadGraph:
    """A small area with three ways from Home to Office:
    direct through a railway underpass, south via a market road, or north via a canal road.
    Coordinates are around Dehradun but the roads are illustrative, not surveyed."""
    g = RoadGraph()
    nodes = {
        "home": (30.3200, 78.0000, "Home"),
        "junction_w": (30.3200, 78.0100, "West junction"),
        "junction_e": (30.3200, 78.0200, "East junction"),
        "office": (30.3200, 78.0300, "Office"),
        "north_w": (30.3420, 78.0000, "North-west chowk"),
        "north_e": (30.3420, 78.0300, "North-east chowk"),
        "south_w": (30.3110, 78.0100, "Market west"),
        "south_e": (30.3110, 78.0200, "Market east"),
    }
    for k, (lat, lon, name) in nodes.items():
        g.add_node(k, lat, lon, name)
    g.add_road("home", "junction_w", "Main road (west)")
    g.add_road("junction_w", "junction_e", "Railway underpass", segment_id="underpass")
    g.add_road("junction_e", "office", "Station road", segment_id="station_road")
    g.add_road("home", "north_w", "Ring road (north-west)")
    g.add_road("north_w", "north_e", "Canal road", segment_id="canal_road")
    g.add_road("north_e", "office", "Ring road (north-east)")
    g.add_road("junction_w", "south_w", "Market link (west)")
    g.add_road("south_w", "south_e", "Market road", segment_id="market_road")
    g.add_road("south_e", "junction_e", "Market link (east)")
    return g


def build_osmnx_graph(lat: float, lon: float, dist_m: int = 1500,
                      segment_edges: Optional[Dict[str, List[Tuple]]] = None) -> RoadGraph:
    """Real road graph around (lat, lon). Needs `pip install osmnx` and network access."""
    import osmnx as ox  # lazy, optional

    M = ox.graph_from_point((lat, lon), dist=dist_m, network_type="drive")
    M = ox.add_edge_speeds(M)
    M = ox.add_edge_travel_times(M)
    g = RoadGraph()
    for n, d in M.nodes(data=True):
        g.add_node(n, d["y"], d["x"], str(n))
    for u, v, d in M.edges(data=True):
        if g.G.has_edge(u, v) and g.G.edges[u, v]["time_s"] <= d["travel_time"]:
            continue
        name = d.get("name", "")
        g.G.add_edge(u, v, length_m=d["length"], time_s=d["travel_time"],
                     name=name if isinstance(name, str) else ", ".join(name), segment_id=None)
    for sid, edges in (segment_edges or {}).items():
        g.tag_segment(edges, sid)
    return g
