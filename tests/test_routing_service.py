from datetime import timedelta

import networkx as nx
import numpy as np
import pytest

from jalmarg.demo.scenario import START, replay
from jalmarg.domain.observation import CameraObservation, CitizenReport
from jalmarg.routing.graph import build_demo_graph
from jalmarg.routing.td_dijkstra import Availability, earliest_arrival
from jalmarg.service import JalMargService
from jalmarg.temporal.forecaster import PhysicsForecaster
from jalmarg.validation.report_chain import Verdict


def line_graph():
    """a -> b (flooded edge 'f', 60 s) -> c, plus a long detour a -> d -> c (600 s)."""
    G = nx.DiGraph()
    G.add_edge("a", "b", time_s=60, segment_id=None)
    G.add_edge("b", "c", time_s=60, segment_id="f")
    G.add_edge("a", "d", time_s=300, segment_id=None)
    G.add_edge("d", "c", time_s=300, segment_id=None)
    return G


def closed_for(minutes, horizon=60):
    arr = np.ones(horizon + 1, dtype=bool)
    arr[:minutes] = False
    return Availability(arr)


def test_availability_next_open():
    av = closed_for(5)
    assert av.next_open(0) == 300
    assert av.next_open(400) == 400
    assert Availability(np.zeros(10, bool)).next_open(0) == float("inf")


def test_wait_beats_long_detour():
    r = earliest_arrival(line_graph(), "a", "c", 0, {"f": closed_for(3)}, allow_wait=True)
    # reach b at 60 s, edge opens at 180 s, arrive at 240 s (detour would be 600 s)
    assert r.nodes == ["a", "b", "c"] and r.wait_s == 120 and r.arrive_s == 240


def test_detour_beats_long_wait():
    r = earliest_arrival(line_graph(), "a", "c", 0, {"f": closed_for(30)}, allow_wait=True)
    assert r.nodes == ["a", "d", "c"] and r.wait_s == 0


def test_no_wait_mode_avoids_closed_edge():
    r = earliest_arrival(line_graph(), "a", "c", 0, {"f": closed_for(3)}, allow_wait=False)
    assert r.nodes == ["a", "d", "c"]


def test_unreachable_returns_none():
    G = line_graph()
    G.remove_edge("d", "c")
    assert earliest_arrival(G, "a", "c", 0, {"f": Availability(np.zeros(61, bool))}) is None


def test_fifo_later_start_never_arrives_earlier():
    av = {"f": closed_for(10)}
    arrivals = [earliest_arrival(line_graph(), "a", "c", t, av).arrive_s for t in range(0, 900, 30)]
    assert all(b >= a for a, b in zip(arrivals, arrivals[1:]))


# ---------------------------------------------------------------- service + planner
@pytest.fixture
def svc():
    s = JalMargService(build_demo_graph(), forecaster=PhysicsForecaster())
    for sid in ("underpass", "market_road", "canal_road", "station_road"):
        s.add_segment(sid)
    return s


def flood(svc, sid, t0, depths, step=2):
    for i, h in enumerate(depths):
        svc.ingest_observation(CameraObservation(sid, t0 + timedelta(minutes=i * step), h, 1.0))


def test_dry_city_goes_direct(svc, t0):
    a = svc.plan_route("two_wheeler", "home", "office", t0)
    assert a.decision == "go_now" and "underpass" in a.recommended.segments


def test_deep_underpass_detour_for_scooter_but_not_bus(svc, t0):
    flood(svc, "underpass", t0, [30, 32, 34, 35, 36])
    now = t0 + timedelta(minutes=8)
    scooter = svc.plan_route("two_wheeler", "home", "office", now)
    bus = svc.plan_route("bus", "home", "office", now)
    assert "underpass" not in scooter.recommended.segments
    assert "underpass" in scooter.naive_unsafe_segments
    assert "underpass" in bus.recommended.segments


def test_closed_everywhere_gives_no_safe_route(svc, t0):
    for sid in ("underpass", "market_road", "canal_road"):
        svc.override_segment(sid, "closed")
    a = svc.plan_route("car", "home", "office", t0)
    assert a.decision == "no_safe_route" and a.recommended is None


def test_report_flow_and_moderation(svc, t0):
    flood(svc, "market_road", t0, [10, 10, 10])
    now = t0 + timedelta(minutes=4)
    ok = svc.submit_report(CitizenReport("market_road", now, 11.0, 4.0, reporter_id="u1", image_hash="x"))
    assert ok.verdict == Verdict.ACCEPTED
    odd = CitizenReport("market_road", now, 60.0, 4.0, reporter_id="u2", image_hash="y")
    assert svc.submit_report(odd).verdict == Verdict.PENDING
    assert svc.pending_reports() == [odd]
    before = svc.segments.get("market_road").n_observations
    svc.moderate_report(odd.obs_id, approve=True)
    assert svc.segments.get("market_road").n_observations == before + 1
    assert odd.image is None


def test_set_vehicle_profile_and_subscription(svc, t0):
    svc.set_vehicle_profile("asha", "suv")
    a = svc.plan_route("asha", "home", "office", t0, user_id="asha")
    assert a.vehicle == "suv"
    with pytest.raises(ValueError):
        svc.set_vehicle_profile("asha", "boat")


def test_route_cache_invalidated_by_state_change(svc, t0):
    a1 = svc.plan_route("car", "home", "office", t0)
    assert svc.plan_route("car", "home", "office", t0) is a1
    flood(svc, "underpass", t0, [10, 20, 30])
    assert svc.plan_route("car", "home", "office", t0) is not a1


def test_register_camera_validates_segment(svc):
    svc.register_camera("cam1", "underpass")
    with pytest.raises(ValueError):
        svc.register_camera("cam2", "nowhere")


# ---------------------------------------------------------------- end-to-end demo scenario
def test_demo_scenario_advice_changes_over_time():
    decisions = {}
    for m in (25, 40, 60):
        s = replay(m, forecaster=PhysicsForecaster())
        decisions[m] = s.plan_route("two_wheeler", "home", "office", START + timedelta(minutes=m))
    assert "underpass" not in decisions[25].recommended.segments          # detour at peak
    assert "underpass" in decisions[60].recommended.segments              # open again later


def test_demo_never_routes_scooter_through_unsafe_water():
    """Unsafe-rate check on the demo: whenever the planner sends a two-wheeler through a
    flooded segment, the TRUE depth at that moment is below the safe depth."""
    from jalmarg.demo.scenario import true_depths

    truth = true_depths()
    for m in range(10, 120, 10):
        s = replay(m, forecaster=PhysicsForecaster())
        a = s.plan_route("two_wheeler", "home", "office", START + timedelta(minutes=m))
        if a.recommended is None:
            continue
        for leg in a.recommended.legs:
            if leg.segment_id:
                minute = m + int(leg.depart_s // 60)
                assert truth[leg.segment_id][minute] < 20.0, (m, leg.segment_id)


def test_canal_camera_dropout_goes_unknown():
    s = replay(100, forecaster=PhysicsForecaster())
    assert s.segments.get("canal_road").state.name == "Unknown"
