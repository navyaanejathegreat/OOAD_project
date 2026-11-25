from datetime import timedelta

import pytest

from jalmarg.domain.observation import CameraObservation, CitizenReport
from jalmarg.domain.observers import AlertService, RouteCacheInvalidator
from jalmarg.domain.segment import OperatorOverride, RoadSegment
from jalmarg.domain.states import DRY, RECEDING, RISING, UNKNOWN, WET
from jalmarg.domain.vehicles import Bus, Car, TwoWheeler, VehicleProfileFactory, Z90


def obs(t, h, sigma=1.0, sid="s1"):
    return CameraObservation(sid, t, h, sigma, 1.0, camera_id="c1")


# ---------------------------------------------------------------- vehicles
def test_factory_creates_all_six_classes():
    f = VehicleProfileFactory()
    keys = {v.key for v in f.all()}
    assert keys == {"pedestrian", "two_wheeler", "auto", "car", "suv", "bus"}


def test_factory_overrides_and_unknown_key():
    f = VehicleProfileFactory({"car": 30.0})
    assert f.create("car").safe_depth_cm == 30.0
    with pytest.raises(ValueError):
        f.create("boat")


def test_passability_uses_pessimistic_quantile():
    v = TwoWheeler()  # 20 cm, margin 2 -> threshold 18
    assert v.is_passable(15.0, 1.0)
    assert not v.is_passable(15.0, 3.0 / Z90 + 0.01)  # q90 just over 18
    assert not v.is_passable(18.0, 0.0)


def test_low_vehicle_flag():
    assert TwoWheeler().is_low and Car().is_low
    assert not Bus().is_low


def test_less_reliable_source_has_larger_sigma(t0):
    cam = CameraObservation("s", t0, 10, 2.0, 1.0)
    rep = CitizenReport("s", t0, 10, 2.0, reliability=0.25)
    assert rep.effective_sigma_cm == pytest.approx(4.0)
    assert cam.effective_sigma_cm == pytest.approx(2.0)


# ---------------------------------------------------------------- states
def feed(seg, t0, depths, step=2):
    for i, h in enumerate(depths):
        seg.assimilate(obs(t0 + timedelta(minutes=i * step), h))


def test_dry_to_wet_on_rain(t0):
    seg = RoadSegment("s1")
    seg.tick(t0, rain_mm_h=20)
    assert seg.state == WET


def test_wet_to_rising_then_receding_then_wet(t0):
    seg = RoadSegment("s1")
    feed(seg, t0, [2, 6, 10, 14, 18, 20])
    assert seg.state == RISING
    feed(seg, t0 + timedelta(minutes=12), [18, 15, 12, 10, 8, 6])
    assert seg.state == RECEDING
    feed(seg, t0 + timedelta(minutes=24), [4, 3, 2, 1.5, 1, 1])
    assert seg.state == WET


def test_receding_back_to_rising_when_rain_resumes(t0):
    seg = RoadSegment("s1")
    feed(seg, t0, [5, 10, 15, 20, 18, 16, 14, 12, 10])
    assert seg.state == RECEDING
    seg.tick(t0 + timedelta(minutes=19), rain_mm_h=30)
    assert seg.state == RISING


def test_wet_to_dry_after_30_dry_minutes(t0):
    seg = RoadSegment("s1")
    seg.tick(t0, rain_mm_h=10)
    seg.assimilate(obs(t0, 0.5))
    seg.tick(t0 + timedelta(minutes=1), rain_mm_h=0)
    assert seg.state == WET
    for m in range(2, 40, 5):
        seg.assimilate(obs(t0 + timedelta(minutes=m), 0.0))
    assert seg.state == DRY


def test_stale_goes_unknown_and_recovers(t0):
    seg = RoadSegment("s1")
    feed(seg, t0, [10, 12, 14])
    seg.tick(t0 + timedelta(minutes=30))
    assert seg.state == UNKNOWN
    seg.assimilate(obs(t0 + timedelta(minutes=31), 3.0, 0.5))
    assert seg.state in (WET, RECEDING, RISING)
    assert seg.state != UNKNOWN


# ---------------------------------------------------------------- observers
def test_observers_notified_on_state_change(t0):
    seg = RoadSegment("s1", "Underpass")
    alerts, cache = AlertService(hotspot_depth_cm=5), RouteCacheInvalidator()
    seg.attach(alerts)
    seg.attach(cache)
    cache.put("route-a", "advice", ["s1"])
    cache.put("route-b", "advice", ["other"])
    feed(seg, t0, [2, 8, 12])
    assert any(a.new_state == "Rising" for a in alerts.alerts)
    assert cache.get("route-a") is None and cache.get("route-b") == "advice"


def test_subscribed_user_gets_alerts(t0):
    seg = RoadSegment("s1")
    alerts = AlertService()
    seg.attach(alerts)
    alerts.subscribe("u1", ["s1"])
    feed(seg, t0, [2, 8, 20])
    assert alerts.for_user("u1")
    assert alerts.for_user("nobody") == []


def test_detach_stops_notifications(t0):
    seg = RoadSegment("s1")
    alerts = AlertService()
    seg.attach(alerts)
    seg.detach(alerts)
    feed(seg, t0, [2, 8, 20])
    assert alerts.alerts == []


# ---------------------------------------------------------------- passability
def draining_segment(t0):
    seg = RoadSegment("s1")
    # rise then a steady drain from ~30 cm
    feed(seg, t0, [10, 20, 30, 30, 28, 26, 24, 22, 20.5], step=2)
    return seg, t0 + timedelta(minutes=16)


def test_time_to_passable_depends_on_vehicle(t0):
    seg, now = draining_segment(t0)
    bus = seg.time_to_passable(Bus(), now)
    car = seg.time_to_passable(Car(), now)
    scooter = seg.time_to_passable(TwoWheeler(), now)
    assert bus.status == "now"
    assert scooter.status in ("in", "beyond_horizon")
    if scooter.status == "in" and car.status == "in":
        assert scooter.minutes >= car.minutes


def test_time_to_passable_is_first_minute_q90_below_threshold(t0):
    seg, now = draining_segment(t0)
    v = TwoWheeler()
    ttp = seg.time_to_passable(v, now)
    q = seg.forecast(now)
    if ttp.status == "in":
        i = int(ttp.minutes)
        assert q.q90[i] < v.threshold_cm
        assert all(q.q90[:i] >= v.threshold_cm)


def test_unknown_blocks_low_vehicles_only(t0):
    seg = RoadSegment("s1")
    feed(seg, t0, [3, 3, 3])
    later = t0 + timedelta(minutes=40)
    seg.tick(later)
    assert seg.state == UNKNOWN
    assert seg.time_to_passable(TwoWheeler(), later).status == "blocked"
    assert not seg.passable_minutes(TwoWheeler(), later).any()
    assert seg.time_to_passable(Bus(), later).status == "now"


def test_operator_override_beats_model(t0):
    seg, now = draining_segment(t0)
    seg.set_override(OperatorOverride("closed", "ward-7"))
    assert seg.time_to_passable(Bus(), now).status == "blocked"
    seg.set_override(OperatorOverride("open", "ward-7"))
    assert seg.time_to_passable(TwoWheeler(), now).status == "now"
    seg.set_override(OperatorOverride("closed", until=now - timedelta(minutes=1)))
    assert seg.time_to_passable(Bus(), now).status == "now"  # expired


def test_never_observed_segment_is_passable(t0):
    seg = RoadSegment("s1")
    assert seg.time_to_passable(TwoWheeler(), t0).status == "now"
