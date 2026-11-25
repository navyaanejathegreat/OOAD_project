from datetime import timedelta

import cv2
import numpy as np
import pytest

from jalmarg.adapters.camera_source import PhotoAdapter, VideoFileAdapter
from jalmarg.adapters.weather import StaticRainfall
from jalmarg.domain.observation import CameraObservation, CitizenReport
from jalmarg.domain.segment import RoadSegment
from jalmarg.repositories.sqlite_repos import (
    InMemorySegmentRepository, SqliteObservationRepository, SqliteSegmentRepository,
)
from jalmarg.validation.report_chain import ValidationContext, Verdict, build_default_chain


def sharp_image():
    rng = np.random.default_rng(0)
    return (rng.integers(0, 255, (64, 64, 3))).astype(np.uint8)


def report(t, h=10.0, who="u1", img_hash="h1", image=None, sigma=4.0, seg="s1"):
    return CitizenReport(seg, t, h, sigma, reporter_id=who, image_hash=img_hash, image=image)


def ctx(t, tracked=None):
    return ValidationContext(now=t, known_segments={"s1"}, tracked_depth=lambda sid: tracked)


def test_good_report_accepted(t0):
    head, _ = build_default_chain()
    res = head.handle(report(t0, image=sharp_image()), ctx(t0))
    assert res.verdict == Verdict.ACCEPTED
    assert res.checked_by == ["format", "duplicate", "quality", "confidence", "moderation"]


@pytest.mark.parametrize("bad,why", [
    (dict(seg="nowhere"), "unknown"),
    (dict(h=500.0), "range"),
    (dict(img_hash=None), "photo"),
])
def test_format_rejections(t0, bad, why):
    head, _ = build_default_chain()
    res = head.handle(report(t0, **bad), ctx(t0))
    assert res.verdict == Verdict.REJECTED and why in res.reasons[0]


def test_old_report_rejected(t0):
    head, _ = build_default_chain()
    res = head.handle(report(t0 - timedelta(hours=3)), ctx(t0))
    assert res.verdict == Verdict.REJECTED


def test_duplicates_rejected(t0):
    head, _ = build_default_chain()
    assert head.handle(report(t0), ctx(t0)).verdict == Verdict.ACCEPTED
    assert head.handle(report(t0, who="u2"), ctx(t0)).verdict == Verdict.REJECTED  # same photo
    assert head.handle(report(t0, img_hash="h2"), ctx(t0)).verdict == Verdict.REJECTED  # same reporter too soon


def test_dark_or_blurry_photo_rejected(t0):
    head, _ = build_default_chain()
    dark = np.zeros((64, 64, 3), np.uint8)
    assert head.handle(report(t0, image=dark), ctx(t0)).verdict == Verdict.REJECTED
    flat = np.full((64, 64, 3), 128, np.uint8)
    res = head.handle(report(t0, image=flat, img_hash="h3", who="u3"), ctx(t0))
    assert res.verdict == Verdict.REJECTED and "blurry" in res.reasons[0]


def test_inconsistent_report_goes_to_moderation(t0):
    head, mod = build_default_chain()
    r = report(t0, h=40.0)
    res = head.handle(r, ctx(t0, tracked=(10.0, 1.0)))
    assert res.verdict == Verdict.PENDING
    assert r.obs_id in mod.queue
    approved = mod.approve(r.obs_id)
    assert approved.verified and approved.reliability >= 0.8 and not mod.queue


def test_photo_adapter_and_video_adapter(tmp_path, t0):
    frames = list(PhotoAdapter(sharp_image(), "phone", "s1", t0).frames())
    assert len(frames) == 1 and frames[0].segment_id == "s1"

    path = str(tmp_path / "clip.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    for _ in range(50):
        vw.write(sharp_image()[:32, :32])
    vw.release()
    fr = list(VideoFileAdapter(path, "cam", "s1", start_time=t0, every_s=1.0).frames())
    assert len(fr) == 5
    assert fr[1].timestamp - fr[0].timestamp == timedelta(seconds=1)


def test_static_rainfall(t0):
    w = StaticRainfall(lambda t: 30.0 if t < t0 + timedelta(minutes=5) else 0.0)
    r = w.rainfall(0, 0, t0, 10)
    assert list(r[:5]) == [30.0] * 5 and r[5:].sum() == 0


def test_repositories(t0):
    segs = SqliteSegmentRepository()
    s = RoadSegment("s1", "Underpass")
    segs.add(s)
    s.assimilate(CameraObservation("s1", t0, 12.0, 1.0))
    segs.save(s)
    assert segs.get("s1") is s and len(segs.all()) == 1
    assert segs.history("s1")[-1][2] == pytest.approx(12.0)

    obs = SqliteObservationRepository()
    obs.add(CameraObservation("s1", t0, 12.0, 1.0, camera_id="c"))
    obs.add(CitizenReport("s1", t0 + timedelta(minutes=5), 14.0, 4.0, reporter_id="u"))
    rows = obs.for_segment("s1")
    assert [r["kind"] for r in rows] == ["camera", "citizen"]
    assert len(obs.for_segment("s1", since=t0 + timedelta(minutes=1))) == 1

    mem = InMemorySegmentRepository()
    mem.add(s)
    assert mem.get("nope") is None
