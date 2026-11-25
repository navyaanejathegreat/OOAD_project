import math

import cv2
import numpy as np
import pytest

from jalmarg.perception.depth_fusion import ReferenceMeasurement, fuse, fuse_values
from jalmarg.perception.estimators import (
    EstimatorFactory, FallbackChainEstimator, LearnedDepthEstimator, ReferenceObjectEstimator,
)
from jalmarg.perception.segmenter import Detection, FrameAnalysis
from jalmarg.perception.wheel_geometry import measure_wheel


def test_depth_and_sigma_formula():
    m = ReferenceMeasurement("car_wheel", 0.5, 0.05, 60.0, 4.0)
    assert m.depth_cm == 30.0
    assert m.sigma_cm == pytest.approx(math.sqrt((60 * 0.05) ** 2 + (0.5 * 4.0) ** 2))


def test_fusion_inverse_variance():
    h, s = fuse_values([10.0, 20.0], [1.0, 1.0])
    assert h == pytest.approx(15.0) and s == pytest.approx(1 / math.sqrt(2))
    h, _ = fuse_values([10.0, 20.0], [1.0, 10.0])
    assert h < 11.0  # tight estimate dominates


def test_fusion_reduces_uncertainty():
    a = ReferenceMeasurement.from_catalogue("car_wheel", 0.4, 0.04)
    b = ReferenceMeasurement.from_catalogue("two_wheeler_wheel", 0.5, 0.04)
    _, s = fuse([a, b])
    assert s < min(a.sigma_cm, b.sigma_cm)
    assert fuse([]) is None


def synthetic_wheel(frac, H=200, W=200, r=60):
    """Circle wheel of radius r centred at (100,100); water fills from the bottom up to `frac` of the wheel."""
    wheel = np.zeros((H, W), np.uint8)
    cv2.circle(wheel, (100, 100), r, 1, -1)
    water = np.zeros((H, W), np.uint8)
    bottom = 100 + r
    waterline = int(round(bottom - frac * 2 * r))
    water[waterline:, :] = 1
    return wheel, water


@pytest.mark.parametrize("frac", [0.2, 0.35, 0.5, 0.7])
def test_wheel_submerged_fraction(frac):
    wheel, water = synthetic_wheel(frac)
    m = measure_wheel(wheel, water)
    assert m is not None
    assert m.submerged_fraction == pytest.approx(frac, abs=0.03)


def test_dry_wheel_has_no_waterline():
    wheel, _ = synthetic_wheel(0.3)
    m = measure_wheel(wheel, np.zeros_like(wheel))
    assert m.waterline_y is None and m.submerged_fraction == 0.0


def test_ragged_waterline_increases_sigma():
    wheel, water = synthetic_wheel(0.4)
    rough = water.copy()
    rng = np.random.default_rng(0)
    for x in range(200):
        off = int(rng.integers(-8, 9))
        rough[:, x] = 0
        rough[max(0, 136 + off):, x] = 1
    a, b = measure_wheel(wheel, water), measure_wheel(wheel, rough)
    assert b.sigma_fraction > a.sigma_fraction


def test_reference_estimator_from_analysis(t0):
    wheel, water = synthetic_wheel(0.5)
    analysis = FrameAnalysis(water, [Detection("car_wheel", 0.9, wheel, (40, 40, 160, 160))])
    obs = ReferenceObjectEstimator().estimate(None, "s1", t0, "cam", analysis)
    assert obs.depth_cm == pytest.approx(30.0, abs=2.5)
    assert obs.n_references == 1 and obs.camera_id == "cam"


def test_no_water_means_zero_depth(t0):
    analysis = FrameAnalysis(np.zeros((50, 50), np.uint8), [])
    obs = ReferenceObjectEstimator().estimate(None, "s1", t0, analysis=analysis)
    assert obs.depth_cm == 0.0


def test_fallback_used_when_no_references(t0):
    water = np.zeros((100, 100), np.uint8)
    water[50:, :] = 1
    analysis = FrameAnalysis(water, [])
    chain = FallbackChainEstimator([ReferenceObjectEstimator(), LearnedDepthEstimator()])
    obs = chain.estimate(None, "s1", t0, analysis=analysis)
    assert obs.estimator == "learned" and obs.sigma_cm >= 10


def test_estimator_factory():
    assert isinstance(EstimatorFactory.create("reference_object"), ReferenceObjectEstimator)
    assert isinstance(EstimatorFactory.create("default"), FallbackChainEstimator)
    with pytest.raises(ValueError):
        EstimatorFactory.create("magic")
