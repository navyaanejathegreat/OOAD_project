from datetime import timedelta

import numpy as np
import pytest

from jalmarg.temporal.forecaster import (
    GRUForecaster, HybridForecaster, PhysicsForecaster, build_inputs, make_net, pinball_loss,
)
from jalmarg.temporal.kalman import KalmanDepthTracker
from jalmarg.temporal.synth import generate_dataset, make_windows, simulate


def test_kalman_smooths_noise_and_tracks_rate(t0):
    true = 30 - 0.3 * np.arange(40)
    depth_err, rate_err = [], []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        kf = KalmanDepthTracker()
        for i, h in enumerate(true):
            kf.update(t0 + timedelta(minutes=i), h + rng.normal(0, 2), 2.0)
        depth_err.append(kf.x[0] - true[-1])
        rate_err.append(kf.x[1] + 0.3)
    assert np.sqrt(np.mean(np.square(depth_err))) < 1.5   # better than the 2 cm raw noise
    assert np.sqrt(np.mean(np.square(rate_err))) < 0.15


def test_kalman_follows_sudden_drain(t0):
    kf = KalmanDepthTracker()
    for i in range(20):
        kf.update(t0 + timedelta(minutes=i), 30, 1.0)
    for i in range(20, 30):
        kf.update(t0 + timedelta(minutes=i), 30 - (i - 19), 1.0)
    assert kf.x[1] < -0.7


def test_kalman_weights_by_sigma(t0):
    a, b = KalmanDepthTracker(), KalmanDepthTracker()
    for kf in (a, b):
        kf.update(t0, 10.0, 1.0)
    a.update(t0 + timedelta(minutes=1), 20.0, 1.0)
    b.update(t0 + timedelta(minutes=1), 20.0, 10.0)
    assert a.x[0] > b.x[0]


def test_kalman_uncertainty_grows_during_gaps(t0):
    kf = KalmanDepthTracker()
    kf.update(t0, 10, 1)
    kf.update(t0 + timedelta(minutes=1), 10, 1)
    s_now = kf.predict_at(t0 + timedelta(minutes=1)).sigma_cm
    s_later = kf.predict_at(t0 + timedelta(minutes=30)).sigma_cm
    assert s_later > s_now


def test_kalman_never_negative(t0):
    kf = KalmanDepthTracker()
    for i, h in enumerate([5, 3, 1, 0, 0]):
        kf.update(t0 + timedelta(minutes=i), h, 1)
    assert kf.predict_at(t0 + timedelta(minutes=60)).depth_cm >= 0


def test_simulate_drains_exponentially():
    h = simulate(30.0, np.zeros(100), alpha=0.02, k=0.05, h_inf=2.0)
    assert h[0] == 30 and h[-1] == pytest.approx(2 + 28 * (0.95 ** 100), rel=1e-6)
    assert np.all(np.diff(h) <= 0)


def test_rain_raises_depth():
    h = simulate(5.0, np.full(20, 40.0), alpha=0.02, k=0.05, h_inf=0.0)
    assert h[-1] > 5.0


def test_windows_split_by_event():
    series = generate_dataset(10, seed=1)
    w = make_windows(series[:8], 2)
    assert {x.series_idx for x in w} <= set(range(8))
    assert all(len(x.h_future) == 60 and len(x.rain_future) == 60 for x in w)


def test_physics_recovers_drain_rate():
    h = simulate(30.0, np.zeros(60), alpha=0.02, k=0.06, h_inf=1.0)
    p = PhysicsForecaster().fit(h, np.zeros(61))
    assert p.k == pytest.approx(0.06, rel=0.1)


def test_physics_quantiles_ordered_and_widening():
    h = simulate(30.0, np.zeros(40), alpha=0.02, k=0.05, h_inf=0.0)
    q = PhysicsForecaster().forecast(h, np.zeros(41), np.zeros(60), sigma_now=1.0)
    assert len(q.q50) == 61
    assert np.all(q.q10 <= q.q50) and np.all(q.q50 <= q.q90)
    assert (q.q90[-1] - q.q50[-1]) > (q.q90[1] - q.q50[1])
    assert q.q50[-1] < q.q50[0]  # draining


def test_short_history_uses_defaults():
    q = PhysicsForecaster().forecast([12.0], [0.0], None, 1.0, horizon=30)
    assert len(q.minutes) == 31 and q.q50[0] == 12.0


def test_build_inputs_pads_history():
    seq, rf = build_inputs([10, 11, 12], [0, 0, 5], [1, 2], hist_len=60, horizon=60)
    assert seq.shape == (60, 3) and rf.shape == (60,)
    assert seq[:57, 2].sum() == 0 and seq[57:, 2].sum() == 3


def test_pinball_loss_penalises_asymmetrically():
    torch = pytest.importorskip("torch")
    y = torch.zeros(1, 5)
    under = torch.full((1, 3, 5), -1.0)
    over = torch.full((1, 3, 5), 1.0)
    # all quantiles predicting the same value: symmetric overall, but per-quantile differs
    assert pinball_loss(under, y).item() == pytest.approx(pinball_loss(over, y).item())
    pred = torch.stack([torch.full((1, 5), -1.0), torch.zeros(1, 5), torch.full((1, 5), 1.0)], dim=1)
    assert pinball_loss(pred, y).item() < pinball_loss(under, y).item()


def test_untrained_neural_falls_back_to_physics():
    h = simulate(30.0, np.zeros(40), alpha=0.02, k=0.05, h_inf=0.0)
    a = HybridForecaster(weights_path="does-not-exist.pt").forecast(h, np.zeros(41), None)
    b = PhysicsForecaster().forecast(h, np.zeros(41), None)
    assert np.allclose(a.q50, b.q50)


def test_trained_networks_produce_valid_quantiles(tmp_path):
    torch = pytest.importorskip("torch")
    for cls, hybrid in ((GRUForecaster, False), (HybridForecaster, True)):
        path = tmp_path / f"{cls.name}.pt"
        torch.save(make_net(hybrid).state_dict(), path)
        f = cls(str(path))
        assert f.is_trained
        h = simulate(25.0, np.zeros(60), alpha=0.02, k=0.05, h_inf=0.0)
        q = f.forecast(h, np.zeros(61), np.zeros(60), 1.0)
        assert len(q.q50) == 61
        assert np.all(q.q10 <= q.q50 + 1e-6) and np.all(q.q50 <= q.q90 + 1e-6)
        assert np.all(q.q10 >= 0)
