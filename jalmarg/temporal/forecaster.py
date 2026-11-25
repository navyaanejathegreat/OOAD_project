"""Recession forecasters: physics-only, pure GRU, and hybrid physics + GRU.

All forecasters share one interface:

    forecast(h_hist, rain_hist, rain_future, sigma_now, horizon) -> Quantiles

h_hist / rain_hist are per-minute arrays whose last element is "now".
Quantiles hold arrays for minutes 0..horizon (index 0 = now).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from .synth import simulate

Z90 = 1.2816
QUANTILES = (0.1, 0.5, 0.9)
HIST_LEN = 60
HORIZON = 60
SCALE_H = 50.0     # cm normaliser for networks
SCALE_R = 50.0     # mm/h normaliser
SCALE_OUT = 10.0   # network outputs are in units of 10 cm


@dataclass
class Quantiles:
    minutes: np.ndarray
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray

    def at(self, minute: int):
        i = int(np.clip(minute, 0, len(self.minutes) - 1))
        return self.q10[i], self.q50[i], self.q90[i]


def _pad_future(rain_future, horizon: int) -> np.ndarray:
    rf = np.zeros(horizon) if rain_future is None else np.asarray(rain_future, dtype=float)[:horizon]
    if len(rf) < horizon:
        rf = np.concatenate([rf, np.full(horizon - len(rf), rf[-1] if len(rf) else 0.0)])
    return rf


class RecessionForecaster(ABC):
    name = "forecaster"

    @abstractmethod
    def forecast(self, h_hist, rain_hist, rain_future=None, sigma_now: float = 1.0,
                 horizon: int = HORIZON) -> Quantiles: ...


# ---------------------------------------------------------------------------
# Physics baseline
# ---------------------------------------------------------------------------
@dataclass
class PhysicsParams:
    alpha: float = 0.02
    k: float = 0.05
    h_inf: float = 0.0
    resid_std: float = 1.0


class PhysicsForecaster(RecessionForecaster):
    """Fits alpha, k, h_inf by least squares on the recent history, then
    integrates forward. Uncertainty grows with horizon: s^2 = sigma0^2 + resid^2 + c*tau."""

    name = "physics"

    def __init__(self, drift_var_per_min: float = 0.15, min_points: int = 5):
        self.drift = drift_var_per_min
        self.min_points = min_points

    def fit(self, h_hist, rain_hist) -> PhysicsParams:
        h = np.asarray(h_hist, dtype=float)
        r = np.asarray(rain_hist, dtype=float)
        if len(h) < self.min_points:
            return PhysicsParams(h_inf=0.0, resid_std=1.0)

        def resid(theta):
            a, k, hi = theta
            return simulate(h[0], r[:-1], a, k, hi)[: len(h)] - h

        x0 = [0.02, 0.05, min(float(h.min()), 2.0)]
        hi_max = max(float(h.min()) + 1e-3, 1e-3)
        lo, hi = [0.0, 0.002, 0.0], [0.1, 0.5, hi_max]
        x0 = list(np.clip(x0, lo, hi))
        try:
            sol = least_squares(resid, x0, bounds=(lo, hi), max_nfev=80)
            a, k, hinf = sol.x
            rs = float(np.sqrt(np.mean(sol.fun ** 2)))
        except Exception:  # pragma: no cover - numerical failure fallback
            a, k, hinf, rs = 0.02, 0.05, 0.0, 2.0
        return PhysicsParams(float(a), float(k), float(hinf), rs)

    def median_path(self, h_now: float, rain_future, horizon: int, p: PhysicsParams) -> np.ndarray:
        return simulate(h_now, _pad_future(rain_future, horizon), p.alpha, p.k, p.h_inf)

    def forecast(self, h_hist, rain_hist, rain_future=None, sigma_now: float = 1.0,
                 horizon: int = HORIZON) -> Quantiles:
        h_hist = np.asarray(h_hist, dtype=float)
        p = self.fit(h_hist, rain_hist)
        q50 = self.median_path(float(h_hist[-1]), rain_future, horizon, p)
        tau = np.arange(horizon + 1, dtype=float)
        s = np.sqrt(sigma_now ** 2 + p.resid_std ** 2 * (tau > 0) + self.drift * tau)
        return Quantiles(tau, np.clip(q50 - Z90 * s, 0, None), q50, q50 + Z90 * s)


# ---------------------------------------------------------------------------
# Neural forecasters (torch imported lazily so the domain layer has no torch dependency)
# ---------------------------------------------------------------------------
def build_inputs(h_hist, rain_hist, rain_future, hist_len: int = HIST_LEN, horizon: int = HORIZON):
    """Left-pad history to hist_len; returns (seq [hist_len, 3], future rain [horizon])."""
    h = np.asarray(h_hist, dtype=float)[-hist_len:]
    r = np.asarray(rain_hist, dtype=float)[-hist_len:]
    if len(r) < len(h):
        r = np.concatenate([np.zeros(len(h) - len(r)), r])
    pad = hist_len - len(h)
    mask = np.concatenate([np.zeros(pad), np.ones(len(h))])
    h = np.concatenate([np.full(pad, h[0]), h])
    r = np.concatenate([np.zeros(pad), r])
    seq = np.stack([h / SCALE_H, r / SCALE_R, mask], axis=1).astype(np.float32)
    rf = (_pad_future(rain_future, horizon) / SCALE_R).astype(np.float32)
    return seq, rf


def make_net(hybrid: bool, horizon: int = HORIZON, hidden: int = 48):
    import torch
    from torch import nn

    class QuantileGRU(nn.Module):
        """GRU over the history; head sees future rain (and the physics path if hybrid).
        Outputs a centre correction and two positive spreads per future minute."""

        def __init__(self):
            super().__init__()
            self.hybrid = hybrid
            self.gru = nn.GRU(3, hidden, batch_first=True)
            extra = horizon * (2 if hybrid else 1)
            self.head = nn.Sequential(
                nn.Linear(hidden + extra, 128), nn.ReLU(),
                nn.Linear(128, 3 * horizon),
            )
            self.horizon = horizon

        def forward(self, seq, rain_future, base):
            """base: [B, H] in cm - physics median (hybrid) or persistence (pure)."""
            _, hN = self.gru(seq)
            feats = [hN[-1], rain_future]
            if self.hybrid:
                feats.append(base / SCALE_H)
            out = self.head(torch.cat(feats, dim=1)).view(-1, 3, self.horizon)
            q50 = base + SCALE_OUT * out[:, 0]
            lo = SCALE_OUT * nn.functional.softplus(out[:, 1]) + 0.05
            hi = SCALE_OUT * nn.functional.softplus(out[:, 2]) + 0.05
            return torch.stack([q50 - lo, q50, q50 + hi], dim=1)  # [B, 3, H]

    return QuantileGRU()


def pinball_loss(pred, target, quantiles=QUANTILES):
    """pred [B, 3, H], target [B, H] (same units). Mean pinball loss over quantiles."""
    import torch

    total = 0.0
    for i, tau in enumerate(quantiles):
        diff = target - pred[:, i]
        total = total + torch.maximum(tau * diff, (tau - 1) * diff).mean()
    return total / len(quantiles)


class _NeuralForecaster(RecessionForecaster):
    hybrid = False

    def __init__(self, weights_path: Optional[str] = None, physics: Optional[PhysicsForecaster] = None):
        self.physics = physics or PhysicsForecaster()
        self.net = None
        self.weights_path = weights_path
        if weights_path and os.path.exists(weights_path):
            import torch
            self.net = make_net(self.hybrid)
            self.net.load_state_dict(torch.load(weights_path, map_location="cpu"))
            self.net.eval()

    @property
    def is_trained(self) -> bool:
        return self.net is not None

    def base_path(self, h_hist, rain_hist, rain_future, horizon) -> np.ndarray:
        raise NotImplementedError

    def forecast(self, h_hist, rain_hist, rain_future=None, sigma_now: float = 1.0,
                 horizon: int = HORIZON) -> Quantiles:
        if self.net is None or horizon != HORIZON:
            return self.physics.forecast(h_hist, rain_hist, rain_future, sigma_now, horizon)
        import torch

        h_hist = np.asarray(h_hist, dtype=float)
        seq, rf = build_inputs(h_hist, rain_hist, rain_future)
        base = self.base_path(h_hist, rain_hist, rain_future, horizon)
        with torch.no_grad():
            out = self.net(
                torch.from_numpy(seq)[None], torch.from_numpy(rf)[None],
                torch.from_numpy(base.astype(np.float32))[None],
            )[0].numpy()
        h0 = float(h_hist[-1])
        z = Z90 * sigma_now
        q10 = np.concatenate([[max(h0 - z, 0)], np.clip(out[0], 0, None)])
        q50 = np.concatenate([[h0], np.clip(out[1], 0, None)])
        q90 = np.concatenate([[h0 + z], np.maximum(out[2], q50[1:])])
        q10 = np.minimum(q10, q50)
        return Quantiles(np.arange(horizon + 1, dtype=float), q10, q50, q90)


class GRUForecaster(_NeuralForecaster):
    """Baseline 2: pure GRU (correction on top of persistence)."""
    name = "gru"
    hybrid = False

    def base_path(self, h_hist, rain_hist, rain_future, horizon):
        return np.full(horizon, float(h_hist[-1]))


class HybridForecaster(_NeuralForecaster):
    """Ours: physics prior + GRU correction with quantile outputs."""
    name = "hybrid"
    hybrid = True

    def base_path(self, h_hist, rain_hist, rain_future, horizon):
        p = self.physics.fit(h_hist, rain_hist)
        return self.physics.median_path(float(h_hist[-1]), rain_future, horizon, p)[1:]


DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def default_forecaster(models_dir: Optional[str] = None) -> RecessionForecaster:
    """Hybrid if trained weights exist, otherwise the physics baseline."""
    path = os.path.join(models_dir or DEFAULT_MODELS_DIR, "forecaster_hybrid.pt")
    if os.path.exists(path):
        try:
            return HybridForecaster(path)
        except Exception:  # pragma: no cover
            pass
    return PhysicsForecaster()
