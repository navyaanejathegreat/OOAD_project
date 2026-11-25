"""Synthetic rise-and-drain depth series from the physics model

    dh/dt = alpha * R(t) - k * (h - h_inf)

with random rain pulses, observation noise and occasional drain blockage
(k drops mid-event: an effect the constant-k physics prior misses, which the
GRU correction is supposed to learn).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SyntheticSeries:
    h_true: np.ndarray        # cm, per minute
    h_obs: np.ndarray         # noisy tracked depth, cm
    rain: np.ndarray          # mm/h, per minute (observed)
    rain_forecast: np.ndarray  # mm/h, per minute (noisy forecast of the same)
    params: Dict[str, float] = field(default_factory=dict)


def simulate(h0: float, rain: np.ndarray, alpha: float, k, h_inf: float) -> np.ndarray:
    """Euler integration at 1-minute steps. k may be a scalar or a per-minute array."""
    n = len(rain)
    k_arr = np.broadcast_to(np.asarray(k, dtype=float), (n,))
    h = np.empty(n + 1)
    h[0] = h0
    for i in range(n):
        h[i + 1] = max(h[i] + alpha * rain[i] - k_arr[i] * (h[i] - h_inf), 0.0)
    return h


def generate_series(rng: np.random.Generator, length: int = 240) -> SyntheticSeries:
    alpha = rng.uniform(0.008, 0.035)       # cm/min per mm/h
    k = rng.uniform(0.012, 0.12)            # 1/min
    h_inf = rng.uniform(0.0, 3.0)
    h0 = h_inf + rng.uniform(0.0, 35.0)

    rain = np.zeros(length)
    for _ in range(rng.integers(1, 4)):
        start = int(rng.integers(0, int(length * 0.65)))
        dur = int(rng.integers(8, 45))
        rain[start:start + dur] += rng.uniform(5.0, 60.0)

    k_arr = np.full(length, k)
    blocked = rng.random() < 0.3
    if blocked:
        t_block = int(rng.integers(30, length - 30))
        k_arr[t_block:] *= rng.uniform(0.25, 0.6)

    h_true = simulate(h0, rain, alpha, k_arr, h_inf)[:length]
    noise_sigma = rng.uniform(0.3, 1.5)
    h_obs = np.clip(h_true + rng.normal(0, noise_sigma, length), 0, None)
    rain_fc = np.clip(rain * rng.uniform(0.7, 1.3) + rng.normal(0, 1.0, length), 0, None)
    return SyntheticSeries(
        h_true, h_obs, rain, rain_fc,
        dict(alpha=alpha, k=k, h_inf=h_inf, h0=h0, blocked=float(blocked), noise_sigma=noise_sigma),
    )


def generate_dataset(n_series: int, seed: int = 0, length: int = 240) -> List[SyntheticSeries]:
    rng = np.random.default_rng(seed)
    return [generate_series(rng, length) for _ in range(n_series)]


@dataclass
class Window:
    h_hist: np.ndarray     # observed depths, last element = now
    rain_hist: np.ndarray
    rain_future: np.ndarray
    h_future: np.ndarray   # true depth at minutes 1..H
    series_idx: int


def make_windows(
    series: List[SyntheticSeries],
    per_series: int = 3,
    hist: int = 60,
    horizon: int = 60,
    seed: int = 0,
    min_hist: int = 15,
) -> List[Window]:
    """Sample (history, future) windows. Windows never mix series, so splitting
    the *series* list gives a split by event with no leakage."""
    rng = np.random.default_rng(seed)
    out: List[Window] = []
    for idx, s in enumerate(series):
        n = len(s.h_true)
        for _ in range(per_series):
            now = int(rng.integers(hist, n - horizon))
            L = int(rng.integers(min_hist, hist + 1))
            out.append(Window(
                h_hist=s.h_obs[now - L + 1: now + 1].copy(),
                rain_hist=s.rain[now - L + 1: now + 1].copy(),
                rain_future=s.rain_forecast[now + 1: now + 1 + horizon].copy(),
                h_future=s.h_true[now + 1: now + 1 + horizon].copy(),
                series_idx=idx,
            ))
    return out


def save_dataset(series: List[SyntheticSeries], path: str) -> None:
    np.savez_compressed(
        path,
        h_true=np.stack([s.h_true for s in series]),
        h_obs=np.stack([s.h_obs for s in series]),
        rain=np.stack([s.rain for s in series]),
        rain_forecast=np.stack([s.rain_forecast for s in series]),
    )
