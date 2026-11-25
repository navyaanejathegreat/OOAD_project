"""1-D Kalman tracker per road segment: state x = [h (cm), dh/dt (cm/min)]."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TrackPoint:
    time: datetime
    depth_cm: float
    sigma_cm: float
    rate_cm_min: float


class KalmanDepthTracker:
    """Constant-velocity Kalman filter.

    Each observation is weighted by its (effective) sigma, so a blurry citizen
    photo moves the estimate less than a sharp CCTV frame. Gaps are handled by
    predicting forward, which grows the uncertainty.
    """

    def __init__(self, process_noise: float = 0.005, init_rate_sigma: float = 0.5, max_history: int = 2000):
        self.q = process_noise
        self.init_rate_var = init_rate_sigma ** 2
        self.max_history = max_history
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.t: Optional[datetime] = None
        self.history: List[TrackPoint] = []

    @property
    def initialized(self) -> bool:
        return self.x is not None

    def _F_Q(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        return F, Q

    def _predict(self, t: datetime) -> Tuple[np.ndarray, np.ndarray]:
        dt = max((t - self.t).total_seconds() / 60.0, 0.0)
        F, Q = self._F_Q(dt)
        x = F @ self.x
        P = F @ self.P @ F.T + Q
        if x[0] < 0:
            x[0] = 0.0
            x[1] = max(x[1], 0.0)
        return x, P

    def update(self, t: datetime, depth_cm: float, sigma_cm: float) -> TrackPoint:
        r = max(sigma_cm, 0.1) ** 2
        if self.x is None:
            self.x = np.array([max(depth_cm, 0.0), 0.0])
            self.P = np.diag([r, self.init_rate_var])
            self.t = t
        else:
            x, P = self._predict(t)
            H = np.array([[1.0, 0.0]])
            S = (H @ P @ H.T)[0, 0] + r
            K = (P @ H.T)[:, 0] / S
            innovation = depth_cm - x[0]
            x = x + K * innovation
            P = (np.eye(2) - np.outer(K, H[0])) @ P
            if x[0] < 0:
                x[0] = 0.0
            self.x, self.P = x, 0.5 * (P + P.T)
            self.t = max(t, self.t)
        tp = TrackPoint(self.t, float(self.x[0]), float(np.sqrt(self.P[0, 0])), float(self.x[1]))
        self.history.append(tp)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        return tp

    def predict_at(self, t: datetime) -> Optional[TrackPoint]:
        if self.x is None:
            return None
        if t <= self.t:
            return TrackPoint(self.t, float(self.x[0]), float(np.sqrt(self.P[0, 0])), float(self.x[1]))
        x, P = self._predict(t)
        return TrackPoint(t, float(x[0]), float(np.sqrt(P[0, 0])), float(x[1]))
