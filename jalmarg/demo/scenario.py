"""A scripted monsoon evening on the demo graph, replayed as if live.

Heavy rain for the first 30 minutes; each segment's true depth follows the
physics model with its own drain rate; cameras report every 2 minutes with
noise. The canal-road camera drops out after 70 minutes, so that segment goes
Unknown (fail-safe for low vehicles). A citizen report arrives at minute 40.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np

from ..adapters.weather import StaticRainfall
from ..domain.observation import CameraObservation, CitizenReport
from ..routing.graph import build_demo_graph
from ..service import JalMargService
from ..temporal.forecaster import RecessionForecaster
from ..temporal.synth import simulate

START = datetime(2026, 7, 15, 17, 0)
DURATION_MIN = 180
RAIN_STOP_MIN = 30
RAIN_MM_H = 40.0


@dataclass
class SegmentScript:
    segment_id: str
    name: str
    camera_id: str
    h0: float
    alpha: float
    k: float
    h_inf: float
    noise_cm: float = 1.5
    camera_until_min: Optional[int] = None


SCRIPTS = [
    SegmentScript("underpass", "Railway underpass", "cam_underpass", 4.0, 0.035, 0.035, 0.0),
    SegmentScript("market_road", "Market road", "cam_market", 3.0, 0.03, 0.018, 1.0),
    SegmentScript("canal_road", "Canal road", "cam_canal", 1.0, 0.012, 0.06, 0.0, camera_until_min=70),
    SegmentScript("station_road", "Station road", "cam_station", 0.0, 0.006, 0.08, 0.0),
]


def rain_at_minute(m: float) -> float:
    return RAIN_MM_H if 0 <= m < RAIN_STOP_MIN else 0.0


def true_depths() -> Dict[str, np.ndarray]:
    rain = np.array([rain_at_minute(m) for m in range(DURATION_MIN)])
    return {s.segment_id: simulate(s.h0, rain, s.alpha, s.k, s.h_inf) for s in SCRIPTS}


def build_service(forecaster: Optional[RecessionForecaster] = None) -> JalMargService:
    weather = StaticRainfall(lambda t: rain_at_minute((t - START).total_seconds() / 60.0))
    svc = JalMargService(build_demo_graph(), weather=weather, forecaster=forecaster)
    for s in SCRIPTS:
        svc.add_segment(s.segment_id, s.name)
        svc.register_camera(s.camera_id, s.segment_id)
    return svc


def replay(until_min: int, forecaster: Optional[RecessionForecaster] = None, seed: int = 7,
           every_min: int = 2) -> JalMargService:
    """Build a fresh service and feed it everything observed up to `until_min`."""
    rng = np.random.default_rng(seed)
    truth = true_depths()
    svc = build_service(forecaster)
    for m in range(0, until_min + 1):
        t = START + timedelta(minutes=m)
        if m % every_min == 0:
            for s in SCRIPTS:
                noise = rng.normal(0, s.noise_cm)
                if s.camera_until_min is not None and m > s.camera_until_min:
                    continue
                h = max(truth[s.segment_id][m] + noise, 0.0)
                svc.ingest_observation(CameraObservation(s.segment_id, t, h, s.noise_cm, 0.9,
                                                         camera_id=s.camera_id, n_references=2))
        if m == 40:
            svc.submit_report(CitizenReport(
                "market_road", t, float(truth["market_road"][m]) + 2.0, 5.0,
                reporter_id="citizen_demo", image_hash="demo-photo-001", note="water up to scooter wheel hub"))
        svc.tick(t)
    return svc
