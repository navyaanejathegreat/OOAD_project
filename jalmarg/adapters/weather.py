"""Rainfall sources: Open-Meteo (free, no key) and a static/scripted source for tests and demos."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Tuple

import numpy as np


class WeatherService(ABC):
    @abstractmethod
    def rainfall(self, lat: float, lon: float, start: datetime, minutes: int) -> np.ndarray:
        """Rain intensity in mm/h for each minute from `start`."""

    def now(self, lat: float, lon: float, t: datetime) -> float:
        return float(self.rainfall(lat, lon, t, 1)[0])


class StaticRainfall(WeatherService):
    """Rain as a function of time; e.g. lambda t: 30.0 if t < stop else 0.0"""

    def __init__(self, fn: Callable[[datetime], float] = lambda t: 0.0):
        self.fn = fn

    def rainfall(self, lat, lon, start, minutes):
        return np.array([self.fn(start + timedelta(minutes=i)) for i in range(minutes)], dtype=float)


class OpenMeteoClient(WeatherService):
    """15-minutely precipitation from https://api.open-meteo.com (mm per 15 min -> mm/h)."""

    URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout_s: float = 10.0):
        self.timeout = timeout_s
        self._cache: dict = {}

    def _fetch(self, lat: float, lon: float) -> List[Tuple[datetime, float]]:
        key = (round(lat, 3), round(lon, 3), datetime.now().strftime("%Y%m%d%H%M")[:-1])
        if key in self._cache:
            return self._cache[key]
        q = urllib.parse.urlencode({
            "latitude": lat, "longitude": lon, "minutely_15": "precipitation",
            "past_days": 1, "forecast_days": 1, "timezone": "Asia/Kolkata",
        })
        with urllib.request.urlopen(f"{self.URL}?{q}", timeout=self.timeout) as resp:
            data = json.load(resp)
        m = data["minutely_15"]
        series = [(datetime.fromisoformat(t), (p or 0.0) * 4.0) for t, p in zip(m["time"], m["precipitation"])]
        self._cache = {key: series}
        return series

    def rainfall(self, lat, lon, start, minutes):
        series = self._fetch(lat, lon)
        times = np.array([(t - start).total_seconds() / 60.0 for t, _ in series])
        vals = np.array([v for _, v in series])
        grid = np.arange(minutes, dtype=float)
        idx = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(vals) - 1)
        return vals[idx]
