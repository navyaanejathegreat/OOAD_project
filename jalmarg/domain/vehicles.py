"""Vehicle profiles (Strategy pattern) and their factory.

Passability is a property of the (segment, vehicle) pair, so each vehicle
class decides for itself whether a depth estimate is safe.
"""
from __future__ import annotations

from abc import ABC
from typing import Dict, Optional, Type

# z-score of the 90th percentile of a normal distribution
Z90 = 1.2816

# Vehicles below this safe depth are blocked on segments in the Unknown state
LOW_VEHICLE_THRESHOLD_CM = 30.0


class VehicleProfile(ABC):
    """Abstract strategy: how deep may the water be for this vehicle?"""

    key: str = "vehicle"
    label: str = "Vehicle"
    default_safe_depth_cm: float = 0.0
    speed_factor: float = 1.0  # multiplier on free-flow travel time

    def __init__(self, safe_depth_cm: Optional[float] = None, margin_cm: float = 2.0):
        self.safe_depth_cm = float(safe_depth_cm if safe_depth_cm is not None else self.default_safe_depth_cm)
        self.margin_cm = float(margin_cm)

    @property
    def threshold_cm(self) -> float:
        """Depth the pessimistic estimate must stay below: d_v - m."""
        return self.safe_depth_cm - self.margin_cm

    @property
    def is_low(self) -> bool:
        return self.safe_depth_cm < LOW_VEHICLE_THRESHOLD_CM

    def is_passable(self, depth_cm: float, sigma_cm: float = 0.0) -> bool:
        """True when the 90th-percentile depth is below d_v - m."""
        return depth_cm + Z90 * max(sigma_cm, 0.0) < self.threshold_cm

    def is_passable_quantile(self, q90_cm: float) -> bool:
        return q90_cm < self.threshold_cm

    def __repr__(self) -> str:
        return f"{type(self).__name__}(safe_depth_cm={self.safe_depth_cm}, margin_cm={self.margin_cm})"


class Pedestrian(VehicleProfile):
    key, label, default_safe_depth_cm, speed_factor = "pedestrian", "Pedestrian", 15.0, 8.0


class TwoWheeler(VehicleProfile):
    key, label, default_safe_depth_cm, speed_factor = "two_wheeler", "Two-wheeler", 20.0, 1.0


class AutoRickshaw(VehicleProfile):
    key, label, default_safe_depth_cm, speed_factor = "auto", "Auto-rickshaw", 20.0, 1.2


class Car(VehicleProfile):
    key, label, default_safe_depth_cm, speed_factor = "car", "Hatchback / sedan", 28.0, 1.0


class SUV(VehicleProfile):
    key, label, default_safe_depth_cm, speed_factor = "suv", "SUV", 45.0, 1.0


class Bus(VehicleProfile):
    key, label, default_safe_depth_cm, speed_factor = "bus", "Bus / truck", 50.0, 1.4


class VehicleProfileFactory:
    """Factory: build a VehicleProfile from a key and optional config overrides."""

    _registry: Dict[str, Type[VehicleProfile]] = {
        cls.key: cls for cls in (Pedestrian, TwoWheeler, AutoRickshaw, Car, SUV, Bus)
    }

    def __init__(self, safe_depth_overrides: Optional[Dict[str, float]] = None, margin_cm: float = 2.0):
        self.overrides = dict(safe_depth_overrides or {})
        self.margin_cm = margin_cm

    @classmethod
    def register(cls, profile_cls: Type[VehicleProfile]) -> None:
        cls._registry[profile_cls.key] = profile_cls

    @classmethod
    def keys(cls) -> list[str]:
        return list(cls._registry)

    def create(self, key: str) -> VehicleProfile:
        try:
            profile_cls = self._registry[key]
        except KeyError:
            raise ValueError(f"Unknown vehicle class '{key}'. Known: {', '.join(self._registry)}") from None
        return profile_cls(safe_depth_cm=self.overrides.get(key), margin_cm=self.margin_cm)

    def all(self) -> list[VehicleProfile]:
        return [self.create(k) for k in self._registry]
