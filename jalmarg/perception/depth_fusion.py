"""Depth from reference objects and inverse-variance fusion.

    h_r     = f_r * L_r
    sigma_r^2 = (L_r * sigma_f)^2 + (f_r * sigma_L)^2
    h_hat   = sum(h_r / sigma_r^2) / sum(1 / sigma_r^2)
    sigma_hat^2 = 1 / sum(1 / sigma_r^2)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class ReferenceSpec:
    """Real-world height of a reference object class and its uncertainty (cm)."""
    name: str
    length_cm: float
    sigma_length_cm: float


# Indicative, class-based ranges; carried into sigma (see README viva answers).
REFERENCE_CATALOGUE: Dict[str, ReferenceSpec] = {
    "car_wheel": ReferenceSpec("car_wheel", 60.0, 4.0),
    "suv_wheel": ReferenceSpec("suv_wheel", 70.0, 5.0),
    "bus_wheel": ReferenceSpec("bus_wheel", 100.0, 6.0),
    "two_wheeler_wheel": ReferenceSpec("two_wheeler_wheel", 45.0, 4.0),
    "auto_wheel": ReferenceSpec("auto_wheel", 40.0, 4.0),
    "kerb": ReferenceSpec("kerb", 15.0, 3.0),
    "person_leg": ReferenceSpec("person_leg", 85.0, 10.0),  # ground to hip
    "bollard": ReferenceSpec("bollard", 90.0, 15.0),
}


@dataclass
class ReferenceMeasurement:
    kind: str
    submerged_fraction: float   # f_r in [0, 1]
    sigma_fraction: float       # sigma_f from waterline sharpness
    length_cm: float
    sigma_length_cm: float

    @classmethod
    def from_catalogue(cls, kind: str, f: float, sigma_f: float) -> "ReferenceMeasurement":
        spec = REFERENCE_CATALOGUE[kind]
        return cls(kind, f, sigma_f, spec.length_cm, spec.sigma_length_cm)

    @property
    def depth_cm(self) -> float:
        return self.submerged_fraction * self.length_cm

    @property
    def sigma_cm(self) -> float:
        s2 = (self.length_cm * self.sigma_fraction) ** 2 + (self.submerged_fraction * self.sigma_length_cm) ** 2
        return max(math.sqrt(s2), 0.5)


def fuse(measurements: Iterable[ReferenceMeasurement]) -> Optional[Tuple[float, float]]:
    """Inverse-variance weighted fusion; returns (h_hat, sigma_hat) or None if empty."""
    ms = list(measurements)
    if not ms:
        return None
    w = [1.0 / (m.sigma_cm ** 2) for m in ms]
    h = sum(wi * m.depth_cm for wi, m in zip(w, ms)) / sum(w)
    return h, math.sqrt(1.0 / sum(w))


def fuse_values(depths, sigmas) -> Tuple[float, float]:
    w = [1.0 / max(s, 1e-6) ** 2 for s in sigmas]
    return sum(wi * d for wi, d in zip(w, depths)) / sum(w), math.sqrt(1.0 / sum(w))
