"""Observations of water depth on a road segment."""
from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Observation(ABC):
    """A single depth measurement h +/- sigma with a source reliability in (0, 1]."""

    segment_id: str
    timestamp: datetime
    depth_cm: float
    sigma_cm: float
    reliability: float = 1.0
    obs_id: str = field(default_factory=_new_id)

    @property
    def source(self) -> str:
        return "observation"

    @property
    def effective_sigma_cm(self) -> float:
        """Less reliable sources are trusted less: sigma / sqrt(reliability)."""
        r = min(max(self.reliability, 1e-3), 1.0)
        return max(self.sigma_cm, 0.1) / math.sqrt(r)


@dataclass
class CameraObservation(Observation):
    camera_id: str = ""
    estimator: str = "reference_object"
    n_references: int = 0

    @property
    def source(self) -> str:
        return f"camera:{self.camera_id}"


@dataclass
class CitizenReport(Observation):
    reporter_id: str = ""
    image_hash: Optional[str] = None
    image: Optional[object] = None  # numpy array when available; never stored long-term
    lat: Optional[float] = None
    lon: Optional[float] = None
    note: str = ""
    verified: bool = False
    reliability: float = 0.5

    @property
    def source(self) -> str:
        return f"citizen:{self.reporter_id}"
