"""DepthEstimator strategies and their factory."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np

from ..domain.observation import CameraObservation
from .depth_fusion import REFERENCE_CATALOGUE, ReferenceMeasurement, fuse
from .segmenter import WHEEL_CLASSES, FrameAnalysis, Segmenter
from .wheel_geometry import measure_wheel

# Which reference spec to use for each detected wheel class
WHEEL_SPEC = {
    "car_wheel": "car_wheel", "wheel": "car_wheel", "suv_wheel": "suv_wheel",
    "bus_wheel": "bus_wheel", "two_wheeler_wheel": "two_wheeler_wheel", "auto_wheel": "auto_wheel",
}


class DepthEstimator(ABC):
    name = "estimator"

    @abstractmethod
    def estimate(self, image: np.ndarray, segment_id: str, timestamp: datetime,
                 camera_id: str = "", analysis: Optional[FrameAnalysis] = None) -> Optional[CameraObservation]:
        """Return an observation, or None if this estimator can't say anything."""


class ReferenceObjectEstimator(DepthEstimator):
    """Wheels (and other references) as rulers, fused by inverse variance."""

    name = "reference_object"

    def __init__(self, segmenter: Optional[Segmenter] = None, min_water_fraction: float = 0.01):
        self.segmenter = segmenter
        self.min_water_fraction = min_water_fraction

    def measurements(self, analysis: FrameAnalysis) -> List[ReferenceMeasurement]:
        out = []
        for det in analysis.of(WHEEL_CLASSES):
            wm = measure_wheel(det.mask, analysis.water_mask)
            if wm is None or wm.waterline_y is None:
                continue
            out.append(ReferenceMeasurement.from_catalogue(WHEEL_SPEC.get(det.cls, "car_wheel"),
                                                           wm.submerged_fraction, wm.sigma_fraction))
        return out

    def estimate(self, image, segment_id, timestamp, camera_id="", analysis=None):
        if analysis is None:
            if self.segmenter is None:
                raise ValueError("ReferenceObjectEstimator needs a Segmenter or a precomputed FrameAnalysis")
            analysis = self.segmenter.analyse(image)
        if analysis.water_fraction < self.min_water_fraction:
            return CameraObservation(segment_id, timestamp, 0.0, 1.5, 0.9, camera_id=camera_id,
                                     estimator=self.name, n_references=0)
        ms = self.measurements(analysis)
        fused = fuse(ms)
        if fused is None:
            return None
        h, s = fused
        return CameraObservation(segment_id, timestamp, h, s, 0.9, camera_id=camera_id,
                                 estimator=self.name, n_references=len(ms))


class LearnedDepthEstimator(DepthEstimator):
    """Fallback when no reference is visible: a regressor with a deliberately large sigma.

    `regressor(image, analysis) -> depth_cm`. Until a CNN head is trained, the
    default is a crude heuristic on the water-mask fraction, so its sigma is large.
    """

    name = "learned"

    def __init__(self, regressor: Optional[Callable] = None, sigma_cm: float = 12.0,
                 segmenter: Optional[Segmenter] = None):
        self.regressor = regressor or (lambda img, a: 40.0 * a.water_fraction)
        self.sigma_cm = sigma_cm
        self.segmenter = segmenter

    def estimate(self, image, segment_id, timestamp, camera_id="", analysis=None):
        if analysis is None and self.segmenter is not None:
            analysis = self.segmenter.analyse(image)
        if analysis is None:
            return None
        h = float(max(self.regressor(image, analysis), 0.0))
        return CameraObservation(segment_id, timestamp, h, self.sigma_cm, 0.6, camera_id=camera_id,
                                 estimator=self.name, n_references=0)


class FallbackChainEstimator(DepthEstimator):
    """Try estimators in order; the first that returns an observation wins."""

    name = "fallback_chain"

    def __init__(self, estimators: List[DepthEstimator], segmenter: Optional[Segmenter] = None):
        self.estimators = estimators
        self.segmenter = segmenter

    def estimate(self, image, segment_id, timestamp, camera_id="", analysis=None):
        if analysis is None and self.segmenter is not None:
            analysis = self.segmenter.analyse(image)
        for est in self.estimators:
            obs = est.estimate(image, segment_id, timestamp, camera_id, analysis)
            if obs is not None:
                return obs
        return None


class EstimatorFactory:
    """Build estimators from a config name."""

    @staticmethod
    def create(kind: str = "default", segmenter: Optional[Segmenter] = None, **kw) -> DepthEstimator:
        if kind == "reference_object":
            return ReferenceObjectEstimator(segmenter, **kw)
        if kind == "learned":
            return LearnedDepthEstimator(segmenter=segmenter, **kw)
        if kind == "default":
            return FallbackChainEstimator(
                [ReferenceObjectEstimator(segmenter), LearnedDepthEstimator(segmenter=segmenter)], segmenter)
        raise ValueError(f"Unknown estimator '{kind}'")
