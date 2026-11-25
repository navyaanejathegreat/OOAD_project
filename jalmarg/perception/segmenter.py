"""Thin wrapper around an Ultralytics YOLO11-seg model.

Water comes from our fine-tuned model; vehicles/people from COCO classes;
wheels from the Roboflow-labelled wheel model. Ultralytics is imported lazily
so the domain layer and tests don't need it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

WHEEL_CLASSES = {"car_wheel", "two_wheeler_wheel", "suv_wheel", "bus_wheel", "auto_wheel", "wheel"}


@dataclass
class Detection:
    cls: str
    confidence: float
    mask: np.ndarray        # HxW uint8
    box: tuple              # x1, y1, x2, y2


@dataclass
class FrameAnalysis:
    water_mask: np.ndarray
    detections: List[Detection] = field(default_factory=list)

    @property
    def water_fraction(self) -> float:
        return float((self.water_mask > 0).mean()) if self.water_mask.size else 0.0

    def of(self, classes) -> List[Detection]:
        classes = {classes} if isinstance(classes, str) else set(classes)
        return [d for d in self.detections if d.cls in classes]


class Segmenter:
    """Runs one or more YOLO-seg models and merges their masks into a FrameAnalysis."""

    def __init__(self, weights: Optional[List[str]] = None, conf: float = 0.35, device: Optional[str] = None,
                 class_aliases: Optional[Dict[str, str]] = None):
        self.weights = weights or ["models/water_wheels_yolo11s-seg.pt", "yolo11n-seg.pt"]
        self.conf = conf
        self.device = device
        self.aliases = {"motorcycle": "two_wheeler", "person": "person"} | (class_aliases or {})
        self._models = None

    def _load(self):
        if self._models is None:
            from ultralytics import YOLO  # lazy

            self._models = [YOLO(w) for w in self.weights]
        return self._models

    def analyse(self, image: np.ndarray) -> FrameAnalysis:
        import cv2

        H, W = image.shape[:2]
        water = np.zeros((H, W), np.uint8)
        dets: List[Detection] = []
        for model in self._load():
            res = model.predict(image, conf=self.conf, device=self.device, verbose=False)[0]
            if res.masks is None:
                continue
            masks = res.masks.data.cpu().numpy()
            for m, cls_id, conf, box in zip(masks, res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                            res.boxes.xyxy.cpu().numpy()):
                name = model.names[int(cls_id)]
                name = self.aliases.get(name, name)
                mask = cv2.resize((m > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                if name == "water":
                    water |= mask
                else:
                    dets.append(Detection(name, float(conf), mask, tuple(map(float, box))))
        return FrameAnalysis(water, dets)
