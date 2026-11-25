"""Camera sources (Adapter pattern): one frame interface for RTSP, video files and photos."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator, Optional

import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    timestamp: datetime
    camera_id: str
    segment_id: str


class CameraSource(ABC):
    def __init__(self, camera_id: str, segment_id: str):
        self.camera_id = camera_id
        self.segment_id = segment_id

    @abstractmethod
    def frames(self) -> Iterator[Frame]: ...


class VideoFileAdapter(CameraSource):
    """Replays a recorded clip as a 'live' stream, sampling one frame every `every_s` seconds.
    Timestamps start at `start_time` and advance with video time (optionally sped up)."""

    def __init__(self, path: str, camera_id: str, segment_id: str, start_time: Optional[datetime] = None,
                 every_s: float = 5.0, time_scale: float = 1.0):
        super().__init__(camera_id, segment_id)
        self.path = path
        self.start_time = start_time or datetime.now()
        self.every_s = every_s
        self.time_scale = time_scale

    def frames(self):
        import cv2

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {self.path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(int(round(fps * self.every_s)), 1)
        idx = 0
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    t = self.start_time + timedelta(seconds=idx / fps * self.time_scale)
                    yield Frame(img, t, self.camera_id, self.segment_id)
                idx += 1
        finally:
            cap.release()


class RtspAdapter(CameraSource):
    """Live RTSP stream; timestamps are wall-clock."""

    def __init__(self, url: str, camera_id: str, segment_id: str, every_s: float = 5.0, max_frames: Optional[int] = None):
        super().__init__(camera_id, segment_id)
        self.url = url
        self.every_s = every_s
        self.max_frames = max_frames

    def frames(self):
        import cv2

        cap = cv2.VideoCapture(self.url)
        last, n = None, 0
        try:
            while cap.isOpened():
                ok, img = cap.read()
                if not ok:
                    break
                now = datetime.now()
                if last is None or (now - last).total_seconds() >= self.every_s:
                    last = now
                    yield Frame(img, now, self.camera_id, self.segment_id)
                    n += 1
                    if self.max_frames and n >= self.max_frames:
                        break
        finally:
            cap.release()


class PhotoAdapter(CameraSource):
    """A single uploaded photo (citizen report) seen through the same interface."""

    def __init__(self, image, camera_id: str, segment_id: str, timestamp: Optional[datetime] = None):
        super().__init__(camera_id, segment_id)
        self.image = image
        self.timestamp = timestamp or datetime.now()

    def frames(self):
        img = self.image
        if isinstance(img, str):
            import cv2

            img = cv2.imread(img)
            if img is None:
                raise IOError(f"Cannot read image {self.image}")
        yield Frame(img, self.timestamp, self.camera_id, self.segment_id)
