"""Citizen report validation (Chain of Responsibility).

FormatCheck -> DuplicateCheck -> QualityCheck -> ConfidenceCheck -> HumanModeration

Each handler can pass the report on, reject it, or escalate it to a human.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np

from ..domain.observation import CitizenReport


class Verdict(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PENDING = "pending_moderation"


@dataclass
class ValidationResult:
    verdict: Verdict
    reasons: List[str] = field(default_factory=list)
    checked_by: List[str] = field(default_factory=list)


@dataclass
class ValidationContext:
    now: datetime
    known_segments: set
    tracked_depth: Callable[[str], Optional[tuple]] = lambda sid: None  # -> (h, sigma) or None
    needs_review: bool = False
    notes: List[str] = field(default_factory=list)


class ReportValidator(ABC):
    name = "validator"

    def __init__(self):
        self._next: Optional[ReportValidator] = None

    def set_next(self, nxt: "ReportValidator") -> "ReportValidator":
        self._next = nxt
        return nxt

    def handle(self, report: CitizenReport, ctx: ValidationContext,
               result: Optional[ValidationResult] = None) -> ValidationResult:
        result = result or ValidationResult(Verdict.ACCEPTED)
        result.checked_by.append(self.name)
        reason = self.check(report, ctx)
        if reason is not None:
            result.verdict = Verdict.REJECTED
            result.reasons.append(f"{self.name}: {reason}")
            return result
        if self._next is not None:
            return self._next.handle(report, ctx, result)
        result.reasons.extend(ctx.notes)
        return result

    @abstractmethod
    def check(self, report: CitizenReport, ctx: ValidationContext) -> Optional[str]:
        """Return a rejection reason, or None to pass. May set ctx.needs_review."""


class FormatCheck(ReportValidator):
    name = "format"

    def __init__(self, max_age_min: float = 60.0, max_depth_cm: float = 200.0):
        super().__init__()
        self.max_age = timedelta(minutes=max_age_min)
        self.max_depth = max_depth_cm

    def check(self, r, ctx):
        if r.segment_id not in ctx.known_segments:
            return "unknown road segment"
        if r.timestamp > ctx.now + timedelta(minutes=2):
            return "timestamp in the future"
        if ctx.now - r.timestamp > self.max_age:
            return "report too old"
        if not (0.0 <= r.depth_cm <= self.max_depth):
            return "depth out of range"
        if r.image is None and not r.image_hash:
            return "no photo attached"
        return None


class DuplicateCheck(ReportValidator):
    name = "duplicate"

    def __init__(self, window_min: float = 10.0):
        super().__init__()
        self.window = timedelta(minutes=window_min)
        self._seen_hashes: Dict[str, datetime] = {}
        self._last_by_reporter: Dict[tuple, datetime] = {}

    def check(self, r, ctx):
        if r.image_hash and r.image_hash in self._seen_hashes:
            return "same photo already submitted"
        key = (r.reporter_id, r.segment_id)
        last = self._last_by_reporter.get(key)
        if last is not None and abs(r.timestamp - last) < self.window:
            return "same reporter, same segment, too soon"
        if r.image_hash:
            self._seen_hashes[r.image_hash] = r.timestamp
        self._last_by_reporter[key] = r.timestamp
        return None


class QualityCheck(ReportValidator):
    """Rejects very dark or very blurry photos (variance of Laplacian)."""
    name = "quality"

    def __init__(self, min_sharpness: float = 15.0, min_brightness: float = 20.0):
        super().__init__()
        self.min_sharpness = min_sharpness
        self.min_brightness = min_brightness

    def check(self, r, ctx):
        img = r.image
        if img is None:
            ctx.notes.append("quality: no pixels to check (hash only)")
            return None
        import cv2

        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) < self.min_brightness:
            return "photo too dark"
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < self.min_sharpness:
            return "photo too blurry"
        return None


class ConfidenceCheck(ReportValidator):
    """Escalates reports that disagree with the tracked depth or are very uncertain."""
    name = "confidence"

    def __init__(self, max_sigma_cm: float = 15.0, n_sigma: float = 3.0):
        super().__init__()
        self.max_sigma = max_sigma_cm
        self.n_sigma = n_sigma

    def check(self, r, ctx):
        if r.sigma_cm > self.max_sigma:
            ctx.needs_review = True
            ctx.notes.append("confidence: estimate too uncertain")
        tracked = ctx.tracked_depth(r.segment_id)
        if tracked is not None:
            h, s = tracked
            if abs(r.depth_cm - h) > self.n_sigma * np.hypot(s, r.effective_sigma_cm):
                ctx.needs_review = True
                ctx.notes.append(f"confidence: disagrees with tracked depth {h:.0f} cm")
        return None


class HumanModeration(ReportValidator):
    """End of the chain: reports flagged for review wait for a municipal operator (UC-09)."""
    name = "moderation"

    def __init__(self):
        super().__init__()
        self.queue: Dict[str, CitizenReport] = {}

    def handle(self, report, ctx, result=None):
        result = result or ValidationResult(Verdict.ACCEPTED)
        result.checked_by.append(self.name)
        result.reasons.extend(ctx.notes)
        if ctx.needs_review:
            self.queue[report.obs_id] = report
            result.verdict = Verdict.PENDING
        return result

    def check(self, r, ctx):  # pragma: no cover - handle() is overridden
        return None

    def approve(self, obs_id: str) -> Optional[CitizenReport]:
        r = self.queue.pop(obs_id, None)
        if r is not None:
            r.verified = True
            r.reliability = max(r.reliability, 0.8)
        return r

    def reject(self, obs_id: str) -> Optional[CitizenReport]:
        return self.queue.pop(obs_id, None)


def build_default_chain():
    """Returns (head, moderation) so callers can reach the moderation queue."""
    head = FormatCheck()
    moderation = HumanModeration()
    head.set_next(DuplicateCheck()).set_next(QualityCheck()).set_next(ConfidenceCheck()).set_next(moderation)
    return head, moderation
