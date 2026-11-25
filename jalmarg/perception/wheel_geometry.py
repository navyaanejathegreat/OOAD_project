"""Wheel-as-a-ruler geometry: fit an ellipse to a wheel mask and find where the
water mask cuts it. The submerged fraction f is the share of the wheel's
vertical extent below the waterline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class WheelMeasurement:
    center: tuple
    axes: tuple            # full axis lengths (px)
    angle: float
    top_y: float
    bottom_y: float
    waterline_y: Optional[float]
    submerged_fraction: float
    sigma_fraction: float


def fit_wheel_ellipse(wheel_mask: np.ndarray):
    """Fit an ellipse to the largest contour of a binary mask. Returns cv2 ellipse or None."""
    m = (wheel_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if len(c) < 5:
        return None
    return cv2.fitEllipse(c)


def _ellipse_vertical_extent(ellipse):
    (cx, cy), (w, h), ang = ellipse
    a, b = w / 2.0, h / 2.0
    t = np.deg2rad(ang)
    half = np.sqrt((a * np.sin(t)) ** 2 + (b * np.cos(t)) ** 2)
    return cy - half, cy + half


def measure_wheel(wheel_mask: np.ndarray, water_mask: np.ndarray, min_cover: float = 0.5,
                  base_sigma_f: float = 0.03) -> Optional[WheelMeasurement]:
    """Estimate the submerged fraction of a wheel.

    Label wheels as the full (amodal) ellipse where possible; if only the
    visible part is segmented, fitEllipse extrapolates the hidden arc, with
    some bias that shows up in the depth error analysis. The waterline is the
    first row, scanning down inside the ellipse, where water covers at least
    `min_cover` of the ellipse width.
    """
    ell = fit_wheel_ellipse(wheel_mask)
    if ell is None:
        return None
    top, bottom = _ellipse_vertical_extent(ell)
    height = bottom - top
    if height < 4:
        return None

    H, W = water_mask.shape[:2]
    ell_mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(ell_mask, ell, 1, -1)
    water = water_mask > 0

    waterline = None
    col_lines = []
    rows = range(max(int(np.floor(top)), 0), min(int(np.ceil(bottom)) + 1, H))
    for y in rows:
        inside = ell_mask[y] > 0
        n = inside.sum()
        if n == 0:
            continue
        if (water[y] & inside).sum() / n >= min_cover:
            waterline = float(y)
            break
    if waterline is None:
        return WheelMeasurement(ell[0], ell[1], ell[2], top, bottom, None, 0.0, base_sigma_f)

    # Waterline sharpness: spread of the first water row per column inside the ellipse
    xs = np.where(ell_mask.any(axis=0))[0]
    for x in xs:
        ys = np.where(ell_mask[:, x] > 0)[0]
        wy = ys[water[ys, x]]
        if len(wy):
            col_lines.append(wy.min())
    spread = float(np.std(col_lines)) if len(col_lines) > 1 else 0.0
    f = float(np.clip((bottom - waterline) / height, 0.0, 1.0))
    sigma_f = float(np.hypot(base_sigma_f, spread / height))
    return WheelMeasurement(ell[0], ell[1], ell[2], top, bottom, waterline, f, sigma_f)
