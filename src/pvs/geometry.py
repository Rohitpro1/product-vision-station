"""Orientation geometry contract (architecture PDF p.3, build prompt section 5).

Image frame: rectified, unmirrored source image after orientation metadata is
applied. Origin top-left, +x right, +y down, angles clockwise-positive.

* Axis    : undirected long line, angle in [0, 180).
* Heading : physical end A -> B, angle in [0, 360), atan2(yB-yA, xB-xA).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Reason(str, Enum):
    """Machine-readable reason codes. Ambiguity is distinct from missing data."""

    OK = "ok"
    # missing data
    ENDPOINT_HIDDEN = "endpoint_hidden"
    NO_MASK = "no_mask"
    NO_EVIDENCE = "no_evidence"
    EVIDENCE_EXPIRED = "evidence_expired"
    # ambiguity / instability
    AXIS_UNSTABLE = "axis_unstable"
    NEAR_ROUND = "near_round"
    TOO_SMALL = "too_small"
    TIED_CANDIDATES = "tied_candidates"
    CONFLICTING_CODE = "conflicting_code"
    LOW_INLIERS = "low_inliers"
    CLUSTERED_MATCHES = "clustered_matches"
    HIGH_REPROJECTION = "high_reprojection_error"
    UNSTABLE_TEMPORAL = "unstable_temporal"
    BELOW_THRESHOLD = "below_threshold"
    NOT_IN_CATALOGUE = "not_in_catalogue"
    FACE_UNKNOWN = "face_unknown"
    ORIENTATION_UNAVAILABLE = "orientation_unavailable"


def wrap360(deg: float) -> float:
    v = math.fmod(deg, 360.0)
    if v < 0:
        v += 360.0
    return 0.0 if v >= 360.0 else v


def wrap180(deg: float) -> float:
    v = math.fmod(deg, 180.0)
    if v < 0:
        v += 180.0
    return 0.0 if v >= 180.0 else v


def heading_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Directed angle A->B in image coordinates (clockwise positive), [0, 360)."""
    return wrap360(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))


def axis_from_heading(h: float) -> float:
    return wrap180(h)


def periodic_error(pred: float, truth: float, period: float) -> float:
    """min(|d| mod P, P - (|d| mod P)) as defined on PDF p.8."""
    d = abs(pred - truth) % period
    return min(d, period - d)


@dataclass(frozen=True)
class AxisFit:
    angle_deg: float | None
    elongation: float  # major/minor std ratio
    residual: float  # RMS distance to the fitted line, normalised by major extent
    extent_px: float  # visible length along major axis
    reason: Reason
    centre: tuple[float, float] | None = None


@dataclass(frozen=True)
class AxisGate:
    min_elongation: float = 1.6
    min_extent_px: float = 40.0
    max_residual: float = 0.35


def fit_axis(points_xy: np.ndarray, gate: AxisGate | None = None) -> AxisFit:
    """Principal-axis fit to mask pixels / contour points.

    The eigenvector sign carries no directional meaning, so only the axial
    angle in [0, 180) is returned.
    """
    gate = gate or AxisGate()
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 5:
        return AxisFit(None, 0.0, 0.0, 0.0, Reason.NO_MASK)
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    major = evecs[:, 0]
    sd_major = math.sqrt(max(evals[0], 1e-12))
    sd_minor = math.sqrt(max(evals[1], 1e-12))
    elong = sd_major / sd_minor
    proj = (pts - c) @ major
    extent = float(proj.max() - proj.min())
    perp = (pts - c) @ evecs[:, 1]
    residual = float(np.sqrt(np.mean(perp**2)) / max(extent, 1e-9))
    angle = wrap180(math.degrees(math.atan2(major[1], major[0])))
    centre = (float(c[0]), float(c[1]))
    if extent < gate.min_extent_px:
        return AxisFit(None, elong, residual, extent, Reason.TOO_SMALL, centre)
    if elong < gate.min_elongation:
        return AxisFit(None, elong, residual, extent, Reason.NEAR_ROUND, centre)
    if residual > gate.max_residual:
        return AxisFit(None, elong, residual, extent, Reason.AXIS_UNSTABLE, centre)
    return AxisFit(angle, elong, residual, extent, Reason.OK, centre)


def mask_points(mask: np.ndarray, max_points: int = 20000) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    return pts


# ---------------------------------------------------------------- smoothing
@dataclass
class _Obs:
    t: float
    angle: float


class PeriodicSmoother:
    """Windowed circular mean that respects the angle period.

    period=180 uses double-angle vectors (axial data); period=360 uses
    ordinary unit vectors (directed data). The estimate is cleared when the
    mean resultant length falls below ``min_resultant`` or when the newest
    observation is older than ``max_age_s``. Identical frame IDs are counted
    once so a repeated frame cannot accumulate confidence.
    """

    def __init__(self, period: float, window_s: float = 0.5, min_resultant: float = 0.8,
                 max_age_s: float = 0.4) -> None:
        if period not in (180.0, 360.0):
            raise ValueError("period must be 180 or 360")
        self.period = period
        self.window_s = window_s
        self.min_resultant = min_resultant
        self.max_age_s = max_age_s
        self._obs: list[_Obs] = []
        self._seen_frames: set[str] = set()

    def reset(self) -> None:
        self._obs.clear()
        self._seen_frames.clear()

    def add(self, t: float, angle: float | None, frame_key: str | None = None) -> None:
        if angle is None:
            return
        if frame_key is not None:
            if frame_key in self._seen_frames:
                return
            self._seen_frames.add(frame_key)
        self._obs.append(_Obs(t, angle))

    def estimate(self, now: float) -> tuple[float | None, float, Reason]:
        self._obs = [o for o in self._obs if now - o.t <= self.window_s]
        if not self._obs:
            return None, 0.0, Reason.NO_EVIDENCE
        if now - self._obs[-1].t > self.max_age_s:
            return None, 0.0, Reason.EVIDENCE_EXPIRED
        k = 360.0 / self.period  # 2 for axial, 1 for directed
        rad = np.radians(np.array([o.angle for o in self._obs]) * k)
        s, c = float(np.sin(rad).mean()), float(np.cos(rad).mean())
        r = math.hypot(s, c)
        if r < self.min_resultant:
            return None, r, Reason.UNSTABLE_TEMPORAL
        mean = math.degrees(math.atan2(s, c)) / k
        mean = wrap180(mean) if self.period == 180.0 else wrap360(mean)
        return mean, r, Reason.OK
