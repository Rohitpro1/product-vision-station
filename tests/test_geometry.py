"""Geometry contract tests. Expected values below are hand-computed (independent of the implementation)."""
import math

import cv2
import numpy as np
import pytest

from pvs.geometry import (
    AxisGate,
    PeriodicSmoother,
    Reason,
    fit_axis,
    heading_deg,
    mask_points,
    periodic_error,
    wrap180,
    wrap360,
)

# image frame: +x right, +y down, clockwise positive
CARDINAL = [((0, 0), (10, 0), 0.0),     # right
            ((0, 0), (0, 10), 90.0),    # down (clockwise from +x)
            ((0, 0), (-10, 0), 180.0),  # left
            ((0, 0), (0, -10), 270.0),  # up
            ((0, 0), (10, 10), 45.0),
            ((0, 0), (10, -10), 315.0)]


@pytest.mark.parametrize("a,b,exp", CARDINAL)
def test_heading_cardinals(a, b, exp):
    assert heading_deg(a, b) == pytest.approx(exp)


def test_wraps():
    assert wrap360(360.0) == 0.0 and wrap360(-1) == 359.0 and wrap360(725) == 5.0
    assert wrap180(180.0) == 0.0 and wrap180(-1) == 179.0 and wrap180(359) == 179.0


@pytest.mark.parametrize("p,t,P,exp", [(179, 1, 180, 2), (1, 179, 180, 2), (359, 1, 360, 2), (0, 180, 360, 180),
                                       (90, 270, 180, 0), (10, 200, 360, 170)])
def test_periodic_error(p, t, P, exp):
    assert periodic_error(p, t, P) == pytest.approx(exp)


def _bar_mask(angle, L=200, W=50, size=400):
    m = np.zeros((size, size), np.uint8)
    rect = ((size / 2, size / 2), (L, W), angle)
    cv2.fillPoly(m, [cv2.boxPoints(rect).astype(np.int32)], 1)
    return m.astype(bool)


@pytest.mark.parametrize("ang", [0, 30, 89, 90, 135, 179])
def test_axis_from_mask(ang):
    fit = fit_axis(mask_points(_bar_mask(ang)))
    assert fit.reason == Reason.OK
    assert periodic_error(fit.angle_deg, ang % 180, 180) < 1.5
    assert 0 <= fit.angle_deg < 180


def test_axis_rejects_round_and_tiny():
    m = np.zeros((200, 200), np.uint8)
    cv2.circle(m, (100, 100), 60, 1, -1)
    assert fit_axis(mask_points(m.astype(bool))).reason == Reason.NEAR_ROUND
    assert fit_axis(mask_points(_bar_mask(10, L=20, W=5))).reason == Reason.TOO_SMALL
    assert fit_axis(np.zeros((0, 2))).reason == Reason.NO_MASK


def test_axis_gate_configurable():
    f = fit_axis(mask_points(_bar_mask(0, L=120, W=90)), AxisGate(min_elongation=1.2))
    assert f.reason == Reason.OK


def test_axial_smoothing_wraps_179_1():
    s = PeriodicSmoother(180.0, window_s=1.0, min_resultant=0.8, max_age_s=1.0)
    for i, a in enumerate([179, 1, 179, 1]):
        s.add(i * 0.1, a, f"f{i}")
    m, r, why = s.estimate(0.3)
    assert why == Reason.OK and periodic_error(m, 0, 180) < 0.5 and r > 0.99


def test_directed_smoothing_wraps_359_1():
    s = PeriodicSmoother(360.0, 1.0, 0.8, 1.0)
    for i, a in enumerate([359, 1, 358, 2]):
        s.add(i * 0.1, a, f"f{i}")
    m, _, why = s.estimate(0.3)
    assert why == Reason.OK and periodic_error(m, 0, 360) < 0.5


def test_heading_180_swap_clears():
    s = PeriodicSmoother(360.0, 1.0, 0.8, 1.0)
    for i, a in enumerate([10, 190, 10, 190]):
        s.add(i * 0.1, a, f"f{i}")
    m, r, why = s.estimate(0.3)
    assert m is None and why == Reason.UNSTABLE_TEMPORAL and r < 0.1


def test_repeated_frame_does_not_accumulate():
    s = PeriodicSmoother(360.0, 1.0, 0.8, 1.0)
    for _ in range(10):
        s.add(0.0, 45.0, "same-hash")
    assert len(s._obs) == 1


def test_evidence_expires_after_gap():
    s = PeriodicSmoother(360.0, window_s=0.5, min_resultant=0.8, max_age_s=0.3)
    s.add(0.0, 45.0, "a")
    assert s.estimate(0.2)[0] == pytest.approx(45.0)
    assert s.estimate(0.4)[2] == Reason.EVIDENCE_EXPIRED
    assert s.estimate(0.9)[2] == Reason.NO_EVIDENCE


def test_back_face_turn_keeps_physical_heading():
    """Turning over about the long axis keeps A left in the canonical back view: same heading."""
    from pvs.fixtures import PACK_H, PACK_W
    # canonical front and back references both define A=(0,H/2), B=(W,H/2)
    assert heading_deg((0, PACK_H / 2), (PACK_W, PACK_H / 2)) == 0.0
    assert math.isclose(heading_deg((PACK_W, PACK_H / 2), (0, PACK_H / 2)), 180.0)


def test_rounding_never_returns_period_upper_bound():
    from pvs.geometry import wrap180, wrap360
    assert wrap360(round(359.999, 2)) == 0.0
    assert wrap180(round(179.996, 2)) == 0.0
