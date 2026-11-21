"""Axis from masks, direction + face from validated front/back template correspondences.

Prototype direction (PDF p.4): SIFT features between the instance crop and each
enrolled front/back reference, RANSAC homography, then the reference drawing's
physical end points A and B are projected into the source frame. Gates:
inlier count, inlier spatial spread, reprojection error, and orientation-
preserving (non-mirrored) homography. An unordered mask axis never supplies a
direction.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Reason, heading_deg
from .transforms import Transform


@dataclass
class TemplateRef:
    sku: str
    side: str
    A: tuple[float, float]
    B: tuple[float, float]
    size_wh: tuple[int, int]
    kp: list
    des: np.ndarray | None
    image: np.ndarray | None = None
    cues: list | None = None  # [[x0,y0,x1,y1], ...] variant-specific regions in reference pixels
    family: str | None = None


@dataclass
class DirectionResult:
    heading_deg: float | None
    face: str
    sku: str | None
    A: list[float] | None
    B: list[float] | None
    inliers: int
    spread: float
    reproj_px: float
    score: float
    reason: Reason
    runner_up_inliers: int = 0
    H: np.ndarray | None = None  # reference -> crop homography of the winning template
    cue_scores: dict | None = None  # sku -> similarity of variant-specific cue regions
    cue_margin: float | None = None
    family: str | None = None


class TemplateMatcher:
    backend_id = "sift-homography-v1"

    def __init__(self, min_inliers: int = 14, min_spread: float = 0.12, max_reproj_px: float = 4.0) -> None:
        self.sift = cv2.SIFT_create(nfeatures=1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.refs: list[TemplateRef] = []
        self.min_inliers = min_inliers
        self.min_spread = min_spread
        self.max_reproj = max_reproj_px

    def add_reference(self, sku: str, side: str, img: np.ndarray, A, B, cues=None, family=None) -> None:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(g, None)
        self.refs.append(TemplateRef(sku, side, tuple(A), tuple(B), (img.shape[1], img.shape[0]), kp, des, img,
                                     cues, family))

    def cue_check(self, crop: np.ndarray, res: DirectionResult) -> None:
        """Compare variant-specific cue regions of same-family, same-side references in the reference frame."""
        if res.H is None or res.sku is None:
            return
        win = next(r for r in self.refs if r.sku == res.sku and r.side == res.face)
        if not win.cues:
            return
        peers = [r for r in self.refs if r.side == win.side and r.family == win.family and r.cues]
        if len(peers) < 2:
            return
        w, h = win.size_wh
        back = cv2.warpPerspective(crop, np.linalg.inv(res.H), (w, h), flags=cv2.INTER_LINEAR)
        valid = cv2.warpPerspective(np.full(crop.shape[:2], 255, np.uint8), np.linalg.inv(res.H), (w, h)) > 0
        lab = cv2.cvtColor(back, cv2.COLOR_BGR2LAB).astype(np.float32)
        scores = {}
        for r in peers:
            rl = cv2.cvtColor(r.image, cv2.COLOR_BGR2LAB).astype(np.float32)
            diffs = []
            for x0, y0, x1, y1 in win.cues:
                m = valid[y0:y1, x0:x1]
                if m.mean() < 0.8:  # cue not visible (outside crop / occluded): no evidence
                    continue
                d = np.abs(lab[y0:y1, x0:x1] - rl[y0:y1, x0:x1])[m].mean() / 255.0
                diffs.append(d)
            if diffs:
                scores[r.sku] = round(float(1.0 - np.mean(diffs)), 4)
        if len(scores) >= 2:
            ranked = sorted(scores.values(), reverse=True)
            res.cue_scores = scores
            res.cue_margin = round(ranked[0] - ranked[1], 4)

    def match(self, crop: np.ndarray, crop_T: Transform) -> DirectionResult:
        """``crop_T`` maps source-frame pixels to crop pixels."""
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(g, None)
        if des is None or len(kp) < 8:
            return DirectionResult(None, "unknown", None, None, None, 0, 0.0, 0.0, 0.0, Reason.LOW_INLIERS)
        results = []
        for ref in self.refs:
            if ref.des is None or len(ref.kp) < 8:
                continue
            pairs = self.matcher.knnMatch(ref.des, des, k=2)
            good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
            if len(good) < 8:
                results.append((0, ref, None, None, 0.0, 99.0))
                continue
            src = np.float32([ref.kp[m.queryIdx].pt for m in good])
            dst = np.float32([kp[m.trainIdx].pt for m in good])
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if H is None:
                results.append((0, ref, None, None, 0.0, 99.0))
                continue
            inl = inl.ravel().astype(bool)
            n_in = int(inl.sum())
            proj = cv2.perspectiveTransform(src[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
            reproj = float(np.sqrt(((proj - dst[inl]) ** 2).sum(1)).mean()) if n_in else 99.0
            spread = 0.0
            if n_in >= 3:
                hull = cv2.convexHull(src[inl])
                spread = float(cv2.contourArea(hull) / (ref.size_wh[0] * ref.size_wh[1]))
            results.append((n_in, ref, H, inl, spread, reproj))
        if not results:
            return DirectionResult(None, "unknown", None, None, None, 0, 0.0, 0.0, 0.0, Reason.LOW_INLIERS)
        results.sort(key=lambda r: -r[0])
        n_in, ref, H, _, spread, reproj = results[0]
        # runner-up = best inlier count from a *different* SKU (variant discrimination margin)
        runner = max([r[0] for r in results[1:] if r[1].sku != ref.sku], default=0)
        base = dict(inliers=n_in, spread=spread, reproj_px=reproj, runner_up_inliers=runner)
        if H is None or n_in < self.min_inliers:
            return DirectionResult(None, "unknown", None, None, None, score=0.0, reason=Reason.LOW_INLIERS, **base)
        if spread < self.min_spread:
            return DirectionResult(None, "unknown", None, None, None, score=0.0, reason=Reason.CLUSTERED_MATCHES, **base)
        if reproj > self.max_reproj:
            return DirectionResult(None, "unknown", None, None, None, score=0.0, reason=Reason.HIGH_REPROJECTION, **base)
        if np.linalg.det(H[:2, :2]) <= 0:  # mirrored artwork cannot be a physical view
            return DirectionResult(None, "unknown", None, None, None, score=0.0, reason=Reason.HIGH_REPROJECTION, **base)
        ab_crop = cv2.perspectiveTransform(np.float32([ref.A, ref.B]).reshape(-1, 1, 2), H).reshape(-1, 2)
        ab_src = crop_T.inverse(ab_crop)
        A, B = ab_src[0].tolist(), ab_src[1].tolist()
        score = float(min(1.0, n_in / (3.0 * self.min_inliers)))
        # face/side is ambiguous if the other side of another SKU matched nearly as well
        return DirectionResult(heading_deg(tuple(A), tuple(B)), ref.side, ref.sku, A, B, score=score,
                               reason=Reason.OK, H=H, family=ref.family, **base)


def rectify_crop(image: np.ndarray, centre: tuple[float, float], axis_deg: float, length: float, width: float,
                 base: Transform | None = None) -> tuple[np.ndarray, Transform]:
    """Rotate so the axis is horizontal and cut an axis-aligned window (original resolution)."""
    W, H = int(max(16, length)), int(max(16, width))
    M = cv2.getRotationMatrix2D(centre, axis_deg, 1.0)  # rotates by -axis in image (clockwise-positive) terms
    M[0, 2] += W / 2 - centre[0]
    M[1, 2] += H / 2 - centre[1]
    A = np.vstack([M, [0, 0, 1]])
    out = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
    return out, (base or Transform()).then(A, f"rectify_{axis_deg:.1f}")
