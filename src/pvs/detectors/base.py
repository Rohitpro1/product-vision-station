"""Configurable detector / mask-provider interfaces."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np


class ModelUnavailable(RuntimeError):
    """Raised when a required model cannot be loaded. Never silently replaced."""


@dataclass
class Detection:
    box_xyxy: list[float]  # source-frame pixels
    score: float
    label: str
    source: str  # backend id, e.g. "grounding-dino-tiny@<rev>"
    mask: np.ndarray | None = None  # bool HxW in source frame
    mask_source: str = "none"
    extra: dict = field(default_factory=dict)


class Detector(Protocol):
    backend_id: str

    def detect(self, image_bgr: np.ndarray, context: dict | None = None) -> list[Detection]: ...


class MaskProvider(Protocol):
    backend_id: str

    def masks(self, image_bgr: np.ndarray, boxes: list[list[float]]) -> list[tuple[np.ndarray | None, str]]: ...


class ThresholdMaskFallback:
    """Explicitly named geometry fallback: GrabCut seeded by the box.

    Every mask from here is labelled ``geometry_fallback_threshold`` in results.
    If GrabCut collapses, the box rectangle is used and labelled
    ``geometry_fallback_box`` (axis from a box has no long-axis meaning beyond
    the box aspect and is gated by the axis elongation test).
    """

    backend_id = "grabcut-fallback"

    def __init__(self, work_side: int = 384, pad: float = 0.15) -> None:
        # GrabCut runs on a padded region around the box, downscaled so its long side is <= work_side;
        # the mask is scaled back. Full-image GrabCut cost ~4 s on a 1280 px frame for no axis benefit.
        self.work_side, self.pad = work_side, pad

    def _grabcut(self, image_bgr: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        x, y, rw, rh = rect
        px, py = int(rw * self.pad) + 2, int(rh * self.pad) + 2
        rx0, ry0, rx1, ry1 = max(0, x - px), max(0, y - py), min(w, x + rw + px), min(h, y + rh + py)
        roi = image_bgr[ry0:ry1, rx0:rx1]
        s = min(1.0, self.work_side / max(roi.shape[:2]))
        small = cv2.resize(roi, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else roi
        sh, sw = small.shape[:2]
        r = (max(0, int((x - rx0) * s)), max(0, int((y - ry0) * s)), max(1, int(rw * s)), max(1, int(rh * s)))
        r = (r[0], r[1], min(r[2], sw - 1 - r[0]), min(r[3], sh - 1 - r[1]))
        m = np.zeros((sh, sw), np.uint8)
        bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
        cv2.grabCut(small, m, r, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fg_small = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
        fg_roi = cv2.resize(fg_small, (rx1 - rx0, ry1 - ry0), interpolation=cv2.INTER_NEAREST) if s < 1 else fg_small
        fg = np.zeros((h, w), bool)
        fg[ry0:ry1, rx0:rx1] = fg_roi.astype(bool)
        return fg

    def masks(self, image_bgr: np.ndarray, boxes: list[list[float]]) -> list[tuple[np.ndarray | None, str]]:
        h, w = image_bgr.shape[:2]
        out: list[tuple[np.ndarray | None, str]] = []
        for x0, y0, x1, y1 in boxes:
            rect = (int(max(0, x0)), int(max(0, y0)), int(min(w - 1, x1) - max(0, x0)), int(min(h - 1, y1) - max(0, y0)))
            if rect[2] < 8 or rect[3] < 8:
                out.append((None, "none"))
                continue
            try:
                fg = self._grabcut(image_bgr, rect)
            except cv2.error:
                fg = np.zeros((h, w), bool)
            area = rect[2] * rect[3]
            if fg.sum() < 0.2 * area:
                fg = np.zeros((h, w), bool)
                fg[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = True
                out.append((fg, "geometry_fallback_box"))
            else:
                out.append((fg, "geometry_fallback_threshold"))
        return out
