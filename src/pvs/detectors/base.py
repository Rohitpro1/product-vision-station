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

    def masks(self, image_bgr: np.ndarray, boxes: list[list[float]]) -> list[tuple[np.ndarray | None, str]]:
        h, w = image_bgr.shape[:2]
        out: list[tuple[np.ndarray | None, str]] = []
        for x0, y0, x1, y1 in boxes:
            rect = (int(max(0, x0)), int(max(0, y0)), int(min(w - 1, x1) - max(0, x0)), int(min(h - 1, y1) - max(0, y0)))
            if rect[2] < 8 or rect[3] < 8:
                out.append((None, "none"))
                continue
            m = np.zeros((h, w), np.uint8)
            bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
            try:
                cv2.grabCut(image_bgr, m, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
                fg = (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)
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
