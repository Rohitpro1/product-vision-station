"""Fixture profile only: scripted detections replayed from generated ground truth.

These outputs exercise plumbing deterministically. They are tagged
``scripted_fixture`` and the ``fixture`` profile banner, and are never used by
the ``pretrained`` or ``station`` profiles.
"""
from __future__ import annotations

import numpy as np

from .base import Detection


class ScriptedDetector:
    backend_id = "scripted-fixture"

    def __init__(self, script: dict[str, list[dict]] | None = None, mask_from_truth: bool = True) -> None:
        self.script = script or {}
        self.mask_from_truth = mask_from_truth

    def load_ground_truth(self, gt: dict) -> None:
        for sc in gt["scenes"]:
            self.script[sc["image"]] = sc["instances"]
            self.script[sc["scene_id"]] = sc["instances"]

    def detect(self, image_bgr: np.ndarray, context: dict | None = None) -> list[Detection]:
        key = (context or {}).get("source_key")
        if not key or key not in self.script:
            if context is not None:
                context["reason"] = "fixture_no_script_for_source"
            return []
        items = self.script[key]
        h, w = image_bgr.shape[:2]
        out = []
        for it in items:
            if it.get("occluded_frac", 0) > 0.9:
                continue
            mask = None
            if self.mask_from_truth and it.get("A") and it.get("B"):
                # oriented rectangle from the truth endpoints and box: a scripted mask
                import cv2

                a, b = np.array(it["A"]), np.array(it["B"])
                d = b - a
                L = np.linalg.norm(d)
                n = np.array([-d[1], d[0]]) / max(L, 1e-9)
                x0, y0, x1, y1 = it["box_xyxy"]
                half = 0.143 * L if it.get("face") != "edge" else 0.017 * L
                poly = np.array([a + n * half, b + n * half, b - n * half, a - n * half], np.int32)
                m = np.zeros((h, w), np.uint8)
                cv2.fillPoly(m, [poly], 1)
                mask = m.astype(bool)
            out.append(Detection(box_xyxy=list(map(float, it["box_xyxy"])), score=0.99, label="package",
                                 source=self.backend_id, mask=mask,
                                 mask_source="manual" if mask is not None else "none"))
        return out
