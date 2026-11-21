"""Preview overlay: box, track, identity level, axis line (undirected), arrow (directed only)."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .schema import FrameResult, IdentityLevel
from .transforms import Transform, mirror_preview

COL = {IdentityLevel.SKU: (60, 200, 60), IdentityLevel.FAMILY: (0, 200, 255), IdentityLevel.UNKNOWN: (180, 180, 180),
       IdentityLevel.CONFLICT: (0, 0, 255), IdentityLevel.AMBIGUOUS: (255, 120, 0)}


def draw(image: np.ndarray, r: FrameResult, mirror: bool = False, stale: bool = False) -> np.ndarray:
    img = image.copy()
    T = Transform()
    if mirror:
        img, T = mirror_preview(img, T)

    def P(x, y):
        p = T.forward(np.array([[x, y]]))[0]
        return int(round(p[0])), int(round(p[1]))

    for inst in r.instances:
        c = COL[inst.identity.level]
        x0, y0, x1, y1 = inst.bbox_xyxy_px
        p0, p1 = P(x0, y0), P(x1, y1)
        cv2.rectangle(img, (min(p0[0], p1[0]), p0[1]), (max(p0[0], p1[0]), p1[1]), c, 2)
        name = inst.identity.sku or inst.identity.family or inst.identity.level.value
        label = f"{inst.track_id or inst.instance_id.split(':')[-1]} {inst.identity.level.value}:{name.split(':')[-1]}"
        label += f" {inst.pose.face.value}"
        cv2.putText(img, label, (min(p0[0], p1[0]), max(14, p0[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        L = 0.45 * max(x1 - x0, y1 - y0)
        if inst.pose.axis.available:
            a = math.radians(inst.pose.axis.value_deg)
            cv2.line(img, P(cx - L * math.cos(a), cy - L * math.sin(a)), P(cx + L * math.cos(a), cy + L * math.sin(a)),
                     (255, 255, 255), 2, cv2.LINE_AA)
        if inst.pose.heading.available:
            h = math.radians(inst.pose.heading.value_deg)
            cv2.arrowedLine(img, P(cx - 0.8 * L * math.cos(h), cy - 0.8 * L * math.sin(h)),
                            P(cx + 0.8 * L * math.cos(h), cy + 0.8 * L * math.sin(h)), (255, 0, 255), 3,
                            cv2.LINE_AA, tipLength=0.15)
            txt = f"{inst.pose.heading.value_deg:.0f}deg"
            cv2.putText(img, txt, P(cx, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
    banner = f"profile={r.profile.value} det={r.backend.get('detector', '-')[:40]} bundle={r.model_bundle} status={r.status.value}"
    if stale:
        banner += " CAMERA STALE"
    if mirror:
        banner += " (mirrored preview)"
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, banner, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img
