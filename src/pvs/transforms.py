"""Coordinate transforms between the declared source frame and every derived image.

Every derived image (rectified, resized, letterboxed, cropped, preview) keeps a
3x3 homogeneous matrix ``M`` mapping *source-frame* pixels to that image. All
reported points are mapped back with ``M^-1`` so outputs stay in the declared
source pixel frame regardless of preview mirroring or model input size.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# EXIF orientation tag -> (rotate_code or None, flip_horizontal)
_EXIF = {
    1: (None, False),
    2: (None, True),
    3: (cv2.ROTATE_180, False),
    4: (cv2.ROTATE_180, True),
    5: (cv2.ROTATE_90_CLOCKWISE, True),
    6: (cv2.ROTATE_90_CLOCKWISE, False),
    7: (cv2.ROTATE_90_COUNTERCLOCKWISE, True),
    8: (cv2.ROTATE_90_COUNTERCLOCKWISE, False),
}


@dataclass
class Transform:
    """Source-frame -> derived-image mapping."""

    M: np.ndarray = field(default_factory=lambda: np.eye(3))
    steps: list[str] = field(default_factory=list)

    def then(self, A: np.ndarray, name: str) -> Transform:
        return Transform(A @ self.M, self.steps + [name])

    def forward(self, pts: np.ndarray) -> np.ndarray:
        return _apply(self.M, pts)

    def inverse(self, pts: np.ndarray) -> np.ndarray:
        return _apply(np.linalg.inv(self.M), pts)


def _apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    h = np.hstack([p, np.ones((len(p), 1))]) @ M.T
    return h[:, :2] / h[:, 2:3]


def translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def scale(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)


def hflip(width: int) -> np.ndarray:
    return np.array([[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]], dtype=np.float64)


def apply_exif(img: np.ndarray, orientation: int | None) -> tuple[np.ndarray, Transform]:
    """Apply EXIF orientation exactly once. Returns the upright image and the
    transform from the *stored* (raw sensor) pixels to the upright pixels.

    The upright image is the declared source frame for the API.
    """
    rot, flip = _EXIF.get(int(orientation or 1), (None, False))
    h, w = img.shape[:2]
    T = Transform()
    out = img
    if flip:
        out = cv2.flip(out, 1)
        T = T.then(hflip(w), "exif_flip")
    if rot is not None:
        hh, ww = out.shape[:2]
        if rot == cv2.ROTATE_90_CLOCKWISE:
            A = np.array([[0, -1, hh - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        elif rot == cv2.ROTATE_90_COUNTERCLOCKWISE:
            A = np.array([[0, 1, 0], [-1, 0, ww - 1], [0, 0, 1]], dtype=np.float64)
        else:
            A = np.array([[-1, 0, ww - 1], [0, -1, hh - 1], [0, 0, 1]], dtype=np.float64)
        out = cv2.rotate(out, rot)
        T = T.then(A, f"exif_rot_{orientation}")
    return out, T


def resize(img: np.ndarray, size_wh: tuple[int, int], base: Transform | None = None
           ) -> tuple[np.ndarray, Transform]:
    h, w = img.shape[:2]
    W, H = size_wh
    # pixel-centre convention: x' = (x + 0.5) * s - 0.5
    sx, sy = W / w, H / h
    A = translate(-0.5, -0.5) @ scale(sx, sy) @ translate(0.5, 0.5)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), (base or Transform()).then(A, f"resize_{W}x{H}")


def letterbox(img: np.ndarray, size: int, base: Transform | None = None, pad: int = 114
              ) -> tuple[np.ndarray, Transform]:
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = round(w * s), round(h * s)
    r, T = resize(img, (nw, nh), base)
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size) + img.shape[2:], pad, dtype=img.dtype)
    out[py:py + nh, px:px + nw] = r
    return out, T.then(translate(px, py), f"pad_{px}_{py}")


def crop(img: np.ndarray, box_xyxy: tuple[float, float, float, float], expand: float = 0.0,
         base: Transform | None = None) -> tuple[np.ndarray, Transform, tuple[int, int, int, int]]:
    """Crop a box (in ``img`` pixels), expanded by ``expand`` fraction, clipped to the image."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box_xyxy
    bw, bh = x1 - x0, y1 - y0
    x0 -= bw * expand / 2
    x1 += bw * expand / 2
    y0 -= bh * expand / 2
    y1 += bh * expand / 2
    ix0, iy0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    ix1, iy1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    out = img[iy0:iy1, ix0:ix1]
    return out, (base or Transform()).then(translate(-ix0, -iy0), f"crop_{ix0}_{iy0}"), (ix0, iy0, ix1, iy1)


def mirror_preview(img: np.ndarray, base: Transform | None = None) -> tuple[np.ndarray, Transform]:
    """Mirror for display only. Reported geometry must never use this frame."""
    return cv2.flip(img, 1), (base or Transform()).then(hflip(img.shape[1]), "preview_mirror")


def angle_in_image(T: Transform, a: tuple[float, float], b: tuple[float, float]) -> float:
    """Heading of source-frame segment A->B as it appears in a derived image."""
    from .geometry import heading_deg

    p = T.forward(np.array([a, b]))
    return heading_deg(tuple(p[0]), tuple(p[1]))
