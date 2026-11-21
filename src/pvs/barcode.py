"""Printed GTIN evidence: ZXing-C++ decoding, GTIN validation and instance association."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_L = ["0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111", "0001011"]
_G = ["0100111", "0110011", "0011011", "0100001", "0011101", "0111001", "0000101", "0010001", "0001001", "0010111"]
_R = ["1110010", "1100110", "1101100", "1000010", "1011100", "1001110", "1010000", "1000100", "1001000", "1110100"]
_PARITY = ["LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG", "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL"]


def gtin_check_digit(body: str) -> int:
    """GS1 mod-10 check digit for the digits *without* the check digit."""
    total = 0
    for i, ch in enumerate(reversed(body)):
        total += int(ch) * (3 if i % 2 == 0 else 1)
    return (10 - total % 10) % 10


def validate_gtin(code: str) -> tuple[bool, str]:
    """Validate a GTIN-8/12/13/14 string; leading zeros are preserved as given."""
    if not code.isdigit():
        return False, "non_digit"
    if len(code) not in (8, 12, 13, 14):
        return False, "bad_length"
    if gtin_check_digit(code[:-1]) != int(code[-1]):
        return False, "bad_check_digit"
    return True, "ok"


def to_gtin14(code: str) -> str:
    return code.zfill(14)


def ean13_modules(code: str) -> str:
    ok, why = validate_gtin(code)
    if not ok or len(code) != 13:
        raise ValueError(f"invalid EAN-13 {code}: {why}")
    par = _PARITY[int(code[0])]
    s = "101"
    for i, ch in enumerate(code[1:7]):
        s += (_L if par[i] == "L" else _G)[int(ch)]
    s += "01010"
    for ch in code[7:]:
        s += _R[int(ch)]
    return s + "101"


def render_ean13(code: str, module_px: int = 3, height_px: int = 70, quiet: int = 11) -> np.ndarray:
    """Render an EAN-13 symbol (black on white, uint8 HxW)."""
    mods = ean13_modules(code)
    w = (len(mods) + 2 * quiet) * module_px
    img = np.full((height_px, w), 255, np.uint8)
    for i, m in enumerate(mods):
        if m == "1":
            x = (quiet + i) * module_px
            img[:, x:x + module_px] = 0
    return img


@dataclass
class BarcodeRead:
    text: str
    format: str
    valid: bool
    reason: str
    polygon_px: list[list[float]]  # source-frame pixels


_SUPPORTED = {"EAN13", "EAN8", "UPCA", "UPCE", "ITF", "DataBar", "DataBarExpanded", "DataBarLimited"}


def decode(img_bgr: np.ndarray, offset_xy: tuple[float, float] = (0.0, 0.0)) -> list[BarcodeRead]:
    """Decode retail linear symbologies. ``offset_xy`` maps crop pixels to source pixels."""
    import zxingcpp

    out: list[BarcodeRead] = []
    for r in zxingcpp.read_barcodes(img_bgr):
        fmt = getattr(r.format, "name", str(r.format)).split(".")[-1]
        if fmt not in _SUPPORTED:
            continue
        text = r.text
        if fmt == "UPCA" and len(text) == 12:
            pass
        valid, why = validate_gtin(text)
        p = r.position
        poly = [[p.top_left.x, p.top_left.y], [p.top_right.x, p.top_right.y],
                [p.bottom_right.x, p.bottom_right.y], [p.bottom_left.x, p.bottom_left.y]]
        poly = [[float(x + offset_xy[0]), float(y + offset_xy[1])] for x, y in poly]
        out.append(BarcodeRead(text, fmt, valid, why, poly))
    return out


def associate(poly: list[list[float]], boxes: list[list[float]], min_inside: float = 0.9) -> int | None:
    """Return the index of the single box containing the code polygon, else None.

    The polygon must lie (>= ``min_inside`` of its corners, by centroid and
    corners) inside exactly one box; overlap with two boxes is ambiguous and
    is not bound to either.
    """
    pts = np.asarray(poly, dtype=float)
    owners = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        inside = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)).mean()
        if inside >= min_inside:
            owners.append(i)
    return owners[0] if len(owners) == 1 else None
