"""Turn OCR lines from one pack into fields: MRP, dates (MFG / EXP / best before), batch, brand.

Pure functions, no models: unit-tested on typical Indian pack wording. Anything not clearly matched is left out
rather than guessed (a wrong MRP or expiry is worse than none).
"""
from __future__ import annotations

import re
from datetime import date

from ..training.brand_map import BRANDS, brand_for

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
_MON = r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEPT?|OCT|NOV|DEC)[A-Z]*"
_DMY = re.compile(r"\b(?P<d>\d{1,2})[./\- ]\s?(?P<m>\d{1,2}|" + _MON + r")[./\- ]\s?(?P<y>\d{4}|\d{2})\b")
_MY = re.compile(r"\b(?P<m>\d{1,2}|" + _MON + r")[./\- ]\s?(?P<y>\d{4}|\d{2})\b")
_MRP = re.compile(r"(?:M\.?\s*R\.?\s*P\.?|PRICE|RS\.?|₹|INR)[^0-9]{0,25}?(\d{1,5}(?:[.,]\d{1,2})?)", re.I)
_LABELS = {
    "mfg": re.compile(r"\b(MFG|MFD|MANUF\w*|PKD|PACKED|PKG|DOM|MFD\.?\s*ON|PKD\.?\s*ON)\b", re.I),
    "exp": re.compile(r"\b(EXP\w*|USE\s*BY|BEST\s*BEFORE|BB|BBE|BEFORE)\b", re.I),
}
_BATCH = re.compile(r"\b(?:BATCH\s*(?:NO\.?)?|B\.?\s*NO\.?|LOT\s*(?:NO\.?)?|B/N|BN)\s*[:.\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,14})", re.I)
_REL = re.compile(r"BEST\s*BEFORE\s*(\d{1,2})\s*(MONTHS?|DAYS?)", re.I)


def _norm(t: str) -> str:
    t = t.upper().replace("O0", "00")
    return re.sub(r"\s+", " ", t).strip()


def _year(y: str) -> int | None:
    y = int(y)
    y = 2000 + y if y < 100 else y
    return y if 2000 <= y <= 2099 else None


def parse_date(text: str) -> tuple[date | None, str | None]:
    """First date in text -> (date, original string). Month-year dates map to the 1st of the month."""
    t = _norm(text)
    for m in list(_DMY.finditer(t)) + list(_MY.finditer(t)):
        g = m.groupdict()
        d, mo, y = int(g.get("d") or 1), g["m"], _year(g["y"])
        mo = _MONTHS.get(mo[:3]) if mo[0].isalpha() else int(mo)
        if not y or not mo or not 1 <= mo <= 12 or not 1 <= d <= 31:
            continue
        try:
            return date(y, mo, d), m.group(0)
        except ValueError:
            continue
    return None, None


def parse_fields(lines: list[str]) -> dict:
    """OCR lines of one pack (reading order) -> {mrp, mfg, exp, best_before, batch, brand, brand_text}."""
    out: dict = {}
    joined = " | ".join(lines)
    up = _norm(joined)
    m = _MRP.search(up)
    if m:
        v = float(m.group(1).replace(",", "."))
        if 1 <= v <= 20000:
            out["mrp"] = v
    # dates: look at each line and the one after its label (labels and values are often split across lines)
    for i, line in enumerate(lines):
        for key, rx in _LABELS.items():
            if key in out or not rx.search(line):
                continue
            tail = rx.split(line, maxsplit=1)[-1]
            for cand in (tail, line, lines[i + 1] if i + 1 < len(lines) else ""):
                d, raw = parse_date(cand)
                if d:
                    out[key] = {"date": d.isoformat(), "text": raw}
                    break
    r = _REL.search(up)
    if r:
        out["best_before"] = f"{r.group(1)} {r.group(2).lower()} from manufacture"
    b = _BATCH.search(up)
    if b and re.search(r"\d", b.group(1)):
        out["batch"] = b.group(1)
    for line in lines:  # brand from printed text, e.g. "Kit Kat", "Dairy Milk"
        fam = brand_for(line)
        if fam:
            out["brand"], out["brand_text"] = fam, line
            break
    if "brand" in out:
        out["brand_name"] = BRANDS.get(out["brand"], out["brand"])
    return out
