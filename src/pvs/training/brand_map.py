"""Map free-text dataset class names to chocolate brand families.

Public datasets spell the same brand many ways ("DairyMilk", "Cadbury Diary Milk", "dairy milk silk hazel nut",
"5Star_Bites_Cadbury", "fivestar"). Rules are checked in order; exclusions come first so that, for example,
"Kurkure Masala Munch" (a snack) never becomes Munch and "Bournvita" (a drink) never becomes Bournville.
Unmapped names return None: the crop is still a "package" for the detector and an "other product" negative
for the brand gallery, unless `ambiguous()` says the name could be one of the brands (generic "cadbury",
"chocolate", "Cadbury Chocolate Berry"...). Ambiguous crops are left out of brand training and evaluation
altogether: used as negatives they would teach the model that a Dairy Milk pack is "not a brand".
"""
from __future__ import annotations

import re

# family id -> display name (shown on the site)
BRANDS: dict[str, str] = {
    "cadbury:5-star": "Cadbury 5 Star",
    "cadbury:dairy-milk": "Cadbury Dairy Milk",
    "cadbury:perk": "Cadbury Perk",
    "cadbury:bournville": "Cadbury Bournville",
    "cadbury:gems": "Cadbury Gems",
    "cadbury:eclairs": "Cadbury Eclairs",
    "cadbury:fuse": "Cadbury Fuse",
    "nestle:kitkat": "Nestlé KitKat",
    "nestle:munch": "Nestlé Munch",
    "nestle:milkybar": "Nestlé Milkybar",
    "mars:snickers": "Snickers",
    "mars:bounty": "Bounty",
    "mars:mars": "Mars",
    "mars:twix": "Twix",
    "mars:galaxy": "Galaxy",
    "mars:mms": "M&M's",
    "amul:chocolate": "Amul Chocolate",
    "ferrero:rocher": "Ferrero Rocher",
    "ferrero:kinder": "Kinder",
    "mondelez:toblerone": "Toblerone",
    "mondelez:milka": "Milka",
    "hersheys:chocolate": "Hershey's",
}

# names that contain a brand word but are not that chocolate
_EXCLUDE = re.compile(
    r"kurkure|masala ?munch|munchies|biskit ?munch|bournvita|horlicks|syrup|ice ?cream|cone|milkshake|"
    r"chocobakes|choco ?pie|chocos\b|cookie|biscuit|cake|wafer ?roll|drink|shake|milk ?powder|cocoa ?powder|"
    r"ketchup|spread",
    re.I)

_RULES: list[tuple[str, re.Pattern]] = [
    ("cadbury:5-star", re.compile(r"5 ?_?star|five ?star|fivestar|^\s*star ?(3d|oreo|bites)\b", re.I)),
    ("cadbury:dairy-milk", re.compile(r"dair[yi] ?_?milk|diary ?milk|\bsilk\b|fruit ?(and|&|n) ?nut|\bcdm\b", re.I)),
    ("cadbury:perk", re.compile(r"\bperk", re.I)),
    ("cadbury:bournville", re.compile(r"bournville|bourneville", re.I)),
    ("cadbury:gems", re.compile(r"\bgems\b", re.I)),
    ("cadbury:eclairs", re.compile(r"eclairs?|choclairs", re.I)),
    ("cadbury:fuse", re.compile(r"\bfuse\b", re.I)),
    ("nestle:kitkat", re.compile(r"kit ?_?kat", re.I)),
    ("nestle:munch", re.compile(r"\bmunch\b|munch ?max|munch ?_?\d", re.I)),
    ("nestle:milkybar", re.compile(r"milky ?_?bar", re.I)),
    ("mars:snickers", re.compile(r"snicke?rs|snikers", re.I)),
    ("mars:bounty", re.compile(r"\bbounty\b", re.I)),
    ("mars:twix", re.compile(r"\btwix\b", re.I)),
    ("mars:galaxy", re.compile(r"\bgalaxy\b", re.I)),
    ("mars:mms", re.compile(r"\bm ?(&|_|and)? ?m'?s\b|m_m fun", re.I)),
    ("mars:mars", re.compile(r"^\s*mars(\s+(bar|chocolate))?\s*$", re.I)),
    ("amul:chocolate", re.compile(r"amul.*(dark|chocolate|choco|almondo|fruit)", re.I)),
    ("ferrero:rocher", re.compile(r"ferr?ero|rocher|rucher", re.I)),
    ("ferrero:kinder", re.compile(r"\bkinder\b", re.I)),
    ("mondelez:toblerone", re.compile(r"toblerone", re.I)),
    ("mondelez:milka", re.compile(r"\bmilka\b", re.I)),
    ("hersheys:chocolate", re.compile(r"hershey|hersey|harsys|\bkisses\b", re.I)),
]


# a chocolate-ish name that did not map to a brand may still be one of them ("cadbury", "catbory chocolate")
_CHOCOLATEY = re.compile(r"cadbury|catbory|cadbary|nestl[eé]|\bmars\b|ferr?ero|mondelez|choc(olate|olates|o)?\b|"
                         r"chocolate|candy|candies|sweets?\b|\bbars?\b|confection", re.I)


def _norm(name: str) -> str:
    s = re.sub(r"[_\-/]+", " ", name)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)  # DairyMilk -> Dairy Milk
    return re.sub(r"\s+", " ", s).strip()


def brand_for(cls_name: str) -> str | None:
    n = _norm(cls_name)
    # "5Star Oreo"/"5star 3D" are still 5 Star; check before the exclusion list (which has "cookie" etc.)
    if _RULES[0][1].search(n) and not re.search(r"bournvita", n, re.I):
        return "cadbury:5-star"
    if _EXCLUDE.search(n):
        return None
    for fam, rx in _RULES[1:]:
        if rx.search(n):
            return fam
    return None


def ambiguous(cls_name: str) -> bool:
    """True when a name did not map to a brand but could be one: generic maker or 'chocolate' labels.

    Clear non-chocolate products (biscuits, drinks, ice cream... see _EXCLUDE) are not ambiguous: they are useful
    hard negatives. Names that map to a brand are never ambiguous."""
    if brand_for(cls_name):
        return False
    n = _norm(cls_name)
    return bool(_CHOCOLATEY.search(n)) and not _EXCLUDE.search(n)
