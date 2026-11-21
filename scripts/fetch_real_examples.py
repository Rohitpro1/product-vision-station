#!/usr/bin/env python3
"""Fetch the third-party photographs used for real-image inference evidence and verify SHA-256.

They are NOT committed (redistribution is governed by their own licences); only derived JSON results and
attributed overlays are. Files land in realdata/ (git-ignored).
"""
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

FILES = [
    ("off_5star_1.jpg", "https://images.openfoodfacts.org/images/products/762/220/231/8078/1.jpg",
     "a868288a6996526ef3ebcadf4a796279368a7a819ef736744ccfd6ffac63beca",
     "Open Food Facts contributors, CC BY-SA 3.0 — product 7622202318078 (5 Star, 18g), image 1"),
    ("milka_wrapper.jpg",
     "https://thumb.wikimedia.org/wikipedia/commons/thumb/a/ac/Milka_Alpine_Milk_Chocolate_chunks_and_wrapper.jpg/1920px-Milka_Alpine_Milk_Chocolate_chunks_and_wrapper.jpg",
     "e545bf2771ea6978c2128c2515aee964314b87b6ca4d4d8d8797a8602a52f142",
     "Ubcule, Wikimedia Commons, CC BY-SA 4.0 — 'Milka Alpine Milk Chocolate chunks and wrapper.jpg'"),
]


def main() -> int:
    out = Path(__file__).resolve().parents[1] / "realdata"
    out.mkdir(exist_ok=True)
    ok = True
    for name, url, sha, credit in FILES:
        p = out / name
        if not p.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "pvs-evidence-fetch/0.1"})
            p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
            time.sleep(1)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        print(("OK  " if h == sha else "BAD ") + name, "—", credit)
        ok &= h == sha
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
