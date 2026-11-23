"""Open Food Facts as a brand-training source: real front-of-pack photos, one product per barcode.

    python -m pvs.training.off scan     --parquet food.parquet --out off_products.json
    python -m pvs.training.off download --products off_products.json --exports /data/rf

`scan` reads the Open Food Facts product database (Hugging Face dataset `openfoodfacts/product-database`,
`food.parquet`, ODbL) column by column and keeps:

- products of the chocolate makers we care about, labelled with a brand family only when the brand tags and
  product names agree (a tag that is itself a product line such as `milka` or `kitkat` gives the family; a
  maker tag such as `cadbury` needs the name to say which line). Conflicts, assortments, and non-chocolate
  products that still carry the brand (Milka cookies, KitKat cereal) are skipped;
- products from those makers that are clearly not one of the brands (Caramilk, Oreo, Milo, Tic Tac): hard
  negatives;
- a random sample of other products with a front photo (other chocolate brands, snacks and sweets, Indian
  products): negatives.

`download` fetches the selected front photos (400 px) and writes them as one more export directory in the
Roboflow COCO layout (`openfoodfacts__products/{train,valid,test}`), with one whole-image box per photo, and
adds it to `download_report.json`, so `pvs.training.retail prepare` merges it with the same near-duplicate
grouping, split rule and leakage check as every other source. The split is decided by barcode (70/15/15), so
all photos of one product stay together.

Images are CC BY-SA 3.0 (Open Food Facts contributors). They are used for training and evaluation only and are
not redistributed; `off_credits.json` lists every product used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .brand_map import _EXCLUDE, BRANDS, ambiguous, brand_for
from .progress import Progress, stage_name

UA = "pvs-brand-training/1.0 (github.com/Rohitpro1/product-vision-station)"
LICENCE = "CC BY-SA 3.0 (Open Food Facts images)"
NEG_CLASS = "openfoodfacts other product"  # must stay unmapped and not ambiguous (see tests)

# brand tags that name a product line by themselves
LINE_TAGS = {
    "milka": "mondelez:milka", "toblerone": "mondelez:toblerone", "kinder": "ferrero:kinder",
    "ferrero-rocher": "ferrero:rocher", "rocher": "ferrero:rocher", "snickers": "mars:snickers",
    "bounty": "mars:bounty", "twix": "mars:twix", "galaxy": "mars:galaxy", "m-m-s": "mars:mms", "m-ms": "mars:mms",
    "kitkat": "nestle:kitkat", "kit-kat": "nestle:kitkat", "nestle-kitkat": "nestle:kitkat",
    "milkybar": "nestle:milkybar", "munch": "nestle:munch", "bournville": "cadbury:bournville",
    "cadbury-bournville": "cadbury:bournville", "cadbury-dairy-milk": "cadbury:dairy-milk", "dairy-milk": "cadbury:dairy-milk",
    "perk": "cadbury:perk", "5-star": "cadbury:5-star", "cadbury-5-star": "cadbury:5-star", "gems": "cadbury:gems",
    "fuse": "cadbury:fuse", "hershey-s": "hersheys:chocolate", "hersheys": "hersheys:chocolate",
    "hershey": "hersheys:chocolate",
}
# umbrella makers: the product name decides the line; unmapped products become hard negatives
MAKER_TAGS = {"cadbury", "nestle", "mars", "ferrero", "mondelez", "mondelez-international", "amul", "nestle-india",
              "cadbury-india", "mars-incorporated", "mars-wrigley"}
CONFECTIONERY = re.compile(r"chocolate|confectioner|candies|candy|sweet-snacks|cocoa|bars\b|snacks", re.I)
SKIP_WORDS = re.compile(r"frozen|ice ?cream|cereal|easter|\begg|advent|hamper|gift|selection|assorted|celebrations|"
                        r"heroes|miniatures|variety|collection|tin\b|pack of|multipack|spread|drink|powder|"
                        r"biscuit|cookie|cake|brownie|muffin|dessert|pudding|yogh?urt|sauce|syrup", re.I)
ASSORTMENT = re.compile(r"selection|assorted|celebrations|heroes|miniatures|variety|collection|advent|hamper|gift",
                        re.I)
NEG_BUCKETS = {"chocolate": 2500, "snack": 2500, "india": 1500}


def label_product(brand_tags: list[str], names: list[str], cats: list[str]) -> tuple[str, str | None]:
    """-> ("brand", family) | ("negative", None) | ("skip", reason). Pure function (unit-tested)."""
    tags = {t.lower() for t in brand_tags or []}
    names = [n for n in (names or []) if n and n.strip()]
    tag_fams = {LINE_TAGS[t] for t in tags if t in LINE_TAGS}
    name_fams = {f for n in names for f in [brand_for(n)] if f}
    if any(t.startswith("amul") for t in tags) and any(re.search(r"choc|dark|almondo", n, re.I) for n in names):
        name_fams.add("amul:chocolate")
    fams = tag_fams | name_fams
    text = " ".join(names)
    if len(fams) > 1:
        return "skip", "conflicting brands"
    if fams:
        if SKIP_WORDS.search(text) or _EXCLUDE.search(text):
            return "skip", "brand on a non-bar product or assortment"
        if not names and not tag_fams:
            return "skip", "no name"
        if CONFECTIONERY.search(" ".join(cats or [])) or re.search(r"choc|bar\b|wafer|bites|candy|caramel", text, re.I) \
                or (tag_fams and not cats):
            return "brand", next(iter(fams))
        return "skip", "not confectionery"
    if not names:
        return "skip", "no name"
    if tags & (MAKER_TAGS | set(LINE_TAGS)) and ASSORTMENT.search(text):
        return "skip", "maker assortment (shows brand logos)"
    if any(ambiguous(n) for n in names) and not (SKIP_WORDS.search(text) or _EXCLUDE.search(text)):
        return "skip", "generic chocolate name"
    return "negative", None


def _fronts(images) -> list[tuple[str, int, int]]:
    seen, out = set(), []
    for im in images or []:
        k = im.get("key") or ""
        if k.startswith("front") and im.get("rev") and im.get("imgid") not in seen:
            seen.add(im.get("imgid"))
            out.append((k, int(im["rev"]), int(im.get("imgid") or 0)))
    return out


def image_url(code: str, key: str, rev: int, size: int = 400) -> str:
    c = code if len(code) <= 8 else code.zfill(13)
    path = c if len(c) <= 8 else f"{c[0:3]}/{c[3:6]}/{c[6:9]}/{c[9:]}"
    return f"https://images.openfoodfacts.org/images/products/{path}/{key}.{rev}.{size}.jpg"


def split_of(code: str) -> str:
    h = int(hashlib.sha256(code.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else "valid" if h < 85 else "test"


def scan(parquet: Path, out: Path, seed: int = 0, max_groups: int | None = None) -> dict:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet))
    cols = ["code", "brands_tags", "product_name", "categories_tags", "images", "countries_tags"]
    rng = random.Random(seed)
    keep, reasons = [], Counter()
    reservoir = {b: [] for b in NEG_BUCKETS}
    seen_n = Counter()
    n_groups = pf.metadata.num_row_groups if not max_groups else min(max_groups, pf.metadata.num_row_groups)
    prog = Progress(stage_name("off") + ":scan", n_groups, "row groups")
    for g in range(n_groups):
        for r in pf.read_row_group(g, columns=cols).to_pylist():
            fronts = _fronts(r["images"])
            if not fronts or not r["code"]:
                continue
            tags = r["brands_tags"] or []
            names = [x["text"] for x in (r["product_name"] or []) if x and x.get("text")][:6]
            cats = r["categories_tags"] or []
            rec = {"code": r["code"], "brands": tags[:8], "names": names[:3], "fronts": fronts[:3]}
            if set(tags) & (MAKER_TAGS | set(LINE_TAGS)):
                kind, fam = label_product(tags, names, cats)
                reasons[kind if kind != "skip" else f"skip:{fam}"] += 1
                if kind != "skip":
                    keep.append({**rec, "label": fam if kind == "brand" else "negative", "bucket": "maker"})
                continue
            if any(brand_for(n) for n in names) or any(ambiguous(n) for n in names) or not names:
                continue
            cs = " ".join(cats)
            bucket = ("india" if "en:india" in (r["countries_tags"] or []) else
                      "chocolate" if "chocolate" in cs else
                      "snack" if re.search(r"snack|biscuit|candies|confection|chips|crisps|beverage|cereal", cs) else None)
            if not bucket:
                continue
            seen_n[bucket] += 1
            res, cap = reservoir[bucket], NEG_BUCKETS[bucket]
            item = {**rec, "label": "negative", "bucket": bucket}
            if len(res) < cap:
                res.append(item)
            else:
                j = rng.randrange(seen_n[bucket])
                if j < cap:
                    res[j] = item
        prog.update(kept=len(keep))
    for res in reservoir.values():
        keep += res
    stats = {"row_groups": n_groups, "kept": len(keep), "maker_decisions": dict(reasons),
             "by_label": dict(Counter(k["label"] for k in keep)), "negatives_by_bucket": dict(Counter(
                 k["bucket"] for k in keep if k["label"] == "negative"))}
    out.write_text(json.dumps({"stats": stats, "products": keep}))
    return stats


def download(products: Path, exports: Path, max_per_brand: int = 1500, max_neg: int = 7000,
             per_product: int = 2, workers: int = 16, seed: int = 0) -> dict:
    import cv2
    import requests

    data = json.loads(products.read_text())["products"]
    rng = random.Random(seed)
    rng.shuffle(data)
    by = defaultdict(list)
    for p in data:
        by[p["label"]].append(p)
    sel = [p for lab, ps in by.items() for p in ps[:(max_neg if lab == "negative" else max_per_brand)]]
    jobs = [(p, k, rev) for p in sel for (k, rev, _imgid) in p["fronts"][:per_product]]
    dest = exports / "openfoodfacts__products"
    sess = requests.Session()
    sess.headers["User-Agent"] = UA

    def fetch(job):
        p, k, rev = job
        sp = split_of(p["code"])
        f = dest / sp / f"{p['code']}_{k}_{rev}.jpg"
        if not f.exists():
            for attempt in range(3):
                try:
                    r = sess.get(image_url(p["code"], k, rev), timeout=30)
                    if r.status_code == 200 and len(r.content) > 2000:
                        f.parent.mkdir(parents=True, exist_ok=True)
                        f.write_bytes(r.content)
                    break
                except Exception:  # noqa: BLE001 - retry, then give up on this photo
                    time.sleep(1 + attempt)
        if not f.exists():
            return None
        img = cv2.imread(str(f))
        if img is None or min(img.shape[:2]) < 64:
            f.unlink(missing_ok=True)
            return None
        return sp, f.name, img.shape[1], img.shape[0], p

    prog = Progress(stage_name("off") + ":download", len(jobs), "photos")
    got = []
    with ThreadPoolExecutor(workers) as ex:
        for res in ex.map(fetch, jobs):
            prog.update()
            if res:
                got.append(res)
    cats = sorted({BRANDS[p["label"]] if p["label"] != "negative" else NEG_CLASS for *_x, p in got})
    cat_id = {c: i + 1 for i, c in enumerate(cats)}
    for sp in ("train", "valid", "test"):
        rows = [g for g in got if g[0] == sp]
        coco = {"images": [], "annotations": [], "categories": [{"id": i, "name": c} for c, i in cat_id.items()]}
        for n, (_sp, name, w, h, p) in enumerate(rows, 1):
            coco["images"].append({"id": n, "file_name": name, "width": w, "height": h})
            c = BRANDS[p["label"]] if p["label"] != "negative" else NEG_CLASS
            coco["annotations"].append({"id": n, "image_id": n, "category_id": cat_id[c], "bbox": [0, 0, w, h]})
        (dest / sp).mkdir(parents=True, exist_ok=True)
        (dest / sp / "_annotations.coco.json").write_text(json.dumps(coco))
    (dest / "README.dataset.txt").write_text(f"Open Food Facts front-of-pack photos\nLicense: {LICENCE}\n")
    credits = [{"code": p["code"], "names": p["names"][:1], "label": p["label"], "photo": f"{sp}/{name}"}
               for sp, name, _w, _h, p in got]
    (exports / "off_credits.json").write_text(json.dumps(credits))
    rep_path = exports / "download_report.json"
    report = json.loads(rep_path.read_text()) if rep_path.exists() else []
    report = [r for r in report if not (r.get("ws") == "openfoodfacts")]
    report.append({"ws": "openfoodfacts", "project": "products", "version": "food.parquet", "licence": LICENCE,
                   "ok": True, "photos": len(got)})
    rep_path.write_text(json.dumps(report, indent=1))
    stats = {"photos": len(got), "products": len({g[4]["code"] for g in got}),
             "photos_by_label": dict(Counter(g[4]["label"] for g in got)),
             "photos_by_split": dict(Counter(g[0] for g in got)), "failed": len(jobs) - len(got)}
    (exports / "off_download_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pvs.training.off")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--parquet", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max-groups", type=int, default=None)
    d = sub.add_parser("download")
    d.add_argument("--products", required=True)
    d.add_argument("--exports", required=True)
    d.add_argument("--max-per-brand", type=int, default=1500)
    d.add_argument("--max-neg", type=int, default=7000)
    d.add_argument("--per-product", type=int, default=2)
    d.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    if a.cmd == "scan":
        print(json.dumps(scan(Path(a.parquet), Path(a.out), max_groups=a.max_groups), indent=1))
    else:
        print(json.dumps(download(Path(a.products), Path(a.exports), a.max_per_brand, a.max_neg, a.per_product,
                                  a.workers), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
