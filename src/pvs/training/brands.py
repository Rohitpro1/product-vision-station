"""Chocolate-brand identity from public data: DINOv2 (ONNX) gallery + thresholds chosen on validation only.

    python -m pvs.training.brands build --data /data/retail --out /run/brands --device cuda
    python -m pvs.training.brands e2e   --data /data/retail --bundle /run/bundle --out /run/brands

`build` exports DINOv2-small to ONNX (parity-checked against PyTorch), embeds ground-truth crops, builds a
gallery from TRAIN crops (each brand is a catalogue family; crops of every other product are explicit
`negative:` references), chooses `family_score`/`tie_margin` on VALIDATION so that wrong accepts stay
<= 1 % (other products called a brand, or the wrong brand), then reports TEST metrics using the production
fusion rules (pvs.fusion.decide). `e2e` runs the full hosted Pipeline on held-out test photos.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..gallery import Gallery, Ref, dino_pool, dino_preprocess
from .brand_map import BRANDS

NEG = "negative:other-product"


def _crops(data: Path, items: list[dict], split: str, max_other_per_class: int, rng: random.Random):
    out, labels, per = [], [], Counter()
    sel = [it for it in items if it["split"] == split]
    rng.shuffle(sel)
    for it in sel:
        img = None
        for b in it["boxes"]:
            lab = b["brand"] or f"other:{b['class']}"
            if not b["brand"] and per[lab] >= max_other_per_class:
                continue
            if img is None:
                img = cv2.imread(str(data / it["image"]))
                if img is None:
                    break
            x0, y0, x1, y1 = b["box"]
            w, h = x1 - x0, y1 - y0
            X0, Y0 = int(max(0, x0 - 0.1 * w)), int(max(0, y0 - 0.1 * h))  # same 10 % expansion as the pipeline
            X1, Y1 = int(min(img.shape[1], x1 + 0.1 * w)), int(min(img.shape[0], y1 + 0.1 * h))
            c = img[Y0:Y1, X0:X1]
            if c.size and min(c.shape[:2]) >= 16:
                out.append(np.ascontiguousarray(c))
                labels.append(lab)
                per[lab] += 1
    return out, labels


class _TorchDino:
    def __init__(self, repo: str, revision: str, device: str):
        import torch
        from transformers import AutoModel

        self.torch, self.device = torch, device
        self.model = AutoModel.from_pretrained(repo, revision=revision).to(device).eval()

    def hidden(self, x: np.ndarray):
        with self.torch.inference_mode():
            t = self.torch.from_numpy(x).to(self.device)
            with self.torch.autocast("cuda", enabled=self.device.startswith("cuda")):
                h = self.model(pixel_values=t).last_hidden_state
            return h.float().cpu().numpy()

    def embed(self, crops: list[np.ndarray], batch: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(crops), batch):
            out.append(dino_pool(self.hidden(dino_preprocess(crops[i:i + batch]))))
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    def export_onnx(self, path: Path) -> None:
        m = self.model.float().to("cpu")
        dummy = self.torch.from_numpy(dino_preprocess([np.zeros((60, 200, 3), np.uint8)] * 2))
        self.torch.onnx.export(m, (dummy,), str(path), input_names=["pixel_values"], output_names=["last_hidden_state"],
                               dynamic_axes={"pixel_values": {0: "batch"}, "last_hidden_state": {0: "batch"}},
                               opset_version=17, dynamo=False)
        self.model = m.to(self.device)


def _search(V: np.ndarray, G: np.ndarray, gl: np.ndarray, labels: list[str]):
    """Vectorised Gallery.search: best score per label, top label, margin to best *different* label."""
    order = np.argsort(gl, kind="stable")
    Gs, gls = G[order], gl[order]
    starts = np.r_[0, np.flatnonzero(np.diff(gls)) + 1]
    tops, s0s, s1s, comps = [], [], [], []
    for i in range(0, len(V), 512):
        S = V[i:i + 512] @ Gs.T
        best = np.maximum.reduceat(S, starts, axis=1)  # (q, n_labels)
        idx = np.argsort(-best, axis=1)[:, :2]
        r = np.arange(len(best))
        tops.append(gls[starts][idx[:, 0]])
        comps.append(gls[starts][idx[:, 1]])
        s0s.append(best[r, idx[:, 0]])
        s1s.append(best[r, idx[:, 1]])
    top, comp = np.concatenate(tops), np.concatenate(comps)
    s0, s1 = np.concatenate(s0s), np.concatenate(s1s)
    return [labels[t] for t in top], s0, s0 - s1, [labels[c] for c in comp]


def _accept(top, s0, margin, fam_score, tie_margin):
    """Fusion rule R4_visual_family with sku_score disabled (one SKU per brand => competitor never same family)."""
    is_brand = np.array([not t.startswith("negative:") for t in top])
    return is_brand & (s0 >= fam_score) & (margin >= tie_margin)


def _cp(k: int, n: int):
    from ..evaluation import clopper_pearson, one_sided_upper

    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": clopper_pearson(k, n) if n else None,
            "upper95": one_sided_upper(k, n) if n else None}


def build(a) -> dict:
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads((data / "manifest.json").read_text())["items"]
    man = json.loads(Path(a.manifest).read_text())["models"]["dinov2"]
    rng = random.Random(0)
    t0 = time.time()
    dino = _TorchDino(man["repo"], man["revision"], a.device)

    tr_c, tr_l = _crops(data, items, "train", a.neg_per_class, rng)
    va_c, va_l = _crops(data, items, "val", a.neg_per_class, rng)
    te_c, te_l = _crops(data, items, "test", a.neg_per_class, rng)
    counts = {s: Counter(lab if not lab.startswith("other:") else "other" for lab in L)
              for s, L in (("train", tr_l), ("val", va_l), ("test", te_l))}
    keep = [b for b in BRANDS if counts["train"][b] >= a.min_train and counts["val"][b] + counts["test"][b] >= 1]
    print(json.dumps({"crop_counts": {s: dict(c) for s, c in counts.items()}, "brands_kept": keep}), flush=True)

    def lab_of(lab):  # brands not kept become negatives
        return lab if lab in keep else NEG

    # gallery from TRAIN: <= refs_per_brand per brand, <= max_negatives others; canonical + rot180 views
    by = defaultdict(list)
    for c, lab in zip(tr_c, tr_l):
        by[lab_of(lab)].append(c)
    sel_c, sel_l = [], []
    for lab, cs in by.items():
        rng.shuffle(cs)
        cap = a.max_negatives if lab == NEG else a.refs_per_brand
        sel_c += cs[:cap]
        sel_l += [lab] * min(cap, len(cs))
    views = sel_c + [cv2.rotate(c, cv2.ROTATE_180) for c in sel_c]
    view_l = sel_l + sel_l
    G = dino.embed(views)
    labels = sorted(set(view_l))
    gl = np.array([labels.index(x) for x in view_l])

    def run(crops, labs):
        V = dino.embed(crops)
        truth = [lab_of(x) for x in labs]
        return V, truth, _search(V, G, gl, labels)

    Vv, tv, (topv, s0v, mv, _) = run(va_c, va_l)
    Vt, tt, (topt, s0t, mt, compt) = run(te_c, te_l)

    # threshold choice on VALIDATION: maximise correct brand accepts with wrong accepts <= max_wrong_rate
    tv_a = np.array(tv)
    best = None
    for fs in np.round(np.arange(0.40, 0.96, 0.01), 2):
        for tm in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
            acc = _accept(topv, s0v, mv, fs, tm)
            pred = np.array(topv)
            correct = int((acc & (pred == tv_a)).sum())
            neg_fa = int((acc & (tv_a == NEG)).sum())
            wrong_brand = int((acc & (tv_a != NEG) & (pred != tv_a)).sum())
            n_neg, n_brand = int((tv_a == NEG).sum()), int((tv_a != NEG).sum())
            ok = (neg_fa <= a.max_wrong_rate * max(1, n_neg)) and (wrong_brand <= a.max_wrong_rate * max(1, n_brand))
            if ok and (best is None or correct > best["correct"]):
                best = {"family_score": float(fs), "tie_margin": float(tm), "correct": correct, "neg_fa": neg_fa,
                        "wrong_brand": wrong_brand, "n_neg": n_neg, "n_brand": n_brand}
    if best is None:
        raise SystemExit("no threshold meets the validation false-accept limit")

    # thresholds file for the hosted profile (production rules read it); sku_score > 1 disables SKU-level claims
    base = yaml.safe_load((Path(a.base_thresholds)).read_text())
    base.update(version="thr-brands-1", calibrated=False, sku_score=1.01, family_score=best["family_score"],
                tie_margin=best["tie_margin"])
    (out / "thresholds_brands.yaml").write_text(
        "# Chosen on the VALIDATION split of the public retail datasets (pvs.training.brands); raw cosines,\n"
        "# not probability-calibrated. Family (brand) level only: sku_score > 1 disables SKU claims.\n"
        + yaml.safe_dump(base, sort_keys=False))

    # TEST metrics with the real fusion rules (one call per crop) + agreement check with the vectorised rule
    from ..config import Thresholds
    from ..fusion import FusionInput, decide
    from ..gallery import GalleryHit
    from ..schema import ScoreType

    thr = Thresholds.model_validate(base)
    fam = {lab: (None if lab == NEG else lab) for lab in labels}
    pred_level, pred_fam = [], []
    for i in range(len(Vt)):
        hit = GalleryHit([{"sku": topt[i], "family": fam[topt[i]] or "", "side": "any", "score": float(s0t[i]),
                           "reference_revision": "brands-1"},
                          {"sku": compt[i], "family": fam[compt[i]] or "", "side": "any",
                           "score": float(s0t[i] - mt[i]), "reference_revision": "brands-1"}],
                         topt[i], fam[topt[i]], "any", float(s0t[i]), float(mt[i]), compt[i])
        idn = decide(FusionInput(gallery=hit, barcodes=[], template=None), thr, {}, ScoreType.RAW_COSINE)
        pred_level.append(idn.level.value)
        pred_fam.append(idn.family if idn.level.value == "family" else None)
    vec = _accept(topt, s0t, mt, best["family_score"], best["tie_margin"])
    agree = float(np.mean([(p is not None) == bool(v) for p, v in zip(pred_fam, vec)]))
    tt_a = np.array(tt)
    per_brand = {}
    for b in keep:
        n = int((tt_a == b).sum())
        tp = sum(1 for p, t in zip(pred_fam, tt) if p == b and t == b)
        claimed = sum(1 for p in pred_fam if p == b)
        per_brand[b] = {"test_crops": n, "recall": _cp(tp, n), "precision": _cp(tp, claimed),
                        "train_refs": int(sum(1 for x in sel_l if x == b))}
    neg_n = int((tt_a == NEG).sum())
    neg_fa = sum(1 for p, t in zip(pred_fam, tt) if t == NEG and p is not None)
    brand_n = int((tt_a != NEG).sum())
    wrong = sum(1 for p, t in zip(pred_fam, tt) if t != NEG and p is not None and p != t)
    correct = sum(1 for p, t in zip(pred_fam, tt) if t != NEG and p == t)
    confusion = Counter(f"{t}->{p or 'unknown'}" for p, t in zip(pred_fam, tt) if t != NEG)

    dino.export_onnx(out / "dinov2_small.onnx")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out / "dinov2_small.onnx"), providers=["CPUExecutionProvider"])
    probe = te_c[:16] or va_c[:16]
    x = dino_preprocess(probe)
    parity = float(np.abs(dino_pool(sess.run(None, {"pixel_values": x})[0]) - dino_pool(dino.hidden(x))).max())

    import hashlib

    onnx_sha = hashlib.sha256((out / "dinov2_small.onnx").read_bytes()).hexdigest()
    cat_rev = f"brands-{onnx_sha[:8]}-{len(view_l)}"
    refs = [Ref(sku=lab, family=fam[lab], side="any", revision="brands-1", view=v)
            for lab, v in zip(view_l, ["canonical"] * len(sel_l) + ["rot180"] * len(sel_l))]
    g = Gallery(G, refs, f"dinov2-onnx:{onnx_sha[:12]}", cat_rev)
    np.savez_compressed(out / "gallery_brands.npz", vectors=g.vectors.astype(np.float16),
                        refs=json.dumps([r.__dict__ for r in refs]),
                        meta=json.dumps({"backend": g.backend_id, "catalogue_revision": cat_rev}))
    catalogue = {"catalogue": "brands", "revision": cat_rev, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "products": [{"sku": b, "family": b, "display_name": BRANDS[b], "gtins": [], "quantity": None,
                               "variant": None, "artwork_revision": "public-data-1", "references": [],
                               "physical_ends": {}, "supported_faces": ["front", "back"],
                               "provenance": "embedding references from public Roboflow Universe datasets "
                                             "(see brands_report.json)",
                               "usage_rights": "derived embeddings only; source images not redistributed",
                               "approval_state": "approved"} for b in keep],
                 "negatives": [], "history": [{"revision": cat_rev, "change": "built by pvs.training.brands"}]}
    (out / "catalogue_brands.json").write_text(json.dumps(catalogue, indent=1))

    rep = {"brands": keep, "display_names": {b: BRANDS[b] for b in keep}, "crop_counts": {s: dict(c) for s, c in counts.items()},
           "gallery": {"refs": len(view_l), "brand_refs_per_view": dict(Counter(sel_l)),
                       "backend": g.backend_id, "catalogue_revision": cat_rev},
           "threshold_choice_on_val": best,
           "test_crop_level": {"brand_crops": brand_n, "correct_brand": _cp(correct, brand_n),
                               "wrong_brand": _cp(wrong, brand_n),
                               "other_products_called_a_brand": _cp(neg_fa, neg_n),
                               "per_brand": per_brand, "confusion": dict(confusion.most_common(60)),
                               "fusion_vs_vectorised_agreement": agree},
           "onnx_parity_max_abs": parity, "dinov2": f"{man['repo']}@{man['revision'][:12]}",
           "seconds": round(time.time() - t0, 1),
           "caveats": ["Ground-truth crops from community annotations; frames of one product are correlated.",
                       "Thresholds are raw cosines chosen on validation, not calibrated probabilities.",
                       "Brand level only; pack size / flavour variants are not distinguished."]}
    (out / "brands_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def _e2e_worker(args):
    bundle, data, chunk = args
    import os

    os.chdir(bundle)
    from ..capture import frame_from_file
    from ..config import load_station
    from ..evaluation import match
    from ..pipeline import Pipeline
    from ..schema import Profile

    cfg = load_station(Path(bundle) / "configs/station.hosted.yaml")
    p = Pipeline(cfg, Profile.STATION, root=Path(bundle))
    kept = {x["family"] for x in json.loads((Path(bundle) / cfg.catalogue).read_text())["products"]}
    rows = []
    for it in chunk:
        r = p.process(frame_from_file(str(Path(data) / it["image"])))
        preds = [(i.bbox_xyxy_px, i.identity.family if i.identity.level.value == "family" else None)
                 for i in r.instances]
        gts = [(b["box"], b["brand"] if b["brand"] in kept else None) for b in it["boxes"]]
        pairs = match([g for g, _ in gts], [p_ for p_, _ in preds], 0.5)
        rows.append({"image": it["image"], "gt": len(gts), "pred": len(preds), "status": r.status.value,
                     "pairs": [(gts[gi][1], preds[pi][1]) for gi, pi in pairs],
                     "unmatched_pred_brands": [preds[k][1] for k in range(len(preds)) if k not in {pi for _, pi in pairs}],
                     "ms": r.timing_ms.get("total")})
    return rows


def e2e(a) -> dict:
    from multiprocessing import get_context

    data, bundle = Path(a.data), Path(a.bundle)
    items = [it for it in json.loads((data / "manifest.json").read_text())["items"] if it["split"] == "test"]
    rng = random.Random(1)
    branded = [it for it in items if any(b["brand"] for b in it["boxes"])]
    other = [it for it in items if not any(b["brand"] for b in it["boxes"])]
    rng.shuffle(branded)
    rng.shuffle(other)
    sel = branded[:a.max_branded] + other[:a.max_other]
    chunks = [(str(bundle), str(data), sel[i::a.workers]) for i in range(a.workers)]
    with get_context("spawn").Pool(a.workers) as pool:
        rows = [r for part in pool.map(_e2e_worker, chunks) for r in part]
    gt = sum(r["gt"] for r in rows)
    matched = sum(len(r["pairs"]) for r in rows)
    pairs = [p for r in rows for p in r["pairs"]]
    b_pairs = [(t, p) for t, p in pairs if t]
    o_pairs = [(t, p) for t, p in pairs if not t]
    kept = {x["family"] for x in json.loads((bundle / "catalogues/brands/catalogue.json").read_text())["products"]}
    n_brand_gt = sum(1 for it in sel for b in it["boxes"] if b["brand"] in kept)
    rep = {"images": len(rows), "branded_images": min(len(branded), a.max_branded),
           "detection_recall_iou50": _cp(matched, gt),
           "pred_boxes": sum(r["pred"] for r in rows),
           "unmatched_pred_boxes": sum(len(r["unmatched_pred_brands"]) for r in rows),
           "brand_gt_boxes": n_brand_gt,
           "brand_correct_end_to_end": _cp(sum(1 for t, p in b_pairs if p == t), n_brand_gt),
           "brand_wrong_end_to_end": _cp(sum(1 for t, p in b_pairs if p and p != t), max(1, len(b_pairs))),
           "other_products_called_a_brand": _cp(sum(1 for t, p in o_pairs if p), len(o_pairs)),
           "unmatched_boxes_called_a_brand": sum(1 for r in rows for b in r["unmatched_pred_brands"] if b),
           "median_ms": float(np.median([r["ms"] for r in rows if r["ms"]])) if rows else None,
           "note": "full hosted Pipeline (trained detector + GrabCut + DINOv2-ONNX gallery + fusion rules) on held-out "
                   "test photos; boxes matched to ground truth at IoU >= 0.5; brands not in the gallery count as other."}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "e2e_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pvs.training.brands")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--device", default="cuda")
    b.add_argument("--manifest", default="models/manifest.json")
    b.add_argument("--base-thresholds", default="configs/thresholds.yaml")
    b.add_argument("--refs-per-brand", type=int, default=250)
    b.add_argument("--max-negatives", type=int, default=3000)
    b.add_argument("--neg-per-class", type=int, default=25)
    b.add_argument("--min-train", type=int, default=15)
    b.add_argument("--max-wrong-rate", type=float, default=0.01)
    e = sub.add_parser("e2e")
    e.add_argument("--data", required=True)
    e.add_argument("--bundle", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--max-branded", type=int, default=300)
    e.add_argument("--max-other", type=int, default=200)
    a = ap.parse_args(argv)
    rep = build(a) if a.cmd == "build" else e2e(a)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("crop_counts",)}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
