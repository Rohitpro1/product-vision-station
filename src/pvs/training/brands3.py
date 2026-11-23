"""Run 3: fine-tuned DINOv2 brand model, compared fairly with what is live.

    python -m pvs.training.brands3 crops    --data /data/retail --out /run/crops
    python -m pvs.training.brands3 train    --crops /run/crops --out /run/ft --device cuda:0
    python -m pvs.training.brands3 evaluate --crops /run/crops --ft /run/ft --v2 /run/v2bundle --out /run/brands
    python -m pvs.training.brands3 external --bundle /run/bundle --list configs/brand_external_check.yaml --out x.json

`crops` cuts every ground-truth box (same 10 % expansion as the pipeline) from the prepared manifest into a crop
cache, tagged with its split, label and source: `shelf` (Roboflow photos) or `closeup` (Open Food Facts
front-of-pack photos). Boxes whose class name is ambiguous (generic "cadbury", "chocolate") are left out.

`train` fine-tunes DINOv2-small as a brand embedder: CosFace head over the kept brands plus one "other product"
class, class- and source-balanced batches (half brands, half others; each brand split between shelf and close-up
photos when it has both), and augmentation for what the station and phones produce (rotation incl. 180°, crop
jitter, blur, low resolution, JPEG, colour, perspective). The pooled feature is the one serving uses
(`dino_pool`), so the exported backbone drops into the existing hosted pipeline unchanged. The checkpoint kept is
the one with the most correct names on a validation subset at ≤ 1 % wrong names.

`evaluate` builds the gallery for (a) the original DINOv2 and (b) the fine-tuned one from TRAIN crops, chooses
thresholds on VALIDATION (≤ 1 % wrong names and ≤ 1 % other products called a brand), retires brands whose
VALIDATION precision is low (decided on validation, not test), picks the better model on validation, and only
then reports TEST, overall and per source. The live v2 release is scored on the same TEST crops for comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..gallery import Gallery, Ref, dino_pool, dino_preprocess
from .brand_map import BRANDS
from .brands import _accept, _cp, _search
from .progress import Progress, stage_name

NEG = "negative:other-product"
OFF_PREFIX = "openfoodfacts__"


# ------------------------------------------------------------------ crops
def crops(a) -> dict:
    data, out = Path(a.data), Path(a.out)
    items = json.loads((data / "manifest.json").read_text())["items"]
    rng = random.Random(0)
    rng.shuffle(items)
    cap = {"train": a.neg_per_class_train, "val": a.neg_per_class_eval, "test": a.neg_per_class_eval}
    per = Counter()
    per_split = Counter()
    index = []
    prog = Progress(stage_name("crops"), len(items), "images")
    for it in items:
        prog.update(crops=len(index))
        sp = it["split"]
        src = "closeup" if it["export"].startswith(OFF_PREFIX) else "shelf"
        if a.max_crops and per_split[sp] >= a.max_crops:
            continue
        img = None
        for k, b in enumerate(it["boxes"]):
            if b.get("ambiguous"):
                continue
            lab = b["brand"] or (f"other:{b['class']}" if src == "shelf" else "other:openfoodfacts")
            if not b["brand"] and src == "shelf" and per[(sp, lab)] >= cap[sp]:
                continue
            if img is None:
                img = cv2.imread(str(data / it["image"]))
                if img is None:
                    break
            x0, y0, x1, y1 = b["box"]
            w, h = x1 - x0, y1 - y0
            X0, Y0 = int(max(0, x0 - 0.1 * w)), int(max(0, y0 - 0.1 * h))
            X1, Y1 = int(min(img.shape[1], x1 + 0.1 * w)), int(min(img.shape[0], y1 + 0.1 * h))
            c = img[Y0:Y1, X0:X1]
            if not c.size or min(c.shape[:2]) < 16:
                continue
            s = min(1.0, a.max_side / max(c.shape[:2]))
            if s < 1:
                c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            rel = f"{sp}/{len(index):07d}.jpg"
            (out / sp).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / rel), c, [cv2.IMWRITE_JPEG_QUALITY, 95])
            index.append({"path": rel, "split": sp, "label": lab, "source": src, "image": it["image"],
                          "group": it.get("group"), "box": k})
            per[(sp, lab)] += 1
            per_split[sp] += 1
    (out / "index.json").write_text(json.dumps(index))
    stats = {s: {"crops": sum(1 for x in index if x["split"] == s),
                 "brand_crops": dict(Counter(x["label"] for x in index if x["split"] == s and not x["label"].startswith("other:"))),
                 "by_source": dict(Counter(x["source"] for x in index if x["split"] == s)),
                 "other_crops": sum(1 for x in index if x["split"] == s and x["label"].startswith("other:"))}
             for s in ("train", "val", "test")}
    (out / "crop_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def _load_index(crops_dir: Path):
    return json.loads((crops_dir / "index.json").read_text())


def kept_brands(index, min_train: int, min_val: int) -> list[str]:
    c = Counter((x["split"], x["label"]) for x in index)
    return [b for b in BRANDS if c[("train", b)] >= min_train and c[("val", b)] >= min_val]


# ------------------------------------------------------------------ augmentation
def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    if rng.random() < 0.8:  # crop jitter: the detector box is never exact
        s = rng.uniform(0.72, 1.0)
        cw, ch = max(16, int(w * rng.uniform(s, 1.0))), max(16, int(h * rng.uniform(s, 1.0)))
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        img = img[y0:y0 + ch, x0:x0 + cw]
    if rng.random() < 0.15:  # loose box: background around the pack
        p = int(0.15 * max(img.shape[:2]))
        img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        img = cv2.rotate(img, cv2.ROTATE_180)
    if rng.random() < 0.5:
        ang = rng.uniform(-12, 12)
        hh, ww = img.shape[:2]
        M = cv2.getRotationMatrix2D((ww / 2, hh / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (ww, hh), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.25:
        hh, ww = img.shape[:2]
        d = 0.08
        src = np.float32([[0, 0], [ww, 0], [ww, hh], [0, hh]])
        dst = src + np.float32([[rng.uniform(-d, d) * ww, rng.uniform(-d, d) * hh] for _ in range(4)])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (ww, hh), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.3:  # small / far-away packs
        f = rng.uniform(0.2, 0.6)
        small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.8:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4)) % 180
        hsv[..., 1] *= rng.uniform(0.75, 1.25)
        hsv[..., 2] *= rng.uniform(0.7, 1.3)
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.4:
        q = rng.randint(35, 90)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(img)


# ------------------------------------------------------------------ model
def _build_model(repo: str, revision: str, n_classes: int):
    import torch
    from torch import nn
    from transformers import AutoModel

    class BrandNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(repo, revision=revision)
            self.W = nn.Parameter(torch.randn(n_classes, 768) * 0.01)

        def embed(self, x):
            h = self.backbone(pixel_values=x).last_hidden_state
            v = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)  # == gallery.dino_pool
            return nn.functional.normalize(v.float(), dim=1)

        def forward(self, x, y=None, s: float = 30.0, m: float = 0.25):
            e = self.embed(x)
            logits = e @ nn.functional.normalize(self.W, dim=1).T
            if y is not None:  # CosFace: subtract the margin from the true class
                logits = logits - m * nn.functional.one_hot(y, logits.shape[1]).float()
            return s * logits, e

    return BrandNet()


def _embed(model_or_backbone, crops_list: list[np.ndarray], device: str, batch: int = 128, what: str = "") -> np.ndarray:
    import torch

    bb = getattr(model_or_backbone, "backbone", model_or_backbone)
    bb.eval()
    out = []
    prog = Progress(stage_name("embed") + (f":{what}" if what else ""), len(crops_list), "crops", every_s=30)
    with torch.inference_mode():
        for i in range(0, len(crops_list), batch):
            x = torch.from_numpy(dino_preprocess(crops_list[i:i + batch])).to(device)
            with torch.autocast("cuda", enabled=device.startswith("cuda")):
                h = bb(pixel_values=x).last_hidden_state
            out.append(dino_pool(h.float().cpu().numpy()))
            prog.update(len(crops_list[i:i + batch]))
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def _read(crops_dir: Path, recs) -> list[np.ndarray]:
    return [cv2.imread(str(crops_dir / r["path"])) for r in recs]


def _gallery_select(train, keep, refs_per_brand, max_negatives, rng):
    """Balanced gallery: per brand half shelf, half close-up (when both exist); negatives from both sources."""
    by = defaultdict(lambda: defaultdict(list))
    for r in train:
        lab = r["label"] if r["label"] in keep else NEG
        by[lab][r["source"]].append(r)
    sel = []
    for lab, srcs in by.items():
        cap = max_negatives if lab == NEG else refs_per_brand
        pools = [v[:] for v in srcs.values()]
        for p in pools:
            rng.shuffle(p)
        take, i = [], 0
        while len(take) < cap and any(pools):  # round-robin over sources
            p = pools[i % len(pools)]
            if p:
                take.append(p.pop())
            i += 1
        sel += [(r, lab) for r in take]
    return sel


def _threshold_search(top, s0, margin, truth, max_wrong: float, grid=None):
    truth = np.array(truth)
    pred = np.array(top)
    n_neg, n_brand = int((truth == NEG).sum()), int((truth != NEG).sum())
    best = None
    grid = grid or [(float(fs), tm) for fs in np.round(np.arange(0.30, 0.995, 0.01), 2)
                    for tm in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08)]
    for fs, tm in grid:
        acc = _accept(top, s0, margin, fs, tm)
        correct = int((acc & (pred == truth)).sum())
        neg_fa = int((acc & (truth == NEG)).sum())
        wrong = int((acc & (truth != NEG) & (pred != truth)).sum())
        if neg_fa <= max_wrong * max(1, n_neg) and wrong <= max_wrong * max(1, n_brand):
            if best is None or correct > best["correct"]:
                best = {"family_score": fs, "tie_margin": tm, "correct": correct, "neg_fa": neg_fa,
                        "wrong_brand": wrong, "n_neg": n_neg, "n_brand": n_brand}
    return best


def _predict(V, G, gl, labels, thr):
    top, s0, m, comp = _search(V, G, gl, labels)
    acc = _accept(top, s0, m, thr["family_score"], thr["tie_margin"])
    return [t if a else None for t, a in zip(top, acc)], (top, s0, m, comp)


def _metrics(pred, truth, sources, brands):
    """pred: brand or None per crop; truth: brand or NEG. Overall + per source + per brand."""
    def block(idx):
        t = [truth[i] for i in idx]
        p = [pred[i] for i in idx]
        nb = sum(1 for x in t if x != NEG)
        nn = sum(1 for x in t if x == NEG)
        return {"brand_crops": nb, "correct_brand": _cp(sum(1 for a, b in zip(p, t) if b != NEG and a == b), nb),
                "wrong_brand": _cp(sum(1 for a, b in zip(p, t) if b != NEG and a and a != b), nb),
                "other_products_called_a_brand": _cp(sum(1 for a, b in zip(p, t) if b == NEG and a), nn),
                "names_given": sum(1 for a in p if a),
                "name_precision": _cp(sum(1 for a, b in zip(p, t) if a and a == b), sum(1 for a in p if a))}
    allidx = list(range(len(truth)))
    out = {"all": block(allidx)}
    for s in sorted(set(sources)):
        out[s] = block([i for i in allidx if sources[i] == s])
    per = {}
    for b in brands:
        n = sum(1 for x in truth if x == b)
        tp = sum(1 for a, t in zip(pred, truth) if a == b and t == b)
        claimed = sum(1 for a in pred if a == b)
        per[b] = {"test_crops": n, "recall": _cp(tp, n), "precision": _cp(tp, claimed),
                  "by_source": {s: {"n": sum(1 for x, ss in zip(truth, sources) if x == b and ss == s),
                                    "tp": sum(1 for a, t, ss in zip(pred, truth, sources) if a == b and t == b and ss == s)}
                                for s in sorted(set(sources))}}
    out["per_brand"] = per
    out["confusion"] = dict(Counter(f"{t}->{p or 'unknown'}" for p, t in zip(pred, truth)
                                    if t != NEG and p != t).most_common(40))
    return out


# ------------------------------------------------------------------ train
def train(a) -> dict:
    import torch
    from torch.utils.data import DataLoader, IterableDataset

    crops_dir, out = Path(a.crops), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = _load_index(crops_dir)
    keep = kept_brands(index, a.min_train, a.min_val)
    classes = keep + ["other"]
    cid = {c: i for i, c in enumerate(classes)}
    train_recs = [r for r in index if r["split"] == "train"]
    pos = defaultdict(lambda: defaultdict(list))
    neg = defaultdict(list)
    for i, r in enumerate(train_recs):
        if r["label"] in cid:
            pos[r["label"]][r["source"]].append(i)
        else:
            neg[r["source"]].append(i)
    print(json.dumps({"classes": classes, "train_pos": {b: {s: len(v) for s, v in d.items()} for b, d in pos.items()},
                      "train_neg": {s: len(v) for s, v in neg.items()}}), flush=True)

    class Stream(IterableDataset):
        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            rng = random.Random(1000 + (info.id if info else 0) + int(time.time()))
            while True:
                if rng.random() < a.pos_frac:
                    b = rng.choice(keep)
                    src = rng.choice(list(pos[b].keys()))
                    i = rng.choice(pos[b][src])
                    y = cid[b]
                else:
                    src = rng.choice(list(neg.keys()))
                    i = rng.choice(neg[src])
                    y = cid["other"]
                img = cv2.imread(str(crops_dir / train_recs[i]["path"]))
                if img is None:
                    continue
                yield torch.from_numpy(dino_preprocess([augment(img, rng)])[0]), y

    man = json.loads(Path(a.manifest).read_text())["models"]["dinov2"]
    dev = a.device
    torch.manual_seed(0)
    model = _build_model(man["repo"], man["revision"], len(classes)).to(dev)
    bb_params = list(model.backbone.parameters())
    opt = torch.optim.AdamW([{"params": bb_params, "lr": a.lr_backbone}, {"params": [model.W], "lr": a.lr_head}],
                            weight_decay=0.05)
    warm = max(1, int(0.05 * a.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.startswith("cuda"))
    dl = DataLoader(Stream(), batch_size=a.batch, num_workers=a.workers, pin_memory=dev.startswith("cuda"))

    # fixed validation probe: gallery from train, queries from val (no augmentation)
    rng = random.Random(0)
    gal = _gallery_select(train_recs, set(keep), a.probe_refs, a.probe_negs, rng)
    val = [r for r in index if r["split"] == "val"]
    rng.shuffle(val)
    val = val[:a.probe_val]
    g_imgs = _read(crops_dir, [r for r, _ in gal])
    v_imgs = _read(crops_dir, val)
    g_lab = [lab for _, lab in gal]
    v_truth = [r["label"] if r["label"] in cid else NEG for r in val]
    labels = sorted(set(g_lab))
    gl = np.array([labels.index(x) for x in g_lab])

    def probe(tag):
        G = _embed(model, g_imgs, dev, what=f"probe-gallery-{tag}")
        V = _embed(model, v_imgs, dev, what=f"probe-val-{tag}")
        top, s0, m, _ = _search(V, G, gl, labels)
        best = _threshold_search(top, s0, m, v_truth, a.max_wrong) or {"correct": 0}
        model.train()
        return best

    curve = [{"step": 0, **probe("step0")}]
    print(json.dumps({"probe": curve[-1]}), flush=True)
    best_correct, best_step = curve[0]["correct"], 0
    torch.save({k: v.cpu() for k, v in model.backbone.state_dict().items()}, out / "best_backbone.pt")
    model.train()
    prog = Progress(stage_name("train"), a.steps, "steps", every_s=30)
    t0, it, loss_ema = time.time(), iter(dl), None
    for step in range(1, a.steps + 1):
        x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", enabled=dev.startswith("cuda")):
            logits, _ = model(x, y, s=a.scale, m=a.margin)
            loss = torch.nn.functional.cross_entropy(logits.float(), y, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
        prog.update(loss=round(loss_ema, 3))
        if step % a.eval_every == 0 or step == a.steps:
            r = probe(f"step{step}")
            curve.append({"step": step, "loss": round(loss_ema, 4), **r})
            print(json.dumps({"probe": curve[-1]}), flush=True)
            if r["correct"] > best_correct:
                best_correct, best_step = r["correct"], step
                torch.save({k: v.cpu() for k, v in model.backbone.state_dict().items()}, out / "best_backbone.pt")
            if a.stop_at and time.time() > a.stop_at:
                print("time budget reached: stopping training", flush=True)
                break
    rep = {"classes": classes, "steps_done": step, "best_step": best_step, "best_probe_correct": best_correct,
           "probe": {"gallery_refs": len(gal), "val_crops": len(val)}, "curve": curve,
           "minutes": round((time.time() - t0) / 60, 1),
           "hyper": {k: getattr(a, k) for k in ("steps", "batch", "lr_backbone", "lr_head", "scale", "margin",
                                                 "pos_frac", "min_train", "min_val")}}
    (out / "train_report.json").write_text(json.dumps(rep, indent=1))
    return rep


# ------------------------------------------------------------------ evaluate
def _load_backbone(man, device, state: Path | None):
    import torch
    from transformers import AutoModel

    m = AutoModel.from_pretrained(man["repo"], revision=man["revision"])
    if state is not None:
        m.load_state_dict(torch.load(state, map_location="cpu"))
    return m.to(device).eval()


def _export_onnx(backbone, path: Path, sample: list[np.ndarray], device: str) -> dict:
    import onnxruntime as ort
    import torch

    m = backbone.float().to("cpu")
    dummy = torch.from_numpy(dino_preprocess([np.zeros((60, 200, 3), np.uint8)] * 2))
    torch.onnx.export(m, (dummy,), str(path), input_names=["pixel_values"], output_names=["last_hidden_state"],
                      dynamic_axes={"pixel_values": {0: "batch"}, "last_hidden_state": {0: "batch"}},
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = dino_preprocess(sample[:16])
    with torch.inference_mode():
        ref = dino_pool(m(pixel_values=torch.from_numpy(x)).last_hidden_state.numpy())
    parity = float(np.abs(dino_pool(sess.run(None, {"pixel_values": x})[0]) - ref).max())
    backbone.to(device)
    return {"parity_max_abs": parity, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _score_model(name, backbone, device, crops_dir, index, keep, a, rng_seed=0):
    """Gallery from TRAIN, thresholds + brand retention on VAL, TEST metrics. Returns everything needed."""
    rng = random.Random(rng_seed)
    train_recs = [r for r in index if r["split"] == "train"]
    val = [r for r in index if r["split"] == "val"]
    test = [r for r in index if r["split"] == "test"]
    gal = _gallery_select(train_recs, set(keep), a.refs_per_brand, a.max_negatives, rng)
    g_imgs = _read(crops_dir, [r for r, _ in gal])
    views = g_imgs + [cv2.rotate(c, cv2.ROTATE_180) for c in g_imgs]
    view_lab = [lab for _, lab in gal] * 2
    G = _embed(backbone, views, device, what=f"{name}-gallery")
    Vv = _embed(backbone, _read(crops_dir, val), device, what=f"{name}-val")
    Vt = _embed(backbone, _read(crops_dir, test), device, what=f"{name}-test")

    kept = list(keep)
    history = []
    for _round in range(4):  # retire brands with low VALIDATION precision, then re-choose thresholds
        lab_of = {b: b for b in kept}
        g_lab = [lab_of.get(x, NEG if x == NEG else f"negative:retired:{x}") for x in view_lab]
        labels = sorted(set(g_lab))
        gl = np.array([labels.index(x) for x in g_lab])
        v_truth = [r["label"] if r["label"] in lab_of else NEG for r in val]
        top, s0, m, _ = _search(Vv, G, gl, labels)
        top = [t if not t.startswith("negative:") else NEG for t in top]
        thr = _threshold_search(top, s0, m, v_truth, a.max_wrong)
        if thr is None:
            thr = {"family_score": 0.99, "tie_margin": 0.05, "correct": 0, "neg_fa": 0, "wrong_brand": 0,
                   "note": "no feasible threshold"}
        acc = _accept(top, s0, m, thr["family_score"], thr["tie_margin"])
        vpred = [t if ok else None for t, ok in zip(top, acc)]
        drop = []
        for b in kept:
            n_val = sum(1 for t in v_truth if t == b)
            claimed = sum(1 for p in vpred if p == b)
            tp = sum(1 for p, t in zip(vpred, v_truth) if p == b and t == b)
            if n_val < a.min_val_keep or (claimed >= 3 and tp / claimed < a.min_val_precision):
                drop.append({"brand": b, "val_crops": n_val, "val_claimed": claimed, "val_tp": tp})
        history.append({"round": _round, "kept": list(kept), "thresholds": thr, "retire": drop})
        if not drop:
            break
        kept = [b for b in kept if b not in {d["brand"] for d in drop}]
    lab_of = {b: b for b in kept}
    g_lab = [lab_of.get(x, NEG if x == NEG else f"negative:retired:{x}") for x in view_lab]
    labels = sorted(set(g_lab))
    gl = np.array([labels.index(x) for x in g_lab])
    t_truth = [r["label"] if r["label"] in lab_of else NEG for r in test]
    top, s0, m, comp = _search(Vt, G, gl, labels)
    topn = [t if not t.startswith("negative:") else NEG for t in top]
    acc = _accept(topn, s0, m, thr["family_score"], thr["tie_margin"])
    tpred = [t if ok else None for t, ok in zip(topn, acc)]
    metrics = _metrics(tpred, t_truth, [r["source"] for r in test], kept)
    return {"name": name, "kept": kept, "thresholds": thr, "retention_history": history, "val_correct": thr["correct"],
            "test": metrics, "G": G, "g_lab": g_lab, "views_meta": [r for r, _ in gal]}


def _score_v2(v2dir: Path, frozen, device, crops_dir, index):
    """Live release v2 (frozen DINOv2 + its 5-brand gallery + its thresholds) on the same TEST crops."""
    z = np.load(v2dir / "models/trained/gallery_brands.npz", allow_pickle=False)
    refs = json.loads(str(z["refs"]))
    G = z["vectors"].astype(np.float32)
    thr = yaml.safe_load((v2dir / "configs/thresholds_brands.yaml").read_text())
    g_lab = [r["sku"] if not r["sku"].startswith("negative:") else NEG for r in refs]
    labels = sorted(set(g_lab))
    gl = np.array([labels.index(x) for x in g_lab])
    v2_brands = sorted({x for x in g_lab if x != NEG})
    test = [r for r in index if r["split"] == "test"]
    Vt = _embed(frozen, _read(crops_dir, test), device, what="v2-test")
    top, s0, m, _ = _search(Vt, G, gl, labels)
    acc = _accept(top, s0, m, thr["family_score"], thr["tie_margin"])
    pred = [t if ok else None for t, ok in zip(top, acc)]
    truth_v2 = [r["label"] if r["label"] in v2_brands else NEG for r in test]
    return {"brands": v2_brands, "thresholds": {"family_score": thr["family_score"], "tie_margin": thr["tie_margin"]},
            "test": _metrics(pred, truth_v2, [r["source"] for r in test], v2_brands), "pred": pred}


def evaluate(a) -> dict:
    crops_dir, out = Path(a.crops), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = _load_index(crops_dir)
    keep = kept_brands(index, a.min_train, a.min_val)
    man = json.loads(Path(a.manifest).read_text())["models"]["dinov2"]
    t0 = time.time()
    frozen = _load_backbone(man, a.device, None)
    results = {"frozen": _score_model("frozen", frozen, a.device, crops_dir, index, keep, a)}
    ft_state = Path(a.ft) / "best_backbone.pt" if a.ft else None
    tuned = None
    if ft_state and ft_state.exists():
        tuned = _load_backbone(man, a.device, ft_state)
        results["finetuned"] = _score_model("finetuned", tuned, a.device, crops_dir, index, keep, a)
    v2 = _score_v2(Path(a.v2), frozen, a.device, crops_dir, index) if a.v2 and Path(a.v2).exists() else None

    # choose on VALIDATION: most correct names under the same <= max_wrong limits
    choice = max(results.values(), key=lambda r: r["val_correct"])
    backbone = tuned if choice["name"] == "finetuned" else frozen
    test = [r for r in index if r["split"] == "test"]
    sample = _read(crops_dir, test[:16] or [r for r in index if r["split"] == "val"][:16])
    onnx = _export_onnx(backbone, out / "dinov2_small.onnx", sample, a.device)
    cat_rev = f"brands3-{onnx['sha256'][:8]}-{len(choice['g_lab'])}"
    fam = {lab: (lab if not lab.startswith("negative:") else None) for lab in set(choice["g_lab"])}
    n = len(choice["views_meta"])
    refs = [Ref(sku=lab, family=fam[lab], side="any", revision="brands3-1", view=v)
            for lab, v in zip(choice["g_lab"], ["canonical"] * n + ["rot180"] * n)]
    g = Gallery(choice["G"], refs, f"dinov2-onnx:{onnx['sha256'][:12]}", cat_rev)
    np.savez_compressed(out / "gallery_brands.npz", vectors=g.vectors.astype(np.float16),
                        refs=json.dumps([r.__dict__ for r in refs]),
                        meta=json.dumps({"backend": g.backend_id, "catalogue_revision": cat_rev}))
    base = yaml.safe_load(Path(a.base_thresholds).read_text())
    base.update(version="thr-brands-3", calibrated=False, sku_score=1.01,
                family_score=choice["thresholds"]["family_score"], tie_margin=choice["thresholds"]["tie_margin"])
    (out / "thresholds_brands.yaml").write_text(
        "# Chosen on the VALIDATION split (pvs.training.brands3); raw cosines, not probability-calibrated.\n"
        "# Family (brand) level only: sku_score > 1 disables SKU claims.\n" + yaml.safe_dump(base, sort_keys=False))
    catalogue = {"catalogue": "brands", "revision": cat_rev, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "products": [{"sku": b, "family": b, "display_name": BRANDS[b], "gtins": [], "quantity": None,
                               "variant": None, "artwork_revision": "public-data-3", "references": [],
                               "physical_ends": {}, "supported_faces": ["front", "back"],
                               "provenance": "embedding references from public Roboflow Universe datasets and Open "
                                             "Food Facts front-of-pack photos (see brands_report.json)",
                               "usage_rights": "derived embeddings only; source images not redistributed",
                               "approval_state": "approved"} for b in choice["kept"]],
                 "negatives": [], "history": [{"revision": cat_rev, "change": "built by pvs.training.brands3",
                                               "model": choice["name"]}]}
    (out / "catalogue_brands.json").write_text(json.dumps(catalogue, indent=1))

    def slim(r):
        return {k: v for k, v in r.items() if k not in ("G", "g_lab", "views_meta")}

    ct = choice["test"]
    rep = {"chosen_model": choice["name"], "chosen_on": "validation correct names at <= 1% wrong names and <= 1% "
                                                         "other products called a brand",
           "brands": choice["kept"], "display_names": {b: BRANDS[b] for b in choice["kept"]},
           "candidates": {k: slim(v) for k, v in results.items()},
           "v2_live_on_same_test": ({k: v for k, v in v2.items() if k != "pred"} if v2 else None),
           # run-2-compatible summary keys (read by pvs.training.hub)
           "test_crop_level": {"brand_crops": ct["all"]["brand_crops"], "correct_brand": ct["all"]["correct_brand"],
                               "wrong_brand": ct["all"]["wrong_brand"],
                               "other_products_called_a_brand": ct["all"]["other_products_called_a_brand"],
                               "per_brand": ct["per_brand"], "by_source": {k: v for k, v in ct.items()
                                                                          if k in ("shelf", "closeup")}},
           "threshold_choice_on_val": choice["thresholds"], "onnx": onnx,
           "dinov2": f"{man['repo']}@{man['revision'][:12]}" + (" + fine-tuned (brands3)" if choice["name"] == "finetuned" else ""),
           "crop_stats": json.loads((crops_dir / "crop_stats.json").read_text()),
           "seconds": round(time.time() - t0, 1),
           "caveats": ["Test crops come from the same public projects / the same Open Food Facts database as training "
                       "(split by group / barcode); frames of one product are correlated.",
                       "Thresholds are raw cosines chosen on validation, not calibrated probabilities.",
                       "Brand level only; pack size / flavour variants are not distinguished."]}
    (out / "brands_report.json").write_text(json.dumps(rep, indent=1))
    return rep


# ------------------------------------------------------------------ live release (the baseline to beat)
RELEASE_URL = "https://raw.githubusercontent.com/Rohitpro1/product-vision-station/model-assets/release.json"


def fetch_live(a) -> dict:
    """Download the release the hosted apps serve right now (release.json), verifying every sha256."""
    import io
    import tarfile

    import requests

    out = Path(a.out)
    rel = requests.get(a.release_url, timeout=60).json()
    out.mkdir(parents=True, exist_ok=True)
    for f in rel["files"]:
        r = requests.get(f["url"], timeout=600)
        r.raise_for_status()
        if hashlib.sha256(r.content).hexdigest() != f["sha256"]:
            raise SystemExit(f"sha256 mismatch for {f['url']}")
        if f.get("extract"):
            with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
                tf.extractall(out, filter="data")
        else:
            dest = (out / f["path"]).resolve()
            if not dest.is_relative_to(out.resolve()):
                raise SystemExit(f"path escapes the release folder: {f['path']}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
    (out / "release.json").write_text(json.dumps(rel, indent=1))
    return {"release": rel["release"], "folder": str(out)}


# ------------------------------------------------------------------ release bundle
def bundle(a) -> dict:
    """v3 bundle = the v2 bundle (same detector) with the new embedder, gallery, catalogue and thresholds."""
    import shutil

    base, brands_dir, out = Path(a.base), Path(a.brands), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out, ignore=shutil.ignore_patterns("var", "release.json", ".complete"))
    shutil.copy(brands_dir / "dinov2_small.onnx", out / "models/trained/dinov2_small.onnx")
    shutil.copy(brands_dir / "gallery_brands.npz", out / "models/trained/gallery_brands.npz")
    shutil.copy(brands_dir / "thresholds_brands.yaml", out / "configs/thresholds_brands.yaml")
    shutil.copy(brands_dir / "catalogue_brands.json", out / "catalogues/brands/catalogue.json")
    st = out / "configs/station.hosted.yaml"
    txt = st.read_text()
    old = yaml.safe_load(txt)["bundle_id"]
    txt = txt.replace(f"bundle_id: {old}", f"bundle_id: {a.bundle_id}")
    txt = txt.replace("# Hosted profile v2:", "# Hosted profile v3 (brands3: fine-tuned DINOv2 + Open Food Facts close-ups):", 1)
    st.write_text(txt)
    files = {str(f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest()
             for f in sorted(out.rglob("*")) if f.is_file()}
    (out.parent / f"{out.name}_sha256.json").write_text(json.dumps(files, indent=1))
    return {"bundle": str(out), "bundle_id": a.bundle_id, "files": files}


# ------------------------------------------------------------------ external photos (full pipeline)
def external(a) -> dict:
    import os
    import urllib.parse

    import requests

    from ..capture import frame_from_file
    from ..config import load_station
    from ..pipeline import Pipeline
    from ..schema import Profile

    spec = yaml.safe_load(Path(a.list).read_text())
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    bundle = Path(a.bundle).resolve()
    cwd = os.getcwd()
    os.chdir(bundle)
    try:
        p = Pipeline(load_station(bundle / "configs/station.hosted.yaml"), Profile.STATION, root=bundle)
    finally:
        os.chdir(cwd)
    rows = []
    for ph in spec["photos"]:
        f = cache / f"{ph['file']}.jpg"
        if not f.exists():
            u = "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(ph["commons"].replace(" ", "_")) + "?width=1280"
            try:
                r = requests.get(u, headers={"User-Agent": "pvs-external-check/1.0 (github.com/Rohitpro1)"}, timeout=60)
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                    f.write_bytes(r.content)
            except Exception as e:  # noqa: BLE001
                print(f"[external] {ph['commons']}: {type(e).__name__}", flush=True)
            time.sleep(2)
        if not f.exists():
            rows.append({**ph, "error": "download failed"})
            continue
        res = p.process(frame_from_file(str(f)))
        ids = [i.identity.family if i.identity.level.value == "family" else None for i in res.instances]
        rows.append({**ph, "packs": len(ids), "names": ids,
                     "correct": ph["brand"] in ids if ph.get("brand") else None,
                     "wrong": [x for x in ids if x and x != ph.get("brand")]})
    summ = {"photos": len(rows), "with_true_brand": sum(1 for r in rows if r.get("brand")),
            "named_correctly": sum(1 for r in rows if r.get("correct")),
            "photos_with_a_wrong_name": sum(1 for r in rows if r.get("wrong")), "rows": rows,
            "bundle": str(a.bundle)}
    Path(a.out).write_text(json.dumps(summ, indent=1))
    return summ


# ------------------------------------------------------------------ deploy decision
def decide(a) -> dict:
    """Recommend deploying v3 only if it beats the live v2 on the same test crops and on the external photos,
    without more wrong names. The final call stays with a person (this only writes decision.json)."""
    rep = json.loads(Path(a.report).read_text())
    v3 = rep["test_crop_level"]
    v2 = (rep.get("v2_live_on_same_test") or {}).get("test", {}).get("all")
    x2 = json.loads(Path(a.ext_v2).read_text()) if a.ext_v2 and Path(a.ext_v2).exists() else None
    x3 = json.loads(Path(a.ext_v3).read_text()) if a.ext_v3 and Path(a.ext_v3).exists() else None
    checks = []

    def rate(m):
        return m["rate"] or 0.0

    checks.append(("wrong brand on test <= 1%", rate(v3["wrong_brand"]) <= 0.01))
    checks.append(("other products called a brand on test <= 1%", rate(v3["other_products_called_a_brand"]) <= 0.01))
    if v2:
        checks.append(("more correct names than v2 on the same test crops",
                       v3["correct_brand"]["k"] > v2["correct_brand"]["k"]))
    if x2 and x3:
        checks.append(("external photos: at least as many correct as v2", x3["named_correctly"] >= x2["named_correctly"]))
        checks.append(("external photos: no wrong names", x3["photos_with_a_wrong_name"] == 0))
    ok = all(c for _, c in checks)
    out = {"verdict": "deploy v3" if ok else "keep v2", "checks": [{"check": n, "pass": c} for n, c in checks],
           "reason": "all checks pass" if ok else "failed: " + "; ".join(n for n, c in checks if not c),
           "v3_test": {k: v3[k] for k in ("correct_brand", "wrong_brand", "other_products_called_a_brand")},
           "v2_test": v2, "external": {"v2": x2 and {k: x2[k] for k in ("named_correctly", "photos_with_a_wrong_name")},
                                       "v3": x3 and {k: x3[k] for k in ("named_correctly", "photos_with_a_wrong_name")}}}
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pvs.training.brands3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crops")
    c.add_argument("--data", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--max-side", type=int, default=288)
    c.add_argument("--neg-per-class-train", type=int, default=40)
    c.add_argument("--neg-per-class-eval", type=int, default=25)
    c.add_argument("--max-crops", type=int, default=None, help="smoke: cap crops per split")
    for name in ("train", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--crops", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--device", default="cuda:0")
        s.add_argument("--manifest", default="models/manifest.json")
        s.add_argument("--min-train", type=int, default=15)
        s.add_argument("--min-val", type=int, default=1)
        s.add_argument("--max-wrong", type=float, default=0.01)
    t = sub.choices["train"]
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch", type=int, default=96)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--lr-backbone", type=float, default=2e-5)
    t.add_argument("--lr-head", type=float, default=2e-3)
    t.add_argument("--scale", type=float, default=30.0)
    t.add_argument("--margin", type=float, default=0.25)
    t.add_argument("--pos-frac", type=float, default=0.5)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--probe-refs", type=int, default=200)
    t.add_argument("--probe-negs", type=int, default=3000)
    t.add_argument("--probe-val", type=int, default=4000)
    t.add_argument("--stop-at", type=float, default=None)
    e = sub.choices["evaluate"]
    e.add_argument("--ft", default=None)
    e.add_argument("--v2", default=None)
    e.add_argument("--base-thresholds", default="configs/thresholds.yaml")
    e.add_argument("--refs-per-brand", type=int, default=600)
    e.add_argument("--max-negatives", type=int, default=8000)
    e.add_argument("--min-val-keep", type=int, default=5)
    e.add_argument("--min-val-precision", type=float, default=0.9)
    fl = sub.add_parser("fetch-live")
    fl.add_argument("--out", required=True)
    fl.add_argument("--release-url", default=RELEASE_URL)
    b = sub.add_parser("bundle")
    b.add_argument("--base", required=True, help="v2 release folder (configs/, models/, catalogues/)")
    b.add_argument("--brands", required=True, help="output of evaluate")
    b.add_argument("--out", required=True)
    b.add_argument("--bundle-id", default="hosted-retail-brands-v3")
    dc = sub.add_parser("decide")
    dc.add_argument("--report", required=True)
    dc.add_argument("--ext-v2", default=None)
    dc.add_argument("--ext-v3", default=None)
    dc.add_argument("--out", required=True)
    x = sub.add_parser("external")
    x.add_argument("--bundle", required=True)
    x.add_argument("--list", default="configs/brand_external_check.yaml")
    x.add_argument("--cache", default="/tmp/pvs-external")
    x.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = {"crops": crops, "train": train, "evaluate": evaluate, "bundle": bundle, "external": external, "fetch-live": fetch_live,
           "decide": decide}[a.cmd](a)
    slim = {k: v for k, v in rep.items() if k not in ("curve", "rows", "candidates", "crop_stats")} if isinstance(rep, dict) else rep
    print(json.dumps(slim, indent=1, default=str)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
