"""Train + evaluate on public Roboflow COCO exports containing Cadbury 5 Star packs.

Pipeline (architecture ADR-01 p.4-5): a *generic* ``package`` detector (RT-DETR-R18, COCO init) is
fine-tuned on every annotated product box; identity comes from a frozen DINOv2 reference gallery built
from TRAIN crops only (5 Star variants as positives, every other product class as explicit negatives).
Splits are group-aware: exact/near-duplicate images (dHash) that appear in more than one export are
forced into one split before training, and a leakage check must pass.

    python -m pvs.training.real5star prepare --exports data/rf/sct data/rf/scancart --out data/real5star
    python -m pvs.training.real5star train   --data data/real5star --out runs/real5star --epochs 30 --device cuda
    python -m pvs.training.real5star evaluate --data data/real5star --run runs/real5star --device cuda

Identity metrics count one decision per annotated test box (independent presentations are NOT known for
these datasets: frames of one pack in one session may be correlated - reported as a caveat).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

FIVE_STAR = re.compile(r"5\s*-?\s*star", re.I)


def sku_for(cls_name: str) -> str | None:
    """Map a dataset class name to a 5 Star SKU id, or None for other products."""
    n = cls_name.lower()
    if not FIVE_STAR.search(n):
        return None
    if " and " in n or "&" in n:  # mixed label ("5Star 3D and 5Star Bites"): ambiguous -> family only
        return "cadbury:5star-mixed"
    if "3d" in n:
        return "cadbury:5star-3d"
    if "bite" in n:
        return "cadbury:5star-bites"
    if "oreo" in n:
        return "cadbury:5star-oreo"
    return "cadbury:5star-classic"


def _dhash(path: Path) -> int:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (17, 16))
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


# --------------------------------------------------------------------------------------------- prepare
def prepare(exports: list[Path], out: Path, ham: int = 6) -> dict:
    """Merge exports into one manifest with leakage-safe train/val/test splits."""
    items = []
    for ex in exports:
        for split in ("train", "valid", "test"):
            ann = ex / split / "_annotations.coco.json"
            if not ann.exists():
                continue
            coco = json.loads(ann.read_text())
            cats = {c["id"]: c["name"] for c in coco["categories"]}
            by_img = defaultdict(list)
            for a in coco["annotations"]:
                x, y, w, h = a["bbox"]
                if w < 4 or h < 4:
                    continue
                name = cats[a["category_id"]]
                by_img[a["image_id"]].append({"box": [x, y, x + w, y + h], "class": name, "sku": sku_for(name)})
            for im in coco["images"]:
                p = ex / split / im["file_name"]
                if p.exists():
                    items.append({"src": str(p), "export": ex.name, "orig_split": split, "w": im["width"],
                                  "h": im["height"], "boxes": by_img.get(im["id"], [])})
    # near-duplicate grouping (union-find over dHash Hamming distance)
    hashes = [_dhash(Path(i["src"])) for i in items]
    parent = list(range(len(items)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    buckets = defaultdict(list)
    for i, h in enumerate(hashes):
        buckets[h >> 200].append(i)  # coarse prefix bucket keeps this O(n * bucket)
    order = sorted(range(len(items)), key=lambda i: hashes[i])
    for k in range(len(order) - 1):  # neighbours in sorted order + exact prefix buckets
        a, b = order[k], order[k + 1]
        if bin(hashes[a] ^ hashes[b]).count("1") <= ham:
            parent[find(a)] = find(b)
    for idx in buckets.values():
        for a in idx[1:]:
            if bin(hashes[a] ^ hashes[idx[0]]).count("1") <= ham:
                parent[find(a)] = find(idx[0])
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)
    # a group goes to test if any member was test; else valid if any valid; else train (keeps upstream test sets sealed)
    rank = {"test": 2, "valid": 1, "train": 0}
    for g in groups.values():
        s = max((items[i]["orig_split"] for i in g), key=rank.get)
        for i in g:
            items[i]["split"] = {"valid": "val"}.get(s, s)
            items[i]["group"] = f"g{find(i)}"
            items[i]["dhash"] = hashes[i]
    # leakage check
    split_groups = defaultdict(set)
    for it in items:
        split_groups[it["split"]].add(it["group"])
    leaks = [f"{a}/{b}" for a in split_groups for b in split_groups if a < b and split_groups[a] & split_groups[b]]
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    for it in items:
        h = hashlib.sha256(Path(it["src"]).read_bytes()).hexdigest()
        dst = img_dir / f"{h[:20]}{Path(it['src']).suffix}"
        if not dst.exists():
            shutil.copyfile(it["src"], dst)
        it["image"] = f"images/{dst.name}"
        it["sha256"] = h
        del it["src"]
    dupe_imgs = sum(len(g) for g in groups.values() if len(g) > 1)
    stats = {
        "exports": [e.name for e in exports], "images": len(items), "groups": len(groups),
        "images_in_duplicate_groups": dupe_imgs, "leakage": leaks,
        "split_images": dict(Counter(i["split"] for i in items)),
        "boxes_by_split_sku": {s: dict(Counter(b["sku"] or "other" for i in items if i["split"] == s for b in i["boxes"]))
                               for s in ("train", "val", "test")},
        "classes": dict(Counter(b["class"] for i in items for b in i["boxes"])),
        "licence": "CC BY 4.0 (Roboflow Universe exports; see THIRD_PARTY_NOTICES.md)",
    }
    (out / "manifest.json").write_text(json.dumps({"stats": stats, "items": items}))
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    if leaks:
        raise SystemExit(f"leakage between splits: {leaks}")
    return stats


# ----------------------------------------------------------------------------------------------- train
def _load(data: Path):
    m = json.loads((data / "manifest.json").read_text())
    return m["items"], m["stats"]


def train(a) -> dict:
    import torch
    from transformers import RTDetrForObjectDetection

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    data, out = Path(a.data), Path(a.run)
    out.mkdir(parents=True, exist_ok=True)
    items, stats = _load(data)
    tr = [i for i in items if i["split"] == "train" and i["boxes"]]
    va = [i for i in items if i["split"] == "val" and i["boxes"]]
    man = json.loads(Path(a.manifest).read_text())["models"]["rtdetr_r18"]
    model = RTDetrForObjectDetection.from_pretrained(man["repo"], revision=man["revision"], num_labels=1,
                                                     id2label={0: "package"}, label2id={"package": 0},
                                                     ignore_mismatched_sizes=True).to(a.device)
    opt = torch.optim.AdamW([{"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": a.lr * 0.1},
                             {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": a.lr}],
                            weight_decay=1e-4)
    steps_per_epoch = max(1, len(tr) // a.batch)
    if a.max_steps:
        steps_per_epoch = min(steps_per_epoch, a.max_steps)
    total = max(1, a.epochs * steps_per_epoch)
    warm = max(1, int(0.05 * total))

    def lr_factor(step: int) -> float:  # linear warm-up then cosine decay to 5%
        if step < warm:
            return (step + 1) / warm
        return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(1.0, (step - warm) / max(1, total - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler(enabled=a.device.startswith("cuda"))
    ck = out / "last.pt"
    start, log, best = 0, [], (-1.0, None)
    if ck.exists():
        st = torch.load(ck, map_location=a.device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, log, best = st["epoch"], st["log"], tuple(st["best"])
    t0 = time.time()
    for ep in range(start, a.epochs):
        model.train()
        random.shuffle(tr)
        losses = []
        for b in range(steps_per_epoch):
            batch = tr[b * a.batch:(b + 1) * a.batch]
            x, y = _batch(data, batch, a.size, a.device, augment=True)
            with torch.autocast(device_type="cuda" if a.device.startswith("cuda") else "cpu",
                                enabled=a.device.startswith("cuda")):
                loss = model(pixel_values=x, labels=y).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss))
        rec = detection_recall(model, data, va, a.size, a.device, limit=a.val_limit)
        log.append({"epoch": ep + 1, "loss": float(np.mean(losses)), "val_recall@0.5": rec["recall"],
                    "val_precision@0.5": rec["precision"], "elapsed_s": round(time.time() - t0, 1)})
        print(json.dumps(log[-1]), flush=True)
        if rec["recall"] + rec["precision"] > best[0]:
            best = (rec["recall"] + rec["precision"], ep + 1)
            model.save_pretrained(out / "best")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep + 1, "log": log, "best": list(best)}, ck)
    rep = {"epochs": a.epochs, "best_epoch": best[1], "log": log, "train_images": len(tr), "val_images": len(va),
           "init": f"{man['repo']}@{man['revision'][:12]}", "size": a.size, "batch": a.batch, "lr": a.lr,
           "device": a.device, "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
           "data_stats": stats}
    (out / "train_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def _batch(data: Path, items: list[dict], size: int, device: str, augment: bool):
    import torch

    xs, ys = [], []
    for it in items:
        img = cv2.imread(str(data / it["image"]))[:, :, ::-1]
        h, w = img.shape[:2]
        boxes = np.array([b["box"] for b in it["boxes"]], np.float32).reshape(-1, 4)
        if augment:
            if random.random() < 0.5:  # horizontal flip
                img = img[:, ::-1]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if random.random() < 0.5:  # photometric
                img = np.clip(img.astype(np.float32) * random.uniform(0.7, 1.3) + random.uniform(-25, 25), 0, 255)
        r = cv2.resize(np.ascontiguousarray(img), (size, size)).astype(np.float32) / 255.0
        xs.append(torch.tensor(r).permute(2, 0, 1))
        cx = (boxes[:, 0] + boxes[:, 2]) / 2 / w
        cy = (boxes[:, 1] + boxes[:, 3]) / 2 / h
        bw = (boxes[:, 2] - boxes[:, 0]) / w
        bh = (boxes[:, 3] - boxes[:, 1]) / h
        ys.append({"class_labels": torch.zeros(len(boxes), dtype=torch.long, device=device),
                   "boxes": torch.tensor(np.stack([cx, cy, bw, bh], 1), dtype=torch.float32, device=device)})
    return torch.stack(xs).to(device), ys


def predict(model, data: Path, it: dict, size: int, device: str, thr: float = 0.5) -> list[tuple[list[float], float]]:
    import torch

    img = cv2.imread(str(data / it["image"]))[:, :, ::-1]
    h, w = img.shape[:2]
    x = torch.tensor(cv2.resize(np.ascontiguousarray(img), (size, size)).astype(np.float32) / 255).permute(2, 0, 1)[None]
    model.eval()
    with torch.no_grad():
        o = model(pixel_values=x.to(device))
    s = o.logits[0, :, 0].sigmoid().cpu().numpy()
    bx = o.pred_boxes[0].cpu().numpy()
    out = []
    for sc, (cx, cy, bw, bh) in zip(s, bx):
        if sc >= thr:
            out.append(([float((cx - bw / 2) * w), float((cy - bh / 2) * h), float((cx + bw / 2) * w),
                         float((cy + bh / 2) * h)], float(sc)))
    from ..detectors.hf_models import Detection, _nms

    keep = _nms([Detection(b, sc, "package", "rtdetr") for b, sc in out], 0.6)
    return [(d.box_xyxy, d.score) for d in keep]


def detection_recall(model, data: Path, items: list[dict], size: int, device: str, limit: int | None = None) -> dict:
    from ..evaluation import match

    tp = gt = pred = 0
    for it in items[:limit]:
        p = predict(model, data, it, size, device)
        g = [b["box"] for b in it["boxes"]]
        m = match(g, [b for b, _ in p], 0.5)
        tp += len(m)
        gt += len(g)
        pred += len(p)
    return {"recall": tp / gt if gt else 0.0, "precision": tp / pred if pred else 0.0, "tp": tp, "gt": gt, "pred": pred}


# -------------------------------------------------------------------------------------------- evaluate
def evaluate(a) -> dict:
    from transformers import RTDetrForObjectDetection

    from ..evaluation import clopper_pearson, env_info, match, one_sided_upper
    from ..gallery import DinoV2Embedder

    data, run = Path(a.data), Path(a.run)
    items, stats = _load(data)
    man = json.loads(Path(a.manifest).read_text())["models"]
    emb = DinoV2Embedder(man["dinov2"]["repo"], man["dinov2"]["revision"], device=a.device)

    def crops(split, max_per_class=None):
        cs, labels, cnt = [], [], Counter()
        for it in items:
            if it["split"] != split:
                continue
            img = cv2.imread(str(data / it["image"]))
            for b in it["boxes"]:
                key = b["sku"] or f"other:{b['class']}"
                if max_per_class and cnt[key] >= max_per_class:
                    continue
                x0, y0, x1, y1 = map(int, b["box"])
                c = img[max(0, y0):y1, max(0, x0):x1]
                if c.size and min(c.shape[:2]) >= 12:
                    cs.append(c)
                    labels.append(key)
                    cnt[key] += 1
        return cs, labels

    def embed(cs):
        out = []
        for i in range(0, len(cs), 64):
            out.append(emb.embed(cs[i:i + 64]))
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    # gallery from TRAIN crops only
    g_c, g_l = crops("train", max_per_class=a.refs_per_class)
    G = embed(g_c)

    def classify(V):
        sims = V @ G.T
        res = []
        for s in sims:
            best = {}
            for v, lab in zip(s.tolist(), g_l):
                if lab not in best or v > best[lab]:
                    best[lab] = v
            r = sorted(best.items(), key=lambda t: -t[1])
            top, s0 = r[0]
            s1 = r[1][1] if len(r) > 1 else -1
            comp = r[1][0] if len(r) > 1 else None
            res.append((top, s0, s0 - s1, comp))
        return res

    def decide(top, s0, margin, comp, thr_score, thr_margin):
        if top.startswith("cadbury:5star") and s0 >= thr_score and margin >= thr_margin:
            if top == "cadbury:5star-mixed":
                return "family", "cadbury:5star"
            return "sku", top
        if top.startswith("cadbury:5star") and s0 >= thr_score and comp and comp.startswith("cadbury:5star"):
            return "family", "cadbury:5star"
        return "unknown", None

    # threshold selection on VAL only (maximise recall subject to zero false accepts on val negatives)
    v_c, v_l = crops("val")
    vres = classify(embed(v_c))
    best = None
    for ts in np.arange(0.40, 0.96, 0.02):
        for tm in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08):
            fa = sum(1 for (r, lab) in zip(vres, v_l) if not lab.startswith("cadbury") and decide(*r, ts, tm)[0] == "sku")
            ok = sum(1 for (r, lab) in zip(vres, v_l) if lab.startswith("cadbury:5star") and lab != "cadbury:5star-mixed"
                     and decide(*r, ts, tm) == ("sku", lab))
            key = (fa == 0, ok, -ts)
            if best is None or key > best[0]:
                best = (key, float(ts), float(tm), fa, ok)
    _, thr_s, thr_m, val_fa, val_ok = best
    # sealed TEST: ground-truth crops
    t_c, t_l = crops("test")
    tres = classify(embed(t_c))
    pos = [(r, lab) for r, lab in zip(tres, t_l) if lab.startswith("cadbury:5star") and lab != "cadbury:5star-mixed"]
    neg = [(r, lab) for r, lab in zip(tres, t_l) if not lab.startswith("cadbury")]
    acc_sku = [(decide(*r, thr_s, thr_m), lab) for r, lab in zip(tres, t_l)]
    accepted = [(d, lab) for d, lab in acc_sku if d[0] == "sku"]
    correct = sum(1 for d, lab in accepted if d[1] == lab)
    rec_ok = sum(1 for r, lab in pos if decide(*r, thr_s, thr_m) == ("sku", lab))
    fam_ok = sum(1 for r, lab in pos if decide(*r, thr_s, thr_m)[0] in ("sku", "family"))
    fa = sum(1 for r, lab in neg if decide(*r, thr_s, thr_m)[0] in ("sku", "family"))
    conf = Counter((lab, decide(*r, thr_s, thr_m)[1] or "unknown") for r, lab in zip(tres, t_l) if lab.startswith("cadbury"))
    # detector on TEST images
    det = None
    if (run / "best").exists():
        model = RTDetrForObjectDetection.from_pretrained(run / "best").to(a.device)
        test_items = [i for i in items if i["split"] == "test"]
        t0 = time.time()
        det = detection_recall(model, data, test_items, a.size, a.device)
        det["ms_per_image"] = round((time.time() - t0) * 1000 / max(1, len(test_items)), 1)
        det["recall_ci95"] = clopper_pearson(det["tp"], det["gt"])
        # 5 Star boxes specifically
        tp5 = gt5 = 0
        for it in test_items:
            g5 = [b["box"] for b in it["boxes"] if b["sku"]]
            if not g5:
                continue
            p = predict(model, data, it, a.size, a.device)
            tp5 += len(match(g5, [b for b, _ in p], 0.5))
            gt5 += len(g5)
        det["five_star_recall"] = {"tp": tp5, "gt": gt5, "value": tp5 / gt5 if gt5 else None,
                                   "ci95": clopper_pearson(tp5, gt5)}
        export_onnx(model, run / "rtdetr_r18_package.onnx", a.size)
    np.savez(run / "gallery_5star.npz", vectors=G, labels=np.array(g_l))
    rep = {
        "implementation": env_info(), "data_stats": stats,
        "thresholds_selected_on_val": {"score": thr_s, "margin": thr_m, "val_false_accepts": val_fa,
                                       "val_correct_sku": val_ok},
        "gallery": {"refs": len(g_l), "per_class_cap": a.refs_per_class, "embedder": emb.backend_id,
                    "classes": dict(Counter(g_l))},
        "test_identity_on_gt_crops": {
            "exact_sku_precision": {"k": correct, "n": len(accepted), "value": correct / len(accepted) if accepted else None,
                                    "ci95": clopper_pearson(correct, len(accepted))},
            "exact_sku_recall": {"k": rec_ok, "n": len(pos), "value": rec_ok / len(pos) if pos else None,
                                 "ci95": clopper_pearson(rec_ok, len(pos))},
            "family_or_better_recall": {"k": fam_ok, "n": len(pos), "value": fam_ok / len(pos) if pos else None},
            "false_accept_non_5star_products": {"k": fa, "n": len(neg), "value": fa / len(neg) if neg else None,
                                                "one_sided_upper95": one_sided_upper(fa, len(neg))},
            "confusion_5star": {f"{k[0]} -> {k[1]}": v for k, v in conf.items()},
        },
        "test_detection": det,
        "caveats": [
            "Boxes are Roboflow community annotations; images are shelf/hand photos from two exports by one author.",
            "One decision per annotated box; boxes of the same pack across frames may be correlated (not independent presentations).",
            "No A/B endpoint labels exist in these datasets: orientation (heading) is NOT trained or evaluated here.",
            "Thresholds chosen on the validation split only; test split was not used for any choice.",
        ],
    }
    (run / "eval_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("thresholds_selected_on_val", "test_identity_on_gt_crops", "test_detection")},
                     indent=1, default=float))
    return rep


def export_onnx(model, path: Path, size: int) -> dict:
    import onnxruntime as ort
    import torch

    model = model.float().cpu().eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            o = self.m(pixel_values=pixel_values)
            return o.logits, o.pred_boxes

    x = torch.rand(1, 3, size, size)
    with torch.no_grad():
        ref = Wrap(model)(x)
        torch.onnx.export(Wrap(model), (x,), str(path), input_names=["pixel_values"],
                          output_names=["logits", "pred_boxes"], opset_version=17, dynamo=False)
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"pixel_values": x.numpy()})
    d = max(float(np.abs(r.numpy() - g).max()) for r, g in zip(ref, got))
    info = {"onnx": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "max_abs_diff": d}
    (path.with_suffix(".json")).write_text(json.dumps(info, indent=1))
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pvs.training.real5star")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--exports", nargs="+", required=True)
    p.add_argument("--out", required=True)
    for name in ("train", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--data", required=True)
        s.add_argument("--run", required=True)
        s.add_argument("--device", default="cuda")
        s.add_argument("--size", type=int, default=640)
        s.add_argument("--manifest", default="models/manifest.json")
        if name == "train":
            s.add_argument("--epochs", type=int, default=30)
            s.add_argument("--batch", type=int, default=16)
            s.add_argument("--lr", type=float, default=2e-4)
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--max-steps", type=int, default=0, help="smoke: cap steps per epoch")
            s.add_argument("--val-limit", type=int, default=None)
        else:
            s.add_argument("--refs-per-class", type=int, default=40)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        print(json.dumps(prepare([Path(e) for e in a.exports], Path(a.out)), indent=1))
    elif a.cmd == "train":
        train(a)
    else:
        evaluate(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
