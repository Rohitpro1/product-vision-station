"""YOLO26-seg for the laptop station: dataset with masks, training, NMS-free export, honest evaluation.

    python -m pvs.training.yolo dataset  --data /data/retail --out /run/yolo-data [--shard 0 --shards 2]
    python -m pvs.training.yolo finalize --out /run/yolo-data
    python -m pvs.training.yolo train    --yaml /run/yolo-data/data.yaml --out /run/yolo --device 0,1 --time-h 4
    python -m pvs.training.yolo export   --weights /run/yolo/train/weights/best.pt --out /run/edge --ocr-dir ...
    python -m pvs.training.yolo evaluate --data /run/yolo-data --bundle /run/edge --out /run/yolo_report.json

`dataset` turns the prepared manifest (pvs.training.retail prepare: Roboflow + Open Food Facts, deduplicated,
leakage-safe splits) into YOLO segmentation labels. Our sources only have boxes, so each box gets a mask from
SAM2 (box prompt, hiera-tiny); a mask that does not fit its box (too small, or spilling out) falls back to the box
rectangle, and the report counts both. Classes: "package" (any pack) + every brand with enough training boxes;
rarer or ambiguous brands stay "package" (still detected and segmented; OCR can name them).

`evaluate` runs the exported ONNX through the same code the laptop uses (pvs.edge), chooses the brand
confidence on VALIDATION (<= 1 % wrong brand names, <= 1 % other packs called a brand) and reports TEST.

Ultralytics (training + the YOLO26 weights) is AGPL-3.0: see docs/EDGE.md before shipping in a closed product.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .brand_map import BRANDS
from .progress import Progress, stage_name

GENERIC = "package"


# ------------------------------------------------------------------ dataset
def _classes(items, min_boxes: int) -> list[str]:
    c = Counter(b["brand"] for it in items if it["split"] == "train" for b in it["boxes"]
                if b["brand"] and not b.get("ambiguous"))
    return [GENERIC] + [b for b, n in c.most_common() if n >= min_boxes]


def _select(items, max_plain: int, rng: random.Random):
    """All photos with a brand box or from Open Food Facts; a sample of the plain retail photos."""
    keep, plain = [], defaultdict(list)
    for it in items:
        if it["export"].startswith("openfoodfacts__") or any(b["brand"] for b in it["boxes"]):
            keep.append(it)
        else:
            plain[it["split"]].append(it)
    for sp, lst in plain.items():
        rng.shuffle(lst)
        keep += lst[:int(max_plain * (1.0 if sp == "train" else 0.15))]
    return sorted(keep, key=lambda it: it["image"])


def _polygon(mask: np.ndarray | None, box, W, H):
    """Mask -> polygon in pixels; falls back to the box rectangle when the mask does not fit the box."""
    x1, y1, x2, y2 = box
    rect = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    if mask is None:
        return rect, "rect"
    bw, bh = x2 - x1, y2 - y1
    mx0, my0 = int(max(0, x1 - 0.03 * bw)), int(max(0, y1 - 0.03 * bh))
    mx1, my1 = int(min(W, x2 + 0.03 * bw)), int(min(H, y2 + 0.03 * bh))
    m = np.zeros_like(mask)
    m[my0:my1, mx0:mx1] = mask[my0:my1, mx0:mx1]
    outside = mask.sum() - m.sum()
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return rect, "rect"
    c = max(cs, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.35 * bw * bh or outside > 0.5 * max(1, m.sum()):
        return rect, "rect"
    xs, ys = c[:, 0, 0], c[:, 0, 1]
    if (xs.max() - xs.min()) < 0.6 * bw or (ys.max() - ys.min()) < 0.6 * bh:
        return rect, "rect"
    c = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
    return (c.tolist() if len(c) >= 3 else rect), ("sam2" if len(c) >= 3 else "rect")


def dataset(a) -> dict:
    data, out = Path(a.data), Path(a.out)
    items = json.loads((data / "manifest.json").read_text())["items"]
    classes = _classes(items, a.min_boxes)
    cid = {c: i for i, c in enumerate(classes)}
    sel = _select(items, a.max_plain, random.Random(0))
    if a.max_images:
        sel = sel[:a.max_images]
    mine = [it for k, it in enumerate(sel) if k % a.shards == a.shard]
    sam = None
    if not a.no_sam:
        from ..detectors.hf_models import Sam2MaskProvider

        man = json.loads(Path(a.manifest).read_text())["models"]["sam2"]
        sam = Sam2MaskProvider(man["repo"], man["revision"], device=a.device)
    stats = {"classes": classes, "images": Counter(), "boxes": defaultdict(Counter), "mask_source": Counter()}
    prog = Progress(stage_name("masks") + (f":{a.shard}" if a.shards > 1 else ""), len(mine), "images", every_s=30)
    for it in mine:
        prog.update(**dict(stats["mask_source"]))
        img = cv2.imread(str(data / it["image"]))
        if img is None:
            continue
        H, W = img.shape[:2]
        sp = it["split"]
        boxes = [b for b in it["boxes"] if b["box"][2] - b["box"][0] >= 4 and b["box"][3] - b["box"][1] >= 4]
        if not boxes:
            continue
        masks = [None] * len(boxes)
        if sam is not None:
            for k in range(0, len(boxes), 48):
                chunk = [b["box"] for b in boxes[k:k + 48]]
                try:
                    for j, (m, _src) in enumerate(sam.masks(img, chunk)):
                        masks[k + j] = m
                except Exception as e:  # noqa: BLE001 - a failed SAM2 call falls back to rectangles
                    print(f"[masks] {it['image']}: {type(e).__name__}: {e}"[:200], flush=True)
        lines = []
        for b, m in zip(boxes, masks):
            poly, src = _polygon(m, b["box"], W, H)
            c = cid.get(b["brand"], 0) if b["brand"] and not b.get("ambiguous") else 0
            stats["boxes"][sp][classes[c]] += 1
            stats["mask_source"][src] += 1
            pts = " ".join(f"{min(max(x / W, 0), 1):.5f} {min(max(y / H, 0), 1):.5f}" for x, y in poly)
            lines.append(f"{c} {pts}")
        stem = Path(it["image"]).stem
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        dst = out / "images" / sp / f"{stem}{Path(it['image']).suffix}"
        if not dst.exists():
            dst.symlink_to((data / it["image"]).resolve())
        stats["images"][sp] += 1
    s = {"classes": classes, "images": dict(stats["images"]),
         "boxes": {k: dict(v) for k, v in stats["boxes"].items()}, "mask_source": dict(stats["mask_source"])}
    (out / f"stats_shard{a.shard}.json").write_text(json.dumps(s, indent=1))
    (out / "classes.json").write_text(json.dumps(classes))
    return s


def finalize(a) -> dict:
    out = Path(a.out)
    classes = json.loads((out / "classes.json").read_text())
    tot = {"classes": classes, "images": Counter(), "boxes": defaultdict(Counter), "mask_source": Counter()}
    for f in sorted(out.glob("stats_shard*.json")):
        s = json.loads(f.read_text())
        tot["images"].update(s["images"])
        tot["mask_source"].update(s["mask_source"])
        for sp, c in s["boxes"].items():
            tot["boxes"][sp].update(c)
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
                                   f"test: images/test\nnames:\n{names}\n")
    rep = {"classes": classes, "display_names": {c: BRANDS.get(c, "Package (brand not named)") for c in classes},
           "images": dict(tot["images"]), "boxes": {k: dict(v) for k, v in tot["boxes"].items()},
           "mask_source": dict(tot["mask_source"])}
    (out / "dataset_report.json").write_text(json.dumps(rep, indent=1))
    return rep


# ------------------------------------------------------------------ train / export
def train(a) -> dict:
    from ultralytics import YOLO

    m = YOLO(a.model)
    kw = dict(data=a.yaml, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
              project=a.out, name="train", exist_ok=True, seed=0, cos_lr=True, patience=a.patience, plots=True,
              fliplr=0.0,  # mirrored logos and text do not exist on real packs
              degrees=10.0, scale=0.5, translate=0.1, hsv_h=0.01, close_mosaic=3, amp=True, cache=False)
    if a.time_h:
        kw["time"] = a.time_h
    if a.fraction:
        kw["fraction"] = a.fraction
    t0 = time.time()
    stop = _watch_epochs(Path(a.out) / "train" / "results.csv", a.epochs, t0)
    try:
        m.train(**kw)
    finally:
        stop.set()
    best = Path(a.out) / "train" / "weights" / "best.pt"
    rep = {"weights": str(best), "minutes": round((time.time() - t0) / 60, 1),
           "args": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in kw.items()}}
    res = Path(a.out) / "train" / "results.csv"
    if res.exists():
        rows = res.read_text().strip().splitlines()
        rep["epochs_done"] = len(rows) - 1
        rep["last_row"] = dict(zip([h.strip() for h in rows[0].split(",")], [x.strip() for x in rows[-1].split(",")]))
    (Path(a.out) / "train_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def _watch_epochs(csv: Path, epochs: int, t0: float):
    """Report each finished epoch to the run status (works with multi-GPU training, where Ultralytics runs in a
    child process): a background thread reads results.csv every 30 s."""
    import threading

    from .progress import read_status, write_status

    stop = threading.Event()

    def loop():
        seen = 0
        while not stop.wait(30):
            if not csv.exists():
                continue
            rows = [r for r in csv.read_text().strip().splitlines() if r.strip()]
            if len(rows) - 1 <= seen:
                continue
            head = [h.strip() for h in rows[0].split(",")]
            st = read_status()
            for r in rows[seen + 1:]:
                v = dict(zip(head, [x.strip() for x in r.split(",")]))

                def f(k, v=v):
                    try:
                        return round(float(v[k]), 4)
                    except (KeyError, ValueError):
                        return None
                st.setdefault("epochs", []).append(
                    {"epoch": int(float(v.get("epoch", 0) or 0)), "box_map50_95": f("metrics/mAP50-95(B)"),
                     "mask_map50_95": f("metrics/mAP50-95(M)"), "box_recall": f("metrics/recall(B)"),
                     "elapsed_s": round(time.time() - t0, 1)})
            seen = len(rows) - 1
            st["epochs"] = st["epochs"][-40:]
            st["line"] = f"[{st.get('stage', 'train')}] epoch {seen}/{epochs} done | box mAP50-95 " \
                         f"{st['epochs'][-1]['box_map50_95']} | mask {st['epochs'][-1]['mask_map50_95']}"
            st["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
            write_status(st)

    threading.Thread(target=loop, daemon=True).start()
    return stop


OCR_URLS = {
    "ocr/PP-OCRv5_det_mobile.onnx":
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
    "ocr/en_PP-OCRv5_rec_mobile.onnx":
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx",
}


def _ocr_files(dest: Path) -> dict:
    """PP-OCRv5 mobile detector + English recogniser (Apache-2.0) and the 0/180 text-direction model that ships
    inside the rapidocr wheel."""
    import rapidocr
    import requests

    out = {}
    for rel, url in OCR_URLS.items():
        f = dest / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        if not f.exists():
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            f.write_bytes(r.content)
        out[rel] = {"url": url, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    cls_src = Path(rapidocr.__file__).parent / "models" / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
    shutil.copy(cls_src, dest / "ocr/ppocr_mobile_v2.0_cls.onnx")
    out["ocr/ppocr_mobile_v2.0_cls.onnx"] = {"url": "rapidocr wheel",
                                            "sha256": hashlib.sha256(cls_src.read_bytes()).hexdigest()}
    return out


def export(a) -> dict:
    from ultralytics import YOLO

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for sz in a.imgsz:
        for half in ([False, True] if a.fp16 else [False]):
            m = YOLO(a.weights)
            p = Path(m.export(format="onnx", imgsz=sz, end2end=True, simplify=True, half=half,
                              device=(a.device if half else "cpu")))
            name = f"model_{sz}{'_fp16' if half else ''}.onnx"
            shutil.move(str(p), out / name)
            files[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    ocr = _ocr_files(out)
    names = YOLO(a.weights).names
    cfg = {"model": f"model_{a.default_size}.onnx",
           "model_gpu": f"model_{a.default_size}_fp16.onnx" if a.fp16 else None,
           "conf": 0.35, "brand_conf": 0.5, "classes": [names[i] for i in sorted(names)],
           "display_names": {c: BRANDS.get(c, "Package") for c in names.values()},
           "licences": {"model": "AGPL-3.0 (Ultralytics YOLO26; Enterprise licence needed for closed products)",
                        "ocr": "Apache-2.0 (PaddleOCR PP-OCRv5 via RapidOCR)"}}
    (out / "edge.json").write_text(json.dumps(cfg, indent=1))
    rep = {"onnx": files, "ocr": ocr, "edge.json": cfg}
    (out / "export_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def val(a) -> dict:
    """Ultralytics mAP on the TEST split (PyTorch weights, square letterbox like the ONNX)."""
    from ultralytics import YOLO

    m = YOLO(a.weights)
    r = m.val(data=a.yaml, split="test", imgsz=a.imgsz, batch=16, device=a.device, plots=False, rect=False)
    names = m.names
    rep = {"split": "test", "imgsz": a.imgsz,
           "box": {"map50_95": float(r.box.map), "map50": float(r.box.map50), "precision": float(r.box.mp),
                   "recall": float(r.box.mr)},
           "mask": {"map50_95": float(r.seg.map), "map50": float(r.seg.map50)},
           "per_class": {names[int(c)]: {"box_map50_95": float(r.box.maps[int(c)]),
                                         "mask_map50_95": float(r.seg.maps[int(c)])} for c in r.box.ap_class_index},
           "speed_ms_torch": {k: round(float(v), 2) for k, v in r.speed.items()}}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return rep


# ------------------------------------------------------------------ evaluate
def _gt(lbl: Path, W: int, H: int, classes):
    out = []
    for line in lbl.read_text().split("\n"):
        v = line.split()
        if len(v) < 7:
            continue
        pts = np.array(v[1:], float).reshape(-1, 2) * [W, H]
        out.append({"cls": classes[int(v[0])], "box": [*pts.min(0), *pts.max(0)], "poly": pts.astype(np.int32)})
    return out


def _match(gts, dets, thr=0.5):
    from ..edge.station import _iou

    pairs, used = [], set()
    cand = sorted(((_iou(g["box"], d.box), gi, di) for gi, g in enumerate(gts) for di, d in enumerate(dets)),
                  reverse=True)
    gused = set()
    for iou, gi, di in cand:
        if iou < thr:
            break
        if gi in gused or di in used:
            continue
        gused.add(gi)
        used.add(di)
        pairs.append((gi, di))
    return pairs


def _cp(k, n):
    from ..evaluation import clopper_pearson

    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": clopper_pearson(k, n) if n else None}


def _collect(seg, root: Path, split: str, classes, limit: int, conf: float):
    imgs = sorted((root / "images" / split).glob("*"))[:limit]
    rows, mask_ious, dets_n, gts_n, matched, lat = [], [], 0, 0, 0, []
    prog = Progress(stage_name("evaluate") + f":{split}", len(imgs), "photos", every_s=30)
    for p in imgs:
        prog.update()
        img = cv2.imread(str(p))
        if img is None:
            continue
        H, W = img.shape[:2]
        gts = _gt(root / "labels" / split / f"{p.stem}.txt", W, H, classes)
        t = time.perf_counter()
        dets = seg(img, conf=conf)
        lat.append((time.perf_counter() - t) * 1000)
        pairs = _match(gts, dets)
        dets_n += len(dets)
        gts_n += len(gts)
        matched += len(pairs)
        for gi, di in pairs:
            rows.append((gts[gi]["cls"], dets[di].name, dets[di].score))
            g = np.zeros((H, W), np.uint8)
            cv2.fillPoly(g, [gts[gi]["poly"]], 1)
            pm = dets[di].full_mask(img.shape)
            u = (g.astype(bool) | pm).sum()
            if u:
                mask_ious.append(float((g.astype(bool) & pm).sum() / u))
        for gi in set(range(len(gts))) - {g for g, _ in pairs}:
            rows.append((gts[gi]["cls"], None, 0.0))
    return {"rows": rows, "mask_iou": mask_ious, "dets": dets_n, "gts": gts_n, "matched": matched, "lat": lat}


def _brand_metrics(rows, t):
    """rows: (true class, predicted class or None if missed, score). A brand name counts when score >= t."""
    def name(p, s):
        return p if (p and p != GENERIC and s >= t) else None
    nb = sum(1 for g, _, _ in rows if g != GENERIC)
    nn = sum(1 for g, _, _ in rows if g == GENERIC)
    correct = sum(1 for g, p, s in rows if g != GENERIC and name(p, s) == g)
    wrong = sum(1 for g, p, s in rows if g != GENERIC and name(p, s) and name(p, s) != g)
    fa = sum(1 for g, p, s in rows if g == GENERIC and name(p, s))
    return {"brand_boxes": nb, "correct_brand": _cp(correct, nb), "wrong_brand": _cp(wrong, nb),
            "other_packs_called_a_brand": _cp(fa, nn)}


def evaluate(a) -> dict:
    from ..edge.seg import YoloSeg

    root, bundle = Path(a.data), Path(a.bundle)
    classes = json.loads((root / "classes.json").read_text())
    cfg = json.loads((bundle / "edge.json").read_text())
    rep = {"models": {}}
    for mf in sorted(bundle.glob("model_*.onnx")):
        if "fp16" in mf.name and not a.device.startswith("cuda"):
            continue
        seg = YoloSeg(mf, device=a.device)
        val = _collect(seg, root, "val", classes, a.limit, cfg["conf"])
        best = None
        for t in np.round(np.arange(0.25, 0.96, 0.025), 3):
            m = _brand_metrics(val["rows"], t)
            ok = (m["wrong_brand"]["rate"] or 0) <= 0.01 and (m["other_packs_called_a_brand"]["rate"] or 0) <= 0.01
            if ok and (best is None or m["correct_brand"]["k"] > best[1]["correct_brand"]["k"]):
                best = (float(t), m)
        t = best[0] if best else 0.95
        test = _collect(seg, root, "test", classes, a.limit, cfg["conf"])
        per = {}
        for c in classes[1:]:
            n = sum(1 for g, _, _ in test["rows"] if g == c)
            tp = sum(1 for g, p, s in test["rows"] if g == c and p == c and s >= t)
            claimed = sum(1 for _, p, s in test["rows"] if p == c and s >= t)
            per[c] = {"test_boxes": n, "recall": _cp(tp, n), "precision": _cp(tp, claimed)}
        rep["models"][mf.name] = {
            "provider": seg.provider, "input_px": seg.imgsz, "brand_conf_chosen_on_val": t,
            "val_at_choice": best[1] if best else None,
            "test": {"photos": len(test["lat"]),
                     "pack_detection_recall_iou50": _cp(test["matched"], test["gts"]),
                     "pack_detection_precision_iou50": _cp(test["matched"], test["dets"]),
                     "mask_iou_mean_vs_sam2_labels": round(float(np.mean(test["mask_iou"])), 3) if test["mask_iou"] else None,
                     **_brand_metrics(test["rows"], t), "per_brand": per,
                     "confusion": dict(Counter(f"{g}->{p if (p and s >= t) else 'package/none'}"
                                               for g, p, s in test["rows"] if g != GENERIC and p != g).most_common(30))},
            "latency_ms_incl_mask_decode": {"p50": round(float(np.percentile(test["lat"][3:], 50)), 1),
                                            "p95": round(float(np.percentile(test["lat"][3:], 95)), 1)}}
    if a.ocr_sample:
        rep["ocr_brand_check"] = _ocr_check(root, bundle, classes, a.ocr_sample, a.device)
    default = cfg["model"]
    if default in rep["models"]:
        cfg["brand_conf"] = rep["models"][default]["brand_conf_chosen_on_val"]
        (bundle / "edge.json").write_text(json.dumps(cfg, indent=1))
    rep["caveats"] = ["Masks in the labels come from SAM2 box prompts, not hand-drawn outlines: mask IoU is "
                      "agreement with those labels.",
                      "Test photos come from the same public projects as training (split by near-duplicate group).",
                      "Brand confidence is a raw model score chosen on validation, not a calibrated probability."]
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    return rep


def _ocr_check(root: Path, bundle: Path, classes, n: int, device: str) -> dict:
    """How often printed text names the right brand on test brand packs (OCR as a second opinion)."""
    from ..edge.fields import parse_fields
    from ..edge.ocr import PackOCR

    ocr = PackOCR(bundle, use_cuda=device.startswith("cuda"))
    rng = random.Random(0)
    items = []
    for lbl in sorted((root / "labels" / "test").glob("*.txt")):
        for line in lbl.read_text().split("\n"):
            v = line.split()
            if len(v) >= 7 and int(v[0]) > 0:
                items.append((lbl.stem, int(v[0]), np.array(v[1:], float).reshape(-1, 2)))
    rng.shuffle(items)
    found = right = wrong = 0
    lat = []
    for stem, c, pts in items[:n]:
        img_p = next((root / "images" / "test").glob(stem + ".*"), None)
        img = cv2.imread(str(img_p)) if img_p else None
        if img is None:
            continue
        H, W = img.shape[:2]
        (x1, y1), (x2, y2) = (pts.min(0) * [W, H]).astype(int), (pts.max(0) * [W, H]).astype(int)
        crop = img[max(0, y1):y2, max(0, x1):x2]
        t = time.perf_counter()
        f = parse_fields(ocr.read(crop))
        lat.append((time.perf_counter() - t) * 1000)
        if "brand" in f:
            found += 1
            right += f["brand"] == classes[c]
            wrong += f["brand"] != classes[c]
    k = len(lat)
    return {"brand_packs": k, "text_named_a_brand": _cp(found, k), "text_brand_right": _cp(right, k),
            "text_brand_wrong": _cp(wrong, k),
            "ocr_ms_per_pack": {"p50": round(float(np.percentile(lat, 50)), 1) if lat else None,
                                "p95": round(float(np.percentile(lat, 95)), 1) if lat else None}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pvs.training.yolo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--data", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--min-boxes", type=int, default=80)
    d.add_argument("--max-plain", type=int, default=15000)
    d.add_argument("--max-images", type=int, default=None)
    d.add_argument("--shard", type=int, default=0)
    d.add_argument("--shards", type=int, default=1)
    d.add_argument("--device", default="cuda")
    d.add_argument("--manifest", default="models/manifest.json")
    d.add_argument("--no-sam", action="store_true")
    f = sub.add_parser("finalize")
    f.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--yaml", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model", default="yolo26n-seg.pt")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--device", default="0")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--patience", type=int, default=15)
    t.add_argument("--time-h", type=float, default=None)
    t.add_argument("--fraction", type=float, default=None)
    x = sub.add_parser("export")
    x.add_argument("--weights", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--imgsz", type=int, nargs="+", default=[640, 480])
    x.add_argument("--default-size", type=int, default=480)
    x.add_argument("--fp16", action="store_true")
    x.add_argument("--device", default="0")
    v = sub.add_parser("val")
    v.add_argument("--weights", required=True)
    v.add_argument("--yaml", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--imgsz", type=int, default=640)
    v.add_argument("--device", default="0")
    e = sub.add_parser("evaluate")
    e.add_argument("--data", required=True)
    e.add_argument("--bundle", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--device", default="cpu")
    e.add_argument("--limit", type=int, default=3000)
    e.add_argument("--ocr-sample", type=int, default=300)
    a = ap.parse_args(argv)
    rep = {"dataset": dataset, "finalize": finalize, "train": train, "export": export, "val": val,
           "evaluate": evaluate}[a.cmd](a)
    print(json.dumps(rep, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
