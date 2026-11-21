"""Landmark (ordered A/B) + visibility + face head: training, resume, ONNX export, parity check.

This model predicts *ordered* physical endpoints A and B (not an unordered
shape-endpoint pair). Loss = visibility-masked smooth-L1 on A/B + BCE on
visibility + CE on face. Augmentation applies one affine matrix to the image
and both labels, so endpoint order is preserved under rotation.

    pvs train landmark --gt data/fixtures/scenes/ground_truth.json --steps 40 --out runs/lm
    pvs train landmark --gt ... --steps 60 --out runs/lm --resume
    pvs train export --ckpt runs/lm/last.pt --out runs/lm/landmark.onnx
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np

FACES = ["front", "back", "edge"]
S = 128


def _sample(img, ins, rng, train=True):
    x0, y0, x1, y1 = ins["box_xyxy"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1.15 + (rng.uniform(-0.1, 0.15) if train else 0))
    if train:
        cx += rng.uniform(-0.06, 0.06) * side
        cy += rng.uniform(-0.06, 0.06) * side
    ang = rng.uniform(-180, 180) if train else 0.0
    M = cv2.getRotationMatrix2D((cx, cy), ang, S / side)
    M[0, 2] += S / 2 - cx
    M[1, 2] += S / 2 - cy
    crop = cv2.warpAffine(img, M, (S, S), borderValue=(114, 114, 114))
    if train and rng.random() < 0.5:
        crop = np.clip(crop.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
    A = M @ np.array([*ins["A"], 1.0])
    B = M @ np.array([*ins["B"], 1.0])
    vis = 1.0 if ins["heading_observable"] else 0.0
    return crop, np.array([*A / S, *B / S], np.float32), vis, FACES.index(ins["face"]) if ins["face"] in FACES else 0


def build_model():
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(blk(3, 16), blk(16, 32), blk(32, 64), blk(64, 96), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.lm = nn.Linear(96, 4)
            self.vis = nn.Linear(96, 1)
            self.face = nn.Linear(96, 3)

        def forward(self, x):
            h = self.f(x)
            return self.lm(h).sigmoid(), self.vis(h), self.face(h)

    return Head()


def _load(gt_path: Path):
    gt = json.loads(gt_path.read_text())
    items = []
    for sc in gt["scenes"]:
        for k, ins in enumerate(sc["instances"]):
            if ins.get("A") and ins["occluded_frac"] <= 0.4:
                items.append({"id": f"{sc['scene_id']}#{k}", "session_id": sc["scene_id"], "image": sc["image"], "ins": ins})
    return gt_path.parent, items


def train(a) -> dict:
    import torch
    import torch.nn.functional as F

    from .data import assign_split, leakage_check, split_hash

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    root, items = _load(Path(a.gt))
    splits = {"train": [], "val": [], "test": []}
    for it in items:
        splits[assign_split(it["session_id"])].append(it)
    leaks = leakage_check(splits, keys=("session_id",))
    if leaks:
        raise SystemExit(f"leakage: {leaks}")
    imgs = {it["image"]: cv2.imread(str(root / it["image"])) for it in items}
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    step0 = 0
    ck = out / "last.pt"
    if a.resume and ck.exists():
        st = torch.load(ck, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step0 = st["step"]
        rng.setstate(st["rng"])
        torch.set_rng_state(st["torch_rng"])
    log = []
    tr = splits["train"] or items
    t0 = time.perf_counter()
    for step in range(step0, a.steps):
        batch = [_sample(imgs[it["image"]], it["ins"], rng) for it in rng.choices(tr, k=a.batch)]
        x = torch.tensor(np.stack([b[0] for b in batch])).permute(0, 3, 1, 2).float() / 255
        y = torch.tensor(np.stack([b[1] for b in batch]))
        v = torch.tensor([b[2] for b in batch]).float()
        fc = torch.tensor([b[3] for b in batch])
        model.train()
        lm, vl, fl = model(x)
        l_lm = (F.smooth_l1_loss(lm, y, reduction="none").mean(1) * v).sum() / v.sum().clamp(min=1)
        loss = l_lm + F.binary_cross_entropy_with_logits(vl.squeeze(1), v) + F.cross_entropy(fl, fc)
        opt.zero_grad()
        loss.backward()
        opt.step()
        log.append({"step": step + 1, "loss": round(float(loss), 4), "lm": round(float(l_lm), 4)})
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": a.steps, "rng": rng.getstate(),
                "torch_rng": torch.get_rng_state(), "seed": a.seed}, ck)
    val = evaluate(model, splits["val"], imgs)
    rep = {"steps_run": a.steps - step0, "resumed_from": step0, "final_step": a.steps, "train_s": round(time.perf_counter() - t0, 1),
           "n_items": {k: len(v) for k, v in splits.items()}, "split_hash": split_hash(splits), "leakage": leaks,
           "val": val, "log_tail": log[-5:], "predicts": "ordered A/B landmarks + visibility + face",
           "data": str(a.gt), "data_sha256": hashlib.sha256(Path(a.gt).read_bytes()).hexdigest()[:16],
           "note": "fixture-data smoke run; weights are NOT eligible for the production bundle"}
    with open(out / "metrics.jsonl", "a") as fh:
        for r in log:
            fh.write(json.dumps(r) + "\n")
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


def evaluate(model, items, imgs) -> dict:
    import torch

    from ..geometry import heading_deg, periodic_error

    if not items:
        return {"status": "insufficient_evidence", "n": 0}
    rng = random.Random(0)
    errs, face_ok = [], 0
    model.eval()
    with torch.no_grad():
        for it in items:
            c, y, v, f = _sample(imgs[it["image"]], it["ins"], rng, train=False)
            lm, vl, fl = model(torch.tensor(c[None]).permute(0, 3, 1, 2).float() / 255)
            p = lm[0].numpy() * S
            if v > 0:
                errs.append(periodic_error(heading_deg(tuple(p[:2]), tuple(p[2:])), heading_deg(tuple(y[:2] * S), tuple(y[2:] * S)), 360))
            face_ok += int(int(fl.argmax()) == f)
    return {"n": len(items), "heading_n": len(errs), "heading_median_err_deg": float(np.median(errs)) if errs else None,
            "face_acc": face_ok / len(items)}


def inspect_augmentation(a) -> Path:
    """Write a grid of augmented crops with A (red) and B (blue) to check label transforms by eye."""
    root, items = _load(Path(a.gt))
    rng = random.Random(a.seed)
    tiles = []
    for it in items[:16]:
        img = cv2.imread(str(root / it["image"]))
        c, y, v, f = _sample(img, it["ins"], rng)
        c = c.copy()
        cv2.circle(c, (int(y[0] * S), int(y[1] * S)), 5, (0, 0, 255), -1)
        cv2.circle(c, (int(y[2] * S), int(y[3] * S)), 5, (255, 0, 0), -1)
        cv2.putText(c, FACES[f][0], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        tiles.append(c)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    p = Path(a.out) / "augmentation_check.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), grid)
    return p


def export(a) -> dict:
    import onnxruntime as ort
    import torch

    model = build_model()
    model.load_state_dict(torch.load(a.ckpt, weights_only=False)["model"])
    model.eval()
    x = torch.rand(2, 3, S, S)
    torch.onnx.export(model, (x,), a.out, input_names=["image"], output_names=["landmarks", "visibility", "face"],
                      dynamic_axes={"image": {0: "n"}}, opset_version=17, dynamo=False)
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = [t.numpy() for t in model(x)]
    got = sess.run(None, {"image": x.numpy()})
    diff = max(float(np.abs(r - g).max()) for r, g in zip(ref, got))
    rep = {"onnx": a.out, "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest(), "max_abs_diff": diff,
           "provider": sess.get_providers()[0], "parity_ok": diff < 1e-4}
    print(json.dumps(rep))
    return rep


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="pvs train")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("landmark")
    t.add_argument("--gt", required=True)
    t.add_argument("--out", default="runs/landmark")
    t.add_argument("--steps", type=int, default=40)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--resume", action="store_true")
    e = sub.add_parser("export")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", required=True)
    i = sub.add_parser("inspect-aug")
    i.add_argument("--gt", required=True)
    i.add_argument("--out", default="runs/landmark")
    i.add_argument("--seed", type=int, default=3)
    d = sub.add_parser("detector-smoke", help="RT-DETR-R18 fine-tune smoke + ONNX export")
    d.add_argument("--gt", required=True)
    d.add_argument("--out", default="runs/rtdetr")
    d.add_argument("--steps", type=int, default=2)
    d.add_argument("--size", type=int, default=320)
    a = ap.parse_args(argv)
    if a.cmd == "landmark":
        print(json.dumps(train(a), indent=1))
    elif a.cmd == "export":
        return 0 if export(a)["parity_ok"] else 1
    elif a.cmd == "inspect-aug":
        print(inspect_augmentation(a))
    elif a.cmd == "detector-smoke":
        from .detector import smoke
        print(json.dumps(smoke(a), indent=1))
    _ = math
    return 0
