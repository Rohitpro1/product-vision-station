"""RT-DETR-R18 (COCO-initialised) generic ``package`` detector: fine-tune entry point + ONNX export.

Smoke mode runs a few real optimizer steps on fixture data, checks the loss is
finite, saves/reloads the checkpoint and exports ONNX, then compares ONNX
Runtime logits with the PyTorch reference. Fixture-trained weights are never
eligible for a production bundle.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np


def _batch(gt_root: Path, scenes: list[dict], size: int):
    import torch

    pix, labels = [], []
    for sc in scenes:
        img = cv2.imread(str(gt_root / sc["image"]))[:, :, ::-1]
        h, w = img.shape[:2]
        r = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
        pix.append(torch.tensor(r).permute(2, 0, 1))
        boxes = []
        for ins in sc["instances"]:
            x0, y0, x1, y1 = ins["box_xyxy"]
            boxes.append([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h])
        labels.append({"class_labels": torch.zeros(len(boxes), dtype=torch.long),
                       "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)})
    return torch.stack(pix), labels


def smoke(a) -> dict:
    import onnxruntime as ort
    import torch
    from transformers import RTDetrForObjectDetection

    man = json.loads(Path("models/manifest.json").read_text())["models"]["rtdetr_r18"]
    torch.manual_seed(0)
    model = RTDetrForObjectDetection.from_pretrained(man["repo"], revision=man["revision"], num_labels=1,
                                                     id2label={0: "package"}, label2id={"package": 0},
                                                     ignore_mismatched_sizes=True)
    gt_path = Path(a.gt)
    gt = json.loads(gt_path.read_text())
    scenes = [s for s in gt["scenes"] if s["instances"]][:4]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    t0 = time.perf_counter()
    model.train()
    for step in range(a.steps):
        x, y = _batch(gt_path.parent, scenes[step % 2 * 2: step % 2 * 2 + 2], a.size)
        out = model(pixel_values=x, labels=y)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "ckpt")
    reloaded = RTDetrForObjectDetection.from_pretrained(out_dir / "ckpt").eval()
    model.eval()
    x, _ = _batch(gt_path.parent, scenes[:1], a.size)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            o = self.m(pixel_values=pixel_values)
            return o.logits, o.pred_boxes

    onnx_path = out_dir / "rtdetr_r18_package.onnx"
    with torch.no_grad():
        ref_l, ref_b = Wrap(reloaded)(x)
        torch.onnx.export(Wrap(reloaded), (x,), str(onnx_path), input_names=["pixel_values"],
                          output_names=["logits", "pred_boxes"], opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got_l, got_b = sess.run(None, {"pixel_values": x.numpy()})
    rep = {"steps": a.steps, "losses": [round(v, 4) for v in losses], "finite": all(np.isfinite(losses)),
           "train_s": round(time.perf_counter() - t0, 1), "input_size": a.size, "init": f"{man['repo']}@{man['revision'][:12]}",
           "onnx": str(onnx_path), "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
           "max_abs_diff_logits": float(np.abs(ref_l.numpy() - got_l).max()),
           "max_abs_diff_boxes": float(np.abs(ref_b.numpy() - got_b).max()), "provider": sess.get_providers()[0],
           "note": "fixture smoke only; not a trained station detector"}
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    return rep
