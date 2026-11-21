"""Real pretrained backends via Hugging Face transformers (pinned revisions in models/manifest.json).

* Grounding DINO (prompted package proposals)
* SAM 2 (box-prompted masks)
* DINOv2 (gallery embeddings) - see ``pvs.gallery``
"""
from __future__ import annotations

import os
import time

import numpy as np

from .base import Detection, ModelUnavailable


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1"


class GroundingDinoDetector:
    def __init__(self, model_id: str, revision: str, prompts: list[str], box_threshold: float = 0.3,
                 text_threshold: float = 0.25, max_side: int = 800, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            t0 = time.perf_counter()
            self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, revision=revision).to(device).eval()
            self.load_s = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001 - surfaced as a degraded state
            raise ModelUnavailable(f"grounding-dino load failed: {type(e).__name__}: {e}") from e
        self.torch = torch
        self.device = device
        self.prompts = prompts
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_side = max_side
        self.backend_id = f"grounding-dino:{model_id}@{revision[:12]}"

    def detect(self, image_bgr: np.ndarray, context: dict | None = None) -> list[Detection]:
        from PIL import Image

        h, w = image_bgr.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        img = Image.fromarray(image_bgr[:, :, ::-1])
        if s < 1.0:
            img = img.resize((round(w * s), round(h * s)))
        text = ". ".join(self.prompts) + "."
        inputs = self.processor(images=img, text=text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[(h, w)])[0]  # boxes returned directly in source pixels
        out = []
        labels = res.get("text_labels", res.get("labels"))
        for box, score, label in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
            x0, y0, x1, y1 = box
            # discard whole-image proposals
            if (x1 - x0) * (y1 - y0) > 0.9 * w * h:
                continue
            out.append(Detection([max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)],
                                 float(score), str(label), self.backend_id))
        return _nms(out, 0.6)


class Sam2MaskProvider:
    def __init__(self, model_id: str, revision: str, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor

            self.processor = Sam2Processor.from_pretrained(model_id, revision=revision)
            self.model = Sam2Model.from_pretrained(model_id, revision=revision).to(device).eval()
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailable(f"sam2 load failed: {type(e).__name__}: {e}") from e
        self.torch = torch
        self.device = device
        self.backend_id = f"sam2:{model_id}@{revision[:12]}"

    def masks(self, image_bgr: np.ndarray, boxes: list[list[float]]) -> list[tuple[np.ndarray | None, str]]:
        from PIL import Image

        if not boxes:
            return []
        img = Image.fromarray(image_bgr[:, :, ::-1])
        inputs = self.processor(images=img, input_boxes=[[list(map(float, b)) for b in boxes]],
                                return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]
        res = []
        for i in range(len(boxes)):
            m = masks[i, 0].numpy().astype(bool)
            res.append((m, "segmentation") if m.sum() > 50 else (None, "none"))
        return res


def _iou(a: list[float], b: list[float]) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(dets: list[Detection], thr: float) -> list[Detection]:
    dets = sorted(dets, key=lambda d: -d.score)
    keep: list[Detection] = []
    for d in dets:
        if all(_iou(d.box_xyxy, k.box_xyxy) < thr for k in keep):
            keep.append(d)
    return keep
