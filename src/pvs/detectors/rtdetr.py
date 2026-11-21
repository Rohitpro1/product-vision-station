"""Station detector candidate: exported RT-DETR ONNX served by ONNX Runtime.

Reports the execution provider that actually handled inference. Requested
providers that are unavailable fall back to CPU and the fallback is visible
in ``backend_id``.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import Detection, ModelUnavailable


class RTDetrOnnxDetector:
    def __init__(self, path: Path, provider: str = "CPUExecutionProvider", size: int = 640, threshold: float = 0.5) -> None:
        import onnxruntime as ort

        if not Path(path).exists():
            raise ModelUnavailable(f"rtdetr onnx missing: {path}")
        avail = ort.get_available_providers()
        providers = [provider, "CPUExecutionProvider"] if provider in avail else ["CPUExecutionProvider"]
        try:
            self.sess = ort.InferenceSession(str(path), providers=providers)
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailable(f"rtdetr onnx load failed: {e}") from e
        self.size, self.threshold = size, threshold
        self.provider = self.sess.get_providers()[0]
        self.backend_id = f"rtdetr-onnx:{Path(path).name}@{self.provider}"

    def detect(self, image_bgr: np.ndarray, context: dict | None = None) -> list[Detection]:
        h, w = image_bgr.shape[:2]
        x = cv2.resize(image_bgr[:, :, ::-1], (self.size, self.size)).astype(np.float32).transpose(2, 0, 1)[None] / 255
        logits, boxes = self.sess.run(None, {"pixel_values": x})
        scores = 1 / (1 + np.exp(-logits[0, :, 0]))
        out = []
        for s, (cx, cy, bw, bh) in zip(scores, boxes[0]):
            if s >= self.threshold:
                out.append(Detection([float((cx - bw / 2) * w), float((cy - bh / 2) * h), float((cx + bw / 2) * w),
                                      float((cy + bh / 2) * h)], float(s), "package", self.backend_id))
        return out
