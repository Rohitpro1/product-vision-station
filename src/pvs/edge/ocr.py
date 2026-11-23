"""PP-OCRv5 mobile (Apache-2.0) through RapidOCR + ONNX Runtime, run on one pack crop at a time.

Model files come from the bundle (no download at run time). Crops are capped at `max_side` pixels so a close-up
photo does not make OCR slow.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FILES = {"det": "ocr/PP-OCRv5_det_mobile.onnx", "rec": "ocr/en_PP-OCRv5_rec_mobile.onnx",
         "cls": "ocr/ppocr_mobile_v2.0_cls.onnx"}


class PackOCR:
    def __init__(self, root: str | Path, use_cuda: bool = False, max_side: int = 640, threads: int = 2) -> None:
        from rapidocr import RapidOCR

        root = Path(root)
        missing = [str(root / f) for f in FILES.values() if not (root / f).exists()]
        if missing:
            raise FileNotFoundError("OCR model files missing: " + ", ".join(missing))
        self.max_side = max_side
        self.engine = RapidOCR(params={
            "Global.log_level": "error",
            "Global.text_score": 0.6,
            "Det.model_path": str(root / FILES["det"]),
            "Rec.model_path": str(root / FILES["rec"]),
            "Cls.model_path": str(root / FILES["cls"]),
            "Det.limit_type": "max",
            "Det.limit_side_len": max_side,
            "EngineConfig.onnxruntime.use_cuda": use_cuda,
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
        })

    def read(self, crop: np.ndarray) -> list[str]:
        """Text lines of one pack in reading order (top to bottom, left to right)."""
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 12:
            return []
        s = self.max_side / max(crop.shape[:2])
        if s < 1:
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        r = self.engine(crop)
        if not r.txts:
            return []
        items = sorted(zip(r.boxes, r.txts), key=lambda bt: (round(float(np.mean(bt[0][:, 1])) / 12), float(bt[0][0][0])))
        return [t.strip() for _, t in items if t.strip()]
