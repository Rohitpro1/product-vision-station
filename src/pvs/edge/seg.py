"""YOLO26-seg ONNX runtime for the laptop: NMS-free export, ONNX Runtime (CUDA if present, else CPU).

No PyTorch / Ultralytics at run time. The model file carries its class names and input size in its metadata.
Masks are decoded only inside each box, at the prototype resolution, then resized to the box: cheap.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Det:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in original image pixels
    score: float
    cls: int
    name: str
    mask: np.ndarray | None = field(default=None, repr=False)  # bool, covers the integer box (y0:y1, x0:x1)
    origin: tuple[int, int] = (0, 0)  # top-left of `mask` in image pixels
    polygon: list[list[int]] = field(default_factory=list)
    angle: float = 0.0  # long side vs image x axis, degrees in (-90, 90]

    def full_mask(self, shape) -> np.ndarray:
        m = np.zeros(shape[:2], bool)
        if self.mask is not None:
            x0, y0 = self.origin
            h, w = self.mask.shape
            m[y0:y0 + h, x0:x0 + w] = self.mask
        return m

    def upright_crop(self, img: np.ndarray, pad: float = 0.04) -> np.ndarray:
        """Crop of the pack rotated so its long side is horizontal (what OCR reads best)."""
        x1, y1, x2, y2 = self.box
        if self.polygon and len(self.polygon) >= 3:
            (cx, cy), (w, h), ang = cv2.minAreaRect(np.array(self.polygon, np.float32))
            if w < h:
                w, h, ang = h, w, ang + 90
            w, h = w * (1 + 2 * pad), h * (1 + 2 * pad)
            if abs(((ang + 90) % 180) - 90) > 5:  # rotate only when the pack is visibly tilted
                M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
                M[0, 2] += w / 2 - cx
                M[1, 2] += h / 2 - cy
                return cv2.warpAffine(img, M, (max(8, int(w)), max(8, int(h))), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        X0, Y0 = int(max(0, x1 - px)), int(max(0, y1 - py))
        X1, Y1 = int(min(img.shape[1], x2 + px)), int(min(img.shape[0], y2 + py))
        c = img[Y0:Y1, X0:X1]
        return cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE) if c.shape[0] > 1.3 * c.shape[1] else c


def providers(device: str = "auto") -> list:
    import onnxruntime as ort

    avail = ort.get_available_providers()
    if device in ("auto", "cuda") and "CUDAExecutionProvider" in avail:
        if hasattr(ort, "preload_dlls"):  # onnxruntime-gpu[cuda,cudnn]: load the pip-installed CUDA libraries
            try:
                ort.preload_dlls()
            except Exception:  # noqa: BLE001, S110 - system CUDA may still work
                pass
        return [("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kSameAsRequested"}),
                "CPUExecutionProvider"]
    if device == "cuda":
        raise RuntimeError("CUDA requested but onnxruntime-gpu / CUDA is not available: " + ", ".join(avail))
    return ["CPUExecutionProvider"]


class YoloSeg:
    def __init__(self, path: str, device: str = "auto", threads: int | None = None) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_cpu_mem_arena = False  # keeps resident memory low on a 4 GB laptop
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=providers(device))
        meta = self.sess.get_modelmeta().custom_metadata_map
        if meta.get("end2end") != "True":
            raise ValueError("export the model with end2end=True (NMS-free); see pvs.training.yolo export")
        self.names = {int(k): v for k, v in ast.literal_eval(meta["names"]).items()}
        self.imgsz = int(self.sess.get_inputs()[0].shape[2])
        self.input = self.sess.get_inputs()[0].name
        self.provider = self.sess.get_providers()[0]
        self.fp16 = self.sess.get_inputs()[0].type == "tensor(float16)"

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = self.imgsz / max(h, w)
        nh, nw = round(h * r), round(w * r)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, left, top

    def __call__(self, img: np.ndarray, conf: float = 0.25, masks: bool = True) -> list[Det]:
        lb, r, left, top = self._letterbox(img)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float16 if self.fp16 else np.float32) / 255.0
        out, protos = self.sess.run(None, {self.input: np.ascontiguousarray(x)})
        out, protos = out[0].astype(np.float32), protos[0].astype(np.float32)
        out = out[out[:, 4] >= conf]
        nm, ph, pw = protos.shape
        P = protos.reshape(nm, -1)
        s = ph / self.imgsz
        H, W = img.shape[:2]
        dets = []
        for row in out:
            bx = row[:4]
            ob = np.array([(bx[0] - left) / r, (bx[1] - top) / r, (bx[2] - left) / r, (bx[3] - top) / r])
            ob = np.clip(ob, 0, [W, H, W, H])
            if ob[2] - ob[0] < 2 or ob[3] - ob[1] < 2:
                continue
            c = int(row[5])
            d = Det(tuple(float(v) for v in ob), float(row[4]), c, self.names.get(c, str(c)))
            if masks:
                px0, py0 = int(max(0, np.floor(bx[0] * s))), int(max(0, np.floor(bx[1] * s)))
                px1, py1 = int(min(pw, np.ceil(bx[2] * s))), int(min(ph, np.ceil(bx[3] * s)))
                if px1 > px0 and py1 > py0:
                    full = (row[6:6 + nm] @ P).reshape(ph, pw)
                    sub = full[py0:py1, px0:px1]
                    X0, Y0 = int(ob[0]), int(ob[1])
                    X1, Y1 = int(np.ceil(ob[2])), int(np.ceil(ob[3]))
                    # the proto crop covers [px0/s, px1/s] in letterbox pixels -> map to original pixels
                    sx0, sy0 = (px0 / s - left) / r, (py0 / s - top) / r
                    sx1, sy1 = (px1 / s - left) / r, (py1 / s - top) / r
                    big = cv2.resize(sub, (max(1, round(sx1 - sx0)), max(1, round(sy1 - sy0))),
                                     interpolation=cv2.INTER_LINEAR) > 0  # logit > 0 == sigmoid > 0.5
                    ox, oy = X0 - round(sx0), Y0 - round(sy0)
                    m = np.zeros((Y1 - Y0, X1 - X0), bool)
                    src = big[max(0, oy):max(0, oy) + m.shape[0], max(0, ox):max(0, ox) + m.shape[1]]
                    m[:src.shape[0], :src.shape[1]] = src
                    d.mask, d.origin = m, (X0, Y0)
                    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cs:
                        cnt = max(cs, key=cv2.contourArea)
                        cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
                        d.polygon = (cnt.reshape(-1, 2) + [X0, Y0]).astype(int).tolist()
                        if len(cnt) >= 3:
                            (_, _), (rw, rh), ang = cv2.minAreaRect(cnt.astype(np.float32))
                            a = ang if rw >= rh else ang - 90
                            d.angle = float(-((-a + 90) % 180 - 90))  # fold to (-90, 90]
            dets.append(d)
        return dets
