"""Reference-gallery identity: frozen embeddings + deterministic cosine search + open-set rejection."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detectors.base import ModelUnavailable


class ColourHistEmbedder:
    """Classical, rotation-invariant HSV histogram + edge-orientation embedding.

    Used by the fixture profile. It is a real (if weak) algorithm, not a scripted
    output, and is labelled ``colour-hist-v1`` in every result.
    """

    backend_id = "colour-hist-v1"

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        out = []
        for c in crops:
            hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
            h = np.sqrt(h / max(h.sum(), 1))
            out.append(h / (np.linalg.norm(h) + 1e-9))
        return np.asarray(out, np.float32)


class DinoV2Embedder:
    def __init__(self, model_id: str, revision: str, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            self.proc = AutoImageProcessor.from_pretrained(model_id, revision=revision)
            self.model = AutoModel.from_pretrained(model_id, revision=revision).to(device).eval()
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailable(f"dinov2 load failed: {type(e).__name__}: {e}") from e
        self.torch = torch
        self.device = device
        self.backend_id = f"dinov2:{model_id}@{revision[:12]}"

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 384), np.float32)
        rgb = [np.ascontiguousarray(c[:, :, ::-1]) for c in crops]
        inputs = self.proc(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            o = self.model(**inputs)
        cls = o.last_hidden_state[:, 0]
        patch = o.last_hidden_state[:, 1:].mean(1)
        v = self.torch.cat([cls, patch], 1).cpu().numpy().astype(np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


@dataclass
class Ref:
    sku: str  # "negative:<name>" for explicit non-target references
    family: str | None
    side: str
    revision: str
    view: str  # canonical | rot180


@dataclass
class GalleryHit:
    candidates: list[dict]  # best score per SKU, sorted desc
    top_sku: str | None
    top_family: str | None
    top_side: str | None
    score: float
    margin: float  # top score minus best competing *different* SKU (incl. negatives)
    competitor: str | None


class Gallery:
    def __init__(self, vectors: np.ndarray, refs: list[Ref], backend_id: str, catalogue_revision: str) -> None:
        self.vectors = vectors.astype(np.float32)
        self.refs = refs
        self.backend_id = backend_id
        self.catalogue_revision = catalogue_revision

    @staticmethod
    def views(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
        return [("canonical", img), ("rot180", cv2.rotate(img, cv2.ROTATE_180))]

    @classmethod
    def build(cls, catalogue: dict, root: Path, embedder) -> Gallery:
        crops, refs = [], []
        for prod in catalogue["products"]:
            if prod.get("approval_state") not in ("approved", "fixture"):
                continue
            for r in prod["references"]:
                img = cv2.imread(str(root / r["path"]))
                if img is None:
                    raise FileNotFoundError(root / r["path"])
                for view, v in cls.views(img):
                    crops.append(v)
                    refs.append(Ref(prod["sku"], prod.get("family"), r["side"], prod["artwork_revision"], view))
        for neg in catalogue.get("negatives", []):
            img = cv2.imread(str(root / neg["path"]))
            if img is None:
                raise FileNotFoundError(root / neg["path"])
            for view, v in cls.views(img):
                crops.append(v)
                refs.append(Ref(f"negative:{neg['name']}", None, "any", neg.get("revision", "1"), view))
        return cls(embedder.embed(crops), refs, embedder.backend_id, catalogue["revision"])

    def save(self, path: Path) -> None:
        np.savez(path, vectors=self.vectors,
                 refs=json.dumps([r.__dict__ for r in self.refs]),
                 meta=json.dumps({"backend": self.backend_id, "catalogue_revision": self.catalogue_revision}))

    @classmethod
    def load(cls, path: Path) -> Gallery:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(z["vectors"], [Ref(**r) for r in json.loads(str(z["refs"]))], meta["backend"],
                   meta["catalogue_revision"])

    def digest(self) -> str:
        return hashlib.sha256(self.vectors.tobytes()).hexdigest()[:16]

    def search(self, q: np.ndarray, k: int = 5) -> GalleryHit:
        sims = self.vectors @ q.astype(np.float32)
        best: dict[str, tuple[float, Ref]] = {}
        for s, r in zip(sims.tolist(), self.refs):
            if r.sku not in best or s > best[r.sku][0]:
                best[r.sku] = (s, r)
        ranked = sorted(best.values(), key=lambda t: -t[0])
        cands = [{"sku": r.sku, "family": r.family or "", "side": r.side, "score": float(s),
                  "reference_revision": r.revision} for s, r in ranked[:k]]
        if not ranked:
            return GalleryHit([], None, None, None, 0.0, 0.0, None)
        s0, r0 = ranked[0]
        s1, r1 = ranked[1] if len(ranked) > 1 else (-1.0, None)
        return GalleryHit(cands, r0.sku, r0.family, r0.side, float(s0), float(s0 - s1),
                          r1.sku if r1 else None)
