"""Group-aware splitting, leakage checks and CVAT-compatible export.

Groups (physical pack, clip, capture session) are assigned to train / val /
sealed test *before* frames are extracted, by hashing the group key. Every
item in a connected group lands on the same side.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def assign_split(group_key: str, ratios=(0.7, 0.15, 0.15), salt: str = "pvs-split-v1") -> str:
    h = int(hashlib.sha256(f"{salt}|{group_key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < ratios[0]:
        return "train"
    return "val" if h < ratios[0] + ratios[1] else "test"


def connected_groups(items: list[dict], keys=("pack_id", "clip_id", "session_id")) -> dict[str, str]:
    """Union-find over shared pack/clip/session IDs -> representative group per item id."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for it in items:
        nodes = [f"{k}:{it[k]}" for k in keys if it.get(k)]
        find(f"item:{it['id']}")
        for n in nodes:
            union(f"item:{it['id']}", n)
    return {it["id"]: find(f"item:{it['id']}") for it in items}


def dhash(img: np.ndarray, size: int = 8) -> int:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img, (size + 1, size))
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def leakage_check(split_items: dict[str, list[dict]], keys=("pack_id", "clip_id", "session_id"),
                  max_hamming: int = 4) -> list[str]:
    """Return violations: shared group IDs or near-duplicate images across splits."""
    problems = []
    names = list(split_items)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for k in keys:
                sa = {x[k] for x in split_items[a] if x.get(k)}
                sb = {x[k] for x in split_items[b] if x.get(k)}
                if sa & sb:
                    problems.append(f"{k} shared between {a} and {b}: {sorted(sa & sb)[:3]}")
            ha = [x["dhash"] for x in split_items[a] if "dhash" in x]
            hb = [x["dhash"] for x in split_items[b] if "dhash" in x]
            for x in ha:
                for y in hb:
                    if bin(x ^ y).count("1") <= max_hamming:
                        problems.append(f"near-duplicate across {a}/{b} (hamming<={max_hamming})")
                        break
    return problems


def split_hash(split_items: dict[str, list[dict]]) -> dict[str, str]:
    return {k: hashlib.sha256(json.dumps(sorted(x["id"] for x in v)).encode()).hexdigest()[:16]
            for k, v in split_items.items()}


def to_cvat_coco(gt: dict, out: Path) -> Path:
    """COCO-keypoints JSON (importable by CVAT): box, A/B landmarks with visibility, face/identity attributes."""
    images, anns = [], []
    aid = 1
    for i, sc in enumerate(gt["scenes"], 1):
        images.append({"id": i, "file_name": sc["image"], "width": sc["width"], "height": sc["height"]})
        for ins in sc["instances"]:
            x0, y0, x1, y1 = ins["box_xyxy"]
            v = 2 if ins["heading_observable"] else 1
            kps = [*ins["A"], v, *ins["B"], v] if ins.get("A") else [0, 0, 0, 0, 0, 0]
            anns.append({"id": aid, "image_id": i, "category_id": 1, "bbox": [x0, y0, x1 - x0, y1 - y0],
                         "area": (x1 - x0) * (y1 - y0), "iscrowd": 0, "keypoints": kps, "num_keypoints": 2,
                         "attributes": {"face": ins["face"], "sku": ins["sku"], "family": ins["family"],
                                        "occluded_frac": ins["occluded_frac"],
                                        "ambiguity": None if ins["heading_observable"] else "endpoint_unobservable"}})
            aid += 1
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "package", "keypoints": ["end_A", "end_B"], "skeleton": [[1, 2]]}]}
    out.write_text(json.dumps(coco))
    return out
