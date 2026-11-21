#!/usr/bin/env python3
"""Retrieve permitted pretrained weights at pinned revisions and verify SHA-256.

Usage:
  python scripts/fetch_models.py                 # download + verify against models/manifest.json
  python scripts/fetch_models.py --pin           # resolve current revisions and (re)write the manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "manifest.json"

# (key, repo, licence, intended use, files to hash)
MODELS = [
    ("grounding_dino", "IDEA-Research/grounding-dino-tiny", "Apache-2.0",
     "pretrained profile: prompted package proposals; annotation assistance", ["model.safetensors"]),
    ("sam2", "facebook/sam2.1-hiera-tiny", "Apache-2.0",
     "pretrained profile: box-prompted masks for geometry and dataset assistance", ["model.safetensors"]),
    ("dinov2", "facebook/dinov2-small", "Apache-2.0",
     "gallery embeddings (frozen) for reference matching", ["model.safetensors"]),
    ("rtdetr_r18", "PekingU/rtdetr_r18vd", "Apache-2.0",
     "COCO-initialised station detector candidate (fine-tune + export path)", ["model.safetensors"]),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    from huggingface_hub import HfApi, snapshot_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"models": {}}
    api = HfApi()
    ok = True
    for key, repo, lic, use, files in MODELS:
        if a.only and key not in a.only:
            continue
        entry = manifest["models"].get(key, {})
        rev = entry.get("revision") if not a.pin else api.model_info(repo).sha
        path = Path(snapshot_download(repo, revision=rev))
        hashes = {f: sha256(path / f) for f in files if (path / f).exists()}
        if a.pin:
            manifest["models"][key] = {"repo": repo, "revision": rev, "licence": lic, "intended_use": use,
                                       "sha256": hashes, "retrieval": f"python scripts/fetch_models.py --only {key}",
                                       "commercial_station_ok": lic == "Apache-2.0"}
            print(f"pinned {key} {repo}@{rev}")
        else:
            for f, h in hashes.items():
                exp = entry.get("sha256", {}).get(f)
                status = "OK" if h == exp else "MISMATCH"
                ok &= h == exp
                print(f"{status} {key} {f} {h[:16]}")
    if a.pin:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
