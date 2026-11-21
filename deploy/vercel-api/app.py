"""Hosted demo API for Vercel (Python on Fluid Compute, Hobby plan: 1 vCPU / 2 GB, 300 s).

The deployment holds only this file, requirements.txt and vercel.json. The `pvs` package is installed from
a pinned GitHub commit (requirements.txt). On a cold start the runtime bundle (trained ONNX detector,
hosted config, thresholds, demo catalogue) is downloaded from the `model-assets` branch, SHA-256 verified,
and unpacked under /tmp, which is the only writable path on Vercel.

Profile: RT-DETR-R18 package detector (Kaggle 2xT4 run) + crop GrabCut masks + colour-histogram gallery.
Identity on real packs is NOT validated; see evidence/training/real5star_kaggle_t4x2/README.md.
"""
from __future__ import annotations

import hashlib
import io
import os
import tarfile
import urllib.request
from pathlib import Path

BUNDLE_URL = os.environ.get(
    "PVS_BUNDLE_URL",
    "https://raw.githubusercontent.com/Rohitpro1/product-vision-station/model-assets/pvs-hosted-bundle.tgz")
BUNDLE_SHA256 = os.environ.get("PVS_BUNDLE_SHA256", "8ba6fdaa22773af1dd0e144b239ec825689f8cad10e844f8912cfcdd11fc6787")
ROOT = Path(os.environ.get("PVS_ROOT", "/tmp/pvs"))
os.environ.setdefault("PVS_CORS_ORIGINS", "https://pack-station-psi.vercel.app")


def _ensure_bundle() -> None:
    marker = ROOT / ".bundle-sha256"
    if marker.exists() and marker.read_text() == BUNDLE_SHA256:
        return
    data = urllib.request.urlopen(BUNDLE_URL, timeout=120).read()  # noqa: S310 - fixed https URL
    got = hashlib.sha256(data).hexdigest()
    if got != BUNDLE_SHA256:
        raise RuntimeError(f"bundle sha256 mismatch: {got}")
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        t.extractall(ROOT, filter="data")
    marker.write_text(BUNDLE_SHA256)


def _build():
    from pvs.api import create_app
    from pvs.config import load_station
    from pvs.pipeline import Pipeline
    from pvs.schema import Profile

    _ensure_bundle()
    os.chdir(ROOT)
    cfg = load_station(ROOT / "configs/station.hosted.yaml")
    pipeline = Pipeline(cfg, Profile.STATION, root=ROOT)
    return create_app(pipeline, cfg)


app = _build()
