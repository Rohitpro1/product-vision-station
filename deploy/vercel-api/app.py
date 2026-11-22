"""Hosted API for Vercel (Python on Fluid Compute, Hobby plan: 1 vCPU / 2 GB, 300 s).

The deployment holds only this file, requirements.txt and vercel.json. The `pvs` package is installed from
a pinned GitHub commit (requirements.txt). On a cold start the model release named in `release.json` on
the repository's `model-assets` branch is downloaded, every file SHA-256 verified, and unpacked under /tmp
(the only writable path on Vercel). The Hugging Face Space reads the same file, so both hosts serve the
same release; see deploy/hf-space/app.py for the schema. Each warm instance keeps the release it started
with; new instances pick up a changed release.json.

If release.json can't be read, the built-in v1 release below is served.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

RELEASE_URL = os.environ.get(
    "PVS_RELEASE_URL",
    "https://raw.githubusercontent.com/Rohitpro1/product-vision-station/model-assets/release.json")
ROOT = Path(os.environ.get("PVS_ROOT", "/tmp/pvs"))
os.environ.setdefault("PVS_CORS_ORIGINS", "https://pack-station-psi.vercel.app")

BUILTIN_RELEASE = {
    "release": "v1-hosted-onnx-demo",
    "config": "configs/station.hosted.yaml",
    "files": [{
        "url": "https://raw.githubusercontent.com/Rohitpro1/product-vision-station/model-assets/pvs-hosted-bundle.tgz",
        "sha256": "8ba6fdaa22773af1dd0e144b239ec825689f8cad10e844f8912cfcdd11fc6787",
        "extract": True,
    }],
}


def _fetch_release() -> dict:
    try:
        with urllib.request.urlopen(RELEASE_URL, timeout=20) as r:  # noqa: S310 - fixed https URL
            rel = json.loads(r.read())
        assert isinstance(rel["release"], str) and rel["release"] and "/" not in rel["release"]
        assert isinstance(rel["config"], str) and rel["files"]
        for f in rel["files"]:
            assert f["url"].startswith("https://") and len(f["sha256"]) == 64 and (f.get("extract") or f.get("path"))
        return rel
    except Exception as e:  # noqa: BLE001 - fall back to the built-in release
        print("release.json unreadable, using built-in release:", e, flush=True)
        return BUILTIN_RELEASE


def _download(url: str, sha256: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as out:  # noqa: S310 - https only
        while chunk := r.read(1 << 20):
            h.update(chunk)
            out.write(chunk)
    if h.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {url}: {h.hexdigest()}")
    tmp.replace(dest)


def _materialise(rel: dict) -> Path:
    d = ROOT / "releases" / rel["release"]
    marker = d / ".complete"
    if marker.exists() and marker.read_text() == json.dumps(rel, sort_keys=True):
        return d
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for i, f in enumerate(rel["files"]):
        if f.get("extract"):
            tgz = ROOT / f"dl-{i}.tgz"
            _download(f["url"], f["sha256"], tgz)
            with tarfile.open(tgz, mode="r:gz") as t:
                t.extractall(d, filter="data")
            tgz.unlink(missing_ok=True)
        else:
            target = (d / f["path"]).resolve()
            if not target.is_relative_to(d.resolve()):
                raise ValueError(f"path escapes the release folder: {f['path']}")
            _download(f["url"], f["sha256"], target)
    marker.write_text(json.dumps(rel, sort_keys=True))
    return d


def _build():
    from pvs.api import create_app
    from pvs.config import load_station
    from pvs.pipeline import Pipeline
    from pvs.schema import Profile

    rel = _fetch_release()
    try:
        d = _materialise(rel)
    except Exception as e:  # noqa: BLE001 - a broken release must not take the API down
        if rel is BUILTIN_RELEASE:
            raise
        print(f"release {rel['release']} failed, using built-in release:", e, flush=True)
        rel = BUILTIN_RELEASE
        d = _materialise(rel)
    os.chdir(d)
    cfg = load_station(d / rel["config"])
    pipeline = Pipeline(cfg, Profile.STATION, root=d)
    print(f"serving {rel['release']} (bundle {cfg.bundle_id})", flush=True)
    return create_app(pipeline, cfg)


app = _build()
