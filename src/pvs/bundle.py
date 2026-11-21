"""Signed model bundles: build, Ed25519 sign/verify, atomic install and rollback.

A bundle is a directory with ``bundle.json`` (files + SHA-256, schema and
catalogue compatibility) and ``bundle.sig`` (Ed25519 over the canonical JSON).
Private keys live outside the repository; ``deploy/trust/*.pub`` holds the
operator-provisioned public keys. Install copies into
``<station>/bundles/<id>`` and swaps the ``current`` symlink atomically; the
previous target is kept as ``previous`` for rollback. Corrupt or unsigned
bundles are rejected before the swap.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .schema import SCHEMA_VERSION


class BundleError(RuntimeError):
    pass


def keygen(priv: Path, pub: Path) -> None:
    k = Ed25519PrivateKey.generate()
    priv.parent.mkdir(parents=True, exist_ok=True)
    pub.parent.mkdir(parents=True, exist_ok=True)
    priv.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                     serialization.NoEncryption()))
    os.chmod(priv, 0o600)
    pub.write_bytes(k.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _canon(d: dict) -> bytes:
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


def build(src: Path, out: Path, bundle_id: str, key: Path, meta: dict) -> dict:
    """``src`` holds preprocessing config, models, catalogue, references, calibration and thresholds."""
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("bundle.json", "bundle.sig"))
    files = {str(p.relative_to(out)): _sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    man = {"bundle_id": bundle_id, "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
           "schema_version": SCHEMA_VERSION, "files": files, **meta}
    (out / "bundle.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    k = serialization.load_pem_private_key(key.read_bytes(), None)
    (out / "bundle.sig").write_bytes(k.sign(_canon(man)))
    return man


def verify(bdir: Path, trust: Path) -> dict:
    mp, sp = bdir / "bundle.json", bdir / "bundle.sig"
    if not mp.exists() or not sp.exists():
        raise BundleError("missing manifest or signature")
    man = json.loads(mp.read_text())
    sig = sp.read_bytes()
    keys = [serialization.load_pem_public_key(p.read_bytes()) for p in sorted(trust.glob("*.pub"))]
    if not keys:
        raise BundleError(f"no trusted keys in {trust}")
    if not any(_ok(k, sig, _canon(man)) for k in keys):
        raise BundleError("signature verification failed")
    for rel, h in man["files"].items():
        p = bdir / rel
        if not p.exists() or _sha(p) != h:
            raise BundleError(f"file hash mismatch: {rel}")
    extra = {str(p.relative_to(bdir)) for p in bdir.rglob("*") if p.is_file()} - set(man["files"]) - {"bundle.json", "bundle.sig"}
    if extra:
        raise BundleError(f"unlisted files: {sorted(extra)[:3]}")
    if man.get("schema_version") != SCHEMA_VERSION:
        raise BundleError(f"incompatible schema {man.get('schema_version')} != {SCHEMA_VERSION}")
    return man


def _ok(k: Ed25519PublicKey, sig: bytes, data: bytes) -> bool:
    try:
        k.verify(sig, data)
        return True
    except InvalidSignature:
        return False


def _swap(link: Path, target: Path) -> None:
    tmp = link.with_name(link.name + ".tmp")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(target)
    os.replace(tmp, link)  # atomic on POSIX


def install(bdir: Path, station: Path, trust: Path) -> str:
    man = verify(bdir, trust)
    dest = station / "bundles" / man["bundle_id"]
    if not dest.exists():
        tmp = dest.with_name(dest.name + ".partial")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(bdir, tmp)
        verify(tmp, trust)
        os.replace(tmp, dest)
    cur = station / "current"
    if cur.is_symlink():
        _swap(station / "previous", cur.resolve())
    _swap(cur, dest.resolve())
    return man["bundle_id"]


def rollback(station: Path, trust: Path) -> str:
    prev = station / "previous"
    if not prev.is_symlink():
        raise BundleError("no previous bundle")
    target = prev.resolve()
    man = verify(target, trust)  # never roll back to something that no longer verifies
    cur = station / "current"
    old = cur.resolve() if cur.is_symlink() else None
    _swap(cur, target)
    if old:
        _swap(prev, old)
    return man["bundle_id"]


def active_bundle(station: Path, trust: Path | None = None) -> dict | None:
    cur = station / "current"
    if not cur.is_symlink():
        return None
    try:
        return verify(cur.resolve(), trust or Path("deploy/trust"))
    except BundleError:
        return None
