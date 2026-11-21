"""Versioned product catalogue: create, enrol, validate, collision test and rollback."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .barcode import to_gtin14, validate_gtin

REQUIRED = ["sku", "family", "artwork_revision", "references", "physical_ends", "supported_faces", "provenance",
            "usage_rights", "approval_state"]
APPROVAL = {"draft", "approved", "fixture", "retired"}


class CatalogueError(ValueError):
    pass


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def create(path: Path, name: str) -> dict:
    cat = {"catalogue": name, "revision": f"{name}-r1", "created_at": _now(), "products": [], "negatives": [],
           "history": []}
    _write(path, cat)
    return cat


def validate(cat: dict, root: Path | None = None) -> list[str]:
    """Return a list of problems (empty = valid)."""
    errs: list[str] = []
    seen_sku: set[str] = set()
    gtin_owner: dict[str, str] = {}
    for p in cat.get("products", []):
        for k in REQUIRED:
            if k not in p:
                errs.append(f"{p.get('sku', '?')}: missing {k}")
        sku = p.get("sku")
        if sku in seen_sku:
            errs.append(f"duplicate sku {sku}")
        seen_sku.add(sku)
        if p.get("approval_state") not in APPROVAL:
            errs.append(f"{sku}: bad approval_state {p.get('approval_state')}")
        if sku and not sku.startswith("fixture:") and p.get("approval_state") == "fixture":
            errs.append(f"{sku}: fixture approval only allowed for fixture: namespace")
        for g in p.get("gtins", []):
            ok, why = validate_gtin(g)
            if not ok:
                errs.append(f"{sku}: GTIN {g} invalid ({why})")
                continue
            g14 = to_gtin14(g)
            if g14 in gtin_owner and gtin_owner[g14] != sku:
                errs.append(f"GTIN {g} mapped to both {gtin_owner[g14]} and {sku}")
            gtin_owner[g14] = sku
        ends = p.get("physical_ends", {})
        if ends and not {"A", "B", "definition"} <= set(ends):
            errs.append(f"{sku}: physical_ends must define A, B, definition")
        for r in p.get("references", []):
            if r.get("side") not in ("front", "back", "edge"):
                errs.append(f"{sku}: reference side {r.get('side')}")
            if root is not None:
                fp = root / r["path"]
                if not fp.exists():
                    errs.append(f"{sku}: missing reference {r['path']}")
                elif r.get("sha256") and hashlib.sha256(fp.read_bytes()).hexdigest() != r["sha256"]:
                    errs.append(f"{sku}: reference hash mismatch {r['path']}")
    return errs


def gtin_index(cat: dict) -> dict[str, dict]:
    idx = {}
    for p in cat.get("products", []):
        if p.get("approval_state") in ("approved", "fixture"):
            for g in p.get("gtins", []):
                idx[to_gtin14(g)] = p
    return idx


def enrol(path: Path, product: dict, root: Path) -> dict:
    """Add or replace a product revision; snapshot the previous catalogue for rollback."""
    cat = load(path)
    trial = copy.deepcopy(cat)
    trial["products"] = [p for p in trial["products"] if p["sku"] != product["sku"]] + [product]
    errs = validate(trial, root)
    if errs:
        raise CatalogueError("; ".join(errs))
    _snapshot(path, cat)
    n = int(cat["revision"].rsplit("-r", 1)[1]) + 1
    trial["revision"] = f"{cat['catalogue']}-r{n}"
    trial["history"] = cat.get("history", []) + [{"revision": trial["revision"], "at": _now(),
                                                  "change": f"enrol {product['sku']} {product['artwork_revision']}"}]
    _write(path, trial)
    return trial


def rollback(path: Path) -> dict:
    snaps = sorted((path.parent / ".history").glob("*.json"))
    if not snaps:
        raise CatalogueError("no snapshot to roll back to")
    prev = load(snaps[-1])
    _write(path, prev)
    snaps[-1].unlink()
    return prev


def collision_test(gallery, product_sku: str, threshold: float) -> list[dict]:
    """Cross-SKU reference similarity above ``threshold`` (should be reviewed before approval)."""
    import numpy as np

    out = []
    V, R = gallery.vectors, gallery.refs
    for i, r in enumerate(R):
        if r.sku != product_sku:
            continue
        sims = V @ V[i]
        for j, s in enumerate(sims.tolist()):
            if R[j].sku != product_sku and s >= threshold:
                out.append({"ref": f"{r.sku}/{r.side}/{r.view}", "other": f"{R[j].sku}/{R[j].side}/{R[j].view}",
                            "cosine": round(float(s), 4)})
    _ = np
    return out


def _snapshot(path: Path, cat: dict) -> None:
    d = path.parent / ".history"
    d.mkdir(exist_ok=True)
    (d / f"{cat['revision']}.json").write_text(json.dumps(cat, indent=2))


def _write(path: Path, cat: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cat, indent=2) + "\n")
    shutil.move(tmp, path)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
