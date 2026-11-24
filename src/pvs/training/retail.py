"""Multi-dataset retail data: licence-filtered Roboflow download + leakage-safe merge.

    python -m pvs.training.retail download --list configs/retail_datasets.yaml --out /data/rf   (needs ROBOFLOW_API_KEY)
    python -m pvs.training.retail prepare  --exports /data/rf --list configs/retail_datasets.yaml --out /data/retail

`prepare` writes the same manifest format as pvs.training.real5star (so its DDP `train` and ONNX export are
reused). Every box is class "package" for the detector; boxes whose class name maps to a chocolate brand also
carry `brand` (pvs.training.brand_map). Near-duplicates (dHash, including horizontally mirrored copies, which
Roboflow augmentation produces) are grouped across ALL projects, and a group goes to the strictest split any
member had (test > valid > train), so an image re-uploaded in another project cannot leak into training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

from .brand_map import ambiguous, brand_for
from .progress import Progress, stage_name


def _licence(export: Path) -> str:
    for name in ("README.roboflow.txt", "README.dataset.txt"):
        p = export / name
        if p.exists():
            m = re.search(r"^License:\s*(.+)$", p.read_text(errors="ignore"), re.M)
            if m:
                return m.group(1).strip()
    return "unknown"


FORK_REASONS = ("foreign_export", "no generated versions")


def _needs_fork(skip: str | None) -> bool:
    return bool(skip) and any(k in skip for k in FORK_REASONS)


def _find_licence(obj) -> str | None:
    """First licence string anywhere in a Roboflow project JSON (the key name differs between endpoints)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in ("license", "licence") and isinstance(v, str) and v.strip():
                return v.strip()
        for v in obj.values():
            r = _find_licence(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_licence(v)
            if r:
                return r
    return None


def _match_fork(project_ids, source: str) -> str | None:
    """Slug of an earlier fork of `source` in our workspace: forks are named <source>-<suffix>."""
    for pid in project_ids:
        slug = str(pid).split("/")[-1]
        if slug == source or (slug.startswith(source + "-") and len(slug) - len(source) <= 8):
            return slug
    return None


def _own_projects(rf, own_url: str) -> list[str]:
    ws = rf.workspace(own_url)
    return [p["id"] if isinstance(p, dict) else str(p) for p in ws.project_list]


def _via_fork(rf, api_key: str, own_url: str, ws: str, name: str, dest: Path, allowed, state: dict) -> dict:
    """Fork a public Universe project into our own workspace, make sure it has a version, download it.
    Downloading our own project is not a foreign export, so the new-workspace weekly export limit does not apply."""
    from roboflow.adapters import rfapi
    from roboflow.core.async_tasks import poll_until_terminal

    rec = {"via": "fork"}
    try:
        rec["licence_source"] = _find_licence(rfapi.get_project(api_key, ws, name))
    except Exception:  # noqa: BLE001 - fall back to the licence in the export
        rec["licence_source"] = None
    if rec["licence_source"] and rec["licence_source"] not in allowed:  # do not spend a fork on it
        return {**rec, "ok": False, "licence": rec["licence_source"],
                "skip": f"licence not allowed: {rec['licence_source']}"}
    slug = _match_fork(_own_projects(rf, own_url), name)
    if slug is None:
        if state.get("stop"):
            raise RuntimeError(f"forking stopped earlier: {state['stop']}")
        try:
            enq = rfapi.fork_project(api_key, own_url, url=f"https://universe.roboflow.com/{ws}/{name}")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "402" in msg or "credit" in msg.lower() or "quota" in msg.lower():
                state["stop"] = msg[:150]  # out of credits / image quota: do not hammer the API
            if "409" in msg or "already" in msg.lower():
                slug = _match_fork(_own_projects(rf, own_url), name)
            if slug is None:
                raise
        else:
            final = poll_until_terminal(api_key, own_url, enq["taskId"], polling_url=enq.get("url"), interval=15,
                                        timeout=1800)
            if final.get("status") != "completed":
                raise RuntimeError(f"fork {final.get('status')}: {final.get('error')}")
            slug = str((final.get("result") or {}).get("id", "")).split("/")[-1] or \
                _match_fork(_own_projects(rf, own_url), name)
            if not slug:
                raise RuntimeError(f"fork finished but the new project was not found: {final}")
    rec["fork"] = f"{own_url}/{slug}"
    proj = rf.workspace(own_url).project(slug)
    versions = sorted(proj.versions(), key=lambda v: int(str(v.version).split("/")[-1]))
    if versions:
        n = int(str(versions[-1].version).split("/")[-1])
    else:  # a fork starts without versions: generate one with no augmentation (prepare resizes later)
        n = proj.generate_version({"preprocessing": {"auto-orient": True}, "augmentation": {}})
        rec["generated_version"] = True
    rec["version"] = n
    if not any((dest / s).exists() for s in ("train", "valid", "test")):
        proj.version(n).download("coco", location=str(dest), overwrite=True)
    rec["licence"] = rec["licence_source"] or _licence(dest)
    rec["ok"] = rec["licence"] in allowed
    if not rec["ok"]:
        rec["skip"] = f"licence not allowed: {rec['licence']}"
    return rec


def _is_rate_limit(e: Exception) -> bool:
    s = str(e)
    return "Too Many Requests" in s or "429" in s or "rate limit" in s.lower()


def _retry_rate_limit(fn, what: str, attempts: int = 7, base_s: float = 45.0, sleep=time.sleep):
    """Run fn again after Roboflow's 'Too Many Requests', waiting longer each time (45 s .. ~10 min, with jitter).
    Safe to repeat: an existing fork is found and reused, a finished download is not fetched again."""
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if not _is_rate_limit(e) or k == attempts - 1:
                raise
            wait = min(600.0, base_s * 2 ** k) * random.uniform(0.8, 1.2)
            print(f"[download] {what}: rate limited, retrying in {wait:.0f}s ({k + 1}/{attempts - 1})", flush=True)
            sleep(wait)
    raise RuntimeError("unreachable")


def download(list_path: Path, out: Path, api_key: str, fork: bool = False, fork_workers: int = 2) -> dict:
    """Latest version of each listed project (COCO export). With fork=True, a project Roboflow will not export to
    us (new-workspace weekly limit on exports from other workspaces) or that has no generated version is forked
    into our own workspace and downloaded from there. Licences are checked the same way either way."""
    from concurrent.futures import ThreadPoolExecutor

    from roboflow import Roboflow

    cfg = yaml.safe_load(list_path.read_text())
    try:
        rf = Roboflow(api_key=api_key)
        own_url = rf.workspace().url if fork else None
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Roboflow login failed ({type(e).__name__}: {e}); check the ROBOFLOW_API_KEY secret") from e
    out.mkdir(parents=True, exist_ok=True)
    report = []
    prog = Progress(stage_name("download"), len(cfg["projects"]), "projects", every_s=0)
    for p in cfg["projects"]:
        ws, name = p["ws"], p["project"]
        dest = out / f"{ws}__{name}"
        rec = {"ws": ws, "project": name}
        try:
            proj = rf.workspace(ws).project(name)
            versions = sorted(proj.versions(), key=lambda v: int(str(v.version).split("/")[-1]))
            if not versions:
                raise RuntimeError("no generated versions")
            v = versions[-1]
            rec["version"] = int(str(v.version).split("/")[-1])
            if not (dest / "train").exists() and not (dest / "test").exists() and not (dest / "valid").exists():
                v.download("coco", location=str(dest), overwrite=True)
            rec["licence"] = _licence(dest)
            rec["ok"] = rec["licence"] in cfg["allowed_licences"]
            if not rec["ok"]:
                rec["skip"] = f"licence not allowed: {rec['licence']}"
        except Exception as e:  # noqa: BLE001 - one bad project must not stop the run
            rec.update(ok=False, skip=f"{type(e).__name__}: {e}"[:200])
        print(json.dumps(rec), flush=True)
        report.append(rec)
        prog.update(ok=sum(r.get("ok", False) for r in report), skipped=sum(not r.get("ok", False) for r in report))
    todo = [r for r in report if fork and not r.get("ok") and _needs_fork(r.get("skip"))]
    if todo:
        state: dict = {}
        fprog = Progress(stage_name("download") + ":fork", len(todo), "projects", every_s=0)

        def work(r):
            dest = out / f"{r['ws']}__{r['project']}"
            try:
                new = _retry_rate_limit(lambda: _via_fork(rf, api_key, own_url, r["ws"], r["project"], dest,
                                                          cfg["allowed_licences"], state), f"{r['ws']}/{r['project']}")
                return {**r, **new, "direct_skip": r.get("skip"), **({} if not new["ok"] else {"skip": None})}
            except Exception as e:  # noqa: BLE001
                return {**r, "via": "fork", "ok": False, "direct_skip": r.get("skip"),
                        "skip": f"fork: {type(e).__name__}: {e}"[:200]}

        by_key = {(r["ws"], r["project"]): k for k, r in enumerate(report)}
        with ThreadPoolExecutor(fork_workers) as ex:
            for new in ex.map(work, todo):
                new = {k: v for k, v in new.items() if v is not None}
                report[by_key[(new["ws"], new["project"])]] = new
                print(json.dumps(new), flush=True)
                fprog.update(ok=sum(1 for r in report if r.get("via") == "fork" and r.get("ok")))
    (out / "download_report.json").write_text(json.dumps(report, indent=1))
    return {"ok": sum(r["ok"] for r in report), "via_fork": sum(1 for r in report if r.get("via") == "fork" and r["ok"]),
            "skipped": [r for r in report if not r["ok"]]}


def _dhash_both(path: Path) -> tuple[int, int] | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    out = []
    for im in (img, img[:, ::-1]):
        s = cv2.resize(im, (17, 16), interpolation=cv2.INTER_AREA)
        bits = (s[:, 1:] > s[:, :-1]).flatten()
        out.append(int("".join("1" if b else "0" for b in bits), 2))
    return out[0], out[1]


def prepare(exports_root: Path, list_path: Path, out: Path, ham: int = 6, seed: int = 0, max_side: int = 1280) -> dict:
    cfg = yaml.safe_load(list_path.read_text())
    report = {f"{r['ws']}__{r['project']}": r for r in json.loads((exports_root / "download_report.json").read_text())}
    caps = {f"{p['ws']}__{p['project']}": p.get("max_images") for p in cfg["projects"]}
    rng = random.Random(seed)
    items: list[dict] = []
    for key, rec in sorted(report.items()):
        if not rec.get("ok"):
            continue
        ex = exports_root / key
        mine = []
        for split in ("train", "valid", "test"):
            ann = ex / split / "_annotations.coco.json"
            if not ann.exists():
                continue
            coco = json.loads(ann.read_text())
            cats = {c["id"]: c["name"] for c in coco["categories"]}
            by_img = defaultdict(list)
            for a in coco["annotations"]:
                x, y, w, h = a["bbox"]
                if w < 4 or h < 4:
                    continue
                name = cats[a["category_id"]]
                by_img[a["image_id"]].append({"box": [x, y, x + w, y + h], "class": name, "brand": brand_for(name),
                                              "ambiguous": ambiguous(name)})
            for im in coco["images"]:
                p = ex / split / im["file_name"]
                if p.exists() and by_img.get(im["id"]):  # images with no boxes say nothing about products
                    mine.append({"src": str(p), "export": key, "orig_split": split, "w": im["width"],
                                 "h": im["height"], "boxes": by_img[im["id"]], "licence": rec["licence"]})
        cap = caps.get(key)
        if cap and len(mine) > cap:  # keep every image with a brand box, sample the rest
            branded = [m for m in mine if any(b["brand"] for b in m["boxes"])]
            rest = [m for m in mine if not any(b["brand"] for b in m["boxes"])]
            rng.shuffle(rest)
            mine = branded + rest[:max(0, cap - len(branded))]
        items += mine
    # near-duplicate grouping incl. mirrored copies (union-find over sorted dHash neighbours)
    keep, hashes = [], []
    prog = Progress(stage_name("prepare") + ":hash", len(items), "images")
    for it in items:
        prog.update()
        h = _dhash_both(Path(it["src"]))
        if h is not None:
            keep.append(it)
            hashes.append(h)
    items = keep
    parent = list(range(len(items)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    keyed = sorted([(h[0], i) for i, h in enumerate(hashes)] + [(h[1], i) for i, h in enumerate(hashes)])
    for (ha, a), (hb, b) in zip(keyed, keyed[1:]):
        if a != b and bin(ha ^ hb).count("1") <= ham:
            parent[find(a)] = find(b)
    exact = defaultdict(list)  # identical files across projects
    for i, it in enumerate(items):
        exact[hashlib.sha256(Path(it["src"]).read_bytes()).hexdigest()].append(i)
    for idx in exact.values():
        for a in idx[1:]:
            parent[find(a)] = find(idx[0])
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)
    rank = {"test": 2, "valid": 1, "train": 0}
    for g in groups.values():
        s = max((items[i]["orig_split"] for i in g), key=rank.get)
        for i in g:
            items[i]["split"] = {"valid": "val"}.get(s, s)
            items[i]["group"] = f"g{find(i)}"
    split_groups = defaultdict(set)
    for it in items:
        split_groups[it["split"]].add(it["group"])
    leaks = [f"{a}/{b}" for a in split_groups for b in split_groups if a < b and split_groups[a] & split_groups[b]]
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    prog = Progress(stage_name("prepare") + ":write", len(items), "images")
    for it in items:
        prog.update()
        src = Path(it.pop("src"))
        h = hashlib.sha256(src.read_bytes()).hexdigest()
        img = cv2.imread(str(src))
        s = min(1.0, max_side / max(img.shape[:2]))
        if s < 1:  # store at <= max_side so decoding never starves the GPUs; boxes scaled to match
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            for b in it["boxes"]:
                b["box"] = [v * s for v in b["box"]]
        it["w"], it["h"] = img.shape[1], img.shape[0]
        dst = img_dir / f"{h[:20]}.jpg"
        if not dst.exists():
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        it["image"] = f"images/{dst.name}"
        it["sha256"] = h
    brand_boxes = {s: dict(Counter(b["brand"] for i in items if i["split"] == s for b in i["boxes"] if b["brand"]))
                   for s in ("train", "val", "test")}
    stats = {
        "projects_used": sorted({i["export"] for i in items}), "images": len(items), "groups": len(groups),
        "images_in_duplicate_groups": sum(len(g) for g in groups.values() if len(g) > 1), "leakage": leaks,
        "split_images": dict(Counter(i["split"] for i in items)),
        "boxes_by_split": {s: sum(len(i["boxes"]) for i in items if i["split"] == s) for s in ("train", "val", "test")},
        "brand_boxes_by_split": brand_boxes,
        "brand_class_names": {b: sorted({x["class"] for i in items for x in i["boxes"] if x["brand"] == b})
                              for b in sorted({x["brand"] for i in items for x in i["boxes"] if x["brand"]})},
        "unmapped_chocolate_like": sorted({x["class"] for i in items for x in i["boxes"] if not x["brand"]
                                           and re.search(r"choc|cadbury|nestle|candy|bar\b", x["class"], re.I)}),
        "ambiguous_class_names": sorted({x["class"] for i in items for x in i["boxes"] if x.get("ambiguous")}),
        "licences": dict(Counter(i["licence"] for i in items)),
        "images_by_project": dict(Counter(i["export"] for i in items)),
    }
    # real5star-compatible fields: `sku` unused here (brand lives in `brand`)
    for it in items:
        for b in it["boxes"]:
            b.setdefault("sku", None)
    (out / "manifest.json").write_text(json.dumps({"stats": stats, "items": items}))
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    if leaks:
        raise SystemExit(f"leakage between splits: {leaks}")
    return stats


def export(data: Path, run: Path, out: Path, size: int, device: str, test_limit: int | None) -> dict:
    """Best checkpoint -> ONNX (parity-checked) + detection recall/precision on the held-out TEST split."""
    from transformers import RTDetrForObjectDetection

    from ..evaluation import clopper_pearson
    from .real5star import detection_recall, export_onnx

    items = json.loads((data / "manifest.json").read_text())["items"]
    test = [i for i in items if i["split"] == "test" and i["boxes"]]
    random.Random(2).shuffle(test)  # a limit must sample every project, not the first ones in the manifest
    model = RTDetrForObjectDetection.from_pretrained(run / "best").to(device).eval()
    t = time.time()
    test = test[:test_limit] if test_limit else test
    prog = Progress(stage_name("export") + ":test", len(test), "images")
    rec = {"tp": 0, "gt": 0, "pred": 0}
    for i in range(0, len(test), 50):
        r = detection_recall(model, data, test[i:i + 50], size, device)
        for k in rec:
            rec[k] += r[k]
        prog.update(len(test[i:i + 50]), recall=rec["tp"] / max(1, rec["gt"]))
    rec["recall"] = rec["tp"] / rec["gt"] if rec["gt"] else 0.0
    rec["precision"] = rec["tp"] / rec["pred"] if rec["pred"] else 0.0
    per_img_ms = (time.time() - t) * 1000 / max(1, len(test))
    out.mkdir(parents=True, exist_ok=True)
    info = export_onnx(model, out / "rtdetr_r18_package.onnx", size)
    rep = {"test_images": len(test), **rec,
           "recall_ci95": clopper_pearson(rec["tp"], rec["gt"]) if rec.get("gt") else None,
           "ms_per_image_incl_decode": round(per_img_ms, 1), "onnx": info,
           "train_report": json.loads((run / "train_report.json").read_text()) if (run / "train_report.json").exists() else None}
    (out / "detector_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pvs.training.retail")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--list", default="configs/retail_datasets.yaml")
    d.add_argument("--out", required=True)
    d.add_argument("--fork", action="store_true", help="fork projects Roboflow will not export to us, then download")
    d.add_argument("--fork-workers", type=int, default=2)
    p = sub.add_parser("prepare")
    p.add_argument("--exports", required=True)
    p.add_argument("--list", default="configs/retail_datasets.yaml")
    p.add_argument("--out", required=True)
    x = sub.add_parser("export")
    x.add_argument("--data", required=True)
    x.add_argument("--run", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--size", type=int, default=640)
    x.add_argument("--device", default="cuda")
    x.add_argument("--test-limit", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "export":
        r = export(Path(a.data), Path(a.run), Path(a.out), a.size, a.device, a.test_limit)
        print(json.dumps({k: v for k, v in r.items() if k != "train_report"}, indent=1))
        return 0
    if a.cmd == "download":
        key = os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            print("ROBOFLOW_API_KEY not set", file=sys.stderr)
            return 2
        print(json.dumps(download(Path(a.list), Path(a.out), key, a.fork, a.fork_workers), indent=1))
    else:
        s = prepare(Path(a.exports), Path(a.list), Path(a.out))
        print(json.dumps({k: s[k] for k in ("images", "split_images", "boxes_by_split", "brand_boxes_by_split",
                                             "leakage")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
