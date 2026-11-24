"""Publish a long run to a (private) Hugging Face model repo: live progress page while it runs, results at the end.

    python -m pvs.training.hub init  --repo USER/pack-station-training
    python -m pvs.training.hub watch --repo ... --status-dir DIR --every 240     (background; stops on DIR/.stop)
    python -m pvs.training.hub final --repo ... --status-dir DIR --out OUT --log LOG

Needs HF_TOKEN (a write token) in the environment; without it every command prints a note and exits 0, so the
training itself never depends on the upload. Network errors are logged and retried on the next cycle.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from .progress import IST, fmt_dur

ARTIFACTS = ["retail-v2-artifacts.tgz", "data_stats.json", "download_report.json", "bundle_sha256.txt",
             "detector/detector_report.json", "detector/train_report.json", "detector/rtdetr_r18_package.json",
             "brands/brands_report.json", "brands/e2e_report.json", "brands/thresholds_brands.yaml",
             "brands/catalogue_brands.json",
             # run 3 (brands3)
             "brands-v3-artifacts.tgz", "off_scan_stats.json", "off_download_stats.json", "crop_stats.json",
             "ft/train_report.json", "external_v2.json", "external_v3.json", "decision.json",
             # laptop model (YOLO26-seg + OCR)
             "yolo-edge-artifacts.tgz", "edge_bundle.tgz", "yolo/dataset_report.json", "yolo/train_report.json",
             "yolo/results.csv", "yolo/yolo_val_test.json", "yolo/yolo_report.json", "yolo/bench_t4_gpu.json",
             "yolo/bench_cpu_4gb.json", "e2e_v2/e2e_report.json", "logs/masks_0.log", "logs/masks_1.log",
             "logs/off_scan.log"]


def _api():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("[hub] HF_TOKEN not set: live page and auto-upload disabled (results stay in the run's output)", flush=True)
        return None
    from huggingface_hub import HfApi

    return HfApi(token=tok)


def _ist(iso: str | None) -> str:
    if not iso:
        return "?"
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return (dt.astimezone(UTC) + IST).strftime("%d %b %H:%M IST")
    except ValueError:
        return iso


def render(st: dict, final: dict | None = None) -> str:
    state = st.get("final") or "running"
    pct = st.get("pct")
    bar = ""
    if isinstance(pct, (int, float)):
        n = int(pct / 5)
        bar = f"`{'█' * n}{'░' * (20 - n)}` {pct:.1f}%"
    lines = [
        "# Pack Station training run",
        "",
        f"**State:** {state}  ·  **updated:** {_ist(st.get('updated_at'))}  ·  **started:** {_ist(st.get('started_at'))}",
        "",
        f"**Now:** `{st.get('line', '')}`",
        "",
    ]
    if bar and state == "running":
        lines += [f"**Current stage:** {st.get('stage')} {bar}", ""]
    if state == "running" and st.get("run_eta_s") is not None:
        lines += [f"**Estimated finish:** {_ist(st.get('run_eta_at'))} (in {fmt_dur(st['run_eta_s'])}; "
                  "estimate, refined as the run goes)", ""]
    done = st.get("stages_done", {})
    if done:
        lines += ["| Stage | Took |", "|---|---|"] + [f"| {k} | {fmt_dur(v)} |" for k, v in done.items()] + [""]
    eps = st.get("epochs") or []
    if eps:
        lines += ["**Detector training (validation sample)**", "",
                  "| Epoch | Loss | Val recall@0.5 | Val precision@0.5 | Elapsed |", "|---|---|---|---|---|"]
        lines += [f"| {e['epoch']} | {e['loss']:.3f} | {e['val_recall@0.5']:.3f} | {e['val_precision@0.5']:.3f} | "
                  f"{fmt_dur(e['elapsed_s'])} |" for e in eps[-25:]] + [""]
    if final:
        lines += ["## Results", ""] + final.get("summary", []) + ["", "## Files", ""] + final.get("files", [])
    if st.get("log_tail"):
        lines += ["", "## Last log lines", "", "```", *st["log_tail"], "```"]
    lines += ["", "Source: https://github.com/Rohitpro1/product-vision-station (`scripts/gpu_train_retail.sh`)."]
    return "\n".join(lines) + "\n"


def _push(api, repo: str, files: dict[str, bytes | str], msg: str) -> None:
    from huggingface_hub import CommitOperationAdd

    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=(v.encode() if isinstance(v, str) else v))
           for k, v in files.items()]
    api.create_commit(repo_id=repo, operations=ops, commit_message=msg)


def resolve(api, repo: str) -> str:
    """A bare name ("pack-station-training") becomes "<token owner>/<name>"."""
    return repo if "/" in repo else f"{api.whoami()['name']}/{repo}"


def init(repo: str) -> int:
    api = _api()
    if not api:
        return 0
    try:
        repo = resolve(api, repo)
        url = api.create_repo(repo_id=repo, private=True, exist_ok=True, repo_type="model")
        print(f"[hub] live page + results: {url}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[hub] could not create {repo}: {type(e).__name__}: {e}", flush=True)
    return 0


def watch(repo: str, status_dir: Path, every: float) -> int:
    api = _api()
    if not api:
        return 0
    repo = resolve(api, repo)
    last = None
    while not (status_dir / ".stop").exists():
        p = status_dir / "status.json"
        if p.exists():
            raw = p.read_text()
            if raw != last:
                try:
                    st = json.loads(raw)
                    _push(api, repo, {"README.md": render(st), "status.json": raw}, f"progress: {st.get('line', '')[:120]}")
                    last = raw
                except Exception as e:  # noqa: BLE001 - never let the publisher break the run
                    print(f"[hub] progress push failed (will retry): {type(e).__name__}: {e}", flush=True)
        for _ in range(int(every)):
            if (status_dir / ".stop").exists():
                break
            time.sleep(1)
    return 0


def _summary(out: Path) -> list[str]:
    s = []

    def j(rel):
        p = out / rel
        return json.loads(p.read_text()) if p.exists() else None

    d = j("detector/detector_report.json")
    if d:
        s.append(f"- **Package detector (held-out test, {d['test_images']} photos):** recall {d['recall']:.3f}, "
                 f"precision {d['precision']:.3f} at IoU 0.5")
    b = j("brands/brands_report.json")
    if b:
        t = b["test_crop_level"]
        s.append(f"- **Brands kept:** {', '.join(b['display_names'].values())}")
        s.append(f"- **Brand naming (test crops):** correct {t['correct_brand']['k']}/{t['correct_brand']['n']}, "
                 f"wrong brand {t['wrong_brand']['k']}, other products called a brand "
                 f"{t['other_products_called_a_brand']['k']}/{t['other_products_called_a_brand']['n']}")
        for k, v in t["per_brand"].items():
            r, p = v["recall"], v["precision"]
            s.append(f"  - {b['display_names'][k]}: recall {r['k']}/{r['n']}, precision {p['k']}/{p['n']}")
    if b and b.get("v2_live_on_same_test"):
        v2 = b["v2_live_on_same_test"]["test"]["all"]
        s.append(f"- **Model chosen on validation:** {b['chosen_model']}; live v2 on the same test crops: correct "
                 f"{v2['correct_brand']['k']}/{v2['correct_brand']['n']}, wrong {v2['wrong_brand']['k']}")
        for src, m in t.get("by_source", {}).items():
            s.append(f"  - {src}: correct {m['correct_brand']['k']}/{m['correct_brand']['n']}, wrong "
                     f"{m['wrong_brand']['k']}, other products called a brand "
                     f"{m['other_products_called_a_brand']['k']}/{m['other_products_called_a_brand']['n']}")
    for name in ("external_v2.json", "external_v3.json"):
        x = j(name)
        if x:
            s.append(f"- **External photos ({name[9:11]}):** named correctly {x['named_correctly']}/"
                     f"{x['with_true_brand']}, photos with a wrong name {x['photos_with_a_wrong_name']}")
    y = j("yolo/yolo_report.json")
    if y:
        for name, m in y.get("models", {}).items():
            tt = m["test"]
            s.append(f"- **YOLO {name} ({'GPU' if m['provider'].startswith('CUDA') else 'CPU'}):** pack recall {tt['pack_detection_recall_iou50']['rate']:.3f}, "
                     f"brand correct {tt['correct_brand']['k']}/{tt['brand_boxes']}, wrong {tt['wrong_brand']['k']}, "
                     f"other packs called a brand {tt['other_packs_called_a_brand']['k']}, mask IoU "
                     f"{tt['mask_iou_mean_vs_sam2_labels']}, {m['latency_ms_incl_mask_decode']['p50']} ms")
        o = y.get("ocr_brand_check")
        if o:
            s.append(f"- **OCR on brand packs:** text named the right brand {o['text_brand_right']['k']}/{o['brand_packs']}, "
                     f"wrong {o['text_brand_wrong']['k']}, {o['ocr_ms_per_pack']['p50']} ms per pack")
    for name in ("yolo/bench_t4_gpu.json", "yolo/bench_cpu_4gb.json"):
        bb = j(name)
        if bb:
            s.append(f"- **Speed {name[11:-5]}:** detect+segment p50 {bb['seg_ms']['p50']} ms, p95 {bb['seg_ms']['p95']} ms; "
                     f"OCR p50 {bb['ocr_ms_per_pack']['p50']} ms per pack; peak memory {bb['peak_rss_mb']} MB")
    yv = j("yolo/yolo_val_test.json")
    if yv:
        s.append(f"- **Test mAP50-95:** box {yv['box']['map50_95']:.3f}, mask {yv['mask']['map50_95']:.3f}")
    dec = j("decision.json")
    if dec:
        s.append(f"- **Deploy decision:** {dec['verdict']} ({dec['reason']})")
    e = j("brands/e2e_report.json")
    if e:
        s.append(f"- **End to end ({e['images']} test photos, full pipeline):** detection recall "
                 f"{e['detection_recall_iou50']['rate']:.3f}, brand correct "
                 f"{e['brand_correct_end_to_end']['k']}/{e['brand_correct_end_to_end']['n']}, other products called "
                 f"a brand {e['other_products_called_a_brand']['k']}/{e['other_products_called_a_brand']['n']}")
    return s or ["- No reports were produced (see the log tail below)."]


def final(repo: str, status_dir: Path, out: Path, log: Path | None) -> int:
    (status_dir / ".stop").touch()
    api = _api()
    if not api:
        return 0
    repo = resolve(api, repo)
    st = json.loads((status_dir / "status.json").read_text()) if (status_dir / "status.json").exists() else {}
    if log and log.exists():
        st["log_tail"] = log.read_text(errors="ignore").splitlines()[-40:]
    files = {}
    for rel in ARTIFACTS:
        p = out / rel
        if p.exists():
            files[f"results/{rel}"] = p.read_bytes()
    listing = [f"- `results/{k[len('results/'):]}`" for k in files] or ["- (none)"]
    bundle = out / "bundle"
    edge = (bundle / "edge.json").exists()
    if bundle.exists():
        listing.append("- `bundle/`: laptop station (YOLO26-seg + OCR); see docs/EDGE.md" if edge else
                       "- `bundle/`: models + config for the hosted API (load with `configs/station.hosted.yaml`)")
    try:
        _push(api, repo, {"README.md": render(st, {"summary": _summary(out), "files": listing}),
                          "status.json": json.dumps(st, indent=1), **files},
              f"run {st.get('final', 'finished')}: reports")
        if bundle.exists():
            # a laptop bundle replaces the folder (older revisions, incl. the pinned v2 release, stay intact)
            api.upload_folder(repo_id=repo, folder_path=str(bundle), path_in_repo="bundle",
                              delete_patterns=["*"] if edge else None,
                              commit_message="bundle: laptop station" if edge else "bundle: models + hosted config")
        print(f"[hub] uploaded results to https://huggingface.co/{repo}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[hub] final upload failed: {type(e).__name__}: {e} (files remain in the run output)", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pvs.training.hub")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("init", "watch", "final"):
        p = sub.add_parser(name)
        p.add_argument("--repo", required=True)
        if name != "init":
            p.add_argument("--status-dir", required=True)
        if name == "watch":
            p.add_argument("--every", type=float, default=240)
        if name == "final":
            p.add_argument("--out", required=True)
            p.add_argument("--log", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "init":
        return init(a.repo)
    if a.cmd == "watch":
        return watch(a.repo, Path(a.status_dir), a.every)
    return final(a.repo, Path(a.status_dir), Path(a.out), Path(a.log) if a.log else None)


if __name__ == "__main__":
    raise SystemExit(main())
