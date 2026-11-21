"""Evaluation harness: one-to-one matching, raw counts, exact intervals, explicit insufficient evidence."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta

from .geometry import periodic_error

GATES = {
    "localisation_recall": (">=", 0.97),
    "exact_sku_precision": (">=", 0.99),
    "exact_sku_recall": (">=", 0.95),
    "exact_sku_recall_identity_observable": (">=", 0.95),
    "false_target_acceptance": ("<=", 0.01),
    "axis_coverage": (">=", 0.95),
    "axis_p95_err_deg": ("<=", 10.0),
    "heading_coverage": (">=", 0.90),
    "heading_p95_err_deg": ("<=", 15.0),
    "heading_reversal_rate": ("<", 0.01),
}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float] | None:
    if n == 0:
        return None
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def one_sided_upper(k: int, n: int, alpha: float = 0.05) -> float | None:
    """One-sided (1-alpha) upper bound; k=0, n=300 -> ~0.00994 (PDF p.8)."""
    if n == 0:
        return None
    return 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))


def rate(k: int, n: int, name: str) -> dict:
    if n == 0:
        return {"metric": name, "status": "insufficient_evidence", "k": 0, "n": 0, "value": None, "ci95": None}
    ci = clopper_pearson(k, n)
    return {"metric": name, "status": "measured", "k": k, "n": n, "value": k / n, "ci95": ci}


def p95(errs: list[float], name: str) -> dict:
    if not errs:
        return {"metric": name, "status": "insufficient_evidence", "n": 0, "value": None}
    return {"metric": name, "status": "measured", "n": len(errs), "value": float(np.percentile(errs, 95)),
            "max": float(max(errs)), "mean": float(np.mean(errs))}


def iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(gt_boxes, pred_boxes, thr: float = 0.5) -> list[tuple[int, int]]:
    if not gt_boxes or not pred_boxes:
        return []
    cost = np.array([[1 - iou(g, p) for p in pred_boxes] for g in gt_boxes])
    r, c = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(r, c) if 1 - cost[i, j] >= thr]


def score_results(pairs: list[tuple[dict, dict]], target_skus: set[str] | None = None) -> dict:
    """``pairs``: (scene truth, FrameResult dict). Every presentation is one independent trial."""
    c = dict(gt=0, matched=0, sku_accepted=0, sku_correct=0, target_presentations=0, target_correct=0,
             neg_presentations=0, neg_false_accept=0, axis_obs=0, axis_ret=0, head_obs=0, head_ret=0,
             reversals=0, empty_scenes=0, empty_false_pos=0,
             target_id_observable=0, target_id_observable_correct=0)
    neg_slices: dict[str, list[int]] = {}
    axis_err, head_err = [], []
    failures = []
    for truth, res in pairs:
        gts = truth["instances"]
        preds = res.get("instances", [])
        if not gts:
            c["empty_scenes"] += 1
            c["empty_false_pos"] += int(bool(preds))
        m = match([g["box_xyxy"] for g in gts], [p["bbox_xyxy_px"] for p in preds])
        mg = {i: j for i, j in m}
        for gi, g in enumerate(gts):
            in_env = g["occluded_frac"] <= 0.4
            if in_env:
                c["gt"] += 1
                c["matched"] += int(gi in mg)
            p = preds[mg[gi]] if gi in mg else None
            ident = p["identity"] if p else None
            is_target = g["in_catalogue"] and (target_skus is None or g["sku"] in target_skus)
            accepted = ident is not None and ident["level"] == "sku"
            if accepted:
                c["sku_accepted"] += 1
                c["sku_correct"] += int(ident["sku"] == g["sku"])
            if is_target and in_env:
                c["target_presentations"] += 1
                c["target_correct"] += int(accepted and ident["sku"] == g["sku"])
                if g["face"] in ("front", "back"):
                    c["target_id_observable"] += 1
                    c["target_id_observable_correct"] += int(accepted and ident["sku"] == g["sku"])
            if not is_target:
                c["neg_presentations"] += 1
                fa = int(accepted)
                c["neg_false_accept"] += fa
                sl = "similar_gold_wrapper" if g["sku"] == "unknown" else "same_family_variant"
                neg_slices.setdefault(sl, [0, 0])
                neg_slices[sl][0] += fa
                neg_slices[sl][1] += 1
            if g["axis_observable"]:
                c["axis_obs"] += 1
                if p and p["pose"]["axis"]["available"]:
                    c["axis_ret"] += 1
                    axis_err.append(periodic_error(p["pose"]["axis"]["value_deg"], g["axis_deg"], 180.0))
            if g["heading_observable"]:
                c["head_obs"] += 1
                if p and p["pose"]["heading"]["available"]:
                    c["head_ret"] += 1
                    e = periodic_error(p["pose"]["heading"]["value_deg"], g["heading_deg"], 360.0)
                    head_err.append(e)
                    c["reversals"] += int(e > 150)
            if p is None or (is_target and not (accepted and ident["sku"] == g["sku"])) or (not is_target and accepted):
                failures.append({"scene": truth["scene_id"], "truth_sku": g["sku"], "face": g["face"],
                                 "pred": None if p is None else {"level": ident["level"], "sku": ident["sku"],
                                                                 "rule": ident["rule"], "reason": ident["reason"]}})
    metrics = {
        "localisation_recall": rate(c["matched"], c["gt"], "localisation_recall"),
        "exact_sku_precision": rate(c["sku_correct"], c["sku_accepted"], "exact_sku_precision"),
        "exact_sku_recall": rate(c["target_correct"], c["target_presentations"], "exact_sku_recall"),
        "exact_sku_recall_identity_observable": rate(c["target_id_observable_correct"], c["target_id_observable"],
                                                     "exact_sku_recall_identity_observable"),
        "false_target_acceptance": rate(c["neg_false_accept"], c["neg_presentations"], "false_target_acceptance"),
        "axis_coverage": rate(c["axis_ret"], c["axis_obs"], "axis_coverage"),
        "axis_p95_err_deg": p95(axis_err, "axis_p95_err_deg"),
        "heading_coverage": rate(c["head_ret"], c["head_obs"], "heading_coverage"),
        "heading_p95_err_deg": p95(head_err, "heading_p95_err_deg"),
        "heading_reversal_rate": rate(c["reversals"], c["head_ret"], "heading_reversal_rate"),
    }
    fta = metrics["false_target_acceptance"]
    fta["one_sided_upper95"] = one_sided_upper(c["neg_false_accept"], c["neg_presentations"])
    for k, (op, gate) in GATES.items():
        v = metrics[k].get("value")
        if k == "false_target_acceptance" and v is not None:
            ub = fta["one_sided_upper95"]
            metrics[k]["gate"] = (f"pass (upper95={ub:.4f} <= {gate})" if ub <= gate else
                                  f"insufficient_evidence (point={v:.4f}, upper95={ub:.4f} > {gate}; need >=299 clean independent negatives)")
            continue
        if v is None:
            metrics[k]["gate"] = "insufficient_evidence"
        else:
            ok = v >= gate if op == ">=" else (v <= gate if op == "<=" else v < gate)
            metrics[k]["gate"] = f"{'pass' if ok else 'fail'} ({op}{gate})"
    return {"counts": c, "metrics": metrics,
            "negative_slices": {k: {"false_accept": v[0], "n": v[1]} for k, v in neg_slices.items()},
            "failures": failures[:200]}


def env_info() -> dict:
    def _git() -> str | None:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:  # noqa: BLE001
            return None

    import onnxruntime

    info = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": __import__("os").cpu_count(), "implementation_commit": _git(),
            "onnxruntime": onnxruntime.__version__}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass
    return info


def evaluate_fixture_set(pipeline, gt_path: Path, out: Path, limit: int | None = None) -> dict:
    from .capture import frame_from_file

    gt = json.loads(gt_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    pairs, times = [], []
    with open(out / "results.jsonl", "w") as fh:
        for sc in gt["scenes"][:limit]:
            f = frame_from_file(gt_path.parent / sc["image"], camera="eval")
            t0 = time.perf_counter()
            r = pipeline.process(f)
            times.append((time.perf_counter() - t0) * 1000)
            d = json.loads(r.model_dump_json())
            fh.write(json.dumps(d) + "\n")
            pairs.append((sc, d))
    rep = score_results(pairs)
    rep["run"] = {"profile": pipeline.profile.value, "backend": pipeline.backend, "thresholds": pipeline.thr.version,
                  "catalogue": pipeline.catalogue["revision"], "ground_truth": str(gt_path),
                  "ground_truth_sha256": hashlib.sha256(gt_path.read_bytes()).hexdigest(),
                  "input_sha256": [s["sha256"] for s in gt["scenes"][:limit]][:3] + ["..."], "n_scenes": len(pairs),
                  "timing_method": "time.perf_counter around Pipeline.process (processing-only; no capture delay)",
                  "processing_ms": p95(times, "processing_ms") | {"median": float(np.median(times)) if times else None},
                  "env": env_info(),
                  "caveat": ("fixture profile uses scripted detections; only identity/orientation/barcode/fusion "
                             "algorithms are exercised" if pipeline.profile.value == "fixture" else
                             "synthetic scenes; not real 5 Star presentations")}
    rep["summary"] = {k: {kk: v.get(kk) for kk in ("k", "n", "value", "gate")} for k, v in rep["metrics"].items()}
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    (out / "report.md").write_text(to_markdown(rep))
    return rep


def to_markdown(rep: dict) -> str:
    lines = [f"# Evaluation report — profile `{rep['run']['profile']}`", "",
             f"- Implementation commit: `{rep['run']['env'].get('implementation_commit')}`",
             f"- Ground truth: `{rep['run']['ground_truth']}` sha256 `{rep['run']['ground_truth_sha256'][:16]}…`",
             f"- Scenes: {rep['run']['n_scenes']} · thresholds `{rep['run']['thresholds']}` · catalogue `{rep['run']['catalogue']}`",
             f"- Backends: `{json.dumps(rep['run']['backend'])}`",
             f"- Caveat: **{rep['run']['caveat']}**", "",
             "| Metric | k / n | Value | 95% CI | Gate |", "|---|---|---|---|---|"]
    for k, m in rep["metrics"].items():
        val = "—" if m.get("value") is None else (f"{m['value']:.4f}" if "deg" not in k else f"{m['value']:.2f}°")
        kn = f"{m.get('k', '')}/{m.get('n', '')}" if "k" in m else f"n={m.get('n')}"
        ci = m.get("ci95")
        ci_s = "—" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(f"| {k} | {kn} | {val} | {ci_s} | {m.get('gate')} |")
    pm = rep["run"]["processing_ms"]
    lines += ["", f"Processing time (ms, processing-only): median {pm.get('median')}, p95 {pm.get('value')}", "",
              f"Raw counts: `{json.dumps(rep['counts'])}`", "",
              f"Negative slices: `{json.dumps(rep['negative_slices'])}`", "",
              f"Failures listed: {len(rep['failures'])} (see report.json)"]
    if not math.isfinite(0):
        pass
    return "\n".join(lines) + "\n"


def rescore(gt_path: Path, out: Path) -> dict:
    """Recompute metrics from a saved results.jsonl (no inference); keeps the original run metadata."""
    gt = json.loads(gt_path.read_text())
    rows = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    by_img = {r["source"].split("/")[-1]: r for r in rows}
    pairs = [(sc, by_img[sc["image"]]) for sc in gt["scenes"] if sc["image"] in by_img]
    old = json.loads((out / "report.json").read_text())
    rep = score_results(pairs)
    rep["run"] = old["run"] | {"rescored_with_commit": env_info().get("implementation_commit"),
                               "rescored_from": "results.jsonl (no re-inference)"}
    rep["summary"] = {k: {kk: v.get(kk) for kk in ("k", "n", "value", "gate")} for k, v in rep["metrics"].items()}
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    (out / "report.md").write_text(to_markdown(rep))
    return rep
