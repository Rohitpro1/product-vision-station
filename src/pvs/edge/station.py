"""Laptop station: segment + recognise every pack each frame, read its text once per pack in the background.

    python -m pvs.edge.station run   --bundle edge_bundle --source 0              # webcam, live window
    python -m pvs.edge.station run   --bundle edge_bundle --source photo.jpg --out result.jpg --json result.json
    python -m pvs.edge.station bench --bundle edge_bundle --images folder/ --mem-limit-gb 4

Per frame: YOLO26-seg (detection + mask + brand class) on the GPU if there is one, else the CPU. Packs are
tracked across frames (box overlap), and each new pack's upright crop goes to an OCR worker thread once (again
only if the first read found nothing useful), so the frame loop never waits for OCR. The brand shown is the
model's class; printed text can confirm it or, for a generic "package", supply it. If the two disagree the pack
is flagged for a person to check instead of guessing.
"""
from __future__ import annotations

import argparse
import json
import queue
import resource
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .fields import parse_fields
from .seg import Det, YoloSeg

GENERIC = "package"


def _iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


class Track:
    _next = 1

    def __init__(self, det: Det, frame: int) -> None:
        self.id = Track._next
        Track._next += 1
        self.det, self.seen, self.hits = det, frame, 1
        self.votes: dict[str, float] = {}
        self.fields: dict = {}
        self.ocr_lines: list[str] = []
        self.ocr_tries = 0
        self.ocr_pending = False
        self.vote(det)

    def vote(self, det: Det) -> None:
        self.votes[det.name] = self.votes.get(det.name, 0.0) + det.score

    def brand(self) -> tuple[str, str]:
        """(name, status): status is 'model', 'model+text', 'text', 'check' or 'unknown'."""
        model = max(self.votes, key=self.votes.get) if self.votes else GENERIC
        text = self.fields.get("brand")
        if model != GENERIC:
            if text and text != model:
                return model, "check"
            return model, "model+text" if text == model else "model"
        if text:
            return text, "text"
        return GENERIC, "unknown"


class Station:
    def __init__(self, bundle: str | Path, device: str = "auto", conf: float | None = None, ocr: bool = True,
                 threads: int | None = None) -> None:
        bundle = Path(bundle)
        cfg = json.loads((bundle / "edge.json").read_text()) if (bundle / "edge.json").exists() else {}
        from .seg import providers

        gpu = providers(device)[0] != "CPUExecutionProvider"
        model = cfg.get("model_gpu") if gpu and cfg.get("model_gpu") else cfg.get("model", "model.onnx")
        self.seg = YoloSeg(bundle / model, device=device, threads=threads)
        self.conf = conf if conf is not None else float(cfg.get("conf", 0.35))
        self.brand_conf = float(cfg.get("brand_conf", self.conf))
        self.display = cfg.get("display_names", {})
        self.tracks: list[Track] = []
        self.frame = 0
        self.ocr = None
        self.q: queue.Queue = queue.Queue(maxsize=8)
        if ocr:
            from .ocr import PackOCR

            self.ocr = PackOCR(bundle, use_cuda=self.seg.provider.startswith("CUDA"))
            threading.Thread(target=self._ocr_worker, daemon=True).start()
        self.timing = {"seg_ms": [], "ocr_ms": []}

    def _ocr_worker(self) -> None:
        while True:
            tr, crop = self.q.get()
            t = time.perf_counter()
            try:
                lines = self.ocr.read(crop)
                tr.ocr_lines = lines
                tr.fields = parse_fields(lines)
            except Exception as e:  # noqa: BLE001 - one bad crop must not stop the station
                tr.fields = {"ocr_error": f"{type(e).__name__}"}
            self.timing["ocr_ms"].append((time.perf_counter() - t) * 1000)
            tr.ocr_pending = False

    def _relabel(self, d: Det) -> Det:
        if d.name != GENERIC and d.score < self.brand_conf:  # brand not sure enough: still a pack
            d.name = GENERIC
        return d

    def process(self, img: np.ndarray, wait_ocr: bool = False) -> list[dict]:
        self.frame += 1
        t = time.perf_counter()
        dets = [self._relabel(d) for d in self.seg(img, conf=self.conf)]
        self.timing["seg_ms"].append((time.perf_counter() - t) * 1000)
        live = []
        for d in sorted(dets, key=lambda x: -x.score):
            best = max(self.tracks, key=lambda tr: _iou(tr.det.box, d.box), default=None)
            if best is not None and best not in live and _iou(best.det.box, d.box) > 0.3:
                best.det, best.seen = d, self.frame
                best.hits += 1
                best.vote(d)
                live.append(best)
            else:
                tr = Track(d, self.frame)
                self.tracks.append(tr)
                live.append(tr)
        self.tracks = [tr for tr in self.tracks if self.frame - tr.seen <= 15]
        if self.ocr:
            for tr in live:
                useful = any(k in tr.fields for k in ("mrp", "exp", "mfg", "batch", "brand"))
                if not tr.ocr_pending and tr.ocr_tries < 3 and not useful and (tr.ocr_tries == 0 or tr.hits % 10 == 0):
                    try:
                        tr.ocr_pending = True
                        tr.ocr_tries += 1
                        self.q.put_nowait((tr, tr.det.upright_crop(img)))
                    except queue.Full:
                        tr.ocr_pending = False
                        tr.ocr_tries -= 1
            if wait_ocr:
                while any(tr.ocr_pending for tr in live):
                    time.sleep(0.002)
        return [self._row(tr) for tr in live]

    def _row(self, tr: Track) -> dict:
        name, status = tr.brand()
        d = tr.det
        return {"track": tr.id, "box": [round(v, 1) for v in d.box], "score": round(d.score, 3),
                "brand": name, "brand_name": self.display.get(name, name), "brand_status": status,
                "angle_deg": round(d.angle, 1), "polygon": d.polygon,
                **{k: v for k, v in tr.fields.items() if k in ("mrp", "mfg", "exp", "best_before", "batch")},
                "text": tr.ocr_lines[:12]}


def draw(img: np.ndarray, rows: list[dict]) -> np.ndarray:
    out = img.copy()
    over = img.copy()
    for r in rows:
        col = (40, 170, 60) if r["brand"] != GENERIC else (200, 140, 30)
        if r["brand_status"] == "check":
            col = (30, 60, 220)
        if r["polygon"]:
            cv2.fillPoly(over, [np.array(r["polygon"], np.int32)], col)
        x1, y1, x2, y2 = map(int, r["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = r["brand_name"] + (f"  MRP {r['mrp']:g}" if "mrp" in r else "") + \
            (f"  EXP {r['exp']['text']}" if "exp" in r else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), col, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.addWeighted(over, 0.35, out, 0.65, 0)


def _pct(v, q):
    return round(float(np.percentile(v, q)), 1) if v else None


def _rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def run(a) -> int:
    st = Station(a.bundle, a.device, a.conf, ocr=not a.no_ocr, threads=a.threads)
    print(f"model on {st.seg.provider}, input {st.seg.imgsz}px, classes {list(st.seg.names.values())}", flush=True)
    src = a.source
    if src.isdigit() or Path(src).suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        t0, n = time.time(), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rows = st.process(frame)
            n += 1
            vis = draw(frame, rows)
            fps = n / max(1e-6, time.time() - t0)
            cv2.putText(vis, f"{st.timing['seg_ms'][-1]:.0f} ms  {fps:.0f} fps  {st.seg.provider[:4]}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if a.json:
                with open(a.json, "a") as f:
                    f.write(json.dumps({"frame": st.frame, "packs": rows}) + "\n")
            if not a.headless:
                cv2.imshow("pack station (q to quit)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        cv2.destroyAllWindows()
    else:
        img = cv2.imread(src)
        if img is None:
            raise SystemExit(f"cannot read {src}")
        rows = st.process(img, wait_ocr=True)
        print(json.dumps(rows, indent=1, default=str))
        if a.out:
            cv2.imwrite(a.out, draw(img, rows))
        if a.json:
            Path(a.json).write_text(json.dumps(rows, indent=1, default=str))
    s = st.timing
    print(json.dumps({"seg_ms_p50": _pct(s["seg_ms"], 50), "seg_ms_p95": _pct(s["seg_ms"], 95),
                      "ocr_ms_p50": _pct(s["ocr_ms"], 50), "peak_rss_mb": _rss_mb()}))
    return 0


def bench(a) -> dict:
    """Latency + memory on a folder of photos. With --mem-limit-gb the process cannot use more than that."""
    if a.mem_limit_gb:  # address-space cap: stricter than RSS (CPU runs; the CUDA driver maps more than it uses)
        lim = int(a.mem_limit_gb * 2**30)
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    imgs = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:a.max_images]
    st = Station(a.bundle, a.device, a.conf, ocr=not a.no_ocr, threads=a.threads)
    frames = [cv2.imread(str(p)) for p in imgs]
    frames = [f for f in frames if f is not None]
    for f in frames[:3]:  # warm-up (first CUDA / ORT runs allocate)
        st.seg(f)
    st.timing = {"seg_ms": [], "ocr_ms": []}
    t_ocr = []
    packs = 0
    for f in frames:
        st.tracks = []  # independent photos: every pack is new, so every pack gets one OCR read
        t = time.perf_counter()
        rows = st.process(f, wait_ocr=True)
        t_ocr.append((time.perf_counter() - t) * 1000)
        packs += len(rows)
    rep = {"provider": st.seg.provider, "input_px": st.seg.imgsz, "photos": len(frames), "packs": packs,
           "seg_ms": {"p50": _pct(st.timing["seg_ms"], 50), "p95": _pct(st.timing["seg_ms"], 95),
                      "mean": round(float(np.mean(st.timing["seg_ms"])), 1) if frames else None},
           "ocr_ms_per_pack": {"p50": _pct(st.timing["ocr_ms"], 50), "p95": _pct(st.timing["ocr_ms"], 95)},
           "photo_with_ocr_ms": {"p50": _pct(t_ocr, 50), "p95": _pct(t_ocr, 95)},
           "peak_rss_mb": _rss_mb(), "mem_limit_gb": a.mem_limit_gb, "threads": a.threads,
           "note": "seg_ms = detection + segmentation + brand class, every frame. OCR runs once per new pack in a "
                   "background thread, so in live video it does not add to the frame time."}
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pvs.edge.station")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "bench"):
        p = sub.add_parser(name)
        p.add_argument("--bundle", required=True)
        p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
        p.add_argument("--conf", type=float, default=None)
        p.add_argument("--threads", type=int, default=None)
        p.add_argument("--no-ocr", action="store_true")
        p.add_argument("--out", default=None)
    r = sub.choices["run"]
    r.add_argument("--source", required=True, help="webcam index, video file or photo")
    r.add_argument("--json", default=None)
    r.add_argument("--headless", action="store_true")
    b = sub.choices["bench"]
    b.add_argument("--images", required=True)
    b.add_argument("--max-images", type=int, default=200)
    b.add_argument("--mem-limit-gb", type=float, default=None)
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else (bench(a) and 0)


if __name__ == "__main__":
    raise SystemExit(main())
