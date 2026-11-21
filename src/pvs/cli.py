"""``pvs`` command-line entry point."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2


def _pipeline(a, need_scripted_gt: Path | None = None):
    from .config import load_station
    from .pipeline import Pipeline
    from .schema import Profile

    cfg = load_station(a.config)
    if a.bundle_id:
        cfg.bundle_id = a.bundle_id
    profile = Profile(a.profile)
    scripted = None
    if profile == Profile.FIXTURE:
        from .detectors.scripted import ScriptedDetector

        scripted = ScriptedDetector()
        gts = [need_scripted_gt] if need_scripted_gt else [Path(g) for g in (getattr(a, "ground_truth", None) or [])]
        for gt in gts:
            if gt and gt.exists():
                scripted.load_ground_truth(json.loads(gt.read_text()))
    if profile == Profile.STATION and cfg.bundle_id == "dev-unbundled":
        from .bundle import active_bundle

        b = active_bundle(Path(a.station_dir))
        if b is None:
            print("station profile: no verified active bundle installed", file=sys.stderr)
        else:
            cfg.bundle_id = b["bundle_id"]
    p = Pipeline(cfg, profile, Path(a.root), scripted=scripted)
    print(f"[pvs] profile={profile.value} bundle={cfg.bundle_id} backends={json.dumps(p.backend)} "
          f"load_s={p.load_s:.1f}" + (f" DEGRADED: {p.failed}" if p.failed else ""), file=sys.stderr)
    return p, cfg


def cmd_detect(a) -> int:
    from .capture import DecodeError, frame_from_file
    from .render import draw
    from .schema import FrameStatus

    p, cfg = _pipeline(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for img_path in a.images:
        try:
            f = frame_from_file(img_path, camera=cfg.camera_id, max_pixels=cfg.limits.max_pixels)
        except (DecodeError, OSError) as e:
            print(json.dumps({"image": img_path, "status": FrameStatus.DECODE_FAILED.value, "error": str(e)}))
            rc = 2
            continue
        r = p.process(f)
        stem = Path(img_path).stem
        (out / f"{stem}.json").write_text(r.model_dump_json(indent=2))
        cv2.imwrite(str(out / f"{stem}_overlay.jpg"), draw(f.image, r, mirror=cfg.capture.mirror_preview))
        summ = [(i.identity.level.value, i.identity.sku or i.identity.family, i.pose.axis.value_deg,
                 i.pose.heading.value_deg, i.pose.face.value) for i in r.instances]
        print(json.dumps({"image": img_path, "status": r.status.value, "reason": r.reason,
                          "instances": summ, "timing_ms": r.timing_ms.get("total")}))
        if r.status == FrameStatus.MODEL_FAILED:
            rc = 3
    return rc


def cmd_replay(a) -> int:
    from .capture import replay
    from .outbox import Outbox, transitions
    from .render import draw
    from .tracker import Tracker

    p, cfg = _pipeline(a)
    tr = Tracker(cfg.camera_id)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ob = Outbox(Path(a.outbox) if a.outbox else out / "outbox.sqlite", cfg.station_id)
    writer = None
    n = 0
    t0 = time.perf_counter()
    with open(out / "results.jsonl", "w") as fh:
        for f in replay(a.video, cfg.camera_id, a.sample_hz):
            r = p.process(f, tr)
            fh.write(r.model_dump_json() + "\n")
            for _ev in transitions(r, ob):
                pass
            if a.overlay:
                vis = draw(f.image, r, mirror=cfg.capture.mirror_preview)
                if writer is None:
                    writer = cv2.VideoWriter(str(out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             a.sample_hz or 15, (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            n += 1
    if writer:
        writer.release()
    dt = time.perf_counter() - t0
    print(json.dumps({"frames": n, "processing_fps": round(n / dt, 2) if dt else None,
                      "events": ob.count(), "timing": "processing-only (file replay; no live capture delay)"}))
    return 0


def cmd_camera(a) -> int:
    from .capture import LiveCamera
    from .render import draw
    from .tracker import Tracker

    p, cfg = _pipeline(a)
    cam = LiveCamera(a.device or cfg.capture.device, cfg.camera_id, cfg.capture.width, cfg.capture.height,
                     cfg.capture.queue_size, cfg.capture.max_frame_age_ms, cfg.capture.reconnect_s)
    cam.start()
    tr = Tracker(cfg.camera_id)
    period = 1.0 / cfg.capture.sample_hz
    scheduled = processed = 0
    t_end = time.monotonic() + a.seconds if a.seconds else None
    try:
        while t_end is None or time.monotonic() < t_end:
            tick = time.monotonic()
            scheduled += 1
            f = cam.latest()
            if f is not None:
                r = p.process(f, tr)
                processed += 1
                if a.show:
                    cv2.imshow("pvs", draw(f.image, r, cfg.capture.mirror_preview, cam.stale(cfg.capture.stall_timeout_s)))
                    if cv2.waitKey(1) == 27:
                        break
                else:
                    print(r.model_dump_json())
            time.sleep(max(0.0, period - (time.monotonic() - tick)))
    finally:
        cam.stop()
    print(json.dumps({"scheduled": scheduled, "processed": processed, "dropped_stale": cam.dropped_stale,
                      "dropped_overflow": cam.dropped_overflow, "camera_resets": cam.resets}), file=sys.stderr)
    return 0 if processed else 4


def cmd_fixtures(a) -> int:
    from . import fixtures as fx

    root = Path(a.out)
    if a.what in ("catalogue", "all"):
        fx.build_fixture_catalogue(Path(a.catalogue_dir))
        print("catalogue ->", a.catalogue_dir)
    if a.what in ("scenes", "all"):
        print("scenes ->", fx.generate_set(root / "scenes", a.n, a.seed))
    if a.what in ("crossing", "all"):
        (root / "video").mkdir(parents=True, exist_ok=True)
        truths = fx.make_crossing_video(root / "video" / "crossing.mp4")
        from dataclasses import asdict

        scenes = [{"image": f"crossing.mp4#{i}", **asdict(t)} for i, t in enumerate(truths)]
        (root / "video" / "crossing_truth.json").write_text(json.dumps({"generator": "make_crossing_video",
                                                                         "scenes": scenes}))
        print("crossing ->", root / "video" / "crossing.mp4")
    return 0


def cmd_catalogue(a) -> int:
    from . import catalogue as c

    path = Path(a.catalogue)
    if a.action == "create":
        c.create(path, a.name)
    elif a.action == "validate":
        errs = c.validate(c.load(path), path.parent)
        print(json.dumps({"catalogue": str(path), "valid": not errs, "errors": errs}, indent=1))
        return 1 if errs else 0
    elif a.action == "enrol":
        prod = json.loads(Path(a.product).read_text())
        cat = c.enrol(path, prod, path.parent)
        print("enrolled", prod["sku"], "->", cat["revision"])
    elif a.action == "rollback":
        print("rolled back to", c.rollback(path)["revision"])
    elif a.action in ("build-gallery", "collisions"):
        from .config import load_station
        from .gallery import ColourHistEmbedder, DinoV2Embedder, Gallery

        cat = c.load(path)
        if a.embedder == "dinov2":
            m = json.loads((Path(a.root) / "models/manifest.json").read_text())["models"]["dinov2"]
            emb = DinoV2Embedder(m["repo"], m["revision"])
        else:
            emb = ColourHistEmbedder()
        g = Gallery.build(cat, path.parent, emb)
        if a.action == "build-gallery":
            g.save(Path(a.out))
            print(json.dumps({"gallery": a.out, "refs": len(g.refs), "backend": g.backend_id, "digest": g.digest()}))
        else:
            thr = a.threshold
            res = {p["sku"]: c.collision_test(g, p["sku"], thr) for p in cat["products"]}
            print(json.dumps(res, indent=1))
            return 1 if any(res.values()) else 0
        _ = load_station
    return 0


def cmd_serve(a) -> int:
    import uvicorn

    from .api import create_app

    p, cfg = _pipeline(a)
    app = create_app(p, cfg, video=a.video, device=a.device)
    uvicorn.run(app, host=a.host or cfg.bind, port=a.port or cfg.port, log_level="info")
    return 0


def cmd_eval(a) -> int:
    from .evaluation import evaluate_fixture_set, rescore

    if a.rescore:
        rep = rescore(Path(a.ground_truth), Path(a.out))
        print(json.dumps(rep["summary"], indent=1))
        return 0

    p, _ = _pipeline(a, Path(a.ground_truth))
    rep = evaluate_fixture_set(p, Path(a.ground_truth), Path(a.out))
    print(json.dumps(rep["summary"], indent=1))
    return 0


def cmd_bundle(a) -> int:
    from . import bundle as b

    if a.action == "keygen":
        b.keygen(Path(a.key), Path(a.pub))
    elif a.action == "build":
        man = b.build(Path(a.src), Path(a.out), a.bundle_id, Path(a.key), json.loads(a.meta or "{}"))
        print(json.dumps({"bundle": a.out, "files": len(man["files"])}))
    elif a.action == "verify":
        b.verify(Path(a.src), Path(a.trust))
        print("verified", a.src)
    elif a.action == "install":
        print("active ->", b.install(Path(a.src), Path(a.station_dir), Path(a.trust)))
    elif a.action == "rollback":
        print("active ->", b.rollback(Path(a.station_dir), Path(a.trust)))
    return 0


def cmd_train(a) -> int:
    from .training import landmark

    return landmark.main(a.rest)


def cmd_doctor(a) -> int:
    import platform

    import onnxruntime as ort

    info = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": __import__("os").cpu_count(),
            "onnxruntime_providers": ort.get_available_providers()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
    except ImportError:
        info["torch"] = None
    import glob

    info["video_devices"] = glob.glob("/dev/video*")
    print(json.dumps(info, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(prog="pvs")
    ap.add_argument("--root", default=".")
    ap.add_argument("--config", default="configs/station.fixture.yaml")
    ap.add_argument("--profile", choices=["fixture", "pretrained", "station"], default="fixture")
    ap.add_argument("--bundle-id", default=None)
    ap.add_argument("--station-dir", default="var/station")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("detect", help="still images -> JSON + overlay")
    s.add_argument("images", nargs="+")
    s.add_argument("--out", default="out/detect")
    s.add_argument("--ground-truth", action="append", help="fixture profile: ground_truth.json (repeatable)")
    s.set_defaults(fn=cmd_detect)

    s = sub.add_parser("replay", help="prerecorded video replay with tracking + events")
    s.add_argument("video")
    s.add_argument("--out", default="out/replay")
    s.add_argument("--sample-hz", type=float, default=None)
    s.add_argument("--overlay", action="store_true")
    s.add_argument("--outbox")
    s.add_argument("--ground-truth", action="append")
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("camera", help="live local camera")
    s.add_argument("--device")
    s.add_argument("--seconds", type=float, default=0)
    s.add_argument("--show", action="store_true")
    s.set_defaults(fn=cmd_camera)

    s = sub.add_parser("fixtures", help="generate redistributable fixtures")
    s.add_argument("what", choices=["catalogue", "scenes", "crossing", "all"])
    s.add_argument("--out", default="data/fixtures")
    s.add_argument("--catalogue-dir", default="catalogues/fixture")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--seed", type=int, default=1000)
    s.set_defaults(fn=cmd_fixtures)

    s = sub.add_parser("catalogue", help="catalogue management")
    s.add_argument("action", choices=["create", "validate", "enrol", "rollback", "build-gallery", "collisions"])
    s.add_argument("--catalogue", default="catalogues/fixture/catalogue.json")
    s.add_argument("--name", default="station")
    s.add_argument("--product")
    s.add_argument("--embedder", choices=["dinov2", "colour_hist"], default="dinov2")
    s.add_argument("--out", default="var/gallery.npz")
    s.add_argument("--threshold", type=float, default=0.95)
    s.set_defaults(fn=cmd_catalogue)

    s = sub.add_parser("serve", help="FastAPI service (loopback by default)")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--video", help="replay a video as the live source")
    s.add_argument("--device", help="camera device for the live source")
    s.add_argument("--ground-truth", action="append", help="fixture profile: scripted detections source (repeatable)")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("eval", help="evaluate on a ground-truth manifest")
    s.add_argument("--ground-truth", required=True)
    s.add_argument("--out", default="evidence/eval")
    s.add_argument("--rescore", action="store_true", help="recompute metrics from <out>/results.jsonl without inference")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("bundle", help="signed model bundles")
    s.add_argument("action", choices=["keygen", "build", "verify", "install", "rollback"])
    s.add_argument("--src")
    s.add_argument("--out")
    s.add_argument("--bundle-id")
    s.add_argument("--key")
    s.add_argument("--pub")
    s.add_argument("--trust", default="deploy/trust")
    s.add_argument("--meta")
    s.set_defaults(fn=cmd_bundle)

    s = sub.add_parser("train", help="training entry points (landmark head)")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("doctor", help="environment inventory")
    s.set_defaults(fn=cmd_doctor)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
