"""Shared per-frame pipeline for stills, replay and live capture (PDF p.2)."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from . import barcode as bc
from . import catalogue as catmod
from .capture import Frame, frame_quality
from .config import StationConfig, Thresholds, load_thresholds
from .detectors.base import Detection, ModelUnavailable, ThresholdMaskFallback
from .fusion import RULES_VERSION, FusionInput, decide
from .gallery import ColourHistEmbedder, Gallery
from .geometry import AxisGate, PeriodicSmoother, Reason, fit_axis, mask_points, periodic_error, wrap180, wrap360
from .orientation import TemplateMatcher, rectify_crop
from .schema import (
    Face,
    Field_,
    FrameResult,
    FrameStatus,
    IdentityLevel,
    Instance,
    Pose,
    Profile,
    Quality,
    ScoreType,
)
from .tracker import Tracker
from .transforms import crop

log = logging.getLogger("pvs.pipeline")


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Pipeline:
    def __init__(self, cfg: StationConfig, profile: Profile, root: Path = Path("."), scripted=None,
                 model_manifest: Path | None = None) -> None:
        self.cfg, self.profile, self.root = cfg, profile, root
        self.thr: Thresholds = load_thresholds(root / cfg.thresholds)
        self.cat_path = root / cfg.catalogue
        self.catalogue = catmod.load(self.cat_path)
        self.cat_root = self.cat_path.parent
        self.gtin_idx = catmod.gtin_index(self.catalogue)
        self.failed: str | None = None
        self.backend: dict[str, str] = {}
        self.detector = self.masker = self.embedder = None
        self.gallery: Gallery | None = None
        manifest = json.loads((model_manifest or root / "models/manifest.json").read_text())["models"]
        load_t0 = time.perf_counter()
        try:
            self._load_backends(scripted, manifest)
        except ModelUnavailable as e:
            self.failed = str(e)
            log.error("model load failed: %s", e)
        except Exception as e:  # noqa: BLE001 - any backend/gallery init error is a visible degraded state
            self.failed = f"backend_init_failed:{type(e).__name__}: {e}"
            log.exception("backend init failed")
        self.load_s = time.perf_counter() - load_t0
        self.matcher = TemplateMatcher(self.thr.template_min_inliers, self.thr.template_min_spread,
                                       self.thr.template_max_reproj_px)
        for p in self.catalogue["products"]:
            if p.get("approval_state") in ("approved", "fixture"):
                for r in p["references"]:
                    if r["side"] in ("front", "back") and r.get("A") and r.get("B"):
                        img = cv2.imread(str(self.cat_root / r["path"]))
                        self.matcher.add_reference(p["sku"], r["side"], img, r["A"], r["B"], r.get("variant_cues"),
                                                   p.get("family"))
        self.backend["template"] = self.matcher.backend_id
        self.backend["barcode"] = "zxing-cpp"
        self.backend["ocr"] = "unavailable"
        self.axis_gate = AxisGate(self.thr.axis_min_elongation, self.thr.axis_min_extent_px, self.thr.axis_max_residual)
        self.score_type = ScoreType.CALIBRATED if self.thr.calibrated else ScoreType.RAW_COSINE

    # ------------------------------------------------------------ backends
    def _load_backends(self, scripted, manifest: dict) -> None:
        m = self.cfg.models
        if self.profile == Profile.FIXTURE:
            if scripted is None:
                raise ModelUnavailable("fixture profile requires a scripted ground-truth source")
            self.detector = scripted
            self.masker = None  # scripted masks travel with detections
            self.embedder = ColourHistEmbedder()
        else:
            from .detectors.hf_models import GroundingDinoDetector, Sam2MaskProvider
            from .gallery import DinoV2Embedder

            if m.detector == "grounding_dino":
                gd = manifest["grounding_dino"]
                self.detector = GroundingDinoDetector(gd["repo"], gd["revision"], m.prompts, m.box_threshold,
                                                      m.text_threshold, m.detector_max_side)
            elif m.detector == "rtdetr_onnx":
                from .detectors.rtdetr import RTDetrOnnxDetector

                if not m.rtdetr_onnx:
                    raise ModelUnavailable("rtdetr_onnx selected but no exported model configured")
                self.detector = RTDetrOnnxDetector(self.root / m.rtdetr_onnx, m.provider)
            else:
                raise ModelUnavailable(f"detector {m.detector} not allowed in profile {self.profile.value}")
            if m.masks == "sam2":
                s = manifest["sam2"]
                self.masker = Sam2MaskProvider(s["repo"], s["revision"])
            else:
                self.masker = ThresholdMaskFallback()
            if m.embedder == "dinov2":
                d = manifest["dinov2"]
                self.embedder = DinoV2Embedder(d["repo"], d["revision"])
            else:
                self.embedder = ColourHistEmbedder()
        self.backend.update({"detector": self.detector.backend_id,
                             "masks": self.masker.backend_id if self.masker else "scripted-fixture",
                             "embedder": self.embedder.backend_id})
        self.gallery = self._gallery()
        self.backend["gallery"] = f"{self.gallery.backend_id}#{self.gallery.digest()}"

    def _gallery(self) -> Gallery:
        cache_dir = self.root / "var" / "gallery"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(f"{self.embedder.backend_id}|{self.catalogue['revision']}|"
                             f"{self.cat_path.read_bytes().hex()[:4096]}".encode()).hexdigest()[:16]
        p = cache_dir / f"{key}.npz"
        if self.cfg.gallery and (self.root / self.cfg.gallery).exists():
            g = Gallery.load(self.root / self.cfg.gallery)
            if g.backend_id == self.embedder.backend_id and g.catalogue_revision == self.catalogue["revision"]:
                return g
        if p.exists():
            return Gallery.load(p)
        g = Gallery.build(self.catalogue, self.cat_root, self.embedder)
        g.save(p)
        return g

    # ------------------------------------------------------------ process
    def _envelope(self, frame: Frame, status: FrameStatus, reason: str = "ok") -> FrameResult:
        return FrameResult(status=status, reason=reason, profile=self.profile, backend=dict(self.backend),
                           model_bundle=self.cfg.bundle_id, catalogue_revision=self.catalogue["revision"],
                           thresholds_version=self.thr.version, fusion_rules_version=RULES_VERSION,
                           camera=frame.camera, stream_epoch=frame.epoch, frame_id=frame.frame_id,
                           captured_at=_iso(frame.captured_at), processed_at=_iso(datetime.now(UTC)),
                           source=frame.source, image_size_px=list(frame.size_wh))

    def process(self, frame: Frame, tracker: Tracker | None = None) -> FrameResult:
        T: dict[str, float] = {}
        t0 = time.perf_counter()
        if self.failed:
            r = self._envelope(frame, FrameStatus.MODEL_FAILED, self.failed)
            return r
        q = frame_quality(frame.image)
        qs = "ok"
        if q["mean_luma"] < self.thr.quality_min_luma:
            qs = "dark"
        quality = Quality(status=qs, **{k: round(v, 3) for k, v in q.items()})
        if qs != "ok":
            r = self._envelope(frame, FrameStatus.POOR_QUALITY, f"quality_{qs}")
            r.quality = quality
            return r
        T["quality"] = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        try:
            ctx: dict = {"source_key": frame.source_key}
            dets: list[Detection] = self.detector.detect(frame.image, ctx)
        except Exception as e:  # noqa: BLE001 - runtime model failure is a distinct outcome
            log.exception("detector failed")
            r = self._envelope(frame, FrameStatus.MODEL_FAILED, f"detector_runtime_error:{type(e).__name__}")
            r.quality = quality
            return r
        T["detect"] = (time.perf_counter() - t1) * 1000
        reason = ctx.get("reason", "ok")
        dets = sorted(dets, key=lambda d: -d.score)
        if len(dets) > self.cfg.limits.max_instances:
            dets = dets[: self.cfg.limits.max_instances]
            reason = "instances_capped"
        t1 = time.perf_counter()
        if self.masker is not None and dets:
            for d, (m, src) in zip(dets, self.masker.masks(frame.image, [d.box_xyxy for d in dets])):
                d.mask, d.mask_source = m, src
        T["masks"] = (time.perf_counter() - t1) * 1000
        if dets:
            # blur is judged on the detected crops: a featureless empty scene is not "blurred"
            # blur and glare are judged on detected crops: a plain or white backdrop is not poor quality
            sharp, sat = [], []
            for d in dets:
                c, _, _ = crop(frame.image, tuple(d.box_xyxy))
                if c.size:
                    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
                    sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
                    sat.append(float((g >= 250).mean()))
            bad = None
            if sharp and max(sharp) < self.thr.quality_min_sharpness:
                bad = "blurred"
            elif sat and min(sat) > self.thr.quality_max_saturated:
                bad = "saturated"
            if bad:
                quality.status = bad
                r = self._envelope(frame, FrameStatus.POOR_QUALITY, f"quality_{bad}")
                r.quality = quality
                return r
        result = self._envelope(frame, FrameStatus.OK if dets else FrameStatus.NO_PACKAGES, reason)
        result.quality = quality
        if not dets:
            result.timing_ms = {k: round(v, 2) for k, v in T.items()}
            return result
        tracks = None
        if tracker is not None:
            if frame.epoch != tracker.epoch:
                tracker.reset(frame.epoch)
            tracks = tracker.update([d.box_xyxy for d in dets], [d.score for d in dets], frame.mono_t,
                                    frame.frame_id, frame.content_hash)
        stage = {"identity": 0.0, "orientation": 0.0, "barcode": 0.0}
        per = []
        reads_all: list[tuple[int, bc.BarcodeRead]] = []
        for i, d in enumerate(dets):
            ts = time.perf_counter()
            axis = fit_axis(mask_points(d.mask), self.axis_gate) if d.mask is not None else None
            crop_img, crop_T, _ = crop(frame.image, tuple(d.box_xyxy), expand=0.1)
            rect = None
            if axis is not None and axis.angle_deg is not None and axis.centre is not None:
                pts = mask_points(d.mask)
                rad = np.radians(axis.angle_deg)
                minor = (pts - axis.centre) @ np.array([-np.sin(rad), np.cos(rad)])
                rect = rectify_crop(frame.image, axis.centre, axis.angle_deg, axis.extent_px * 1.08,
                                    float(minor.max() - minor.min()) * 1.15 + 4)
            tmpl = self.matcher.match(crop_img, crop_T) if crop_img.size else None
            if tmpl is not None and tmpl.reason == Reason.OK:
                self.matcher.cue_check(crop_img, tmpl)
            stage["orientation"] += (time.perf_counter() - ts) * 1000
            ts = time.perf_counter()
            reads = bc.decode(crop_img, (crop_T.inverse(np.zeros((1, 2)))[0][0], crop_T.inverse(np.zeros((1, 2)))[0][1]))
            if not reads and rect is not None:
                for r in bc.decode(rect[0]):
                    r.polygon_px = rect[1].inverse(np.array(r.polygon_px)).tolist()
                    reads.append(r)
            reads_all += [(i, r) for r in reads]
            stage["barcode"] += (time.perf_counter() - ts) * 1000
            ts = time.perf_counter()
            emb_src = rect[0] if rect is not None else crop_img
            hit = self.gallery.search(self.embedder.embed([emb_src])[0]) if emb_src.size else None
            stage["identity"] += (time.perf_counter() - ts) * 1000
            per.append((d, axis, tmpl, hit))
        # bind each code to exactly one instance (dedupe identical reads from overlapping crops)
        boxes = [d.box_xyxy for d in dets]
        bound: dict[int, list[bc.BarcodeRead]] = {i: [] for i in range(len(dets))}
        seen = set()
        for _, r in reads_all:
            key = (r.text, tuple(round(v) for pt in r.polygon_px for v in pt))
            if key in seen:
                continue
            seen.add(key)
            owner = bc.associate(r.polygon_px, boxes)
            if owner is not None:
                bound[owner].append(r)
        now_iso = _iso(frame.captured_at)
        for i, (d, axis, tmpl, hit) in enumerate(per):
            tr = tracks[i] if tracks is not None else None
            if tr is not None:
                for r in bound[i]:
                    if r.valid:
                        tracker.add_evidence(tr, "barcode", r, frame.frame_id, frame.mono_t, frame.content_hash)
                held = tr.state.get("identity")
                held_age = frame.mono_t - held[1] if held else None
                fin = FusionInput(hit, bound[i], tmpl, held[0] if held else None, held_age, frame.frame_id, now_iso)
            else:
                fin = FusionInput(hit, bound[i], tmpl, None, None, frame.frame_id, now_iso)
            ident = decide(fin, self.thr, self.gtin_idx, self.score_type)
            if tr is not None and ident.level in (IdentityLevel.SKU, IdentityLevel.FAMILY) and ident.rule != "R6_track_hold":
                tr.state["identity"] = (ident, frame.mono_t)
            if tr is not None and ident.level == IdentityLevel.CONFLICT:
                tr.state.pop("identity", None)
            pose = self._pose(d, axis, tmpl, tr, frame, now_iso)
            result.instances.append(Instance(
                instance_id=f"{frame.camera}:{frame.epoch}:{frame.frame_id}:{i}", track_id=tr.tid if tr else None,
                bbox_xyxy_px=[round(v, 1) for v in d.box_xyxy], det_score=round(d.score, 4), det_source=d.source,
                mask_source=d.mask_source if d.mask is not None else "none", identity=ident, pose=pose))
        T.update(stage)
        T["total"] = (time.perf_counter() - t0) * 1000
        result.timing_ms = {k: round(v, 2) for k, v in T.items()}
        return result

    def _pose(self, d: Detection, axis, tmpl, tr, frame: Frame, now_iso: str) -> Pose:
        thr = self.thr
        ax_val, ax_reason = None, Reason.NO_MASK.value
        if axis is not None:
            ax_val, ax_reason = axis.angle_deg, axis.reason.value
        hd_val, hd_reason, face, face_score, face_reason = None, Reason.ORIENTATION_UNAVAILABLE.value, Face.UNKNOWN, None, "ok"
        endpoints = None
        if tmpl is not None and tmpl.reason == Reason.OK:
            hd_val = tmpl.heading_deg
            face = Face(tmpl.face)
            face_score = round(tmpl.score, 3)
            endpoints = {"A": [round(v, 1) for v in tmpl.A], "B": [round(v, 1) for v in tmpl.B]}
            hd_reason = "ok"
            if ax_val is not None and periodic_error(hd_val % 180.0, ax_val, 180.0) > 20.0:
                hd_val, endpoints, hd_reason = None, None, Reason.AXIS_UNSTABLE.value  # template disagrees with shape
        elif tmpl is not None:
            hd_reason = tmpl.reason.value if tmpl.reason != Reason.LOW_INLIERS else Reason.ENDPOINT_HIDDEN.value
            face_reason = Reason.FACE_UNKNOWN.value
        if face == Face.UNKNOWN and axis is not None and axis.elongation >= 7.0 and axis.angle_deg is not None:
            face, face_reason = Face.EDGE, "elongation_edge_rule"
        ax_age = hd_age = 0.0 if hd_val is not None else None
        ax_age = 0.0 if ax_val is not None else None
        if tr is not None:
            sa = tr.state.setdefault("smoother_axis", PeriodicSmoother(180.0, thr.smoother_window_s, thr.smoother_min_resultant, thr.evidence_ttl_s))
            sh = tr.state.setdefault("smoother_heading", PeriodicSmoother(360.0, thr.smoother_window_s, thr.smoother_min_resultant, thr.evidence_ttl_s))
            key = frame.content_hash or f"{frame.epoch}:{frame.frame_id}"
            sa.add(frame.mono_t, ax_val, key)
            sh.add(frame.mono_t, hd_val, key)
            if face in (Face.FRONT, Face.BACK):
                tr.state["face"] = (face, face_score, frame.mono_t)
            a_est, _, a_r = sa.estimate(frame.mono_t)
            h_est, _, h_r = sh.estimate(frame.mono_t)
            if ax_val is not None:
                ax_val = a_est if a_est is not None else None
                ax_reason = "ok" if a_est is not None else a_r.value
            if hd_val is not None:
                hd_val, hd_reason = (h_est, "ok") if h_est is not None else (None, h_r.value)
                if hd_val is None:
                    endpoints = None
            elif h_est is not None and sh._obs:
                # brief, tested hold of the last supported direction; age is exposed
                age = frame.mono_t - sh._obs[-1].t
                # hold only through transient occlusion (face unknown); a turn to edge/back clears the arrow
                if face == Face.UNKNOWN and age <= thr.evidence_ttl_s and h_r == Reason.OK and hd_reason != Reason.AXIS_UNSTABLE.value:
                    hd_val, hd_reason, hd_age = h_est, "held", round(age * 1000, 1)
        return Pose(face=face, face_score=face_score, face_reason=face_reason,
                    axis=Field_(value_deg=None if ax_val is None else wrap180(round(ax_val, 2)), period_deg=180,
                                available=ax_val is not None, reason=ax_reason,
                                score=round(min(1.0, axis.elongation / 4), 3) if axis and ax_val is not None else None,
                                source=f"mask:{d.mask_source}" if d.mask is not None else None,
                                observed_at=now_iso if ax_val is not None else None, age_ms=ax_age),
                    heading=Field_(value_deg=None if hd_val is None else wrap360(round(hd_val, 2)), period_deg=360,
                                   available=hd_val is not None, reason=hd_reason,
                                   score=face_score if hd_val is not None else None,
                                   source=self.matcher.backend_id if hd_val is not None else None,
                                   observed_at=now_iso if hd_val is not None else None, age_ms=hd_age),
                    endpoints_px=endpoints)
