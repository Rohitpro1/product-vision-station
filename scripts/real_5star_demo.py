#!/usr/bin/env python3
"""Real-image evidence with a community photograph of a 5 Star bar (Open Food Facts, CC BY-SA 3.0).

Steps (all with the *pretrained* profile — real Grounding DINO / SAM 2 / DINOv2 weights):
 1. un-enrolled: the bar against the fixture catalogue -> expected identity `unknown` (+ axis if measurable)
 2. enrolment tooling exercise: rectify the detected bar into a front reference, enrol it in a scratch
    catalogue (var/demo_5star, never committed, approval "demo"), rebuild the gallery
 3. SAME-PHOTOGRAPH self-consistency check (NOT validation): detect on a rotated/re-lit copy of the same photo
 4. real non-target: the Milka wrapper photo against the demo catalogue -> must not be accepted as 5 Star

Outputs: evidence/real/*.json and downscaled overlays (CC BY-SA adaptations, attributed in THIRD_PARTY_NOTICES.md).
Prerequisite: python scripts/fetch_models.py && python scripts/fetch_real_examples.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pvs import catalogue as catmod  # noqa: E402
from pvs.capture import Frame, frame_from_file  # noqa: E402
from pvs.config import load_station  # noqa: E402
from pvs.geometry import fit_axis, mask_points  # noqa: E402
from pvs.orientation import rectify_crop  # noqa: E402
from pvs.pipeline import Pipeline  # noqa: E402
from pvs.render import draw  # noqa: E402
from pvs.schema import Profile  # noqa: E402

OUT = ROOT / "evidence" / "real"
REAL = ROOT / "realdata"
DEMO = ROOT / "var" / "demo_5star"


def save(name: str, f: Frame, r, note: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(r.model_dump_json(indent=1))
    vis = draw(f.image, r)
    s = 1280 / max(vis.shape[:2])
    cv2.imwrite(str(OUT / f"{name}_overlay.jpg"), cv2.resize(vis, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 82])
    summ = {"example": name, "note": note, "status": r.status.value, "timing_ms": r.timing_ms.get("total"),
            "instances": [{"det_score": i.det_score, "mask_source": i.mask_source, "identity": i.identity.level.value,
                           "sku": i.identity.sku, "family": i.identity.family, "rule": i.identity.rule,
                           "reason": i.identity.reason, "gallery_top": (i.identity.candidates[0].model_dump()
                                                                         if i.identity.candidates else None),
                           "axis_deg": i.pose.axis.value_deg, "heading_deg": i.pose.heading.value_deg,
                           "heading_reason": i.pose.heading.reason, "face": i.pose.face.value} for i in r.instances]}
    print(json.dumps(summ))
    return summ


def main() -> int:
    photo = REAL / "off_5star_1.jpg"
    milka = REAL / "milka_wrapper.jpg"
    if not photo.exists():
        print("missing realdata/off_5star_1.jpg: run scripts/fetch_real_examples.py", file=sys.stderr)
        return 2
    cfg = load_station(ROOT / "configs/station.pretrained.yaml")
    results = []
    # 1. un-enrolled
    pl = Pipeline(cfg, Profile.PRETRAINED, ROOT)
    if pl.failed:
        print("model_failed:", pl.failed, file=sys.stderr)
        return 3
    f = frame_from_file(photo, camera="photo")
    r = pl.process(f)
    results.append(save("5star_unenrolled", f, r, "real 5 Star photo vs fixture catalogue (5 Star not enrolled)"))
    best = max(r.instances, key=lambda i: i.det_score)
    # 2. enrol a reference from the detected bar
    det = pl.detector.detect(f.image)
    d = max(det, key=lambda d: d.score)
    (m, src), = pl.masker.masks(f.image, [d.box_xyxy])
    ax = fit_axis(mask_points(m))
    pts = mask_points(m)
    rad = np.radians(ax.angle_deg)
    minor = (pts - ax.centre) @ np.array([-np.sin(rad), np.cos(rad)])
    ref, _ = rectify_crop(f.image, ax.centre, ax.angle_deg, ax.extent_px * 1.02, float(minor.max() - minor.min()) * 1.02)
    if DEMO.exists():
        shutil.rmtree(DEMO)
    (DEMO / "refs").mkdir(parents=True)
    shutil.copy(ROOT / "catalogues/fixture/refs/neg_plain_gold.png", DEMO / "refs/neg_plain_gold.png")
    cv2.imwrite(str(DEMO / "refs/5star_front.png"), ref)
    h, w = ref.shape[:2]
    cat = catmod.create(DEMO / "catalogue.json", "demo5star")
    prod = {"sku": "demo:cadbury-5star-18g-unverified", "family": "cadbury:5star", "gtins": [],
            "quantity": "18g (as photographed; unverified)", "variant": "classic", "artwork_revision": "observed-off-1",
            "references": [{"side": "front", "path": "refs/5star_front.png",
                            "sha256": hashlib.sha256((DEMO / "refs/5star_front.png").read_bytes()).hexdigest(),
                            "A": [0.0, h / 2], "B": [float(w), h / 2]}],
            "physical_ends": {"A": "left crimp when wordmark upright (proposal)", "B": "right crimp",
                              "definition": "docs/reference_drawing.md#cadbury-5-star"},
            "supported_faces": ["front"], "provenance": "rectified from Open Food Facts image 1 of 7622202318078",
            "usage_rights": "CC BY-SA 3.0 demo only; NOT an authorised reference", "approval_state": "approved"}
    cat = catmod.enrol(DEMO / "catalogue.json", prod, DEMO)
    cat["negatives"] = [{"name": "plain-gold-wrapper", "path": "refs/neg_plain_gold.png", "revision": "1"}]
    (DEMO / "catalogue.json").write_text(json.dumps(cat, indent=1))
    errs = catmod.validate(catmod.load(DEMO / "catalogue.json"), DEMO)
    print(json.dumps({"enrolled": prod["sku"], "catalogue_revision": cat["revision"], "validation_errors": errs,
                      "reference_px": [w, h], "mask_source": src}))
    cfg2 = load_station(ROOT / "configs/station.pretrained.yaml")
    cfg2.catalogue = str((DEMO / "catalogue.json").relative_to(ROOT))
    pl2 = Pipeline(cfg2, Profile.PRETRAINED, ROOT)
    # 3. same-photograph self-consistency (rotated + re-lit copy)
    img = f.image
    Hh, Ww = img.shape[:2]
    M = cv2.getRotationMatrix2D((Ww / 2, Hh / 2), 65, 0.8)
    rot = cv2.warpAffine(img, M, (Ww, Hh), borderValue=(200, 200, 200))
    rot = np.clip(rot.astype(np.float32) * 0.85 + 10, 0, 255).astype(np.uint8)
    g = Frame(rot, "photo", 0, 1, f.captured_at, 0.0, "off_5star_1.jpg (rotated 65deg, relit)")
    r3 = pl2.process(g)
    results.append(save("5star_enrolled_same_photo_rotated", g, r3,
                        "SAME photograph as the enrolled reference, rotated and re-lit: self-consistency only, NOT validation"))
    # 4. real non-target
    if milka.exists():
        fm = frame_from_file(milka, camera="photo")
        r4 = pl2.process(fm)
        results.append(save("milka_vs_demo_5star_catalogue", fm, r4, "real non-target (Milka) vs demo 5 Star catalogue"))
    (OUT / "index.json").write_text(json.dumps({"provenance": {
        "off_5star_1.jpg": "Open Food Facts contributors, CC BY-SA 3.0, product 7622202318078 image 1, sha256 a868288a…",
        "milka_wrapper.jpg": "Ubcule, Wikimedia Commons, CC BY-SA 4.0, sha256 e545bf27…"},
        "results": results, "unenrolled_best_detection": best.det_score}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
