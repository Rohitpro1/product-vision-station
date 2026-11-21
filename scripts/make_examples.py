#!/usr/bin/env python3
"""Produce one labelled example per result state (fixture provenance) into evidence/examples/.

States: exact SKU (visual), exact SKU (barcode), family, unknown, conflict, axis-only, supported heading,
empty frame, two-object crossing. Every artifact is synthetic (pvs.fixtures) unless named otherwise.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pvs import fixtures as fx  # noqa: E402
from pvs.barcode import render_ean13  # noqa: E402
from pvs.capture import frame_from_file  # noqa: E402
from pvs.config import load_station  # noqa: E402
from pvs.detectors.scripted import ScriptedDetector  # noqa: E402
from pvs.pipeline import Pipeline  # noqa: E402
from pvs.render import draw  # noqa: E402
from pvs.schema import Profile  # noqa: E402

OUT = ROOT / "evidence" / "examples"


def scene(name, items, post=None):
    img, truth = fx.make_scene(321, forced=items, occlude=False, persp=0.0)
    if post:
        post(img, truth)
    return name, img, truth


def stick_foreign_code(img, truth):
    """Paste a valid GTIN that is NOT in the catalogue onto the pack (a relabelled / foreign code)."""
    bc = render_ean13("4006381333931", module_px=2, height_px=50, quiet=11)
    x0, y0, x1, y1 = map(int, truth.instances[0].box_xyxy)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    h, w = bc.shape
    img[cy - h // 2: cy - h // 2 + h, cx - w // 2: cx - w // 2 + w] = cv2.cvtColor(bc, cv2.COLOR_GRAY2BGR)


def thumb_over_cues(img, truth):
    """A thumb covers the variant-specific stripe/badge: family is still supported, exact variant is not."""
    cv2.ellipse(img, (470, 228), (190, 40), 0, 0, 360, (120, 150, 205), -1)
    cv2.ellipse(img, (650, 318), (60, 26), 0, 0, 360, (120, 150, 205), -1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [
        scene("exact_sku_visual", [{"sku": "fixture:zephyr-lite-40g", "face": "front", "centre": (480, 270), "angle": 32, "scale": 1.0}]),
        scene("exact_sku_barcode_back_face", [{"sku": "fixture:zephyr-classic-40g", "face": "back", "centre": (480, 270), "angle": 0, "scale": 1.0}]),
        scene("unknown_lookalike_axis_only", [{"sku": fx.DISTRACTOR, "face": "front", "centre": (480, 270), "angle": 140, "scale": 1.0}]),
        scene("edge_view_axis_only", [{"sku": "fixture:zephyr-classic-40g", "face": "edge", "centre": (480, 270), "angle": 60, "scale": 1.1}]),
        scene("conflict_foreign_code", [{"sku": "fixture:zephyr-classic-40g", "face": "front", "centre": (480, 270), "angle": 0, "scale": 1.2}],
              stick_foreign_code),
        scene("family_variant_cue_hidden", [{"sku": "fixture:zephyr-lite-40g", "face": "front", "centre": (480, 270), "angle": 0, "scale": 1.0}],
              thumb_over_cues),
        scene("three_packs", [{"sku": "fixture:zephyr-classic-40g", "face": "front", "centre": (220, 160), "angle": 200, "scale": 0.6},
                              {"sku": "fixture:zephyr-lite-40g", "face": "back", "centre": (640, 170), "angle": 95, "scale": 0.6},
                              {"sku": fx.DISTRACTOR, "face": "front", "centre": (480, 420), "angle": 15, "scale": 0.6}]),
    ]
    empty = np.full((540, 960, 3), 205, np.uint8)
    gt = {"scenes": []}
    for name, img, truth in cases:
        cv2.imwrite(str(OUT / f"{name}.png"), img)
        gt["scenes"].append({"image": f"{name}.png", **asdict(truth)})
    cv2.imwrite(str(OUT / "empty_frame.png"), empty)
    sd = ScriptedDetector()
    sd.load_ground_truth(gt)
    pl = Pipeline(load_station(ROOT / "configs/station.fixture.yaml"), Profile.FIXTURE, ROOT, scripted=sd)
    index = []
    for name in [c[0] for c in cases] + ["empty_frame"]:
        f = frame_from_file(OUT / f"{name}.png")
        r = pl.process(f)
        (OUT / f"{name}.json").write_text(r.model_dump_json(indent=1))
        cv2.imwrite(str(OUT / f"{name}_overlay.jpg"), draw(f.image, r), [cv2.IMWRITE_JPEG_QUALITY, 80])
        (OUT / f"{name}.png").unlink()  # input is reproducible from the seed; keep overlay + JSON
        index.append({"example": name, "provenance": "synthetic fixture (pvs.fixtures, seed 321), scripted boxes",
                      "status": r.status.value,
                      "instances": [{"identity": i.identity.level.value, "sku": i.identity.sku, "family": i.identity.family,
                                     "rule": i.identity.rule, "reason": i.identity.reason, "face": i.pose.face.value,
                                     "axis_deg": i.pose.axis.value_deg, "heading_deg": i.pose.heading.value_deg,
                                     "heading_reason": i.pose.heading.reason} for i in r.instances]})
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    for e in index:
        print(e["example"], e["status"], [(i["identity"], i["rule"], i["face"], i["axis_deg"], i["heading_deg"]) for i in e["instances"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
