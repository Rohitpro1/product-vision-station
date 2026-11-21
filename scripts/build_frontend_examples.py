#!/usr/bin/env python3
"""Regenerate frontend/examples: synthetic fixture scenes (lossless PNG) + their FrameResult JSON."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import make_examples as me  # noqa: E402

from pvs import fixtures as fx  # noqa: E402
from pvs.capture import frame_from_file  # noqa: E402
from pvs.config import load_station  # noqa: E402
from pvs.detectors.scripted import ScriptedDetector  # noqa: E402
from pvs.pipeline import Pipeline  # noqa: E402
from pvs.schema import Profile  # noqa: E402

C = (480, 270)
CASES = [
    ("exact_sku_visual", "Exact SKU from artwork",
     [{"sku": "fixture:zephyr-lite-40g", "face": "front", "centre": C, "angle": 32, "scale": 1.0}], None),
    ("barcode_back_face", "Back face: barcode resolves the SKU",
     [{"sku": "fixture:zephyr-classic-40g", "face": "back", "centre": C, "angle": 0, "scale": 1.0}], None),
    ("family_cue_hidden", "Thumb hides the variant cue: family only",
     [{"sku": "fixture:zephyr-lite-40g", "face": "front", "centre": C, "angle": 0, "scale": 1.0}], me.thumb_over_cues),
    ("unknown_lookalike", "Look-alike wrapper: unknown, axis only",
     [{"sku": fx.DISTRACTOR, "face": "front", "centre": C, "angle": 140, "scale": 1.0}], None),
    ("conflict_code", "Foreign barcode on a known pack: conflict",
     [{"sku": "fixture:zephyr-classic-40g", "face": "front", "centre": C, "angle": 0, "scale": 1.2}], me.stick_foreign_code),
    ("three_packs", "Three packs at once",
     [{"sku": "fixture:zephyr-classic-40g", "face": "front", "centre": (220, 160), "angle": 200, "scale": 0.6},
      {"sku": "fixture:zephyr-lite-40g", "face": "back", "centre": (640, 170), "angle": 95, "scale": 0.6},
      {"sku": fx.DISTRACTOR, "face": "front", "centre": (480, 420), "angle": 15, "scale": 0.6}], None),
]


def main() -> int:
    out = ROOT / "frontend" / "examples"
    out.mkdir(parents=True, exist_ok=True)
    gt = {"scenes": []}
    for name, _, items, post in CASES:
        img, truth = fx.make_scene(321, forced=items, occlude=False, persp=0.0)
        if post:
            post(img, truth)
        cv2.imwrite(str(out / f"{name}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        gt["scenes"].append({"image": f"{name}.png", **asdict(truth)})
    sd = ScriptedDetector()
    sd.load_ground_truth(gt)
    pl = Pipeline(load_station(ROOT / "configs/station.fixture.yaml"), Profile.FIXTURE, ROOT, scripted=sd)
    index = []
    for name, title, _, _ in CASES:
        d = json.loads(pl.process(frame_from_file(out / f"{name}.png")).model_dump_json())
        d["source"] = f"examples/{name}.png"
        (out / f"{name}.json").write_text(json.dumps(d))
        index.append({"id": name, "title": title, "image": f"examples/{name}.png", "result": f"examples/{name}.json"})
        print(name, [(i["identity"]["level"], i["identity"]["rule"], i["pose"]["heading"]["value_deg"]) for i in d["instances"]])
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
