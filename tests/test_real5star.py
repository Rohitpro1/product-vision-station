"""real5star data preparation: class mapping, cross-export duplicate grouping, no split leakage."""
import json
import shutil
from pathlib import Path

import cv2

from pvs import fixtures as fx
from pvs.training.real5star import prepare, sku_for


def make_export(root: Path, name: str, seeds: dict[str, list[int]], cls_name: str) -> Path:
    ex = root / name
    for split, ss in seeds.items():
        d = ex / split
        d.mkdir(parents=True)
        imgs, anns = [], []
        for k, s in enumerate(ss):
            img, truth = fx.make_scene(s, n=2, allow_distractor=False, occlude=False)
            fn = f"img_{s}.jpg"
            cv2.imwrite(str(d / fn), img)
            imgs.append({"id": k, "file_name": fn, "width": img.shape[1], "height": img.shape[0]})
            for j, ins in enumerate(truth.instances):
                x0, y0, x1, y1 = ins.box_xyxy
                anns.append({"id": len(anns), "image_id": k, "category_id": 1 if j == 0 else 2,
                             "bbox": [x0, y0, x1 - x0, y1 - y0]})
        (d / "_annotations.coco.json").write_text(json.dumps(
            {"images": imgs, "annotations": anns,
             "categories": [{"id": 0, "name": "retail"}, {"id": 1, "name": cls_name}, {"id": 2, "name": "BingoChips"}]}))
    return ex


def test_sku_mapping():
    assert sku_for("5Star Chocolate") == "cadbury:5star-classic"
    assert sku_for("5Star") == "cadbury:5star-classic"
    assert sku_for("5Star 3D") == "cadbury:5star-3d"
    assert sku_for("5 star") == "cadbury:5star-classic"
    assert sku_for("5Star Bites") == "cadbury:5star-bites"
    assert sku_for("5Star 3D and 5Star Bites") == "cadbury:5star-mixed"
    assert sku_for("BingoChips") is None and sku_for("DairyMilk Chocolate") is None


def test_prepare_groups_cross_export_duplicates_without_leakage(tmp_path):
    a = make_export(tmp_path, "sct", {"train": [1, 2, 3], "valid": [4], "test": [5]}, "5Star Chocolate")
    b = make_export(tmp_path, "scancart", {"train": [6, 5], "valid": [7], "test": [8]}, "5Star")
    # seed 5 appears in sct/test and scancart/train -> must end up in test only
    stats = prepare([a, b], tmp_path / "out")
    assert stats["leakage"] == []
    assert stats["images_in_duplicate_groups"] >= 2
    m = json.loads((tmp_path / "out/manifest.json").read_text())
    dup = [i for i in m["items"] if i["sha256"] == m["items"][[x["export"] == "sct" and x["orig_split"] == "test"
                                                              for x in m["items"]].index(True)]["sha256"]]
    assert {i["split"] for i in dup} == {"test"}
    assert stats["boxes_by_split_sku"]["train"].get("cadbury:5star-classic", 0) > 0
    shutil.rmtree(tmp_path / "out")
