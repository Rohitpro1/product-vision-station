import json

import cv2
import numpy as np

from pvs.gallery import dino_preprocess
from pvs.training.retail import prepare


def _export(root, key, split, imgs, classes):
    d = root / key / split
    d.mkdir(parents=True)
    (root / key / "README.roboflow.txt").write_text("License: CC BY 4.0\n")
    ims, anns = [], []
    for i, img in enumerate(imgs):
        fn = f"{key}_{split}_{i}.jpg"
        cv2.imwrite(str(d / fn), img)
        ims.append({"id": i, "file_name": fn, "width": img.shape[1], "height": img.shape[0]})
        anns.append({"id": i, "image_id": i, "category_id": 0, "bbox": [10, 10, 60, 30]})
    (d / "_annotations.coco.json").write_text(json.dumps(
        {"images": ims, "annotations": anns, "categories": [{"id": 0, "name": classes}]}))


def test_mirrored_copy_in_another_project_follows_the_test_split(tmp_path):
    rng = np.random.default_rng(3)
    base = [rng.integers(0, 255, (120, 160, 3)).astype(np.uint8) for _ in range(6)]
    rf = tmp_path / "rf"
    _export(rf, "a__p", "test", base[:2], "5Star")
    _export(rf, "a__p", "train", base[2:4], "Dettol Soap")
    _export(rf, "b__q", "train", [base[0][:, ::-1].copy(), base[4], base[5]], "KitKat")  # mirrored test image
    (rf / "download_report.json").write_text(json.dumps([
        {"ws": "a", "project": "p", "ok": True, "licence": "CC BY 4.0"},
        {"ws": "b", "project": "q", "ok": True, "licence": "CC BY 4.0"}]))
    lst = tmp_path / "list.yaml"
    lst.write_text("allowed_licences: ['CC BY 4.0']\nprojects:\n  - {ws: a, project: p}\n  - {ws: b, project: q}\n")
    stats = prepare(rf, lst, tmp_path / "data")
    items = json.loads((tmp_path / "data" / "manifest.json").read_text())["items"]
    mirrored = [i for i in items if i["export"] == "b__q" and i["boxes"][0]["class"] == "KitKat"]
    assert stats["leakage"] == []
    assert sorted(i["split"] for i in mirrored) == ["test", "train", "train"]
    assert stats["brand_boxes_by_split"]["test"] == {"cadbury:5-star": 2, "nestle:kitkat": 1}


def test_dino_preprocess_is_landscape_letterboxed():
    tall = np.zeros((300, 60, 3), np.uint8)
    tall[:150] = 255
    x = dino_preprocess([tall])
    assert x.shape == (1, 3, 224, 224)
    # rotated to landscape: the white half is now the left or right half of the middle row
    mid = x[0, :, 112, :]
    assert abs(float(mid[:, :100].mean()) - float(mid[:, 124:].mean())) > 3.0
    # letterbox rows above the pack are the ImageNet mean, i.e. ~0 after normalisation
    assert abs(float(x[0, :, 0, :].mean())) < 0.05
