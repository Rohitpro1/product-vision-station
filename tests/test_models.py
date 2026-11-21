"""Real pretrained inference (Grounding DINO + SAM 2 + DINOv2). Opt-in: PVS_RUN_MODELS=1 after scripts/fetch_models.py."""
import os

import cv2
import pytest

from pvs import fixtures as fx
from pvs.capture import frame_from_file
from pvs.config import load_station
from pvs.pipeline import Pipeline
from pvs.schema import FrameStatus, Profile

pytestmark = pytest.mark.models
if os.environ.get("PVS_RUN_MODELS") != "1":
    pytest.skip("set PVS_RUN_MODELS=1 to run real-model inference", allow_module_level=True)


@pytest.fixture(scope="module")
def pl(repo_root):
    p = Pipeline(load_station(repo_root / "configs/station.pretrained.yaml"), Profile.PRETRAINED, repo_root)
    assert p.failed is None, p.failed
    assert p.backend["detector"].startswith("grounding-dino:")
    return p


def test_real_models_detect_and_axis(pl, tmp_path):
    img, truth = fx.make_scene(31, forced=[{"sku": "fixture:zephyr-classic-40g", "face": "front", "centre": (480, 270),
                                            "angle": 25.0, "scale": 1.0}], occlude=False, persp=0.0)
    p = tmp_path / "s.png"
    cv2.imwrite(str(p), img)
    r = pl.process(frame_from_file(p))
    assert r.status == FrameStatus.OK and r.instances
    best = max(r.instances, key=lambda i: i.det_score)
    assert best.mask_source == "segmentation"
    assert best.pose.axis.available


def test_real_models_empty_scene(pl, tmp_path):
    import numpy as np
    p = tmp_path / "e.png"
    cv2.imwrite(str(p), np.full((540, 960, 3), 210, np.uint8))
    r = pl.process(frame_from_file(p))
    assert r.status in (FrameStatus.NO_PACKAGES, FrameStatus.OK)
    assert r.status != FrameStatus.NO_PACKAGES or r.instances == []
