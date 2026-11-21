import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def fixture_set(tmp_path_factory):
    """Small deterministic scene set + crossing video, generated fresh for the test session."""
    from pvs import fixtures as fx

    d = tmp_path_factory.mktemp("fx")
    gt = fx.generate_set(d / "scenes", 12, seed0=4242)
    return d, gt


@pytest.fixture(scope="session")
def fixture_pipeline(fixture_set, tmp_path_factory):
    from pvs.config import load_station
    from pvs.detectors.scripted import ScriptedDetector
    from pvs.pipeline import Pipeline
    from pvs.schema import Profile

    _, gt = fixture_set
    sd = ScriptedDetector()
    sd.load_ground_truth(json.loads(gt.read_text()))
    cfg = load_station(ROOT / "configs/station.fixture.yaml")
    return Pipeline(cfg, Profile.FIXTURE, ROOT, scripted=sd)
