"""End-to-end through normal entry points (fixture profile) + API contracts."""
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pvs import fixtures as fx
from pvs.capture import Frame, frame_from_file
from pvs.cli import main as cli
from pvs.config import load_station
from pvs.detectors.scripted import ScriptedDetector
from pvs.geometry import periodic_error
from pvs.pipeline import Pipeline
from pvs.schema import FrameResult, FrameStatus, IdentityLevel, Profile


def _forced_frame(tmp_path, items, name):
    img, truth = fx.make_scene(77, forced=items, occlude=False, persp=0.0)
    p = tmp_path / f"{name}.png"
    cv2.imwrite(str(p), img)
    from dataclasses import asdict
    return p, {"scenes": [{"image": p.name, **asdict(truth)}]}, truth


def _pipe(repo_root, gt):
    sd = ScriptedDetector()
    sd.load_ground_truth(gt)
    return Pipeline(load_station(repo_root / "configs/station.fixture.yaml"), Profile.FIXTURE, repo_root, scripted=sd)


def test_front_view_gives_sku_and_heading(tmp_path, repo_root):
    p, gt, truth = _forced_frame(tmp_path, [{"sku": "fixture:zephyr-lite-40g", "face": "front", "centre": (480, 270),
                                             "angle": 212.0, "scale": 0.9}], "front")
    r = _pipe(repo_root, gt).process(frame_from_file(p))
    (inst,) = r.instances
    assert inst.identity.level == IdentityLevel.SKU and inst.identity.sku == "fixture:zephyr-lite-40g"
    assert inst.pose.face.value == "front"
    assert periodic_error(inst.pose.heading.value_deg, truth.instances[0].heading_deg, 360) < 3
    assert periodic_error(inst.pose.axis.value_deg, truth.instances[0].axis_deg, 180) < 3


def test_back_barcode_resolves_sku_and_keeps_heading(tmp_path, repo_root):
    p, gt, truth = _forced_frame(tmp_path, [{"sku": "fixture:zephyr-classic-40g", "face": "back", "centre": (480, 270),
                                             "angle": 90.0, "scale": 0.9}], "back")
    (inst,) = _pipe(repo_root, gt).process(frame_from_file(p)).instances
    assert inst.identity.rule == "R2_verified_gtin" and inst.identity.gtin == "02000000000015"
    assert any(e.kind == "barcode" for e in inst.identity.evidence)
    assert inst.pose.face.value == "back"
    assert periodic_error(inst.pose.heading.value_deg, truth.instances[0].heading_deg, 360) < 3


def test_distractor_unknown_with_axis_and_no_arrow(tmp_path, repo_root):
    p, gt, _ = _forced_frame(tmp_path, [{"sku": fx.DISTRACTOR, "face": "front", "centre": (480, 270), "angle": 30.0,
                                         "scale": 0.9}], "dist")
    (inst,) = _pipe(repo_root, gt).process(frame_from_file(p)).instances
    assert inst.identity.level == IdentityLevel.UNKNOWN and inst.identity.sku is None
    assert inst.pose.axis.available and not inst.pose.heading.available
    assert inst.pose.heading.reason in ("endpoint_hidden", "clustered_matches", "high_reprojection_error")


def test_edge_view_axis_only(tmp_path, repo_root):
    p, gt, _ = _forced_frame(tmp_path, [{"sku": "fixture:zephyr-classic-40g", "face": "edge", "centre": (480, 270),
                                         "angle": 120.0, "scale": 1.0}], "edge")
    (inst,) = _pipe(repo_root, gt).process(frame_from_file(p)).instances
    assert inst.pose.axis.available and not inst.pose.heading.available
    assert inst.identity.level != IdentityLevel.SKU


def test_empty_frame_returns_empty_array(tmp_path, repo_root):
    img = np.full((540, 960, 3), 200, np.uint8)
    p = tmp_path / "empty.png"
    cv2.imwrite(str(p), img)
    r = _pipe(repo_root, {"scenes": [{"image": "empty.png", "scene_id": "e", "instances": []}]}).process(frame_from_file(p))
    assert r.status == FrameStatus.NO_PACKAGES and r.instances == [] and r.reason == "ok"
    r2 = _pipe(repo_root, {"scenes": []}).process(frame_from_file(p))
    assert r2.status == FrameStatus.NO_PACKAGES and r2.reason == "fixture_no_script_for_source"


def test_dark_frame_poor_quality(tmp_path, repo_root):
    p = tmp_path / "dark.png"
    cv2.imwrite(str(p), np.full((100, 100, 3), 3, np.uint8))
    r = _pipe(repo_root, {"scenes": []}).process(frame_from_file(p))
    assert r.status == FrameStatus.POOR_QUALITY and r.reason == "quality_dark"


def test_model_failure_is_visible_and_never_substituted(repo_root, tmp_path):
    cfg = load_station(repo_root / "configs/station.fixture.yaml")
    pl = Pipeline(cfg, Profile.FIXTURE, repo_root, scripted=None)  # fixture profile without scripted source
    assert pl.failed
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), np.full((50, 50, 3), 128, np.uint8))
    r = pl.process(frame_from_file(p))
    assert r.status == FrameStatus.MODEL_FAILED and r.instances == []
    cfg.models.detector = "grounding_dino"
    bad = repo_root / "tests" / "_bad_manifest.json"
    bad.write_text(json.dumps({"models": {"grounding_dino": {"repo": "does-not/exist", "revision": "0" * 40},
                                          "sam2": {}, "dinov2": {}}}))
    try:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        pl2 = Pipeline(cfg, Profile.PRETRAINED, repo_root, model_manifest=bad)
        assert pl2.failed and "grounding-dino load failed" in pl2.failed
        assert pl2.process(frame_from_file(p)).status == FrameStatus.MODEL_FAILED
    finally:
        bad.unlink()
        os.environ.pop("HF_HUB_OFFLINE", None)


def test_cli_detect_and_decode_failure(tmp_path, fixture_set, capsys):
    d, gt = fixture_set
    imgs = sorted((d / "scenes").glob("*.png"))[:2]
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    rc = cli(["detect", *map(str, imgs), str(bad), "--ground-truth", str(gt), "--out", str(tmp_path / "o")])
    out = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert rc == 2 and out[-1]["status"] == "decode_failed"
    for f in (tmp_path / "o").glob("*.json"):
        FrameResult.model_validate_json(f.read_text())  # every written result validates against the schema


def test_replay_crossing_tracks_and_events(tmp_path, repo_root):
    from dataclasses import asdict
    vid = tmp_path / "crossing.mp4"
    truths = fx.make_crossing_video(vid, n_frames=30)
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"scenes": [{"image": f"crossing.mp4#{i}", **asdict(t)} for i, t in enumerate(truths)]}))
    rc = cli(["replay", str(vid), "--ground-truth", str(gt), "--out", str(tmp_path / "r")])
    assert rc == 0
    rows = [json.loads(x) for x in (tmp_path / "r/results.jsonl").read_text().splitlines()]
    first = {i["identity"]["sku"]: i["track_id"] for i in rows[0]["instances"]}
    last = {i["identity"]["sku"]: i["track_id"] for i in rows[-1]["instances"]}
    assert first == last and len(first) == 2  # each SKU kept its track across the swap
    # barcode evidence only ever on the classic (back-facing) pack
    for r in rows:
        for i in r["instances"]:
            if any(e["kind"] == "barcode" for e in i["identity"]["evidence"]):
                assert i["identity"]["sku"] == "fixture:zephyr-classic-40g"


def test_repeated_frame_hash_not_accumulated(repo_root, fixture_set):
    d, gt = fixture_set
    from pvs.tracker import Tracker
    pl = _pipe(repo_root, json.loads(gt.read_text()))
    img = sorted((d / "scenes").glob("*.png"))[0]
    f = frame_from_file(img)
    tr = Tracker("t")
    for k in range(5):
        g = Frame(f.image, "t", 0, k, f.captured_at, k / 15, f.source, f.source_key, content_hash=f.content_hash)
        pl.process(g, tr)
    for t in tr.tracks:
        assert len(t.state["smoother_axis"]._obs) <= 1


# ------------------------------------------------------------------ API
@pytest.fixture()
def client(repo_root, fixture_set):
    from pvs.api import create_app
    d, gt = fixture_set
    pl = _pipe(repo_root, json.loads(gt.read_text()))
    cfg = load_station(repo_root / "configs/station.fixture.yaml")
    cfg.limits.max_upload_bytes = 2_000_000
    return TestClient(create_app(pl, cfg)), d


def test_api_detect_validates_against_schema(client):
    c, d = client
    img = sorted((d / "scenes").glob("*.png"))[0]
    r = c.post("/v1/detect", files={"file": (img.name, img.read_bytes(), "image/png")})
    assert r.status_code == 200
    FrameResult.model_validate(r.json())
    sch = c.get("/v1/schema").json()
    assert sch["properties"]["schema_version"]["const"] == "1.0"


def test_api_errors_are_distinct(client):
    c, _ = client
    assert c.post("/v1/detect", files={"file": ("x.png", b"garbage", "image/png")}).json()["error"] == "undecodable"
    big = c.post("/v1/detect", files={"file": ("x.png", b"0" * 2_000_001, "image/png")})
    assert big.status_code == 413
    ok, gif = cv2.imencode(".tiff", np.zeros((8, 8, 3), np.uint8))
    assert c.post("/v1/detect", files={"file": ("x.tiff", gif.tobytes(), "image/tiff")}).status_code == 415
    assert c.post("/v1/detect", files={"file": ("x.png", b"", "image/png")}).json()["error"] == "empty_upload"


def test_api_health_ready_meta_metrics(client):
    c, _ = client
    assert c.get("/healthz").json() == {"status": "alive"}
    assert c.get("/readyz").json()["ready"] is True
    assert c.get("/v1/meta").json()["profile"] == "fixture"
    assert "pvs_requests_total" in c.get("/metrics").text
    assert c.get("/v1/live/latest").status_code == 404


def test_api_auth_when_configured(repo_root, fixture_set, monkeypatch):
    from pvs.api import create_app
    _, gt = fixture_set
    cfg = load_station(repo_root / "configs/station.fixture.yaml")
    cfg.api_token_env = "PVS_TEST_TOKEN"
    monkeypatch.setenv("PVS_TEST_TOKEN", "s3cret")
    c = TestClient(create_app(_pipe(repo_root, json.loads(gt.read_text())), cfg))
    assert c.get("/v1/meta").status_code == 401
    assert c.get("/v1/meta", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_api_ready_503_on_model_failure(repo_root):
    from pvs.api import create_app
    cfg = load_station(repo_root / "configs/station.fixture.yaml")
    c = TestClient(create_app(Pipeline(cfg, Profile.FIXTURE, repo_root), cfg))
    r = c.get("/readyz")
    assert r.status_code == 503 and r.json()["status"] == "model_failed"


def _replay(tmp_path, maker, name):
    from dataclasses import asdict
    vid = tmp_path / f"{name}.mp4"
    truths = maker(vid)
    gt = tmp_path / f"{name}.json"
    gt.write_text(json.dumps({"scenes": [{"image": f"{name}.mp4#{i}", **asdict(t)} for i, t in enumerate(truths)]}))
    assert cli(["replay", str(vid), "--ground-truth", str(gt), "--out", str(tmp_path / name)]) == 0
    return [json.loads(x) for x in (tmp_path / name / "results.jsonl").read_text().splitlines()], truths


def test_turnover_clears_arrow_and_removal_then_nontarget_never_inherits(tmp_path):
    rows, truths = _replay(tmp_path, fx.make_turnover_video, "turnover")
    for r, t in zip(rows, truths):
        for i in r["instances"]:
            truth_face = t.instances[0].face if t.instances else None
            if truth_face == "edge":
                assert not i["pose"]["heading"]["available"]  # arrow cleared while turning
            if t.instances and t.instances[0].sku == "unknown":
                assert i["identity"]["level"] != "sku", (r["frame_id"], i["identity"])  # never inherits the target
        if not t.instances:
            assert r["status"] == "no_packages" and r["instances"] == []
    tids = {i["track_id"] for r, t in zip(rows, truths) for i in r["instances"] if t.instances and t.instances[0].sku == "unknown"}
    target_tids = {i["track_id"] for r, t in zip(rows, truths) for i in r["instances"] if t.instances and t.instances[0].in_catalogue}
    assert tids and not (tids & target_tids)  # the non-target after removal never reuses the target's track ID


def test_api_cors_only_for_configured_origins(repo_root, fixture_set, monkeypatch):
    from pvs.api import create_app
    _, gt = fixture_set
    cfg = load_station(repo_root / "configs/station.fixture.yaml")
    monkeypatch.setenv("PVS_CORS_ORIGINS", "https://pack-station.vercel.app")
    c = TestClient(create_app(_pipe(repo_root, json.loads(gt.read_text())), cfg))
    ok = c.options("/v1/detect", headers={"Origin": "https://pack-station.vercel.app", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "https://pack-station.vercel.app"
    bad = c.get("/readyz", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers
