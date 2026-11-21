import json
import os
import subprocess
import sys
from argparse import Namespace

import pytest

from pvs.training.data import assign_split, connected_groups, leakage_check


def test_group_split_keeps_connected_items_together():
    items = [{"id": "f1", "pack_id": "P1", "clip_id": "C1"}, {"id": "f2", "clip_id": "C1", "session_id": "S9"},
             {"id": "f3", "session_id": "S9"}, {"id": "f4", "pack_id": "P2"}]
    g = connected_groups(items)
    assert g["f1"] == g["f2"] == g["f3"] != g["f4"]
    assert len({assign_split(g[i["id"]]) for i in items[:3]}) == 1


def test_leakage_check_flags_shared_groups_and_near_duplicates():
    s = {"train": [{"id": "a", "pack_id": "P1", "dhash": 0b1010}], "test": [{"id": "b", "pack_id": "P1", "dhash": 0b1011}]}
    probs = leakage_check(s)
    assert any("pack_id" in p for p in probs) and any("near-duplicate" in p for p in probs)
    assert leakage_check({"train": [{"id": "a", "pack_id": "P1"}], "test": [{"id": "b", "pack_id": "P2"}]}) == []


def test_landmark_smoke_resume_and_onnx_parity(fixture_set, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from pvs.training import landmark

    _, gt = fixture_set
    a = Namespace(gt=str(gt), out=str(tmp_path / "lm"), steps=3, batch=4, lr=1e-3, seed=1, resume=False)
    r1 = landmark.train(a)
    assert r1["steps_run"] == 3 and r1["leakage"] == []
    a.steps, a.resume = 5, True
    r2 = landmark.train(a)
    assert r2["resumed_from"] == 3 and r2["steps_run"] == 2
    e = landmark.export(Namespace(ckpt=str(tmp_path / "lm/last.pt"), out=str(tmp_path / "lm.onnx")))
    assert e["parity_ok"]


def test_cvat_coco_export(fixture_set, tmp_path):
    from pvs.training.data import to_cvat_coco
    _, gt = fixture_set
    coco = json.loads(to_cvat_coco(json.loads(gt.read_text()), tmp_path / "coco.json").read_text())
    assert coco["categories"][0]["keypoints"] == ["end_A", "end_B"]
    assert all(len(a["keypoints"]) == 6 for a in coco["annotations"])


@pytest.mark.postgres
def test_postgres_migrations_apply_and_check(repo_root):
    dsn = os.environ.get("PVS_PG_DSN")
    if not dsn:
        pytest.skip("PVS_PG_DSN not set")
    env = dict(os.environ)
    run = lambda *a: subprocess.run([sys.executable, str(repo_root / "scripts/migrate.py"), *a], env=env,  # noqa: E731
                                    capture_output=True, text=True)
    assert run().returncode == 0
    r = run("--check")
    assert r.returncode == 0, r.stdout + r.stderr
    q = subprocess.run(["psql", dsn, "-qAt", "-c",
                        "INSERT INTO catalogue_revision(revision,catalogue,manifest,manifest_sha256) VALUES "
                        "('t-r1','t','{}',repeat('a',64)) ON CONFLICT DO NOTHING; "
                        "INSERT INTO product VALUES ('x:a','t-r1','x','1',null,null,'approved','r','p') ON CONFLICT DO NOTHING; "
                        "INSERT INTO gtin_mapping VALUES ('t-r1','04006381333931','x:a','test') ON CONFLICT DO NOTHING; "
                        "SELECT count(*) FROM gtin_mapping WHERE revision='t-r1'"], capture_output=True, text=True)
    assert q.stdout.strip().endswith("1")
    dup = subprocess.run(["psql", dsn, "-v", "ON_ERROR_STOP=1", "-qAt", "-c",
                          "INSERT INTO gtin_mapping VALUES ('t-r1','04006381333931','x:b','test')"], capture_output=True, text=True)
    assert dup.returncode != 0  # one GTIN -> one SKU per revision
