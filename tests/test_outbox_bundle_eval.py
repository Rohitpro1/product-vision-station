import json
import os

import pytest

from pvs import bundle as b
from pvs.evaluation import clopper_pearson, one_sided_upper, rate, score_results
from pvs.outbox import DedupConsumer, Outbox, OutboxFull


def test_event_ids_stable_and_state_change_only(tmp_path):
    ob = Outbox(tmp_path / "o.sqlite", "st1")
    e1 = ob.record_transition("cam", 0, "cam:0:1", {"level": "family"}, "state_change", "t0")
    assert ob.record_transition("cam", 0, "cam:0:1", {"level": "family"}, "state_change", "t1") is None
    e2 = ob.record_transition("cam", 0, "cam:0:1", {"level": "sku"}, "state_change", "t2")
    assert e1 == Outbox.event_id("st1", "cam", 0, "cam:0:1", 1) and e2 == Outbox.event_id("st1", "cam", 0, "cam:0:1", 2)


def test_restart_preserves_sequence_and_dedup(tmp_path):
    p = tmp_path / "o.sqlite"
    ob = Outbox(p, "st1")
    ob.record_transition("cam", 0, "t", {"s": 1}, "k", "x")
    ob.close()
    ob = Outbox(p, "st1")  # restart
    assert ob.record_transition("cam", 0, "t", {"s": 1}, "k", "x") is None  # replay after restart: no duplicate
    assert ob.record_transition("cam", 0, "t", {"s": 2}, "k", "x") == Outbox.event_id("st1", "cam", 0, "t", 2)


def test_crash_between_remote_delivery_and_ack_redelivers_and_consumer_dedups(tmp_path, monkeypatch):
    ob = Outbox(tmp_path / "o.sqlite", "st1")
    for i in range(3):
        ob.record_transition("cam", 0, "t", {"s": i}, "k", "x")
    consumer = DedupConsumer()

    def send_then_crash(ev):
        consumer(ev)  # remote side applied it
        raise ConnectionError("crashed before local ack")

    r = ob.deliver(send_then_crash, now=0.0)
    assert r["delivered"] == 0 and r["pending"] == 3
    assert ob.deliver(consumer, now=0.1)["delivered"] == 0  # backoff respected
    assert ob.deliver(consumer, now=100.0)["delivered"] == 3
    assert consumer.duplicates == 1 and len(consumer.applied) == 3 and ob.pending() == 0


def test_full_outbox_refuses_and_alarms(tmp_path):
    ob = Outbox(tmp_path / "o.sqlite", "st1", max_rows=3)
    for i in range(3):
        ob.record_transition("cam", 0, f"t{i}", {"s": 1}, "k", "x")
    assert ob.alarms()["high_water"]
    with pytest.raises(OutboxFull):
        ob.record_transition("cam", 0, "t9", {"s": 1}, "k", "x")
    assert ob.alarms()["data_loss_events"] == 1
    ob.deliver(lambda ev: None, now=0.0)
    assert ob.record_transition("cam", 0, "t9", {"s": 1}, "k", "x")  # delivered rows pruned first


# ----------------------------------------------------------------- bundles
@pytest.fixture()
def keys(tmp_path):
    priv, pub = tmp_path / "keys/test.key", tmp_path / "trust/test.pub"
    b.keygen(priv, pub)
    return priv, pub.parent


def _src(tmp_path, n):
    s = tmp_path / f"src{n}"
    s.mkdir()
    (s / "thresholds.yaml").write_text(f"version: v{n}\n")
    (s / "model.onnx").write_bytes(os.urandom(64))
    return s


def test_bundle_sign_install_rollback(tmp_path, keys):
    priv, trust = keys
    st = tmp_path / "station"
    b1 = b.build(_src(tmp_path, 1), tmp_path / "b1", "bundle-1", priv, {"catalogue": "c-r1"})
    b.build(_src(tmp_path, 2), tmp_path / "b2", "bundle-2", priv, {"catalogue": "c-r2"})
    assert b1["schema_version"] == "1.0"
    assert b.install(tmp_path / "b1", st, trust) == "bundle-1"
    assert b.install(tmp_path / "b2", st, trust) == "bundle-2"
    assert b.active_bundle(st, trust)["bundle_id"] == "bundle-2"
    assert b.rollback(st, trust) == "bundle-1"
    assert b.active_bundle(st, trust)["bundle_id"] == "bundle-1"


def test_corrupt_or_foreign_bundle_rejected_and_active_unchanged(tmp_path, keys):
    priv, trust = keys
    st = tmp_path / "station"
    b.build(_src(tmp_path, 1), tmp_path / "b1", "bundle-1", priv, {})
    b.install(tmp_path / "b1", st, trust)
    b.build(_src(tmp_path, 2), tmp_path / "b2", "bundle-2", priv, {})
    with open(tmp_path / "b2/model.onnx", "ab") as f:
        f.write(b"x")
    with pytest.raises(b.BundleError, match="hash mismatch"):
        b.install(tmp_path / "b2", st, trust)
    other_priv = tmp_path / "other.key"
    b.keygen(other_priv, tmp_path / "other.pub")
    b.build(_src(tmp_path, 3), tmp_path / "b3", "bundle-3", other_priv, {})
    with pytest.raises(b.BundleError, match="signature"):
        b.install(tmp_path / "b3", st, trust)
    (tmp_path / "b1" / "extra.bin").write_bytes(b"x")
    with pytest.raises(b.BundleError, match="unlisted"):
        b.verify(tmp_path / "b1", trust)
    assert b.active_bundle(st, trust)["bundle_id"] == "bundle-1"


def test_incompatible_schema_rejected(tmp_path, keys):
    priv, trust = keys
    b.build(_src(tmp_path, 1), tmp_path / "b1", "bundle-1", priv, {"schema_version": "2.0"})
    with pytest.raises(b.BundleError, match="incompatible schema"):
        b.verify(tmp_path / "b1", trust)


# -------------------------------------------------------------- evaluation
def test_one_sided_bound_matches_design_note():
    assert one_sided_upper(0, 300) == pytest.approx(0.00994, abs=2e-5)
    assert one_sided_upper(0, 30) == pytest.approx(0.0950, abs=5e-4)


def test_insufficient_evidence_on_zero_denominator():
    r = rate(0, 0, "x")
    assert r["status"] == "insufficient_evidence" and r["value"] is None
    assert clopper_pearson(0, 0) is None
    rep = score_results([])
    assert all(m.get("gate") == "insufficient_evidence" for m in rep["metrics"].values())


def test_scoring_counts_and_wraparound():
    truth = {"scene_id": "s", "instances": [{"sku": "a", "in_catalogue": True, "face": "front", "box_xyxy": [0, 0, 100, 50],
                                             "axis_deg": 1.0, "heading_deg": 359.0, "axis_observable": True,
                                             "heading_observable": True, "occluded_frac": 0.0}]}
    pred = {"instances": [{"bbox_xyxy_px": [2, 1, 101, 50], "identity": {"level": "sku", "sku": "a", "rule": "r", "reason": "ok"},
                           "pose": {"axis": {"available": True, "value_deg": 179.0},
                                    "heading": {"available": True, "value_deg": 1.0}}}]}
    rep = score_results([(truth, pred)])
    assert rep["metrics"]["axis_p95_err_deg"]["value"] == pytest.approx(2.0)
    assert rep["metrics"]["heading_p95_err_deg"]["value"] == pytest.approx(2.0)
    assert rep["counts"]["sku_correct"] == 1
    pred["instances"][0]["pose"]["heading"]["value_deg"] = 181.0
    assert score_results([(truth, pred)])["counts"]["reversals"] == 1
    _ = json
