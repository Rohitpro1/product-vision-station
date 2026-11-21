from pvs.barcode import BarcodeRead
from pvs.config import load_thresholds
from pvs.fusion import FusionInput, decide
from pvs.gallery import GalleryHit
from pvs.geometry import Reason
from pvs.orientation import DirectionResult
from pvs.schema import IdentityLevel, ScoreType
from pvs.tracker import Tracker


def _boxes(t):
    """Two packs that swap horizontal positions over t in [0,1] on different rows."""
    xa = 100 + 400 * t
    xb = 500 - 400 * t
    return [[xa, 100, xa + 120, 160], [xb, 140, xb + 120, 200]]


def test_tracks_follow_objects_through_swap():
    tr = Tracker("cam")
    ids = []
    for f in range(41):
        out = tr.update(_boxes(f / 40), [0.9, 0.9], f / 15, f)
        ids.append((out[0].tid, out[1].tid))
    assert ids[0] == ids[-1]  # each object kept its own ID across the crossing


def test_clean_crossing_delayed_read_stays_with_owner():
    tr = Tracker("cam")
    a, b = tr.update(_boxes(0), [0.9, 0.9], 0.0, 0)
    box_at_capture = list(a.box)
    for f in range(1, 41):  # rows differ: never ambiguous
        tr.update(_boxes(f / 40), [0.9, 0.9], f / 15, f)
    assert tr.add_delayed_evidence(a.tid, "ocr", "ZEPHYR", 0, box_at_capture, 0.0)
    assert "ocr" in a.evidence and "ocr" not in b.evidence  # bound to the crop's owner only


def _same_row(t):
    xa, xb = 100 + 400 * t, 500 - 400 * t
    return [[xa, 100, xa + 120, 160], [xb, 105, xb + 120, 165]]


def test_occluding_swap_drops_late_read_and_clears_evidence():
    tr = Tracker("cam")
    a, b = tr.update(_same_row(0), [0.9, 0.9], 0.0, 0)
    tr.add_evidence(a, "barcode", "2000000000015", 0, 0.0)
    box_at_capture = list(a.box)
    for f in range(1, 41):  # same row: boxes merge mid-crossing -> ambiguous
        tr.update(_same_row(f / 40), [0.9, 0.9], f / 15, f)
    assert a.ambiguous_since_frame is not None and a.ambiguous_since_frame > 0
    assert "barcode" not in a.evidence and "barcode" not in b.evidence  # never migrates to the neighbour
    assert tr.add_delayed_evidence(a.tid, "ocr", "ZEPHYR LITE", 0, box_at_capture, 0.0) is False
    assert "ocr" not in a.evidence and "ocr" not in b.evidence


def test_delayed_read_accepted_when_ownership_unchanged():
    tr = Tracker("cam")
    (a,) = tr.update([[0, 0, 100, 50]], [0.9], 0.0, 0)
    tr.update([[2, 0, 102, 50]], [0.9], 0.07, 1)
    assert tr.add_delayed_evidence(a.tid, "ocr", "X", 0, [0, 0, 100, 50], 0.0)
    assert not tr.add_delayed_evidence(a.tid, "ocr", "X", 0, [300, 0, 400, 50], 0.0)  # wrong crop


def test_repeated_identical_frame_not_independent():
    tr = Tracker("cam")
    (a,) = tr.update([[0, 0, 100, 50]], [0.9], 0.0, 0)
    assert tr.add_evidence(a, "barcode", "x", 0, 0.0, content_hash="h1")
    assert not tr.add_evidence(a, "barcode", "x", 1, 0.1, content_hash="h1")


def test_implausible_jump_clears_evidence_and_epoch_reset():
    tr = Tracker("cam", max_jump=0.6)
    (a,) = tr.update([[0, 0, 100, 50]], [0.9], 0.0, 0)
    tr.add_evidence(a, "barcode", "x", 0, 0.0)
    tr.update([[60, 0, 160, 50]], [0.9], 0.07, 1)  # IoU ok-ish but jump > 0.6*diag? diag~112, jump 60 -> no
    assert "barcode" in a.evidence
    tr2 = Tracker("cam", max_jump=0.3, match_iou=0.1)
    (c,) = tr2.update([[0, 0, 100, 50]], [0.9], 0.0, 0)
    tr2.add_evidence(c, "barcode", "x", 0, 0.0)
    tr2.update([[45, 0, 145, 50]], [0.9], 0.07, 1)
    assert "barcode" not in c.evidence
    tr2.reset(1)
    (d,) = tr2.update([[0, 0, 100, 50]], [0.9], 1.0, 0)
    assert d.tid.startswith("cam:1:") and not d.evidence


def test_stale_tracks_expire():
    tr = Tracker("cam", max_gap_s=0.5)
    (a,) = tr.update([[0, 0, 100, 50]], [0.9], 0.0, 0)
    (b,) = tr.update([[0, 0, 100, 50]], [0.9], 2.0, 30)
    assert a.tid != b.tid


# ---------------------------------------------------------------- fusion
THR = None


def thr(repo_root=None):
    from pathlib import Path
    return load_thresholds(Path(__file__).resolve().parents[1] / "configs/thresholds.yaml")


IDX = {"02000000000015": {"sku": "fx:a", "family": "fx"}, "02000000000022": {"sku": "fx:b", "family": "fx"}}


def _g(sku="fx:a", score=0.9, margin=0.1, comp="fx:b"):
    c = [{"sku": sku, "family": "fx", "side": "front", "score": score, "reference_revision": "1"},
         {"sku": comp, "family": "fx" if comp.startswith("fx") else "", "side": "front", "score": score - margin,
          "reference_revision": "1"}]
    return GalleryHit(c, sku, "fx" if sku.startswith("fx") else None, "front", score, margin, comp)


def _code(text):
    return BarcodeRead(text, "EAN13", True, "ok", [[0, 0], [1, 0], [1, 1], [0, 1]])


def _tmpl(sku, cue_margin=0.2):
    return DirectionResult(10.0, "front", sku, [0, 0], [1, 0], 50, 0.5, 1.0, 1.0, Reason.OK, 10,
                           cue_scores={sku: 0.95, "other": 0.95 - cue_margin}, cue_margin=cue_margin)


def test_verified_gtin_promotes_to_sku():
    i = decide(FusionInput(_g(score=0.5, margin=0.0), [_code("2000000000015")], None), thr(), IDX, ScoreType.RAW_COSINE)
    assert i.level == IdentityLevel.SKU and i.sku == "fx:a" and i.rule == "R2_verified_gtin"


def test_code_for_other_product_plus_strong_visual_is_conflict():
    i = decide(FusionInput(_g("fx:a", 0.95, 0.2), [_code("2000000000022")], None), thr(), IDX, ScoreType.RAW_COSINE)
    assert i.level == IdentityLevel.CONFLICT and i.sku is None and i.reason == "conflicting_code"


def test_unknown_code_plus_strong_visual_is_conflict():
    i = decide(FusionInput(_g("fx:a", 0.95, 0.2), [_code("4006381333931")], None), thr(), IDX, ScoreType.RAW_COSINE)
    assert i.level == IdentityLevel.CONFLICT


def test_visual_sku_needs_variant_cue():
    t = thr()
    no_cue = decide(FusionInput(_g("fx:a", 0.95, 0.2), [], None), t, IDX, ScoreType.RAW_COSINE)
    assert no_cue.level == IdentityLevel.FAMILY
    cue = decide(FusionInput(_g("fx:a", 0.95, 0.2), [], _tmpl("fx:a")), t, IDX, ScoreType.RAW_COSINE)
    assert cue.level == IdentityLevel.SKU and cue.sku == "fx:a"
    weak = decide(FusionInput(_g("fx:a", 0.75, 0.005), [], _tmpl("fx:a", cue_margin=0.01)), t, IDX, ScoreType.RAW_COSINE)
    assert weak.level != IdentityLevel.SKU


def test_negative_reference_yields_unknown():
    g = GalleryHit([{"sku": "negative:gold", "family": "", "side": "any", "score": 0.95, "reference_revision": "1"}],
                   "negative:gold", None, "any", 0.95, 0.3, "fx:a")
    i = decide(FusionInput(g, [], None), thr(), IDX, ScoreType.RAW_COSINE)
    assert i.level == IdentityLevel.UNKNOWN and i.reason == "not_in_catalogue"


def test_track_hold_bounded_and_marked():
    t = thr()
    held = decide(FusionInput(_g(score=0.5, margin=0.0), [_code("2000000000015")], None), t, IDX, ScoreType.RAW_COSINE)
    h = decide(FusionInput(None, [], None, held, 0.4), t, IDX, ScoreType.RAW_COSINE)
    assert h.level == IdentityLevel.SKU and h.rule == "R6_track_hold" and h.age_ms == 400.0
    expired = decide(FusionInput(None, [], None, held, t.track_hold_s + 0.1), t, IDX, ScoreType.RAW_COSINE)
    assert expired.level == IdentityLevel.UNKNOWN


def test_two_different_codes_conflict():
    i = decide(FusionInput(None, [_code("2000000000015"), _code("2000000000022")], None), thr(), IDX, ScoreType.RAW_COSINE)
    assert i.level == IdentityLevel.CONFLICT
