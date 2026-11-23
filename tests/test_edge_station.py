import numpy as np

from pvs.edge.seg import Det
from pvs.edge.station import GENERIC, Track
from pvs.training.yolo import _brand_metrics, _polygon


def _det(name, score=0.9):
    return Det((10, 10, 110, 60), score, 0, name)


def test_brand_from_model_and_text():
    t = Track(_det("nestle:kitkat"), 1)
    assert t.brand() == ("nestle:kitkat", "model")
    t.fields = {"brand": "nestle:kitkat"}
    assert t.brand() == ("nestle:kitkat", "model+text")
    t.fields = {"brand": "mars:snickers"}
    assert t.brand() == ("nestle:kitkat", "check")  # disagreement is flagged, never silently resolved


def test_generic_pack_named_by_text_only():
    t = Track(_det(GENERIC), 1)
    assert t.brand() == (GENERIC, "unknown")
    t.fields = {"brand": "cadbury:dairy-milk"}
    assert t.brand() == ("cadbury:dairy-milk", "text")


def test_polygon_fallbacks():
    W = H = 200
    box = [50, 50, 150, 120]
    m = np.zeros((H, W), bool)
    m[52:118, 52:148] = True
    poly, src = _polygon(m, box, W, H)
    assert src == "sam2" and len(poly) >= 4
    tiny = np.zeros((H, W), bool)
    tiny[80:85, 80:85] = True
    assert _polygon(tiny, box, W, H)[1] == "rect"
    assert _polygon(None, box, W, H) == ([[50, 50], [150, 50], [150, 120], [50, 120]], "rect")


def test_brand_metrics_threshold():
    rows = [("nestle:kitkat", "nestle:kitkat", 0.9), ("nestle:kitkat", "mars:snickers", 0.4),
            (GENERIC, "nestle:kitkat", 0.3), ("mars:snickers", None, 0.0)]
    lo = _brand_metrics(rows, 0.25)
    hi = _brand_metrics(rows, 0.5)
    assert (lo["correct_brand"]["k"], lo["wrong_brand"]["k"], lo["other_packs_called_a_brand"]["k"]) == (1, 1, 1)
    assert (hi["correct_brand"]["k"], hi["wrong_brand"]["k"], hi["other_packs_called_a_brand"]["k"]) == (1, 0, 0)
