import copy
import json

import numpy as np
import pytest

from pvs import barcode as bc
from pvs import catalogue as cat


@pytest.mark.parametrize("code,ok", [("4006381333931", True), ("4006381333932", False), ("036000291452", True),
                                     ("00012345600012", True), ("96385074", True), ("12345", False), ("40063813339a1", False)])
def test_gtin_validation(code, ok):
    assert bc.validate_gtin(code)[0] is ok


def test_leading_zeros_preserved():
    assert bc.to_gtin14("036000291452") == "00036000291452"
    assert bc.validate_gtin("00036000291452")[0]


def test_render_decode_roundtrip():
    img = bc.render_ean13("2000000000015", module_px=3, height_px=80, quiet=11)
    reads = bc.decode(np.stack([img] * 3, -1), offset_xy=(100, 50))
    assert [r.text for r in reads] == ["2000000000015"] and reads[0].valid
    assert min(p[0] for p in reads[0].polygon_px) >= 100  # offset applied to polygon


def test_association_requires_exactly_one_owner():
    poly = [[50, 50], [80, 50], [80, 60], [50, 60]]
    assert bc.associate(poly, [[0, 0, 100, 100]]) == 0
    assert bc.associate(poly, [[0, 0, 100, 100], [40, 40, 200, 200]]) is None  # overlapping owners: ambiguous
    assert bc.associate(poly, [[200, 200, 300, 300]]) is None


def _product(sku="x:a", gtin="4006381333931"):
    return {"sku": sku, "family": "x", "gtins": [gtin], "artwork_revision": "1", "references": [],
            "physical_ends": {"A": "a", "B": "b", "definition": "d"}, "supported_faces": ["front"],
            "provenance": "test", "usage_rights": "test", "approval_state": "approved"}


def test_catalogue_rejects_duplicate_gtin_and_bad_check_digit(tmp_path):
    c = {"catalogue": "t", "revision": "t-r1", "products": [_product("x:a"), _product("x:b")]}
    assert any("mapped to both" in e for e in cat.validate(c))
    c2 = {"catalogue": "t", "revision": "t-r1", "products": [_product(gtin="4006381333932")]}
    assert any("bad_check_digit" in e for e in cat.validate(c2))


def test_enrol_and_rollback(tmp_path):
    p = tmp_path / "catalogue.json"
    cat.create(p, "t")
    cat.enrol(p, _product("x:a"), tmp_path)
    c = cat.enrol(p, _product("x:b", "036000291452"), tmp_path)
    assert c["revision"] == "t-r3" and len(c["products"]) == 2
    with pytest.raises(cat.CatalogueError):
        cat.enrol(p, _product("x:c", "036000291452"), tmp_path)  # duplicate GTIN mapping rejected
    assert cat.load(p)["revision"] == "t-r3"  # failed enrol left catalogue untouched
    assert cat.rollback(p)["revision"] == "t-r2"
    assert [x["sku"] for x in cat.load(p)["products"]] == ["x:a"]


def test_fixture_catalogue_valid(repo_root):
    c = json.loads((repo_root / "catalogues/fixture/catalogue.json").read_text())
    assert cat.validate(c, repo_root / "catalogues/fixture") == []
    bad = copy.deepcopy(c)
    bad["products"][0]["approval_state"] = "fixture"
    bad["products"][0]["sku"] = "real:sku"
    assert any("fixture approval" in e for e in cat.validate(bad))
