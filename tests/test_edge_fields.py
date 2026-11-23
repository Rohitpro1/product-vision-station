import pytest

from pvs.edge.fields import parse_date, parse_fields


@pytest.mark.parametrize("text,iso", [
    ("EXP 12/2026", "2026-12-01"),
    ("MFD 05.03.26", "2026-03-05"),
    ("BEST BEFORE 15 JAN 2027", "2027-01-15"),
    ("PKD: MAR 2026", "2026-03-01"),
    ("USE BY 31-12-26", "2026-12-31"),
])
def test_parse_date(text, iso):
    d, _ = parse_date(text)
    assert d is not None and d.isoformat() == iso


def test_no_date_from_noise():
    assert parse_date("NET WT 40 g")[0] is None
    assert parse_date("31/13/26")[0] is None


def test_fields_typical_pack():
    lines = ["Cadbury Dairy Milk", "Net Wt. 50 g", "M.R.P. ₹ 50.00 (Incl. of all taxes)", "MFD: 05/2026",
             "BEST BEFORE 9 MONTHS FROM MANUFACTURE", "Batch No: B26CH045"]
    f = parse_fields(lines)
    assert f["mrp"] == 50.0
    assert f["mfg"]["date"] == "2026-05-01"
    assert f["best_before"] == "9 months from manufacture"
    assert f["batch"] == "B26CH045"
    assert f["brand"] == "cadbury:dairy-milk"


def test_fields_label_on_next_line():
    f = parse_fields(["EXP.", "11/2026", "MRP Rs. 10"])
    assert f["exp"]["date"] == "2026-11-01" and f["mrp"] == 10.0


def test_fields_nothing_guessed():
    assert parse_fields(["Kit Kat"]) == {"brand": "nestle:kitkat", "brand_text": "Kit Kat",
                                         "brand_name": parse_fields(["Kit Kat"])["brand_name"]}
    assert parse_fields(["hello world", "40 g"]) == {}
