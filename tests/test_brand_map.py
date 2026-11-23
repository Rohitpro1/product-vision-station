import pytest

from pvs.training.brand_map import BRANDS, ambiguous, brand_for


@pytest.mark.parametrize("name,brand", [
    ("5Star", "cadbury:5-star"), ("fivestar", "cadbury:5-star"), ("5star 18g", "cadbury:5-star"),
    ("5Star_Bites_Cadbury", "cadbury:5-star"), ("5Star Oreo", "cadbury:5-star"), ("five star", "cadbury:5-star"),
    ("DairyMilk", "cadbury:dairy-milk"), ("Cadbury Diary Milk", "cadbury:dairy-milk"),
    ("Dairy Milk Silk Hazel Nut", "cadbury:dairy-milk"), ("dairymilk", "cadbury:dairy-milk"),
    ("Dairy Milk Oreo", "cadbury:dairy-milk"),
    ("KitKat", "nestle:kitkat"), ("Kit Kat 4 finger 38g", "nestle:kitkat"), ("Nestle KitKat", "nestle:kitkat"),
    ("kitkat-Rs.10", "nestle:kitkat"), ("KITKAT", "nestle:kitkat"),
    ("Munch", "nestle:munch"), ("Nestle Munch Max 19.2g", "nestle:munch"), ("Munch 10.4g", "nestle:munch"),
    ("munch max 20rs", "nestle:munch"), ("Munch Brownie Max", "nestle:munch"),
    ("perk", "cadbury:perk"), ("perk_chocolate", "cadbury:perk"),
    ("Milky_bar", "nestle:milkybar"), ("MilkyBar", "nestle:milkybar"), ("Milkybar Moosha 18g", "nestle:milkybar"),
    ("Bournville", "cadbury:bournville"), ("Bournville Cranberry", "cadbury:bournville"),
    ("gems", "cadbury:gems"), ("Cadbury gems", "cadbury:gems"), ("Choclairs", "cadbury:eclairs"),
    ("eclairs", "cadbury:eclairs"), ("Cadbury fuse 21g", "cadbury:fuse"),
    ("snickers", "mars:snickers"), ("Snickers almonds 14 g 10 rs", "mars:snickers"), ("BOUNTY", "mars:bounty"),
    ("MARS", "mars:mars"), ("Amul Dark Chocolate 35g", "amul:chocolate"), ("Amul_dark_chocolate", "amul:chocolate"),
    ("FERERO RUCHER", "ferrero:rocher"), ("Kinder Bueno", "ferrero:kinder"), ("Hershey", "hersheys:chocolate"),
])
def test_brand_mapping(name, brand):
    assert brand_for(name) == brand
    assert brand in BRANDS


@pytest.mark.parametrize("name", [
    "Kurkure Masala Munch", "Biskit munch", "Farmley-Masala-Munchies-(Rs-30)", "Bournvita", "Bournvita 750gm",
    "HersheysSyrup", "Cadbury Chocobakes", "Chocos", "Amul Butter", "Amul Taaza", "Lays Chips", "Dettol Soap",
    "Dove Shampoo", "Marie Gold", "Choco Pie", "Lotte Choco Pie", "Amul Chocolate Ice Cream 750 ML",
    "Mars Bar Milkshake", "Oreo", "Pears Soap", "marshmallow",
])
def test_not_a_chocolate_brand(name):
    assert brand_for(name) is None


@pytest.mark.parametrize("name,brand", [
    ("CDM medium", "cadbury:dairy-milk"), ("snikers", "mars:snickers"), ("HARSYS", "hersheys:chocolate"),
    ("kisses", "hersheys:chocolate"), ("mars_chocolate", "mars:mars"), ("star 3d", "cadbury:5-star"),
    ("star oreo", "cadbury:5-star"), ("snickers_chocolate", "mars:snickers"), ("kitkat_chocolate", "nestle:kitkat"),
])
def test_new_spellings(name, brand):
    assert brand_for(name) == brand


@pytest.mark.parametrize("name", ["cadbury", "cadbury_chocolate", "CATBORY", "Cadbury Chocolate Berry", "chocolate",
                                  "Chocolates", "candy", "131_chocolate", "nestle", "chocolate-bar"])
def test_ambiguous_names_are_not_negatives(name):
    assert brand_for(name) is None
    assert ambiguous(name)


@pytest.mark.parametrize("name", ["Oreo", "Choco Pie", "Amul Chocolate Ice Cream 750 ML", "Lays Chips", "Dettol Soap",
                                  "Bournvita", "Milky Way", "Cadbury Chocobakes", "DairyMilk", "5Star"])
def test_clear_names_are_not_ambiguous(name):
    assert not ambiguous(name)
