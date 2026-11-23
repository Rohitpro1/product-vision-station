import pytest

from pvs.training.brand_map import BRANDS, ambiguous, brand_for
from pvs.training.off import NEG_CLASS, image_url, label_product, split_of


@pytest.mark.parametrize("tags,names,cats,expect", [
    (["ferrero", "ferrero-rocher"], ["Ferrero rocher, fine hazelnut chocolates"], ["en:chocolates"], ("brand", "ferrero:rocher")),
    (["milka"], ["Milka hazelnut"], ["en:chocolates"], ("brand", "mondelez:milka")),
    (["snickers"], ["Mini Snickers"], ["en:snacks"], ("brand", "mars:snickers")),
    (["nestle"], ["Kit Kat Dark 70%"], ["en:chocolates"], ("brand", "nestle:kitkat")),
    (["cadbury"], ["Dairy Milk Key Lime Pie"], ["en:chocolates"], ("brand", "cadbury:dairy-milk")),
    (["hershey-s"], ["milk chocolate"], ["en:chocolates"], ("brand", "hersheys:chocolate")),
    (["amul"], ["Amul Dark Chocolate 55%"], ["en:chocolates"], ("brand", "amul:chocolate")),
    # tag says one brand, name another -> skip
    (["toblerone"], ["Toblerone fruit &nut"], ["en:chocolates"], ("skip", "conflicting brands")),
    (["hershey-s", "kitkat"], ["Kit Kat"], ["en:chocolates"], ("skip", "conflicting brands")),
    # brand on something that is not a bar
    (["bounty"], ["Bounty Smooth & Creamy Frozen Bars"], ["en:frozen-desserts"], ("skip", "brand on a non-bar product or assortment")),
    (["milka"], ["Chocolate & Hazelnut Cake"], ["en:cakes"], ("skip", "brand on a non-bar product or assortment")),
    (["mars"], ["Celebrations"], ["en:chocolates"], ("skip", "maker assortment (shows brand logos)")),
    (["cadbury"], ["Heroes"], ["en:chocolates"], ("skip", "maker assortment (shows brand logos)")),
    # maker products that are not one of the brands: hard negatives
    (["cadbury"], ["P.S. Caramilk"], ["en:chocolates"], ("negative", None)),
    (["nestle"], ["milo"], ["en:beverages"], ("negative", None)),
    (["ferrero", "tic-tac"], ["Tic Tac"], ["en:candies"], ("negative", None)),
    # generic name under a maker: could be a brand -> skip
    (["nestle"], ["Chocolate bar"], ["en:chocolates"], ("skip", "generic chocolate name")),
    (["nestle"], [], [], ("skip", "no name")),
])
def test_label_product(tags, names, cats, expect):
    assert label_product(tags, names, cats) == expect


def test_display_names_map_back_to_their_family():
    # OFF exports use display names as COCO class names; prepare maps them back with brand_for
    for fam, name in BRANDS.items():
        assert brand_for(name) == fam, name


def test_negative_class_is_a_clean_negative():
    assert brand_for(NEG_CLASS) is None and not ambiguous(NEG_CLASS)


def test_image_url_and_split():
    assert image_url("7622202318078", "front_en", 14) == \
        "https://images.openfoodfacts.org/images/products/762/220/231/8078/front_en.14.400.jpg"
    assert image_url("12345678", "front_fr", 3).endswith("/products/12345678/front_fr.3.400.jpg")
    assert split_of("7622202318078") == split_of("7622202318078")
    counts = {s: sum(split_of(str(10**12 + i)) == s for i in range(3000)) for s in ("train", "valid", "test")}
    assert 0.65 < counts["train"] / 3000 < 0.75 and 0.11 < counts["test"] / 3000 < 0.19
