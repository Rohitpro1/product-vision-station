import numpy as np
import pytest

from pvs.geometry import heading_deg
from pvs.orientation import rectify_crop
from pvs.transforms import Transform, angle_in_image, apply_exif, crop, letterbox, mirror_preview, resize


def _img(w=40, h=20):
    img = np.zeros((h, w, 3), np.uint8)
    img[2, 5] = (255, 255, 255)  # marker at (x=5, y=2)
    return img


def _marker(img):
    ys, xs = np.nonzero(img[:, :, 0])
    return float(xs[0]), float(ys[0])


@pytest.mark.parametrize("o", range(1, 9))
def test_exif_point_mapping(o):
    raw = _img()
    up, T = apply_exif(raw, o)
    assert T.forward(np.array([[5, 2]]))[0] == pytest.approx(_marker(up))
    assert T.inverse(np.array([_marker(up)]))[0] == pytest.approx([5, 2])


def test_exif6_dimensions_and_hand_computed_point():
    # orientation 6 = stored image must be rotated 90 deg clockwise; 40x20 -> 20x40.
    # hand computed: (x, y) -> (H-1-y, x) = (20-1-2, 5) = (17, 5)
    up, T = apply_exif(_img(), 6)
    assert up.shape[:2] == (40, 20)
    assert T.forward(np.array([[5, 2]]))[0] == pytest.approx([17, 5])


def test_non_square_resize_roundtrip():
    img = np.zeros((100, 300, 3), np.uint8)
    _, T = resize(img, (150, 80))
    pts = np.array([[0, 0], [299, 99], [123.4, 56.7]])
    assert T.inverse(T.forward(pts)) == pytest.approx(pts)
    # heading is NOT preserved under anisotropic resize: must always be reported in source frame
    assert angle_in_image(T, (0, 0), (10, 10)) != pytest.approx(45.0, abs=1)


def test_letterbox_roundtrip_and_padding():
    img = np.zeros((100, 300, 3), np.uint8)
    out, T = letterbox(img, 640)
    assert out.shape[:2] == (640, 640)
    p = T.forward(np.array([[0.0, 0.0]]))[0]
    assert p[1] > 200  # vertical padding applied
    pts = np.array([[10, 20], [250, 90]])
    assert T.inverse(T.forward(pts)) == pytest.approx(pts)


def test_crop_near_edge_clips_and_maps_back():
    img = np.zeros((100, 100, 3), np.uint8)
    c, T, (x0, y0, x1, y1) = crop(img, (90, -5, 120, 30), expand=0.2)
    assert (x0, y0) == (87, 0) and x1 == 100 and c.shape[1] == 13
    assert T.inverse(np.array([[0, 0]]))[0] == pytest.approx([87, 0])


def test_mirror_preview_keeps_source_angle():
    img = np.zeros((100, 200, 3), np.uint8)
    a, b = (20.0, 50.0), (120.0, 50.0)
    src_heading = heading_deg(a, b)
    _, T = mirror_preview(img)
    assert angle_in_image(T, a, b) == pytest.approx(180.0)  # drawn reversed in the mirrored preview
    assert heading_deg(a, b) == src_heading == 0.0  # reported angle unchanged


def test_rectify_crop_makes_axis_horizontal():
    img = np.zeros((300, 300, 3), np.uint8)
    out, T = rectify_crop(img, (150, 150), 30.0, 200, 60)
    a = np.array([[150 - 50 * np.cos(np.radians(30)), 150 - 50 * np.sin(np.radians(30))],
                  [150 + 50 * np.cos(np.radians(30)), 150 + 50 * np.sin(np.radians(30))]])
    p = T.forward(a)
    assert heading_deg(tuple(p[0]), tuple(p[1])) == pytest.approx(0.0, abs=1e-6)
    assert T.inverse(p) == pytest.approx(a)


def test_transform_chain_composes():
    T = Transform().then(np.diag([2.0, 2.0, 1.0]), "s").then(np.array([[1, 0, 3], [0, 1, 4], [0, 0, 1.0]]), "t")
    assert T.forward(np.array([[1, 1]]))[0] == pytest.approx([5, 6])
    assert T.steps == ["s", "t"]
