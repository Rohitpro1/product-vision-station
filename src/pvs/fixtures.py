"""Original, redistributable synthetic package fixtures.

The fixture product "Zephyr Bar" is an invented asymmetric wrapper with:

* physical end A (left in the canonical upright front view): a solid notch
  band; end B: a zig-zag crimp. Turning the bar over about its long axis
  keeps A on the left in the canonical *back* view (reference drawing
  ``docs/reference_drawing.md``).
* distinct front artwork (wordmark, stripes, deterministic texture) and back
  artwork (EAN-13 in the GS1 restricted-circulation 2xx range, text lines).
* two look-alike variants (``classic`` / ``lite``) and a plain gold
  distractor that is *not* in the catalogue.

All identifiers are namespaced ``fixture:`` and are not real products.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .barcode import gtin_check_digit, render_ean13
from .geometry import heading_deg

PACK_W, PACK_H = 420, 120  # canonical artwork pixels


def _gtin(body12: str) -> str:
    return body12 + str(gtin_check_digit(body12))


VARIANTS = {
    "fixture:zephyr-classic-40g": {"family": "fixture:zephyr", "stripe": (40, 40, 200), "badge": None,
                                   "gtin": _gtin("200000000001"), "seed": 11},
    "fixture:zephyr-lite-40g": {"family": "fixture:zephyr", "stripe": (200, 120, 30), "badge": "LITE",
                                "gtin": _gtin("200000000002"), "seed": 12},
}
DISTRACTOR = "fixture:plain-gold-wrapper"
# variant-specific regions (reference pixels): coloured stripe and badge area
VARIANT_CUES = {"front": [[40, 8, PACK_W - 40, 26], [296, 90, 372, 114]], "back": [[40, 8, PACK_W - 40, 16]]}  # not in catalogue


def _texture(seed: int, h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.zeros((h, w, 3), np.uint8)
    for _ in range(60):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(3, 12))
        if rng.random() < 0.5:
            cv2.circle(t, (x, y), r, c, -1)
        else:
            cv2.rectangle(t, (x, y), (x + r, y + r // 2 + 2), c, -1)
    return t


def render_face(sku: str, face: str) -> np.ndarray:
    """Canonical artwork (BGR, PACK_H x PACK_W). A is the left end, B the right end."""
    if sku == DISTRACTOR:
        img = np.full((PACK_H, PACK_W, 3), (40, 175, 215), np.uint8)
        cv2.GaussianBlur(img, (0, 0), 3, dst=img)
        return img
    v = VARIANTS[sku]
    img = np.full((PACK_H, PACK_W, 3), (30, 160, 210), np.uint8)  # gold base
    tex = _texture(v["seed"] * (1 if face == "front" else 7), PACK_H, PACK_W)
    m = tex.any(axis=2)
    img[m] = (0.55 * img[m] + 0.45 * tex[m]).astype(np.uint8)
    # A end: solid notch band; B end: zig-zag crimp
    cv2.rectangle(img, (0, 0), (22, PACK_H - 1), (20, 20, 20), -1)
    for y in range(0, PACK_H, 12):
        cv2.line(img, (PACK_W - 22, y), (PACK_W - 1, y + 6), (240, 240, 240), 3)
        cv2.line(img, (PACK_W - 1, y + 6), (PACK_W - 22, y + 12), (240, 240, 240), 3)
    if face == "front":
        cv2.rectangle(img, (40, 8), (PACK_W - 40, 26), v["stripe"], -1)
        cv2.putText(img, "ZEPHYR", (48, 82), cv2.FONT_HERSHEY_DUPLEX, 1.6, (90, 20, 60), 3, cv2.LINE_AA)
        cv2.fillPoly(img, [np.array([[300, 45], [330, 60], [300, 75], [310, 60]])], (90, 20, 60))
        if v["badge"]:
            cv2.putText(img, v["badge"], (300, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.7, v["stripe"], 2, cv2.LINE_AA)
    else:
        bc = render_ean13(v["gtin"], module_px=2, height_px=64, quiet=11)
        bh, bw = bc.shape
        x0, y0 = PACK_W - 26 - bw, 26
        img[y0:y0 + bh, x0:x0 + bw] = cv2.cvtColor(bc, cv2.COLOR_GRAY2BGR)
        for i, line in enumerate(["INGREDIENTS: SUGAR,", "COCOA, MILK SOLIDS", f"NET {sku[-3:].upper()}"]):
            cv2.putText(img, line, (28, 46 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.rectangle(img, (40, 8), (PACK_W - 40, 16), v["stripe"], -1)
    return img


@dataclass
class InstanceTruth:
    sku: str
    family: str | None
    in_catalogue: bool
    face: str
    box_xyxy: list[float]
    A: list[float] | None
    B: list[float] | None
    axis_deg: float | None
    heading_deg: float | None
    axis_observable: bool
    heading_observable: bool
    occluded_frac: float


@dataclass
class SceneTruth:
    scene_id: str
    seed: int
    width: int
    height: int
    instances: list[InstanceTruth] = field(default_factory=list)


def _background(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    base = rng.integers(170, 235, 3)
    bg = np.empty((h, w, 3), np.float32)
    bg[:] = base
    noise = rng.normal(0, 6, (h // 8 + 1, w // 8 + 1)).astype(np.float32)
    bg += cv2.resize(noise, (w, h))[..., None]
    return np.clip(bg, 0, 255).astype(np.uint8)


def place(scene: np.ndarray, art: np.ndarray, centre: tuple[float, float], angle_deg: float, scale: float,
          persp: float = 0.0, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Warp canonical artwork into the scene. Returns (3x3 homography art->scene, mask)."""
    h, w = art.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    cx, cy = w / 2, h / 2
    th = math.radians(angle_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    dst = []
    for _i, (x, y) in enumerate(src):
        p = np.array([(x - cx) * scale, (y - cy) * scale])
        if persp and rng is not None:
            p *= 1 + persp * (rng.random() - 0.5)
        q = R @ p + np.array(centre)
        dst.append(q)
    dst = np.float32(dst)
    H = cv2.getPerspectiveTransform(src, dst)
    H_, W_ = scene.shape[:2]
    warped = cv2.warpPerspective(art, H, (W_, H_), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (W_, H_), flags=cv2.INTER_NEAREST)
    scene[mask > 0] = warped[mask > 0]
    return H, mask


def _ph(H: np.ndarray, pt: tuple[float, float]) -> list[float]:
    p = H @ np.array([pt[0], pt[1], 1.0])
    return [float(p[0] / p[2]), float(p[1] / p[2])]


def make_scene(seed: int, n: int | None = None, width: int = 960, height: int = 540,
               allow_distractor: bool = True, occlude: bool = True, persp: float = 0.15,
               forced: list[dict] | None = None) -> tuple[np.ndarray, SceneTruth]:
    """Generate one scene. ``forced`` items: {sku, face, centre, angle, scale}."""
    rng = np.random.default_rng(seed)
    scene = _background(rng, width, height)
    truth = SceneTruth(scene_id=f"fx-{seed:06d}", seed=seed, width=width, height=height)
    items = forced
    if items is None:
        k = int(rng.integers(0, 4)) if n is None else n
        pool = list(VARIANTS) + ([DISTRACTOR] if allow_distractor else [])
        items = []
        cols = np.linspace(width * 0.22, width * 0.78, max(k, 1))
        for i in range(k):
            items.append({
                "sku": str(rng.choice(pool)),
                "face": str(rng.choice(["front", "back", "front", "edge"])),
                "centre": (float(cols[i] + rng.normal(0, 15)), float(height / 2 + rng.normal(0, 60))),
                "angle": float(rng.uniform(0, 360)),
                "scale": float(rng.uniform(0.55, 0.8)) if k > 1 else float(rng.uniform(0.7, 1.1)),
            })
    masks = []
    for it in items:
        sku, face = it["sku"], it["face"]
        if face == "edge":
            art = np.full((14, PACK_W, 3), (25, 140, 190), np.uint8)
            cv2.rectangle(art, (0, 0), (22, 13), (20, 20, 20), -1)
        else:
            art = render_face(sku, "front" if sku == DISTRACTOR else face)
        H, mask = place(scene, art, it["centre"], it["angle"], it["scale"], persp, rng)
        ah, aw = art.shape[:2]
        A, B = _ph(H, (0, ah / 2)), _ph(H, (aw, ah / 2))
        masks.append(mask)
        ys, xs = np.nonzero(mask)
        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        in_cat = sku in VARIANTS
        hd = heading_deg(tuple(A), tuple(B))
        truth.instances.append(InstanceTruth(
            sku=sku if in_cat else "unknown", family=VARIANTS[sku]["family"] if in_cat else None,
            in_catalogue=in_cat, face=face if sku != DISTRACTOR else "front", box_xyxy=box, A=A, B=B,
            axis_deg=hd % 180.0, heading_deg=hd if (in_cat and face != "edge") else None,
            axis_observable=True, heading_observable=in_cat and face != "edge", occluded_frac=0.0))
    # later packs occlude earlier ones: recompute visibility
    for i, m in enumerate(masks):
        later = np.zeros_like(m)
        for m2 in masks[i + 1:]:
            later |= m2
        vis = (m > 0) & ~(later > 0)
        truth.instances[i].occluded_frac = float(1 - vis.sum() / max((m > 0).sum(), 1))
    if occlude and truth.instances and rng.random() < 0.3:
        tgt = truth.instances[int(rng.integers(0, len(truth.instances)))]
        x0, y0, x1, y1 = tgt.box_xyxy
        cx, cy = (x0 + x1) / 2 + rng.normal(0, 10), (y0 + y1) / 2 + rng.normal(0, 10)
        r = int(0.18 * max(x1 - x0, y1 - y0))
        cv2.ellipse(scene, (int(cx), int(cy)), (r, int(r * 0.6)), float(rng.uniform(0, 180)), 0, 360,
                    (120, 150, 205), -1)  # skin-tone "thumb"
        tgt.occluded_frac = min(1.0, tgt.occluded_frac + 0.1)
    for t in truth.instances:
        if t.occluded_frac > 0.4:
            t.axis_observable = False
            t.heading_observable = False
            t.heading_deg = None
    return scene, truth


def generate_set(out_dir: Path, n: int, seed0: int = 1000) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in range(n):
        seed = seed0 + i
        img, truth = make_scene(seed)
        p = out_dir / f"{truth.scene_id}.png"
        cv2.imwrite(str(p), img)
        manifest.append({"image": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), **asdict(truth)})
    mp = out_dir / "ground_truth.json"
    mp.write_text(json.dumps({"generator": "pvs.fixtures.make_scene", "seed0": seed0, "n": n,
                              "scenes": manifest}, indent=1))
    return mp


def write_references(out_dir: Path) -> list[dict]:
    """Clean front/back reference views for the fixture catalogue (with A/B landmarks)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = []
    for sku in VARIANTS:
        for face in ("front", "back"):
            img = render_face(sku, face)
            name = f"{sku.split(':')[1]}_{face}.png"
            cv2.imwrite(str(out_dir / name), img)
            refs.append({"sku": sku, "side": face, "path": name, "A": [0.0, PACK_H / 2], "B": [float(PACK_W), PACK_H / 2],
                         "sha256": hashlib.sha256((out_dir / name).read_bytes()).hexdigest()})
    return refs


def make_crossing_video(path: Path, n_frames: int = 60, fps: int = 15, size=(960, 540)) -> list[SceneTruth]:
    """Two look-alike packs that swap positions (classic shows its back barcode)."""
    w, h = size
    truths = []
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in range(n_frames):
        t = f / (n_frames - 1)
        xa = w * (0.25 + 0.5 * t)
        xb = w * (0.75 - 0.5 * t)
        items = [
            {"sku": "fixture:zephyr-classic-40g", "face": "back", "centre": (xa, h * 0.35), "angle": 10.0, "scale": 0.8},
            {"sku": "fixture:zephyr-lite-40g", "face": "front", "centre": (xb, h * 0.68), "angle": 170.0, "scale": 0.8},
        ]
        img, truth = make_scene(5000 + f, forced=items, width=w, height=h, occlude=False, persp=0.0)
        truth.scene_id = f"cross-{f:04d}"
        writer.write(img)
        truths.append(truth)
    writer.release()
    return truths


def build_fixture_catalogue(cat_dir: Path) -> dict:
    """Write the namespaced fixture catalogue, references and explicit negative references."""
    ref_dir = cat_dir / "refs"
    refs = write_references(ref_dir)
    neg = render_face(DISTRACTOR, "front")
    cv2.imwrite(str(ref_dir / "neg_plain_gold.png"), neg)
    products = []
    for sku, v in VARIANTS.items():
        products.append({
            "sku": sku, "family": v["family"], "display_name": f"Zephyr {sku.split('-')[1].title()} (fixture)",
            "gtins": [v["gtin"]], "quantity": "40g", "variant": sku.split("-")[1], "artwork_revision": "fx-art-1",
            "references": [{k: r[k] for k in ("side", "A", "B", "sha256")} | {"path": f"refs/{r['path']}",
                            "variant_cues": VARIANT_CUES[r["side"]]} for r in refs if r["sku"] == sku],
            "physical_ends": {"A": "solid dark notch band", "B": "white zig-zag crimp",
                              "definition": "docs/reference_drawing.md#fixture-zephyr"},
            "supported_faces": ["front", "back", "edge"], "provenance": "generated by pvs.fixtures (seeded)",
            "usage_rights": "original synthetic artwork, redistributable (repo licence)", "approval_state": "fixture"})
    cat = {"catalogue": "fixture", "revision": "fixture-r1", "created_at": "2026-11-21T00:00:00+00:00",
           "products": products,
           "negatives": [{"name": "plain-gold-wrapper", "path": "refs/neg_plain_gold.png", "revision": "1",
                          "provenance": "generated look-alike distractor"}],
           "history": [{"revision": "fixture-r1", "at": "2026-11-21T00:00:00+00:00", "change": "initial fixture catalogue"}]}
    (cat_dir / "catalogue.json").write_text(json.dumps(cat, indent=2) + "\n")
    return cat


def make_turnover_video(path: Path, fps: int = 15, size=(960, 540)) -> list[SceneTruth]:
    """Scripted analogue of PDF scenes 2 and 4: front -> edge (turning) -> back -> removed -> non-target."""
    w, h = size
    plan = ([("fixture:zephyr-lite-40g", "front")] * 15 + [("fixture:zephyr-lite-40g", "edge")] * 8
            + [("fixture:zephyr-lite-40g", "back")] * 15 + [None] * 8 + [(DISTRACTOR, "front")] * 14)
    truths = []
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f, item in enumerate(plan):
        forced = [] if item is None else [{"sku": item[0], "face": item[1], "centre": (w / 2 + 2 * f, h / 2),
                                            "angle": 20.0 + 0.5 * f, "scale": 0.9}]
        img, truth = make_scene(7000 + f, forced=forced, width=w, height=h, occlude=False, persp=0.0)
        truth.scene_id = f"turn-{f:04d}"
        writer.write(img)
        truths.append(truth)
    writer.release()
    return truths
