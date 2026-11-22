# Evidence index

Still-image evaluations and real-photo runs: implementation revision `6c36061`. Replays, examples and the
API smoke: `ddadf2e` (tracker-only change; see docs/VERIFICATION.md §6). Provenance labels:
**SYN** synthetic fixture (original artwork, `pvs.fixtures`, seed recorded) · **SCRIPTED** detection boxes
come from fixture ground truth · **REAL-MODEL** real pretrained weights ran · **REAL-PHOTO** third-party photograph.

| Path | Provenance | What it shows |
|---|---|---|
| `examples/exact_sku_visual*` | SYN, SCRIPTED | exact SKU from variant cue + supported heading |
| `examples/exact_sku_barcode_back_face*` | SYN, SCRIPTED | back face: verified GTIN → SKU, heading kept from back artwork |
| `examples/family_variant_cue_hidden*` | SYN, SCRIPTED | thumb over variant cues → family only |
| `examples/unknown_lookalike_axis_only*` | SYN, SCRIPTED | look-alike gold wrapper → unknown, axis, no arrow |
| `examples/edge_view_axis_only*` | SYN, SCRIPTED | edge → axis only, `face: edge` |
| `examples/conflict_foreign_code*` | SYN, SCRIPTED | valid foreign GTIN + strong visual match → conflict, no SKU |
| `examples/three_packs*` | SYN, SCRIPTED | three simultaneous packs (envelope maximum) |
| `examples/empty_frame*` | SYN | `no_packages`, `instances: []` |
| `replay/crossing/` | SYN, SCRIPTED | two look-alikes swap positions; track IDs, barcode stays with its own pack; events |
| `replay/turnover/` | SYN, SCRIPTED | front → edge (arrow clears) → back (GTIN) → removed → non-target (unknown, nothing inherited) |
| `real/5star_unenrolled*` | REAL-PHOTO, REAL-MODEL | 5 Star not enrolled → unknown + axis (Open Food Facts, CC BY-SA 3.0) |
| `real/5star_enrolled_same_photo_rotated*` | REAL-PHOTO, REAL-MODEL | **same-photo self-consistency only**, not validation |
| `real/milka_vs_demo_5star_catalogue*` | REAL-PHOTO, REAL-MODEL | real non-target rejected (Ubcule, Wikimedia Commons, CC BY-SA 4.0) |
| `eval/fixture_sealed150_seed12000/` | SYN, SCRIPTED | sealed synthetic evaluation, frozen thresholds |
| `eval/pretrained_sealed20_seed12000/` | SYN, REAL-MODEL | real models on unseen synthetic scenes |
| `eval/burned/` | SYN | earlier held-out runs, invalidated after failure inspection (kept for audit) |
| `training/` | SYN | landmark + RT-DETR smoke runs, ONNX parity, augmentation grid |
| `api/` | SYN, SCRIPTED | live server smoke transcript, live overlay, generated JSON Schema |
| `training/real5star_kaggle_t4x2/` | REAL-PHOTO (public Roboflow, CC BY 4.0), REAL-MODEL | run 1 on Kaggle 2 × T4: package detector test recall 0.964 / precision 0.967; 5 Star identity 0/3 (reports and logs only; no images, ONNX or gallery) |
| `training/retail_v2_kaggle_t4x2/` | REAL-PHOTO (public Roboflow, CC BY 4.0 / public domain), REAL-MODEL | run 2 on Kaggle 2 × T4: retail package detector test recall 0.808 / precision 0.801 on 2,000 shelf photos; 21-brand gallery, 5 brands kept for deployment; pre-deploy check on public photos (reports only) |

Overlays under `real/` are adaptations of CC BY-SA photographs and carry the same licence; the source photos
themselves are not committed (`scripts/fetch_real_examples.py`).

Frozen copies of the main reports above, plus hosted-service and training-repo observations as of
22 Nov 2026, are in the engineering submission pack: `docs/submission/2026-11-22/evidence/`. Those copies are
checksummed and must not be edited; this folder holds the current files.
