# Verification report

Implementation revisions evaluated: **`473b608`** (still-image evaluations, real-photo runs) and **`84aaa68`**
(tracker fix; unit tests, tracked replays and the API smoke were re-run at this revision). Evidence and
documentation were added in a later evidence/docs-only commit. See §6.

Environment: Linux x86-64 sandbox, **2 vCPU, 7 GB RAM, no GPU, no camera** (`/dev/video*` absent), Python
3.13.18, torch 2.15.0+cpu, transformers 5.23.0, onnxruntime 1.32.0 (CPUExecutionProvider only),
opencv-python-headless 5.1.0, zxing-cpp 3.1.1. Exact versions: `requirements/*.lock`; weights:
`models/manifest.json` (revision + SHA-256).

Timing method everywhere: `time.perf_counter()` around `Pipeline.process` — **processing-only**. File replay
cannot establish live capture-to-result delay; no camera/driver buffering was measured.

## Summary by level

| Level | What | Result |
|---|---|---|
| 1 Deterministic software | lint, 100 tests + Postgres migration test, CLI slice, API server | **PASS** |
| 2 Real pretrained inference | Grounding DINO + SAM 2 + DINOv2 on real photos and on unseen synthetic scenes | **PASS (runs)**: accuracy figures below; ~13 s/frame on CPU |
| 3 Held-out product presentations | synthetic sealed set: measured. Public retail photos (Kaggle 2 × T4 run): detector and gallery measured. **Real 5 Star station presentations: none available** | synthetic: measured · public-data detector: **measured** · 5 Star identity on public data: **FAIL / insufficient** |
| 4 Named deployment hardware | 15 fps, p95 ≤150 ms, 8-hour soak with reconnects | **NOT RUN — missing prerequisite** (no camera, no GPU station) |

## 1. Deterministic software tests

| Command | Result |
|---|---|
| `ruff check src tests scripts` | All checks passed |
| `python -m pytest -q -m "not models and not postgres"` (at `84aaa68`) | 101 passed, 2 skipped (opt-in real-model module; Postgres without DSN) |
| `PVS_PG_DSN=postgresql://postgres@/pvs?host=/tmp&port=55432 python -m pytest -q tests/test_training_postgres.py` | 5 passed (local PostgreSQL 16; migrations apply, `--check` clean, one-GTIN-one-SKU constraint enforced) |
| `python scripts/migrate.py --check` | clean after apply |
| live server: `pvs serve --port 8765 --video …crossing.mp4 …` + curl | `evidence/api/live_server_smoke.txt` |

Covered behaviours include: hand-computed cardinal headings and 8 EXIF orientations; non-square resize,
letterbox, crop at the image edge, mirrored preview keeping the source angle; 179°/1° axial and 359°/1°
directed smoothing; 180° swap clears; repeated frame hash not accumulated; evidence expiry; occluding swap
drops a late OCR result and never moves a barcode to the neighbour; removal followed by a non-target never
inherits the target; turnover clears the arrow; conflict/unknown/tie rules; outbox crash between remote
delivery and local ack → re-delivery deduplicated; head-of-line ordering under backoff; outbox full →
refuse + alarm + loss counter; corrupt/foreign/unlisted/incompatible bundles rejected with active bundle
unchanged; atomic install + rollback; one-sided bound 0/300 → 0.994%; zero denominators → insufficient evidence.

Training smoke (fixture data; **weights not eligible for any bundle**):
`evidence/training/landmark_smoke_report.json` (20 + 10 resumed optimizer steps, split hashes, leakage none),
`evidence/training/landmark_onnx_parity.json` (max |Δ| 4.8e-7), `evidence/training/rtdetr_smoke.json`
(RT-DETR-R18 from `PekingU/rtdetr_r18vd@ac77a11`: 2 steps, loss 311→163, ONNX parity 5.5e-5 logits),
`evidence/training/augmentation_check.png` (A red / B blue follow every rotation).

## 2. Real pretrained inference

| Input (provenance) | Result | Evidence |
|---|---|---|
| Milka wrapper photo (Wikimedia Commons, CC BY-SA 4.0) vs fixture catalogue | wrapper found with SAM 2 mask, axis 179.5°, identity **unknown** (DINOv2 cos 0.14); the loose chocolate pieces are also proposed as packages (unknown) | `evidence/real/milka_vs_demo_5star_catalogue*` |
| real 5 Star photo held in hand (Open Food Facts, CC BY-SA 3.0) — **not enrolled** | bar found (score 0.68, SAM 2), axis 5.7°, identity **unknown**, heading withheld (`endpoint_hidden`); two false proposals (hand+bar box, clothing) also unknown | `evidence/real/5star_unenrolled*` |
| same photo after **enrolling a reference rectified from that same photo**, then rotated 65° and re-lit | exact demo SKU (`R3`), heading 300.7° = hand-computed 5.7° − 65°; **self-consistency of the enrolment and direction tooling only — not recognition evidence** | `evidence/real/5star_enrolled_same_photo_rotated*` |
| Milka photo vs that demo 5 Star catalogue | all instances **unknown** (best cos 0.33): one real non-target rejected (n = 1) | `evidence/real/milka_vs_demo_5star_catalogue*` |

Unseen synthetic scenes, real models, frozen `thr-dev-0.5`, seeds 12000–12019
(`evidence/eval/pretrained_sealed20_seed12000/report.md`): localisation **25/33** (misses: thin edge views and
the plain gold wrapper — the text prompt does not cover them), exact-SKU precision **11/11**, identity-observable
exact recall **11/13**, false target acceptance 0/12 (upper 95% bound 22% → insufficient), axis coverage 25/33
with p95 error 0.52°, heading coverage 13/13, p95 0.02°, reversals 0. Median processing **12.6 s/frame** (2 vCPU,
CPU only) — about 80× over the 150 ms budget; a GPU and the trained single-stage detector are required.

## 3. Held-out presentations

**Synthetic sealed set** (fixture profile; scripted boxes, so detection is *not* tested; thresholds
`thr-dev-0.5` frozen before the run; seeds 12000–12149 never used in development):
`evidence/eval/fixture_sealed150_seed12000/report.md`.

| Metric | k / n | Gate |
|---|---|---|
| localisation recall | 228/228 | pass (boxes are scripted) |
| exact-SKU precision | 103/103 | pass |
| exact-SKU recall, all target presentations (abstentions count) | 103/145 | **fail** — 40 edge views correctly abstain, 2 conflicts |
| exact-SKU recall, identity-observable (front/back) | 103/105 | pass |
| false target acceptance (look-alike gold wrapper) | 0/83, upper95 3.5% | **insufficient evidence** (needs ≥299 clean negatives) |
| axis coverage / p95 | 228/228 / 0.15° | pass |
| heading coverage / p95 / reversals | 105/105 / 0.05° / 0 | pass |

Processing time (fixture profile, scripted boxes, 960×540): median 31 ms, p95 78 ms per frame.

The 2 conflicts are ZXing misdecodes that pass the mod-10 check (`0004208000022`, `0008404000015`): policy
turned them into `conflict` (no wrong SKU). Remedy logged, not applied to the sealed run: accept only the
catalogue's expected symbology and require agreement across ≥2 frames of a track.

Burned set: seeds 9000–9149 were evaluated, then a rule-ordering defect was found by inspecting their
failures (variant-cue rule pre-empted by the gallery tie rule). They are kept under `evidence/eval/burned/`
and no longer count as held-out.

**Public-data GPU run** (`evidence/training/real5star_kaggle_t4x2`, commit `b5ee790`, Kaggle 2 × T4).
The splits keep near-duplicate groups together, and the test split was used for nothing but this report.

| Metric | k / n | Interval / gate |
|---|---|---|
| generic package detector recall @ IoU 0.5 | 347/360 = 0.964 | 95% CI 0.939–0.981 |
| detector precision | 347/359 = 0.967 | — |
| 5 Star exact-SKU recall (gallery, GT crops) | 0/3 | **fail**; n far below any gate |
| 5 Star exact-SKU precision | 0/7 | **fail** |
| false acceptance as 5 Star, other products | 6/357 | upper95 3.3% → **fail** (gate ≤ 1%) |

Why identity fails here:
- The validation split has no 5 Star boxes, so thresholds could only be set against false accepts.
- The 27 reference crops are mostly watermarked web renders.

This confirms that 5 Star recognition needs authorised references and physical presentations.

**Real 5 Star held-out presentations: none.** No authorised reference set, physical packs, look-alike SKUs
or capture sessions exist. Actual counts: physical packs 0, independent presentations 0, sessions 0,
frames 0, lots 0, distractor SKUs 0 (the two real photographs above are single community images).

Remaining commands once assets exist (input manifest = CVAT export with grouped splits):

```bash
pvs catalogue enrol --catalogue catalogues/station/catalogue.json --product cadbury_5star.json
pvs --profile pretrained --config configs/station.pretrained.yaml eval --ground-truth data/station/sealed/ground_truth.json --out evidence/eval/station_sealed
# acceptance: PDF p.8 gates; false acceptance needs ≥299 independent negatives with 0 errors for ≤1%
```

## Clean-clone verification

From a fresh `git clone` of tip `28b68ce` (local clone; see the publication status below):

| Step | Result |
|---|---|
| `./scripts/bootstrap.sh` (venv, pinned locks, editable install, fixtures, tests) | exit 0 in 58 s (pip cache warm); 101 passed, 1 skipped, 1 deselected |
| `pvs detect data/fixtures/scenes/fx-001000.png --ground-truth …` | 2 instances, exact SKUs, headings 51.28° / 166.15° |
| `python scripts/fetch_models.py` | 4/4 weights OK against pinned SHA-256 (served from the local Hugging Face cache) |
| `pvs --profile pretrained … detect data/fixtures/scenes/fx-001002.png` | real models: exact SKU, heading 26.55° (truth 26.5°), 14.9 s |
| `PVS_RUN_MODELS=1 pytest tests/test_models.py` | 2 passed |
| `ruff check src tests scripts` | clean |

Publication: pushed to https://github.com/Rohitpro1/product-vision-station (`main`, remote tip `ff3e3b8`
at push time). GitHub Actions run 44896002149 on `ff3e3b8`: `deterministic`, `train-smoke` and
`postgres-migrations` succeeded; `real-models` skipped (manual opt-in). A fresh clone from GitHub of `ff3e3b8`
ran `./scripts/bootstrap.sh` (exit 0; 101 passed) plus the fixture and pretrained `pvs detect` smoke runs above.
The repository was created **public** by the owner (spec default was private).

## 4. Named deployment hardware — not run

Prerequisites: 1080p camera with locked exposure/focus, NVIDIA GPU host (≈16 GB VRAM class), the trained
station bundle. Runbook:

```bash
pvs doctor                                                    # provider list must include CUDA/TensorRT
pvs --profile station --config /etc/pvs/station.yaml serve --device /dev/video0 &
# 8 h soak with scheduled reconnects (unplug/replug camera each hour); scrape /metrics every 15 s:
#   coverage = rate(pvs_live_processed_total)/rate(pvs_live_scheduled_total)     >= 0.95
#   p95 pvs_frame_age_ms                                                          <= 150 ms
#   processed fps                                                                 >= 15 with 3 packs
# also record p99, RSS trend, track switches, duplicate events (consumer dedup count)
```

TensorRT/FP16 comparison and INT8 quantisation: not run (no NVIDIA hardware). Docker image: Dockerfile
written, **not built** (no Docker daemon in the sandbox).

## Hosted demo check (2026-11-21)

Site: https://pack-station-psi.vercel.app. API: https://pack-station-api.vercel.app (Vercel deployment `dpl_AgNhtw3Y76PExG1b7z6sk2RYknoA`, package at `7453926`, bundle sha256 `8ba6fdaa…6787`).

| Check | Result |
|---|---|
| `GET /readyz` | `{"ready":true,"profile":"station","bundle":"hosted-onnx-demo"}` |
| CORS preflight from the site's origin | 200, `access-control-allow-origin: https://pack-station-psi.vercel.app` |
| `POST /v1/detect`, 1280 px real 5 Star photo | 200, `ok`, 1 pack, same box as the local run, ~2.3 s per request (1.4 s pipeline) |
| Headless Chrome: open the site, upload the photo through the page | "Connected · station"; the box and a 4.6° axis are drawn; identity `unknown` (expected: synthetic demo catalogue) |

Hugging Face Space check (2026-11-22): https://rohitpro1-pack-station-api.hf.space, files from repo commits `3fb7c75` (requirements) and `5ffae01` (app.py).

| Check | Result |
|---|---|
| `/readyz` | ready; `hosted-onnx-demo` |
| `POST /v1/detect`, same photo | 200, identical box `[160.8, 682.4, 744.9, 885.2]`, ~3.3 s |
| CORS preflight from the site | 200 with `access-control-allow-origin: https://pack-station-psi.vercel.app`; other origins get 400 |
| Live site, headless Chrome | "Connected · Hugging Face", detection sent to `*.hf.space`, axis 4.7° |
| Live site with `*.hf.space` blocked | "Connected · Vercel", detection sent to `pack-station-api.vercel.app`, axis 4.6° |

## 5. Gates not yet supported by evidence

Every PDF p.8 gate on **real station** data (the public-data detector figures above are not station presentations); runtime/latency gates; Docker build; TensorRT; OCR (backend
`unavailable`); calibrated probabilities (scores are raw cosine); cluster-aware intervals (no grouped sessions yet).

## 6. Changes after the implementation commit

| Commit | Files | Affected verification | Re-run |
|---|---|---|---|
| `84aaa68` | `src/pvs/tracker.py` (+ tests) | tracked streams only: a track unseen for > `evidence_gap_s` is no longer re-associated, so an object reappearing after removal gets a new ID. `pvs eval` / `pvs detect` never construct a `Tracker`, so still-image evaluations at `473b608` are unaffected. | full test suite, both synthetic replays, live API smoke, examples |
| later | docs + `evidence/` only | none | — |
