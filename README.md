# Product Vision Station

Recognises wrapped retail packs and reports their orientation per instance. For every detected package it
returns its own box, the strongest supported identity level (`sku` › `family` › `unknown`, or `conflict` /
`ambiguous`), an undirected **axis** [0°,180°), a directed **heading** A→B [0°,360°) when the physical ends
are identifiable, and the visible **face**. Each field has its own evidence, availability, reason code and
observation age. Specification: *Product Recognition & Orientation — ADR-01* (summarised in
`docs/ARCHITECTURE.md`; the PDF itself is not committed).

> **Status:** software, real pretrained inference and enrolment tooling work and are tested.
> **Real Cadbury 5 Star recognition has not been validated**: no authorised reference set or physical
> presentations exist yet. The station has **not** been qualified on hardware. See `docs/VERIFICATION.md`.
>
> **Real-data GPU run (Kaggle, 2 × T4):** a generic pack detector fine-tuned on 1,893 public retail photos
> reached test recall **0.964** and precision **0.967**. 5 Star identity on the same public data **failed**:
> 0/3 exact SKU and 6/357 false accepts. See `evidence/training/real5star_kaggle_t4x2/README.md`.
>
>
> **Live demo:**
> - Site: https://pack-station-psi.vercel.app
> - API: https://pack-station-api.vercel.app (`/readyz`, `POST /v1/detect`). It runs on the Vercel Hobby plan from `deploy/vercel-api`, using the trained ONNX detector.
> - The demo catalogue is synthetic, so real packs come back as `unknown`.

## Quick start (tested commands)

```bash
./scripts/bootstrap.sh                 # venv + pinned locks + editable install + fixtures + tests
. .venv/bin/activate

# 1. deterministic demo (fixture profile: scripted boxes + real identity/orientation algorithms)
pvs detect data/fixtures/scenes/fx-001000.png --ground-truth data/fixtures/scenes/ground_truth.json --out out/demo
pvs replay data/fixtures/video/crossing.mp4 --ground-truth data/fixtures/video/crossing_truth.json --overlay --out out/crossing

# 2. real pretrained inference (Grounding DINO + SAM 2 + DINOv2, pinned + hash-verified)
python scripts/fetch_models.py
python scripts/fetch_real_examples.py            # two CC BY-SA photos, SHA-256 checked
pvs --profile pretrained --config configs/station.pretrained.yaml detect realdata/off_5star_1.jpg --out out/real

# 3. service + live preview (loopback)
pvs --profile pretrained --config configs/station.pretrained.yaml serve --device 0      # http://127.0.0.1:8000/
pvs serve --video data/fixtures/video/crossing.mp4 --ground-truth data/fixtures/video/crossing_truth.json

# 4. evaluation (raw counts, exact intervals, insufficient-evidence states)
pvs eval --ground-truth data/fixtures/scenes/ground_truth.json --out out/eval
```

Each command prints the active **profile**, every **backend** id and the **bundle** id. If a model cannot
load, results report `model_failed`; fixture output is never substituted for real predictions.

| Need | Command / doc |
|---|---|
| enrol a real product | `pvs catalogue enrol --catalogue <cat.json> --product <product.json>` · `docs/OPERATOR_GUIDE.md` §4 |
| collision test / roll back enrolment | `pvs catalogue collisions …` · `pvs catalogue rollback --catalogue …` |
| train landmark/face head · detector | `pvs train landmark …` · `pvs train detector-smoke …` · `docs/TRAINING_EVALUATION.md` |
| export + parity | `pvs train export --ckpt runs/lm/last.pt --out runs/lm/landmark.onnx` |
| signed bundle install / rollback | `pvs bundle build|verify|install|rollback` · `docs/OPERATOR_GUIDE.md` §3 |
| deployment | `deploy/Dockerfile`, `deploy/docker-compose.yml`, `deploy/pvs.service`, `deploy/Caddyfile` |
| API reference | `docs/API.md` (schema at `GET /v1/schema`) |
| requirements → evidence | `docs/REQUIREMENTS_EVIDENCE.md` |

## Capability status

| Capability | Implemented | Verified on | Evidence | Remaining dependency |
|---|---|---|---|---|
| Geometry contract, transforms, EXIF, mirror | yes | unit tests (hand-computed cases) | `tests/test_geometry.py`, `tests/test_transforms.py` | — |
| Fixture vertical slice (still, replay, API) | yes | sealed synthetic set (150 scenes) | `evidence/eval/fixture_sealed150_seed12000`, `evidence/examples` | — |
| Pretrained inference (GDINO + SAM 2 + DINOv2) | yes | real photos + 20 unseen synthetic scenes | `evidence/real`, `evidence/eval/pretrained_sealed20_seed12000` | GPU for speed |
| Real 5 Star recognition | enrolment tooling + DINOv2 gallery trained on public data | **not validated**: public-data test 0/3 exact SKU, false accept 6/357 | `evidence/real/5star_*`, `evidence/training/real5star_kaggle_t4x2` | authorised references, 12–20 physical packs, look-alikes, sessions |
| Exact SKU via barcode / variant cues | yes | fixtures | `evidence/examples/exact_sku_*` | verified real GTIN (currently **unset**) |
| Tracking, evidence expiry, crossing safety | yes | unit tests + synthetic replays | `evidence/replay` | physical crossing scene |
| RT-DETR generic package detector | trained (2 × T4, DDP), ONNX parity 1e-5 | public retail photos: test recall 0.964 [0.939–0.981], precision 0.967, 34 ms/img on T4 | `evidence/training/real5star_kaggle_t4x2` | station presentations |
| Landmark (A→B heading) head | training/export paths | smoke only (fixture data) | `evidence/training` | A/B endpoint labels |
| Signed bundles, outbox, Postgres registry | yes | tests (local PG 16) | `tests/test_outbox_bundle_eval.py` | operator signing keys |
| Live camera, 15 fps / 150 ms, 8-h soak | code path + runbook | **not run** | `docs/VERIFICATION.md` §4 | camera + NVIDIA station |
| Docker image | Dockerfile | **not built** | — | Docker daemon |
| Web front end + hosted API | Vercel static site + Vercel Python function | live upload end to end (headless Chrome): photo → API → box and axis drawn; ~2.3 s per request when warm | `frontend/`, `deploy/vercel-api/` | — |

## Layout

```
src/pvs/            capture, transforms, geometry, detectors/, gallery, orientation, barcode, tracker,
                    fusion, pipeline, schema, render, api, outbox, catalogue, bundle, evaluation, training/
configs/            station.fixture.yaml, station.pretrained.yaml, thresholds.yaml (versioned)
catalogues/         fixture/ (original synthetic artwork, redistributable), templates/
models/manifest.json  pinned weights: repo, revision, SHA-256, licence, intended use
migrations/postgres  central catalogue/release registry
deploy/             Dockerfile, compose, systemd unit, Caddy TLS, trust/ (public keys only)
evidence/           committed results; each artifact labelled by provenance (evidence/README.md)
docs/               architecture, API, operator guide, training/evaluation, verification, decisions
```

Licence: proprietary (see `LICENSE`); third-party notices in `THIRD_PARTY_NOTICES.md`.
