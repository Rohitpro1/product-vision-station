# Training run 2: retail package detector + chocolate brands (Kaggle, 2 × T4)

**Provenance:** REAL-PHOTO (public Roboflow Universe exports, CC BY 4.0 and public domain), REAL-MODEL.
Reports and configs only. No images, ONNX models or gallery are committed here. The trained files are in the Hugging Face model repo
[`Rohitpro1/pack-station-training`](https://huggingface.co/Rohitpro1/pack-station-training) at revision
`394f78830d29e6b780239f0f6fcfe6c7b8029dcf`. Their SHA-256 values are in `bundle_sha256.txt`.

| | |
|---|---|
| Run | 22 Nov 2026, 17:55 → 02:44 IST, Kaggle saved version, 2 × T4, DDP via torchrun |
| Code | `scripts/gpu_train_retail.sh` at `ed64b5e` (pre-rewrite ID; see `docs/COMMIT_ID_MAP.md`) |
| Data | 36 public projects, 65,034 images after de-duplication; split train 56,155 / val 5,382 / test 3,497; no leakage between splits (`data_stats.json`) |
| Detector | RT-DETR-R18 (`PekingU/rtdetr_r18vd@ac77a11`), one class "package", 20 epochs (best = 20), 8.0 h training |
| Identity | DINOv2-small (`facebook/dinov2-small@ed25f3a`) ONNX, brand-level gallery, thresholds chosen on val |

## Held-out test results (from the run)

**Package detector** (`detector_report.json`): 2,000 test images (dense shelves, about 4.5 packs per image).

| Recall @ IoU 0.5 | Precision | Speed (T4, incl. decode) | ONNX parity |
|---|---|---|---|
| 0.808 [0.799–0.816] (7,260 / 8,990) | 0.801 | 37 ms/image | max abs diff 3.4e-5 |

Run 1's 0.964 was measured on a different, much easier test set (single 5 Star packs), so the two numbers are
not comparable.

**Brands, all 21 in the gallery** (`brands_report.json`, crop level, test): out of 1,027 brand crops, 446 were named correctly
(0.434) and 19 were named as the wrong brand (0.019). Out of 8,166 other products, 73 (0.009) were called a brand. End to end on
500 test photos (`e2e_report.json`): 41% of brand packs were named correctly and 1.7% were named wrongly. The median time
was 3.7 s per photo on Kaggle CPU.

## What was deployed, and why

The bar for a brand to stay: **at least 10 test crops and precision ≥ 0.85**, where precision means that most of the times the model
names that brand, the name is right. A wrong name is worse than "unknown".

| Kept | Test crops | Named (right / all) | Recall |
|---|---|---|---|
| Cadbury Dairy Milk | 312 | 172 / 174 | 0.55 |
| Cadbury Gems | 15 | 15 / 17 | 1.00 |
| Nestlé KitKat | 233 | 43 / 49 | 0.18 |
| Bounty | 51 | 21 / 24 | 0.41 |
| Kinder | 35 | 20 / 23 | 0.57 |

16 brands were retired. Their gallery references are kept as negatives, so those packs come back as
"unknown". Most of them fell below the precision bar. **Cadbury 5 Star** was named 12 times and was right
only 3 times. Amul and Ferrero Rocher had too few test crops. **Cadbury Bournville** met the bar (17 / 17)
but was retired after the pre-deploy check below. Full table: `brands_report.json` → `test_crop_level.per_brand`.

**Estimated effect of keeping 5** (`keep5_estimate.json`, derived from the counts above, not re-measured):
287 names given on the test crops, 271 right. That is precision 0.944 [0.911–0.968] and recall 0.42 for
the kept brands.

## Pre-deploy check on public photos (`realcheck.json`)

The check used 11 Wikimedia Commons photos of chocolate packs plus the 6 synthetic examples on the website, with the full
pipeline run locally for v1 and v2. The photos were used for this check only and are not stored.

- **No wrong names.** That includes the retired Snickers.
- **No right names either.** 0 of 9 photos of kept brands were named. Close-up web photos look different from the
  shelf crops the gallery was built from, so expect "unknown" on that kind of photo.
- **The v2 detector finds more packs in multi-pack scenes:** synthetic three-pack example 3 vs 1, Kinder photo 2 vs 0,
  shop shelf 3 vs 2.
- With 6 brands kept, one synthetic plain-coloured pack was named **Cadbury Bournville** (score 0.81). Bournville
  was retired as well; with the remaining 5 brands, nothing in the check was named.

## Live checks after deploy (23 Nov 2026, 03:50–04:00 IST)

- **Release:** both APIs report `bundle: hosted-retail-brands-v2`, catalogue `brands-c42c79ff-13156-keep5`.
  - Hugging Face Space: picked up `release.json` within one 10-minute poll.
  - Vercel: redeployed.
- **Timing:** `POST /v1/detect` works on both hosts. The shelf photo takes 6.2 s on Vercel and 7.3 s on the Space; the synthetic examples take about 2.7 s on Vercel. A Vercel cold start, including the 190 MB model download, took about 10 s in a local run.
- **CORS:** the website origin is allowed on both hosts. Vercel refuses other origins (preflight 400). The Space's front proxy, run by
  Hugging Face, echoes any origin. The API has no logins or cookies, so this only means other sites could call
  the public demo API too.
- **Website:** shows brand display names and the new status note.

## Files

| File | What |
|---|---|
| `detector_report.json`, `train_report.json`, `rtdetr_r18_package.json` | detector test metrics, per-epoch log, ONNX export info |
| `brands_report.json`, `e2e_report.json` | brand gallery, val thresholds, test metrics (crop level and end to end) |
| `data_stats.json`, `download_report.json` | projects, licences, split sizes, de-duplication, leakage check |
| `status.json`, `bundle_sha256.txt` | final run status with per-stage durations; SHA-256 of every bundle file |
| `catalogue_brands_keep5.json`, `station.hosted.v2.yaml`, `thresholds_brands.yaml` | deployed catalogue (with retirement history), config, thresholds |
| `release_v2.json` | the release file served from `model-assets/release.json` |
| `keep5_estimate.json`, `realcheck.json` | derived estimate for the kept brands; pre-deploy check |

## Limits

- **Correlated test data:** test photos come from the same public projects as the training data. Frames of one product
  are correlated, so these numbers say little about a real station camera.
- **Thresholds:** they are raw cosine scores chosen on validation, not calibrated probabilities.
- **Brand level only:** pack sizes and flavours are not told apart.
- **Real 5 Star recognition:** still not validated. That needs authorised reference photos of physical packs at the station.
