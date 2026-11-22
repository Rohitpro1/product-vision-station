# Decision log

| # | Decision | Why | Reversible by |
|---|---|---|---|
| D1 | Repository `Rohitpro1/product-vision-station`, branch `main`; **public** (the brief's default was private, but the owner created it public and chose to keep it that way) | build prompt defaults; authenticated account; owner's choice of visibility | GitHub settings |
| D2 | Python 3.13, torch 2.15 CPU, transformers 5.23, ONNX Runtime 1.32 (exact pins in `requirements/*.lock`) | versions installable and mutually compatible in the build environment | edit locks + rerun CI |
| D3 | Pretrained baseline = Grounding DINO tiny + SAM 2.1 tiny + DINOv2-small, all Apache-2.0, revisions + SHA-256 in `models/manifest.json` | PDF p.4 route 1–2; smallest variants fit a 2-CPU / 7 GB sandbox | `fetch_models.py --pin` |
| D4 | Direction prototype = SIFT homography to front/back references with inlier, spread, reprojection, non-mirror and axis-agreement gates | PDF p.4 "use printed asymmetric template matches for the first direction arrow" | swap for trained landmark head once validated |
| D5 | Exact SKU from vision requires a **variant cue** (template agreement or catalogue-declared cue regions), not gallery cosine alone | look-alike variants share most artwork; cosine margin alone was not separating them on fixtures | threshold/rule versions |
| D6 | Template-only match supports **family**, not SKU (`R4b`) | shared artwork proves family, not variant | rule version |
| D7 | Global blur/glare gate replaced by crop-level gates | a white or featureless backdrop was being reported as poor quality (found on a real photo and on empty fixtures) | config |
| D8 | False-accept gate uses one-sided 95% upper bound | PDF p.8 note; a 0/73 point estimate does not support ≤1% | — |
| D9 | Outbox delivery is head-of-line ordered | test exposed later events overtaking a backed-off event | — |
| D10 | Gallery scores reported as `raw_cosine_uncalibrated` | no validation data to calibrate | set `calibrated: true` after calibration |
| D11 | OCR backend reported `unavailable` | PaddleOCR not installed in this environment; brand text is optional evidence | install + implement `ocr` evidence kind |
| D12 | FoundationPose and other non-commercial components excluded | licence (PDF p.4, p.11) | separate research config only |
| D13 | Third-party photographs are fetched by script with SHA-256, not committed; only derived JSON and attributed overlays are committed | redistribution rights | — |
| D14 | GPU training on Kaggle (2 × T4, DDP via torchrun) instead of CyFuture | CyFuture V100 VMs stuck at "initiated" / failed and other GPUs were out of stock; Kaggle is free and had T4 × 2 | `scripts/gpu_train_5star.sh` runs on any CUDA 12 host |
| D15 | torch 2.6.0+cu124 (not the CPU lock's 2.15) for GPU runs | newer wheels dropped sm_60/sm_70; 2.6 covers P100/V100/T4 | edit runner pins |
| D16 | Public Roboflow exports used for the generic detector only; 5 Star images and gallery not committed or published | many 5 Star images are watermarked web renders | — |
| D17 | Front end and API both hosted on Vercel's free Hobby plan; Render not used | Render's free tier (512 MB) is too small. Hugging Face Spaces would have needed a browser sign-in. A Vercel Python function (1 vCPU / 2 GB) fits a torch-free profile. | `render.yaml` remains for a paid Render plan |
| D18 | The hosted profile uses the trained ONNX detector + crop GrabCut + colour-histogram gallery (no torch) | keeps the function bundle at about 480 MB and cold starts short | `configs/station.onnx.yaml` (DINOv2) for GPU or large hosts |
| D19 | The runtime bundle (75 MB) lives on the orphan branch `model-assets`, is SHA-256 pinned in `deploy/vercel-api/app.py`, and is fetched to `/tmp` on cold start | release creation was not permitted from the build session; the branch keeps `main` clones small | re-host anywhere and set `PVS_BUNDLE_URL` + `PVS_BUNDLE_SHA256` |
| D21 | Hugging Face Space as primary API, Vercel as fallback | Docker Spaces are now paid, so the Space uses the free Gradio SDK. Gradio owns the server, and the REST API is attached inside `App.create_app`. | edit `frontend/config.js` `apiUrls` |
| D20 | GrabCut fallback runs on a padded ROI downscaled to ≤384 px | full-frame GrabCut took 3.6–4.4 s per pack; the ROI version takes 0.1–0.3 s, with axes within 1° on the real photos | `ThresholdMaskFallback(work_side=...)` |
| D22 | The model each API serves is named in `release.json` on `model-assets` (files listed with SHA-256). The HF Space re-reads it every 10 min and hot-swaps; Vercel reads it on each cold start | lets a model be deployed or rolled back by pushing one file, with no Space edit or PC/browser session; a broken release is rejected and the current one keeps serving (tested locally and on the live Space) | supersedes the fixed pin in D19; built-in v1 release is the fallback |
| D23 | Deploy run 2 (`v2-retail-brands-keep5`) to both APIs | its detector finds more packs in multi-pack scenes than v1 (synthetic three-pack 3 vs 1); test recall 0.808 / precision 0.801 on 2,000 dense shelf photos (not comparable with v1's 5 Star-only test) | set `release.json` back to the v1 entry |
| D24 | A brand stays only with ≥ 10 test crops and test precision ≥ 0.85; retired brands' gallery refs become negatives, so those packs read "unknown" | a wrong brand name is worse than "unknown"; 5 Star was right on only 3 of 12 names | `evidence/training/retail_v2_kaggle_t4x2/brands_report.json` |
| D25 | Cadbury Bournville retired despite 17/17 on test | it named a synthetic plain-coloured pack with score 0.81: an out-of-distribution false positive on the website's own example | re-add once checked on station photos |
| D26 | Model files over GitHub's size limit (two ONNX, about 170 MB) are served from the HF model repo at a fixed revision; small files (gallery, catalogue, configs) from `model-assets` | GitHub's 100 MB file limit; both sources are pinned by SHA-256 in `release.json` | — |
| D27 | Hugging Face's front proxy echoes any `Origin` in CORS headers for the Space; accepted as a known limitation | the header is added by the platform, not the app; the API is public, unauthenticated and keeps no cookies. Vercel enforces the site-only policy | put a proxy in front if the API ever needs auth |

## Model-selection status

The principal decision the PDF asks for — reference-matching baseline vs fine-tuned RT-DETR + landmark
head — **cannot be made yet**: it must be decided on identical held-out real presentations (p.4 "First
comparison"). Both branches are implemented to the point of running; no real training data exists.
Measured so far:
- The pretrained route takes about 13–20 s/frame on 2 CPU cores. That is far outside the 150 ms budget, which is expected without a GPU.
- The RT-DETR-R18 ONNX export matches PyTorch to within 5.5e-5.
- On public retail photos (`evidence/training/real5star_kaggle_t4x2`), the fine-tuned RT-DETR-R18 package detector reaches test recall 0.964 and precision 0.967 at 34 ms/image on a T4.

- Run 2 (`evidence/training/retail_v2_kaggle_t4x2`): trained on 36 public retail projects, test recall 0.808 / precision 0.801 on 2,000 dense shelf photos; DINOv2 brand gallery precision about 0.94 for the 5 kept brands on test crops (estimate).

This favours the trained single-stage detector for **localisation**. Identity still needs a reference set, and the comparison still has to be repeated on station presentations.

## Quote basis (design context, PDF p.9)

One station, one SKU: 37–66 person-days; illustrative labour ₹2.96–5.28 lakh at ₹8,000/PD. These are the
design's planning assumptions, kept separate from measured items: compute, storage, per-SKU enrolment work,
station hardware and recurring support are quoted separately and have **not** been measured here.
