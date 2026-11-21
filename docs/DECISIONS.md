# Decision log

| # | Decision | Why | Reversible by |
|---|---|---|---|
| D1 | Repository `Rohitpro1/product-vision-station`, private, branch `main` | build prompt defaults; authenticated account | GitHub settings |
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
| D17 | Front end deployed to Vercel; API not deployed (Render blueprint ready) | owner chose to deploy the backend later | `render.yaml` |

## Model-selection status

The principal decision the PDF asks for — reference-matching baseline vs fine-tuned RT-DETR + landmark
head — **cannot be made yet**: it must be decided on identical held-out real presentations (p.4 "First
comparison"). Both branches are implemented to the point of running; no real training data exists.
Measured so far:
- The pretrained route takes about 13–20 s/frame on 2 CPU cores. That is far outside the 150 ms budget, which is expected without a GPU.
- The RT-DETR-R18 ONNX export matches PyTorch to within 5.5e-5.
- On public retail photos (`evidence/training/real5star_kaggle_t4x2`), the fine-tuned RT-DETR-R18 package detector reaches test recall 0.964 and precision 0.967 at 34 ms/image on a T4.

This favours the trained single-stage detector for **localisation**. Identity still needs a reference set, and the comparison still has to be repeated on station presentations.

## Quote basis (design context, PDF p.9)

One station, one SKU: 37–66 person-days; illustrative labour ₹2.96–5.28 lakh at ₹8,000/PD. These are the
design's planning assumptions, kept separate from measured items: compute, storage, per-SKU enrolment work,
station hardware and recurring support are quoted separately and have **not** been measured here.
