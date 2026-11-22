# Real-data run — Kaggle, 2 × Tesla T4 (2026-11-21)

Produced by `scripts/gpu_train_5star.sh 40` at commit `9dbc97a` on a Kaggle notebook (2 × T4 16 GB,
driver 580, Python 3.13.16, torch 2.6.0+cu124, DDP over both GPUs). The training and evaluation code
is `src/pvs/training/real5star.py`.

**Data:** two Roboflow Universe COCO exports (SCt v6 and Scancart v2, CC BY 4.0).
- Size: 2,729 images, 151 product class names (many duplicates under different spellings).
- Near-duplicate groups (dHash, Hamming ≤ 6) were kept within one split. No leakage was found.
- Split: train 1,893 / val 486 / test 350 images.
- 5 Star boxes in each split:
  - train: 21 classic, 4 3D, 2 Bites
  - val: **0**
  - test: **3** classic
- Many 5 Star images are web renders with stock-photo watermarks. The images and the gallery embeddings are therefore **not** committed.

## Results (test split; never used for any choice)

| What | Result | 95% interval |
|---|---|---|
| Generic package detector (RT-DETR-R18, 1 class), recall @ IoU 0.5 | **0.964** (347/360) | 0.939–0.981 |
| Detector precision @ IoU 0.5, score 0.5 | **0.967** (347/359) | — |
| Detector recall on the 3 test 5 Star boxes | 3/3 | 0.29–1.00 |
| Detector latency, PyTorch FP16, 640 px, one T4 | 34.2 ms/image | — |
| ONNX export parity (max abs diff vs PyTorch) | 1.0e-5 | — |
| 5 Star exact SKU recall (DINOv2 gallery, GT crops) | **0/3** | 0.00–0.71 |
| 5 Star exact SKU precision | **0/7** | 0.00–0.41 |
| 5 Star family-or-better recall | 1/3 | — |
| False accepts as 5 Star on other products | **6/357 = 1.7%** | one-sided upper bound 3.3% |

Training ran 40 epochs at about 64 s/epoch (42.5 min) with a global batch of 32 (16 per GPU) and lr 2.83e-4.
- Val recall reached 0.95 after epoch 1 and plateaued at about 0.98.
- The best val recall+precision was at epoch 11. That checkpoint is the one evaluated and exported.

## What this does and does not show

- **Generic pack detection works on this kind of imagery:** shelf and hand-held retail photos from these two exports. This is evidence for the RT-DETR branch of the PDF's model-selection question, on community data and not on station presentations.
- **5 Star recognition is not validated, and this data cannot validate it.**
  - The validation split has no 5 Star boxes, so the gallery thresholds were chosen only to give zero false accepts on other products.
  - The test split has only 3 boxes, and all 3 were missed at SKU level.
  - Six other products were accepted as 5 Star. That fails the PDF p.8 gate of ≤ 1% false acceptance with ≥ 299 independent negatives and 0 errors.
  - The reference gallery is built from 27 crops, mostly web renders. It does not represent real packs under station lighting.
- **Orientation (heading) is not trained or evaluated.** These datasets have no A/B endpoint labels.
- Boxes come from community annotations, and frames of the same pack are correlated. The intervals therefore treat boxes as if they were independent, which overstates the evidence.

## Artifacts not committed

| File | SHA-256 | Why kept out |
|---|---|---|
| `rtdetr_r18_package.onnx` (80.7 MB) | `68782ed8d550fa47334512634239eec067a032063e85ea7ef1fcb697560cde06` | size; publish as a release asset if wanted |
| `gallery_5star.npz` (5.7 MB) | `4c89d5f88d06b5e6fb5b48fe8fad9a3c0fc11c11f7382ae7b7de11184e2cf961` | derived from watermarked third-party images |
| `pvs-run-artifacts.tgz` | `00c62ae0038716ae384920d6bf1584f407bed40b628f1dffe859c24c7e511f15` | bundle of all of the above |

To reproduce the run, follow the runbook in `docs/TRAINING_EVALUATION.md` ("Real-data GPU run").
