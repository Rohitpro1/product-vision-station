# Data, training and evaluation guide

## Data workflow (p.5)

1. **Plan groups first.** Every capture item carries `pack_id`, `clip_id`, `session_id`.
   `pvs.training.data.connected_groups` unions items that share any of them; `assign_split` hashes the
   group to train/val/test (70/15/15, salted, deterministic). Do this *before* extracting frames.
2. **Leakage check.** `leakage_check` fails on any shared pack/clip/session across splits and on dHash
   near-duplicates (Hamming ≤ 4). Training aborts when it reports anything; split hashes are recorded.
3. **Teacher proposals.** Run the pretrained profile (Grounding DINO + SAM 2) over raw frames; export with
   `to_cvat_coco` (COCO keypoints, CVAT-importable): box, end_A/end_B with visibility (2 visible,
   1 labelled-unobservable, 0 unlabelled), face, SKU/family, occlusion, ambiguity reason.
4. **Reviewer protocol.**
   - Inspect a random sample of raw frames *including frames where the teacher found nothing*.
   - Correct missed/merged boxes; mark masks as `manual` when edited.
   - Front/back consistency: face must agree with the visible artwork; double-label a stratified 10% slice.
   - Endpoint swaps: A is defined by `docs/reference_drawing.md`, never by image side; adjudicate any
     A/B disagreement and any angular disagreement > 10° before training.
   - Label "unobservable" separately from "unlabelled".
5. **Generalisation set.** Keep an unseen lot / camera / artwork revision as a separately reported set.

Acquisition plan vs actuals must be reported separately (physical packs, independent presentations,
sessions, frames, lots, distractors). Current actuals: **0 physical packs, 0 sessions** (see `docs/VERIFICATION.md`).

## Training entry points

```bash
pvs train inspect-aug --gt data/fixtures/scenes/ground_truth.json --out runs/lm      # grid: A red, B blue
pvs train landmark --gt <gt.json> --steps 2000 --out runs/lm                        # ordered A/B + visibility + face
pvs train landmark --gt <gt.json> --steps 4000 --out runs/lm --resume               # resumable (optimizer + RNG state)
pvs train export --ckpt runs/lm/last.pt --out runs/lm/landmark.onnx                 # ONNX + ORT parity check
pvs train detector-smoke --gt <gt.json> --steps 2 --size 320 --out runs/rtdetr      # RT-DETR-R18 COCO init → package class, ONNX parity
```

The landmark head predicts **ordered physical landmarks** (A then B). It is not an unordered shape-endpoint
model; checkpoints record `"predicts": "ordered A/B landmarks + visibility + face"`. Loss: visibility-masked
smooth-L1 + BCE(visibility) + CE(face). Augmentation applies one affine to image and labels (rotation over
±180°, scale/translation jitter, photometric) — text may appear rotated, endpoint order is preserved; verify
with `inspect-aug` (committed example: `evidence/training/augmentation_check.png`).

**Fixture-trained weights are pipeline evidence only and are excluded from any production bundle.**

## Real-data GPU run (public Roboflow exports, all GPUs)

`scripts/gpu_train_5star.sh` runs the whole flow on any Linux CUDA 12 host:
1. It sets up a clean Python 3.13 environment with torch 2.6.0+cu124.
2. It runs `prepare`, which merges the exports, groups near-duplicates and makes leak-checked splits.
3. It runs `train`, which fine-tunes RT-DETR-R18 with one process per GPU (DDP); `--batch` is the global batch.
4. It runs `evaluate`, which builds the DINOv2 gallery, sets thresholds on val, measures on test and exports to ONNX.

Kaggle (GPU T4 × 2, Internet on, secret `ROBOFLOW_API_KEY`):

```python
!pip -q install roboflow
from kaggle_secrets import UserSecretsClient
from roboflow import Roboflow
rf = Roboflow(api_key=UserSecretsClient().get_secret("ROBOFLOW_API_KEY"))
rf.workspace("nano-b8y2n").project("sct-ufzbr").version(6).download("coco", location="/kaggle/working/data/sct")
rf.workspace("nano-b8y2n").project("scancart").version(2).download("coco", location="/kaggle/working/data/scancart")
```
```bash
git clone --depth 1 https://github.com/Rohitpro1/product-vision-station && cd product-vision-station
REPO=$PWD OUT=/kaggle/working/pvs-run DATA_ROOT=/kaggle/working/data bash scripts/gpu_train_5star.sh 40
# optional: NGPU=1, BATCH_PER_GPU=8 (out of memory), WORKERS=2
```

The runner calls uv and Python under `env -i`, so a host `PYTHONPATH`/`sitecustomize` cannot interfere.
On 2 × T4 the 40 epochs took 42.5 min. The results are in `evidence/training/real5star_kaggle_t4x2/`.

## Evaluation

```bash
pvs eval --ground-truth <gt.json> --out evidence/eval/<name>                       # fixture profile (scripted boxes)
pvs --profile pretrained --config configs/station.pretrained.yaml eval --ground-truth <gt.json> --out ...
```

Reports (`report.json`, `report.md`) contain the implementation commit, input/ground-truth hashes, backends,
thresholds and catalogue revisions, hardware/versions, timing method, raw counts behind every rate,
Clopper–Pearson 95% intervals, and an explicit `insufficient_evidence` status for any zero denominator.
False target acceptance is gated on the **one-sided 95% upper bound** (0 errors in 300 independent negatives
→ 0.994%); repeated frames of one presentation do not count as independent negatives.

Matching is one-to-one (Hungarian) at IoU ≥ 0.5. Angular error is `min(|Δ| mod P, P − |Δ| mod P)`,
P = 180° (axis) or 360° (heading); reversals are heading errors > 150°.

Freeze thresholds (`configs/thresholds.yaml` `version`), gallery and eligibility rules before running the
sealed split. Use one predeclared decision per presentation. For grouped sessions use cluster-aware
intervals (bootstrap by session) — implement when real sessions exist; the fixture set is i.i.d. by seed.

## When accuracy is poor (section 11)

Look at failures (`report.json → failures`) before changing model size. Decide which stage failed first:
missing optical information, labels, detector recall, gallery confusion, track association, angle convention
or stale evidence. If two SKUs' fronts are identical at the station resolution, keep `family`, ask the
operator to turn the pack, or add a view — more compute cannot recover absent evidence.
