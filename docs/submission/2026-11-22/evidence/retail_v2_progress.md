# Pack Station training run

**State:** running  ·  **updated:** 22 Nov 22:07 IST  ·  **started:** 22 Nov 17:55 IST

**Now:** `[train] 16,386/35,080 steps (46.7%) | 1.21 steps/s | stage ETA 4h17m (02:25 IST) | run ETA 5h06m (03:14 IST) | epoch 10/20 | loss 6.803 | imgs_per_s 38.8`

**Current stage:** train `█████████░░░░░░░░░░░` 46.7%

**Estimated finish:** 23 Nov 03:14 IST (in 5h06m; estimate, refined as the run goes)

| Stage | Took |
|---|---|
| setup | 12s |
| download | 6m10s |
| prepare | 18m07s |
| smoke:train | 1m11s |
| smoke:export | 16s |
| smoke:brands | 24s |
| smoke:bundle | 0s |
| smoke:e2e | 39s |

**Detector training (validation sample)**

| Epoch | Loss | Val recall@0.5 | Val precision@0.5 | Elapsed |
|---|---|---|---|---|
| 1 | 64.951 | 0.409 | 0.463 | 46s |
| 1 | 17.590 | 0.766 | 0.612 | 24m14s |
| 2 | 8.757 | 0.779 | 0.690 | 48m15s |
| 3 | 8.179 | 0.792 | 0.678 | 1h12m |
| 4 | 7.840 | 0.665 | 0.838 | 1h36m |
| 5 | 7.548 | 0.816 | 0.699 | 2h00m |
| 6 | 7.337 | 0.779 | 0.767 | 2h24m |
| 7 | 7.130 | 0.814 | 0.725 | 2h48m |
| 8 | 6.906 | 0.846 | 0.685 | 3h13m |
| 9 | 6.724 | 0.814 | 0.766 | 3h37m |


Source: https://github.com/Rohitpro1/product-vision-station (`scripts/gpu_train_retail.sh`).
